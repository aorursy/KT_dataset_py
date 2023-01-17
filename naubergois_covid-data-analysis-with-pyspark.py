# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install pyspark
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from tqdm import tqdm
import pandas as pd

def load_raw():
    #Create Spark Session
    spark = SparkSession.builder \
        .master('local') \
        .appName('myAppName') \
        .config('spark.executor.memory', '12gb') \
        .config("spark.cores.max", "10") \
        .getOrCreate()

    #Get spark context
    sc = spark.sparkContext


    sqlContext = SQLContext(sc)
    
    df=pd.read_excel('/kaggle/input/covid19/dataset.xlsx')
    
    
    df['Respiratory Syncytial Virus']=df['Respiratory Syncytial Virus'].astype(str)
    df['Influenza A']=df['Influenza A'].astype(str)
    df['Influenza B']=df['Influenza B'].astype(str)
    df['Parainfluenza 1']=df['Parainfluenza 1'].astype(str)
    df['CoronavirusNL63']=df['CoronavirusNL63'].astype(str)
    df['Rhinovirus/Enterovirus']=df['Rhinovirus/Enterovirus'].astype(str)
    df['Coronavirus HKU1']=df['Coronavirus HKU1'].astype(str)
    
    for column in df.columns:
        df[column]=df[column].astype(str)
    
    df=sqlContext.createDataFrame(df)
    
    
    
   
    
    
    return df,sqlContext
df,sqlContext=load_raw()
print('Number of lines ',df.count())
df.printSchema()
df=df.fillna(0)
from pyspark.sql.functions import *
df=df.replace("nan", "0")
pd.DataFrame(df.head(5),columns=df.schema.names)
df_hemoglobin=df.select("Hemoglobin").toPandas()
df_hemoglobin['Hemoglobin']=pd.to_numeric(df_hemoglobin['Hemoglobin'])
df_hemoglobin['Hemoglobin'].hist()
df.select("SARS-Cov-2 exam result").show()
df_=df.select(col("SARS-Cov-2 exam result").alias("result"),col('Patient age quantile').alias('age'))
df_.show()
df_.select("result", "age").write.mode('overwrite').option("header", "true").save("result_age.parquet", format="parquet")
df_ = sqlContext.sql("SELECT * FROM parquet.`/kaggle/working/result_age.parquet`")
df_.printSchema()
import pyspark.sql.functions as func
df_.groupBy("result").agg(func.max("age"), func.avg("age")).show()
df_pandas_age=df_.groupBy("result").agg(func.max("age"), func.avg("age")).toPandas()
df_pandas_age.plot()
from pyspark.sql.types import IntegerType
columns=df.schema.names
for column in columns:
    df= df.withColumn(column, df[column].cast(IntegerType()))
df.printSchema()
import re

df_numeric_pandas=df_numeric.toPandas()
df_class_1=df_numeric_pandas[df_numeric_pandas['SARS-Cov-2 exam result']!='negative']
df_class_0=df_numeric_pandas[df_numeric_pandas['SARS-Cov-2 exam result']=='negative']
df_class_0=df_class_0[:len(df_class_1)]

df_numeric_concat=pd.concat([df_class_0,df_class_1],axis=0)

y=df_numeric_concat['SARS-Cov-2 exam result']

y_l=[0 if r=='negative' else 1 for r in y]


columns_to_drop = ['SARS-Cov-2 exam result','Patient ID']
X = df_numeric_concat.drop('SARS-Cov-2 exam result',axis=1)

columns=X.columns
X.columns=[str(re.sub(r"[^a-zA-Z0-9]+", ' ', column)) for column in columns]
columns=X.columns
X.columns=[column.replace("!@#$%^&*()[]{};:,./<>?\|`~-=_+", " ") for column in columns]

columns=X.columns


for column in X.columns:
    X[column]=pd.to_numeric(X[column],errors='ignore')

X=pd.get_dummies(X)


for column in X.columns:
    if '<' in column:
        X=X.drop([column],axis=1)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X_train,X_test,y_train,y_test=train_test_split(X,y_l,random_state=10,shuffle=True)

#print(y_train)
import shap
import xgboost
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

def xgboost_classifier(X_train,y_train,X_test,y_test):

    model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X_train, label=y_train), 100)
    dtest = xgboost.DMatrix(X_test,label=y_test)
    preds = model.predict(dtest)
    
    y_pred=[int(pred>0.5) for pred in preds]
    
    score=classification_report(y_test,y_pred)
    
    return model,score

def xai_plot_values(model,X_):
    shap.initjs()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_)
    return explainer,shap_values
    


model,report=xgboost_classifier(X_train,y_train,X_test,y_test)
print(report)
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
# A parameter grid for XGBoost
params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }
model = XGBClassifier(learning_rate=0.02, n_estimators=600, objective='binary:logistic',
                    silent=True, nthread=1)


skf = StratifiedKFold(n_splits=5, shuffle = True, random_state = 1001)

random_search = RandomizedSearchCV(model, 
                                   
                                   param_distributions=params, 
                                   n_iter=10, 
                                   scoring='roc_auc', n_jobs=-1, 
                                   cv=skf.split(X_train,y_train), verbose=3, random_state=1001 )



random_search.fit(X_train, y_train)

