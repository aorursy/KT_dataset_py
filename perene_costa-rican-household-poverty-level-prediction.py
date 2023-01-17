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
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from imblearn.over_sampling import (RandomOverSampler,ADASYN,BorderlineSMOTE,
                                    KMeansSMOTE,SMOTE,SVMSMOTE)

from imblearn.under_sampling import (RandomUnderSampler,CondensedNearestNeighbour,
                                     EditedNearestNeighbours,
                                    RepeatedEditedNearestNeighbours,
                                    NeighbourhoodCleaningRule,AllKNN,TomekLinks)

from imblearn.pipeline import Pipeline
df = pd.read_csv("/kaggle/input/costa-rican-household-poverty-prediction/train.csv")
df.info()
g = df.columns.to_series().groupby(df.dtypes).groups
g
def extract_features(df):
    df['bedrooms_to_rooms'] = df['bedrooms']/df['rooms']
    df['rent_to_rooms'] = df['v2a1']/df['rooms']
    df['tamhog_to_rooms'] = df['tamhog']/df['rooms'] # tamhog - size of the household
    df['r4t3_to_tamhog'] = df['r4t3']/df['tamhog'] # r4t3 - Total persons in the household
    df['r4t3_to_rooms'] = df['r4t3']/df['rooms'] # r4t3 - Total persons in the household
    df['v2a1_to_r4t3'] = df['v2a1']/df['r4t3'] # rent to people in household
    df['v2a1_to_r4t3'] = df['v2a1']/(df['r4t3'] - df['r4t1']) # rent to people under age 12
    df['hhsize_to_rooms'] = df['hhsize']/df['rooms'] # rooms per person
    df['rent_to_hhsize'] = df['v2a1']/df['hhsize'] # rent to household size
    df['rent_to_over_18'] = df['v2a1']/df['num_over_18']
    # some households have no one over 18, use the total rent for those
    df.loc[df.num_over_18 == 0, "rent_to_over_18"] = df[df.num_over_18 == 0].v2a1
    
    return df
def trata_cat(df):
    df['dependency'] = np.sqrt(df['SQBdependency'])
    df.loc[(df.v14a ==  1) & (df.sanitario1 ==  1) & (df.abastaguano == 0), "v14a"] = 0
    df.loc[(df.v14a ==  1) & (df.sanitario1 ==  1) & (df.abastaguano == 0), "sanitario1"] = 0
    df['num_over_18'] = 0
    df['num_over_18'] = df[df.age >= 18].groupby('idhogar').transform("count")
    df['num_over_18'] = df.groupby("idhogar")["num_over_18"].transform("max")
    df['num_over_18'] = df['num_over_18'].fillna(0)
    cols = ['edjefe', 'edjefa']
    df[cols] = df[cols].replace({'yes':1, 'no':0})
    df["dependency"] = df["dependency"].astype(float)
    df["edjefe"] = df["edjefe"].astype(int)
    df["edjefa"] = df["edjefa"].astype(int)
    df['edjef'] = np.max(df[['edjefa','edjefe']], axis=1)
    df = extract_features(df)
    return df
def dropa_trata(df):
    df_limpo = df.drop(["Id"],axis=1).copy()
    df_tratado = trata_cat(df_limpo)
    return df_tratado.drop(["idhogar"],axis=1)
def treino(df,model,train_submit=False):
    if train_submit:
        X, y = df.drop("Target",axis=1),df["Target"]
        model.fit(X,y)
        return model
    if "Target" in df.columns:
        X, y = df.drop("Target",axis=1),df["Target"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify=y, random_state=42)
        model.fit(X_train,y_train)
        predicted = model.predict(X_test)
        print('Classifcation report:\n', classification_report(y_test, predicted))
        print('Confusion matrix:\n', confusion_matrix(y_test, predicted))
        return model
    else:
        return model.predict(df)
def treino_sampling(df,model,train_submit=False):
    if train_submit:
        X, y = df.drop("Target",axis=1),df["Target"]
        model.fit(X,y)
        return model
    if "Target" in df.columns:
        resampling = RandomOverSampler()
        pipeline = Pipeline([('Resampling', resampling), ('XGBClassifier', model)])
        X, y = df.drop("Target",axis=1),df["Target"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify=y, random_state=42)
        pipeline.fit(X_train, y_train)
        predicted = model.predict(X_test)
        print('Classifcation report:\n', classification_report(y_test, predicted))
        print('Confusion matrix:\n', confusion_matrix(y_test, predicted))
        return model
    else:
        return model.predict(df)
def sampling_predict(df,func):
    
    resampling = func()
    model = XGBClassifier()
    pipeline = Pipeline([('Resampling', resampling), ('XGBClassifier', model)])
    
    X=df.drop("Target",axis=1)
    y=df.Target
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)
    pipeline.fit(X_train, y_train) 
    predicted = pipeline.predict(X_test)
    print('Classifcation report:\n', classification_report(y_test, predicted))
    print('Confusion matrix:\n', confusion_matrix(y_test, predicted))
xgb=XGBClassifier()
modelo_final_sem_sampler = treino(df=dropa_trata(df),model=xgb)
under = ["RandomUnderSampler","EditedNearestNeighbours",
                                    "RepeatedEditedNearestNeighbours",
                                    "NeighbourhoodCleaningRule","AllKNN","TomekLinks"]
over = ("RandomOverSampler","ADASYN","BorderlineSMOTE","SMOTE","SVMSMOTE")
df_d1 = df.fillna(0)
for model in under:
    print("\n"+model+"\n")
    sampling_predict(dropa_trata(df_d1),eval(model))
for model in over:
    print("\n"+model+"\n")
    sampling_predict(dropa_trata(df_d1),eval(model))
modelo_final = treino_sampling(dropa_trata(df),model=xgb)
df_test = pd.read_csv("/kaggle/input/costa-rican-household-poverty-prediction/test.csv")
preds = treino(dropa_trata(df_test),modelo_final_sem_sampler)
z = pd.Series(preds,name="Target")
df_entrega = pd.concat([df_test.Id,z], axis=1)
df_entrega.to_csv("/kaggle/working/submission.csv",index=False)