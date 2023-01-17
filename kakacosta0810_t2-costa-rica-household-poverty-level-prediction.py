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
df = pd.read_csv('/kaggle/input/costa-rican-household-poverty-prediction/train.csv')

df.head(10)
df.info()
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
g = df.columns.to_series().groupby(df.dtypes).groups
g
def trata_cat(df):

    cols = ['dependency', 'edjefe', 'edjefa']

    df[cols] = df[cols].replace({'yes':1, 'no':0})

    df["dependency"] = df["dependency"].astype(float)

    df["edjefe"] = df["edjefe"].astype(int)

    df["edjefa"] = df["edjefa"].astype(int)

    return df
def dropa_trata(df):

    df_limpo = df.drop(["Id","idhogar"],axis=1).copy()

    return trata_cat(df_limpo)
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
under = ["TomekLinks"]

over = ("RandomOverSampler","ADASYN","BorderlineSMOTE","SMOTE","SVMSMOTE")

df_d1 = df.fillna(-1)
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