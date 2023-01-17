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
import re

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import plotly.figure_factory as ff

from plotly.subplots import make_subplots

%matplotlib inline
train_df=pd.read_csv('../input/titanic/train.csv')

train_df
train_df.info()
null_percent=train_df.isnull().sum()/len(train_df)*100

null_percent.sort_values(ascending=False)
sns.heatmap(train_df.isnull(),yticklabels=False,cbar=False)
train_df.describe()
train_df.columns=map(str.lower,train_df.columns)

train_df
train_df.groupby('survived').count().passengerid
fig=px.sunburst(train_df,path=['sex','survived'],values='passengerid')

fig.show()
fig = px.violin(train_df, y="age", x="sex", color="survived",points='all', box=False, hover_data=train_df.columns, range_y=[train_df.age.min()-.5, train_df.age.max()+.5])

fig.show()
fig = px.histogram(train_df, x="age",y="survived",color="pclass", marginal="box",hover_data=train_df.columns)

fig.show()
sns.countplot(data=train_df,x='pclass')
sns.countplot(data=train_df,x='pclass',hue='survived')
sns.countplot(data=train_df,x='sex',hue='survived')
train_df.ticket.describe()
train_df.loc[train_df['ticket'] == '1601']

train_df.loc[train_df['ticket']=='CA. 2343']
train_df.loc[train_df['name'].str.contains("Sage")]

train_df['new_cabin'] = train_df.cabin.dropna().astype(str).str[0] 

train_df.groupby(by=['new_cabin','pclass']).pclass.count()

train_df.groupby(by='pclass').pclass.count()

train_df.loc[(train_df.survived==0) & (train_df['cabin'].isnull())].count()
train_df=train_df.drop(columns='new_cabin')
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(train_df.drop(columns=['survived']),train_df.survived,random_state=42)
def CExist(df):

    df['cabin']=df['cabin'].fillna(0)

    df.cabin=df.cabin.apply(lambda x: 0 if x==0 else 1)

    return df

    
x_train=CExist(x_train)

x_train
train_df['name'].apply(lambda name: name.split(',')[1].split('.')[0].strip()).value_counts()
normalized_titles = {"Capt":"o",

                     "Col":"o",

                     "Major":"o",

                     "Jonkheer":"r",

                     "Don":"r",

                     "Sir" :"r",

                     "Dr":"o",

                     "Rev":"o",

                     "the Countess":"r",

                     "Dona":"r",

                     "Mme":"Mrs",

                     "Mlle":"Miss",

                     "Ms":"Mrs",

                     "Mr" :"Mr",

                     "Mrs" :"Mrs",

                     "Miss":"Miss",

                     "Master":"Master",

                     "Lady":"r"}

def normalize_titles(df):

    df['title']=df['name'].apply(lambda name:name.split(',')[1].split('.')[0].strip()).map(normalized_titles)

    return df

x_train=normalize_titles(x_train)
x_train
train_df.name.apply(lambda x:len(x)).describe()
def name_length(df):

    df['name_len']=df.name.apply(lambda x:1 if len(x)>25 else 0)

    return df
x_train=name_length(x_train)

x_train
def fill_age(trainset,testset=None):

    if testset is None:

        trainset=trainset.fillna(trainset.median())

        return trainset

    else:

        testset=testset.fillna(trainset.age.median())

        return testset
x_train = fill_age(x_train)

x_train
def age_categorize(trainSet):

  interval = (0, 5, 12, 18, 25, 35, 60, 100)

  age_cat = ['babies', 'children', 'teenage', 'student', 'young', 'adult', 'senior']

  trainSet["age_cat"] = pd.cut(trainSet.age, interval, labels=age_cat)

  return trainSet
x_train = age_categorize(x_train)

x_train
def family(df):

    df['family']=df['sibsp']+df['parch']

    df['family']=df['family'].apply(lambda x:1 if x>0 else 0)

    return df
x_train=family(x_train)

x_train
def fare_categorize(trainSet):

  quant = (-1, 0, 8, 15, 31, 600)

  label_quants = ['NoInf', 'quart_1', 'quart_2', 'quart_3', 'quart_4']

  trainSet["fare_cat"] = pd.cut(trainSet.fare, quant, labels=label_quants)

  return trainSet
x_train=fare_categorize(x_train)

x_train
def get_dummies_t(dataFrame):

  for column in dataFrame.columns:

    if (dataFrame[column].nunique()<10  and dataFrame[column].dtype==np.dtype('O')) or (dataFrame[column].nunique()<10 and dataFrame[column].nunique()>2):

      if column == "sibsp" or column == "parch" or column == "ticket_token":

        continue

      if column == "title":

        dataFrame = dataFrame.join(pd.get_dummies(dataFrame[column], prefix=column))

        dataFrame.drop(columns=column, inplace = True)

        continue

      dataFrame = dataFrame.join(pd.get_dummies(dataFrame[column], prefix=column, drop_first=True))

      dataFrame.drop(columns=column, inplace = True)

  return dataFrame
x_train=get_dummies_t(x_train)

x_train
def drop_text(dataFrame):

  for column in dataFrame.columns:

    if dataFrame[column].dtype==object:

      dataFrame.drop(columns=column, inplace = True)

  return dataFrame
x_train=drop_text(x_train)
x_train
from sklearn.preprocessing import StandardScaler



def scale_test(x_train,x_test=None):

    scaler=StandardScaler()

    scaler.fit(x_train)

    if x_test is None:

        return pd.DataFrame(scaler.transform(x_train),columns=x_train.columns)

    return pd.DataFrame(scaler.transform(x_test),columns=x_test.columns)
#x_train=scale_test(x_train)
x_train
def prepare_test(x_train, x_test):

  x_test = CExist(x_test)

  x_test = normalize_titles(x_test)

  x_test = name_length(x_test)

  x_test = fill_age(x_train, x_test)

  x_test = age_categorize(x_test)

  x_test = fare_categorize(x_test)

  x_test = family(x_test)

  x_test = get_dummies_t(x_test)

  x_test = drop_text(x_test)

  #x_test = scale_test(x_train, x_test)



  return x_test
prepare_test(x_train,x_test)
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report



def classification_metrics(y_test,predict):

    print("Confusion Matrix:\n")

    print(confusion_matrix(y_test,predict))

    print("\nAccuracy: ",accuracy_score(y_test,predict))

    print("\nClassification report:\n")

    print(classification_report(y_test,predict))
from sklearn.linear_model import RidgeClassifier



rc= RidgeClassifier(class_weight=None, solver='auto', fit_intercept=True,tol=0.001)

rc.fit(x_train,y_train)

predict=pd.DataFrame(data=rc.predict(prepare_test(x_train,x_test)),index=x_test.index)
classification_metrics(y_test,predict)
from sklearn.linear_model import LogisticRegression



lr=LogisticRegression(max_iter=10000,random_state=0)

lr.fit(x_train,y_train)

predict=pd.DataFrame(data=lr.predict(prepare_test(x_train,x_test)),index=x_test.index)

classification_metrics(y_test,predict)
from sklearn.svm import SVC

svC=SVC(random_state=0)

svC.fit(x_train,y_train)

predict=pd.DataFrame(data=svC.predict(prepare_test(x_train, x_test)),index=x_test.index)

classification_metrics(y_test,predict)
from sklearn.tree import DecisionTreeClassifier

dtc=DecisionTreeClassifier(random_state=0)

dtc.fit(x_train,y_train)

predict=pd.DataFrame(data=dtc.predict(prepare_test(x_train,x_test)),index=x_test.index)

classification_metrics(y_test,predict)
from sklearn.ensemble import ExtraTreesClassifier

etc=ExtraTreesClassifier(random_state=0, n_estimators=10)

etc.fit(x_train,y_train)

predict=pd.DataFrame(data=etc.predict(prepare_test(x_train,x_test)),index=x_test.index)

classification_metrics(y_test,predict)
from sklearn.ensemble import RandomForestClassifier

rfC = RandomForestClassifier(criterion='entropy', max_depth= 8, max_leaf_nodes=20, min_samples_leaf=4, n_estimators= 600, random_state=0)



rfC.fit(x_train, y_train)

predict = pd.DataFrame(data=rfC.predict(prepare_test(x_train, x_test)), index = x_test.index)

classification_metrics(y_test, predict)
from sklearn.ensemble import AdaBoostClassifier



abC = AdaBoostClassifier(random_state=0, n_estimators=7, learning_rate=0.9)

abC.fit(x_train, y_train)

predict = pd.DataFrame(data=abC.predict(prepare_test(x_train, x_test)), index = x_test.index)

classification_metrics(y_test, predict)
from sklearn.ensemble import GradientBoostingClassifier



gbC = GradientBoostingClassifier(random_state=0, n_estimators=10)

gbC.fit(x_train, y_train)

predict = pd.DataFrame(data=gbC.predict(prepare_test(x_train, x_test)), index = x_test.index)

classification_metrics(y_test, predict)
from xgboost import XGBClassifier



xgB = XGBClassifier(random_state=0)

xgB.fit(x_train, y_train)

predict = pd.DataFrame(data=xgB.predict(prepare_test(x_train, x_test)), index = x_test.index)

classification_metrics(y_test, predict)
#Importing the auxiliar and preprocessing librarys 

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.pipeline import Pipeline



from sklearn.model_selection import train_test_split, KFold, cross_validate

from sklearn.metrics import accuracy_score



#Models

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

from sklearn.svm import SVC

from sklearn.linear_model import RidgeClassifier, SGDClassifier, LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_selection import SelectFromModel

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, BaggingClassifier, VotingClassifier, RandomTreesEmbedding



clfs = []

seed = 3



clfs.append(("LogReg", 

             Pipeline([("Scaler", StandardScaler()),

                       ("LogReg", LogisticRegression())])))



clfs.append(("XGBClassifier",

             Pipeline([("Scaler", StandardScaler()),

                       ("XGB", XGBClassifier())]))) 

clfs.append(("KNN", 

             Pipeline([("Scaler", StandardScaler()),

                       ("KNN", KNeighborsClassifier())]))) 



clfs.append(("DecisionTreeClassifier", 

             Pipeline([("Scaler", StandardScaler()),

                       ("DecisionTrees", DecisionTreeClassifier())]))) 



clfs.append(("RandomForestClassifier", 

             Pipeline([("Scaler", StandardScaler()),

                       ("RandomForest", RandomForestClassifier())]))) 



clfs.append(("GradientBoostingClassifier", 

             Pipeline([("Scaler", StandardScaler()),

                       ("GradientBoosting", GradientBoostingClassifier(max_features=15, n_estimators=150))]))) 



clfs.append(("RidgeClassifier", 

             Pipeline([("Scaler", StandardScaler()),

                       ("RidgeClassifier", RidgeClassifier())])))



clfs.append(("BaggingRidgeClassifier",

             Pipeline([("Scaler", StandardScaler()),

                       ("BaggingClassifier", BaggingClassifier())])))



clfs.append(("ExtraTreesClassifier",

             Pipeline([("Scaler", StandardScaler()),

                       ("ExtraTrees", ExtraTreesClassifier())])))



#'neg_mean_absolute_error', 'neg_mean_squared_error','r2'

scoring = 'accuracy'

n_folds = 10



results, names  = [], [] 

params = []

for name, model  in clfs:

    #kfold = KFold(n_splits=n_folds, random_state=seed, shuffle=True)

    cv_results = cross_val_score(model, x_train, y_train, cv= 5, scoring=scoring, n_jobs=-1)    

    params.append(model.get_params)

    names.append(name)

    results.append(cv_results)    

    msg = "%s: %f (+/- %f)" % (name, cv_results.mean(),  cv_results.std())

    print(msg)
params

testData=pd.read_csv('../input/titanic/test.csv')

sub=pd.read_csv('../input/titanic/gender_submission.csv',index_col='PassengerId')

testData.columns=map(str.lower,testData.columns)

prepare_test(x_train,testData)
accuracies = []

for model in [rc, lr, svC, dtc, etc, rfC, abC, gbC, xgB]:

  predict = model.predict(prepare_test(x_train, testData))

  accuracies.append((accuracy_score(sub, predict), model))

sorted(accuracies, key=lambda x: x[0])[-1]
accuracies
testData['Survived']=abC.predict(prepare_test(x_train,testData))
testData
testData.rename(columns={"passengerid":"PassengerId"},inplace=True)
testData[["PassengerId","Survived"]].to_csv("submission.csv",index=False)