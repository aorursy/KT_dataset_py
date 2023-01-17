# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import missingno
from sklearn import preprocessing
labelEncoder = preprocessing.LabelEncoder()

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
        
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_train = pd.read_csv('../input/titanic/train.csv')
df_test = pd.read_csv('../input/titanic/test.csv')
df_gender_submission = pd.read_csv('../input/titanic/gender_submission.csv')
df_test = pd.merge(df_test,df_gender_submission,on="PassengerId",how="inner")
df_test.head()
missingno.matrix(df_test, figsize = (30,10))
df_train.info()
def f1(df):
    df[['Age']] = df[['Age']].fillna(0)
    df[['Fare']] = df[['Fare']].fillna(0)
    df[['Cabin']] = df[['Cabin']].fillna('')
    df[['Embarked']] = df[['Embarked']].fillna('')  
    return df
df_train = f1(df_train)
df_test = f1(df_test)
df_train.Sex.value_counts()
def f_sex(df):
    df.Sex = df.Sex.replace('male',0)
    df.Sex = df.Sex.replace('female',1)
    return df
df_train = f_sex(df_train)
df_test = f_sex(df_test)
df_train.SibSp.value_counts()
df_train['Pclass'].value_counts()
df_train[['Survived','Sex','Age','SibSp','Parch','Fare']]
train_X = df_train[['Sex','Age','SibSp','Parch','Fare']]
test_X = df_test[['Sex','Age','SibSp','Parch','Fare']]
train_Y = df_train[['Survived']].values.ravel()
test_Y = df_test[['Survived']].values.ravel()
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import metrics
classifierType =[[DecisionTreeClassifier(), 'DecisionTreeClassifier'],[GaussianNB(), 'Naïve Bayes'],[MLPClassifier(),'Neural Network']]
classifierScore=[]
sumFMeasure =0
cv = KFold(n_splits=5, random_state=1, shuffle=True) #ให้แบ่งชุดข้อมูลเป็น 5-fold cross validation
for typeC in classifierType:
    model = typeC[0]
#     model.fit(train_X ,train_Y)
    scores = cross_val_score(model, train_X, train_Y, scoring='accuracy', cv=cv, n_jobs=-1)
    score = model.score(test_X ,test_Y)
    recall = metrics.recall_score(test_Y,model.predict(test_X))
    precision = metrics.precision_score(test_Y,model.predict(test_X))
    f1 = metrics.f1_score(test_Y,model.predict(test_X))
    classifierScore.append({'score':score,'precision':precision,'recall':recall,'F-Measure':f1,'classifierName': typeC[1]})
    sumFMeasure += f1
    print('classifierName :',typeC[1])
    print('accuracy : ',score)
    print('Recall : ',recall)
    print('Precision : ',precision)
    print('F-Measure : ',f1)
    print('5-fold cross validation Score',np.average(scores))
    print()
print('Average F-Measure : ',sumFMeasure/3)