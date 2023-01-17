# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


train = pd.read_csv("../input/train.csv")

train
gender = pd.read_csv("../input/gender_submission.csv")



import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd
from sklearn import preprocessing,tree
train.head()
train.describe()
train.groupby('Pclass').mean()
class_sex_grouping=train.groupby(['Pclass','Sex']).mean()
class_sex_grouping

class_sex_grouping['Survived'].plot.bar()
train.count()
train=train.drop(['Name','Ticket','Fare'],axis=1)
train
train1=train
#train1["Cabin"] = train1['Cabin'].fillna("NA")
train1.count()
train2=train1.dropna(subset=['Age'])
train2.count()

train2=train2.drop(['Cabin'],axis=1)
#preprocessin
train2['Embarked']=train2['Embarked'].fillna('NA')

def pre_train2(df):
    pre_df=df.copy()
    le=preprocessing.LabelEncoder()
    pre_df.Sex=le.fit_transform(pre_df.Sex)
    #pre_df.Cabin=le.fit_transform(pre_df.Cabin)
    pre_df.Embarked=le.fit_transform(pre_df.Embarked)
    return pre_df
preprocessed_train=pre_train2(train2)

preprocessed_train.head()
test = pd.read_csv("../input/test.csv")
test=test.drop(['Cabin'],axis=1)
test['Embarked']=test['Embarked'].fillna('NA')
preprocess_test=pre_train2(test)
preprocess_test['Age']=preprocess_test['Age'].fillna(25)
X=preprocessed_train.drop(['Survived'],axis=1).values
y=preprocessed_train['Survived'].values

#from sklearn.model_selection import train_test_split
#X_train, X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
clf_dt=tree.DecisionTreeClassifier(max_depth=10)
clf_dt.fit(X,y)
#clf_dt.score(X_test,y_test)
preprocess_test
preprocess_test=preprocess_test.drop(['Name'],axis=1)
preprocess_test=preprocess_test.drop(['Fare'],axis=1)
preprocess_test=preprocess_test.drop(['Ticket'],axis=1)
predict=clf_dt.predict(preprocess_test)
mysubmission=pd.DataFrame({'PassengerId':preprocess_test.PassengerId,'Survived':predict})
mysubmission.to_csv("submission.csv",index=False)
mysubmission
temp = pd.read_csv("submission.csv")
temp