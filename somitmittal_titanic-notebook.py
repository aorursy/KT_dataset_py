import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



data_train = pd.read_csv('../input/train.csv')

data_test = pd.read_csv('../input/test.csv')

#data_train.sample(2)

#data_test.sample(2)

#data_train.describe()

#data_train.Embarked=data_train.Embarked.fillna(0)

#data_train['Embarked'].describe()
def simplify_ages(df):

    parameter=df['Age'].mean()

    df.Age.fillna(parameter,inplace=True)

    bins=[0,5,14,20,35,60,90]

    group_names=['Baby','Teenager','Student','YoungAdult','Adult','Senior']

    categories = pd.cut(df.Age, bins, labels=group_names)

    df.Age=categories

    return df



def embark(df):

    df.Embarked=df.Embarked.fillna("C")

    return df

    

def drop_features(df):

    df=df.drop(['Ticket','Name','Cabin','Fare'],axis=1)

    return df



def transform_features(df):

    df=simplify_ages(df)

    df=drop_features(df)

    df=embark(df)

    return df



data_train=transform_features(data_train)

data_test=transform_features(data_test)

#data_train.describe()

#data_train.head()
from sklearn import preprocessing

def encode_features(df_train,df_test):

    features=['Age','Sex','Embarked']

    df=pd.concat([df_train[features],df_test[features]])

    for feature in features:

        le=preprocessing.LabelEncoder()

        le = le.fit(df[feature])

        df_train[feature] = le.transform(df_train[feature])

        df_test[feature] = le.transform(df_test[feature])

    return df_train, df_test



data_train,data_test=encode_features(data_train,data_test)

#data_train.head()

#data_test.head()

        
from sklearn.model_selection import train_test_split

features=data_train.drop(['PassengerId','Survived'],axis=1)

label=data_train['Survived']



X_train,X_test,Y_train,Y_test=train_test_split(features,label,test_size=0.20,random_state=23)
from sklearn.ensemble import RandomForestClassifier

from sklearn import svm

from sklearn.grid_search import GridSearchCV

from sklearn.metrics import make_scorer,accuracy_score



#clf=RandomForestClassifier()

clf=svm.SVC(C=10,kernel='rbf',gamma=1)



'''parameters={'n_estimators':[4,6,9],'max_features':['log2', 'sqrt','auto'], 

            'criterion': ['entropy', 'gini'],

            'max_depth': [2, 3, 5, 10], 

            'min_samples_split': [2, 3, 5],

            'min_samples_leaf': [1,5,8]

            }

acc_scorer=make_scorer(accuracy_score)

grid_obj=GridSearchCV(clf,parameters,scoring=acc_scorer)

grid_obj=grid_obj.fit(X_train,Y_train)

clf = grid_obj.best_estimator_

'''

# Fit the best algorithm to the data. 

clf.fit(X_train, Y_train)
predictions=clf.predict(X_test)

#print(accuracy_score(Y_test,predictions))
ids=data_test['PassengerId']

predictions=clf.predict(data_test.drop('PassengerId',axis=1))



out=pd.DataFrame({"PassengerId":ids,"Survived":predictions})

out.to_csv('submissions.csv', index = False)

#out.sample(5)