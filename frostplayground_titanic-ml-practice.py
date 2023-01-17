import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



titanic_df=pd.read_csv('../input/train.csv')

titanic_df
titanic_df.info()
titanic_df['Age'].fillna(titanic_df['Age'].mean(),inplace=True)

titanic_df['Cabin'].fillna('N',inplace=True)

titanic_df['Embarked'].fillna('N',inplace=True)
titanic_df.info()
titanic_df['Sex'].value_counts()
titanic_df['Cabin'].value_counts()
titanic_df['Embarked'].value_counts()
titanic_df['Cabin']=titanic_df['Cabin'].str[:1]

titanic_df['Cabin'].value_counts()
titanic_df.groupby(['Sex','Survived'])['Survived'].count()
sns.barplot(x='Sex',y='Survived',data=titanic_df)
sns.barplot(x='Pclass',y='Survived',hue='Sex',data=titanic_df)
def age_cat(age):

    cat=''

    if age <=-1:

        cat='Unknown'

    elif age <=13:

        cat='Child'

    elif age <=19:

        cat='Teenager'

    elif age <= 30:

        cat='Young Adult'

    elif age <=40:

        cat='Adult'

    elif age <=60:

        cat='Old'

    else :

        cat='Elderly'

    

    return cat
plt.figure(figsize=(10,6))



cat_names=['Unknown','Child','Teenager','Young Adult','Adult','Old','Elderly']



titanic_df['Age_cat']=titanic_df['Age'].apply(lambda x : age_cat(x))
sns.barplot(x='Age_cat',y='Survived',hue='Sex',data=titanic_df,order=cat_names)
print(titanic_df['Cabin'].value_counts())

print(titanic_df['Sex'].value_counts())

print(titanic_df['Embarked'].value_counts())
from sklearn import preprocessing



def encoding_features(data):

    features=['Cabin','Sex','Embarked']

    

    for feature in features:

        le=preprocessing.LabelEncoder()

        le=le.fit(data[feature])

        data[feature]=le.transform(data[feature])

    return data





titanic_df=encoding_features(titanic_df)

titanic_df
# Explain : encoding values

# N : 7    687

# C : 2     59

# B : 1     47

# D : 3    33

# E : 4    32

# A : 0   15

# F : 5    13

# G : 6      4

# T : 8      1

# Name: Cabin, dtype: int64

# male : 1     577

# female : 0   314

# Name: Sex, dtype: int64

# S : 3    644

# C : 0   168

# Q : 2    77

# N : 1     2

# Name: Embarked, dtype: int64
print(titanic_df['Cabin'].value_counts())

print(titanic_df['Sex'].value_counts())

print(titanic_df['Embarked'].value_counts())
def fill_null(data):

    data['Age'].fillna(data['Age'].mean(),inplace=True)

    data['Cabin'].fillna('N',inplace=True)

    data['Embarked'].fillna('N',inplace=True)

    data['Fare'].fillna(0,inplace=True)

    

    return data



def drop_col(data):

    data.drop(['PassengerId','Name','Ticket'],axis=1,inplace=True)

    

    return data



def label_encoding(data):

    data['Cabin']=data['Cabin'].str[:1]

    features=['Cabin','Sex','Embarked']

    for feature in features:

        le=preprocessing.LabelEncoder()

        le=le.fit(data[feature])

        data[feature]=le.transform(data[feature])

        

    return data
titanic_df=pd.read_csv('../input/train.csv')

y_titanic_df=titanic_df['Survived']

x_titanic_df=titanic_df.drop('Survived',axis=1)



x_titanic_df=fill_null(x_titanic_df)

x_titanic_df=drop_col(x_titanic_df)

x_titanic_df=label_encoding(x_titanic_df)
test_df=pd.read_csv('../input/test.csv')



test_df=fill_null(test_df)

test_df=label_encoding(test_df)
X_train = x_titanic_df

y_train = y_titanic_df

X_test  = test_df.drop(['PassengerId','Name','Ticket'], axis=1).copy()

X_train.shape, y_train.shape, X_test.shape
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score



dt_clf=DecisionTreeClassifier()

rf_clf=RandomForestClassifier()

lr_clf=LogisticRegression()



# accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)



dt_clf.fit(X_train,y_train)

dt_pred=dt_clf.predict(X_test)

dt_acc=round(dt_clf.score(X_train,y_train)*100,2)

print(dt_acc)
rf_clf.fit(X_train,y_train)

rf_pred=rf_clf.predict(X_test)

rf_acc=round(rf_clf.score(X_train,y_train)*100,2)

print(rf_acc)
lr_clf.fit(X_train,y_train)

lr_pred=lr_clf.predict(X_test)

# print('LogisticRegression Accuracy : {0:.4f}'.format(accuracy_score(y_test,lr_pred)))

lr_acc=round(lr_clf.score(X_train,y_train)*100,2)

print(lr_acc)
submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": dt_pred

    })



print(submission)



submission.to_csv('submission.csv', index=False)