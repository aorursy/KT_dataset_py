# Import some useful libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
%pylab inline

# Load data in pandas dataframe
df = pd.read_csv("../input/train.csv")
df_test=pd.read_csv("../input/test.csv")

#Data srumbling for training(Let's drop some features which are not helpful for prediction)
df_train= df.drop(['Name','PassengerId','Ticket','Cabin','Survived'],axis=1)
print(df_train["Age"].median(skipna=True))
sns.countplot(x='Embarked',data=df_train,palette='Set2')

df_test1=df_test.drop(['Name','PassengerId','Ticket','Cabin'],axis=1)
final_test= pd.get_dummies(df_test1,columns=['Embarked','Pclass'])
label= LabelEncoder()
final_test['Sex']=label.fit_transform(final_test['Sex'])
#Fill missing values in our Age and Embarked columns with median and most embarked value in dataset
df_train['Age'].fillna(df_train['Age'].median(skipna=True),inplace=True)
df_train['Embarked'].fillna(df_train['Embarked'].value_counts().idxmax(),inplace=True)
final_test['Age'].fillna(final_test['Age'].median(skipna=True),inplace=True)
final_test['Fare'].fillna(final_test['Fare'].median(skipna=True),inplace=True)
sns.distplot(df_train['Age'])
df_train.head()
# Using pandas get_dummies to work with Pclass and Embarked column
final_train= pd.get_dummies(df_train,columns=['Embarked','Pclass'])
final_train.head()
# Use sklearn to do labelencoding for 'Sex' category in our final_train dataset
label= LabelEncoder()
final_train['Sex']=label.fit_transform(final_train['Sex'])
final_train.head()
from sklearn.linear_model import LogisticRegression 
model = LogisticRegression()
X=final_train
y=df['Survived']
model.fit(X,y)
final_test.head()
# Predicting Survival on test data
pred_y=model.predict(final_test)
submission= pd.DataFrame(columns=['PassengerId','Survived'])
submission['PassengerId']= df_test['PassengerId']
submission['Survived']= pred_y
submission.head()