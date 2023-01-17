import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



# Machine Learning

import xgboost as xgb

from xgboost import XGBClassifier



sns.set_style("darkgrid")
df = pd.read_csv("../input/train.csv")

df.sample(5)
df.shape
df.info()
df_test = pd.read_csv("../input/test.csv")

df_test.sample(5)
df_test.shape
df_test.info()
example = pd.read_csv("../input/gender_submission.csv")

example.head(5)
example.shape
df['PassengerId'].nunique() == df.shape[0]
print("Class frequency:")

for index,value in zip(df['Pclass'].value_counts().index,df['Pclass'].value_counts().values):

    print("Class {} => {} ({:.2f}%)".format(index,value,(value/df.shape[0])*100))
sns.countplot(x="Pclass",hue="Survived",data=df,palette="YlGnBu")
df['Name'].apply(lambda x: x.split(".")[0].split(",")[1]).value_counts()
print("Frequency:")

for index,value in zip(df['Sex'].value_counts().index,df['Sex'].value_counts().values):

    print("{} => {} ({:.2f}%)".format(index.capitalize(),value,(value/df.shape[0])*100))
sns.countplot(x="Sex",hue="Survived",data=df,palette="YlGnBu")
print("Age:\nMin: {} | Max: {} | Mean: {} | Std: {}".format(df.Age.min(),df.Age.max(),df.Age.mean(),df.Age.std()))
sns.distplot(df['Age'].dropna())
sns.boxplot(hue="Sex",y="Age",x="Survived",data=df)
sns.countplot(df['SibSp'])
sns.countplot(x="SibSp",hue="Survived",data=df)
sns.countplot(df['Parch'])
sns.countplot(x="Parch",hue="Survived",data=df)
df['Ticket'].value_counts().head()
df[df['Ticket']=='CA. 2343']
print("Age:\nMin: {} | Max: {} | Mean: {} | Std: {}".format(df.Fare.min(),df.Fare.max(),df.Fare.mean(),df.Fare.std()))
df[~df.Cabin.isna()].sample(10)
sns.countplot(x="Embarked",hue="Survived",data=df,palette="YlGnBu")
def extractFeatures(df,type="train"):

    df['Sex'] = df['Sex'].apply(lambda x: np.where(x=='male',1,0))

    df['Age'].fillna((int(df['Age'].mean())+1),inplace=True)

    df['Cabin_letter'] = df['Cabin'].str[0]

    df['Cabin_num'] = df['Cabin'].apply(lambda x: x.split(" ")[0][1:] if isinstance(x, str) else "X")

    df = pd.concat([df,pd.get_dummies(df['Cabin_letter'],prefix="Cabin")],axis=1)

    df = pd.concat([df,pd.get_dummies(df['Embarked'],prefix="Embarked")],axis=1)

    if type=="train":

        return df[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked_C','Embarked_S','Cabin_A','Cabin_B','Cabin_C','Cabin_D','Cabin_E','Cabin_F','Cabin_G','Cabin_num','Survived']]

    else:

        return df[['PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked_C','Embarked_S','Cabin_A','Cabin_B','Cabin_C','Cabin_D','Cabin_E','Cabin_F','Cabin_G','Cabin_num']]
df = extractFeatures(df)
df_test = extractFeatures(df_test,type="test")
df.corr()
df2 = df[(df['Cabin_num']!='X')&(df['Cabin_num']!='')]

df2_test = df_test[(df_test['Cabin_num']!='X')&(df_test['Cabin_num']!='')]

df2['Cabin_num'] = df2['Cabin_num'].astype(int)

df2_test['Cabin_num'] = df2_test['Cabin_num'].astype(int)
df = df[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked_C','Embarked_S','Survived']]

df_test = df_test[['PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked_C','Embarked_S']]
X_train,y_train = df.iloc[:,:-1],df.iloc[:,-1]

X_train2,y_train2 = df2.iloc[:,:-1],df2.iloc[:,-1]
model = XGBClassifier()

model.fit(X_train,y_train)
y_pred = model.predict(df_test.drop(['PassengerId'],axis=1))
df_test['Survived'] = y_pred

sub = df_test[['PassengerId','Survived']]
#sub = pd.read_csv("../output/submission1.csv")
model.fit(X_train2,y_train2)

y_pred = model.predict(df2_test.drop(['PassengerId'],axis=1))

df2_test['Survived'] = y_pred

sub2 = df2_test[['PassengerId','Survived']]
#sub.to_csv('submission1.csv',index=False)
sub2
sub = sub.merge(sub2,on="PassengerId",how="left")
sub['Survived_y'] = sub['Survived_y'].fillna(-1)
sub['Survived'] = sub.apply(lambda x: x['Survived_y'] if ((x['Survived_y']!=-1)and(x['Survived_x']!=x['Survived_y'])) else x['Survived_x'],axis=1)
sub['Survived'] = sub['Survived'].astype(int)
sub[['PassengerId','Survived']].to_csv("submission2.csv",index=False)
#sns.countplot(x="Cabin_letter",hue="Survived",data=df,palette="YlGnBu")
#df[df['Cabin_letter']=='E']