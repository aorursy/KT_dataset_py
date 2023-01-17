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
# import dataset:

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# import dataset :

df=pd.read_csv("../input/train.csv")
df.head()
# some missing we can see with the help of seaborn:

# visualization:

sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
# yellow is missing data:

# so now going dealing with missing data:

sns.set_style('whitegrid')
# with sex:

sns.countplot(x='Survived',data=df,hue='Sex',palette='RdBu_r')
# now passengerclass:

sns.countplot(x='Survived',hue='Pclass',data=df,palette='RdBu_r')
# age of the people on titanic:

sns.distplot(df['Age'].dropna(),bins=20,kde=False)
# explore other the column:

df.info()
sns.countplot(x='SibSp',data=df)
# Now check the fair (The price he pay):

sns.distplot(df['Fare'],bins=20,kde=False)
# Cleaning a data in this part:

sns.boxplot(x='Pclass',y='Age',data=df)
# filling the age cloumn:

def input_age(cols):

    Age=cols[0]

    Pclass=cols[1]

    

    if(pd.isnull(Age)):

        if(Pclass==1):

            return 37

        if(Pclass==2):

            return 29

        else:

            return 24

    else:

        return Age
df['Age']=df[['Age','Pclass']].apply(input_age,axis=1)
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
# for cabin column:

df.drop('Cabin',axis=1,inplace=True)
df.head()
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
# cleaning the dataset( this make for machine learning model):

pd.get_dummies(df['Sex'])
sex=pd.get_dummies(df['Sex'],drop_first=True)
embarked=pd.get_dummies(df['Embarked'],drop_first=True)
df=pd.concat([df,sex,embarked],axis=1)
df.head()
df.drop(['Sex','Embarked','Ticket'],axis=1,inplace=True)
df.head()
df.drop('PassengerId',axis=1,inplace=True)
df.head()
df.drop('Name',axis=1,inplace=True)
df.head()
# make dataset into dependent and independent set:

X=df.iloc[:,1:8].values

y=df.iloc[:,0].values
X
y
#  spliting the dataset into train and test set:

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
# feature scaling :

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train=sc.fit_transform(X_train)

X_test=sc.fit_transform(X_test)
#fiting the model into the Logistic classification:

from sklearn.linear_model import LogisticRegression

classifier=LogisticRegression()

classifier.fit(X_train,y_train)
# prediction of new result:

y_pred=classifier.predict(X_test)
# making the classification repeort and confusion matrix:

from sklearn.metrics import classification_report,confusion_matrix

cm=confusion_matrix(y_test,y_pred)
cm
cr=classification_report(y_test,y_pred)
cr
(148+71)/268