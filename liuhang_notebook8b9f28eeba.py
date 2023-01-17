import pandas as pd

from pandas import Series,DataFrame

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC,LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB
titanic_df = pd.read_csv("../input/train.csv",dtype={"Age":np.float},)

test_df= pd.read_csv("../input/test.csv",dtype={"Age":np.float},)



titanic_df.head()
titanic_df.info()

print("------------------------------------------")

test_df.info()
titanic_df = titanic_df.drop(['PassengerId','Name','Ticket'],axis=1)

test_df = test_df.drop(['Name','Ticket'],axis=1)
titanic_df["Embarked"] = titanic_df["Embarked"].fillna("S")



sns.factorplot('Embarked','Survived',data=titanic_df,size=2,aspect=3)



fig,(axis1,axis2,axis3) = plt.subplots(1,3,figsize= (15,5))



sns.countplot(x='Embarked',data=titanic_df,ax=axis1)

sns.countplot(x='Survived',data = titanic_df,order=[1,0],ax=axis2)

embark_perc = titanic_df[["Embarked","Survived"]].groupby(['Embarked'],as_index=False).mean()

sns.barplot(x = "Embarked",y='Survived',data=embark_perc,order=['S','C','Q'],ax=axis3)







test_df['Fare'].fillna(test_df["Fare"].median(),inplace=True)



titanic_df['Fare'] = titanic_df['Fare'].astype(int)

test_df['Fare'] = test_df['Fare'].astype(int)



fare_not_survived = titanic_df["Fare"][titanic_df["Survived"] ==0]

fare_survived = titanic_df["Fare"][titanic_df["Survived"] ==1]





average_fare = DataFrame([fare_not_survived.mean(),fare_survived.mean()])

std_fare = DataFrame([fare_not_survived.std(),fare])


