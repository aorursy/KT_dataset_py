import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
df=pd.read_csv('../input/train.csv')

df.head()
df.info()
#Total Survived Passengers

sns.set_style("whitegrid")

sns.countplot(x="Survived",data=df)
#Passengers survived in terms of their sex

sns.countplot(x="Survived",data=df,hue="Sex")

#Clearly we can see that the number among those who couldn't survive, the majority were males,

#and among those survived the majority were females
#Exploring those survived in terms of class

sns.countplot(x="Survived",data=df,hue="Pclass")

#Clearly it is seen majority of those who failed to survive were from the lowest available class.
#Exploring the distribution of age of the passengers

sns.distplot(df["Age"].dropna(),kde=False,bins=30)

#As we can see majority of the passengers lie between late teens to early thrities
#Exploring distribution of fares

sns.distplot(df["Fare"],kde=False, bins=50)

#Majority of the passengers bought tickets that were on the cheaper side
sns.heatmap(df.isnull(),yticklabels= False,cbar=False,cmap="viridis")

#We can see a major chunk of the data is missing from Age column, and Cabin column almost all of the data is missing
#We need to impute the missing values with the average age values based on Pclass and Age

#First let us find out the average age based on Pclass and Age using a boxplot

plt.figure(figsize=(10,5))

sns.boxplot(x="Pclass",y="Age",data=df)

#From the box plot we can estimate the average age of the passengers of each class
#Create a function to impute

def impute(cols):

    Age=cols[0]

    Pclass=cols[1]

    if pd.isnull(Age):

        if Pclass==1:

            return 37

        elif Pclass==2:

            return 29

        else:

            return 24

    else:

        return Age
#Replacing the missing values with the average age values estimated from the box plot

df["Age"]=df[["Age","Pclass"]].apply(impute,axis=1)
#Checking the heatmap once again

sns.heatmap(df.isnull(),yticklabels= False,cbar=False,cmap="viridis")

#We can see all the missing values in the Age column is replaced. Since Cabin has most of its values missing, its better to drop the column
#Dropping Cabin column

df.drop("Cabin",axis=1,inplace=True)
sex = pd.get_dummies(df['Sex'],drop_first=True)

embark = pd.get_dummies(df['Embarked'],drop_first=True)
df.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)

#Dropping the columns with categorical variables...
#...instead adding the newly created columns of "sex" and "embark"

df=pd.concat([df,sex,embark],axis=1)
df.columns=["PassengerId","Survived","Pclass","Age","SibSp","Parch","Fare","Sex","Embarked-Q","Embarked-S"]

df.head()

#"1" is actually sex, "Q" and "S" represents embark
X=df.drop("Survived",axis=1) #Dropping Survived, because all other variables are included in X

y=df["Survived"]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)
from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))