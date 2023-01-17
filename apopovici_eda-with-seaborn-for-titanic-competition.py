import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('seaborn-pastel')

%matplotlib inline
# The contest files are located in /kaggle/input. Let´s see what is there: 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Import train and test data as DataFrame using pandas'read_csv function

import pandas as pd

train_data=pd.read_csv("/kaggle/input/titanic/train.csv")

test_data=pd.read_csv("/kaggle/input/titanic/test.csv")
# View the data structure 

print("Traininig data set has {} rows and {} columns".format(train_data.shape[0],train_data.shape[1]))

print(train_data.columns)

print("-"*100)

print("Test data set has {} rows and {} columns".format(test_data.shape[0],test_data.shape[1]))

print(test_data.columns)

#train_data.info()
# View first 5 rows of train data

train_data.head()
# Check the overall survival rate

plt.rcParams["figure.figsize"] = (5,5)

train_data["Survived"].value_counts(normalize=True).plot(kind="pie",autopct='%.2f')
train_data.isnull().sum()
plt.rcParams["figure.figsize"] = (20,10)



plt.subplot(1,3,1)

train_data["Pclass"].value_counts(normalize=True).plot(kind="pie",autopct='%.2f')



plt.subplot(1,3,2)

train_data["Sex"].value_counts(normalize=True).plot(kind="pie",autopct='%.2f')



plt.subplot(1,3,3)

train_data["Embarked"].value_counts(normalize=True).plot(kind="pie",autopct='%.2f')



plt.show()
plt.rcParams["figure.figsize"] = (20,10)



plt.subplot(1,3,1)

train_data["Pclass"].value_counts(normalize=True).plot(kind="pie",autopct='%.2f')





plt.subplot(1,3,2)

train_data["Sex"].value_counts(normalize=True).plot(kind="pie",autopct='%.2f')



plt.subplot(1,3,3)

train_data["Embarked"].value_counts(normalize=True).plot(kind="pie",autopct='%.2f')



plt.show()
pd.crosstab(train_data.Pclass,train_data.Survived,margins=False,normalize='index')
sns.factorplot('Pclass','Survived','Sex',col='Embarked',data=train_data)

plt.show()
women=train_data.loc[train_data.Sex=="female"]["Survived"]

rate_women=sum(women)/len(women)

print("% of women survived: ",rate_women)
men=train_data.loc[train_data.Sex=="male"]["Survived"]

rate_men=sum(men)/len(men)

print("% of men survived: ",rate_men)
from sklearn.ensemble import RandomForestClassifier

y=train_data["Survived"]

features=["Pclass","Sex","SibSp","Parch"]

X=pd.get_dummies(train_data[features])

X_test=pd.get_dummies(test_data[features])



model=RandomForestClassifier(n_estimators=100,max_depth=5,random_state=1)

model.fit(X,y)

predictions=model.predict(X_test)



output=pd.DataFrame({'PassengerId':test_data.PassengerId, 'Survived':predictions})

output.to_csv("my_submission.csv",index=False)

print("output generated")
