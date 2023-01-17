#import libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings("ignore")



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/titanic/train.csv")

test=pd.read_csv("../input/titanic/test.csv")

df.head()
df.drop("Name",axis=1,inplace=True)

df.drop("Ticket",axis=1,inplace=True)

df.drop(["Fare"],axis=1,inplace=True)
df.head()
df.isna().sum()
#fill the missing cabin values with mode

df["Cabin"].fillna(str(df["Cabin"].mode().values[0]),inplace=True)
df["Cabin"]=df["Cabin"].apply(lambda x:str(x).replace(' ','')if ' ' in str(x) else str(x))
df["Deck"] = df["Cabin"].str.slice(0,1)
df.drop(["Cabin"],axis=1,inplace=True)
def impute_median(series):

    return series.fillna(series.median())
df.Age=df.Age.transform(impute_median)
df.isnull().sum()
df["Embarked"]=df["Embarked"].fillna("S")
df.isnull().sum()
df['Is_Married'] = np.where(df['SibSp']==1, 1, 0)

df.head()
df["Family_Size"]=df.SibSp+df.Parch

df.head()
df['Elderly'] = np.where(df['Age']>=50, 1, 0)
df.head()
df.dtypes
#encode the categorical data values

from sklearn.preprocessing import LabelEncoder

labelEncoder_Y=LabelEncoder()

df.iloc[:,3]=labelEncoder_Y.fit_transform(df.iloc[:,3].values)

df.iloc[:,7]=labelEncoder_Y.fit_transform(df.iloc[:,7].values)

df.iloc[:,8]=labelEncoder_Y.fit_transform(df.iloc[:,8].values)
df.dtypes
#get a count ofthe number female or male 

df.Sex.value_counts()
sns.countplot(df.Sex,label="count")

plt.show()
#get a count of the number survived or not

df.Survived.value_counts()
#visualize the count 

sns.countplot(df.Survived,label="count")

plt.show()
#create a pair plot

sns.pairplot(df.iloc[:,1:12],hue="Survived")

plt.show()
#get the correlation of the columns

df.iloc[:,1:12].corr()
#visualize the correlation

plt.figure(figsize=(10,10))

sns.heatmap(df.iloc[:,1:12].corr(), annot=True,fmt=".0%")

plt.show()
test.head()
test['Is_Married'] = np.where(test['SibSp']==1, 1, 0)

test.head()
test["Family_Size"]=test.SibSp+test.Parch

test.head()
test['Elderly'] = np.where(test['Age']>=50, 1, 0)

test.head()
test.drop("Name",axis=1,inplace=True)

test.drop("Ticket",axis=1,inplace=True)

test.drop("Fare",axis=1,inplace=True)
test.isnull().sum()
test.Age=test.Age.transform(impute_median)
#fill the missing cabin values with mode

test["Cabin"].fillna(str(test["Cabin"].mode().values[0]),inplace=True)
test["Cabin"]=test["Cabin"].apply(lambda x:str(x).replace(' ','')if ' ' in str(x) else str(x))
test["Deck"] = test["Cabin"].str.slice(0,1)
test.drop(["Cabin"],axis=1,inplace=True)
test.dtypes
#encode the categorical data values

from sklearn.preprocessing import LabelEncoder

labelEncoder_Y=LabelEncoder()

test.iloc[:,2]=labelEncoder_Y.fit_transform(test.iloc[:,2].values)

test.iloc[:,6]=labelEncoder_Y.fit_transform(test.iloc[:,6].values)

test.iloc[:,10]=labelEncoder_Y.fit_transform(test.iloc[:,10].values)

test.head()
#Split the data set into independent(x) and dependent (y) data sets

x=df.iloc[:,2:12].values

y=df.iloc[:,1].values.reshape(-1,1)

x_test  = test.drop("PassengerId",axis=1).copy()
#0.469

#split the data set

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.469,random_state=42)
#scale the data(feature scaling)

from sklearn.preprocessing import StandardScaler



sc=StandardScaler()

x_train=sc.fit_transform(x_train)

x_test=sc.fit_transform(x_test)
x_train.shape
x_test.shape
y_train.shape
y_test.shape
#create a function for the models

def models(x_train,y_train):

  #Logistic Regression Model

  from sklearn.linear_model import LogisticRegression

  log=LogisticRegression(random_state=42)

  log.fit(x_train,y_train)

  

  #Decision Tree

  from sklearn.tree import DecisionTreeClassifier

  tree=DecisionTreeClassifier(criterion='entropy',random_state=0)

  tree.fit(x_train,y_train)

  

  #Random Forest Classifier

  from sklearn.ensemble import RandomForestClassifier

  forest = RandomForestClassifier(n_estimators=15,criterion="entropy",random_state=0)

  forest.fit(x_train,y_train)



  #Print the models accuracy on the training data

  print("[0]Logistic Regression Training Accuracy:",log.score(x_train,y_train))

  print("[1]Decision Tree Classifier Training Accuracy:",tree.score(x_train,y_train))

  print("[2]Random Forest Classifier Training Accuracy:",forest.score(x_train,y_train))

  

  return log,tree,forest
#Getting all of the models

model = models(x_train,y_train)
#test model accuracy on confusion matrix

from sklearn.metrics import confusion_matrix





for i in range(len(model)):

  print("Model ", i)

  cm =confusion_matrix(y_test,model[i].predict(x_test))



  TP=cm[0][0]

  TN=cm[1][1]

  FN=cm[1][0]

  FP=cm[0][1]



  print(cm)

  print("Testing Accuracy = ", (TP+TN) / (TP+TN+FN+FP))

  print()
#show another way to get metrics of the models

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score



for i in range(len(model) ):

  print("Model ",i)

  print( classification_report(y_test,model[i].predict(x_test)))

  print( accuracy_score(y_test,model[i].predict(x_test)))

  print()
pred=model[0].predict(x_test)

print(pred)

PassengerId = test['PassengerId']

submission = pd.DataFrame({ 'PassengerId': PassengerId,

                            'Survived': pred })

submission.to_csv(r'submission.csv',index=False)
from IPython.display import FileLink

FileLink(r'submission.csv')