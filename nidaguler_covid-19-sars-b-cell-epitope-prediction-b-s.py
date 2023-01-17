# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import warnings

warnings.filterwarnings("ignore")



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_bcell=pd.read_csv("../input/epitope-prediction/input_bcell.csv")

df_covid=pd.read_csv("../input/epitope-prediction/input_covid.csv")

df_sars=pd.read_csv("../input/epitope-prediction/input_sars.csv")
df_bcell.head()
df_bcell.tail()
df_bcell.isna().sum()
df_bcell.dtypes
from sklearn.preprocessing import LabelEncoder

labelEncoder_Y=LabelEncoder()

df_bcell.iloc[:,0]=labelEncoder_Y.fit_transform(df_bcell.iloc[:,0].values)

df_bcell.iloc[:,1]=labelEncoder_Y.fit_transform(df_bcell.iloc[:,1].values)

df_bcell.iloc[:,4]=labelEncoder_Y.fit_transform(df_bcell.iloc[:,4].values)
df_bcell.head()
df_bcell.dtypes
#visualize the correlation

plt.figure(figsize=(10,10))

sns.heatmap(df_bcell.corr(), annot=True,fmt=".0%")

plt.show()
df_covid.head()
df_covid.tail()
df_covid.isna().sum()
df_covid.dtypes
from sklearn.preprocessing import LabelEncoder

labelEncoder_Y=LabelEncoder()

df_covid.iloc[:,0]=labelEncoder_Y.fit_transform(df_covid.iloc[:,0].values)

df_covid.iloc[:,1]=labelEncoder_Y.fit_transform(df_covid.iloc[:,1].values)

df_covid.iloc[:,4]=labelEncoder_Y.fit_transform(df_covid.iloc[:,4].values)
df_covid.head()
df_covid.dtypes
#visualize the correlation

plt.figure(figsize=(10,10))

sns.heatmap(df_covid.corr(), annot=True,fmt=".0%")

plt.show()
df_sars.head()
df_sars.tail()
df_sars.isna().sum()
df_sars.dtypes
from sklearn.preprocessing import LabelEncoder

labelEncoder_Y=LabelEncoder()

df_sars.iloc[:,0]=labelEncoder_Y.fit_transform(df_sars.iloc[:,0].values)

df_sars.iloc[:,1]=labelEncoder_Y.fit_transform(df_sars.iloc[:,1].values)

df_sars.iloc[:,4]=labelEncoder_Y.fit_transform(df_sars.iloc[:,4].values)
df_sars.head()
df_sars.dtypes
#visualize the correlation

plt.figure(figsize=(10,10))

sns.heatmap(df_sars.corr(), annot=True,fmt=".0%")

plt.show()
#Split the data set into independent(x) and dependent (y) data sets

x=df_bcell.iloc[:,1:14].values

y=df_bcell.iloc[:,0].values.reshape(-1,1)

x_test  = df_sars.drop("parent_protein_id",axis=1).copy()
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
pred=model[2].predict(x_test)

print(pred)