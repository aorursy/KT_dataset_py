import numpy as np
import pandas as pd 
import math
import seaborn as sns
import matplotlib as plt
%matplotlib inline

df = pd.read_csv("../input/all-space-missions-data-set/spacemissions.csv")
df.head()

sns.heatmap(df.isnull(),yticklabels= False, cbar= False)
sns.countplot('Status Mission',data=df)
sns.countplot('Status Mission',hue='Status Rocket',data=df)
sns.distplot(df['Year'],bins=40,kde=False)
## Data Cleaning

df.drop('Million$',axis=1,inplace = True)
df.head()
sns.heatmap(df.isnull(),yticklabels=False,cbar=False)
## Converting categorical values
len(df['Country'])
rocket=pd.get_dummies(df['Status Rocket'],drop_first=True)
rocket.head()

status=pd.get_dummies(df['Status Mission'])
status.head()

df.drop(['Status Rocket','Status Mission'],axis =1, inplace=True)

df.head()

df=pd.concat([df,rocket],axis=1)
df.head()
df=pd.concat([df,status],axis=1)
df.head()
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
df["Month"] = lb_make.fit_transform(df["Month"])

df["Country"] = lb_make.fit_transform(df["Country"])
df["Company Name"] = lb_make.fit_transform(df["Company Name"])

df.head()

## Building the logistic regression model
df.drop('Success',axis=1).head()
df.drop(['Failure','Partial Failure','Prelaunch Failure'],axis =1,inplace=True)
df.head()
df.drop('Success',axis=1).head()
df['Success'].head()
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(df.drop('Success',axis=1),df['Success'],test_size=0.30,random_state=101)
## Training and Predicting
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,Y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import confusion_matrix
accuracy= confusion_matrix(Y_test,predictions)
accuracy
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(Y_test,predictions)
accuracy
predictions

