#!/usr/bin/env python

# coding: utf-8



# In[152]:





import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import math

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score, mean_squared_error

get_ipython().run_line_magic('matplotlib', 'inline')





# In[153]:





df=pd.read_csv("Titanic.csv")

df





# In[154]:





sns.countplot(x="Survived",data=df)





# In[155]:





sns.countplot(x="Survived",hue='Sex',data=df)





# In[156]:





sns.countplot(x="Survived",hue='Pclass',data=df)





# In[157]:





sns.countplot(x="Survived",hue="Embarked",data=df)





# In[158]:





df





# In[159]:





df.columns





# In[160]:





sns.countplot(x="Survived",hue="SibSp",data=df)





# In[161]:





df['Age'].plot.hist()





# In[162]:





df['Fare'].plot.hist(bins=20)





# In[163]:





df.info()





# In[164]:





sns.countplot(x='SibSp',data=df)





# In[165]:





df.isnull()





# In[166]:





df.isnull().sum()





# In[167]:





sns.heatmap(df.isnull(),cmap="viridis")





# In[168]:





sns.boxplot(x='Pclass',y="Age",data=df)





# In[169]:





df.head()





# In[170]:





###after droping cabin column





# In[171]:





df.head(5)





# In[172]:





df.dropna(inplace=True)





# In[173]:





sns.heatmap(df.isnull(),cbar=False)





# # NICE



# In[174]:





df.isnull().sum()





# In[175]:





df.head()





# In[176]:





sex=pd.get_dummies(df['Sex'],drop_first=True)





# In[177]:





sex.head(5)





# In[178]:





embark=pd.get_dummies(df['Embarked'],drop_first=True)





# In[179]:





embark





# In[180]:





pc1=pd.get_dummies(df['Pclass'],drop_first=True)





# In[181]:





df=pd.concat([df,sex,embark,pc1],axis=1)





# In[182]:





df.drop(df.columns[[9]],axis=1,inplace=True)

df





# In[184]:





df.drop(['Embarked','Sex','PassengerId','Pclass','Name','Ticket'],axis=1,inplace=True)





# In[190]:





df.drop(['Cabin'],axis=1,inplace=True)





# In[191]:





df





# In[192]:





X=df.drop('Survived',axis=1)

y=df["Survived"]





# In[193]:





X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)





# In[194]:





from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()





# In[195]:





lr.fit(X_train,y_train) 





# In[197]:





predictions=lr.predict(X_test)





# In[198]:





from sklearn.metrics import classification_report





# In[200]:





classification_report(y_test,predictions)





# In[202]:





from sklearn.metrics import confusion_matrix





# In[206]:





confusion_matrix(y_test,predictions)





# In[207]:





from sklearn.metrics import accuracy_score





# In[208]:





accuracy_score(y_test,predictions)