#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv("../input/students-performance-in-exams/StudentsPerformance.csv")


# In[3]:


df.head()


# In[4]:


df.isnull().sum()


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


print(df['race/ethnicity'].value_counts())


# In[8]:


labels='group A','group B','group C','group D','group E'
sizes=[89,190,319,262,140]
explode=[0,0,0,0,0]
fig,ax=plt.subplots(figsize=(7,6))
ax.pie(sizes,labels=labels,explode=explode,autopct='%1.1f%%',startangle=90)
ax.axis('equal')
plt.show()


# In[9]:


print(df['parental level of education'].value_counts())


# In[10]:


labels='some college',"associate's degree",'high school','some high school',"bachelor's degree","master's degree"
sizes=[226,222,196,179,118,59]
explode=(0,0,0,0,0,0)
fig,ax=plt.subplots(figsize=(7,6))
ax.pie(sizes,labels=labels,explode=explode,autopct='%1.1f%%',startangle=90)
plt.show()


# In[11]:


print(df['gender'].value_counts())


# In[12]:


labels='female','male'
sizes=[518,482]
explode=(0,0)
fig,ax=plt.subplots(figsize=(7,6))
ax.pie(sizes,labels=labels,explode=explode,autopct='%1.1f%%',startangle=90)

plt.show()


# In[13]:


print(df['test preparation course'].value_counts())


# In[14]:


labels='none','completed'
sizes=[642,358]
explode=(0,0)
fig,ax=plt.subplots(figsize=(7,6))
ax.pie(sizes,labels=labels,explode=explode,autopct='%1.1f%%',startangle=90)
plt.show()


# In[15]:


df['lunch'].value_counts()


# In[16]:


labels='standard','free/reduced'
sizes=[645,355]
explode=(0,0)
fig,ax=plt.subplots(figsize=(7,6))
ax.pie(sizes,labels=labels,explode=explode,autopct='%1.1f%%',startangle=90)
plt.show()


# In[17]:


df.head()


# In[18]:


df['gender']=[1 if each=='female' else 0 for each in df['gender']]


# In[19]:


df.head()


# In[20]:


df['test preparation course']=[1 if each=='completed' else 0 for each in df['test preparation course']]


# In[21]:


df['lunch']=[1 if each=='standard' else 0 for each in df['lunch']]


# In[22]:


def race_ethnicity(race):
    if race=='group A':
        return 1
    elif race=='group B':
        return 2
    elif race=='group C':
        return 3
    elif race=='group D':
        return 4
    else:
        return 5


# In[23]:


df['race/ethnicity']=df['race/ethnicity'].apply(race_ethnicity)


# In[24]:


def education(education):
    if education=="bachelor's degree":
        return 1
    elif education=='some college':
        return 2
    elif education=="master's degree":
        return 3
    elif education=="associate's degree":
        return 4
    elif education=='high school':
        return 5
    else:
        return 6
    
df['parental level of education']=df['parental level of education'].apply(education)


# In[25]:


df.head()


# In[26]:


df['total_score']=df['math score']+df['reading score']+df['writing score']


# In[27]:


df.head()


# In[28]:


y=df['gender']
df.drop(['gender'],axis=1,inplace=True)
df.columns


# In[29]:


df=pd.get_dummies(df,columns=['race/ethnicity','lunch','parental level of education','test preparation course'])
df.info()


# In[30]:


x=(df-np.min(df))/(np.max(df)-np.min(df))
x


# In[31]:


from sklearn.model_selection import train_test_split


# In[32]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[33]:


x_train.shape,x_test.shape,y_train.shape,y_test.shape


# In[34]:


from sklearn.svm import SVC


# In[35]:


svm=SVC(gamma=0.01,C=500,kernel="rbf")
svm.fit(x_train,y_train)


# In[36]:


svm_score=svm.score(x_test,y_test)


# In[37]:


svm_score


# In[38]:


from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix


# In[39]:


logreg=LogisticRegression()


# In[40]:


logreg.fit(x_train,y_train)
score=logreg.score(x_test,y_test)
score


# In[41]:


knn=[]
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5,weights="uniform",metric="euclidean")


# In[42]:


knn.fit(x_train,y_train)
score_knn=knn.score(x_test,y_test)


# In[43]:


score_knn


# In[44]:


results={'SVM':svm_score,'Logistic Regression': score,'KNeighbors':score_knn}


# In[45]:


results


# In[ ]:
