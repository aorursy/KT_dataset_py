#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


df=pd.read_csv("train.csv")


# In[3]:


df.head()


# In[4]:


sns.heatmap(df.isnull(),yticklabels=False)


# In[5]:


sns.boxplot('Pclass','Age',data=df)


# In[6]:


def input_age(cols):
    Age=cols[0]
    Pclass=cols[1]
    if pd.isnull(Age):
        if Pclass==1:
            return 38
        elif Pclass==2:
            return 29
        else:
            return 24 
    else:
          return Age


# In[7]:


df['Age'] = df[['Age','Pclass']].apply(input_age,axis=1)


# In[8]:


df.head()


# In[9]:


sns.heatmap(df.isnull(),yticklabels=False)


# In[10]:


df.drop('Cabin',axis=1,inplace=True)


# In[11]:


sns.heatmap(df.isnull(),yticklabels=False)


# In[12]:


df.head()


# In[13]:


sex=pd.get_dummies(df['Sex'],drop_first=True).head()


# In[14]:


embarked=pd.get_dummies(df['Embarked'],drop_first=True).head()


# In[15]:


sex.head()


# In[16]:


embarked.head()


# In[17]:


df.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[18]:


df.head()


# In[19]:


pd.concat([df,sex,embarked],axis=1)


# In[51]:


from sklearn.model_selection import train_test_split


# In[52]:


X_train,X_test,y_train,y_test=train_test_split(df.drop('Survived',axis=1),df['Survived'],test_size =0.30,random_state=101)


# In[53]:


from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
import xgboost


# In[54]:


classifier=xgboost.XGBClassifier()


# In[55]:


params={"learning_rate":[0.05,0.10,0.15,0.20,0.25,0.30],"max_depth":[3,4,5,6,7,8,10,12,15],"min_child_weight":[1,3,5,7],"gamma":[0.0,0.1,0.2,0.3,0.4],"colsample_bytree":[0.3,0.4,0.5,0.7]}


# In[56]:


random_search=RandomizedSearchCV(classifier,param_distributions=params,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)


# In[57]:


random_search.fit(X_train,y_train)


# In[58]:


random_search.best_estimator_


# In[59]:


random_search.best_params_


# In[66]:


from sklearn.model_selection import cross_val_score
score=cross_val_score(classifier,X_test,y_test,cv=10)


# In[67]:


score


# In[68]:


score.mean()


# In[70]:


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


# In[75]:


from sklearn.ensemble import RandomForestClassifier
df_model=RandomForestClassifier()
df_model.fit(X_train,y_train)


# In[76]:


y_train_pred=df_model.predict_proba(X_train)


# In[77]:


print('DF train roc-auc:{}'.format(roc_auc_score(y_train,y_train_pred[:,1])))


# In[79]:


y_test_pred=df_model.predict_proba(X_test)


# In[80]:


print('DF train roc-auc:{}'.format(roc_auc_score(y_test,y_test_pred[:,1])))


# In[81]:


from sklearn.linear_model  import LogisticRegression


# In[82]:


log_classifier=LogisticRegression()


# In[83]:


log_classifier.fit(X_train,y_train)


# In[84]:


y_train_pred=log_classifier.predict_proba(X_train)


# In[85]:


print('DF train roc-auc:{}'.format(roc_auc_score(y_train,y_train_pred[:,1])))


# In[86]:


y_test_pred=log_classifier.predict_proba(X_test)


# In[87]:


print('DF test roc-auc:{}'.format(roc_auc_score(y_test,y_test_pred[:,1])))


# In[ ]: