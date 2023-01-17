#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


d4 = pd.read_csv('public-covid-19-cases-canada.csv')


# In[3]:


d4.head()


# In[4]:


d4.describe(include = "all")


# In[5]:


pd.options.display.max_rows=None


# In[6]:


d4.describe(include="all")


# In[28]:


d4.dtypes
cols_drop =['case_id']
d4
 


# In[8]:


import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(20,10))
cmap = sns.cubehelix_palette(light=1, as_cmap=True, reverse= True)
sns.heatmap(d4.isnull(),cmap= cmap)


# In[9]:


d4['has_travel_history'].value_counts()


# In[10]:


d4['has_travel_history'].value_counts().idxmax()


# In[11]:


d4['has_travel_history'].replace(np.nan,"t", inplace=True)


# In[12]:


plt.figure(figsize=(20,10))
cmap = sns.cubehelix_palette(light=1, as_cmap=True, reverse= True)
sns.heatmap(d4.isnull(),cmap= cmap)


# In[13]:


d4['locally_acquired'].value_counts()


# In[14]:


d4['locally_acquired'].value_counts().idxmax()


# In[15]:


d4['locally_acquired'].replace(np.nan,"Close contact", inplace=True)


# In[16]:


plt.figure(figsize=(20,10))
cmap = sns.cubehelix_palette(light=1, as_cmap=True, reverse= True)
sns.heatmap(d4.isnull(),cmap= cmap)


# In[17]:


d4.dtypes


# In[19]:


#univariat analysis
d4.describe()


# In[ ]:


import seaborn as sns


# In[ ]:


sns.distplot(d4['provincial_case_id'],hist=False)


# In[ ]:


sns.boxplot(d4['provincial_case_id'])


# In[ ]:


d4['country'].value_counts()
d4['health_region'].value_counts()
d4['sex'].value_counts()


# In[ ]:


plt.figure(figsize=(50,20))
sns.countplot(d4['health_region'])


# In[ ]:


#Bivariate 


# In[ ]:


d4.corr()


# In[ ]:


d5= d4.select_dtypes(include=[np.number])
corrr= d5.corr()
corrr['provincial_case_id'].sort_values(ascending=False)*100


# In[ ]:


sns.boxplot(x="sex",y="provincial_case_id",data=d4)