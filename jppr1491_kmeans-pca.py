#!/usr/bin/env python
# coding: utf-8

# In[44]:


import pandas as pd
import numpy as np
from numpy import arange
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# In[22]:


full_data = pd.read_csv(r'\Users\Pedro\Documents\high_diamond_ranked_10min.csv')


# In[23]:


# just throw the model at the data and see where it goes


# In[24]:


full_data.head()


# In[39]:


no_gameId = full_data.drop(['gameId'], axis=1)
cluster = KMeans(n_clusters =2)
full_data['blueWinsPred'] = cluster.fit_predict(no_gameId)


# In[40]:


full_data.head()


# In[41]:


blueWins = full_data['blueWins'].to_numpy()
blueWinsPred = full_data['blueWinsPred'].to_numpy()
goodPred = 0
for i in range(len(blueWins)):
    if blueWins[i]==blueWinsPred[i]:
        good += 1

print(good/len(blueWins))


# In[ ]:


# 76% good predictions 
# now lets try PCA and see if we can get better prediction
# In a previous file we saw that 2 components were enough to explain 98% of the variance of the data


# In[42]:


pca = PCA(n_components = 2)
full_data['x'] = pca.fit_transform(full_data)[:,0]
full_data['y'] = pca.fit_transform(full_data)[:,1]


# In[43]:


full_data.head()


# In[52]:


plt.figure(figsize=(20,20))
plt.scatter(full_data['x'],full_data['y'], c = full_data['blueWinsPred'])
# yellow corresponds to blueWinsPred == 1 
# purple was predicted to blueWinsPred == 0


# In[ ]: