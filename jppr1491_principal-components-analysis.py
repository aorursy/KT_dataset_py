#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt


# In[6]:


full_data = pd.read_csv(r'\Users\Pedro\Documents\high_diamond_ranked_10min.csv')


# In[11]:


# full_data.head()


# In[8]:


df = full_data.drop(['gameId'], axis = 1)


# In[10]:


# df.head()


# In[12]:


scaled_data = preprocessing.scale(df.T)


# In[14]:


pca = PCA()
pca.fit(scaled_data)


# In[15]:


pca_data = pca.transform(scaled_data)


# In[16]:


percent_var = np.round(pca.explained_variance_ratio_*100, decimals=1)


# In[18]:


labels = ['PC' + str(x) for x in range(1, len(percent_var)+1)]
plt.figure(figsize=(20,20))
plt.bar(x=range(1, len(percent_var)+1), height = percent_var, tick_label = labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show()


# In[19]:


# for this data we see that the two fist principal components are enough to explain the variance as shown in the graph


# In[21]:


# df.columns


# In[22]:


pca_df = pd.DataFrame(pca_data, index = [df.columns], columns = labels)


# In[33]:


plt.figure(figsize=(20,20))
plt.scatter(pca_df.PC1, pca_df.PC2)
plt.title('Principal Components')
plt.xlabel('PC1 - {0}'.format(percent_var[0]))
plt.ylabel('PC2 - {0}'.format(percent_var[1]))
for sample in pca_df.index:
    plt.annotate(sample, (pca_df.PC1.loc[sample], pca_df.PC2.loc[sample]))
plt.show()


# In[38]:


loading_scores = pd.Series(pca.components_[0], index = full_data['gameId'])
sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)

top_10 = sorted_loading_scores[0:20].index.values
print(loading_scores[top_10])


# In[ ]:


# we see 2 clusters that spiit all 40 features and thus suggesting a strong correlation between each other.
# the score values are similar if not all equal suggesting that every match registered in the dataset contribuited for this
# separation 
# this pca suggests that if we want to predict the outcome of a match 2 variables are enough, one from each cluster


# In[ ]: