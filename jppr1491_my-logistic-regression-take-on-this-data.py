#!/usr/bin/env python
# coding: utf-8

# In[82]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[70]:


df = pd.read_csv(r'\Users\Pedro\Documents\high_diamond_ranked_10min.csv')


# In[71]:


df.head()


# In[72]:


# check all the variables
df.info()


# In[73]:



fig, ax = plt.subplots(figsize=(15,15)) 
sns.heatmap(df.corr(), 
        xticklabels=df.corr().columns,
        yticklabels=df.corr().columns, ax=ax)


# In[74]:


# from the correlation matrix we see that gold, kills, assists, gold difference, cs, mosters and toers 
# from the same team are highly correlated. 
# So for our first model, from the mention above we will only use both teams total gold
# since we will be trying to predict the result for the blue team we will exclude red team first blood 
# we will exclude total experience because its highly correlated with average level which we will use 


# In[75]:


df_model = df[['blueWins', 'blueWardsPlaced', 'blueWardsDestroyed', 'blueTotalGold', 'blueAvgLevel', 'redWardsPlaced',
              'redWardsDestroyed', 'redTotalGold', 'redAvgLevel']]


# In[76]:


# df_model.corr() 
# the calculations show a correlation of 0.61 between total gold and average level but still I want to keep those


# In[80]:


X = df_model[['blueWardsPlaced', 'blueWardsDestroyed', 'blueTotalGold', 'blueAvgLevel', 'redWardsPlaced',
              'redWardsDestroyed', 'redTotalGold', 'redAvgLevel']]
Y = df_model[['blueWins']]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,train_size=0.75, random_state=69)


# In[128]:


model_1 = LogisticRegression(solver='liblinear')


# In[129]:


model_1.fit(X_train,Y_train)


# In[130]:


model_1.predict(X_test)


# In[132]:


#model_1.predict_proba(X_test)


# In[133]:


model_1.score(X_test,Y_test)


# In[134]:


# so now we need to add variables possibly some that we thought we would be needing
# At this point we are using the forward method (we keep adding variables until our model score gets better)
# Later we maybe we need to remove some turning into a step wise
# so where are we at the moment? we decided that to predict the outcome for the blue team we needed the TotalGold, WardsPlaced 
# and destroyed by the opposing team and both teams average levels
# from my current knowledge of the game objectives such as epic monsters such as dragons and heralds give the team that gets them
# a big advantage so thats what we will include


# In[135]:


df_model2 = df[['blueWins', 'blueWardsPlaced', 'blueWardsDestroyed', 'blueDragons', 'blueHeralds', 
                'blueTotalGold', 'blueAvgLevel', 'redWardsPlaced','redWardsDestroyed', 'redTotalGold', 'redAvgLevel']]


# In[136]:


X2 = df_model2[['blueWardsPlaced', 'blueWardsDestroyed', 'blueDragons', 'blueHeralds', 
                'blueTotalGold', 'blueAvgLevel', 'redWardsPlaced','redWardsDestroyed', 'redTotalGold', 'redAvgLevel']]
Y2 = df_model2[['blueWins']]


# In[137]:


X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2,Y2,train_size=0.75, random_state=69)


# In[138]:


model_2 = LogisticRegression(solver='liblinear')


# 

# In[139]:


model_2.fit(X2_train,Y2_train)


# In[140]:


model_2.predict(X2_test)


# In[141]:


model_2.score(X2_test,Y2_test)


# In[142]:


# with more of information our model improved
# now ill include towers destroyed 
# if the model doesnt improve, I'll work with different train, test size groups since
# we have a lot of data. At the moment I'm doing 75% train, 25% test


# In[156]:


df_model3 = df[['blueWins', 'blueWardsPlaced', 'blueWardsDestroyed', 'blueDragons', 'blueHeralds', 'blueTowersDestroyed',
                'blueTotalGold', 'blueAvgLevel', 'redWardsPlaced','redWardsDestroyed', 'redTowersDestroyed',
                'redTotalGold', 'redAvgLevel']]

X3 = df_model3[['blueWardsPlaced', 'blueWardsDestroyed', 'blueDragons', 'blueHeralds', 'blueTowersDestroyed',
                'blueTotalGold', 'blueAvgLevel', 'redWardsPlaced','redWardsDestroyed', 'redTowersDestroyed',
                'redTotalGold', 'redAvgLevel']]

Y3 = df_model3[['blueWins']]


# In[174]:


X3_train, X3_test, Y3_train, Y3_test = train_test_split(X3,Y3,train_size=0.75, random_state=69)


# In[175]:


model_3=LogisticRegression(solver='liblinear')


# In[176]:


model_3.fit(X3_train,Y3_train)


# In[177]:


model_3.predict(X3_test)


# In[178]:


model_3.score(X3_test, Y3_test)


# In[ ]:


# so after adding 2 new variable of towers destroyed the model we obtained has the same quality as the previous one, without 
# this variables so for this data set turrets destroyed have no influence on the response 


# In[182]:


win = df['blueWins']
rtd = df['redTowersDestroyed']
btd = df['blueTowersDestroyed']

win.corr(rtd)


# In[183]:


win.corr(btd)


# In[184]:


# observing the correlation between bluewins and towers destroyed and the previous results for the models 2 and 3 
# we can conclude in fact that turrets destroyed have next to no impact on the outcome of the game

# keep the current model, model 2, and work around test, train sets size

# for the next experiments we will have train sizes of 80%, 85%, 90% and 95% and we will keep the model with the best score 


# In[186]:


X21_train, X21_test, Y21_train, Y21_test = train_test_split(X2,Y2,train_size=0.80, random_state=69)
model_21 = LogisticRegression(solver='liblinear')
model_21.fit(X21_train,Y21_train)
model_21.score(X21_test,Y21_test)


# In[187]:


X22_train, X22_test, Y22_train, Y22_test = train_test_split(X2,Y2,train_size=0.85, random_state=69)
model_22 = LogisticRegression(solver='liblinear')
model_22.fit(X22_train,Y22_train)
model_22.score(X22_test,Y22_test)


# In[189]:


X23_train, X23_test, Y23_train, Y23_test = train_test_split(X2,Y2,train_size=0.90, random_state=69)
model_23 = LogisticRegression(solver='liblinear')
model_23.fit(X23_train,Y23_train)
model_23.score(X23_test,Y23_test)


# In[190]:


X24_train, X24_test, Y24_train, Y24_test = train_test_split(X2,Y2,train_size=0.95, random_state=69)
model_24 = LogisticRegression(solver='liblinear')
model_24.fit(X24_train,Y24_train)
model_24.score(X24_test,Y24_test)


# In[197]:


# Our conclusion is we should use model_2 with a training set of 90% of the data_set
# Our data size is very big and probably is the case where too much information produces to much noise with a score of 75,8%
# this model doesn't have a convincing performance
# The variables were almost handpicked

# so now I'll check a second time the correlation matrix and check for variables that have at least 0.9 in absolute value with
# the variable blueWins

df.corr()['blueWins']


# In[198]:


# looking at this correlations I will try another model using the variables which have higher correlation whether
# positive or negative with blueWins 
# given the interval of correlations we will use the variables with 30% or higher and -30% or lower
# blueGoldDifference (this time we'll use this variable instead of blueTotalGold because both of these are highly 
# correlated and blueDGoldDifference has a higher correlation with blueWins), blueExperienceDifference (same thought as 
# blueTotalGold) and finally redDeaths


# In[201]:


df_model4 = df[['blueWins', 'blueGoldDiff', 'blueExperienceDiff', 'redDeaths']]

X4 = df_model4[['blueGoldDiff', 'blueExperienceDiff', 'redDeaths']]
Y4 = df_model4[['blueWins']]


# In[210]:


X4_train, X4_test, Y4_train, Y4_test = train_test_split(X4,Y4,train_size=0.99, random_state=69)
model_4 = LogisticRegression(solver='liblinear')
model_4.fit(X4_train,Y4_train)
model_4.score(X4_test,Y4_test)


# In[213]:


# some conclusion
# the goal for this project was to find what variables explain better the outcome for blueWins
# none of the models obtained were satisfying because of scores between 70 and 80% in the last model. 
# the dataset is too big (40 variables + thousands of entries) which means a lot of variance and a lot of noise 
# this being the reason I didnt start with the complete model
# a lot of variables show high correlation with each other which makes the task of picking variables somewhat harder 
# Finally if I had to choose a model to make predictions with I would use the last one.


# In[ ]: