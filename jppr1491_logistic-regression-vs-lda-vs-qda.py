#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix

get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


full_data = pd.read_csv(r'\Users\Pedro\Documents\high_diamond_ranked_10min.csv')
# lets exclude gameId

data = full_data.drop(['gameId'],axis=1)


# In[6]:


# Win-loss count 0 means loss, 1 means win
blueWins = data['blueWins']
ax = sns.countplot(x = blueWins, data = data)


# In[10]:


X = data.drop(['blueWins'],axis=1)
Y = blueWins.values.reshape(-1,1)

# if memory doesnt fail, last time we fitted a logistic regression model to this data, the best result came from a train 
# size of 0.9 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.9, random_state=69)


# In[11]:


# logistic regression
from sklearn.linear_model import LogisticRegression


# In[13]:


log_reg = LogisticRegression()
log_reg.fit(X_train, Y_train.ravel())


# In[14]:


Y_prob = log_reg.predict_proba(X_test)[:,1]
Y_pred = np.where(Y_prob > 0.5, 1, 0) 


# In[15]:


log_confusion_matrix = confusion_matrix(Y_test, Y_pred)
log_confusion_matrix
# 390 + 358 correct predictions
# 122 + 119 wrong predictions


# In[38]:


false_positive_rate_log, true_positive_rate_log, thresholds_log = roc_curve(Y_test, Y_prob)
roc_auc_log = auc(false_positive_rate_log, true_positive_rate_log)
roc_auc_log


# In[19]:


# lets take a look at the graph

plt.figure(figsize=(9,9))
plt.title('ROC')
plt.plot(false_positive_rate_log, true_positive_rate_log, color='red', label = 'AUC = %2f' % roc_auc_log)
plt.legend(loc='lower right')
plt.plot([0,1], [0,1], linestyle = '--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.plot(roc_auc_log)


# In[20]:


# linear discriminant analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis()
lda.fit(X_train, Y_train.ravel())


# In[21]:


Y_prob_lda = lda.predict_proba(X_test)[:,1]
Y_pred_lda = np.where(Y_prob_lda > 0.5, 1, 0) 


# In[23]:


lda_confusion_matrix = confusion_matrix(Y_test, Y_pred_lda)
lda_confusion_matrix


# In[24]:


false_positive_rate_lda, true_positive_rate_lda, thresholds_lda = roc_curve(Y_test, Y_prob_lda)
roc_auc_lda = auc(false_positive_rate_lda, true_positive_rate_lda)
roc_auc_lda


# In[25]:


# the auc for lda is slightly lower than the auc for the logistic regression 
# this means logistic regression performs slightly better, but a very very close to zero
# roc_auc_log - roc_auc_lda


# In[26]:


plt.figure(figsize=(9,9))
plt.title('ROC')
plt.plot(false_positive_rate_lda, true_positive_rate_lda, color='red', label = 'AUC = %2f' % roc_auc_lda)
plt.legend(loc='lower right')
plt.plot([0,1], [0,1], linestyle = '--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.plot(roc_auc_lda)


# In[27]:


# finally lets see how quadratic discriminant analysis performs
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, Y_train.ravel())


# In[28]:


Y_prob_qda = qda.predict_proba(X_test)[:,1]
Y_pred_qda = np.where(Y_prob_qda > 0.5, 1, 0) 


# In[29]:


qda_confusion_matrix = confusion_matrix(Y_test, Y_pred_qda)
qda_confusion_matrix


# In[30]:


false_positive_rate_qda, true_positive_rate_qda, thresholds_qda = roc_curve(Y_test, Y_prob_qda)
roc_auc_qda = auc(false_positive_rate_qda, true_positive_rate_qda)
roc_auc_qda


# In[31]:


plt.figure(figsize=(9,9))
plt.title('ROC')
plt.plot(false_positive_rate_qda, true_positive_rate_qda, color='red', label = 'AUC = %2f' % roc_auc_qda)
plt.legend(loc='lower right')
plt.plot([0,1], [0,1], linestyle = '--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.plot(roc_auc_qda)


# In[45]:


# Conclusions
# The best perfoming method was logistic regression, followed by lda, followed by qda
print('Logistic regression', '\t' , 'Lda', '\t', '                Qda')
print(roc_auc_log, '\t', roc_auc_lda, '\t', roc_auc_qda)


# In[ ]: