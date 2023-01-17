#load packages
import sys
import pandas as pd
import matplotlib
import numpy as np
import scipy as sp
import IPython
from IPython import display
import sklearn

import matplotlib.pyplot as plt
import seaborn as sns
import statistics as stat

%matplotlib inline
#Read train and test data
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )
train.dtypes
train.describe()
# define function to count unique values
def num_unique(series): 
    return len(series.unique())
    
train.apply(num_unique, axis = 0)
train.head()
train.median()
# declare lists of categorical and numerical variables
CAT_ATTRIBS = ['Survived','Sex','Embarked','Pclass','SibSp','Parch']
CAT_ATTRIBS_WO_SUR = ['Sex','Embarked','Pclass','SibSp','Parch'] # categorical w/o Survive
NUM_ATTRIBS = ['Age','Fare']

n_cat_attribs = len(CAT_ATTRIBS)
n_cat_attribs_wo_sur = len(CAT_ATTRIBS_WO_SUR)
n_num_attribs = len(NUM_ATTRIBS)
# TBD - rewrite the following code snippet so it is not that hard-coded (for loop)
f,ax = plt.subplots(2,3,figsize=(12,8))
sns.countplot('Sex',data=train,ax=ax[0,0])
sns.countplot('Pclass',data=train,ax=ax[0,1])
sns.countplot('Embarked',data=train,ax=ax[0,2])
sns.countplot('Survived',data=train,ax=ax[1,0])
sns.countplot('SibSp',data=train,ax=ax[1,1])
sns.countplot('Parch',data=train,ax=ax[1,2])
plt.suptitle('Categorical variables distributions', fontsize = 20)
plt.subplots_adjust(top=0.92)
# plt.tight_layout()
f,ax = plt.subplots(2,2,figsize=(14,8))

sns.distplot(train['Age'].dropna(),ax=ax[0,0])
sns.distplot(train['Fare'].dropna(),ax=ax[0,1])
sns.boxplot(train['Age'].dropna(),ax=ax[1,0])
sns.boxplot(train['Fare'].dropna(),ax=ax[1,1])

plt.suptitle('Continuous variables distributions', fontsize = 20)
plt.subplots_adjust(top=0.90)
# use categorical variables
data_hist_grid = train.loc[:, CAT_ATTRIBS] #.dropna(axis=0, how='any')

f,ax = plt.subplots(n_cat_attribs,n_cat_attribs,figsize=(20,16))

# iterate over categorical variables (two times)
for i in range(n_cat_attribs):
    for j in range(n_cat_attribs):
        
        # histogram on diagonal
        if i==j:
            sns.countplot(data_hist_grid.columns[i],
                          data=data_hist_grid,
                          ax=ax[i,j])
            
        # histogram with categories off diagonal
        else:
            sns.countplot(data_hist_grid.columns[i],
                          data=data_hist_grid,
                          ax=ax[i,j],
                          hue=data_hist_grid.columns[j])

# The next two lines do not work together with plt.tight_layout()
# plt.suptitle('Continuous variables distributions', fontsize = 20)
# plt.subplots_adjust(top=0.92) #left=0.2, wspace=0.8, )
        
plt.tight_layout()
data_for_pairplot = train[['Age','Fare','Survived']].dropna(axis=0, how='any')
sns.pairplot(data_for_pairplot, diag_kind="kde", kind="reg", hue='Survived',
                   vars=('Age','Fare'))
# plt.fig.suptitle('Continuous variables distributions', fontsize = 20)
# plt.fig.subplots_adjust(top=0.90) 
# cut the outliers in the Fare variable
train_wo_outliers = train[train['Fare'] < 180]
f,ax = plt.subplots(n_cat_attribs_wo_sur,n_num_attribs,figsize=(20,16))

# iterate over categorical variables
for i in range(n_cat_attribs_wo_sur):
    
    # iterate over numerical varibles
    for j in range(n_num_attribs):
        # print('i' + str(i) + ', j' + str(j))
        
        # create list of unique values
        unique_vals = train_wo_outliers[CAT_ATTRIBS_WO_SUR[i]].unique()
        
        # iterate over each unique value
        for unique_val in unique_vals:
        
            # subset the data
            data_subset = train_wo_outliers.loc[train_wo_outliers[CAT_ATTRIBS_WO_SUR[i]] == unique_val,NUM_ATTRIBS[j]]
            
            # kernel density estimation only works for certain number of observations
            if len(data_subset) > 10:
                sns.kdeplot(data_subset,ax=ax[i,j],label=unique_val)
                
            ax[i,j].set_title(NUM_ATTRIBS[j] + ' vs ' + CAT_ATTRIBS_WO_SUR[i])

plt.tight_layout()
f,ax = plt.subplots(n_cat_attribs_wo_sur,n_num_attribs,figsize=(20,16))

# iterate over categorical variables
for i in range(n_cat_attribs_wo_sur):
    
    # iterate over numerical varibles
    for j in range(n_num_attribs):
        
        sns.violinplot(x=CAT_ATTRIBS_WO_SUR[i], 
                       y=NUM_ATTRIBS[j],
                       data=train_wo_outliers,
                       palette="muted",
                       split=True,
                       ax=ax[i,j])

plt.tight_layout()
f,ax = plt.subplots(n_cat_attribs_wo_sur,n_num_attribs,figsize=(20,16))

# iterate over categorical variables
for i in range(n_cat_attribs_wo_sur):
    
    # iterate over numerical varibles
    for j in range(n_num_attribs):
        
        sns.violinplot(x=CAT_ATTRIBS_WO_SUR[i], 
                       y=NUM_ATTRIBS[j],
                       hue="Survived",
                       data=train_wo_outliers,
                       palette="muted",
                       split=True,
                       ax=ax[i,j],
                       inner="stick")

plt.tight_layout()