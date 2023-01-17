# General
import numpy as np
import pandas as pd
import os

# Visualization
import missingno as mn
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import gc

train = pd.read_csv('../input/application_train.csv', index_col = "SK_ID_CURR")
traindex = train.index
test = pd.read_csv('../input/application_test.csv', index_col = "SK_ID_CURR")
testdex = test.index

y = train.TARGET.copy()
train.drop("TARGET",axis=1,inplace= True)
df = pd.concat([train,test],axis=0)

del train, test
gc.collect();
mn.matrix(df.sample(100))
plt.show()
# Visualize Train/Test Difference
settypes= df.dtypes.reset_index()
def test_train_mis(test, train):
    missing_test = test.isnull().sum(axis=0).reset_index()
    missing_test.columns = ['column_name', 'test_missing_count']
    missing_test['test_missing_ratio'] = (missing_test['test_missing_count']/test.shape[0])*100
    missing_train = train.isnull().sum(axis=0).reset_index()
    missing_train.columns = ['column_name', 'train_missing_count']
    missing_train['train_missing_ratio'] = (missing_train['train_missing_count'] / train.shape[0])*100
    missing = pd.merge(missing_train, missing_test,
                       on='column_name', how='outer',indicator=True,)
    missing = pd.merge(missing,settypes, left_on='column_name', right_on='index',how='inner')
    missing = missing.loc[(missing['train_missing_ratio']>0) | (missing['test_missing_ratio']>0)]\
    .sort_values(by=["train_missing_ratio"], ascending=False)
    missing['Diff'] = missing.train_missing_ratio - missing.test_missing_ratio
    return missing

# Create
missing = test_train_mis(df.loc[testdex,:], df.loc[traindex,:])
missing["column_name"] = missing["column_name"].str.replace("_", " ").str.title()

f, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(15,15), sharey=True)
ax1 = sns.barplot(y ="column_name", x ="train_missing_ratio", data=missing,ax=ax1,
                  palette=plt.cm.magma(missing['train_missing_ratio']*.01))
ax1.set_xlim(75,0)
ax1.set_xlabel('Percent of Data Missing')
ax1.set_title('Train Set Percent Missing')
ax1.set_ylabel('Independent Variables')

ax2 = sns.barplot(y ="column_name", x ="Diff", data=missing,ax=ax2,
                  palette=plt.cm.magma(missing['Diff']*.01))
ax2.set_xlabel('Difference between Test/Train Missing count')
ax2.set_title('Difference between Test/Train Missing count')
ax2.set_xlim(-15,15)
ax2.axvline(x=0, c = "r")
ax2.set_ylabel('')

ax3 = sns.barplot(y ="column_name", x ="test_missing_ratio", data=missing,ax=ax3,
                  palette=plt.cm.magma(missing['test_missing_ratio']*.01))
ax3.set_xlabel('Percent of Data Missing')
ax3.set_title('Test Set Percent Missing')
ax3.set_ylabel('')

f.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
mn.heatmap(df)
plt.show()
mn.dendrogram(df)
plt.show()
