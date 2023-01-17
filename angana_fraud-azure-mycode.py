# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#import os
#print(os.listdir("../input"))

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np
%matplotlib inline
#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import average_precision_score
from xgboost.sklearn import XGBClassifier
from xgboost import plot_importance, to_graphviz

import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

df = pd.read_csv('../input/Filter_data_file_1.csv')
print(df.head())
df.isnull().values.any()

# Any results you write to the current directory are saved as output.
print("\n The types of fraudulent transactions are {}".format(\
list(df.loc[df.isFraud == 1].Transaction_Type.drop_duplicates().values)))

dfFraudTransfer = df.loc[(df.isFraud == 1) & (df.Transaction_Type == 'TRANSFER')]
dfFraudCashout = df.loc[(df.isFraud == 1) & (df.Transaction_Type == 'CASH_OUT')]

print ('\n The number of fraudulent TRANSFERs = {}'.\
       format(len(dfFraudTransfer)))
       
print ('\n The number of fraudulent CASH_OUTs = {}'.\
       format(len(dfFraudCashout))) 
       
       
# Are there account labels common to fraudulent TRANSFERs and CASH_OUTs
print('\nIn fraudulent transactions, are there destinations for TRANSFERS \
that are also originators for CASH_OUTs? {}'.format(\
(dfFraudTransfer.Destination_Account.isin(dfFraudCashout.Source_Account)).any()))
dfNotFraud = df.loc[df.isFraud == 0]

print ('\n The number of non fraudulent transaction = {}'.\
       format(len(dfNotFraud)))

#print('\nFraudulent TRANSFERs whose destination accounts are originators of \
#genuine CASH_OUTs: \n\n{}'.format(dfFraudTransfer.loc[dfFraudTransfer.Destination_Account.isin(dfNotFraud.loc[dfNotFraud.Transaction_Type  == 'CASH_OUT'].Source_Account.drop_duplicates())]))

#-------------------------Data Visualization -----------------------------------------------
X = df.loc[(df.Transaction_Type == 'TRANSFER') | (df.Transaction_Type == 'CASH_OUT')]

# construct a random number generator
randomState = 5
np.random.seed(randomState) # for repeatable results

Y = X['isFraud']
del X['isFraud']

# Eliminate columns shown to be irrelevant for analysis
X = X.drop(['Source_Account', 'Destination_Account', 'isFlaggedFraud'], axis = 1)



# Binary-encoding of labelled data in 'type'
X.loc[X.Transaction_Type == 'TRANSFER', 'Transaction_Type'] = 0
X.loc[X.Transaction_Type == 'CASH_OUT', 'Transaction_Type'] = 1
X.Transaction_Type = X.Transaction_Type.astype(int)

Xfraud = X.loc[Y == 1]
XnonFraud = X.loc[Y == 0]

limit = len(X)
colours = plt.cm.tab10(np.linspace(0,1,9))
#plt.subplot(2,1,1)
#plt.plot(X.Hours_Per_Month, X.Transaction_Type==0, color ='blue')
ax = sns.stripplot(Y[:limit],X.Hours_Per_Month[:limit], hue = X.Transaction_Type[:limit], jitter= 0.4, marker = '.', size = 4, palette = colours)
ax.set_xlabel('')
ax.set_xticklabels(['genuine', 'fraudulent'], size = 16)
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles, ['Transfer', 'Cash out'], bbox_to_anchor=(1,1), loc=2, borderaxespad=0, fontsize = 16);
#for axis in ['top', 'bottom', 'left', 'right']:
            #ax.spines[axis].set_liewidth(2)
#handles, labels = ax.get_legend_handles_labels()
plt.title('Striped vs. homogenous fingerprints of genuine and fraudulent transactions over time', size = 18)
ax.set_ylabel('time [hour]', size = 16)
#plt.subplot(2,1,2)
#plt.plot(X.Hours_Per_Month, X.Transaction_Type, color ='blue')
#handles, labels = ax.get_legend_handles_labels()
plt.show();

ax = sns.stripplot(Y[:limit],X.Transaction_Amount[:limit], hue = X.Transaction_Type[:limit], jitter= 0.4, marker = '.', size = 4, palette = colours)
ax.set_xlabel('')
ax.set_xticklabels(['genuine', 'fraudulent'], size = 16)
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles, ['Transfer', 'Cash out'], bbox_to_anchor=(1,1), loc=2, borderaxespad=0, fontsize = 16);
plt.title('Same-signed fingerprints of genuine and fraudulent transactions over amount', size = 18)
ax.set_ylabel('Amount', size = 16)
plt.show();

ax = sns.stripplot(Y[:limit],X.errorBalanceDest[:limit], hue = X.Transaction_Type[:limit], jitter= 0.4, marker = '.', size = 4, palette = colours)
ax.set_xlabel('')
ax.set_xticklabels(['genuine', 'fraudulent'], size = 16)
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles, ['Transfer', 'Cash out'], bbox_to_anchor=(1,1), loc=2, borderaxespad=0, fontsize = 16);
plt.title('Opposite polarity fingerprints over the error in destination account balances', size = 18)
ax.set_ylabel('ErrorBalanceDest', size = 16)
plt.show();




# Any results you write to the current directory are saved as output.
