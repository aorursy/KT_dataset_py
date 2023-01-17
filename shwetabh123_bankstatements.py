# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
%matplotlib inline 
data =pd.read_csv("../input/Step2_raw_Transactions.csv - Step2_raw_public - Transactions.csv (2).csv")
data.head()
data.describe()
data.info()
data.columns
sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
data.drop('ref_no',axis=1,inplace=True)
data.head(5)
data['transaction amount  - Debit'].fillna((data['transaction amount  - Debit'].mean()), inplace=True)
sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
data['transaction amount   - Credit'].fillna((data['transaction amount   - Credit'].mean()), inplace=True)
sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
data['value_date'] = data['value_date'].astype('datetime64[ns]')
data.head(5)
data.drop('Date',axis=1,inplace=True)
data.drop('value_date',axis=1,inplace=True)


data.head(5)

data1=pd.get_dummies(data)


data1.head(5)
sns.heatmap(data1.isnull(),yticklabels=False,cbar=False,cmap='viridis')
data1.columns
data1.columns
X =data1[['Transaction ID', 'transaction amount  - Debit',
       'transaction amount   - Credit',
       'narration_ACH DEBIT:HDFC BANK LIMITED,0000048995809',
       'narration_ACH DEBIT:HDFC BANK LIMITED,0000049597028',
       'narration_ACH DEBIT:HDFC BANK LIMITED,0001149597028',
       'narration_ACH DEBIT:SHRIRAM CITY UNION F,PMPRCCM1708290018',
       'narration_ACH DEBIT:TP ACH BAJAJ FINANAC,71110941',
       'narration_ACH DEBIT:TP ACH BAJAJ FINANAC,71212322',
       'narration_ACH DEBIT:TP ACH BAJAJ FINANAC,71331999',
       'balance_76116.99', 'balance_77524.99', 'balance_7932.99',
       'balance_8029.99', 'balance_8433.99', 'balance_88640.99',
       'balance_88909.99', 'balance_931.99', 'balance_932.99',
       'balance_94536.99']]
# Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('mode of transaction')
plt.ylabel('Euclidean distances')
plt.show()
X1 =data1[['Transaction ID', 'transaction amount  - Debit',
       'transaction amount   - Credit',
       'narration_ACH DEBIT:HDFC BANK LIMITED,0000048995809',
       'narration_ACH DEBIT:HDFC BANK LIMITED,0000049597028',
       'narration_ACH DEBIT:HDFC BANK LIMITED,0001149597028',
       'narration_ACH DEBIT:SHRIRAM CITY UNION F,PMPRCCM1708290018']]
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X1, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('category of merchant')
plt.ylabel('Euclidean distances')
plt.show()




