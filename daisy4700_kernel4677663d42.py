# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data1=pd.read_csv('/kaggle/input/ipl-data/IPL2013.csv')
data1.head(4)
data=pd.read_csv('/kaggle/input/ipl-data/IPL2013.csv')
data.head(5)
d=data.iloc[:5,:10]
d
data.columns
data['PLAYING ROLE'].unique()
data['COUNTRY'].unique()
data['TEAM'].unique()
data=data.drop(columns=['ECON'])
data.head()
data.columns
replace1={'Allrounder':1, 'Bowler':2, 'Batsman':3, 'W. Keeper':4}
replace1
data['PLAYING ROLE']=data['PLAYING ROLE'].replace(replace1)
data.head()
replace2={'KXIP':1, 'RCB':2, 'KKR':3, 'CSK':4, 'CSK+':4, 'RR':5, 'RCB+':5, 'MI+':6, 'DD+':6,'KKR+':3, 'DC':7, 'MI':6, 'DC+':7, 'RR+':5, 'KXIP+':1, 'KXI+':8, 'DD':6}
replace2
data['TEAM']=data['TEAM'].replace(replace2)
data.head()
replace3={'SA':1, 'BAN':2, 'IND':3, 'AUS':4, 'WI':5, 'SL':6, 'NZ':7, 'ENG':8, 'PAK':9, 'ZIM':10}
replace3
data['COUNTRY']=data['COUNTRY'].replace(replace3)
data.head()
data=data.drop(columns=['PLAYER NAME'])
data.head()
x=data.iloc[:,1:-2]
y=data.iloc[:,-1]

import matplotlib.pyplot as plt
cor=data.corr()
cor
#using Pearson cprelation
import seaborn as sns
plt.figure(figsize=(12,10))
sns.heatmap(cor,annot=True,cmap=plt.cm.Reds)
plt.show()
#selecting highly corelated features
cor_target=abs(cor["SOLD PRICE"])
cor_target
essential_features=cor_target[cor_target>0.3]
essential_features
cor_target.min()
cor_target.mean()
#selecting highly corelated features
essential_features=cor_target[cor_target>0.2]
essential_features
data.head(3)
from patsy import dmatrices
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
vif["features"] = x.columns
vif.round(1)
vif[vif['VIF Factor']>10]
#vif['VIF Factor'].mean()
x=x.drop(columns=['SR-BL','AUCTION YEAR','AVE-BL','WKTS','AGE','AVE','RUNS-C','SR-B','ODI-RUNS-S','HS','PLAYING ROLE','ODI-SR-B'])
x.head(3)
x.shape
from sklearn.model_selection import train_test_split 
x.train,x.test,y.train,y.test=train_test_split(x,y,test_size=0.2)
import statsmodels.api as sm
model=sm.OLS(y.train,x.train).fit()
prediction=model.predict(x.test)
model.summary()
