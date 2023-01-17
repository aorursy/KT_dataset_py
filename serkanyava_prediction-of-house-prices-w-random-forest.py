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
fitrain=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
fitest= pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
fitrain['train']  = 1
fitest['train']  = 0
df = pd.concat([fitrain, fitest], axis=0,sort=False)
df
df.isnull().sum()
df.columns
df['SalePrice'].describe()
df.info()
def show_missing_info(df):
    missing_info = df.isna().sum().reset_index(drop=False)
    missing_info.columns = ["column","rows"]
    missing_info["missing_pct"] = (missing_info["rows"]/fitrain.shape[0])*100
    missing_info = missing_info[missing_info["rows"]>0].sort_values(by="missing_pct",ascending=False)
    return missing_info
missing_df = show_missing_info(fitrain)
missing_df
df.drop(['Alley','PoolQC','Fence','MiscFeature','FireplaceQu'], axis=1,inplace=True)
obj_cols = df.select_dtypes(include=['object'])
num_cols =df.select_dtypes(exclude=['object'])
obj_cols
num_cols
obj_cols.isna().sum()
obj_cols =obj_cols.fillna(obj_cols.mode().iloc[0])
num_cols.isnull().sum()
num_cols =num_cols.fillna(num_cols.mode().iloc[0])
num_cols
obj_cols = pd.get_dummies(obj_cols, columns=obj_cols.columns) 
obj_cols.head()
df_final = pd.concat([obj_cols, num_cols], axis=1,sort=False)
df_final.head()
import matplotlib.pyplot as plt
cor=fitrain.corr()
import seaborn as sns
f,ax = plt.subplots(figsize=(30, 30))
sns.heatmap(cor, annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
cor["SalePrice"].sort_values(ascending=False)

x_1 = df_final.assign(Age = df_final['YrSold']-df_final['YearBuilt']) 
drop_list2 = ['YrSold','YearBuilt']
x_1 = x_1.drop(drop_list2,axis = 1 )       
x_1.head()
drop_list3=['OverallCond','EnclosedPorch', 'KitchenAbvGr']
x_1 = x_1.drop(drop_list3,axis = 1 )        
x_1.head()
x_1.SalePrice.plot(kind = 'hist',bins = 50,figsize = (30,15))
plt.show()
msval=fitrain['MSZoning'].value_counts()
msval.plot(kind = 'line',figsize = (30,10))
plt.show()

cmap = sns.cubehelix_palette(dark=.3, light=.9, as_cmap=True)
plt.figure(figsize=(20,20))
sns.scatterplot(data=fitrain, x="SalePrice", y="Neighborhood", palette=cmap)
plt.show()
sns.relplot(x="OverallQual", y="SalePrice",
            size_order=["T1", "T2"], palette=cmap,
            height=10, aspect=1.75, facet_kws=dict(sharex=False),
            kind="line", legend="full", data=df_final)
sns.relplot(x="Age", y="SalePrice",
            size_order=["T1", "T2"],
            height=10, aspect=1.75, facet_kws=dict(sharex=False),
            kind="line", legend="full", data=x_1)

df_train = x_1[x_1['train'] == 1]
df_train = df_train.drop(['train',],axis=1)


df_test = x_1[x_1['train'] == 0]
df_test = df_test.drop(['SalePrice'],axis=1)
df_test = df_test.drop(['train',],axis=1)
y_train= df_train['SalePrice']
x_train = df_train.drop(['SalePrice'],axis=1)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score,mean_squared_error,make_scorer
from sklearn.ensemble import RandomForestRegressor
est = make_pipeline(StandardScaler(),  RandomForestRegressor(max_samples=1460, n_estimators=5000, min_samples_leaf=1, random_state=14))
est.fit(x_train, y_train)
pred=est.predict(df_test)
test=fitest['Id']
test.shape
pred.shape
result = pd.DataFrame()
result['Id']= test
result['SalePrice'] = pred
result.to_csv('submission.csv',index=False)
