import pandas as pd

import seaborn as sns

import numpy as np
test_df=pd.read_csv('../input/test.csv')

train_df=pd.read_csv('../input/train.csv')

train_df.shape
train_df.head()
train_obj=train_df.select_dtypes(include=[object])

train_obj.columns
def value(obj_df):

    name_list=obj_df.columns

    dic_val={}

    for i in range(len(name_list)):

        print(name_list[i])

        print(obj_df.iloc[:,i].value_counts())

        print('\n')

       
value(train_obj)
from sklearn.cluster import KMeans

from sklearn import datasets

model=KMeans(n_clusters=10)

model.fit(train_obj)



train_df.isnull().sum()
corr = train_df.corr()

sns.heatmap(corr,xticklabels=corr.columns.values,yticklabels=corr.columns.values)
corr
train_num=train_df.select_dtypes(include=[np.int64,np.float64])

train_num.drop(['Id'],axis=1,inplace=True)

train_num.isnull().sum()
train_obj=train_df.select_dtypes(include=[object])

train_obj.isnull().sum()

#train_obj.shape
train_obj.drop(['PoolQC','Fence','MiscFeature','Alley','Street','Utilities','LandSlope','Condition2','RoofMat1'],axis=1,inplace=True)

train_obj.isnull().sum()
train_obj.columns
train_obj=pd.get_dummies(train_obj)

train_obj.head()