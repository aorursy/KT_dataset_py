# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/santander-df-1/df_1',delimiter=',')
df.drop(columns=['Unnamed: 0','fecha_alta','canal_entrada','ult_fec_cli_1t'],axis=1, inplace=True)
def lower_than(group,num):
    if len(group['fecha_dato']) < num:
        return None
    else :
        return group
df=df.groupby('ncodpers',as_index=False).apply(lower_than,num=17)
df.head()
df.index=df.index.droplevel(level=0)
# let's check
for i in df['ncodpers'].value_counts():
    if i != 17 :
        print('fuck')
target_cols = df.iloc[:1,].filter(regex="ind_+.*ult.*").columns.values
feature_cols = [_ for _ in df.iloc[:1,].columns.values if _ not in set(target_cols)]
feature_df=df.loc[:,feature_cols]
target_df=df.loc[:,target_cols]
obj_col=df.select_dtypes(include='object').columns
obj_col
feature_df=pd.get_dummies(feature_df, prefix=obj_col,drop_first=True)
len(feature_df.columns)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_e_s=scaler.fit_transform(feature_df)
#scaler.transform(df)
#feature_df = df_e_s
# target_df와 concat하기위해 df로 바꿈
feature_df =pd.DataFrame(df_e_s,columns=feature_df.columns)
print(type(target_df))
print(target_df.shape)
display(type(feature_df))
display(feature_df.shape)
feature_df.to_csv('santander_feature_df1.csv')
target_df.to_csv('santander_target_df1.csv')



