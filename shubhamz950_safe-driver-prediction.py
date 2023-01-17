# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import plotly.graph_objs as go
import plotly.offline as py
py.init_notebook_mode(connected=True)


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_sub = pd.read_csv('../input/sample_submission.csv')

df_sub.head()
df_test.head()
df_train.target.value_counts()
df_test.columns
df_copy = df_train
df_copy = df_copy.replace(-1, np.NaN)
df_test = df_test.replace(-1, np.NaN)
df_null = df_copy.isnull().sum().sort_values(ascending=False)
df_null_percent = (df_copy.isnull().sum() / df_copy.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([df_null,df_null_percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
df_copy = df_copy.drop((missing_data[missing_data['Total'] > 5810]).index,1)
df_test = df_test.drop((missing_data[missing_data['Total'] > 5810]).index,1)
df_copy.columns
Counter(df_copy.dtypes.values)
df_float = df_copy.select_dtypes(include=['float64'])
df_int = df_copy.select_dtypes(include=['int64'])
df_int.columns
colormap = plt.cm.magma
plt.figure(figsize=(16,24))
sns.heatmap(df_float.corr(), linewidths=0.1, vmax=1.0, square= True, cmap=colormap, linecolor='white', annot = True)
bin_col = [col for col in df_copy.columns if '_bin' in col]
zeroes = []
ones = []
for col in bin_col:
    zeroes.append((df_copy[col] == 0).sum())
    ones.append((df_copy[col] == 1).sum())
trace1 = go.Bar(
    x=bin_col,
    y=zeroes ,
    name='Zero count'
)
trace2 = go.Bar(
    x=bin_col,
    y=ones,
    name='One count'
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='stack',
    title='Count of 1 and 0 in binary variables'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='stacked-bar')
#plt.show()
y = df_copy['target']
X = df_copy.drop('target', axis=1, inplace=True)
y.head()
from sklearn.preprocessing import Imputer
imr = Imputer(missing_values='NaN', strategy='median', axis=0)
#Other options for the strategy parameter are median or most_frequent
imr.fit(df_copy)
imurated_data = imr.transform(df_copy)
test_data = imr.transform(df_test)
imurated_data
imurated_data = pd.DataFrame(imurated_data)
test_data = pd.DataFrame(test_data)
imurated_data.columns = ['id', 'ps_ind_01', 'ps_ind_02_cat', 'ps_ind_03','ps_ind_04_cat', 'ps_ind_05_cat', 'ps_ind_06_bin', 'ps_ind_07_bin','ps_ind_08_bin', 'ps_ind_09_bin', 'ps_ind_10_bin', 'ps_ind_11_bin','ps_ind_12_bin', 'ps_ind_13_bin', 'ps_ind_14', 'ps_ind_15','ps_ind_16_bin', 'ps_ind_17_bin', 'ps_ind_18_bin', 'ps_reg_01','ps_reg_02', 'ps_car_01_cat', 'ps_car_02_cat', 'ps_car_04_cat','ps_car_06_cat', 'ps_car_08_cat', 'ps_car_09_cat', 'ps_car_10_cat','ps_car_11_cat', 'ps_car_11', 'ps_car_12', 'ps_car_13', 'ps_car_15','ps_calc_01', 'ps_calc_02', 'ps_calc_03', 'ps_calc_04', 'ps_calc_05','ps_calc_06', 'ps_calc_07', 'ps_calc_08', 'ps_calc_09', 'ps_calc_10','ps_calc_11', 'ps_calc_12', 'ps_calc_13', 'ps_calc_14','ps_calc_15_bin', 'ps_calc_16_bin', 'ps_calc_17_bin', 'ps_calc_18_bin','ps_calc_19_bin', 'ps_calc_20_bin']
test_data.columns = ['id', 'ps_ind_01', 'ps_ind_02_cat', 'ps_ind_03','ps_ind_04_cat', 'ps_ind_05_cat', 'ps_ind_06_bin', 'ps_ind_07_bin','ps_ind_08_bin', 'ps_ind_09_bin', 'ps_ind_10_bin', 'ps_ind_11_bin','ps_ind_12_bin', 'ps_ind_13_bin', 'ps_ind_14', 'ps_ind_15','ps_ind_16_bin', 'ps_ind_17_bin', 'ps_ind_18_bin', 'ps_reg_01','ps_reg_02', 'ps_car_01_cat', 'ps_car_02_cat', 'ps_car_04_cat','ps_car_06_cat', 'ps_car_08_cat', 'ps_car_09_cat', 'ps_car_10_cat','ps_car_11_cat', 'ps_car_11', 'ps_car_12', 'ps_car_13', 'ps_car_15','ps_calc_01', 'ps_calc_02', 'ps_calc_03', 'ps_calc_04', 'ps_calc_05','ps_calc_06', 'ps_calc_07', 'ps_calc_08', 'ps_calc_09', 'ps_calc_10','ps_calc_11', 'ps_calc_12', 'ps_calc_13', 'ps_calc_14','ps_calc_15_bin', 'ps_calc_16_bin', 'ps_calc_17_bin', 'ps_calc_18_bin','ps_calc_19_bin', 'ps_calc_20_bin']
imurated_data.drop(['ps_ind_10_bin','ps_ind_11_bin','ps_ind_12_bin','ps_ind_13_bin'], axis=1, inplace=True)
imurated_data.head()
test_data.drop(['ps_ind_10_bin','ps_ind_11_bin','ps_ind_12_bin','ps_ind_13_bin'], axis=1, inplace=True)
from sklearn.model_selection import train_test_split

# y = imurated_data['target']
# X = imurated_data.drop('target', axis=1, inplace=True)

df_copy.head()
X_train,X_test,y_train,y_test = train_test_split(imurated_data, y, test_size=0.3, random_state=1)


from  sklearn.linear_model  import LogisticRegression
# from sklearn.decomposition import PCA
# pca = PCA(n_components = 20)
lr = LogisticRegression()
# X_train_pca = pca.fit_transform(X_train)
# X_test_pca = pca.transform(X_test)
# test_data_pca = pca.transform(test_data)
lr.fit(X_train,y_train)
lr.score(X_test,y_test)
# corrmat = df_train.corr()
# f, ax = plt.subplots(figsize=(12, 9))
# sns.heatmap(corrmat, vmax=.8, square=True);
output = lr.predict(test_data)
te = df_test = pd.read_csv('../input/test.csv')
ids = te['id']

my_submission = pd.DataFrame({'ID': ids, 'target': output})
my_submission.to_csv('submission.csv', index=False)
