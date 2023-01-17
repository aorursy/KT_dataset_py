# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns



pd.set_option.max_columns = 300
train_df = pd.read_csv("../input/train_2016.csv", parse_dates=["transactiondate"])
train_df['transaction_month'] = train_df.transactiondate.dt.month
train_df.head()
f,ax = plt.subplots(figsize = (10,8))

plt.scatter(x=range(train_df.shape[0]),y=np.sort(train_df.logerror.values))

plt.show()
cnt_trans = train_df.transaction_month.value_counts()
cnt_trans.index
cnt_trans.values
plt.figure(figsize=(10,12))

sns.distplot(train_df.logerror.values,bins=50,kde=False)

sns.plt.show()
plt.figure(figsize=(10,12))

sns.barplot(x=cnt_trans.index,y=cnt_trans.values)

plt.xlabel('MONTH OF TRANSACTION',fontsize=12)

plt.ylabel('NUMBER OF OCCURENCES',fontsize=12)
prop_df =  pd.read_csv("../input/properties_2016.csv")
miss_df = prop_df.isnull().sum(axis=0).reset_index()
miss_df.columns = ['column','miss_count']
miss_df.head()
miss1_df = miss_df.sort_values(by='miss_count')
plt.figure(figsize=(30,40))

sns.set(font_scale=1.5)

sns.barplot(x='miss_count',y='column',data=miss1_df)

sns.plt.show()
train_df.corr()
pro_df = prop_df.copy()

tr_df = train_df.copy()
con_df = pd.merge(tr_df,pro_df,how='left',on='parcelid')
miss2_df = con_df.isnull().sum(axis=0).reset_index()
miss2_df.columns = ['col','miss_cnt']
miss2_df.head()
miss3_df = miss2_df.sort_values(by='miss_cnt')
miss3_df.head()
plt.figure(figsize=(40,50))

sns.set(font_scale=1.5)

sns.barplot(x='miss_cnt',y='col',data=miss3_df)



sns.plt.show()
corrmat = con_df.corr()
plt.figure(figsize=(30,40))

sns.heatmap(corrmat,vmax=.8, square=True)

sns.plt.show()
dtype_df = con_df.dtypes.reset_index()
dtype_df.columns = ['col','dtyp']
d = dtype_df.groupby(['dtyp'])['col'].sum().reset_index()
sns.regplot(x='numberofstories',y='logerror',data=con_df)

sns.plt.show()
np.mean(con_df['bedroomcnt'])

sns.boxplot(y='bedroomcnt',data=con_df)

sns.plt.show()
for col in con_df.columns:

    if con_df[col].dtypes=='object':

        print (col)
sns.regplot(x='bedroomcnt',y='bathroomcnt',data=con_df)

sns.plt.show()
sns.regplot(x='numberofstories',y='logerror',data=con_df)

sns.plt.show()
sns.regplot(x='lotsizesquarefeet',y='logerror',data=con_df)

sns.plt.show()
con1_df = con_df.copy()
ob_col=[]

for col in con1_df.columns:

    if con1_df[col].dtypes=='object':

        ob_col.append(col)
ob_col.append('transaction_month')

ob_col.append('transactiondate')


for col in con1_df:

    if col not in ob_col:

        con1_df[col].fillna(np.mean(con1_df[col]),inplace=True)
featto_use=[]

for col in con1_df.columns:

    if col not in ob_col:

        featto_use.append(col)
featto_use.remove('logerror')
con2_df = con1_df[featto_use]
X_train,y_train=con2_df,con1_df['logerror']
from sklearn.ensemble import GradientBoostingRegressor
clf = GradientBoostingRegressor()
clf = clf.fit(X_train,y_train)
feat_imp_num = list(clf.feature_importances_)
feat_dict = dict(zip(featto_use,feat_imp_num))
sorted_feat = reversed(sorted(feat_dict,key=feat_dict.__getitem__))
feat_name_rank=[]

feat_rank=[]

for k in sorted_feat:

    if feat_dict[k]!=0.0:

        feat_name_rank.append(k)

        feat_rank.append(feat_dict[k])
rankwise_dict = dict(zip(feat_name_rank,feat_rank))
feat_df = pd.DataFrame.from_dict(rankwise_dict,orient='index').reset_index()
feat_df.columns = ['f_name','rank']
plt.figure(figsize=(20,30))

sns.barplot(x='rank',y='f_name',data=feat_df)

sns.plt.show()
#please upvote, if you find it useful