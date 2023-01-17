!pip install git+https://github.com/fastai/fastai@2e1ccb58121dc648751e2109fc0fbf6925aa8887
!apt update && apt install -y libsm6 libxext6
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from fastai.imports import *

from fastai.structured import *



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from pandas_summary import DataFrameSummary

from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier

from sklearn import metrics
!ls ../input/nfl-big-data-bowl-2020/
path = Path('../input/nfl-big-data-bowl-2020/')

path_train_csv = path/'train.csv'
df = pd.read_csv(path_train_csv,low_memory=False)

df.head()
df.columns
df['TimeSnap'].head()
df = pd.read_csv(path_train_csv,low_memory=False,parse_dates=['TimeSnap'])

df['TimeSnap'].head()
def display_all(df):

    with pd.option_context('display.max_rows',1000,'display.max_columns',1000):

        display(df)
display_all(df.tail().T)

print(df.shape)
df = pd.read_csv(path_train_csv,low_memory=False,parse_dates=['TimeSnap','TimeHandoff'])
display_all(df.tail().T)

print(df.shape)
display_all(df.describe(include='all').T)
add_datepart(df,'TimeSnap')

add_datepart(df,'TimeHandoff')
display_all(df.tail().T)

print(df.shape)
display_all(df.isnull().sum().sort_index()/len(df))
df['Yards'].describe()
train_cats(df)
display_all(df.tail().T)

print(df.shape)
from sklearn.model_selection import GroupShuffleSplit

train_idxs, valid_idxs = next(GroupShuffleSplit(test_size=.2, n_splits=2, random_state = 42).split(df, groups=df['GameId']))

df_train = df.iloc[train_idxs]

df_valid = df.iloc[valid_idxs]
df_train_final,y,nas = proc_df(df_train,'Yards')

df_valid_final,_,_ = proc_df(df_valid,na_dict=nas)

df_train_final.shape,y.shape,df_valid_final.shape
len(df_train['GameId'].unique()),len(df_valid['GameId'].unique())
df_train['GameId'].equals(df_valid['GameId'])
set_rf_samples(80000)
model_first = RandomForestRegressor(n_estimators=40, min_samples_leaf=3,n_jobs=-1)

%time model_first.fit(df_train_final,y)
df_valid_final.shape
y_valid = df_valid_final['Yards']

df_valid_final.drop(['Yards'],axis=1,inplace=True)

df_valid_final.shape,y_valid.shape
preds = model_first.predict(df_train_final)

preds_valid = model_first.predict(df_valid_final)
y_ans = np.zeros((len(df_train_final),199))



for i,p in enumerate(y):

    for j in range(199):

        if j-99>=p:

            y_ans[i][j]=1.0
train_cdf = np.histogram(preds, bins=199,

                 range=(-99,100), density=True)[0].cumsum()
print("Train score:",np.sum(np.power(train_cdf-y_ans,2))/(199*(len(df_train_final))))
valid_cdf = np.histogram(preds_valid, bins=199,

                 range=(-99,100), density=True)[0].cumsum()
y_ans_valid = np.zeros((len(df_valid_final),199))



for i,p in enumerate(y_valid):

    for j in range(199):

        if j-99>=p:

            y_ans_valid[i][j]=1.0
print("Valid score:",np.sum(np.power(valid_cdf-y_ans_valid,2))/(199*(len(df_valid_final))))
fi = rf_feat_importance(model_first,df_train_final)

fi
fi[fi['imp']>0.01]
fi[:15].plot('cols','imp','barh',figsize=(12,7))
df_train_final['GameClock'].describe()
plt.scatter(df_train_final['GameClock'],y)
plt.scatter(df_train_final['Distance'],y)
df['GameClock'].min(),df['GameClock'].max()
df['GameClock'].unique()