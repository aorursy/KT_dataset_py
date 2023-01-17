import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('seaborn')



from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split as train_valid_split

from sklearn.multioutput import MultiOutputRegressor,RegressorChain



from xgboost import XGBRegressor

from sklearn.metrics import r2_score,mean_squared_error



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/bigquery-geotab-intersection-congestion/train.csv')

test = pd.read_csv('/kaggle/input/bigquery-geotab-intersection-congestion/test.csv')

train.info()
target_cols = ['TotalTimeStopped_p20','TotalTimeStopped_p50', 'TotalTimeStopped_p80',

               'DistanceToFirstStop_p20','DistanceToFirstStop_p50','DistanceToFirstStop_p80']

train[target_cols].head()
nrow=3

ncol=2

fig, axes = plt.subplots(nrow, ncol,figsize=(20,10))

count=0

for r in range(nrow):

    for c in range(ncol):

        if(count==len(target_cols)):

            break

        col = target_cols[count]

        

        axes[r,c].hist(np.log1p(train[col]),bins=100)

        axes[r,c].set_title('log1p( '+str(col)+' )',fontsize=15)

        count = count+1



plt.show()
feature_cols = ['IntersectionId', 'Latitude', 'Longitude', 'EntryStreetName',

                'ExitStreetName', 'EntryHeading', 'ExitHeading', 'Hour', 'Weekend',

                'Month','Path','City']

train[feature_cols].head()
train[feature_cols].describe(include='all').T
def category_mapping(df,map_dict):

    for col in map_dict.keys():

        df[col] = df[col].map(map_dict[col])

        df[col] = df[col].fillna(0).astype(np.int16)

    return df
cat_maps = {}

cat_cols = ['EntryStreetName','ExitStreetName','EntryHeading','ExitHeading','Path','City']

for col in cat_cols:

    values = list(train[col].unique())+list(test[col].unique())

    LE = LabelEncoder().fit(values)

    cat_maps[col] = dict(zip(LE.classes_, LE.transform(LE.classes_)))

    

train = category_mapping(train,cat_maps)
non_feature_cols = ['RowId','IntersectionId','TimeFromFirstStop_p20',

                    'TimeFromFirstStop_p40','TimeFromFirstStop_p50',

                    'TimeFromFirstStop_p60','TimeFromFirstStop_p80',

                    'DistanceToFirstStop_p40', 'DistanceToFirstStop_p60',

                    'TotalTimeStopped_p40','TotalTimeStopped_p60']

feature_cols = set(train.columns)-set(non_feature_cols+target_cols)



ys = train[target_cols]

Xs = train[feature_cols]

del train

X_train,X_valid, y_train, y_valid = train_valid_split(Xs, ys, test_size = .05, random_state=0)

X_train.shape,y_train.shape
%%time

multioutput_model = MultiOutputRegressor(XGBRegressor(random_state=0,n_jobs=-1,n_estimators=1000,

                                                      objective='reg:squarederror',max_depth=10,

                                                      tree_method='gpu_hist', predictor='gpu_predictor'))

multioutput_model.fit(X_train,y_train);
preds = multioutput_model.predict(X_train)

print('Training R2: ',r2_score(y_train,preds),

      ' MSE: ',mean_squared_error(y_train,preds))

preds = multioutput_model.predict(X_valid)

print('Validation R2: ',r2_score(y_valid,preds),

      ' MSE: ',mean_squared_error(y_valid,preds))
nrow=3

ncol=2

fig, axes = plt.subplots(nrow, ncol,

                         sharex=True,

                         sharey=True,

                         figsize=(15,10))

count=0

for r in range(nrow):

    for c in range(ncol):

        if(count==len(target_cols)):

            break

        col = target_cols[count]

        

        axes[r,c].hist(np.log1p(preds[:,count]),bins=100)

        axes[r,c].set_title('log1p( '+str(col)+' )',fontsize=15)

        count = count+1



plt.show()
test = test[feature_cols]

test = category_mapping(test,cat_maps)

test.head()
preds = multioutput_model.predict(test)
nrow=3

ncol=2

fig, axes = plt.subplots(nrow, ncol,

                         sharex=True,

                         sharey=True,

                         figsize=(15,10))

count=0

for r in range(nrow):

    for c in range(ncol):

        if(count==len(target_cols)):

            break

        col = target_cols[count]

        

        axes[r,c].hist(np.log1p(preds[:,count]),bins=100)

        axes[r,c].set_title('log1p( '+str(col)+' )',fontsize=15)

        count = count+1



plt.show()
TargetIds = pd.read_csv('/kaggle/input/bigquery-geotab-intersection-congestion/sample_submission.csv')['TargetId'].values



sub_df = pd.DataFrame()

sub_df['Target'] = list(preds)

sub_df = sub_df.explode('Target')



sub_df['TargetId'] = TargetIds

sub_df = sub_df[['TargetId','Target']]

sub_df.to_csv('the-sub-mission.csv',index=False)

sub_df.head(10)