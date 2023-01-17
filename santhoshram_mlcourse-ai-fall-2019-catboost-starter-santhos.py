import warnings

import numpy as np

import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

from catboost import CatBoostClassifier
PATH_TO_DATA = Path('../input/flight-delays-fall-2018/')
train_df = pd.read_csv(PATH_TO_DATA / 'flight_delays_train.csv')
train_df.head()

test_df = pd.read_csv(PATH_TO_DATA / 'flight_delays_test.csv')
test_df.head()
train_split = train_df.shape[0]

full_df = pd.concat((train_df, test_df))
full_df['flight'] = full_df['Origin'] + '-->' + full_df['Dest']



# Hour and minute

full_df['hour'] = full_df['DepTime'] // 100

full_df.loc[full_df['hour'] == 24, 'hour'] = 0

full_df.loc[full_df['hour'] == 25, 'hour'] = 1

full_df['minute'] = full_df['DepTime'] % 100



# Daytime

full_df['daytime'] = pd.cut(full_df['hour'], bins=[0, 6, 12, 18, 23], include_lowest=True)





# Label Encoding

for col in ['Origin', 'Dest', 'UniqueCarrier', 'daytime']:

    full_df[col] = pd.factorize(full_df[col])[0]

#test_df['flight'] = test_df['Origin'] + '-->' + test_df['Dest']

#full_df['DayOfWeek'] = full_df['DayOfWeek'].apply(lambda x: x.replace('c-',''))

#full_df['Month'] = full_df['Month'].apply(lambda x: x.replace('c-',''))

#full_df['DayofMonth'] = full_df['DayofMonth'].apply(lambda x: x.replace('c-',''))
bin = [0, 1, 2, 3, 4]

full_df['DistanceBin'] = pd.cut(full_df['Distance'], bins=[0,100,300,800,1500,3000], labels = bin, include_lowest = True)

full_df['DistanceBin'].fillna(0, inplace = True)
full_df.head()
full_df['summer'] = (full_df['Month'].isin(['c-6', 'c-7', 'c-8'])).astype(np.int32)

full_df['autumn'] = (full_df['Month'].isin(['c-9', 'c-10', 'c-11'])).astype(np.int32)

full_df['winter'] = (full_df['Month'].isin(['c-12', 'c-1', 'c-2'])).astype(np.int32)

full_df['spring'] = (full_df['Month'].isin(['c-3', 'c-4', 'c-5'])).astype(np.int32)
full_df.info()

train_df, test_df = full_df.iloc[:train_split], full_df.iloc[train_split:]
categ_feat_idx = np.where(train_df.drop('dep_delayed_15min', axis=1).dtypes)[0]

categ_feat_idx
train_df.info()
X_train = train_df.drop('dep_delayed_15min', axis=1).values

y_train = train_df['dep_delayed_15min'].map({'Y': 1, 'N': 0}).values

X_test = test_df.drop('dep_delayed_15min', axis=1).values
for col in test_df.columns:

    print("Column:"+col+"Nulls::"+str(test_df[col].isnull().any()))

#X_test.isnull().any()
X_train_part, X_valid, y_train_part, y_valid = train_test_split(X_train, y_train, 

                                                                test_size=0.3, 

                                                                random_state=17)
ctb = CatBoostClassifier(random_seed=17, silent=True)
%%time

ctb.fit(X_train_part, y_train_part,

        cat_features=categ_feat_idx);
ctb_valid_pred = ctb.predict_proba(X_valid)[:, 1]
roc_auc_score(y_valid, ctb_valid_pred)
%%time

ctb.fit(X_train, y_train,

        cat_features=categ_feat_idx);
ctb_test_pred = ctb.predict_proba(X_test)[:, 1]
with warnings.catch_warnings():

    warnings.simplefilter("ignore")

    

    sample_sub = pd.read_csv(PATH_TO_DATA / 'sample_submission.csv', 

                             index_col='id')

    sample_sub['dep_delayed_15min'] = ctb_test_pred

    sample_sub.to_csv('ctb_pred.csv')
!head ctb_pred.csv