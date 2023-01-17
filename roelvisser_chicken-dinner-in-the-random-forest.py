import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read training data
df_train_raw = pd.read_csv('../input/train.csv')
df_train_raw.sample(5)
df_train_raw.describe()
print(('{m:d} unique matches').format(m=len(df_train_raw['matchId'].unique())))

print(('{tk:d} data entries in {m:d} different matches with exactly 3 team kills').format(
    tk=len(df_train_raw[df_train_raw['teamKills']==3]),
    m=len(df_train_raw[df_train_raw['teamKills']==3]['matchId'].unique())))

print(('{tk:d} data entries in {m:d} different matches with more than 3 team kills').format(
    tk=len(df_train_raw[df_train_raw['teamKills']>3]),
    m=len(df_train_raw[df_train_raw['teamKills']>3]['matchId'].unique())))
    
print(('{g:d} data entries in {m:d} different matches with exactly 1 group').format(
    g=len(df_train_raw[df_train_raw['numGroups']==1]),
    m=len(df_train_raw[df_train_raw['numGroups']==1]['matchId'].unique())))

n_group = 10
print(('{g:d} data entries in {m:d} different matches with less than {n:d} groups').format(
    g=len(df_train_raw[df_train_raw['numGroups']<n_group]),
    m=len(df_train_raw[df_train_raw['numGroups']<n_group]['matchId'].unique()),
    n=n_group))
df_train = df_train_raw#.iloc[0:100000]

del df_train_raw
f, ax = plt.subplots(figsize=(20, 15))
sns.heatmap(df_train.corr(), annot=True, fmt='.2f', linewidths=.5, ax=ax)
# Reference: https://www.kaggle.com/anycode/simple-nn-baseline/code
def FeatureEngineering(df):
    df_size = df.groupby(['matchId','groupId']).size().reset_index(name='group_size')
        
    df_mean = df.groupby(['matchId','groupId']).mean().reset_index()
    
    df_sum = df.groupby(['matchId','groupId']).sum().reset_index()
    
    df_max = df.groupby(['matchId','groupId']).max().reset_index()
    
    df_min = df.groupby(['matchId','groupId']).min().reset_index()
    
    df_match_mean = df.groupby(['matchId']).mean().reset_index()
    
    df = pd.merge(df, df_size, how='left', on=['matchId', 'groupId'])
    del df_size
    df = pd.merge(df, df_mean, suffixes=["", "_mean"], how='left', on=['matchId', 'groupId'])
    del df_mean
    df = pd.merge(df, df_sum, suffixes=["", "_sum"], how='left', on=['matchId', 'groupId'])
    del df_sum
    df = pd.merge(df, df_max, suffixes=["", "_max"], how='left', on=['matchId', 'groupId'])
    del df_max
    df = pd.merge(df, df_min, suffixes=["", "_min"], how='left', on=['matchId', 'groupId'])
    del df_min
    df = pd.merge(df, df_match_mean, suffixes=["", "_match_mean"], how='left', on=['matchId'])
    del df_match_mean
        
    columns = list(df.columns)
    columns.remove("Id")
    columns.remove("matchId")
    columns.remove("groupId")
    columns.remove("Id_mean")
    columns.remove("Id_sum")
    columns.remove("Id_max")
    columns.remove("Id_min")
    columns.remove("Id_match_mean")

    df = df[columns]
    return df

def MAE(y_estimate, y_true):
    return sum(abs(y_estimate-y_true))/len(y_estimate)
from sklearn.model_selection import train_test_split

matchId = df_train['matchId'].unique()
matchIdTrain = np.random.choice(matchId, int(0.80*len(matchId)))

df_train2 = df_train[df_train['matchId'].isin(matchIdTrain)]
df_test = df_train[~df_train['matchId'].isin(matchIdTrain)]

y_train = df_train2['winPlacePerc']
X_train = df_train2.drop(columns=['winPlacePerc'])
y_test = df_test['winPlacePerc']
X_test = df_test.drop(columns=['winPlacePerc'])

X_train = FeatureEngineering(X_train)
X_test = FeatureEngineering(X_test)

# This commented out section is the train/test split without keeping the matches intact
#X = df_train.drop(columns=['winPlacePerc'])
#y = df_train['winPlacePerc']
#X = FeatureEngineering(X)
#X_train, X_test, y_train, y_test = train_test_split(X, y)

print(('Training set size: {train:d}, test set size: {test:d}').format(train=len(X_train), test=len(X_test)))
print(X_train.describe())

del df_train2, df_test, df_train#, X, y
def RandomForestModel():
    print('\nCreating and training random forest regressor')
    from sklearn.ensemble import RandomForestRegressor
    rfr = RandomForestRegressor(n_jobs=4, n_estimators=10)
    rfr.fit(X_train, y_train)

    y_rfr = rfr.predict(X_test)
    score_rfr = MAE(y_rfr, y_test)
    print(('Random Forest training testset score: {s:.3f}').format(s=score_rfr))
    
    # Read the test set data and make predictions
    X_submit = pd.read_csv('../input/test.csv')
    df_submit = X_submit[['Id', 'matchId', 'groupId']]
    X_submit = FeatureEngineering(X_submit)
    y_submit = rfr.predict(X_submit)
    df_submit['prediction'] = y_submit
    
    # Return a dataframe with ID's and the prediction
    return df_submit    
df_submit = RandomForestModel()

df_submit.head()
print('Correcting predictions')
        
df_submit['prediction_mod'] = -1.0
matchId = df_submit['matchId'].unique()

for match in matchId:
    df_match = df_submit[df_submit['matchId']==match]

    df_max = df_match.groupby(['groupId']).max()
    pred_sort = sorted(df_max['prediction'])

    for i in df_max.index:
        groupPlace = pred_sort.index(df_max.loc[i]['prediction'])
        if len(pred_sort) > 1:
            df_max.at[i,'prediction_mod'] = groupPlace/(len(pred_sort)-1)
        else:
            df_max.at[i,'prediction_mod'] = 1.0

    for i in df_match.index:
        df_submit.at[i, 'prediction_mod'] = df_max['prediction_mod'].loc[df_match['groupId'].loc[i]]

y_submit_cor = df_submit['prediction_mod']
print('Submission scores corrected')

df_submit.head()
df_test = pd.read_csv('../input/sample_submission.csv')
df_test['winPlacePerc'] = df_submit['prediction_mod'].copy()

df_test.to_csv('submission_rfr.csv', index=False) 
print('Random Forest submission file made\n')