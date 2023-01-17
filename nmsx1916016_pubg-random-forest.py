# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import datetime

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt

from matplotlib.ticker import MultipleLocator

from sklearn.ensemble import RandomForestClassifier as forest

from sklearn.ensemble import RandomForestRegressor

import matplotlib

# Metric used for the PUBG competition (Mean Absolute Error (MAE))

from sklearn.metrics import mean_absolute_error

def split_vals(a, n: int):

    # Function for splitting training and validation data

    return a[:n].copy(), a[n:].copy()

# Function to print the MAE (Mean Absolute Error) score

# This is the metric used by Kaggle in this competition



def print_score(m : RandomForestRegressor):

    res = ['mae train: ', mean_absolute_error(m.predict(X_train), y_train),

           'mae val: ', mean_absolute_error(m.predict(X_valid), y_valid)]

    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)

    print(res)

def get_sample(df,n):

    idxs = sorted(np.random.permutation(len(df))[:n])

    return df.iloc[idxs].copy()

def readcsv(path):

    df =pd.read_csv(path)

    return df



def rf_feat_importance(m, df):

    return pd.DataFrame({'cols': df.columns, 'imp': m.feature_importances_}

                        ).sort_values('imp', ascending=False)





def set_rf_samples(n):

    """ Changes Scikit learn's random forests to give each tree a random sample of

    n random rows.

    """

    forest._generate_sample_indices = (lambda rs, n_samples:

                                       forest.check_random_state(rs).randint(0, n_samples, n))





def reset_rf_samples():

    """ Undoes the changes produced by set_rf_samples.

    """

    forest._generate_sample_indices = (lambda rs, n_samples:forest.check_random_state(rs).randint(0, n_samples, n_samples))









def delete_cheater(train = 'none'):

    # Create feature totalDistance

    train['totalDistance'] = train['rideDistance'] + train['walkDistance'] + train['swimDistance']

    # Create feature killsWithoutMoving

    train['killsWithoutMoving'] = ((train['kills'] > 0) & (train['totalDistance'] == 0))

    train['headshot_rate'] = train['headshotKills'] / train['kills']

    train['headshot_rate'] = train['headshot_rate'].fillna(0)

    train.drop(train[train['killsWithoutMoving'] == True].index, inplace=True)

    # Players who got more than 10 roadKills

    train[train['roadKills'] > 10]

    # Drop roadKill 'cheaters'

    train.drop(train[train['roadKills'] > 10].index, inplace=True)

    # Remove outliers

    train.drop(train[train['kills'] > 30].index, inplace=True)



    # Players who made a minimum of 10 kills and have a headshot_rate of 100%

    train[(train['headshot_rate'] == 1) & (train['kills'] > 9)].head(10)

    train.drop(train[train['longestKill'] >= 1000].index, inplace=True)

    train.drop(train[train['walkDistance'] >= 10000].index, inplace=True)

    train.drop(train[train['rideDistance'] >= 20000].index, inplace=True)

    train.drop(train[train['swimDistance'] >= 2000].index, inplace=True)

    train.drop(train[train['weaponsAcquired'] >= 80].index, inplace=True)

    train.drop(train[train['heals'] >= 40].index, inplace=True)

    return train

def test_modify(test ='none',to_keep ='none'):

    # Add engineered features to the test set

    test['headshot_rate'] = test['headshotKills'] / test['kills']

    test['headshot_rate'] = test['headshot_rate'].fillna(0)

    test['totalDistance'] = test['rideDistance'] + test['walkDistance'] + test['swimDistance']

    test['playersJoined'] = test.groupby('matchId')['matchId'].transform('count')

    test['healsandboosts'] = test['heals'] + test['boosts']

    test['killsWithoutMoving'] = ((test['kills'] > 0) & (test['totalDistance'] == 0))



    # Turn groupId and match Id into categorical types

    test['groupId'] = test['groupId'].astype('category')

    test['matchId'] = test['matchId'].astype('category')



    # Get category coding for groupId and matchID

    test['groupId_cat'] = test['groupId'].cat.codes

    test['matchId_cat'] = test['matchId'].cat.codes





    return test[to_keep].copy()

if __name__ == '__main__':

    start_time = datetime.datetime.now()

    train_path = '../input/pubg-finish-placement-prediction/train_V2.csv'

    test_path = '../input/pubg-finish-placement-prediction/test_V2.csv'

    n = 10

    train = readcsv(train_path)

    test = readcsv(test_path)



    #删除只有一名玩家的样本

    #print(train[train['winPlacePerc'].isnull()])

    train.drop(2744604, inplace=True)



    #删除作弊者:

    train=delete_cheater(train)

    #print('There are {} different Match types in the dataset.'.format(train['matchType'].nunique()))

    # One hot encode matchType

    train = pd.get_dummies(train, columns=['matchType'])



    # Take a look at the encoding

    matchType_encoding = train.filter(regex='matchType')



    matchType_encoding.head()

    #将两个ID转化为category编码

    train['groupId'] = train['groupId'].astype('category')

    train['matchId'] = train['matchId'].astype('category')



    train['groupId_cat'] = train['groupId'].cat.codes

    train['matchId_cat'] = train['matchId'].cat.codes



    train.drop(columns=['groupId', 'matchId'], inplace=True)



    train[['groupId_cat', 'matchId_cat']].head()

    train.drop(columns = ['Id'], inplace=True)





    # Take sample for debugging and exploration

    sample = 500000

    df_sample = train.sample(sample)

    # Split sample into training data and target variable

    df = df_sample.drop(columns=['winPlacePerc'])  # all columns except target

    y = df_sample['winPlacePerc']  # Only target variable





   

   

    

    
    val_perc = 0.12  # % to use for validation set

    n_valid = int(val_perc * sample)

    n_trn = len(df) - n_valid

    # Split data

    raw_train, raw_valid = split_vals(df_sample, n_trn)

    X_train, X_valid = split_vals(df, n_trn)

    y_train, y_valid = split_vals(y, n_trn)



    # Check dimensions of samples

    print('Sample train shape: ', X_train.shape,

           'Sample target shape: ', y_train.shape,

           'Sample validation shape: ', X_valid.shape)

    # Train basic model

    m1 = RandomForestRegressor(n_estimators=70, min_samples_leaf=3, max_features='sqrt',

                               n_jobs=-1)

    m1.fit(X_train, y_train)

    print_score(m1)



    # What are the most predictive features according to our basic random forest model

    fi = rf_feat_importance(m1, df);

   # fi[:10]



    # Plot a feature importance graph for the 20 most important features



    plot1 = fi[:30].plot('cols', 'imp', figsize=(14, 6), legend=False, kind='barh')

   # x_major_locator = MultipleLocator(0.005)

    # 把x轴的刻度间隔设置为1，并存在变量里

    ax = plt.gca()

    # # ax为两条坐标轴的实例

    # ax.xaxis.set_major_locator(x_major_locator)

    ax.axvline(x=0.005, linestyle = '--',color='r', linewidth=1,label='x=0.005');

    #plt.show()



    # Use this code if you want to save the figure

    # fig = plot1.get_figure()

    # fig.savefig("Feature_importances(AllFeatures).png")



    # Keep only significant features

    to_keep = fi[fi.imp > 0.005].cols

    print('after m1:Significant features: ', len(to_keep))

    print(to_keep)



    # Make a DataFrame with only significant features

    df_keep = df[to_keep].copy()

    X_train, X_valid = split_vals(df_keep, n_trn)



   

    val_perc_full = 0.12  # % to use for validation set

    n_valid_full = int(val_perc_full * len(train))

    n_trn_full = len(train) - n_valid_full

    df_full = train.drop(columns=['winPlacePerc'])  # all columns except target

    y = train['winPlacePerc']  # target variable

    df_full = df_full[to_keep]  # Keep only relevant features

    X_train, X_valid = split_vals(df_full, n_trn_full)

    y_train, y_valid = split_vals(y, n_trn_full)



    # Check dimensions of data

    print('Sample train shape: ', X_train.shape,

           'Sample target shape: ', y_train.shape,

           'Sample validation shape: ', X_valid.shape)

    # Train final model

    # You should get better results by increasing n_estimators

    # and by playing around with the parameters

   
m3 = RandomForestRegressor(n_estimators=70, min_samples_leaf=3, max_features=0.5,

                       n_jobs=-1)

m3.fit(X_train, y_train)

#print_score(m3)

test_pred = test_modify(test, to_keep)

# Fill NaN with 0 (temporary)

test_pred.fillna(0, inplace=True)

print(test_pred.head())

# Make submission ready for Kaggle

# We use our final Random Forest model (m3) to get the predictions

predictions = np.clip(a=m3.predict(test_pred), a_min=0.0, a_max=1.0)

pred_df = pd.DataFrame({'Id': test['Id'], 'winPlacePerc': predictions})



# Create submission file

pred_df.to_csv("submission.csv", index=False)

print("运行完毕")

end_time = datetime.datetime.now()

print('耗费时间:',(end_time-start_time))