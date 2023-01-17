# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# numpy and pandas for data manipulation
import numpy as np
import pandas as pd 

# sklearn preprocessing for dealing with categorical variables
from sklearn.preprocessing import LabelEncoder

# File system manangement
import os

# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')

# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns

# print out files
print(os.listdir("../input/"))

# training data

traindata = pd.read_csv('../input/train_V2.csv')
traindata.head()

# testing data

testdata = pd.read_csv('../input/test_V2.csv')
testdata.head()
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns
missing_values = missing_values_table(traindata)
missing_values

traindata[traindata['winPlacePerc'].isnull()]
traindata.drop(2744604, inplace= True)
traindata = pd.get_dummies(traindata, columns = ['matchType'])
matchType_encoding = traindata.filter(regex='matchType')
matchType_encoding.head()
# Turn groupId and match Id into categorical types
traindata['groupId'] = traindata['groupId'].astype('category')
traindata['matchId'] = traindata['matchId'].astype('category')

# Get category coding for groupId and matchID
traindata['groupId_cat'] = traindata['groupId'].cat.codes
traindata['matchId_cat'] = traindata['matchId'].cat.codes

# Get rid of old columns
traindata.drop(columns=['groupId', 'matchId'], inplace=True)

# Lets take a look at our newly created features
traindata[['groupId_cat', 'matchId_cat']].head()
traindata.drop(columns = ['Id'], inplace=True)
traindata['totalDistance'] = traindata['walkDistance'] + traindata['rideDistance'] + traindata['swimDistance']
traindata['headshotRate'] = traindata['headshotKills']/traindata['kills']
traindata['headshotRate'] = traindata['headshotRate'].fillna(0)
traindata['playersJoined'] = traindata.groupby('matchId_cat')['matchId_cat'].transform('count')

# Create normalized features
traindata['killsNorm'] = traindata['kills']*((100-traindata['playersJoined'])/100 + 1)
traindata['damageDealtNorm'] = traindata['damageDealt']*((100-traindata['playersJoined'])/100 + 1)
traindata['maxPlaceNorm'] = traindata['maxPlace']*((100-traindata['playersJoined'])/100 + 1)
traindata['matchDurationNorm'] = traindata['matchDuration']*((100-traindata['playersJoined'])/100 + 1)
traindata['healsandboosts'] = traindata['heals'] + traindata['boosts']
traindata['killsWithoutMoving'] = ((traindata['kills'] > 0) & (traindata['totalDistance'] == 0))


# Check players who kills without moving
display(traindata[traindata['killsWithoutMoving'] == True].shape)
traindata[traindata['killsWithoutMoving'] == True].head(10)
# Remove outliers
traindata.drop(traindata[traindata['killsWithoutMoving'] == True].index, inplace=True)
# Drop roadKill 'cheaters'
traindata.drop(traindata[traindata['roadKills'] > 10].index, inplace=True)
# Remove outliers
traindata.drop(traindata[traindata['kills'] > 30].index, inplace=True)
# Remove outliers
traindata.drop(traindata[traindata['longestKill'] >= 1000].index, inplace=True)
traindata.drop(traindata[traindata['walkDistance'] >= 10000].index, inplace=True)
traindata.drop(traindata[traindata['rideDistance'] >= 20000].index, inplace=True)
# Remove outliers
traindata.drop(traindata[traindata['swimDistance'] >= 2000].index, inplace=True)
traindata.drop(traindata[traindata['weaponsAcquired'] >= 80].index, inplace=True)
# Remove outliers
traindata.drop(traindata[traindata['heals'] >= 40].index, inplace=True)               
sample = 500000
df_sample = traindata.sample(sample)
# Split sample into training data and target variable
df = df_sample.drop(columns = ['winPlacePerc']) #all columns except target
y = df_sample['winPlacePerc'] # Only target variable
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.12, random_state=42)

from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

# Function to print the MAE (Mean Absolute Error) score
# This is the metric used by Kaggle in this competition
def print_score(m : RandomForestRegressor):
    res = ['mae train: ', mean_absolute_error(m.predict(X_train), y_train), 
           'mae val: ', mean_absolute_error(m.predict(X_test), y_test)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)
m1 = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features='sqrt',
                          n_jobs=-1)
m1.fit(X_train, y_train)
print_score(m1)
# provides a way to analyze feature importance 
#takes in a model and a dataframe and pulls the importances and make them into a seperate table 
def rf_feat_importance(m, df):
    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}
                       ).sort_values('imp', ascending=False)


fi = rf_feat_importance(m1, df); fi[:10]
to_keep = fi[fi.imp>0.005].cols
print('Significant features: ', len(to_keep))
to_keep
df_keep = df[to_keep].copy()
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.12, random_state=42)

m2 = RandomForestRegressor(n_estimators=80, min_samples_leaf=3, max_features='sqrt',
                          n_jobs=-1)
m2.fit(X_train, y_train)
print_score(m2)
from sklearn import metrics
from scipy.cluster import hierarchy as hc
from fastai.imports import *

corr = np.round(scipy.stats.spearmanr(df_keep).correlation, 4)
corr_condensed = hc.distance.squareform(1-corr)
z = hc.linkage(corr_condensed, method='average')
fig = plt.figure(figsize=(14,10))
dendrogram = hc.dendrogram(z, labels=df_keep.columns, orientation='left', leaf_font_size=16)
plt.plot()
# Prepare data
val_perc_full = 0.12 # % to use for validation set
n_valid_full = int(val_perc_full * len(traindata)) 
n_trn_full = len(traindata)-n_valid_full
df_full = traindata.drop(columns = ['winPlacePerc']) # all columns except target
y = traindata['winPlacePerc'] # target variable
df_full = df_full[to_keep] # Keep only relevant features
X_train, X_test, y_train, y_test = train_test_split(df_full, y, test_size=0.12, random_state=42)

# Check dimensions of data
print('Sample train shape: ', X_train.shape, 
      'Sample target shape: ', y_train.shape, 
      'Sample validation shape: ', X_test.shape)
m3 = RandomForestRegressor(n_estimators=70, min_samples_leaf=3, max_features=0.5,
                          n_jobs=-1)
m3.fit(X_train, y_train)
print_score(m3)
# Add engineered features to the testdata set
testdata['totalDistance'] = testdata['walkDistance'] + testdata['rideDistance'] + testdata['swimDistance']
testdata['headshotRate'] = testdata['headshotKills']/testdata['kills']
testdata['headshotRate'] = testdata['headshotRate'].fillna(0)
testdata['playersJoined'] = testdata.groupby('matchId')['matchId'].transform('count')

# Create normalized features
testdata['killsNorm'] = testdata['kills']*((100-testdata['playersJoined'])/100 + 1)
testdata['damageDealtNorm'] = testdata['damageDealt']*((100-testdata['playersJoined'])/100 + 1)
testdata['maxPlaceNorm'] = testdata['maxPlace']*((100-testdata['playersJoined'])/100 + 1)
testdata['matchDurationNorm'] = testdata['matchDuration']*((100-testdata['playersJoined'])/100 + 1)
testdata['healsandboosts'] = testdata['heals'] + testdata['boosts']

testdata['killsWithoutMoving'] = ((testdata['kills'] > 0) & (testdata['totalDistance'] == 0))


# Turn groupId and match Id into categorical types
testdata['groupId'] = testdata['groupId'].astype('category')
testdata['matchId'] = testdata['matchId'].astype('category')

# Get category coding for groupId and matchID
testdata['groupId_cat'] = testdata['groupId'].cat.codes
testdata['matchId_cat'] = testdata['matchId'].cat.codes

# Remove irrelevant features from the testdata set
test_pred = testdata[to_keep].copy()

# Fill NaN with 0 (temporary)
test_pred.fillna(0, inplace=True)
test_pred.head()

# Make submission ready for Kaggle
# We use our final Random Forest model (m3) to get the predictions
predictions = np.clip(a = m3.predict(test_pred), a_min = 0.0, a_max = 1.0)
pred_df = pd.DataFrame({'Id' : testdata['Id'], 'winPlacePerc' : predictions})

# Create submission file
pred_df.to_csv("submission.csv", index=False)