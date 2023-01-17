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
import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats as sc

fig,ax = plt.figsize=(12,8)
train_df=pd.read_csv('../input/train_V2.csv')

print('training data is loaded...')

test_df=pd.read_csv('../input/test_V2.csv')

print('testing data is loaded...')
train_df.head()
train_df.dtypes
null_cnt = train_df.isnull().sum().sort_values()

print('null count:', null_cnt[null_cnt>0])

# dropna

train_df.dropna(inplace=True)
def reduce_mem_usage(props):

    start_mem_usg = props.memory_usage().sum() / 1024**2 

    print("Memory usage of properties dataframe is :",start_mem_usg," MB")

    NAlist = [] # Keeps track of columns that have missing values filled in. 

    for col in props.columns:

        if props[col].dtype != object:  # Exclude strings

            

            # Print current column type

            print("******************************")

            print("Column: ",col)

            print("dtype before: ",props[col].dtype)

            

            # make variables for Int, max and min

            IsInt = False

            mx = props[col].max()

            mn = props[col].min()

            

            # Integer does not support NA, therefore, NA needs to be filled

            if not np.isfinite(props[col]).all(): 

                NAlist.append(col)

                props[col].fillna(mn-1,inplace=True)  

                   

            # test if column can be converted to an integer

            asint = props[col].fillna(0).astype(np.int64)

            result = (props[col] - asint)

            result = result.sum()

            if result > -0.01 and result < 0.01:

                IsInt = True



            

            # Make Integer/unsigned Integer datatypes

            if IsInt:

                if mn >= 0:

                    if mx < 255:

                        props[col] = props[col].astype(np.uint8)

                    elif mx < 65535:

                        props[col] = props[col].astype(np.uint16)

                    elif mx < 4294967295:

                        props[col] = props[col].astype(np.uint32)

                    else:

                        props[col] = props[col].astype(np.uint64)

                else:

                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:

                        props[col] = props[col].astype(np.int8)

                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:

                        props[col] = props[col].astype(np.int16)

                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:

                        props[col] = props[col].astype(np.int32)

                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:

                        props[col] = props[col].astype(np.int64)    

            

            # Make float datatypes 32 bit

            else:

                props[col] = props[col].astype(np.float32)

            

            # Print new column type

            print("dtype after: ",props[col].dtype)

            print("******************************")

    

    # Print final result

    print("___MEMORY USAGE AFTER COMPLETION:___")

    mem_usg = props.memory_usage().sum() / 1024**2 

    print("Memory usage is: ",mem_usg," MB")

    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")

    return props, NAlist
props, NAlist = reduce_mem_usage(train_df)

print("_________________")

print("")

print("Warning: the following columns have missing values filled with 'df['column_name'].min() -1': ")

print("_________________")

print("")

print(NAlist)
train_df.nunique()
sns.set(style="darkgrid")

ax = sns.barplot(x="assists", y="assists", data=train_df, estimator=lambda x: len(x) / len(train_df) * 100)

ax.set(ylabel="Percent")
#assists vs win percentage

sns.jointplot(x="winPlacePerc", y="kills", data=train_df, height=10, ratio=3, color="r")

plt.show()
#damage delt vs winning percentage

sns.scatterplot(x="winPlacePerc", y="kills", data=train_df)

plt.show()
#Damage dealt by zero kill players

data = train_df.copy()

data = data[data['kills']==0]

plt.figure(figsize=(15,10))

plt.title("Damage Dealt by 0 killers",fontsize=15)

sns.distplot(data['damageDealt'])

plt.show()
#let us investigate further 

print("There are {} players won the match without single win".format(len(data[data['winPlacePerc']==0])))

print("There are {} players won the match without single Damage".format(len(data[data['damageDealt']==0])))
#Ride Distance 

plt.figure(figsize=(15,10))

sns.distplot(data['rideDistance'])

plt.show()
print("On average the persons runs {}m, while the Maxium distance rided is {}m and 90% of the people rided distance is {}m".format(data['rideDistance'].mean(),data['rideDistance'].max(),data['rideDistance'].quantile(0.9)))
plt.figure(figsize=(15,10))

sns.distplot(data['walkDistance'])

plt.show()
print("On average the persons walks {}m, while the Maxium distance walked is {}m and 90% of the people walks distance is {}m".format(data['walkDistance'].mean(),data['walkDistance'].max(),data['walkDistance'].quantile(0.9)))
#vehicle destoys vs winnnig percentage

plt.figure(figsize=(15,10))

sns.pointplot(x='vehicleDestroys',y='winPlacePerc',data=train_df)

plt.title('vehicleDestroys Vs Winning %')

plt.show()
data = data[data['heals'] < data['heals'].quantile(0.99)]

data
#match duration vs winning percentage

plt.figure(figsize=(20,10))

sns.distplot(train_df['matchDuration'])

plt.title('Duration vs Winnign percentage')

plt.show()
#Match Type vs winning percentage

# plt.figure(figsize=(20,10))

# sns.catplot(x='matchType',y='winPlacePerc',kind='bar',data=train_df)

# plt.title('Match Type vs Winning perc')

# plt.show()
#Swimming vs the win distribution

print("The average person swims for {:.1f}m, 99% of people have swimemd {}m or less, while the olympic champion swimmed for {}m.".format(train_df['swimDistance'].mean(), train_df['swimDistance'].quantile(0.99), train_df['swimDistance'].max()))
data = train_df[train_df['swimDistance'] < train_df['swimDistance'].quantile(0.95)]

plt.figure(figsize=(15,10))

plt.title("Swim Distance Distribution",fontsize=15)

sns.distplot(data['swimDistance'])

plt.show()
swim=train_df.copy()

swim['swimDistance']=pd.cut(swim['swimDistance'],[-1, 0, 5, 20, 5286], labels=['0m','1-5m', '6-20m', '20m+'])

plt.figure(figsize=(15,8))

sns.boxplot(x="swimDistance", y="winPlacePerc", data=swim)

plt.show()
print("The average person uses {:.1f} heal items, 99% of people use {} or less, while the maximum % used is {}.".format(train_df['heals'].mean(), train_df['heals'].quantile(0.99), train_df['heals'].max()))

print("The average person uses {:.1f} boost items, 99% of people use {} or less, while the maximum % used is {}.".format(train_df['boosts'].mean(), train_df['boosts'].quantile(0.99), train_df['boosts'].max()))
#pointplot to determin heals vs boosts vs winplacePerc

data = train_df.copy()

data = data[data['heals'] < data['heals'].quantile(0.99)]

data = data[data['boosts'] < data['boosts'].quantile(0.99)]



f,ax1 = plt.subplots(figsize =(20,10))

sns.pointplot(x='heals',y='winPlacePerc',data=data,color='red',alpha=0.8)

sns.pointplot(x='boosts',y='winPlacePerc',data=data,color='green',alpha=0.8)

plt.text(4,0.6,'Heals',color='red',fontsize = 17,style = 'italic')

plt.text(4,0.55,'Boosts',color='green',fontsize = 17,style = 'italic')

plt.xlabel('Number of heal/boost items',fontsize = 15,color='blue')

plt.ylabel('Win Percentage',fontsize = 15,color='blue')

plt.title('Heals vs Boosts',fontsize = 20,color='blue')

plt.legend(loc='best')

plt.grid()

plt.show()
k = 5 #number of variables for heatmap

f,ax = plt.subplots(figsize=(11, 11))

cols = train_df.corr().nlargest(k, 'winPlacePerc')['winPlacePerc'].index

cm = np.corrcoef(train_df[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
# Create headshot_rate feature

train_df['headshot_rate'] = train_df['headshotKills'] / train_df['kills']

train_df['headshot_rate'] = train_df['headshot_rate'].fillna(0)
#Total Distance covered

train_df['totalDistance'] = train_df['rideDistance'] + train_df['walkDistance'] + train_df['swimDistance']
# Create feature killsWithoutMoving

train_df['killsWithoutMoving'] = ((train_df['kills'] > 0) & (train_df['totalDistance'] == 0))
#Encoding matchType to categorical data type

# One hot encode matchType

train_df = pd.get_dummies(train_df, columns=['matchType'])



# Take a look at the encoding

matchType_encoding = train_df.filter(regex='matchType')

matchType_encoding.head()
# Turn groupId and match Id into categorical types

train_df['groupId'] = train_df['groupId'].astype('category')

train_df['matchId'] = train_df['matchId'].astype('category')



# Get category coding for groupId and matchID

train_df['groupId_cat'] = train_df['groupId'].cat.codes

train_df['matchId_cat'] = train_df['matchId'].cat.codes



# Get rid of old columns

train_df.drop(columns=['groupId', 'matchId'], inplace=True)



# Lets take a look at our newly created features

train_df[['groupId_cat', 'matchId_cat']].head()
train_df.head()
train_df.drop(columns = ['Id'], inplace=True)
#Using the subset of data for the splitting purpose

sample = 500000

df_sample = train_df.sample(sample)
# Split sample into training data and target variable

df = df_sample.drop(columns = ['winPlacePerc']) #all columns except target

y = df_sample['winPlacePerc'] # Only target variable
# Function for splitting training and validation data

def split_vals(a, n : int): 

    return a[:n].copy(), a[n:].copy()

val_perc = 0.12 # % to use for validation set

n_valid = int(val_perc * sample) 

n_trn = len(df)-n_valid

# Split data

raw_train, raw_valid = split_vals(df_sample, n_trn)

X_train, X_valid = split_vals(df, n_trn)

y_train, y_valid = split_vals(y, n_trn)



# Check dimensions of samples

print('Sample train shape: ', X_train.shape, 

      'Sample target shape: ', y_train.shape, 

      'Sample validation shape: ', X_valid.shape)
# Metric used for the PUBG competition (Mean Absolute Error (MAE))

from sklearn.metrics import mean_absolute_error

from sklearn.ensemble import RandomForestRegressor

# Function to print the MAE (Mean Absolute Error) score

# This is the metric used by Kaggle in this competition

def print_score(m : RandomForestRegressor):

    res = ['mae train: ', mean_absolute_error(m.predict(X_train), y_train), 

           'mae val: ', mean_absolute_error(m.predict(X_valid), y_valid)]

    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)

    print(res)
m1 = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features='sqrt',

                          n_jobs=-1)

m1.fit(X_train, y_train)

print_score(m1)
def rf_feat_importance(m, df):

    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}

                       ).sort_values('imp', ascending=False)
fi = rf_feat_importance(m1, df); fi[:10]
# Plot a feature importance graph for the 20 most important features

plot1 = fi[:20].plot('cols', 'imp', figsize=(14,6), legend=False, kind = 'barh')

plot1
to_keep = fi[fi.imp>0.005].cols

print('Significant features: ', len(to_keep))

to_keep
# Make a DataFrame with only significant features

df_keep = df[to_keep].copy()

X_train, X_valid = split_vals(df_keep, n_trn)
m2 = RandomForestRegressor(n_estimators=80, min_samples_leaf=3, max_features='sqrt',

                          n_jobs=-1)

m2.fit(X_train, y_train)

print_score(m2)
fi_to_keep = rf_feat_importance(m2, df_keep)

plot2 = fi_to_keep.plot('cols', 'imp', figsize=(14,6), legend=False, kind = 'barh')

plot2
# Correlation heatmap

corr = df_keep.corr()



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(15, 11))



# Create heatmap

heatmap = sns.heatmap(corr,annot=True)
# Add engineered features to the test_df set

test_df['headshot_rate'] = test_df['headshotKills'] / test_df['kills']

test_df['headshot_rate'] = test_df['headshot_rate'].fillna(0)

test_df['totalDistance'] = test_df['rideDistance'] + test_df['walkDistance'] + test_df['swimDistance']

test_df['playersJoined'] = test_df.groupby('matchId')['matchId'].transform('count')

test_df['healsandboosts'] = test_df['heals'] + test_df['boosts']

test_df['killsWithoutMoving'] = ((test_df['kills'] > 0) & (test_df['totalDistance'] == 0))



# Turn groupId and match Id into categorical types

test_df['groupId'] = test_df['groupId'].astype('category')

test_df['matchId'] = test_df['matchId'].astype('category')



# Get category coding for groupId and matchID

test_df['groupId_cat'] = test_df['groupId'].cat.codes

test_df['matchId_cat'] = test_df['matchId'].cat.codes



# Remove irrelevant features from the test_df set

test_df_pred = test_df[to_keep].copy()



# Fill NaN with 0 (temporary)

test_df_pred.fillna(0, inplace=True)

test_df_pred.head()
# Make submission ready for Kaggle

# We use our final Random Forest model (m3) to get the predictions

predictions = np.clip(a = m2.predict(test_df_pred), a_min = 0.0, a_max = 1.0)

pred_df = pd.DataFrame({'Id' : test_df['Id'], 'winPlacePerc' : predictions})



# Create submission file

pred_df.to_csv("submission.csv", index=False)