# Import necessary libraries

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns 

import os

import warnings



warnings.filterwarnings("ignore")

plt.rcParams["figure.figsize"] = (18,8)

sns.set(rc={'figure.figsize':(18,8)})
data = pd.read_csv("../input/train_V2.csv")

print("Done loading the data")

# Extra original dataset in case I need it

data2 = data.copy()

print("Done copying the data")
data.shape
# Let's get some information about the data

data.info()
# A look into data

data.head()
data.drop(columns=['rankPoints'], inplace=True)
# Now check for missing values

data.isnull().values.any()
data.isnull().sum()
data.dropna(inplace=True)

data.isnull().values.any()
# A more detailed look into the data

data.describe()
sns.countplot(data['kills']).set_title("Kills");
sns.lineplot(x="kills", y='killPoints', data=data);
sns.lineplot(x="kills", y='winPlacePerc', data=data);
zero_kills = data.copy()

zero_kills = zero_kills[zero_kills['kills']==0]

# Scatter plot instead of lineplot since line is hard to see 

sns.scatterplot(x='kills', y='killPoints', data=zero_kills);
# Same reason as previous line

sns.scatterplot(x="kills", y='winPlacePerc', data=zero_kills);
sns.lineplot(x="killPlace", y='winPlacePerc', data=zero_kills);
sns.distplot(data['walkDistance'], color = 'sandybrown');


sns.lineplot(x="walkDistance", y='winPlacePerc', data=data, color='sandybrown');
sns.jointplot(x="heals", y="winPlacePerc",  data=data, height = 12, ratio = 4, color='seagreen');
sns.jointplot(x="boosts", y="winPlacePerc",  data=data, height = 12, ratio = 4, color='seagreen');
sns.jointplot(x="weaponsAcquired", y="winPlacePerc",  data=data, height = 10, ratio = 4, color='orchid');
sns.lineplot(x="damageDealt", y='winPlacePerc', data=data, color='darkgreen');
sns.distplot(data['matchDuration'], color='darkgreen');
data['killsPerMeter'] = data['kills']/data['walkDistance']

data['killsPerMeter'].fillna(0, inplace=True)

data['killsPerMeter'].replace(np.inf, 0, inplace=True)
data['healsPerMeter'] = data['heals'] / data['walkDistance']

data['healsPerMeter'].fillna(0, inplace=True)

data['healsPerMeter'].replace(np.inf, 0, inplace=True)
data['killsPerHeal'] = data['kills'] / data['heals']

data['killsPerHeal'].fillna(0, inplace=True)

data['killsPerHeal'].replace(np.inf, 0, inplace=True)
data['killsPerSecond'] = data['kills'] / data['matchDuration']

data['killsPerSecond'].fillna(0, inplace=True)

data['killsPerSecond'].replace(np.inf, 0, inplace=True)
data['TotalHealsPerTotalDistance'] = (data['boosts'] + data['heals']) / (data['walkDistance'] + data['rideDistance'] + data['swimDistance'])

data['TotalHealsPerTotalDistance'].fillna(0, inplace=True)

data['TotalHealsPerTotalDistance'].replace(np.inf, 0, inplace=True)
data['killPlacePerMaxPlace'] = data['killPlace'] / data['maxPlace']

data['killPlacePerMaxPlace'].fillna(0, inplace=True)

data['killPlacePerMaxPlace'].replace(np.inf, 0, inplace=True)
len(data.columns)
# Check Correlations

f, ax = plt.subplots(figsize=(20,20))

sns.heatmap(data.corr(), annot=True, linewidths=1,fmt='.2f', ax=ax)

plt.show()
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error as mae

from sklearn.metrics import mean_squared_error as mse

from sklearn.model_selection import train_test_split
def lin_reg_exp(df):

    

    # i is columns that we want to test 

    i = 29

    num_features = []

    error_mae = []

    error_mse = []

    feature_dropped = []

    

    target = 'winPlacePerc'

    # Right now ignorning categorical variables but will look at them and incorporate soon

    drop = ['Id', 'matchId', 'groupId', 'matchType', target]

    

    X = df.copy()

    X.dropna(inplace=True)

    y = df.copy()

    y.dropna(inplace=True)

    y = y[target]

    X.drop(columns=drop, axis=1, inplace=True)

        

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=12)

    

    model = LinearRegression()

    model.fit(X_train, y_train)

        

    y_pred = model.predict(X_test)

    

    num_features.append(len(X_train.columns))

    error_mae.append(mae(y_test, y_pred))

    error_mse.append(mse(y_test, y_pred))

    feature_dropped.append('None')

    print("First pass done")

    

    while(i >= 1):               

        X = df.copy()

        X.dropna(inplace=True)

        y = df.copy()

        y.dropna(inplace=True)

        y = y[target]

        X.drop(columns=drop, axis=1, inplace=True)

        

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=12)

        

        feature_dropped.append(X_train.columns[i-1])

        X_train.drop(X_train.columns[i-1], axis= 1, inplace=True)

        X_test.drop(X_test.columns[i-1], axis = 1, inplace=True)

        

        

        model = LinearRegression()

        model.fit(X_train, y_train)

        

        y_pred = model.predict(X_test)

        num_features.append(i)

        error_mae.append(mae(y_test, y_pred))

        error_mse.append(mse(y_test, y_pred))

        print(i)

        i -= 1

        

    results = pd.DataFrame({'MAE Error': error_mae,

                          'MSE Error': error_mse,

                          'Dropped Feature': feature_dropped})

    

    return(results)
lin_reg_exp(data)
data.to_csv(r'Training_Data_New.csv')
