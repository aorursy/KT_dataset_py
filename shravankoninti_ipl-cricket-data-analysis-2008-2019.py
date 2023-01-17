# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# import libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error

from sklearn.ensemble import RandomForestRegressor



from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

from sklearn.model_selection import GroupKFold, GridSearchCV

from sklearn.model_selection import train_test_split



import lightgbm as lgb



from plotly import tools

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot

import warnings

warnings.filterwarnings("ignore")

init_notebook_mode(connected=True)
deliveries_df = pd.read_csv('/kaggle/input/ipldata/deliveries.csv')
deliveries_df.head()
matches_df = pd.read_csv('/kaggle/input/ipldata/matches.csv')
matches_df.head()
matches_df.describe()
matches_df.info()
#Missing vlaues checking

matches_df.isnull().sum()
# Get some basic stats on the data

print("Number of matches played so far in IPL : ", matches_df.shape[0])

print("Number of seasons in IPL : ", len(matches_df.season.unique()))

print("Number of Teams participated in IPL : ", len(matches_df.team1.unique()))

print("Number of Teams participated in IPL : ", len(matches_df.team2.unique()))
plt.figure(figsize=(12,6))

sns.countplot(x='season', data=matches_df)

plt.title('The total number of matches played in each year')

plt.show()
plt.figure(figsize=(12,6))

sns.countplot(x='venue', data=matches_df)

plt.xticks(rotation='vertical')

plt.show()
df = pd.melt(matches_df, id_vars=['id','season'], value_vars=['team1', 'team2'])

df.head()
df.columns = ['id', 'season', 'varaible', 'Team']
df.head()
plt.figure(figsize=(12,6))

sns.countplot(x='Team', data=df)

plt.xticks(rotation='vertical')

plt.show()
eden_df = matches_df[matches_df['venue'] == 'Eden Gardens']
eden_df.head()
eden_df_1 = pd.melt(eden_df, id_vars=['id','season'], value_vars=['team1', 'team2'])

eden_df_1.head()
eden_df_1.columns = ['id', 'season', 'varaible', 'Team']
plt.figure(figsize=(12,6))

sns.countplot(x='Team', data=eden_df_1)

plt.xticks(rotation='vertical')

plt.show()
matches_df.isnull().sum()
deliveries_df.isnull().sum()
deliveries_df.drop(['player_dismissed', 'dismissal_kind', 'fielder'],axis=1,inplace=True)
matches_df['date'] = pd.to_datetime(matches_df['date'])
matches_df["WeekDay"] = matches_df["date"].dt.weekday
matches_df.head()
df = matches_df[(matches_df['toss_decision'] == 'field') &  (matches_df['venue'] == 'Wankhede Stadium') &

             (matches_df['season'] >= 2008) & (matches_df['season'] <= 2019)

             ]
df.head()
df.shape
print('The win percentage of a team batting second at Wankhede Stadium during 2008 to 2016 is {}%'.format((df[df['win_by_wickets']>0].shape[0])*100/ df.shape[0]))
df[(df['win_by_wickets']>0)]['winner'].value_counts()
df[df['win_by_wickets']>0]['winner'].value_counts().plot(kind='bar', color='Orange', figsize=(12,6))

plt.xlabel("Team")

plt.ylabel("Count")

plt.title('Top Teams who win batting second are')

plt.show()
plt.figure(figsize=(12,6))



plt.title('Top Teams who win batting second are')

sns.countplot(x='winner', data=df[df['win_by_wickets']>0])

plt.xlabel("Team")

plt.ylabel("Count")

plt.xticks(rotation='vertical')

plt.show()
df = matches_df[['id', 'WeekDay','winner']]

df = df[df['winner'] == 'Kolkata Knight Riders']
df.head()
df['WeekDay'].value_counts().plot(kind='bar', color='green', figsize=(12,6))

plt.xlabel("Team")

plt.ylabel("Count")

plt.title('Kolkata Knight Riders winning on weekdays - where Monday is 0 and Sunday is 6')
df = matches_df.loc[matches_df.groupby('season').date.idxmax()]
df.head()
plt.figure(figsize=(12,6))



plt.title('Top winning teams in IPL history')

sns.countplot(x='winner', data=df)

plt.xlabel("Team")

plt.ylabel("Count")

plt.xticks(rotation='vertical')

plt.show()
# Let us take only the matches played in 2019 for this analysis #

matches_df_2019 = matches_df.ix[matches_df.season==2019,:]

matches_df_2019 = matches_df_2019.ix[matches_df_2019.dl_applied == 0,:]

matches_df_2019.head()
train_df = matches_df[matches_df['season'] != 2019]
test_df = matches_df[matches_df['season'] == 2019]
train_df.columns
train_df.head()
train_df = train_df[['city',  'team1', 'team2', 'toss_winner',

       'toss_decision', 'result', 'dl_applied', 'winner',  'venue',

        'WeekDay']]

test_df = test_df[['city',  'team1', 'team2', 'toss_winner',

       'toss_decision', 'result', 'dl_applied', 'winner',  'venue',

        'WeekDay']]
train_df.head()
train_df.isnull().sum()
test_df.isnull().sum()
train_df= train_df.dropna()

test_df = test_df.dropna()
train_df.head()
train_df.dtypes
test_df.dtypes
# Importing LabelEncoder and initializing it

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

# Iterating over all the common columns in train and test

for col in train_df.columns.values:

    # Encoding only categorical variables

    if train_df[col].dtypes=='object':

        print(col)

        # Using whole data to form an exhaustive list of levels

        data=train_df[col].append(test_df[col])

        le.fit(data.values) 

        train_df[col]=le.transform(train_df[col])

        test_df[col]=le.transform(test_df[col])
X = train_df.drop(['winner'],axis=1)

y = train_df['winner']



train_X = X

train_y = y



test_X = test_df.drop(['winner'],axis=1)

y_test = test_df['winner']
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split





# Split into training and test sets

X_train, X_valid , y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=17)



from sklearn import metrics

model = LogisticRegression()

model.fit(X_train,y_train)

prediction=model.predict(X_valid)

print('The accuracy of the Logistic Regression is', metrics.accuracy_score(prediction,y_valid))
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=500)

rf.fit(X_train, y_train)

y_valid_preds = rf.predict(X_valid)

print("The validation accuracy score is", metrics.accuracy_score(y_valid, y_valid_preds))



print("The test accuarcy score is", metrics.accuracy_score(y_test, rf.predict(test_X)))
coefs_df = pd.DataFrame()



coefs_df['Features'] = X_train.columns

coefs_df['Coefs'] = rf.feature_importances_

coefs_df.sort_values('Coefs', ascending=False).head(10)
coefs_df.set_index('Features', inplace=True)

coefs_df.sort_values('Coefs', ascending=False).head(10).plot(kind='bar', color='green', figsize=(12,6))
def check_winner(a,b):

    if (a == b):

        return 1

    else:

        return 0



train_df['win_toss_win_match'] = train_df.apply(lambda row: check_winner(row['toss_winner'],row['winner']),axis=1)

train_df.head()
train_df['win_toss_win_match'].value_counts().plot(kind='bar', color='green')



plt.title('Winning Toss and Winning Match')

plt.xlabel("Team")

plt.ylabel("Count")

plt.xticks(rotation='vertical')

plt.show()