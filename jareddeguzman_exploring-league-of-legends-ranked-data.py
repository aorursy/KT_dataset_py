# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
filepath = '/kaggle/input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv'

lol_df = pd.read_csv(filepath, index_col='gameId')
lol_df.head()
lol_df.columns
blueCols = lol_df.columns[lol_df.columns.str.contains(pat = 'blue')] 
abs(lol_df.corr())['blueWins'][blueCols].sort_values(ascending=False)
abs(lol_df.corr())['blueWardsPlaced'][blueCols].sort_values(ascending=False)
sns.set_style("white")

sns.distplot(lol_df['blueWardsPlaced'])
sns.boxplot(lol_df.blueWardsPlaced)
lol_df.blueWardsPlaced.describe()
q1 = lol_df.blueWardsPlaced.describe()[4]

q3 = lol_df.blueWardsPlaced.describe()[6]

iqr = q3 - q1
lower_bound = q1 -(1.5 * iqr) 

upper_bound = q3 +(1.5 * iqr) 

print('Range of Values that are not considered outliers:')

print(str(lower_bound) +' - ' + str(upper_bound))
outlier_condition = (lol_df['blueWardsPlaced'] > upper_bound) | (lol_df['blueWardsPlaced'] < lower_bound)

blueWard_outliers = len(lol_df['blueWardsPlaced'][outlier_condition])
print('Percentage of Blue Wards in the data that are outliers: ')

print(str(round((blueWard_outliers * 100) / lol_df.shape[0], 2)) + '%')
sns.set_style("white")

sns.distplot(lol_df['blueGoldPerMin'])
sns.scatterplot(data=lol_df, x='blueGoldPerMin', y='blueKills', hue='blueWins')
pd.pivot_table(lol_df, ['blueGoldPerMin', 'blueWardsPlaced', 'blueTotalExperience'], 'blueWins')
print('Average Gold Difference between winning and losing:')

print(round(abs(1586.411113 - 1714.526389)))



print('Average Exp. Difference between winning and losing:')

print(round(abs(17453.47161 - 18404.57789)))
abs(lol_df.corr())['blueGoldPerMin'][blueCols].sort_values(ascending=False)
lol_df.head()
#4 Since Supports don't aim to get gold early on in the game.

lol_df['avg_blue_gpm'] = lol_df['blueGoldPerMin'] / 4

lol_df['avg_blue_gpm'].head()
avg_gold_per_jungle_cs = 25.41

lol_df['avg_blue_jungle_farm'] = lol_df['blueTotalJungleMinionsKilled'] * avg_gold_per_jungle_cs
lol_df[['blueWins', 'blueTotalJungleMinionsKilled', 'avg_blue_jungle_farm']].head()
avgFullClearTime = 3.67

lol_df['avg_blue_jungle_farm_per_min'] = (lol_df['blueTotalJungleMinionsKilled'] * avg_gold_per_jungle_cs) / (avgFullClearTime)
lol_df[['blueWins', 'blueTotalJungleMinionsKilled', 'avg_blue_jungle_farm_per_min']].head()
lol_df['avg_blue_lane_farm_per_min'] = (lol_df['blueGoldPerMin'] - lol_df['avg_blue_jungle_farm_per_min']) / 4

lol_df[['blueGoldPerMin', 'avg_blue_lane_farm_per_min', 'avg_blue_jungle_farm_per_min']].head()
ax1 = lol_df.plot(kind='scatter', x='avg_blue_lane_farm_per_min', y='blueGoldPerMin', color='r')    

ax2 = lol_df.plot(kind='scatter', x='avg_blue_jungle_farm_per_min', y='blueGoldPerMin', color='g')    
lol_df[['blueGoldPerMin', 'avg_blue_lane_farm_per_min', 'avg_blue_jungle_farm_per_min']].head()
lol_df['blue_lane_jg_gold_diff_per_min'] = lol_df['avg_blue_lane_farm_per_min'] - lol_df['avg_blue_jungle_farm_per_min']

lol_df['blue_jg_lane_gold_diff_per_min'] = lol_df['blue_lane_jg_gold_diff_per_min'] * -1

lol_df['blue_lane_jg_gold_diff_per_min'].head()
sns.scatterplot(data=lol_df, y='blueGoldPerMin', x='blue_lane_jg_gold_diff_per_min')
print(lol_df['blueGoldPerMin'].corr(lol_df['blue_lane_jg_gold_diff_per_min']))

print(lol_df['blue_jg_lane_gold_diff_per_min'].corr(lol_df['blueKills']))
lol_df[['blueGoldPerMin', 'blue_jg_lane_gold_diff_per_min']][lol_df['blueWins'] == 1]
sns.lineplot(data=lol_df, x='avg_blue_jungle_farm_per_min', y='blueGoldPerMin')
sns.lineplot(data=lol_df, x='avg_blue_lane_farm_per_min', y='blueGoldPerMin')
lol_df[blueCols].columns
abs(lol_df.corr())['blueWins'][blueCols].sort_values(ascending=False)
X_columns = lol_df[blueCols].columns[blueCols != 'blueWins']

X_list = list(X_columns)

X_list.append('avg_blue_lane_farm_per_min')

X = lol_df[X_list].copy()

y = lol_df['blueWins']



X_list
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
from sklearn.linear_model import LogisticRegression

first_model = LogisticRegression(max_iter = 2000)

first_model.fit(X_train, y_train)



predictions = first_model.predict(X_test)
logistic_regression_results = pd.DataFrame(predictions)

logistic_regression_results.rename(columns={0: 'Predictions'})

logistic_regression_results['Actual'] = y_test.values

logistic_regression_results
from sklearn.metrics import accuracy_score

accuracy_score(y_test.values, predictions)
from sklearn.tree import DecisionTreeClassifier

second_model = DecisionTreeClassifier(criterion='entropy')

second_model.fit(X_train, y_train)

predictions = second_model.predict(X_test)
dectree_regression_results = pd.DataFrame()

dectree_regression_results['Predictions'] = predictions

dectree_regression_results['Actual'] = y_test.values

dectree_regression_results
accuracy_score(y_test.values, predictions)
from sklearn.ensemble import RandomForestClassifier



third_model = RandomForestClassifier()

third_model.fit(X_train, y_train)

predictions = third_model.predict(X_test)
randtree_regression_results = pd.DataFrame()

randtree_regression_results['Predictions'] = predictions

randtree_regression_results['Actual'] = y_test.values

randtree_regression_results
accuracy_score(predictions, y_test)