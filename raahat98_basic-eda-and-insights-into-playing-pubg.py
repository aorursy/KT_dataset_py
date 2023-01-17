import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
train_data.shape, test_data.shape
train_data.info()
train_data.describe()
test_data.describe()
train_data.head()
test_data.tail()
# Plot countplots and distplots depicting how various features are related to the final place, 
# and some features to each other. Ditch some features and combine some others to get more meaningful
# features. Then, train using some basic scikit learn model.
data = train_data.copy()
avg = data['kills'].mean()
ninety = data['kills'].quantile(0.95)
highest = data['kills'].max()
print('Average no. of kills: {0} \nKills of 95% of people: {1} \nMax Kills: {2}'.format(avg, ninety, highest))
# Plotting no. of Kills

plt.figure(figsize = (15, 10))
sns.countplot(data['kills'].sort_values())
plt.title('Kill Count')
plt.show()
# Still plotting no. of Kills

data1 = data.copy()

data1['kills'].astype('str')
data1.loc[data['kills'] > data['kills'].quantile(0.95)] = '4+'

plt.figure(figsize = (15, 10))
sns.countplot(data1['kills'].astype('str').sort_values())
plt.title('Modified Kill Count')
plt.show()
# Kills vs Damage Dealt

plt.figure(figsize = (15, 10))
sns.scatterplot(x = data['kills'], y = data['damageDealt'])
plt.title('Kills vs Damage Dealt')
plt.show()
# Kills vs Winning Percentage

plt.figure(figsize = (15, 10))
sns.jointplot(y = data['kills'], x = data['winPlacePerc'], height=10, ratio=3, color="b")
plt.title('Kills vs Winning Precentage')
plt.show()
data = train_data.copy()
avg = data['headshotKills'].mean()
ninety = data['headshotKills'].quantile(0.95)
highest = data['headshotKills'].max()
print('Average no. of headshots: {0} \nNo. of headshots of 95% of people: {1} \nMax Headshots: {2}'.format(avg, ninety, highest))
# Plotting no. of Headshots

plt.figure(figsize = (15, 10))
sns.countplot(data['headshotKills'].sort_values())
plt.title('Headshot Count')
plt.show()
# Headshots vs Winning Prediction

plt.figure(figsize = (15, 10))
sns.jointplot(y = data['headshotKills'], x = data['winPlacePerc'], height=10, ratio=3, color="c")
plt.title('Headshots vs Winning Precentage')
plt.show()
avg = data['assists'].mean()
ninety = data['assists'].quantile(0.95)
highest = data['assists'].max()
print('Average no. of assists: {0} \nNo. of assists of 95% of people: {1} \nMax assists: {2}'.format(avg, ninety, highest))
# Plotting no. of Assists

plt.figure(figsize = (15, 10))
sns.countplot(data['assists'].sort_values())
plt.title('Assist Count')
plt.show()
# Assists vs Winning Prediction

plt.figure(figsize = (15, 10))
sns.jointplot(y = data['assists'], x = data['winPlacePerc'], height=10, ratio=3, color="y")
plt.title('Assists vs Winning Precentage')
plt.show()
avg = data['DBNOs'].mean()
ninety = data['DBNOs'].quantile(0.95)
highest = data['DBNOs'].max()
print('Average no. of DBNOs: {0} \nNo. of DBNOs of 95% of people: {1} \nMax DBNOs: {2}'.format(avg, ninety, highest))
# Plotting no. of DBNOs

plt.figure(figsize = (15, 10))
sns.countplot(data['DBNOs'].sort_values())
plt.title('DBNOs Count')
plt.show()
# DBNOs vs Kills

plt.figure(figsize = (10, 5))
sns.scatterplot(x = data['DBNOs'], y = data['kills'])
plt.title('DBNOs vs Kills')
plt.show()
# DBNOs vs Winning Prediction

plt.figure(figsize = (15, 10))
sns.jointplot(y = data['DBNOs'], x = data['winPlacePerc'], height=10, ratio=3, color="g")
plt.title('DBNOs vs Winning Precentage')
plt.show()
# heals, boosts, distances, game modes (solo, duo, squad)