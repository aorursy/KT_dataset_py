import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))
training_set = pd.read_csv('../input/train_V2.csv')
training_set.head()
training_set.shape
training_set.columns
training_set.describe()
training_set.isna().sum()
training_set.plot(x = "kills", y = "damageDealt", kind="scatter", figsize = (15,10))
import seaborn as sns
headshots = training_set[training_set['headshotKills'] > 0]
plt.figure(figsize = (15, 5))
sns.countplot(headshots['headshotKills'])
dbno = training_set[training_set['DBNOs'] > 0]
plt.figure(figsize = (15, 5))
sns.countplot(dbno['DBNOs'])
training_set.plot(x = 'kills', y = 'DBNOs', kind = 'scatter', figsize = (15, 10))
walk0 = training_set["walkDistance"] == 0
ride0 = training_set["rideDistance"] == 0
swim0 = training_set["swimDistance"] == 0
print("{} of players didn't walk at all, {} players didn't drive and {} didn't swim." .format(walk0.sum(),ride0.sum(),swim0.sum()))
walk0_data = training_set[walk0]
print("Average place for non walkers is {:.3f}, minimum is {}, and best is {}, 95% players have a score below {}."
     .format(walk0_data['winPlacePerc'].mean(), walk0_data['winPlacePerc'].min(), walk0_data['winPlacePerc'].max(), walk0_data['winPlacePerc'].quantile(0.95)))
walk0_data.hist('winPlacePerc',bins = 50, figsize = (15, 5))
suspicious = training_set.query('walkDistance == 0 & winPlacePerc == 1')
suspicious.head()
print("Maximum ride distance for suspected entries is {:.3f} meters, and swim distance is {:.1f} meters." .format(suspicious["rideDistance"].max(), suspicious["swimDistance"].max()))
plt.plot(suspicious['swimDistance'])
suspicious_non_swimmer = suspicious[suspicious['swimDistance'] == 0]
suspicious_non_swimmer.shape
ride = training_set.query('rideDistance >0 & rideDistance <10000')
walk = training_set.query('walkDistance >0 & walkDistance <4000')
ride.hist('rideDistance', bins=40, figsize = (15,10))
walk.hist('walkDistance', bins=40, figsize = (15,10))