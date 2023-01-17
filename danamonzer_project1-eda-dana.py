import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
plt.style.use('ggplot')
import os
import glob
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

data = pd.concat([pd.read_csv(f) for f in glob.glob("/kaggle/input/cee-498-project1-london-bike-sharing/train.csv") ])
#data = pd.read_csv("train.csv")
data.head()
data.info()
#data.dtypes

#all float except cnt is integer and timestamp is object
#data.shape 
# we have 12222 datapoints 
#We notice the data is clean and doesn't include null values
data['timestamp'] = data['timestamp'] .apply(lambda x :datetime.datetime.strptime(str(x),'%Y-%m-%d %H:%M:%S'))
data['month'] = data['timestamp'].apply(lambda x : float(str(x).split(' ')[0].split('-')[1]))
data['day'] = data['timestamp'].apply(lambda x : float(str(x).split(' ')[0].split('-')[2]))
data['hour'] = data['timestamp'].apply(lambda x : float(str(x).split(' ')[1].split(':')[0]))

data.head()
data_sample = data.sample(10000) 

p = sns.PairGrid(data=data_sample, vars=['t1', 't2', 'hum', 'wind_speed', 'weather_code', 'is_holiday', 'is_weekend','season','month','day','hour', 'cnt'])
p.map_diag(plt.hist)
p.map_offdiag(plt.scatter)
## We plot hour vs cnt alone
data_sample = data.sample(10000) 

p2 = sns.PairGrid(data=data_sample, vars=['hour', 'cnt'])
p2.map_diag(plt.hist)
p2.map_offdiag(plt.scatter)
figure, ((ax1, ax2)) = plt.subplots(nrows=1, ncols=2)
figure.set_size_inches(12, 8)

sns.boxplot(data=data, x='month', y='cnt', ax=ax1)
sns.boxplot(data=data, x='hour', y='cnt', ax=ax2)

fig,(ax1, ax2, ax3, ax4, ax5)= plt.subplots(nrows=5)
fig.set_size_inches(18,25)

sns.pointplot(data=data, x='hour', y='cnt', ax=ax1)
sns.pointplot(data=data, x='hour', y='cnt', hue='is_holiday', ax=ax2)
sns.pointplot(data=data, x='hour', y='cnt', hue='is_weekend', ax=ax3)
sns.pointplot(data=data, x='hour', y='cnt', hue='season', ax=ax4)
sns.pointplot(data=data, x='hour', y='cnt', hue='weather_code',ax=ax5)
df2 = data.describe()
df2

corrmat = data.corr()
f, ax = plt.subplots(figsize = (12,12))
sns.heatmap(corrmat, vmax=1, annot=True);