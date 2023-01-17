import pandas as pd
# Change the path to the dataset file if needed. 

PATH = '../input/athlete_events.csv'
data = pd.read_csv(PATH)

data.head()
data.info()
# You code here

data[data['Year']==1992].groupby('Sex')['Age'].min()
# You code here

all_2012_males = data[(data['Sex']=='M')&(data['Year']==2012)&(data['Sport']=='Basketball')].drop_duplicates('Name')

all_2012_males.head()
print(all_2012_males.shape)
basket_2012_males = all_2012_males[all_2012_males['Sport']=='Basketball']

print(basket_2012_males.shape)
import numpy as np

print(np.round(basket_2012_males.shape[0]/all_2012_males.shape[0]*100, 1))
144/5858
# You code here

data['Sport'].value_counts().index
data[(data['Year']==2000)&(data['Sex']=='F')&(data['Sport']=='Tennis')]['Height'].describe()
# You code here

data[data['Year']==2006].sort_values('Weight', ascending=False).head(1)['Sport']
# You code here

data[data['Name']=='John Aalberg']['Year'].nunique()
# You code here

data[(data['Year']==2008)&(data['Team']=='Switzerland')&(data['Sport']=='Tennis')&(data['Medal']=='Gold')]
# You code here

data1 = data.dropna(subset=['Medal'])

data1[(data1['Team']=='Italy')&(data1['Year']==2016)].shape[0] > data1[(data1['Team']=='Spain')&(data1['Year']==2016)].shape[0]
# You code here

data[data['Year']==2008].drop_duplicates('Name')['Age'].hist(bins=[15, 25, 35, 45, 55]);
# You code here

(data[(data['Season']=='Winter')&(data['City']=='Squaw Valley')].shape[0] > 0), (data[(data['Season']=='Summer')&(data['City']=='Atlanta')].shape[0] > 0)
# You code here

abs(data[data['Year']==2002]['Sport'].nunique() - data[data['Year']==1986]['Sport'].nunique())