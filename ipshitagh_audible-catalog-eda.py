import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

print("Setup complete")
audiob = pd.read_csv('../input/audible-complete-catalog/Audible_Catlog.csv')

audiob_adv = pd.read_csv('../input/audible-complete-catalog/Audible_Catlog_Advanced_Features.csv')
audiob.head(5)
audiob_adv.head(5)
temp_df = audiob_adv['Listening Time'].str.extract(r'(\d+)[^\d]+(\d+)').astype('float64')

audiob_adv["Time"] = temp_df.iloc[:,0]*60 + temp_df.iloc[:,1]
audiob_adv.drop(['Listening Time','Ranks and Genre'], axis = 1)
audiob_adv.isnull().values.any()
sns.heatmap(audiob_adv.isnull(), cbar=False)
audiob_adv["Number of Reviews"].fillna(0, inplace = True)

audiob_adv["Time"].fillna(10, inplace = True)
sns.heatmap(audiob_adv.isnull(), cbar=False)
audiob_adv.rename(columns = {"Number of Reviews": "Number_of_Reviews"},  

                             inplace = True)
audiob_adv["Number_of_Reviews"] = audiob_adv.Number_of_Reviews.astype(float)

audiob_adv["Price"] = audiob_adv.Price.astype(float)

plt.figure(figsize=(30,5))

sns.boxplot(x=audiob_adv['Price'],palette = 'colorblind')
plt.figure(figsize=(30,5))

sns.boxplot(x=audiob_adv['Rating'],palette = 'colorblind')
audiob_adv = audiob_adv[~(audiob_adv['Rating']<=0)]
plt.figure(figsize=(30,5))

sns.boxplot(x=audiob_adv['Rating'],palette = 'colorblind')
plt.figure(figsize=(30,5))

sns.boxplot(x=audiob_adv['Time'],palette = 'colorblind')
sns.set_context('talk')

plt.figure(figsize=(20,10))

cnt = audiob_adv['Author'].value_counts().to_frame()[0:20]

sns.barplot(x= cnt['Author'], y =cnt.index, data=cnt, palette='deep',orient='h')

plt.title('Distribution of Audio Books of Top 20 Authors');
plt.figure(figsize=(16,8))



cnt = audiob_adv.groupby(['Author'])['Price'].max().sort_values(ascending=False).to_frame()[:20]

g2 = sns.barplot(x = cnt['Price'], y = cnt.index)

g2.set_title('Most expensive book by Author')

g2.set_ylabel('Author')

g2.set_xlabel('')
plt.figure(figsize=(16,8))



cnt = audiob_adv.groupby(['Author'])['Rating'].max().sort_values(ascending=False).to_frame()[:20]

g2 = sns.barplot(x = cnt['Rating'], y = cnt.index)

g2.set_title('Highest Rated book by Author')

g2.set_ylabel('Author')

g2.set_xlabel('')
audiob_adv.replace(to_replace='Satyajit Rai', value = 'Satyajit Ray', inplace=True)
plt.figure(figsize=(16,8))



cnt = audiob_adv.groupby(['Author'])['Rating'].max().sort_values(ascending=True).to_frame()[:20]

g2 = sns.barplot(x = cnt['Rating'], y = cnt.index)

g2.set_title('Lowest Rated book by Author')

g2.set_ylabel('Author')

g2.set_xlabel('')
trial = audiob_adv[~(audiob_adv['Time']<20)]
plt.figure(figsize=(16,8))



cnt = trial.groupby(['Author'])['Time'].max().sort_values(ascending=True).to_frame()[:20]

g2 = sns.barplot(x = cnt['Time'], y = cnt.index)

g2.set_title('Shortest book by Author (in minutes)')

g2.set_ylabel('Author')

g2.set_xlabel('')
plt.figure(figsize=(10,10))

rating= audiob_adv.Rating.astype(float)

sns.distplot(rating, bins=20)
plt.figure(figsize=(15,10))

sns.set_context('paper')

ax = sns.jointplot(x="Rating", y="Time", data = audiob_adv, color = 'crimson')

ax.set_axis_labels("Rating", "Time")

plt.show()
plt.figure(figsize=(15,10))

sns.set_context('paper')

ax = sns.jointplot(x="Rating", y="Price", data = audiob_adv, color = 'crimson')

ax.set_axis_labels("Rating", "Price")

plt.show()
trial = audiob_adv[~(audiob_adv['Price']>3000)]
plt.figure(figsize=(15,10))

sns.set_context('paper')

ax = sns.jointplot(x="Rating", y="Price", data = trial, color = 'crimson')

ax.set_axis_labels("Rating", "Price")

plt.show()
ax = sns.jointplot(x="Rating", y="Number_of_Reviews", data = audiob_adv)

ax.set_axis_labels("Rating", "Number of Reviews")
sns.heatmap(audiob_adv.corr(),vmin=-1, vmax=1, annot=True);
plt.figure(figsize=(15,10))

sns.set_context('paper')

ax = sns.jointplot(x="Price", y="Time", data = audiob_adv, color = 'crimson')

ax.set_axis_labels("Price", "Time")

plt.show()
trial = audiob_adv[~(audiob_adv['Price']>3000)]
plt.figure(figsize=(15,10))

sns.set_context('paper')

ax = sns.jointplot(x="Price", y="Time", data = trial, color = 'crimson')

ax.set_axis_labels("Price", "Time")

plt.show()
plt.figure(figsize=(10,10))

rating= audiob_adv.Time.astype(float)

sns.distplot(rating, bins=20)