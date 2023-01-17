import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('../input/train.csv') #import data
df.head(5)
df.shape #check shape
df.describe()
df.dtypes #check type
df.isnull().sum(axis=0) # Check missing values
match = df.groupby(['matchId']).count()['kills']
kills = match.sort_values(axis=0, ascending=False)
kills.head(5)
kills.tail(5)
df_corr = df.iloc[:,3:] #Drop ID's
corr=df_corr.corr()

sns.set(font_scale=1.15)
plt.figure(figsize=(24, 18))

sns.heatmap(corr, vmax=.8, linewidths=0.01,
            square=True,annot=True,cmap='YlGnBu',linecolor="black")
plt.title('Correlation between features');
fig, ax = plt.subplots(3,3, figsize=(20, 14))
sns.distplot(df.rideDistance, bins = 20, ax=ax[0,0])  
sns.distplot(df.damageDealt, bins = 20, ax=ax[0,1]) 
sns.distplot(df.killPlace, bins = 20, ax=ax[0,2]) 
sns.distplot(df.longestKill, bins = 20, ax=ax[1,0]) 
sns.distplot(df.maxPlace, bins = 20, ax=ax[1,1]) 
sns.distplot(df.rideDistance, bins = 20, ax=ax[1,2]) 
sns.distplot(df.swimDistance, bins = 20, ax=ax[2,0]) 
sns.distplot(df.walkDistance, bins = 20, ax=ax[2,1]) 
sns.distplot(df.winPoints, bins = 20, ax=ax[2,2]) 
plt.show()
sns.lmplot(x='kills', y='damageDealt', data=df)
sns.lmplot(x='winPlacePerc', y='walkDistance', data=df)
g = sns.factorplot('kills','DBNOs', data=df,
                   hue='boosts',
                   size=18,
                   aspect=0.7,
                   palette='Blues',
                   join=False,
              )
sns.jointplot(x="damageDealt", y="kills", data=df, height=10, ratio=3, color="r")
plt.show()
