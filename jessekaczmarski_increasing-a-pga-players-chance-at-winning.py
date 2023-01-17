import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from statsmodels.api import Logit as logit



dataset = pd.read_csv('../input/pga-tour-20102018-data/PGA_Data_Historical.csv')
#Mike's code for transposing the variables out

df = dataset.set_index(['Player Name', 'Variable', 'Season'])['Value'].unstack('Variable').reset_index()



#Creating a list for the variables I want to keep

model_vars = ['Player Name', 

              'Season',

              'Top 10 Finishes - (1ST)',

              'Scoring Rating - (RATING)',

              'Accuracy Rating - (RATING)',

              'Short Game Rating - (RATING)',

              'Putting Rating - (RATING)',

              'Power Rating - (RATING)']
df=df[model_vars]

df.head(4)
df['Top 10 Finishes - (1ST)'].fillna(0, inplace = True)

df=df.dropna()
#Column renaming and converting data types

df.rename(columns={'Top 10 Finishes - (1ST)':'wins', 'Scoring Rating - (RATING)':'score','Accuracy Rating - (RATING)':'accu', 'Short Game Rating - (RATING)':'sg', 'Putting Rating - (RATING)':'putt','Power Rating - (RATING)':'power', 'Scoring Rating - (RATING)':'score'}, inplace = True)

for col in  df.columns[2:]:

   df[col] = df[col].astype(float)

df.info()



#Creating a binary win variable where 1 = at least 1 win and 0 = no wins

df['win'] = df['wins']

df['win'].replace([2,3,4,5], 1, inplace = True)

df['win'].value_counts()
plt.subplot(2,3,1)

df['accu'].plot.hist()

plt.xlabel('Accuracy Rating')

plt.subplot(2,3,2)

df['score'].plot.hist()

plt.xlabel('Scoring Rating')

plt.subplot(2,3,3)

df['sg'].plot.hist()

plt.xlabel('Short Game Rating')

plt.subplot(2,3,4)

df['putt'].plot.hist()

plt.xlabel('Putting Rating')

plt.subplot(2,3,5)

df['power'].plot.hist()

plt.xlabel('Power Rating')

plt.tight_layout()

print('Figure 1: Homogeneity in Ratings')

plt.show()
#Logistic regression

df_16 = df[df['Season']==2016]

df_15 = df[df['Season']==2015]

df_14 = df[df['Season']==2014]

df_13 = df[df['Season']==2013]

df_12 = df[df['Season']==2012]

df_11 = df[df['Season']==2011]

df_10 = df[df['Season']==2010]

logit16 = logit(df_16['win'], df_16[['accu','power','sg','putt','score']]).fit()

logit15 = logit(df_15['win'], df_15[['accu','power','sg','putt','score']]).fit()

logit14 = logit(df_14['win'], df_14[['accu','power','sg','putt','score']]).fit()

logit13 = logit(df_13['win'], df_13[['accu','power','sg','putt','score']]).fit()

logit12 = logit(df_12['win'], df_12[['accu','power','sg','putt','score']]).fit()

logit11 = logit(df_11['win'], df_11[['accu','power','sg','putt','score']]).fit()

logit10 = logit(df_10['win'], df_10[['accu','power','sg','putt','score']]).fit()



print('2016 Season:\n', logit16.get_margeff(at='mean',method='dydx').summary(),

      '\n2015 Season:\n', logit15.get_margeff(at='mean',method='dydx').summary(),

      '\n\n2014 Season:\n', logit14.get_margeff(at='mean',method='dydx').summary(),

      '\n\n2013 Season:\n', logit13.get_margeff(at='mean',method='dydx').summary(),

      '\n\n2012 Season:\n', logit12.get_margeff(at='mean',method='dydx').summary(),

      '\n\n2011 Season:\n', logit11.get_margeff(at='mean',method='dydx').summary(),

      '\n\n2010 Season:\n', logit10.get_margeff(at='mean',method='dydx').summary()

)