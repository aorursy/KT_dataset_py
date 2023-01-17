import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline



train_df = pd.read_csv('../input/winequality-red.csv')





corrmat = train_df.corr()

k = 10 #number of variables for heatmap

cols = corrmat.nlargest(k, 'quality')['quality'].index

cm = np.corrcoef(train_df[cols].values.T)

sns.set(font_scale=1.25)

#hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

#plt.show()

# Use alcohol, sulphate, citric acid

# outlier mgmt

# low range of SalePrice is pretty normal

# high range of SalePrice gets crazy - up to 7 stddev out

#scaled = StandardScaler().fit_transform(train_df['alcohol'][:,np.newaxis]);

#low_range = scaled[scaled[:,0].argsort()][:10]

#high_range= scaled[scaled[:,0].argsort()][-10:]

#print('outer range (low) of the distribution:')

#print(low_range)

#print('\nouter range (high) of the distribution:')

#print(high_range)



#var = 'alcohol'

#data = pd.concat([train_df['quality'],train_df[var]], axis = 1)

#data.plot.scatter(x=var, y = 'quality', ylim = (0,10));





# Variables to predict SalePrice:

# alcohol

# sulphate

# citric acid



# Take average Z score of three variables, apply that z score to prediction





train_df['alcoholZScore'] = pd.Series(len(train_df['alcohol']), index = train_df.index)

train_df['alcoholZScore'] = (train_df['alcohol'] - train_df['alcohol'].mean()) / train_df['alcohol'].std(ddof=0)



train_df['sulphatesZScore'] = pd.Series(len(train_df['sulphates']), index = train_df.index)

train_df['sulphatesZScore'] = (train_df['sulphates'] - train_df['sulphates'].mean()) / train_df['sulphates'].std(ddof=0)



train_df['citricAcidZScore'] = pd.Series(len(train_df['citric acid']), index = train_df.index)

train_df['citricAcidZScore'] = (train_df['citric acid'] - train_df['citric acid'].mean()) / train_df['citric acid'].std(ddof=0)



train_df['AvgZScore'] = pd.Series(len(train_df['citricAcidZScore']), index = train_df.index)

train_df['AvgZScore'] = (train_df['alcoholZScore'] + train_df['sulphatesZScore'] + train_df['citricAcidZScore']) / 3



# we get .5 correlation... not sure how to improve that.  No other variables have remote correlation

# probably should use pre-defined machine learning algorithm



train_df['Prediction'] = pd.Series(len(train_df['AvgZScore']), index = train_df.index)

train_df['Prediction'] = train_df['AvgZScore'] * train_df['quality'].std(ddof=0) + train_df['quality'].mean()



print(train_df.head())