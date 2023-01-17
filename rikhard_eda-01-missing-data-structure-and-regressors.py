import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from scipy.stats import pearsonr
# Load train data

df = pd.read_csv('../input/train.csv')
print(df.columns.tolist())

print('\nNumber of columns on train data:',len(df.columns))

print('\nNumber of data points:',len(df))

print('\nNumber of unique timestamp data points:',len(df['timestamp'].unique()))

print('\nNumber of unique id data points:',len(df['id'].unique()))

print('\nData types:',df.dtypes.unique())
df.select_dtypes(include=['O']).columns.tolist()
df.select_dtypes(include=['O']).head()
print('\nNumber of columns which have any NaN:',df.isnull().any().sum(),'/',len(df.columns))

print('\nNumber of rows which have any NaN:',df.isnull().any(axis=1).sum(),'/',len(df))
# Get the number of NaN's for each column, discarding those with zero NaN's

ranking = df.loc[:,df.isnull().any()].isnull().sum().sort_values()

# Turn into %

x = ranking.values/len(df)



# Plot bar chart

index = np.arange(len(ranking))

plt.bar(index, x)

plt.xlabel('Features')

plt.ylabel('% NaN observations')

plt.title('% of null data points for each feature')

plt.show()



print('Features:',ranking.index.tolist())
rank_features = ranking.index

accum_nulls = list()

accum_features = list()

for i, feature in enumerate(rank_features):

    # On each step, we add a new feature to the list

    accum_features.append(feature)

    # We calculate the number of rows with NaN's for that list of features

    accum_nulls.append(len(df)-len(df[accum_features].dropna()))
# Calculate the % of missing data

x = np.array(accum_nulls)/len(df)



# Plot

index = np.arange(len(x))

plt.bar(index, x)

plt.xlabel('Features')

plt.ylabel('% NaN accumulated observations')

plt.title('% of null data points accumulated until each feature')

plt.show()



print('Features:',accum_features)
y = df.groupby('timestamp').apply(lambda f: f.isnull().sum().sum()).values



plt.plot(y)

plt.xlabel('timestamp')

plt.ylabel('Number of NaNs')

plt.title('Missing data structure over timestamp')

plt.show()
y = df.groupby('timestamp').apply(lambda x: x.isnull().sum().sum()/len(x)).values



plt.plot(y)

plt.xlabel('timestamp')

plt.ylabel('Number of NaNs')

plt.title('Missing data structure over timestamp normalized')

plt.show()
# Get the list of features

features = df.iloc[:,2:-1].select_dtypes(exclude=['O']).columns.tolist()

# Get the target name

target = df.iloc[:,-1].name
correlations = dict()

for feat in features:

    df_temp = df[[feat,target]]

    df_temp = df_temp.dropna()

    x1 = df_temp[feat].values

    x2 = df_temp[target].values

    key = feat + ' vs ' + target

    correlations[key] = pearsonr(x1,x2)[0]
df_corrs = pd.DataFrame(correlations, index=['R']).T

df_corrs.loc[df_corrs['R'].abs().sort_values(ascending=False).index].iloc[:20]
y = df.loc[:,['num_room','full_sq',target]].dropna().sort_values(target,ascending=True).values

x = np.arange(y.shape[0])
plt.subplot(3, 1, 1)

plt.plot(x,y[:,0])

plt.title('num_room & full_sq vs price')

plt.ylabel('num_room')



plt.subplot(3, 1, 2)

plt.plot(x,y[:,1])

plt.ylabel('full_sq')



plt.subplot(3, 1, 3)

plt.plot(x,y[:,2],'r')

plt.ylabel('price')

    

plt.show()
x = df[target].values.astype('int')

sns.distplot(x)

plt.show()