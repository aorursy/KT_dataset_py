import pandas as pd

import numpy as np

from sklearn.decomposition import PCA

from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

import seaborn as sns



# load the train data file

train = pd.read_csv("../input/feature-exploration-and-dataset-preparation/train_clean_standarized.csv", index_col=0)

train_resampled = pd.read_csv("../input/resampling/train_resampled.csv", index_col=0)
# let's separate our features from the target

X = train.drop('TARGET', axis=1)

y = train['TARGET'].values
sns.heatmap(X.corr());
X.corr()
pca_test = PCA().fit(X)
# plot the Cumulative Summation of the Explained Variance for the different number of components

plt.figure()

plt.plot(np.cumsum(pca_test.explained_variance_ratio_))

plt.xlabel('Number of Components')

plt.ylabel('Variance (%)') #for each component

plt.title('Dataset Explained Variance')

plt.show()
# instantiate PCA

pca = PCA(n_components=80)



# fit PCA

principalComponents = pca.fit_transform(X)
train_pc = pd.DataFrame(data = principalComponents)

train_target = pd.Series(y, name='TARGET')



train_pc_df = pd.concat([train_pc, train_target], axis=1)

train_pc_df.head(5)
sns.heatmap(train_pc.corr());
# we calculate the variance explained by priciple component

print('Variance of each component:', pca.explained_variance_ratio_)

print('\n Total Variance Explained:', round(sum(list(pca.explained_variance_ratio_))*100, 2))
# let's separate our features from the target

X_resampled = train_resampled.drop('TARGET', axis=1)

y_resampled = train_resampled['TARGET'].values
sns.heatmap(X_resampled.corr());
X_resampled.corr()
pca_resampled_test = PCA().fit(X_resampled)



# plot the Cumulative Summation of the Explained Variance for the different number of components

plt.figure()

plt.plot(np.cumsum(pca_resampled_test.explained_variance_ratio_))

plt.xlabel('Number of Components')

plt.ylabel('Variance (%)') #for each component

plt.title('Dataset Explained Variance')

plt.show()
# instantiate PCA

pca_resampled = PCA(n_components=80)



# fit PCA

principalComponents_resampled = pca_resampled.fit_transform(X_resampled)
train_resampled_pc = pd.DataFrame(data = principalComponents_resampled)

train_resampled_target = pd.Series(y_resampled, name='TARGET')



train_resampled_pc_df = pd.concat([train_resampled_pc, train_resampled_target], axis=1)

train_resampled_pc_df.head(5)
sns.heatmap(train_resampled_pc_df.corr());
# we calculate the variance explained by priciple component

print('Variance of each component:', pca_resampled.explained_variance_ratio_)

print('\n Total Variance Explained:', round(sum(list(pca_resampled.explained_variance_ratio_))*100, 2))
train_pc_df.to_csv('train_PCA.csv')

train_resampled_pc_df.to_csv('train_resampled_PCA.csv')