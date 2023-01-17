import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (Shift+Enter) will list the files in the input directory

import os

path = '../input'

print(os.listdir("../input"))

import matplotlib.pyplot as plt

import seaborn as sns

# Any results you write to the current directory are saved as output.
train_path = f'{path}/train.csv'

test_path = f'{path}/test.csv'

train_df = pd.read_csv(train_path)
print(train_df.shape)

train_df.head()
target = train_df['target']

train_df = train_df.drop(['ID_code'], axis = 1).astype('float16')
target.value_counts().plot.bar()

print('%age value of 0s target variable:', target.value_counts()[0]/len(target) * 100)

print('%age value of 1s target variable:', target.value_counts()[1]/len(target) * 100)
from sklearn.decomposition import PCA

pca = PCA(n_components=3)

x_pca = pca.fit_transform(train_df)

print(pca.explained_variance_ratio_)

print(sum(pca.explained_variance_ratio_))



x_pca = pd.DataFrame(data = x_pca)

plt.scatter(x = x_pca[0], y = x_pca[1], data = x_pca, c = target.values)

plt.xlabel('pc1')

plt.ylabel('pc2')

plt.title('representation of classes with pca')
pca = PCA().fit(train_df)

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('number of components')

plt.ylabel('cumulative explained variance');
def density_feature_plot(df, features, grid_size = (8,8)):

    i = 0

    sns.set_style('whitegrid')

    plt.figure()

    fig, ax = plt.subplots(grid_size[0],grid_size[1],figsize=(16,16))

    

    t0 = df.loc[df['target'] == 0]

    t1 = df.loc[df['target'] == 1]



    for feature in features:

        i += 1

        plt.subplot(grid_size[0],grid_size[1],i)

        sns.kdeplot(t0[feature], bw=0.5,label=0)

        sns.kdeplot(t1[feature], bw=0.5,label=1)

        plt.xlabel(feature, fontsize=9)

    locs, labels = plt.xticks()

    plt.tick_params(axis='x', which='major', labelsize=6, pad=-6)

    plt.tick_params(axis='y', which='major', labelsize=6)

    plt.tight_layout()

    plt.show();
features = train_df.columns.values[2:66]

density_feature_plot(train_df, features)
features = train_df.columns.values[66:130]

density_feature_plot(train_df, features)
features = train_df.columns.values[130:166]

density_feature_plot(train_df, features, (6,6))
features = train_df.columns.values[166:]

density_feature_plot(train_df, features, (6,6))
train_df.iloc[:, 2:100].plot(kind='box', figsize=[16,8])
# Plot last 100 features.

train_df.iloc[:, 100:].plot(kind='box', figsize=[16,8])
corr_df = train_df.corr()
import seaborn as sns

sns.set(style="white")

mask = np.zeros_like(corr_df.iloc[:,1:], dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(16, 16))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr_df.iloc[:,1:], mask=mask, cmap=cmap, vmax=.2, center=0,

            square=True, linewidths=.5)
corr_target=corr_df.loc[corr_df.target>0.05]['target'].iloc[1:] # slight +ve co-rrelation

corr_target.plot(kind='bar')
corr_target=corr_df.loc[corr_df.target < -0.05]['target'].iloc[1:] 

corr_target.plot(kind='bar') # slight -ve co-rrelation
pd.DataFrame(train_df.isnull().sum()).T
train_df.duplicated().sum()