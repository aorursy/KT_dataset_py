import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import os
print(os.listdir("../input"))
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

train_df.head()
train_df.loc[:,'Elevation':'Horizontal_Distance_To_Fire_Points'].describe()
test_df.loc[:,'Elevation':'Horizontal_Distance_To_Fire_Points'].describe()
plt.figure(figsize = (20, 30))

for pos, key in enumerate(train_df.columns[1:11]):
    plt.subplot(5, 2, pos+1)
    vp = plt.violinplot([train_df[key], test_df[key]])

    vp['bodies'][0].set_facecolor('#004488')
    vp['bodies'][0].set_edgecolor('black')
    vp['bodies'][0].set_alpha(1)

    vp['bodies'][1].set_facecolor('#FF4400')
    vp['bodies'][1].set_edgecolor('black')
    vp['bodies'][1].set_alpha(1)

    plt.ylabel(key, fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.xticks([1,2], ['train', 'test'], fontsize = 12)

plt.show()
train_df.drop(train_df.columns[11:-1], axis  = 1).groupby('Cover_Type').describe()
plt.figure(figsize = (20, 30))

for pos, key in enumerate(train_df.columns[1:11]):
    ax = plt.subplot(5, 2, pos+1)
    sns.violinplot(y = key, x = 'Cover_Type', data = train_df)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)

plt.show()
train_numerical_feat_df = train_df.iloc[:,1:11]
temp = train_numerical_feat_df - train_numerical_feat_df.mean()
train_numerical_feat_df = temp/temp.std()

train_numerical_feat_df['Cover_Type'] = train_df['Cover_Type']

train_numerical_feat_df.describe()
from sklearn.decomposition import PCA

n_comp = 2
pca = PCA(n_components = n_comp)
pca.fit(train_numerical_feat_df.iloc[:,0:10])

print('Total variance explained by the first %d principal components = %f ' %(n_comp, sum(pca.explained_variance_ratio_)))
from matplotlib.colors import ListedColormap

feat2D = pca.transform(train_numerical_feat_df.iloc[:,0:10])

old_cmap = plt.get_cmap("Set1")
my_cmap = ListedColormap(old_cmap.colors[:7])

#colors =  list(map(lambda x: my_cmap(x-1), train_cont_feat_df['Cover_Type']))
colors = train_numerical_feat_df['Cover_Type']

plt.figure(figsize = (15, 10))
plt.scatter(x = feat2D[:,0], y = feat2D[:,1], c = colors, cmap = my_cmap, vmin = 0.5, vmax = 7.5)
plt.colorbar()
plt.show()
n_comp = 10
pca10 = PCA(n_components = n_comp)
pca10.fit(train_numerical_feat_df.iloc[:,0:10])

print('Explained variance ratio for each component')
print(pca10.explained_variance_ratio_)

x = range(1, len(pca10.explained_variance_ratio_) + 1)
cumulative_ratios = [sum(pca10.explained_variance_ratio_[0:j+1]) for j in range(len(pca10.explained_variance_ratio_))]

plt.figure(figsize = (15, 10))
plt.plot(x, cumulative_ratios)
plt.plot(x, [0.9]*len(x), linestyle='dashed')
plt.plot([6]*8, np.arange(0.3, 1.1, 0.1), linestyle='dashed')
plt.ylabel('Explained variance for the first N components', fontsize= 12)
plt.xticks(x, fontsize= 12)
plt.yticks(fontsize= 12)
plt.show()
print(pca10.components_[0])
print(pca10.components_[1])
ext_train_df = train_df.copy()

def dist_hyd(row):
    return np.sqrt(row['Horizontal_Distance_To_Hydrology']**2 + row['Vertical_Distance_To_Hydrology']**2)

def avg_dist_bad(row):
    return 0.5*(row['Horizontal_Distance_To_Fire_Points'] + row['Horizontal_Distance_To_Roadways'])

def min_dist_bad(row):
    return np.min([row['Horizontal_Distance_To_Fire_Points'], row['Horizontal_Distance_To_Roadways']])

def avg_shade(row):
    return (row['Hillshade_9am'] + row['Hillshade_Noon'] + row['Hillshade_3pm'])/3.0

def min_shade(row):
    return np.min([row['Hillshade_9am'], row['Hillshade_Noon'], row['Hillshade_3pm']])

def max_shade(row):
    return np.max([row['Hillshade_9am'], row['Hillshade_Noon'], row['Hillshade_3pm']])

ext_train_df['Distance_To_Hydrology'] = ext_train_df.apply(lambda x: dist_hyd(x), axis = 1)
ext_train_df['Average_Distance_To_Bad_Points'] = ext_train_df.apply(lambda x: avg_dist_bad(x), axis = 1)
ext_train_df['Min_Distance_To_Bad_Points'] = ext_train_df.apply(lambda x: min_dist_bad(x), axis = 1)
ext_train_df['Average_Shade'] = ext_train_df.apply(lambda x: avg_shade(x), axis = 1)
ext_train_df['Min_Shade'] = ext_train_df.apply(lambda x: min_shade(x), axis = 1)
ext_train_df['Max_Shade'] = ext_train_df.apply(lambda x: max_shade(x), axis = 1)

ext_train_df.head()
plt.figure(figsize = (20, 18))

for pos, key in enumerate(('Distance_To_Hydrology', 'Average_Distance_To_Bad_Points', 'Min_Distance_To_Bad_Points', 'Average_Shade', 'Min_Shade', 'Max_Shade')):
    ax = plt.subplot(3, 2, pos+1)
    sns.violinplot(y = key, x = 'Cover_Type', data = ext_train_df)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    
plt.show()
train_df.loc[:,'Wilderness_Area1':'Soil_Type40'].sum().to_frame().T/len(train_df)
test_df.loc[:,'Wilderness_Area1':'Soil_Type40'].sum().to_frame().T/len(test_df)
train_df.loc[:,'Wilderness_Area1':'Cover_Type'].groupby('Cover_Type').sum()/2160.0
frequencies = train_df.loc[:,'Wilderness_Area1':'Cover_Type'].groupby('Cover_Type').sum()/2160.0
frequencies = frequencies.append(train_df.loc[:,'Wilderness_Area1':'Soil_Type40'].sum().to_frame().T/len(train_df))
frequencies.rename(index={0:'all'}, inplace = True)

frequencies
frequencies.iloc[:,0:4].plot.barh(figsize = (15,10), stacked = True)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.show()
frequencies.iloc[:,4:].plot.barh(figsize = (20,10), stacked = True)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.show()
