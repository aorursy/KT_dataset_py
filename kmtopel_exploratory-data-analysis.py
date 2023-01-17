#Import libraries for data analysis, visualization, and ML model building

import numpy as np

import pandas as pd

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

from scipy import stats

import seaborn as sns

import matplotlib.pyplot as plt

import os

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/learn-together/train.csv')

test = pd.read_csv('/kaggle/input/learn-together/test.csv')

sample_submission = pd.read_csv('/kaggle/input/learn-together/sample_submission.csv')
#List of all non-binary columns

non_binary_cols = ['Elevation', 'Aspect', 'Slope',

       'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',

       'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon',

       'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points','Cover_Type']
#The 'aspect' variable reflects the azimuth measurement in degrees (0-360).

#Since a reading of 180 degrees indicates that the sun is directly overhead,

#readings of 200 and 160 degrees would be equivalent. 

#Therefore, I updated the column for this variable to situate all readings between 0 and 180.

train['aspect_updated']=np.absolute(180-train['Aspect'])

test['aspect_updated']=np.absolute(180-test['Aspect'])

non_binary_cols.append('aspect_updated')
#Create correlation table

correlation_table = train[non_binary_cols].drop('Aspect',axis=1).corr().round(3)

#Diagonally slice table to remove duplicates using np.tril method

data_viz_cols = correlation_table.columns

data_viz_index = correlation_table.index

correlation_table = pd.DataFrame(np.tril(correlation_table),columns=data_viz_cols,index=data_viz_index).replace(0,np.nan).replace(1,np.nan)

#Adjust plot size

a4_dims = (11.7, 8.27)

fig, ax = plt.subplots(figsize=a4_dims)

sns.heatmap(correlation_table, annot=True,cmap='coolwarm')

plt.show()
train['Cover_Type'] = train['Cover_Type'].astype('category')

dummie_cats = pd.get_dummies(train['Cover_Type'])

train_updated = pd.concat([train,dummie_cats],axis=1)

train_updated = train_updated.drop(['Cover_Type','Id'],axis=1)

a4_dims = (10, 30)

fig, ax = plt.subplots(figsize=a4_dims)

sns.heatmap(abs(train_updated.corr()[[1,2,3,4,5,6,7]]).round(3).drop([1,2,3,4,5,6,7],axis=0),cmap='coolwarm',annot=True)

target_correlations = train.corr()['Cover_Type'].abs().round(3).sort_values(ascending=False)

target_corr_series = pd.Series(target_correlations)

target_corrs = target_corr_series[target_corr_series>.1]

corr_cols = target_corrs.index
#For the purposes of better understanding data, I ran the dataset through a K-Means model.

#This type of unsupervised learning model can help see how features

#naturally group into various "clusters." 



#Scale the features between 0 and 1, so no feature disproportionately contributes to distance calculations

features = train.drop(['Id','Cover_Type'],axis=1)

scaler = MinMaxScaler()

fit = scaler.fit(features)

normalized_features = pd.DataFrame(scaler.transform(features),columns=features.columns)



#Create a KMeans model where 

cluster_model = KMeans(n_clusters=7,random_state=1)

cluster_model.fit_transform(normalized_features)

predicted_clusters = cluster_model.labels_

normalized_features['cluster'] = predicted_clusters

normalized_features['cover_type'] = train['Cover_Type']



#Print value counts for each cluster

for i in normalized_features['cluster'].unique():

    df = normalized_features[normalized_features['cluster']==i]

    counts = df['cover_type'].value_counts()

    print('cluster '+str(i))

    print('\n')

    print(counts)
col_names = ['pca_'+str(i+1) for i in range(len(features.columns))]

pca = PCA(n_components=len(features.columns))

pca.fit_transform(features)

pca_df = pd.DataFrame(pca.components_,columns=col_names)

pca_df['cover_type'] = train['Cover_Type']

x_lbls = [str(i+1) for i in range(len(features.columns))]

y = pca.explained_variance_ratio_

plt.figure(figsize=(20,5))

plt.title('Scree Plot of Eigenvalues')

plt.ylabel('% of Variance Explained')

plt.xlabel('Principal Component')

plt.bar(x_lbls,y)

plt.show()
plt.figure(figsize=(5,5))

pca_two_feature = PCA(n_components=2)

pca_two_feature.fit_transform(features)

pca_two_feature_df = pd.DataFrame(pca_two_feature.components_)

pca_two_feature_df



#plt.scatter(x=pca_two_feature_df['pca_1'],y=pca_two_feature_df['pca_2'],c=pca_df['cover_type'],cmap='prism')

#plt.xlabel('pca_1')

#plt.ylabel('pca_2')

#plt.show()