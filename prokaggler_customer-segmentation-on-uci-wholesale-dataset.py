# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import seaborn as sns
raw_df = pd.read_csv('/kaggle/input/uci-wholesale-customers-data/Wholesale customers data.csv')
raw_df.head(5)
raw_df.drop(['Channel','Region'],axis=1,inplace=True)

raw_df.shape
raw_df.describe()
raw_df.loc[[100,200,300],:]
raw_df.columns
# fresh filter

fresh_q1 = 3127.75000

raw_df[raw_df['Fresh'] < fresh_q1].head()
# Frozen filter

frozen_q1 = 742.250000

raw_df[raw_df.Frozen < frozen_q1].head()
# frozen q3

frozen_q3 = 3554.250000

raw_df[raw_df.Frozen > frozen_q3].head(7)
selected_samples = [43,12,39]



samples = pd.DataFrame(raw_df.loc[selected_samples],columns=raw_df.columns).reset_index(drop = True)

samples
mean_data = raw_df.describe().loc['mean',:]



sample_bars = samples.append(mean_data)



sample_bars.index = selected_samples + ['mean']



sample_bars.plot(kind='bar',figsize=(15,8))
percentiles = raw_df.rank(pct=True)



percentiles = 100 * percentiles.round(decimals=3)



percentiles = percentiles.iloc[selected_samples]



sns.heatmap(percentiles,vmin=1,vmax=99,annot=True)
raw_df.columns
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor
deps_vars = list(raw_df.columns)



for var in deps_vars:

    new_data = raw_df.drop([var],axis=1)

    

    new_feature = pd.DataFrame(raw_df.loc[:,var])

    

    X_train, X_test, y_train, y_test = train_test_split(new_data, new_feature, test_size=0.25, random_state=42)

    

    dtr = DecisionTreeRegressor(random_state=42)

    

    dtr.fit(X_train,y_train)

    

    score = dtr.score(X_test, y_test)

    

    print('R2 score for {} as dependent variable: {}'.format(var, score))
sns.pairplot(data=raw_df,size=5)
import matplotlib.pyplot as plt

%matplotlib inline
def plot_correlation(df,size=10):

    corr = raw_df.corr()

    fig, ax = plt.subplots(figsize=(size,size))

    cax = ax.matshow(df,interpolation='nearest')

    ax.matshow(corr)

    fig.colorbar(cax)

    plt.xticks(range(len(corr.columns)),corr.columns)

    plt.yticks(range(len(corr.columns)),corr.columns)

    

plot_correlation(raw_df)
log_data = np.log(raw_df)

log_sample = np.log(samples)

sns.pairplot(log_data)
log_sample
plot_correlation(log_data)

plot_correlation(log_sample)
np.percentile(raw_df.Milk,25)
import itertools

outlier_list = []



for feature in log_data.columns:

    Q1 = np.percentile(log_data[feature],25)

    Q3 = np.percentile(log_data[feature],75)

    step = 1.5 * (Q3 - Q1)

    print("Data points considered outliers for the feature '{}':".format(feature))

    outlier_rows = log_data.loc[~((log_data[feature] >= Q1- step) & (log_data[feature] <= Q3 + step)),:]

    outlier_list.append(list(outlier_rows.index))

outliers = list(itertools.chain.from_iterable(outlier_list))

uniq_outliers = list(set(outliers))

dup_outliers = list(set([x for x in outliers if outliers.count(x) > 1]))



print('Outliers list:\n', uniq_outliers)

print('Length of outliers list:\n', len(uniq_outliers))

print('Duplicate list:\n', dup_outliers)

print('Length of duplicates list:\n', len(dup_outliers))



good_data = log_data.drop(log_data.index[dup_outliers]).reset_index(drop=True)



print('Original shape of data:\n', raw_df.shape)

print('New shape of data:\n', good_data.shape)
import matplotlib.cm as cm

from sklearn.decomposition import pca



def pca_results(good_data, pca):

	'''

	Create a DataFrame of the PCA results

	Includes dimension feature weights and explained variance

	Visualizes the PCA results

	'''



	# Dimension indexing

	dimensions = dimensions = ['Dimension {}'.format(i) for i in range(1,len(pca.components_)+1)]



	# PCA components

	components = pd.DataFrame(np.round(pca.components_, 4), columns = good_data.keys())

	components.index = dimensions



	# PCA explained variance

	ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)

	variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Variance'])

	variance_ratios.index = dimensions



	# Create a bar plot visualization

	fig, ax = plt.subplots(figsize = (14,8))



	# Plot the feature weights as a function of the components

	components.plot(ax = ax, kind = 'bar');

	ax.set_ylabel("Feature Weights")

	ax.set_xticklabels(dimensions, rotation=0)





	# Display the explained variance ratios

	for i, ev in enumerate(pca.explained_variance_ratio_):

		ax.text(i-0.40, ax.get_ylim()[1] + 0.05, "Explained Variance\n          %.4f"%(ev))



	# Return a concatenated DataFrame

	return pd.concat([variance_ratios, components], axis = 1)



def cluster_results(reduced_data, preds, centers, pca_samples):

	'''

	Visualizes the PCA-reduced cluster data in two dimensions

	Adds cues for cluster centers and student-selected sample data

	'''



	predictions = pd.DataFrame(preds, columns = ['Cluster'])

	plot_data = pd.concat([predictions, reduced_data], axis = 1)



	# Generate the cluster plot

	fig, ax = plt.subplots(figsize = (14,8))



	# Color map

	cmap = cm.get_cmap('gist_rainbow')



	# Color the points based on assigned cluster

	for i, cluster in plot_data.groupby('Cluster'):   

	    cluster.plot(ax = ax, kind = 'scatter', x = 'Dimension 1', y = 'Dimension 2', \

	                 color = cmap((i)*1.0/(len(centers)-1)), label = 'Cluster %i'%(i), s=30);



	# Plot centers with indicators

	for i, c in enumerate(centers):

	    ax.scatter(x = c[0], y = c[1], color = 'white', edgecolors = 'black', \

	               alpha = 1, linewidth = 2, marker = 'o', s=200);

	    ax.scatter(x = c[0], y = c[1], marker='$%d$'%(i), alpha = 1, s=100);



	# Plot transformed sample points 

	ax.scatter(x = pca_samples[:,0], y = pca_samples[:,1], \

	           s = 150, linewidth = 4, color = 'black', marker = 'x');



	# Set plot title

	ax.set_title("Cluster Learning on PCA-Reduced Data - Centroids Marked by Number\nTransformed Sample Data Marked by Black Cross");





def channel_results(reduced_data, outliers, pca_samples):

	'''

	Visualizes the PCA-reduced cluster data in two dimensions using the full dataset

	Data is labeled by "Channel" and cues added for student-selected sample data

	'''



	# Check that the dataset is loadable

	try:

	    full_data = pd.read_csv("customers.csv")

	except:

	    print("Dataset could not be loaded. Is the file missing?")

	    return False



	# Create the Channel DataFrame

	channel = pd.DataFrame(full_data['Channel'], columns = ['Channel'])

	channel = channel.drop(channel.index[outliers]).reset_index(drop = True)

	labeled = pd.concat([reduced_data, channel], axis = 1)

	

	# Generate the cluster plot

	fig, ax = plt.subplots(figsize = (14,8))



	# Color map

	cmap = cm.get_cmap('gist_rainbow')



	# Color the points based on assigned Channel

	labels = ['Hotel/Restaurant/Cafe', 'Retailer']

	grouped = labeled.groupby('Channel')

	for i, channel in grouped:   

	    channel.plot(ax = ax, kind = 'scatter', x = 'Dimension 1', y = 'Dimension 2', \

	                 color = cmap((i-1)*1.0/2), label = labels[i-1], s=30);

	    

	# Plot transformed sample points   

	for i, sample in enumerate(pca_samples):

		ax.scatter(x = sample[0], y = sample[1], \

	           s = 200, linewidth = 3, color = 'black', marker = 'o', facecolors = 'none');

		ax.scatter(x = sample[0]+0.25, y = sample[1]+0.3, marker='$%d$'%(i), alpha = 1, s=125);



	# Set plot title

	ax.set_title("PCA-Reduced Data Labeled by 'Channel'\nTransformed Sample Data Circled");
#!pip install renders

#import renders as rs

from sklearn.decomposition import PCA

pca = PCA(n_components=6)

pca.fit(good_data)

pca_samples = pca.transform(good_data)

pca_results = pca_results(good_data, pca)



pca_results
type(pca_results)
pca_results['Explained Variance'].cumsum()
pca = PCA(n_components=2)

pca.fit(good_data)

reduced_data = pca.transform(good_data)

pca_samples = pca.transform(log_sample)

reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])
from sklearn.mixture import GaussianMixture as GMM

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score
range_n_clusters = list(range(2,11))

print(range_n_clusters)
for n_clusters in range_n_clusters:

    clusterer = GMM(n_components=n_clusters).fit(reduced_data)

    preds = clusterer.predict(reduced_data)

    centers = clusterer.means_

    sample_preds = clusterer.predict(pca_samples)

    score = silhouette_score(reduced_data, preds, metric='mahalanobis')

    print("For n_clusters = {}. The average silhouette_score is : {}".format(n_clusters, score))
lowest_bic = np.infty

bic = []

n_components_range = range(1, 7)

cv_types = ['spherical', 'tied', 'diag', 'full']

for cv_type in cv_types:

    for n_components in n_components_range:

        # Fit a mixture of Gaussians with EM

        gmm = GMM(n_components=n_components, covariance_type=cv_type)

        gmm.fit(reduced_data)

        bic.append(gmm.bic(reduced_data))

        if bic[-1] < lowest_bic:

            lowest_bic = bic[-1]

            best_gmm = gmm
for n_clusters in range_n_clusters:

    clusterer = KMeans(n_clusters=n_clusters).fit(reduced_data)

    preds = clusterer.predict(reduced_data)

    centers = clusterer.cluster_centers_

    sample_preds = clusterer.predict(pca_samples)

    score = silhouette_score(reduced_data, preds, metric='euclidean')

    print("For n_clusters = {}. The average silhouette_score is : {}".format(n_clusters, score))
cluster_results(reduced_data, preds, centers, pca_samples)
log_centers = pca.inverse_transform(centers)

true_centers = np.exp(log_centers)

segments = ['Segment {}'.format(i) for i in range(0,len(centers))]

true_centers = pd.DataFrame(np.round(true_centers), columns = raw_df.columns)

true_centers.index = segments

true_centers
true_centers - raw_df.median()
for i, pred in enumerate(sample_preds):

    print("Sample point", i, "predicted to be in Cluster", pred)
samples
dup_outliers
channel_results(reduced_data, dup_outliers, pca_samples)