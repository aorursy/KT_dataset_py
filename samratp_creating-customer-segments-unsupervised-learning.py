# Import libraries necessary for this project

import numpy as np

import pandas as pd

from IPython.display import display # Allows the use of display() for DataFrames

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score



import seaborn as sns

import matplotlib.pyplot as plt

# Pretty display for notebooks

%matplotlib inline



# Load the wholesale customers dataset

try:

    data = pd.read_csv("../input/customers.csv")

    data.drop(['Region', 'Channel'], axis = 1, inplace = True)

    print("Wholesale customers dataset has {} samples with {} features each.".format(*data.shape))

except:

    print("Dataset could not be loaded. Is the dataset missing?")
###########################################

# Suppress matplotlib user warnings

# Necessary for newer version of matplotlib

import warnings

warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")

#

# Display inline matplotlib plots with IPython

from IPython import get_ipython

get_ipython().run_line_magic('matplotlib', 'inline')

###########################################



import matplotlib.pyplot as plt

import matplotlib.cm as cm

import pandas as pd

import numpy as np



def pca_results(good_data, pca):

	'''

	Create a DataFrame of the PCA results

	Includes dimension feature weights and explained variance

	Visualizes the PCA results

	'''



	# Dimension indexing

	dimensions = dimensions = ['Dimension {}'.format(i) for i in range(1,len(pca.components_)+1)]



	# PCA components

	components = pd.DataFrame(np.round(pca.components_, 4), columns = list(good_data.keys()))

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





def biplot(good_data, reduced_data, pca):

    '''

    Produce a biplot that shows a scatterplot of the reduced

    data and the projections of the original features.

    

    good_data: original data, before transformation.

               Needs to be a pandas dataframe with valid column names

    reduced_data: the reduced data (the first two dimensions are plotted)

    pca: pca object that contains the components_ attribute



    return: a matplotlib AxesSubplot object (for any additional customization)

    

    This procedure is inspired by the script:

    https://github.com/teddyroland/python-biplot

    '''



    fig, ax = plt.subplots(figsize = (14,8))

    # scatterplot of the reduced data    

    ax.scatter(x=reduced_data.loc[:, 'Dimension 1'], y=reduced_data.loc[:, 'Dimension 2'], 

        facecolors='b', edgecolors='b', s=70, alpha=0.5)

    

    feature_vectors = pca.components_.T



    # we use scaling factors to make the arrows easier to see

    arrow_size, text_pos = 7.0, 8.0,



    # projections of the original features

    for i, v in enumerate(feature_vectors):

        ax.arrow(0, 0, arrow_size*v[0], arrow_size*v[1], 

                  head_width=0.2, head_length=0.2, linewidth=2, color='red')

        ax.text(v[0]*text_pos, v[1]*text_pos, good_data.columns[i], color='black', 

                 ha='center', va='center', fontsize=18)



    ax.set_xlabel("Dimension 1", fontsize=14)

    ax.set_ylabel("Dimension 2", fontsize=14)

    ax.set_title("PC plane with original feature projections.", fontsize=16);

    return ax

    



def channel_results(reduced_data, outliers, pca_samples):

	'''

	Visualizes the PCA-reduced cluster data in two dimensions using the full dataset

	Data is labeled by "Channel" and cues added for student-selected sample data

	'''



	# Check that the dataset is loadable

	try:

	    full_data = pd.read_csv("../input/customers.csv")

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
# Print some sample data

display(data.head())
# Display Data Info

display(data.info())
# Display a description of the dataset

display(data.describe())
# Select three indices of your choice you wish to sample from the dataset

np.random.seed(2018)

indices = np.random.randint(low = 0, high = 441, size = 3)

print("Indices of Samples => {}".format(indices))



# Create a DataFrame of the chosen samples

samples = pd.DataFrame(data.loc[indices], columns = data.keys()).reset_index(drop = True)

print("\nChosen samples of wholesale customers dataset:")

display(samples)
# Function to display the sample data vs the population mean for

# each of the categories

def sampl_pop_plotting(sample):

    fig, ax = plt.subplots(figsize=(10,5))

    

    index = np.arange(sample.count())

    bar_width = 0.3

    opacity_pop = 1

    opacity_sample = 0.3



    rect1 = ax.bar(index, data.mean(), bar_width,

                    alpha=opacity_pop, color='g',

                    label='Population Mean')

    

    rect2 = ax.bar(index + bar_width, sample, bar_width,

                    alpha=opacity_sample, color='k',

                    label='Sample')

    

    ax.set_xlabel('Categories')

    ax.set_ylabel('Total Purchase Cost')

    ax.set_title('Sample vs Population Mean')

    ax.set_xticks(index + bar_width / 2)

    ax.set_xticklabels(samples.columns)

    ax.legend(loc=0, prop={'size': 15})

    

    fig.tight_layout()

    plt.show()
# Display data for the first sample wrt to the population mean

display(samples.iloc[0] - data.mean())
# Plot data for the first sample wrt to the population mean

sampl_pop_plotting(samples.iloc[0])
# Display data for the second sample wrt to the population mean

display(samples.iloc[1] - data.mean())
# Plot data for the second sample wrt to the population mean

sampl_pop_plotting(samples.iloc[1])
# Display data for the third sample wrt to the population mean

display(samples.iloc[2] - data.mean())
# Plot data for the third sample wrt to the population mean

sampl_pop_plotting(samples.iloc[2])
# percentile heatmap for sample points

percentiles_data = 100*data.rank(pct=True)

percentiles_samples = percentiles_data.iloc[indices]

plt.subplots(figsize=(10,5))

_ = sns.heatmap(percentiles_samples, annot=True)
def predict_one_feature(dropped_feature):

    # Make a copy of the DataFrame, using the 'drop' function to drop the given feature

    print("Dropping feature -> {}".format(dropped_feature))

    new_data = data.drop([dropped_feature], axis = 1)

    

    # Split the data into training and testing sets(0.25) using the given feature as the target

    # Set a random state.

    X_train, X_test, y_train, y_test = train_test_split(new_data, data[dropped_feature], test_size=0.25, random_state=0)

    

    # Create a decision tree regressor and fit it to the training set

    regressor = DecisionTreeRegressor(random_state=0)

    regressor.fit(X_train, y_train)

    

    # Report the score of the prediction using the testing set

    score = regressor.score(X_test, y_test)

    print("Score for predicting '{}' using other features = {:.3f}\n".format(dropped_feature, score))
# Attempt to predict the score of 'Milk' using other features

predict_one_feature('Milk')
print("Features in data -> {}\n".format(data.columns.values))



# Predict the score of each feature by dropping it and using other features

for cols in data.columns.values:

    predict_one_feature(cols)
# Display the correlation heatmap

corr = data.corr()



plt.figure(figsize = (10,5))

ax = sns.heatmap(corr, annot=True)

ax.legend(loc=0, prop={'size': 15})
# Produce a scatter matrix for each pair of features in the data

_ = sns.pairplot(data, diag_kind = 'kde')
plt.figure(figsize = (20,8))

_ = sns.barplot(data=data, palette="Set2")
plt.figure(figsize = (20,8))

_ = sns.boxplot(data=data, orient='h', palette="Set2")
plt.figure(figsize = (20,8))



for cols in data.columns.values:

    ax = sns.kdeplot(data[cols])

    ax.legend(loc=0, prop={'size': 15})
# Scale the data using the natural logarithm

log_data = np.log(data)



# Scale the sample data using the natural logarithm

log_samples = np.log(samples)



# Produce a scatter matrix for each pair of newly-transformed features

_ = sns.pairplot(log_data, diag_kind = 'kde')
# Display the log-transformed sample data

display(log_samples)
# Display the correlation heatmap

log_corr = log_data.corr()



f = plt.figure(figsize = (16,8))

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = True

with sns.axes_style("white"):

    ax1 = sns.heatmap(corr, annot=True, mask=mask, cbar_kws={'label': 'Before Log Normalization'})



mask2 = np.zeros_like(corr)

mask2[np.tril_indices_from(mask2)] = True

with sns.axes_style("white"):

    ax2 = sns.heatmap(log_corr, annot=True, mask=mask2, cmap="YlGnBu", cbar_kws={'label': 'After Log Normalization'})
# boxplot on the logdata

plt.figure(figsize = (16,8))

_ = sns.boxplot(data=log_data, palette="Set2")
outliers_list = []

# For each feature find the data points with extreme high or low values

for feature in log_data.keys():

    

    # Calculate Q1 (25th percentile of the data) for the given feature

    Q1 = np.percentile(log_data[feature], 25)

    

    # Calculate Q3 (75th percentile of the data) for the given feature

    Q3 = np.percentile(log_data[feature], 75)

    

    # Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)

    step = (Q3 - Q1) * 1.5

    

    # Display the outliers

    print("Data points considered outliers for the feature '{}':".format(feature))

    outliers = list(log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))].index.values)

    display(log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))])

    outliers_list.extend(outliers)

    

print("List of Outliers -> {}".format(outliers_list))

duplicate_outliers_list = list(set([x for x in outliers_list if outliers_list.count(x) >= 2]))

duplicate_outliers_list.sort()

print("\nList of Common Outliers -> {}".format(duplicate_outliers_list))



# Select the indices for data points you wish to remove

outliers  = duplicate_outliers_list



# Remove the outliers, if any were specified

good_data = log_data.drop(log_data.index[outliers]).reset_index(drop = True)
# Apply PCA by fitting the good data with the same number of dimensions as features

pca = PCA(n_components = 6, random_state=0)

pca.fit(good_data)



# Transform log_samples using the PCA fit above

pca_samples = pca.transform(log_samples)

print("Explained Variance Ratio => {}\n".format(pca.explained_variance_ratio_))

print("Explained Variance Ratio(csum) => {}\n".format(pca.explained_variance_ratio_.cumsum()))



# Generate PCA results plot

pca_results = pca_results(good_data, pca)
# Display sample log-data after having a PCA transformation applied

display(pd.DataFrame(np.round(pca_samples, 4), columns = pca_results.index.values))
# Apply PCA by fitting the good data with only two dimensions

pca = PCA(n_components = 2, random_state=0)

pca.fit(good_data)



# Transform the good data using the PCA fit above

reduced_data = pca.transform(good_data)



# Transform log_samples using the PCA fit above

pca_samples = pca.transform(log_samples)



# Create a DataFrame for the reduced data

reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])
# Display sample log-data after applying PCA transformation in two dimensions

display(pd.DataFrame(np.round(pca_samples, 4), columns = ['Dimension 1', 'Dimension 2']))
# Create a biplot

biplot(good_data, reduced_data, pca)
def sil_coeff(no_clusters):

    # Apply your clustering algorithm of choice to the reduced data 

    clusterer_1 = KMeans(n_clusters=no_clusters, random_state=0 )

    clusterer_1.fit(reduced_data)

    

    # Predict the cluster for each data point

    preds_1 = clusterer_1.predict(reduced_data)

    

    # Find the cluster centers

    centers_1 = clusterer_1.cluster_centers_

    

    # Predict the cluster for each transformed sample data point

    sample_preds_1 = clusterer_1.predict(pca_samples)

    

    # Calculate the mean silhouette coefficient for the number of clusters chosen

    score = silhouette_score(reduced_data, preds_1)

    

    print("silhouette coefficient for `{}` clusters => {:.4f}".format(no_clusters, score))

    

clusters_range = range(2,15)

for i in clusters_range:

    sil_coeff(i)
# Display the results of the clustering from implementation for 2 clusters

clusterer = KMeans(n_clusters = 2)

clusterer.fit(reduced_data)

preds = clusterer.predict(reduced_data)

centers = clusterer.cluster_centers_

sample_preds = clusterer.predict(pca_samples)



cluster_results(reduced_data, preds, centers, pca_samples)
# Inverse transform the centers

log_centers = pca.inverse_transform(centers)



# Exponentiate the centers

true_centers = np.exp(log_centers)



# Display the true centers

segments = ['Segment {}'.format(i) for i in range(0,len(centers))]

true_centers = pd.DataFrame(np.round(true_centers), columns = data.keys())

true_centers.index = segments

display(true_centers)
display(data.mean(axis=0))
display(samples)
# Display the predictions

for i, pred in enumerate(sample_preds):

    print("Sample point", i, "predicted to be in Cluster", pred)
# Display the clustering results based on 'Channel' data

channel_results(reduced_data, outliers, pca_samples)