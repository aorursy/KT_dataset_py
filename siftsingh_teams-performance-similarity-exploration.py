%matplotlib inline
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from IPython.display import HTML, display
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
data = pd.read_csv("../input/FIFA 2018 Statistics.csv")
data.head()
data.columns
data.dtypes
# checking how many missing values are present in each of the columns
missing = data.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar()
# add features to quant list if they are int or float type
quant = [f for f in data.columns if data.dtypes[f]!='object']
# add features to qualitative list if they are object type
qualitative = [f for f in data.columns if data.dtypes[f] == 'object']
quant
quant.remove('Own goal Time')
quant.remove('Own goals')
quant.remove('Goals in PSO')
quant.remove('1st Goal')
quant
qualitative.remove('Date')
qualitative
def encode(frame, feature):
    ordering = pd.DataFrame()
    # extracting unique values from a feature(column)
    ordering['val'] = frame[feature].unique()
    # assigning the unique values to the index of the dataframe
    ordering.index = ordering.val
    # creating a column ordering with values assinged from 1 to the number of unique values
    ordering['ordering'] = range(1, ordering.shape[0]+1)
    # creating a dict with the unique values as keys and the corresponding 
    # numbers in the ordering column as values
    ordering = ordering['ordering'].to_dict()
    # adding the encoded values into the original dataframe within new columns for each feature 
    for cat, o in ordering.items():
        frame.loc[frame[feature] == cat, feature+'_E'] = o
    
qual_encoded = []
# encoding all the features in the qualitative list
for q in qualitative:  
    encode(data, q)
    qual_encoded.append(q+'_E')
qual_encoded
# data
qual_encoded.remove('Opponent_E')
# # feature importance
# print("Features Importance...")
# gain = model.feature_importance('gain')
# featureimp = pd.DataFrame({'feature':model.feature_name(), 
#                    'split':model.feature_importance('split'), 
#                    'gain':100 * gain / gain.sum()}).sort_values('gain', ascending=False)
# print(featureimp[:10])

def feat_correlation(frame, features):
    corr = pd.DataFrame()
    corr['feature'] = features
    corr['teams'] = [frame[f].corr(frame['Team_E'], 'spearman') for f in features]
    corr = corr.sort_values('teams')
    plt.figure(figsize=(6, 0.25*len(features)))
    sns.barplot(data=corr, y='feature', x='teams', orient='h')
    
features = quant + qual_encoded
feat_correlation(data, features)
# labels = pd.DataFrame()
# x = kmeans.labels_
# labels = pd.concat([labels, pd.Series(x)])
# labels = pd.concat([labels,data['Team']],axis=1)
# labels
# plotting the heatmap for all the quantitative features to determine which features are most relevant
plt.figure(figsize=(30,10))
sns.heatmap(data[quant].corr(), square=True, annot=True,robust=True, yticklabels=1)
# plotting the heatmap for all the quantitative and qualitative features to determine which features are most relevant
plt.figure(figsize=(30,11))
sns.heatmap(data[quant+qual_encoded].corr(), square=True, annot=True,robust=True, yticklabels=1)
# collect all the qualitative and quantitative features into one list
features = quant + qual_encoded
# Elbow criterion to determine optimal number of clusters
def elbow_plot(data, maxK=10, seed_centroids=None):
    """
        parameters:
        - data: pandas DataFrame (data to be fitted)
        - maxK (default = 10): integer (maximum number of clusters with which to run k-means)
        - seed_centroids (default = None ): float (initial value of centroids for k-means)
    """
    sse = {}
    for k in range(1, maxK):
        print("k: ", k)
        if seed_centroids is not None:
            seeds = seed_centroids.head(k)
            kmeans = KMeans(n_clusters=k, max_iter=500, n_init=100, random_state=0, init=np.reshape(seeds, (k,1))).fit(data)
            data["clusters"] = kmeans.labels_
        else:
            kmeans = KMeans(n_clusters=k, max_iter=300, n_init=100, random_state=0).fit(data)
            data["clusters"] = kmeans.labels_
        # Inertia: Sum of distances of samples to their closest cluster center
        sse[k] = kmeans.inertia_
    plt.figure()
    plt.plot(list(sse.keys()), list(sse.values()))
    plt.show()
    return

elbow_plot(data[features], maxK=10)
# reducing the dimensionality of the data to 2- dimensions using t-SNE
model = TSNE(n_components=2, random_state=0, perplexity=50)
# fill all the na values with zeros and pass all the values to X
X = data[features].fillna(0.).values
# run t-SNE on the data
tsne = model.fit_transform(X)

# standardising the data to values between 0 to 1
std = StandardScaler()
s = std.fit_transform(X)

# changing the data to 5 dimensions and fitting it into the model
pca = PCA(n_components=5)
pca.fit(s)
pc = pca.transform(s)

# fitting the data into clusters using K-means
kmeans = KMeans(n_clusters=3)
kmeans.fit(pc)

# plotting the data points onto a figure
plt.figure(figsize=(30,30))
# fr = pd.DataFrame({'tsne1': tsne[:,0], 'tsne2': tsne[:, 1], 'cluster': kmeans.labels_})
fr = pd.DataFrame({'pca1': pc[:,0], 'pca2': pc[:, 1], 'cluster': kmeans.labels_, 'label': data['Team']})
p1 = sns.regplot(data=fr, x='pca1', y='pca2', fit_reg=False)
# add annotations one by one with a loop
for line in range(0,fr.shape[0]):
    #uncomment this line if you wish to see points for a specific country
#     if fr.label[line]=='France':
    p1.text(fr.pca1[line]+0.2, fr.pca2[line], fr.label[line], horizontalalignment='left', size='large', color='black', weight='semibold')
print(np.sum(pca.explained_variance_ratio_)) 
# lmplot used in case the clusters need to be differentiated from each other based on colour
p1 = sns.lmplot(data=fr, x='pca1', y='pca2', fit_reg=False, hue='cluster')
kmeans.labels_
kmeans.cluster_centers_
plt.plot(pc[:, 0], pc[:, 1], 'k.', markersize=2)
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='r', zorder=10)
plt.title('K-means clustering')
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
# plt.xticks(())
# plt.yticks(())
plt.show()
