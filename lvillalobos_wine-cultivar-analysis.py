import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tools.plotting import parallel_coordinates
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt
from itertools import cycle, islice
%matplotlib inline
## Reading Dataset
wine_df = pd.read_csv('../input/Wine.csv', sep=',',names=["Cultivar","Alcohol","Malic","Ash","Alkalinity","Magnesium","Phenols","Flavanoids","Nonflav","Proanthocyan","Color","Hue","OD280","Proline"])
print(type(wine_df))
# Finds number of rows and columns in dataframe
rows, cols = wine_df.shape
print("Dataframe has", rows, "records and", cols, "variables.\n")
wine_df.head()
attributes = wine_df.columns
features = attributes[1:]
classification = attributes[0]
print("The attributes in the dataframe are:\n", attributes)

# Are there any records with NaN data?
NaN_data_flag = wine_df.isnull().any()
if NaN_data_flag.any():
    print("Some records have NaN values. These will be removed...\n")
    before_rows, before_cols = wine_df.shape
    wine_df = wine_df.dropna()
    after_rows, after_cols = wine_df.shape
    print("Dropped", after_rows - before_rows, "records. Cleaned dataframe has", after_rows, "records.\n")
else:
    print("There are no records with NaN values. Dataframe is already clean.\n")
    
wine_df.describe().round(3)
# Samples per Cultivar
region_counts = wine_df['Cultivar'].value_counts()
explode = (0, 0.1, 0)
region_counts.plot(kind='pie',autopct='%.0f%%', shadow=True, figsize=(4,4), radius=1.0)

hist_quality = wine_df['Alcohol']
plt.hist(hist_quality, 10, normed=False, facecolor='teal')
plt.xlabel('Alcohol')
plt.ylabel('Count')
plt.title('Alcohol Content Distribution')
plt.grid(True)
plt.show()
hist_fixed = wine_df['Phenols']
plt.hist(hist_fixed, 10, normed=False, facecolor='blue')
plt.xlabel('Phenols')
plt.ylabel('Count')
plt.title('Phenols Distribution')
plt.grid(True)
plt.show()

hist_volatile = wine_df['Color']
plt.hist(hist_volatile, 10, normed=False, facecolor='red')
plt.xlabel('Color')
plt.ylabel('Count')
plt.title('Color Distribution')
plt.grid(True)
plt.show()
hist_citric = wine_df['Nonflav']
plt.hist(hist_citric, 10, normed=False, facecolor='lime')
plt.xlabel('Nonflavanoids')
plt.ylabel('Count')
plt.title('Nonflavanoids Distribution')
plt.grid(True)
plt.show()
classification_data = wine_df.copy()
label_mapping = {1:'Cultivar1', 2:'Cultivar2', 3:'Cultivar3'}
classification_data['Cultivar_Label'] = classification_data['Cultivar'].map(label_mapping)
classification_data.head()
# Target is stored in Y
Y = classification_data[['Cultivar_Label']].copy()

# Training features are stored in X
features = ['Alcohol', 'Malic', 'Ash', 'Alkalinity', 'Magnesium', 'Phenols', 
            'Flavanoids', 'Nonflav', 'Proanthocyan', 'Color', 'Hue', 'OD280', 'Proline']
X = classification_data[features].copy()

# Split data into test and training
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.4, random_state = 42)

# Train decision tree
quality_classifier = DecisionTreeClassifier(max_leaf_nodes = 15, random_state = 42)
quality_classifier.fit(X_train, Y_train)

# Testing
predictions = quality_classifier.predict(X_test)
predictions[:10]
# Percentage of target data by label
percentages = Y_test['Cultivar_Label'].value_counts(normalize=True)
percentages
# Measure accuracy
accuracy_score(y_true = Y_test, y_pred = predictions)
features = ['Alcohol', 'Malic', 'Ash', 'Alkalinity', 'Magnesium', 'Phenols', 
            'Flavanoids', 'Nonflav', 'Proanthocyan', 'Color', 'Hue', 'OD280', 'Proline']
X_cluster = StandardScaler().fit_transform(wine_df[features])
# Finds best number of clusters using Inertia_ metric
SSE_data =[]
for n in range(3, 20):
    # Perform the clustering
    kmeans = KMeans(n_clusters = n)
    model = kmeans.fit(X_cluster)
    SSE_data.append(model.inertia_)
    
# Plot the SSE values to find the elbow that gives the best number of clusters
SSE_series = pd.Series(SSE_data)
x = np.arange(3., 20., 1.0)
plt.scatter(x, SSE_series, c="b", marker='o', label="SSE vs. n_clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.legend(loc=1)
plt.ylim(ymin=1)
plt.grid()
plt.show()
# Clustering
kmeans = KMeans(n_clusters = 10)
model = kmeans.fit(X_cluster)
print("Model\n", model)

# Dispay centroids
centers = model.cluster_centers_
centers.round(2)
# Determines cluster for each sample
predictedCluster = kmeans.predict(X_cluster)
predictedCluster
#Plots
# Function that creates a DataFrame with a column for Cluster Number

def pd_centers(featuresUsed, centers):
	colNames = list(featuresUsed)
	colNames.append('prediction')

	# Zip with a column called 'prediction' (index)
	Z = [np.append(A, index) for index, A in enumerate(centers)]

	# Convert to pandas data frame for plotting
	P = pd.DataFrame(Z, columns=colNames)
	P['prediction'] = P['prediction'].astype(int)
	return P

# Function that creates Parallel Plots

def parallel_plot(data):
    my_colors = list(islice(cycle(['b', 'r', 'g', 'y', 'k', 'c', 'm', 'lime', 'salmon', 'grey']), None, len(data)))
    fig = plt.figure(figsize=(15,8)).gca().axes.set_ylim([-2,+5])
    parallel_coordinates(data, 'prediction', color = my_colors, marker='o')
    plt.savefig('FeaturePlot.png')
    
P = pd_centers(features, centers)
parallel_plot(P)

# Cultivar Representation in Clusters
cultivar1Clusters = predictedCluster[wine_df['Cultivar'] == 1]
cultivar2Clusters = predictedCluster[wine_df['Cultivar'] == 2]
cultivar3Clusters = predictedCluster[wine_df['Cultivar'] == 3]

# Centroid counts in each cultivar
totClusters = 10
cultivar1Counts = []
cultivar2Counts = []
cultivar3Counts = []
for i in range(totClusters):
    cultivar1Counts.append(np.count_nonzero(cultivar1Clusters == i))
    cultivar2Counts.append(np.count_nonzero(cultivar2Clusters == i))
    cultivar3Counts.append(np.count_nonzero(cultivar3Clusters == i))
    
# Plot the distribution of cultivar samples per centroid
ind = np.arange(totClusters)    
width = 0.45                    
cult1Pluscult2 = [sum(x) for x in zip(cultivar1Counts, cultivar2Counts)]

p1 = plt.bar(ind, cultivar1Counts, width, color='b')
p2 = plt.bar(ind, cultivar2Counts, width, bottom = cultivar1Counts, color='r')
p3 = plt.bar(ind, cultivar3Counts, width, bottom = cult1Pluscult2, color='lime')

plt.ylabel('Cultivar counts')
plt.xlabel('Centroids')
plt.title('Cultivar counts per centroid')
plt.xticks(ind, ('C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10'))
plt.yticks(np.arange(0, 51, 10))
plt.legend((p1[0], p2[0], p3[0]), ('Cultivar1', 'Cultivar2', 'Cultivar3'))
plt.show()

wine_df['Cultivar1'] = wine_df['Cultivar'] == 1
wine_df['Cultivar2'] = wine_df['Cultivar'] == 2
wine_df['Cultivar3'] = wine_df['Cultivar'] == 3
corrVariables = ['Cultivar1', 'Cultivar2', 'Cultivar3', 'Alcohol', 'Magnesium',
                'Phenols', 'Flavanoids', 'Nonflav', 'Ash', 'Alkalinity', 'Color', 'Hue', 
                 'Proanthocyan', 'OD280', 'Malic', 'Proline']
cultivars_df = wine_df[corrVariables]


# Correlation
corr = cultivars_df.corr().round(2)
corr = corr[['Cultivar1', 'Cultivar2', 'Cultivar3']]
cultivarCorr = corr.drop(['Cultivar1', 'Cultivar2', 'Cultivar3'], axis = 0)
cultivarCorr
            
# Heatmap of the correlation matrix above
heatmapRows = ['Cultivar1', 'Cultivar2', 'Cultivar3']
heatmapCols = ['Alcohol', 'Magnesium', 'Phenols', 'Flavanoids', 'Nonflav', 'Ash', 
               'Alkalinity', 'Color', 'Hue', 'Proanthocyan', 'OD280', 'Malic', 'Proline']

fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(cultivarCorr, interpolation='nearest')

# We want to show all ticks...
ax.set_xticks(np.arange(len(heatmapRows)))
ax.set_yticks(np.arange(len(heatmapCols)))
# ... and label them with the respective list entries
ax.set_xticklabels(heatmapRows)
ax.set_yticklabels(heatmapCols)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(heatmapCols)):
    for j in range(len(heatmapRows)):
        text = ax.text(j, i, cultivarCorr.iloc[i][j], ha="center", va="center", color="w")

ax.set_title("Heatmap between Cultivars and Chemical Attributes")
fig.tight_layout()
plt.show()


