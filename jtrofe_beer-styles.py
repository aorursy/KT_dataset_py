import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

recipes = pd.read_csv('../input/recipeData.csv', index_col=0, delimiter=",", encoding='latin1')

# See what we got here
recipes.head()
data = recipes[['OG','FG','ABV','IBU','Color']]
# Calculate the ABV from the OG and FG and compare it to what the user entered
# Remove recipes where the difference is large, on the assumption they could have entered other things wrong too
data = data.assign(calculatedABV=(data.OG - data.FG) * 131.25)
data = data.assign(abvDiff=np.abs(data.ABV - data.calculatedABV))
print('Removing %d beers where ABV didn\'t match calculated ABV' % data.abvDiff[data.abvDiff >= 1].count())
data = data[data.abvDiff < 0.5]

#Get rid of beer with an ABV less than 2%
print('Removing %d beers with low ABV\n' % data.ABV[data.ABV < 2].count())
data = data[data.ABV >= 2]
data = data.drop(columns=['calculatedABV', 'abvDiff'])

# The Brewer's Friend website and forums suggest that an OG of 1.07 is considered high
# See what the OG of the beer with the highest ABV is
max_abv = data.ABV.max()
max_beer = data[data.ABV == max_abv]
print('Beer with highest ABV:\n')
print(data[data.ABV == max_abv])

# See if there are any beers with an OG higher than that one's and remove them
max_og = max_beer.OG.values[0]
cnt = data.OG[data.OG > max_og].count()
print('\n%d rows with unusually high OG' % cnt)
if cnt > 0:
    data = data[data.OG <= max_og]
# Shuffle the data
data = data.sample(frac=1)

# Training/CV/Test split will be 70/15/15
cnt = data.shape[0]
index1 = int(np.ceil(cnt * 0.7))
index2 = int(np.ceil(cnt * 0.85))
data_train = data.iloc[1:index1]
data_cv = data.iloc[index1+1:index2]
data_test = data.iloc[index2+1:]
print('Sizes:\nTraining set - %d\nCV set       - %d\nTest set     - %d' % (data_train.shape[0], data_cv.shape[0], data_test.shape[0]))
mu = data_train.mean()
s = data_train.std()

def normalizeData(d):
    return (d - mu) / s

train_norm = normalizeData(data_train)
cv_norm = normalizeData(data_cv)
test_norm = normalizeData(data_test)

from sklearn.cluster import KMeans

# Calculate distortion the old fashioned way, unoptimized
def calculate_distortion(model, X):
    centroids = model.cluster_centers_
    indices = model.predict(X)
    m = X.shape[0]
    
    J = 0
    for i in range(0,m):
        c = centroids[indices[i]]
        J += (X.iloc[i] - c).sum() ** 2
    return J/m

# There are 176 unique styles in the original data
cluster_counts = np.arange(10,180,10)
models = [None] * len(cluster_counts)
costs = [None] * len(cluster_counts)

# This is a long process because I don't know much about optimization
for i in np.arange(0,len(cluster_counts)):
    models[i] = KMeans(n_clusters = cluster_counts[i]).fit(train_norm)
    costs[i] = calculate_distortion(models[i], cv_norm)
#Graph the data
plt.figure(figsize=(10,6))
plt.plot(cluster_counts, costs, 'o')
plt.xlabel('Clusters ( K )')
plt.ylabel('Distortion ( J )')
plt.show()
# 80 is at index 7
selected_count = 7
K = cluster_counts[selected_count]
model = models[selected_count]

# Now we can see the distortion on the test set
test_distortion = calculate_distortion(model, test_norm)
print('Distortion on test set with %d clusters: %f' % (K, test_distortion))
def append_style(m, X):
    indices = m.predict(X)
    return X.assign(clusterStyle = indices)

train_style = append_style(model, train_norm)
cv_style = append_style(model, cv_norm)
test_style = append_style(model, test_norm)

style_data = train_style.append(cv_style).append(test_style).sort_index()['clusterStyle']
full_data = recipes.join(style_data)
counts = full_data.sort_values(['clusterStyle'])['clusterStyle'].value_counts().sort_index()
plt.figure(figsize=(10,6))
plt.xlabel('Class')
plt.ylabel('# Members')
counts.plot.bar()
full_data = full_data[np.isfinite(full_data['clusterStyle'])]
styleMap = np.zeros((K, 176))

for index, row in full_data.iterrows():
    styleMap[int(row.clusterStyle)-1][row.StyleID-1] += 1

# Get the total amount of each original StyleID
totals = np.zeros(176)

for i in range(len(totals)):
    totals[i] = full_data.StyleID[full_data.StyleID==(i+1)].count()

# Divide the counts in the style map by the total for each StyleID
for i in range(styleMap.shape[0]):
    for j in range(styleMap.shape[1]):
        styleMap[i][j] /= totals[j]
plt.figure(figsize=(20,20))
heatmap = plt.pcolor(styleMap, cmap=plt.cm.binary)
plt.yticks(np.arange(0, K, 1))
plt.xticks(np.arange(0, 176, 10))
plt.colorbar(heatmap)
plt.xlabel('Original StyleID')
plt.ylabel('New Class')
plt.show()
