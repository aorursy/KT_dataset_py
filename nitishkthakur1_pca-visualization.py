# Import the necessary libraries
import numpy as np
import pandas as pd
from sklearn import datasets, decomposition, linear_model, preprocessing, model_selection
import matplotlib.pyplot as plt
import seaborn as sns

# Load Boston housing Dataset from sklearn 
data_boston = datasets.load_boston()
X = pd.DataFrame(data = data_boston.data, columns = data_boston.feature_names)
y = pd.DataFrame(data = data_boston.target, columns = ['target'])

# Preview the data
X.head()
# Scale data before performing PCA - to zero mean and unit variance
X_scaled = preprocessing.scale(X)

# Convert to DataFrame
X_scaled = pd.DataFrame(X_scaled, columns = X.columns)
X_scaled.describe()
# Initialize PCA transformer
pca = decomposition.PCA()

# Fit it to data
pca.fit(X_scaled)
# Plot the cumulative variance
fig = plt.figure(figsize = (10,4))
plt.plot(np.arange(1, X.shape[1]+1),100 * np.cumsum(pca.explained_variance_ratio_), marker = 'o',
         color = 'teal', alpha = .8)
plt.xticks(np.arange(1, X.shape[1]+1),np.arange(1, X.shape[1]+1))
plt.xlabel('Number of Components')
plt.ylabel('Expained Variance %')
plt.grid(linestyle = '--')
# Transform data
pca = decomposition.PCA().fit(X_scaled)
X_pca = pca.transform(X_scaled)
# Save the pca weights in a dataframe
pca_component_directions = pd.DataFrame(pca.components_, columns = X.columns, 
                                        index = np.arange(1, X_pca.shape[1]+1))

# Make a heatmap to show the contribution of each feature to each principal component
fig = plt.figure(figsize = (12, 9))
sns.heatmap(pca_component_directions, linewidth = .2, annot = True, cmap = 'coolwarm',
            vmax = 1, vmin = -1)
plt.ylabel('Components', fontsize = 13)
plt.xlabel('Features', fontsize = 13)
# Extract the first 2 principal components
X_2comps = decomposition.PCA(n_components = 2).fit_transform(X_scaled)

fig = plt.figure(figsize = (10,7))
plt.plot(X_2comps[:,0], X_2comps[:,1], marker = 'o', color = 'teal', alpha = .75, linewidth = 0)
plt.xlabel('Principal Component 1', fontsize = 14)
plt.ylabel('Principal Component 2', fontsize = 14)
plt.grid(linestyle = '--')

# Find the median value of y.
threshold = y.median()
threshold
# Store binary y and X in a dataframe 
plot_2comps = pd.DataFrame(X_2comps, columns = ['PC1', 'PC2'])

# Binarize y: replace y with 0 where y < threshold and with 1 where y>=threshold
y_copy = y.copy()
y_copy[y_copy < threshold] = 0
y_copy[y_copy >= threshold] = 1

plot_2comps['target'] = y_copy
# Colour code the 2 Principal Component plot depending on whether target is 1 or 0
X_2comps_0 = plot_2comps[plot_2comps.target == 0][['PC1', 'PC2']].copy()
X_2comps_1 = plot_2comps[plot_2comps.target == 1][['PC1', 'PC2']].copy()

# Teal Data points correspond to target < 21.2 and salmon points correspond to target >=21.2
fig = plt.figure(figsize = (10,7))
plt.plot(X_2comps_0['PC1'], X_2comps_0['PC2'], marker = 'o', color = 'teal', alpha = .75,
         linewidth = 0, label = 'target = 0')
plt.plot(X_2comps_1['PC1'], X_2comps_1['PC2'], marker = 'o', color = 'salmon', alpha = .75,
         linewidth = 0, label = 'target = 1')
plt.xlabel('Principal Component 1', fontsize = 14)
plt.ylabel('Principal Component 2', fontsize = 14)
plt.legend()
plt.grid(linestyle = '--')
