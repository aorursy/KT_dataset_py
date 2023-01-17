import pandas as pd        # Data manipulation
import numpy as np         # Array manipulation

# 1.1 Modeling libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score

# 1.1.1 For parameter-search over grid
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostClassifier   # For classification
from catboost import Pool                 # Pool is catboost's internal data structure
                                          
# 1.2 Model performance
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist

# 1.3 For plotting
# conda install -c conda-forge scikit-plot
import scikitplot as skplt      # For roc graphs
import matplotlib.pyplot as plt
import seaborn as sns                # Easier plotting

# 1.4 Misc
import os
import time
import gc
###@@@@@@@@@@@@@@@@@@@@ PART-I @@@@@@@@@@@@@@@@@@@@@@@@@@

# 2.1
train_df = pd.read_csv("../input/sign_mnist_train.csv", header=0)
test_df = pd.read_csv("../input/sign_mnist_test.csv", header=0)

# 2.2 Explore data
train_df.columns        # Column names
train_df.shape          # 114321, 133
train_df.dtypes         # Column types

test_df.columns        # Column names
test_df.shape          # 114321, 133
test_df.dtypes         # Column types

# 2.3 Count of various types of variables
train_df.dtypes.value_counts()          # Same code as all above
test_df.dtypes.value_counts()           # Same code as all above
ktrn_df = train_df.drop(columns='label')    # remove the target column label
ktrn_df.shape[0]                            # check number of rows in new dataframe   
K = [x+1 for x in (np.unique(train_df[['label']].values))] # check number of unique values for optimal clusters
# Normalize the data using scale and fit 
scaler = StandardScaler() 
ktrn_df_scaled = scaler.fit_transform(ktrn_df)
# identify number of clusters using K-Means method with Algorithm: elkan & method of initialization: k-means++'
distortions_elkan = []      # create empty array to collect SSE (sum of squared errors)
start = time.time()         # start the timers for tracking purpose  
for k in K:
    kmeanModel_elkan = KMeans(n_clusters=k, algorithm = 'elkan', init = 'k-means++').fit(ktrn_df)    
    kmeanModel_elkan.fit(ktrn_df)
    distortions_elkan.append(sum(np.min(cdist(ktrn_df, kmeanModel_elkan.cluster_centers_, 'euclidean'), axis=1)) / ktrn_df.shape[0])
end = time.time()           # end the timers for tracking purpose
print(end - start)          # print the time taken for this effort
plt.plot(K, distortions_elkan, 'bx-')   # plot the line chart with distortions
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing optimal k using Algorithm: elkan & method of initialization: k-means++')
plt.show()
# identify number of clusters using K-Means method with Algorithm: full & method of initialization: random'
distortions_full = []       # create empty array to collect SSE (sum of squared errors)
start = time.time()         # start the timers for tracking purpose
for k in K:
    kmeanModel_full = KMeans(n_clusters=k, algorithm = 'full', init = 'random').fit(ktrn_df)    
    kmeanModel_full.fit(ktrn_df)
    distortions_full.append(sum(np.min(cdist(ktrn_df, kmeanModel_full.cluster_centers_, 'euclidean'), axis=1)) / ktrn_df.shape[0])
end = time.time()           # end the timers for tracking purpose
print(end - start)          # print the time taken for this effort
plt.plot(K, distortions_full, 'bx-')    # plot the line chart with distortions
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing optimal k using Algorithm: full & method of initialization: random')
plt.show()