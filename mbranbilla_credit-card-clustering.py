import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing as pp
from sklearn.cluster import KMeans

sns.set()
%matplotlib inline

# Display Options
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = None
# Import data
data = pd.read_csv("../input/CC GENERAL.csv")
data.head()
# Overview
data.describe()
# View missing values (count)
data.isna().sum()
# Fill NAs by mean
data = data.fillna(data.mean())

data.isna().sum()
# Remove CUST_ID (not usefull)
data.drop("CUST_ID", axis=1, inplace=True)
data.dtypes
# Unique values for int64 types
data[['CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'TENURE']].nunique()
# Correlation plot
sns.heatmap(data.corr(),
            xticklabels=data.columns,
            yticklabels=data.columns
           )
# Pairplot - dispersion between variables
sns.pairplot(data)
# Distribution of int64 variables
fig, axes = plt.subplots(nrows=3, ncols=1)
ax0, ax1, ax2 = axes.flatten()

ax0.hist(data['CASH_ADVANCE_TRX'], 65, histtype='bar', stacked=True)
ax0.set_title('CASH_ADVANCE_TRX')

ax1.hist(data['PURCHASES_TRX'], 173, histtype='bar', stacked=True)
ax1.set_title('PURCHASES_TRX')

ax2.hist(data['TENURE'], 7, histtype='bar', stacked=True)
ax2.set_title('TENURE')

fig.tight_layout()
plt.show()
# Create a copy of data
features = data.copy()
list(features)
# Log-transformation

cols =  ['BALANCE',
         'PURCHASES',
         'ONEOFF_PURCHASES',
         'INSTALLMENTS_PURCHASES',
         'CASH_ADVANCE',
         'CASH_ADVANCE_TRX',
         'PURCHASES_TRX',
         'CREDIT_LIMIT',
         'PAYMENTS',
         'MINIMUM_PAYMENTS',
        ]

# Note: Adding 1 for each value to avoid inf values
features[cols] = np.log(1 + features[cols])

features.head()
features.describe()
# Using boxplot for indentify possible outliers values after log-transform

features.boxplot(rot=90, figsize=(30,10))
cols = list(features)
irq_score = {}

for c in cols:
    q1 = features[c].quantile(0.25)
    q3 = features[c].quantile(0.75)
    score = q3 - q1
    outliers = features[(features[c] < q1 - 1.5 * score) | (features[c] > q3 + 1.5 * score)][c]
    values = features[(features[c] >= q1 - 1.5 * score) | (features[c] <= q3 + 1.5 * score)][c]
    
    irq_score[c] = {
        "Q1": q1,
        "Q3": q3,
        "IRQ": score,
        "n_outliers": outliers.count(),
        "outliers_avg": outliers.mean(),
        "outliers_stdev": outliers.std(),
        "outliers_median": outliers.median(),
        "values_avg:": values.mean(),
        "values_stdev": values.std(),
        "values_median": values.median(),
    }
    
irq_score = pd.DataFrame.from_dict(irq_score, orient='index')

irq_score
# Scale All features

for col in cols:
    features[col] = pp.scale(np.array(features[col]))

features.head()
X = np.array(features)
Sum_of_squared_distances = []
K = range(1, 30)

for k in K:
    km = KMeans(n_clusters=k, random_state=0)
    km = km.fit(X)
    Sum_of_squared_distances.append(km.inertia_)

plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()

# Custumers per cluster

n_clusters = 10

clustering = KMeans(n_clusters=n_clusters,
                    random_state=0
                   )

cluster_labels = clustering.fit_predict(X)

# plot cluster sizes

plt.hist(cluster_labels, bins=range(n_clusters+1))
plt.title('# Customers per Cluster')
plt.xlabel('Cluster')
plt.ylabel('# Customers')
plt.show()

# Assing cluster number to features and original dataframe
features['cluster_index'] = cluster_labels
data['cluster_index'] = cluster_labels
# Dispersion between clusterized data
# Pairplot - dispersion between variables
sns.pairplot(features, hue='cluster_index')
# View Features
features
# View results
data
