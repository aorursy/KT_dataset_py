# Set-up libraries
import os
import pandas as pd
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
# Check input data source
for dirname, _, filenames in os.walk('../input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Read-in data
df = pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
# Look at some details
df.info()
# Look at some records
df.head()
# Check for missing values
df.isna().sum()
# Check for duplicate values
df.duplicated().sum()
# Look at breakdown of Gender
df['Gender'].value_counts()
sns.countplot(df['Gender'])
# Look at distribution of Age
sns.distplot(df['Age'])
# Look at distribution of Annual Income
sns.distplot(df['Annual Income (k$)'])
# Look at distribution of Spending
sns.distplot(df['Spending Score (1-100)'])
# Explore data visually with scatter plots
sns.pairplot(df)
# Explore data visually with kernel density estimations
g = sns.PairGrid(df)
g.map_diag(sns.kdeplot)
g.map_offdiag(sns.kdeplot, n_levels=6);
# Summarise
df.describe()
# Rename columns for easier handling
df = df.rename(columns={'Annual Income (k$)': 'Income',
                  'Spending Score (1-100)': 'Spending'
                  })
df.head()
# Transform categorical features to numeric
le = LabelEncoder()
le.fit(df['Gender'].drop_duplicates())
df['Gender'] = le.transform(df['Gender'])
# Look at breakdown of feature Gender
df['Gender'].value_counts()
# Get the features for input
X = df.drop('CustomerID', axis=1)
X.head()
# Build and fit models
wcss_scores = []
iterations = list(range(1,10))

for k in iterations:
    model = KMeans(n_clusters=k)
    model.fit(X)
    model.fit(X)
    wcss_scores.append(model.inertia_)
# Plot performances
plt.figure(figsize=(12,6))
sns.lineplot(iterations, wcss_scores)
# Visualise the clusters, considering Income and Spending
plt.figure(figsize=(27,27))

plt.subplot(3,2,1)
plt.title('K = 2: Income',fontsize=22)
plt.xlabel('Income')
plt.xlabel('Spending')
model = KMeans(n_clusters=2)
X['labels'] = model.fit_predict(X)
plt.scatter(X.Spending[X.labels == 0], X.Income[X.labels == 0])
plt.scatter(X.Spending[X.labels == 1], X.Income[X.labels == 1])

plt.subplot(3,2,2)
plt.title('K = 3: Income',fontsize=22)
plt.xlabel('Income')
plt.xlabel('Spending')
model = KMeans(n_clusters=3)
X['labels'] = model.fit_predict(X)
plt.scatter(X.Spending[X.labels == 0], X.Income[X.labels == 0])
plt.scatter(X.Spending[X.labels == 1], X.Income[X.labels == 1])
plt.scatter(X.Spending[X.labels == 2], X.Income[X.labels == 2])

plt.subplot(3, 2, 3)
plt.title('K = 4: Income', fontsize=22)
plt.xlabel('Income')
plt.ylabel('Spending')
model = KMeans(n_clusters=4)
X['labels'] = model.fit_predict(X)
plt.scatter(X.Spending[X.labels == 0], X.Income[X.labels == 0])
plt.scatter(X.Spending[X.labels == 1], X.Income[X.labels == 1])
plt.scatter(X.Spending[X.labels == 2], X.Income[X.labels == 2])
plt.scatter(X.Spending[X.labels == 3], X.Income[X.labels == 3])

plt.subplot(3, 2, 4)
plt.title('K = 5: Income', fontsize=22)
plt.xlabel('Income')
plt.ylabel('Spending')
model = KMeans(n_clusters=5)
X['labels'] = model.fit_predict(X)
plt.scatter(X.Spending[X.labels == 0], X.Income[X.labels == 0])
plt.scatter(X.Spending[X.labels == 1], X.Income[X.labels == 1])
plt.scatter(X.Spending[X.labels == 2], X.Income[X.labels == 2])
plt.scatter(X.Spending[X.labels == 3], X.Income[X.labels == 3])
plt.scatter(X.Spending[X.labels == 4], X.Income[X.labels == 4])
# Visualise most interesting clusters
plt.figure(figsize=(24,12))

plt.title('K = 5: Income', fontsize=22)
plt.xlabel('Income')
plt.ylabel('Spending')
model = KMeans(n_clusters=5)
X['labels'] = model.fit_predict(X)
plt.scatter(X.Spending[X.labels == 0], X.Income[X.labels == 0])
plt.scatter(X.Spending[X.labels == 1], X.Income[X.labels == 1])
plt.scatter(X.Spending[X.labels == 2], X.Income[X.labels == 2])
plt.scatter(X.Spending[X.labels == 3], X.Income[X.labels == 3])
plt.scatter(X.Spending[X.labels == 4], X.Income[X.labels == 4])
# Visualise the clusters, considering Age and Spending
plt.figure(figsize=(27,27))

plt.subplot(3,2,1)
plt.title('K = 2: Income',fontsize=22)
plt.xlabel('Age')
plt.xlabel('Spending')
model = KMeans(n_clusters=2)
X['labels'] = model.fit_predict(X)
plt.scatter(X.Spending[X.labels == 0], X.Age[X.labels == 0])
plt.scatter(X.Spending[X.labels == 1], X.Age[X.labels == 1])

plt.subplot(3,2,2)
plt.title('K = 3: Income',fontsize=22)
plt.xlabel('Age')
plt.xlabel('Spending')
model = KMeans(n_clusters=3)
X['labels'] = model.fit_predict(X)
plt.scatter(X.Spending[X.labels == 0], X.Age[X.labels == 0])
plt.scatter(X.Spending[X.labels == 1], X.Age[X.labels == 1])
plt.scatter(X.Spending[X.labels == 2], X.Age[X.labels == 2])

plt.subplot(3, 2, 3)
plt.title('K = 4: Income', fontsize=22)
plt.xlabel('Age')
plt.ylabel('Spending')
model = KMeans(n_clusters=4)
X['labels'] = model.fit_predict(X)
plt.scatter(X.Spending[X.labels == 0], X.Age[X.labels == 0])
plt.scatter(X.Spending[X.labels == 1], X.Age[X.labels == 1])
plt.scatter(X.Spending[X.labels == 2], X.Age[X.labels == 2])
plt.scatter(X.Spending[X.labels == 3], X.Age[X.labels == 3])

plt.subplot(3, 2, 4)
plt.title('K = 5: Income', fontsize=22)
plt.xlabel('Age')
plt.ylabel('Spending')
model = KMeans(n_clusters=5)
X['labels'] = model.fit_predict(X)
plt.scatter(X.Spending[X.labels == 0], X.Age[X.labels == 0])
plt.scatter(X.Spending[X.labels == 1], X.Age[X.labels == 1])
plt.scatter(X.Spending[X.labels == 2], X.Age[X.labels == 2])
plt.scatter(X.Spending[X.labels == 3], X.Age[X.labels == 3])
plt.scatter(X.Spending[X.labels == 4], X.Age[X.labels == 4])
# Visualise interesting clusters
plt.figure(figsize=(27,27))

plt.subplot(3,2,1)
plt.title('K = 2: Income',fontsize=22)
plt.xlabel('Age')
plt.xlabel('Spending')
model = KMeans(n_clusters=2)
X['labels'] = model.fit_predict(X)
plt.scatter(X.Spending[X.labels == 0], X.Age[X.labels == 0])
plt.scatter(X.Spending[X.labels == 1], X.Age[X.labels == 1])
