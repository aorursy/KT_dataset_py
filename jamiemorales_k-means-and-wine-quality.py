# Set-up libraries
import os
import pandas as pd
import seaborn as sns
import numpy as np
sns.set()
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
# Check input data source
for dirname, _, filenames in os.walk('../input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Read-in data
df = pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
# Look at some details
df.info()
# Look at some records
df.head()
# Check for missing values
df.isna().sum()
# Check for duplicate values
df.duplicated().sum()
# Look at breakdown of Gender
df['quality'].value_counts()
sns.countplot(df['quality'])
# Explore data visually with boxplots
f, ax = plt.subplots(2, 2, figsize=(16, 12))
sns.boxplot('quality', 'alcohol', data=df, ax=ax[0, 0])
sns.boxplot('quality', 'sulphates', data=df, ax=ax[0, 1])
sns.boxplot('quality', 'volatile acidity', data=df, ax=ax[1, 0])
sns.boxplot('quality', 'citric acid', data=df, ax=ax[1,1])
# Explore correlation of other features to quality
df.corr()['quality'].sort_values(ascending=False)
# Summarise
df.describe()
# Get the features for input
X = df.drop('quality', axis=1)
# Scale the values
X_scaled = StandardScaler().fit_transform(X)
# Build and fit models
wcss_scores = []
iterations = list(range(1,10))

for k in iterations:
    model = KMeans(n_clusters=k)
    model.fit(X_scaled)
    wcss_scores.append(model.inertia_)
# Plot performances
plt.figure(figsize=(12,6))
sns.lineplot(iterations, wcss_scores)
# Visualise the clusterds considerig fixed acidity, residual sugar, and alcohol
fig = plt.figure(figsize=(20, 15))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=15, azim=40)
model = KMeans(n_clusters=2)
model.fit(X)
labels = model.labels_
ax.scatter(X['fixed acidity'], X['residual sugar'], X['alcohol'],c=labels.astype(np.float), edgecolor='k')
ax.set_xlabel('Acidity')
ax.set_ylabel('Sugar')
ax.set_zlabel('Alcohol')
ax.set_title('K=2: Acidity, Sugar, Alcohol', size=22)
fig = plt.figure(figsize=(20, 15))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=15, azim=40)
model = KMeans(n_clusters=5)
model.fit(X)
labels = model.labels_
ax.scatter(X['fixed acidity'], X['residual sugar'], X['alcohol'],c=labels.astype(np.float), edgecolor='k')
ax.set_xlabel('Acidity')
ax.set_ylabel('Sugar')
ax.set_zlabel('Alcohol')
ax.set_title('K=5: Acidity, Sugar, Alcohol', size=22)
# Visualise the clusterds considerig fixed acidity, residual sugar, and alcohol
fig = plt.figure(figsize=(20, 15))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=15, azim=40)
model = KMeans(n_clusters=7)
model.fit(X)
labels = model.labels_
ax.scatter(X['fixed acidity'], X['residual sugar'], X['alcohol'],c=labels.astype(np.float), edgecolor='k')
ax.set_xlabel('Acidity')
ax.set_ylabel('Sugar')
ax.set_zlabel('Alcohol')
ax.set_title('K=7: Acidity, Sugar, Alcohol', size=22)
# Compare clusters generated to the ground truth
fig = plt.figure(figsize=(20, 15))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=15, azim=40)
ax.scatter(df['fixed acidity'], df['residual sugar'], df['alcohol'],c=df['quality'], edgecolor='k')
ax.set_xlabel('Acidity')
ax.set_ylabel('Sugar')
ax.set_zlabel('Alcohol')
ax.set_title('Ground truth', size=22)
# Visualise the clusterds considerig fixed acidity, residual sugar, and alcohol
fig = plt.figure(figsize=(20, 18))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=15, azim=40)
model = KMeans(n_clusters=2)
model.fit(X)
labels = model.labels_
ax.scatter(X['fixed acidity'], X['residual sugar'], X['alcohol'],c=labels.astype(np.float), edgecolor='k')
ax.set_xlabel('Acidity')
ax.set_ylabel('Sugar')
ax.set_zlabel('Alcohol')
ax.set_title('K=2: Acidity, Sugar, Alcohol', size=22)
# Compare generated cluster with ground truth
fig = plt.figure(figsize=(20, 18))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=15, azim=40)
ax.scatter(df['fixed acidity'][df['quality']<6], 
           df['residual sugar'][df['quality']<6], 
           df['alcohol'][df['quality']<6],
           edgecolor='k')
ax.set_xlabel('Acidity')
ax.set_ylabel('Sugar')
ax.set_zlabel('Alcohol')
ax.set_title('Ground truth: Low to average quality wine', size=22)
# Visualise the clusterds considerig fixed acidity, residual sugar, and alcohol
fig = plt.figure(figsize=(20, 18))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=15, azim=40)
ax.scatter(df['fixed acidity'][df['quality']>6], 
           df['residual sugar'][df['quality']>6], 
           df['alcohol'][df['quality']>6],
           edgecolor='k')
ax.set_xlabel('Acidity')
ax.set_ylabel('Sugar')
ax.set_zlabel('Alcohol')
ax.set_title('Ground truth: High quality wine', size=22)
