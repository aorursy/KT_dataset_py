#Mall Customer Segmentation Data

%reset -f

import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt 

import seaborn as sns

%matplotlib inline

import sklearn

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score

cseg= pd.read_csv("../input/mall-customer-segmentation-data/Mall_Customers.csv")
cseg.head()
cseg.shape
cseg.dtypes
cseg['Gender'].unique()
cseg= cseg.rename(columns= {"Annual Income (k$)": "annualincome","Spending Score (1-100)":"spendingscore"})
cseg.head()
cseg.describe()
cseg.isnull().sum()        # no null values found in the given columns
cseg.groupby('Gender').spendingscore.mean()   #Average spending score of Female is higher than Men
cseg.groupby('Gender').annualincome.mean()
cseg.groupby('Gender').spendingscore.mean().plot(kind= 'bar')
cseg.groupby('Gender').spendingscore.mean().plot(kind= 'line')
sns.jointplot(cseg.annualincome, cseg.spendingscore, kind = 'reg')
cseg.info()
sns.pairplot(data=cseg, hue='Gender')
sns.pairplot(cseg, diag_kind="kde", markers="+",

                 plot_kws=dict(s=50, edgecolor="b", linewidth=1),

                 diag_kws=dict(shade=True))
sns.pairplot(cseg, kind="reg")
sns.heatmap(cseg.corr(), cmap='coolwarm')
cseg.drop(columns=['CustomerID','Gender'],inplace= True)  #Dropping columns not needed
cseg.head(10)
# Copy 'Age' column to another variable and then drop it

#     We will not use it in clustering

y = cseg['Age'].values

cseg.drop(columns = ['Age'], inplace = True)
X_train, X_test, _, y_test = train_test_split(cseg,

                                               y,

                                               test_size = 0.25

                                               )
X_train.shape
X_test.shape
# Develop model

# Create an instance of modeling class

clf = KMeans(n_clusters = 2)

# Train the class over data

clf.fit(X_train)
# So what are our clusters?

clf.cluster_centers_
clf.cluster_centers_.shape                      

clf.labels_
clf.labels_.size
clf.inertia_
# 6 Make prediction over our test data and check accuracy

y_pred = clf.predict(X_test)

y_pred
np.sum(y_pred == y_test)/y_test.size
# 7.0 Are clusters distiguisable?

sns.scatterplot('annualincome','spendingscore', hue = y_pred, data = X_test)
# 7.1 Scree plot:

sse = []

for i,j in enumerate(range(10)):

    # How many clusters?

    n_clusters = i+1

    # Create an instance of class

    clf = KMeans(n_clusters = n_clusters)

    # Train the kmeans object over data

    clf.fit(X_train)

    # Store the value of inertia in sse

    sse.append(clf.inertia_ )
sns.lineplot(range(1, 11), sse)
clf1 = KMeans(n_clusters = 5)
clf1.fit(X_train)
clf1.cluster_centers_
clf1.cluster_centers_.shape
clf1.labels_
clf1.labels_.size
clf1.inertia_
y_pred = clf1.predict(X_test)

y_pred
np.sum(y_pred == y_test)/y_test.size
sns.scatterplot('annualincome','spendingscore', hue = y_pred, data = X_test)