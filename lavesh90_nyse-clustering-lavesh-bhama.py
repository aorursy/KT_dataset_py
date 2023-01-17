# import libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import plotly.express as px

import seaborn as sns
# import scalling and clustering libraries.

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score

from yellowbrick.cluster import SilhouetteVisualizer
# to display all the commands result, otherwise it will display only the last command's results

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
# to get rid of scientific notations



pd.options.display.float_format = '{:.2f}'.format
#set maximum columns and row to display in the output

pd.options.display.max_columns=100

pd.options.display.max_rows=100
# read the desired file

data = pd.read_csv("../input/nyse/fundamentals.csv", index_col = 0, parse_dates=['Period Ending'])
# column cleansing

data.columns = data.columns.str.replace(" ", "_")

data.columns = data.columns.str.replace("_&_", "_")

data.columns = data.columns.str.replace("_/_", "_")

data.columns = data.columns.str.replace(".", "")

data.columns = data.columns.str.replace("-", "")

data.columns = data.columns.str.replace("'", "")

data.columns = data.columns.str.replace(",", "")

data.columns = data.columns.str.replace("/", "_")
# data description and further removal of column with null values

data.shape

data.dropna(axis=1,inplace=True)

data.shape

data.head()
# trying to divide the data into two parts

data.Gross_Profit.mean()

data[data.Gross_Profit >= data.Gross_Profit.mean()].shape

data[data.Gross_Profit <data.Gross_Profit.mean() ].shape



data['cluster_group'] = data['Gross_Profit'].apply(lambda x : 1 if x > data.Gross_Profit.mean() else 0)



data.head()

data.cluster_group.value_counts()
# seaborn baplot with 95% Confidende Interval

sns.barplot('cluster_group', 'Gross_Profit', estimator = np.mean, data=data, ci=95)
# distribution plot 

sns.distplot(data.Gross_Margin, rug=True)
# aggregating the data

datagroup = data.groupby("Ticker_Symbol")

mean_dt = datagroup.aggregate('mean')

mean_dt.reset_index(inplace=True)

mean_dt.head()
TopPerformer = mean_dt.sort_values(by = 'Gross_Profit', ascending=False).head(10)



sns.barplot(x = 'Ticker_Symbol', y = 'Gross_Profit', data = TopPerformer)
data_remTick = data.copy() # Deep Copy

data_remTick.drop(["Ticker_Symbol","Period_Ending"], axis=1, inplace=True)

data_remTick.head()

# removing and storing separately the custom column containing the binary values for two kinds of data

y = data_remTick['cluster_group'].values

data_remTick.drop(columns = ['cluster_group'], inplace = True)
# scalling data

ss = StandardScaler()

ss.fit(data_remTick)

X = ss.transform(data_remTick)

X.shape
# train and test data sets

X_train, X_test, y_train, y_test = train_test_split( X,y, test_size = 0.25)

X_train.shape

X_test.shape

y_train.shape

y_test.shape
# For skree ploting

sse1 = []

for k in range(1,10):

    km = KMeans(n_clusters = k)

    km.fit(X_train)

    sse1.append(km.inertia_)
# simple plot. For arriving at number of cluster.

plt.plot(range(1,10), sse1, marker='*')
# applying KMeans algo with 2 clusters

km2 = KMeans(n_clusters = 2)

km2.fit(X_train)

silhouette_score(X_train, km2.labels_)
# applying KMeans algo with 3 clusters

km1 = KMeans(n_clusters = 3)

km1.fit(X_train)

#km1.cluster_centers_

#km1.labels_

#km1.inertia_

silhouette_score(X_train, km1.labels_) 
kmeans = KMeans(n_clusters=2)

pred_y = kmeans.fit(X)

plt.scatter(X[:,0], X[:,1])

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red')

plt.show()
# predicting with 2 clusters

clf = KMeans(n_clusters = 2)

clf.fit(X_train)

y_pred = clf.predict(X_test)

#y_pred

np.sum(y_pred == y_test)/y_test.size



# a values of 0.78 suggests 78% correct predictions.
# Final Silhouette visualiztion 

visualizer = SilhouetteVisualizer(clf, colors='yellowbrick')

visualizer.fit(X_train)        

visualizer.show() 