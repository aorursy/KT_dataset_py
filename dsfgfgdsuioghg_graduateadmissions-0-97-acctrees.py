import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

import os





df = pd.read_csv("../input/Admission_Predict.csv")



df.isna().any

sns.heatmap(df.isna())



df.corr()

sns.heatmap(df.corr(), annot = True)



# plots

# having research

sns.countplot(x='Research', data = df)

plt.xlabel("Canditates")

plt.ylabel("Frequency")





# FOR - gen.stats

for x in df.columns:

    mean = df[x].mean()

    median = df[x].median()

    min = df[x].min()

    max = df[x].max()

    print('mean', x, mean, 'median', x, median,'min', x, min,'max', x, max)

    

# toefl

sns.distplot(df['TOEFL Score'])

mean = df['TOEFL Score'].mean()

median = df['TOEFL Score'].median()

min = df['TOEFL Score'].min()

max = df['TOEFL Score'].max()

print('mean TOEFL Score', mean, 'median TOEFL Score', median,'min TOEFL Score', min,'max TOEFL Score', max)



y1 = np.array([df["TOEFL Score"].min(),df["TOEFL Score"].mean(),df["TOEFL Score"].max()])

x1 = ["Worst","Average","Best"]

plt.bar(x1,y1)



sns.scatterplot(x="TOEFL Score", y="CGPA", data=df)



# GRE Score

sns.distplot(df['GRE Score'])

sns.distplot(df['GRE Score'])

mean_gre = df['GRE Score'].mean()

median_gre = df['GRE Score'].median()

min_gre = df['GRE Score'].min()

max_gre = df['GRE Score'].max()

print('mean GRE Score', mean_gre, 'median GRE Score', median_gre,'min GRE Score', min_gre,'max GRE Score', max_gre)



y_gre = np.array([df["GRE Score"].min(),df["GRE Score"].mean(),df["GRE Score"].max()])

x_gre = ["Worst","Average","Best"]

plt.bar(x_gre,y_gre)



sns.scatterplot(x="GRE Score", y="CGPA", data=df)



# university stats

df.groupby(['University Rating'])['University Rating'].count().sort_values()



# regression linear

x_r = df.iloc[:, 1:7].values

y_r = df.iloc[:, 8].values

from sklearn.model_selection import train_test_split

x_train, x_test,y_train, y_test = train_test_split(x_r,y_r,test_size = 0.20,random_state = 42)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x_train,y_train)

y_head_lr = lr.predict(x_test)

from sklearn.metrics import r2_score

print("r_square score: ", r2_score(y_test,y_head_lr))



y_head_lr_train = lr.predict(x_train)

print("r_square score (train dataset): ", r2_score(y_train,y_head_lr_train))



# regression D trees

# Fitting Decision Tree Regression to the dataset

from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state = 0)

regressor.fit(x_r, y_r)

# Predicting a new result

y_pred_r = regressor.predict(x_test)

from sklearn.metrics import r2_score

print("r_square score: ", r2_score(y_test,y_pred_r))



# regression RF trees

# Fitting Random Forest Regression to the dataset

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators = 500, random_state = 0)

regressor.fit(x_r, y_r)

# Predicting a new result

y_pred_RF = regressor.predict(x_test)

print("r_square score: ", r2_score(y_test,y_pred_RF))





# classification

# Using the elbow method to find the optimal number of clusters

from sklearn.cluster import KMeans

wcss = []

for i in range(1, 11):

    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)

    kmeans.fit(x)

    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()

# Fitting K-Means to the dataset

kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)

y_kmeans = kmeans.fit_predict(x)

# Visualising the clusters

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')

plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')

plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')

plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')

plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')

plt.title('Clusters of customers')

plt.xlabel('Annual Income (k$)')

plt.ylabel('Spending Score (1-100)')

plt.legend()

plt.show()




