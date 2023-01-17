import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from wordcloud import WordCloud

from urllib.request import urlopen

from PIL import Image

import os
print(os.listdir("../input"))
whr2019 = pd.read_csv('../input/world-happiness-report-2019.csv')
whr2019.head()
whr2019.isnull().sum()
whr2019.dtypes
plt.figure(figsize = (10,8))

sns.heatmap(whr2019.corr(), annot=True, linewidths=.2, cmap = 'OrRd',linecolor= 'crimson')
whr2019 = whr2019.rename({'Country (region)' : 'Country'},axis = 1)
whr2019.columns
whr2019 = whr2019.rename({'Positive affect':'Positive_affect'},axis = 1)

plt.figure(figsize = (15,8))



plt.subplot2grid((1,2),(0,0))

whr2019.groupby(['Country']).Positive_affect.sum().nlargest(30).plot(kind = 'barh')

plt.xlabel("Positive affect")



plt.subplot2grid((1,2),(0,1))

whr2019.groupby(['Country']).Positive_affect.sum().nsmallest(30).plot(kind = 'barh')

plt.xlabel("Positive affect")



plt.tight_layout()

plt.show()
whr2019 = whr2019.rename({'Negative affect':'Negative_affect'},axis = 1)

plt.figure(figsize = (15,8))



plt.subplot2grid((1,2),(0,0))

whr2019.groupby(['Country']).Negative_affect.sum().nlargest(30).plot(kind = 'barh')

plt.xlabel("Negative affect")



plt.subplot2grid((1,2),(0,1))

whr2019.groupby(['Country']).Negative_affect.sum().nsmallest(30).plot(kind = 'barh')

plt.xlabel("Negative affect")



plt.tight_layout()

plt.show()
whr2019 = whr2019.rename({'Social support' : 'Social_support'},axis = 1)

fig = plt.figure(figsize = (15,8))

plt.subplots_adjust(hspace = 0.5)



plt.subplot2grid((1,2),(0,0))

whr2019.groupby(['Country']).Social_support.sum().nlargest(30).plot(kind = 'barh')

plt.xlabel("Social_support")



plt.subplot2grid((1,2),(0,1))

whr2019.groupby(['Country']).Social_support.sum().nsmallest(30).plot(kind = 'barh')

plt.xlabel("Social_support")



plt.tight_layout()

plt.show()
fig = plt.figure(figsize = (15,8))

plt.subplots_adjust(hspace = 0.5)



plt.subplot2grid((1,2),(0,0))

whr2019.groupby(['Country']).Freedom.sum().nlargest(30).plot(kind = 'barh')

plt.xlabel("Freedom")



plt.subplot2grid((1,2),(0,1))

whr2019.groupby(['Country']).Freedom.sum().nsmallest(30).plot(kind = 'barh')

plt.xlabel("Freedom")



plt.tight_layout()

plt.show()
plt.figure(figsize = (15,8))



plt.subplot2grid((1,2),(0,0))

whr2019.groupby(['Country']).Corruption.sum().nlargest(30).plot(kind='barh')

plt.xlabel("Corruption")



plt.subplot2grid((1,2),(0,1))

whr2019.groupby(['Country']).Corruption.sum().nsmallest(30).plot(kind='barh')

plt.xlabel("Corruption")



plt.tight_layout()

plt.show()
fig = plt.figure(figsize = (15,8))

plt.subplots_adjust(hspace = 0.5)



plt.subplot2grid((1,2),(0,0))

whr2019.groupby(['Country']).Generosity.sum().nlargest(30).plot(kind = 'barh')

plt.xlabel("Generosity")



plt.subplot2grid((1,2),(0,1))

whr2019.groupby(['Country']).Generosity.sum().nsmallest(30).plot(kind = 'barh')

plt.xlabel("Generosity")



plt.tight_layout()

plt.show()
whr2019 = whr2019.rename({'Log of GDP\nper capita' : 'Log_of_GDP_per_capita'},axis = 1)

fig = plt.figure(figsize = (15,8))

plt.subplots_adjust(hspace = 0.5)



plt.subplot2grid((1,2),(0,0))

whr2019.groupby(['Country']).Log_of_GDP_per_capita.sum().nlargest(30).plot(kind = 'barh')

plt.xlabel("Log of GDP per capita")



plt.subplot2grid((1,2),(0,1))

whr2019.groupby(['Country']).Log_of_GDP_per_capita.sum().nsmallest(30).plot(kind = 'barh')

plt.xlabel("Log of GDP per capita")



plt.tight_layout()

plt.show()
whr2019 = whr2019.rename({'Healthy life\nexpectancy':'Healthy_life_expectancy'},axis = 1)

fig = plt.figure(figsize = (15,8))

plt.subplots_adjust(hspace = 0.5)



plt.subplot2grid((1,2),(0,0))

whr2019.groupby(['Country']).Healthy_life_expectancy.sum().nlargest(30).plot(kind = 'barh')

plt.xlabel("Healthy life expectancy")



plt.subplot2grid((1,2),(0,1))

whr2019.groupby(['Country']).Healthy_life_expectancy.sum().nsmallest(30).plot(kind = 'barh')

plt.xlabel("Healthy life expectancy")



plt.tight_layout()

plt.show()
x = whr2019['Ladder']

y = whr2019['Social_support']

plt.figure(figsize = (15,5))

plt.scatter(x,y)

plt.xlabel("Country")

plt.ylabel("Social_support")

plt.show()
x = whr2019['Ladder']

y = whr2019['Log_of_GDP_per_capita']

plt.figure(figsize = (15,5))

plt.scatter(x,y)

plt.xlabel("Country")

plt.ylabel("Log_of_GDP_per_capita")

plt.show()
x = whr2019['Ladder']

y = whr2019['Healthy_life_expectancy']

plt.figure(figsize = (15,5))

plt.scatter(x,y)

plt.xlabel("Country")

plt.ylabel("Healthy_life_expectancy")

plt.show()
for col in whr2019.columns:

    if whr2019[col].isnull().sum() != 0:

        whr2019[col] = whr2019[col].fillna(0.0)

whr2019.isnull().sum()
x = whr2019.drop(['Country'],axis = 1)
# find n_clusters value



from sklearn.cluster import KMeans



a = []

for k in range (1,12):

    kmeans = KMeans(n_clusters = k)

    kmeans.fit(x)

    a.append(kmeans.inertia_)



plt.plot(range(1,12),a)

plt.xlabel('k values')

plt.ylabel('a')

plt.show()
#n_clusters= 3



kmeans = KMeans(n_clusters= 3)

kmeans.fit(x)
clusters_knn = kmeans.fit_predict(x)
plt.scatter(x[clusters_knn == 0]['Ladder'],x[clusters_knn == 0]['Positive_affect'], color = 'Red')

plt.scatter(x[clusters_knn == 1]['Ladder'],x[clusters_knn == 1]['Positive_affect'], color = 'Blue')

plt.scatter(x[clusters_knn == 2]['Ladder'],x[clusters_knn == 2]['Positive_affect'], color = 'Green')





plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 2], s = 300, c = 'yellow', label = 'Centroids')

plt.title('Effect of Positive_affect in Happiness index')

plt.xlabel('Ladder')

plt.ylabel('Positive_affect')



plt.show()
plt.scatter(x[clusters_knn == 0]['Ladder'],x[clusters_knn == 0]['Negative_affect'], color = 'Red')

plt.scatter(x[clusters_knn == 1]['Ladder'],x[clusters_knn == 1]['Negative_affect'], color = 'Blue')

plt.scatter(x[clusters_knn == 2]['Ladder'],x[clusters_knn == 2]['Negative_affect'], color = 'Green')





plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 3], s = 300, c = 'yellow', label = 'Centroids')

plt.title('Effect of Negative_affect in Happiness index')

plt.xlabel('Ladder')

plt.ylabel('Negative_affect')



plt.show()
plt.scatter(x[clusters_knn == 0]['Ladder'],x[clusters_knn == 0]['Social_support'], color = 'Red')

plt.scatter(x[clusters_knn == 1]['Ladder'],x[clusters_knn == 1]['Social_support'], color = 'Blue')

plt.scatter(x[clusters_knn == 2]['Ladder'],x[clusters_knn == 2]['Social_support'], color = 'Green')

#plt.scatter(x[clusters_knn == 3]['Ladder'],x[clusters_knn == 3]['Social_support'], color = 'lightcoral')





plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 4], s = 300, c = 'yellow', label = 'Centroids')

plt.title('Effect of Social_support in Happiness index')

plt.xlabel('Ladder')

plt.ylabel('Social_support')



plt.show()
plt.scatter(x[clusters_knn == 0]['Ladder'],x[clusters_knn == 0]['Log_of_GDP_per_capita'], color = 'Red')

plt.scatter(x[clusters_knn == 1]['Ladder'],x[clusters_knn == 1]['Log_of_GDP_per_capita'], color = 'Blue')

plt.scatter(x[clusters_knn == 2]['Ladder'],x[clusters_knn == 2]['Log_of_GDP_per_capita'], color = 'Green')





plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 8], s = 300, c = 'yellow', label = 'Centroids')

plt.title('Effect of Log_of_GDP_per_capita in Happiness index')

plt.xlabel('Ladder')

plt.ylabel('Log_of_GDP_per_capita')



plt.show()
plt.scatter(x[clusters_knn == 0]['Ladder'],x[clusters_knn == 0]['Healthy_life_expectancy'], color = 'Red')

plt.scatter(x[clusters_knn == 1]['Ladder'],x[clusters_knn == 1]['Healthy_life_expectancy'], color = 'Blue')

plt.scatter(x[clusters_knn == 2]['Ladder'],x[clusters_knn == 2]['Healthy_life_expectancy'], color = 'Green')





plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 9], s = 300, c = 'yellow', label = 'Centroids')

plt.title('Effect of Healthy_life_expectancy in Happiness index')

plt.xlabel('Ladder')

plt.ylabel('Healthy_life_expectancy')



plt.show()
plt.scatter(x[clusters_knn == 0]['Ladder'],x[clusters_knn == 0]['Corruption'], color = 'Red')

plt.scatter(x[clusters_knn == 1]['Ladder'],x[clusters_knn == 1]['Corruption'], color = 'Blue')

plt.scatter(x[clusters_knn == 2]['Ladder'],x[clusters_knn == 2]['Corruption'], color = 'Green')





plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 6], s = 300, c = 'yellow', label = 'Centroids')

plt.title('Effect of Corruption in Happiness index')

plt.xlabel('Ladder')

plt.ylabel('Corruption')



plt.show()