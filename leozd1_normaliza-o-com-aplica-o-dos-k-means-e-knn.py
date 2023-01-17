import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

game = pd.read_csv('vgsales.csv')

game

jogos = list(game.groupby('Publisher'))

#jogos

MediaPlataforma = game.groupby('Platform').Global_Sales.mean()

#MediaPlataforma

pd.value_counts(game['Genre']).plot.barh()

pd.value_counts(game['Publisher']).head().plot.barh()

pd.value_counts(game['Year']).head().plot.barh()

pd.value_counts(game['Platform']).head().plot.barh()

MediaPlataforma = game.groupby('Platform').Global_Sales.mean().head().plot.barh()

MedianaPlataforma = game.groupby('Platform').Global_Sales.median().head().plot.barh()

patrao = game.groupby('Platform')['Global_Sales'].value_counts()

desviopatrao = patrao.std()

#desviopatrao

game["Year"].plot.hist(bins=30, edgecolor='black')

mediaPublica = game.groupby('Publisher').Global_Sales.mean().head().plot.barh()

medianaPublica = game.groupby('Publisher').Global_Sales.median().head().plot.barh()
import numpy as np

import pandas as pd

import seaborn as sbn

import matplotlib.pyplot as plt



game = pd.read_csv('vgsales.csv', sep=',', encoding='utf-8')

game["Global_Sales"].describe()



game["NA_Sales"].describe()

game["EU_Sales"].describe()

game["JP_Sales"].describe()

game["Other_Sales"].describe()

game["Global_Sales"].describe()



from sklearn import preprocessing

na = preprocessing.StandardScaler().fit(game[["NA_Sales"]])

eu = preprocessing.StandardScaler().fit(game[["EU_Sales"]])

jp = preprocessing.StandardScaler().fit(game[["JP_Sales"]])

ot = preprocessing.StandardScaler().fit(game[["Other_Sales"]])

gs = preprocessing.StandardScaler().fit(game[["Global_Sales"]])

game["NA_Sales"] = na.transform(game[["NA_Sales"]])

game["EU_Sales"] = eu.transform(game[["EU_Sales"]])

game["JP_Sales"] = jp.transform(game[["JP_Sales"]])

game["Other_Sales"] = ot.transform(game[["Other_Sales"]])

game["Global_Sales"] = gs.transform(game[["Global_Sales"]])



game["NA_Sales"].describe()

game["EU_Sales"].describe()

game["JP_Sales"].describe()

game["Other_Sales"].describe()

game["Global_Sales"].describe()



X = game.iloc[:, 6:11].values

X

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 5)

kmeans.fit(X)

kmeans.cluster_centers_

distance = kmeans.fit_transform(X)

distance

labels = kmeans.labels_

labels

plt.scatter(X[:, 0], X[:,1], s = 100, c = kmeans.labels_)

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 50, c = 'red',label = 'Centroids')

plt.title('Jogos Clusters and Centroids')

plt.xlabel('X')

plt.ylabel('Y')

plt.legend()



plt.show()
import numpy as np  

import matplotlib.pyplot as plt  

import pandas as pd



game = pd.read_csv('vgsales.csv', sep=',', encoding='utf-8')

game.head(30)



X = game.iloc[:, 6:11].values  

y = game.iloc[:, 2].values



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)



from sklearn.preprocessing import StandardScaler  

scaler = StandardScaler()  

scaler.fit(X_train)



X_train = scaler.transform(X_train)  

X_test = scaler.transform(X_test)



from sklearn.neighbors import KNeighborsClassifier  

classifier = KNeighborsClassifier(n_neighbors=5)  

classifier.fit(X_train, y_train)  



y_pred = classifier.predict(X_test)  



from sklearn.metrics import classification_report, confusion_matrix  

print(confusion_matrix(y_test, y_pred))  

print(classification_report(y_test, y_pred))  



error = []

for i in range(1, 40):  

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train, y_train)

    pred_i = knn.predict(X_test)

    error.append(np.mean(pred_i != y_test))



plt.figure(figsize=(12, 6))  

plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',  

         markerfacecolor='blue', markersize=10)

plt.title('Taxa erro de valores K ')  

plt.xlabel('K Valores')  

plt.ylabel('Media de Erro')  