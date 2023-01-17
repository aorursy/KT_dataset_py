# for some basic operations

import numpy as np 

import pandas as pd 



# for visualizations

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')



# for providing path

import os

print(os.listdir("../input"))



data = pd.read_csv('../input/web_log_data.csv')



# getting the shape

data.shape
# checking the head of the data



data.head()


# unique date time

print("No. of Unique Dates :", data['date_time'].nunique())

      

# unique ip addresses

print("No. of Unique Ip Addresses :", data['ip'].nunique())

      

# unique user ids

print("No. of Unique User Ids :", data['user_id'].nunique())

      

# unique session id

print("No. of Unique Session Ids :", data['session'].nunique())
# describing the data



data.describe()
# checking if there is any NULL values in the dataset



data.isnull().sum().sum()
plt.rcParams['figure.figsize'] = (18, 7)



color = plt.cm.copper(np.linspace(0, 1, 40))

data['request'].value_counts().head(40).plot.bar(color = color)

plt.title('Most Popular Requests by the Users', fontsize = 20)
plt.rcParams['figure.figsize'] = (18, 7)



color = plt.cm.magma(np.linspace(0, 1, 40))

data['session'].value_counts().head(40).plot.bar(color = color)

plt.title('Most Popular Sessions used by the Users', fontsize = 20)
plt.rcParams['figure.figsize'] = (18, 7)



color = plt.cm.Wistia(np.linspace(0, 1, 40))

data['step'].value_counts().head(40).plot.bar(color = color)

plt.title('Most Popular step used by the Users', fontsize = 20)
# extracting some new features from date-time

data['date_time'] = data['date_time'].str.split(':',n = 1, expand = True)



data['date_time']
# converting the date_time into datetime format



data['date_time'] = pd.to_datetime(data['date_time'])

data['month'] = data['date_time'].dt.month

data['day'] = data['date_time'].dt.day
size = data['month'].value_counts()

color = plt.cm.rainbow(np.linspace(0, 1, 2))

labels = "April", "May"

explode = [0, 0.1]



plt.rcParams['figure.figsize'] = (9, 9)

plt.pie(size, colors = color, labels = labels, explode = explode, shadow = True)

plt.title('Distribution of Web Traffic Monthly Wise', fontsize = 20)

plt.tight_layout()

plt.legend()

plt.show()


plt.rcParams['figure.figsize'] = (18, 8)

sns.countplot(data['day'], palette = 'viridis')

plt.title('Distribution of Web Traffic Daily Basis', fontsize = 20)
# deleting the unnecasary columns



data = data.drop(['date_time'], axis = 1)



# label encoding the ip address and request

from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()

data['ip'] = le.fit_transform(data['ip'])

data['request'] = le.fit_transform(data['request'])



# looking at the columns of the dataset

data.columns
# data preparation



# steps vs Days

x = data.iloc[:, [2, 6]].values



from sklearn.cluster import KMeans



wcss = []

for i in range(1, 11):

  km = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

  km.fit(x)

  wcss.append(km.inertia_)

  

plt.plot(range(1, 11), wcss)

plt.title('The Elbow Method', fontsize = 20)

plt.xlabel('No. of Clusters')

plt.ylabel('wcss')

plt.show()
km = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

y_means = km.fit_predict(x)



plt.scatter(x[y_means == 0, 0], x[y_means == 0, 1], s = 100, c = 'pink', label = 'Customers at the Beginning of the month')

plt.scatter(x[y_means == 1, 0], x[y_means == 1, 1], s = 100, c = 'yellow', label = 'Customers at the Ending of the month')

plt.scatter(x[y_means == 2, 0], x[y_means == 2, 1], s = 100, c = 'cyan', label = 'Customers at the middle of the month')

plt.scatter(x[y_means == 3, 0], x[y_means == 3, 1], s = 100, c = 'magenta', label = 'Target Customers')

plt.scatter(x[y_means == 4, 0], x[y_means == 4, 1], s = 100, c = 'lightblue', label = 'Specific Customer')

plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 50, c = 'blue' , label = 'centeroid')



plt.title('Steps vs Days', fontsize = 20)

plt.ylabel('Days')

plt.xlabel('Steps')

plt.legend()

plt.show()
data.head()
# let's try to classify the users on basis of the month they requested access



y = data['month']



data = data.drop(['month'], axis = 1)

x = data



print("Shape of x: ", x.shape)

print("Shape of y: ", y.shape)

# splitting the data into train and test



from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)



print("Shape of x_train :", x_train.shape)

print("Shape of x_test :", x_test.shape)

print("Shape of y_train :", y_train.shape)

print("Shape of y_test :", y_test.shape)
# standardization



from sklearn.preprocessing import StandardScaler



sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.transform(x_test)
from sklearn.ensemble import RandomForestClassifier



model = RandomForestClassifier()

model.fit(x_train, y_train)

y_pred = model.predict(x_test)



print("Training Accuracy :", model.score(x_train, y_train))

print("Testing Accuracy :", model.score(x_test, y_test))



from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix



cm = confusion_matrix(y_test, y_pred)

plt.rcParams['figure.figsize'] = (5, 5)

sns.heatmap(cm, annot = True, cmap = 'Blues')



cr = classification_report(y_test, y_pred)

print(cr)
importance = model.feature_importances_

labels = x.columns

indices = np.argsort(importance)

color = plt.cm.spring(np.linspace(0, 1, 8))



plt.rcParams['figure.figsize'] = (15, 12)

plt.bar(range(len(importance)), importance[indices], color = color)

plt.grid()

plt.xticks(range(len(labels)), labels[indices])

plt.title('Feature Importance Plot', fontsize = 20)