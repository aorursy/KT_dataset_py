# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.


import numpy as np 



import pandas as pd

import matplotlib.pyplot as plt







from sklearn.impute import SimpleImputer 

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.model_selection import train_test_split 

from sklearn.preprocessing import StandardScaler 

import sqlite3

con = sqlite3.connect("/kaggle/input/twitter-airline-sentiment/database.sqlite")



data = pd.read_sql_query("SELECT * FROM Tweets", con)
data


# Feature Selection

features = ['tweet_id','airline_sentiment','airline','retweet_count','tweet_location']



data = data[features]



from sklearn import preprocessing 

  

# label_encoder object knows how to understand word labels. 

label_encoder = preprocessing.LabelEncoder() 

  

# Encode labels in column 'species'. 

data['airline']= label_encoder.fit_transform(data['airline']) 

data['airline_sentiment']= label_encoder.fit_transform(data['airline_sentiment']) 

data['tweet_location']= label_encoder.fit_transform(data['tweet_location'])

data
X = data.iloc[:, [1,3]].values

# attributes to determine dependent variable / Class

Y = data.iloc[:, 2].values

# encode categorical data
data.isnull().sum()
data
# Calculate the standard error of  the mean of all the rows in dataframe 

data.sem(axis = 1, skipna = False) 
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)
y_pred
from matplotlib.colors import ListedColormap



X_set, y_set = X_train, Y_train

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),

                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))



plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),

             alpha = 0.75, cmap = ListedColormap(('blue', 'yellow')))



plt.xlim(X1.min(), X1.max())



plt.ylim(X2.min(), X2.max())



for i, j in enumerate(np.unique(y_set)):

    

    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],

    c = ListedColormap(('blue', 'yellow'))(i), label = j)



plt.title('K-NN (Training set)')



plt.xlabel('retweet count')



plt.ylabel('tweet_location ')



plt.legend()



plt.show()
from matplotlib.colors import ListedColormap



X_set, y_set = X_test, Y_test

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),

                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))



plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),

 alpha = 0.75, cmap = ListedColormap(('red', 'green')))



plt.xlim(X1.min(), X1.max())



plt.ylim(X2.min(), X2.max())



for i, j in enumerate(np.unique(y_set)):

    

    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],

    c = ListedColormap(('red', 'green'))(i), label = j)



plt.title('K-NN (Test set)')



plt.xlabel('retweet_count')



plt.ylabel('tweeta_location')



plt.legend()



plt.show()
from sklearn.neighbors import KNeighborsClassifier



KNN = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)



KNN.fit(X_train, Y_train)



Y_pred = KNN.predict(X_test)
Y_pred
from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score 

from sklearn.metrics import classification_report 



cm = confusion_matrix(Y_test, Y_pred)





print('confusion_matrics' ,cm) 

print ('Accuracy Score :',accuracy_score(Y_test, Y_pred))

print ('Report : ')

print (classification_report(Y_test, Y_pred) )
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, Y_train)

# Predicting the Test set results

y_pred = regressor.predict(X_test)
y_pred

from sklearn.svm import SVC



classifier = SVC(kernel = 'rbf', random_state = 0)



classifier.fit(X_train, Y_train)


y_pred = classifier.predict(X_test)


y_pred
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)
y_pred
from matplotlib.colors import ListedColormap



X_set, y_set = X_train, Y_train

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),

                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))



plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),

             alpha = 0.75, cmap = ListedColormap(('blue', 'yellow')))



plt.xlim(X1.min(), X1.max())



plt.ylim(X2.min(), X2.max())



for i, j in enumerate(np.unique(y_set)):

    

    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],

    c = ListedColormap(('blue', 'yellow'))(i), label = j)



plt.title('svm(Training set)')



plt.xlabel('retweet count')



plt.ylabel('tweet_location ')



plt.legend()



plt.show()
from matplotlib.colors import ListedColormap



X_set, y_set = X_test, Y_test

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),

                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))



plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),

 alpha = 0.75, cmap = ListedColormap(('red', 'green')))



plt.xlim(X1.min(), X1.max())



plt.ylim(X2.min(), X2.max())



for i, j in enumerate(np.unique(y_set)):

    

    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],

    c = ListedColormap(('red', 'green'))(i), label = j)



plt.title('svm (Test set)')



plt.xlabel('retweet_count')



plt.ylabel('tweeta_location')



plt.legend()



plt.show()