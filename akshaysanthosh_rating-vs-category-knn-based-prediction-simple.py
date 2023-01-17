# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import sklearn
from sklearn.neighbors import KNeighborsClassifier
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from sklearn.naive_bayes import GaussianNB
apps = pd.read_csv("../input/googleplaystore.csv")
apps.info()
apps.head()
apps.Rating.value_counts()
apps[apps.Rating==19.0]
# removing outliers

apps = apps.drop(apps.index[10472])
apps.iloc[10471:10478]
napp=apps.dropna()
napp.head()
napp.Genres.unique()

napp.Rating
c_list=list(napp.Category.unique())
print(c_list)
r_list=[napp.Installs]
print(r_list)
category_list = list(napp.Category.unique())
ratings = []

for category in category_list:
    x = napp[apps.Category == category]
    rating_rate = x.Rating.sum()/len(x)
    ratings.append(rating_rate)
data = pd.DataFrame({'Category':category_list, 'Rating':ratings})
new_index = (data['Rating'].sort_values(ascending=False)).index.values
sorted_data = data.reindex(new_index)

sorted_data

x=category_list
plt.figure(figsize=(25,15))
sns.barplot(x=sorted_data.Category, y=sorted_data.Rating)

plt.xticks(rotation = 45)
plt.xlabel('Application Category')
plt.ylabel('Ratings')
plt.title('Average Ratings by Category')
plt.show()
from sklearn.neighbors import KNeighborsClassifier

from sklearn import preprocessing

 

##Various machine learning algorithms require numerical input data,
##so you need to represent categorical columns in a numerical column.
#LabelEncoder() function is used 

#label encoding 
y=ratings

le = preprocessing.LabelEncoder()

y_encoded=le.fit_transform(x)

print(y_encoded)
#lable encoding 

le = preprocessing.LabelEncoder()

x_encoded=le.fit_transform(x)

xaa=x_encoded.reshape(-1, 1)

print(xaa)


#traning 
result = KNeighborsClassifier(n_neighbors=3)
result.fit(xaa,y_encoded)
#testing 
predicted= result.predict(2)
print(predicted)

""" Import knearest neighbors Classifier model
from sklearn.neighbors import KNeighborsClassifier

#Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)

#Train the model using the training sets
knn.fit(xaa,y_encoded)

#Predict the response for test dataset
y_pred = knn.predict(xaa)*/ # put the label from test data set here """

""" #Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics - for testing accuracy 


print("Accuracy:",metrics.accuracy_score(y_test, y_pred)) 


"""