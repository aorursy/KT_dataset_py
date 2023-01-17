#import "Movie Lens Small Latest Dataset" dataset



import pandas as pd

import numpy as np

import warnings; warnings.simplefilter('ignore')

import matplotlib.pyplot as plt

import seaborn as sns



links= pd.read_csv('../input/movie-lens-small-latest-dataset/links.csv')

movies=pd.read_csv('../input/movie-lens-small-latest-dataset/movies.csv')

ratings=pd.read_csv('../input/movie-lens-small-latest-dataset/ratings.csv')

tags=pd.read_csv('../input/movie-lens-small-latest-dataset/tags.csv')



dataset=movies.merge(ratings,on='movieId').merge(tags,on='movieId').merge(links,on='movieId')

dataset.head()
to_drop=['title','genres','timestamp_x','timestamp_y','userId_y','imdbId','tmdbId']



dataset.drop(columns=to_drop,inplace=True)

print(dataset.describe())

dataset.head()
dataset=pd.get_dummies(dataset) #encode the catergorical data



print(dataset)

dataset.isnull().sum()# check the number of missing data cells
#Divide data into test,train and validation

#train dataset

train_dataset = dataset.sample(frac=0.9,random_state=0) #90% of the dataset

test_dataset=dataset.drop(train_dataset.index) #10% of the Dataset





#seperate labels

train_labels = train_dataset.pop('rating')

test_labels = test_dataset.pop('rating')
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor

from sklearn import metrics



#scale the features

sc = StandardScaler()

train_dataset = sc.fit_transform(train_dataset)

test_dataset = sc.fit_transform(test_dataset)



#train model

regressor = RandomForestRegressor(n_estimators=10, random_state=0)

regressor.fit(train_dataset, train_labels)



#predict ratings on test data

predicted_labels = regressor.predict(test_dataset)

def mean_absolute_percentage_error(y_true, y_pred): 

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100



actual_labels=np.array(test_labels)



print('Mean Absolute Error:', metrics.mean_absolute_error(actual_labels, predicted_labels))

print('Mean Squared Error:', metrics.mean_squared_error(actual_labels, predicted_labels))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(actual_labels, predicted_labels)))

print('Mean Average Percentage Error: ',mean_absolute_percentage_error(actual_labels,predicted_labels))

print('Size of Test Labels',actual_labels.size)

print('Size of Predicted Labels',predicted_labels.size)



#create a new dataframe

predicted_movies=pd.DataFrame({'Actual':actual_labels,'Predicted':predicted_labels}).reset_index()

#print Table of Test Dataset

predicted_movies.head(20)

difference=actual_labels-predicted_labels

plt.hist(difference,normed=True,color='orange',bins=15,alpha=0.8)
sns.set(style="darkgrid")

mapping=predicted_movies['Actual'][:50].plot(kind='line',x='Datetime',y='Volume BTC',color='blue',label='Actual')

predicted_movies['Predicted'][:50].plot(kind='line',x='Datetime',y='Volume BTC',color='orange',label='Predicted',ax=mapping)

mapping.set_xlabel('Users')

mapping.set_ylabel('Ratings')

mapping.set_title('Regression Graph show Actual vs Predicted ratings')

mapping=plt.gcf()

mapping.set_size_inches(20,12)
sns.jointplot(predicted_movies['Actual'],predicted_movies['Predicted'],predicted_movies,kind='kde')