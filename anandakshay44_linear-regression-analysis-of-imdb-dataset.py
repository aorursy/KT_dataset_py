import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.linear_model import LogisticRegression # to apply the Logistic regression

from sklearn.model_selection import train_test_split # to split the data into two parts
file = '../input/movie_metadata.csv'

df = pd.read_csv(file)

df.head()
df.isnull().any()
#Replace every nan values with 0

df.fillna(value=0,axis=1,inplace=True)

df.shape
sns.heatmap(df.corr(), vmax=1, square=True)
#Defining features and target for this dataset based on co-relation

features = ['actor_3_facebook_likes', 'actor_1_facebook_likes', 'gross',

       'num_voted_users', 'cast_total_facebook_likes', 'facenumber_in_poster',

       'num_user_for_reviews', 'budget', 'title_year',

       'actor_2_facebook_likes', 'aspect_ratio',

       'movie_facebook_likes']

target = ['imdb_score']
#splitting data set into training and test data set in 0.7/0.3

train, test = train_test_split(df,test_size=0.30)

train.head()
#Fill the training and test data with require information

X_train = train[features].dropna()

y_train = train[target].dropna()

X_test = test[features].dropna()

y_test = test[target].dropna()
from sklearn import linear_model# compute classification accuracy for the linear regression model

from sklearn import metrics # for the check the error and accuracy of the model

lin = linear_model.LinearRegression()

# train the model on the training set

lin.fit(X_train, y_train)
lin_score_train = lin.score(X_test, y_test)

lin_score_test = lin.score(X_train, y_train)
print("Training score: ",lin_score_train)

print("Testing score: ",lin_score_test)
from sklearn import neighbors

n_neighbors=5

knn=neighbors.KNeighborsRegressor(n_neighbors,weights='uniform')

knn.fit(X_train, y_train)
knn_score_train = knn.score(X_test, y_test)

knn_score_test = knn.score(X_train, y_train)



print("Training score: ",knn_score_train)

print("Testing score: ",knn_score_test)