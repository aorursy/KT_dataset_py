import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.model_selection import train_test_split

from sklearn.cluster import KMeans
def import_data():

    primary = pd.read_csv('../input/ratings.csv')

    secondary = pd.read_csv('../input/books.csv')

    return primary,secondary
primary_df,secondary_df = import_data()

primary_df.head()
corr_primary = primary_df.corr()

corr_primary
sns.heatmap(corr_primary)
secondary_df.head()
corr_secondary = secondary_df.corr()

corr_secondary
sns.heatmap(corr_secondary)
def split():

    train, test = train_test_split(primary_df, test_size=0.2, random_state=42)

    X = train.drop('rating',axis=1) ## select everything except the rating

    y = train[['rating']]    ## select just the rating

    

    ## train test split

    

    ## 80 for train

    ## 20 for test

    

    return X,y,test
X,y,test  = split()
X.head()
y.head()
print(X.shape)

print(y.shape)
n_user = len(primary_df.user_id.unique())

n_books = len(primary_df.book_id.unique())
print("NUMBER OF UNIQUE USER "+str(n_user))

print("NUMBER OF UNIQUE BOOKS "+str(n_books))
kmeans = KMeans(n_clusters=5)
kmeans.fit(X,y)
predictions = kmeans.predict(test[['book_id','user_id']])
from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_squared_log_error
import math

def evaluation():

    mse = mean_squared_error(np.array(test['rating']),predictions)

    rmse = math.sqrt(mean_squared_error(np.array(test['rating']),predictions))    

    mae = mean_absolute_error(np.array(test['rating']),predictions)    

    msle = mean_squared_log_error(np.array(test['rating']),predictions) 

    return mse,rmse,mae,msle
mse,rmse,mae,msle = evaluation()
print ("MSE LOSS  "+str(mse))

print ("RMSE LOSS "+str(rmse))

print ("MAE LOSS  "+str(mae))

print ("MLSE LOSS "+str(msle))
## create a dataframe for user selected on the database

import random



user_list=primary_df.user_id.unique()

random_user=random.choice(user_list)
user_id = np.array([random_user for i in range(len(book_data))])
user_id
book_data = list(np.array(list(set(primary_df.book_id))))
book_data
data = {'book_id':book_data ,'user_id':user_id}
df_test = pd.DataFrame(data)
predict=kmeans.predict(df_test)
print(predict)
recommended_book_ids