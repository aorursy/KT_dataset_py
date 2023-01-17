import numpy as np 

import pandas as pd 
amazon= pd.read_csv(r"../input/amazon-movie-ratings/Amazon.csv")
amazon_pd = pd.DataFrame(amazon)
amazon.head()
amazon.shape
amazon.size
amazon.describe()
#maximum number of views 

amazon.describe().T["count"].sort_values(ascending = False)[0:6]
amazon.index
amazon.columns
Amazon_filtered = amazon.fillna(value=0)

Amazon_filtered


Amazon_filtered1 = Amazon_filtered.drop(columns='user_id')

Amazon_filtered1.head()
Amazon_filtered1.describe()
Amazon_max_views = Amazon_filtered1.sum()

Amazon_max_views
#finding maximum sum of ratings 

max(Amazon_max_views)


Amazon_max_views.head()

Amazon_max_views.tail()
Amazon_max_views.index
#finding which movie has maximum views/ratings

max_views= Amazon_max_views.argmax()

max_views
#checking whether that movie has max views/ratings or not 

Amazon_max_views['Movie127']
sum(Amazon_max_views)
len(Amazon_max_views.index)
#the average rating for each movie

Average_ratings_of_every_movie=sum(Amazon_max_views)/len(Amazon_max_views.index)

Average_ratings_of_every_movie
#the average rating for each movie (alternative way )

Amazon_max_views.mean()
Amazon_df = pd.DataFrame(Amazon_max_views)

Amazon_df.head()
Amazon_df.columns=['rating']
Amazon_df.index
Amazon_df.tail()
#top 5 movie ratings 

Amazon_df.nlargest(5,'rating')

#top 5 movies having least audience 

Amazon_df.nsmallest(5,'rating')
melt_df=amazon_pd.melt(id_vars= amazon.columns[0],value_vars=amazon.columns[1:],var_name='Movie',value_name='rating')
melt_df
melt_df.shape
melt_filtered = melt_df.fillna(0)

melt_filtered.shape
import surprise
from surprise import Reader

from surprise import Dataset

from surprise import SVD

from surprise.model_selection import train_test_split

reader = Reader(rating_scale=(-1,10))





data = Dataset.load_from_df(melt_df.fillna(0), reader=reader)

#Divide the data into training and test data

trainset, testset = train_test_split(data, test_size=0.25)
algo = SVD()
#Building a model

algo.fit(trainset)
#Make predictions on the test data

predict= algo.test(testset)
from surprise.model_selection import cross_validate
cross_validate(algo,data,measures=['RMSE','MAE'],cv=3,verbose=True)
user_id='A1CV1WROP5KTTW'

Movie='Movie6'

rating='5'

algo.predict(user_id,Movie,r_ui=rating)

print(cross_validate(algo,data,measures=['RMSE','MAE'],cv=3,verbose=True))