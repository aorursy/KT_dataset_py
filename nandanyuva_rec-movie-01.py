import pandas as pd

import numpy as np
data=pd.read_csv('../input/ratings.csv')
from surprise import KNNWithMeans #like KNN but taylor made for recommendation engine

from surprise import NMF # one more reccomendation classification algo  famously used by Netflix 

from surprise import Reader # Like Pd - pandas

from surprise.model_selection import train_test_split

from surprise import Dataset

from surprise import accuracy

#suprise package we need 
reader=Reader(rating_scale=(1,5)) #rating range #need by suprise package 

data_surp=Dataset.load_from_df(data,reader)
data_surp
traindata,testdata = train_test_split(data_surp,test_size=0.2,shuffle=True)
recom_mod=KNNWithMeans(K=50,sim_options={'name':'cosine','user_based':True})

# cosine dot product #50 Trees
recom_mod.fit(traindata)
recom_mod.fit(data_surp.build_full_trainset())

#with suprise package we are training our model with the whole dataset
test_pred=recom_mod.test(testdata)
RMSE = accuracy.rmse(test_pred)

#+-(RMSE) 
recom_mod.predict(1,34) 

#user 1 for the movie id 34 , he may rate 2.10 so dont recommend
user_recommend= 15 #suggest movie for the user 15

movies_ids = data['movieId'].unique()

pred_rate = pd.DataFrame(columns=['userId','movieId','rating'])



for mid in movies_ids:

    user_mid = data[(data['userId']==user_recommend) & (data['movieId']==mid)]

    is_rated = True if len(user_mid)>0 else False

    

    if not is_rated:

        curr = {'userId': user_recommend,

               'movieId': mid,

               'rating': recom_mod.predict(user_recommend,mid)[3]}

        pred_rate = pred_rate.append(curr,ignore_index=True)

top_recom=pred_rate.sort_values('rating', ascending=False).head(25)



# for the user 15 we are extrating movies which he dint watched # predicting the ratings he might give

                                                                
top_recom.reset_index() # wov you got it
from surprise import SVD

from surprise import Dataset

from surprise.model_selection import cross_validate
#load the, movielens - 100k dataset (dwonlaod if if needed)

data=Dataset.load_from_df(data,reader)
#well use the famous SVD algorithm

algo = SVD()

nalgo= NMF()
#RUN 5 -FOLD CROS-VALIDATION AND PRINT RESULTS

cross_validate(algo,data,measures=['RMSE','MAE'], cv=5,verbose=True)
#RUN 5 -FOLD CROS-VALIDATION AND PRINT RESULTS

cross_validate(nalgo,data,measures=['RMSE','MAE'], cv=5,verbose=True)