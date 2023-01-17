# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

from surprise import KNNWithMeans,Reader,Dataset,accuracy

from surprise.model_selection import train_test_split

data = pd.read_csv('../input/ratings.csv')
data.head(),data.shape
reader = Reader(rating_scale = (1,5))

data_surp = Dataset.load_from_df(data,reader)
[traindata, testdata] = train_test_split(data_surp,test_size=0.2,shuffle=True)
#KNN

recommender = KNNWithMeans(k = 50,sim_options = {'name':'cosine','user_based':True})
#FIt the Model

recommender.fit(traindata)
test_predict = recommender.test(testdata)
test_predict
RMSE = accuracy.rmse(test_predict)

RMSE
recommender.predict(1,35)
#user1 =data[data['userId']==1]

user1.head()
#li = []

for i in user1['movieId']:

    for j in data['movieId']:

        if j != i:

            li = li.append(j)

        

             
user_recommend = 1

movies_ids = data['movieId'].unique()

pred_rate = pd.DataFrame(columns=['userId','movieId','rating'])



for mid in movies_ids:

    user_mid = data[(data['userId']==user_recommend) & 

                    (data['movieId']==mid)]

    is_rated = True if len(user_mid)>0 else False

    

    if not is_rated:

        curr = {'userId':user_recommend,

                'movieId':mid,

                'rating':recommender.predict(user_recommend,mid)[3]}

        pred_rate = pred_rate.append(curr,ignore_index=True)

top_recommend = pred_rate.sort_values('rating',ascending=False).head(25)
top_recommend.head()