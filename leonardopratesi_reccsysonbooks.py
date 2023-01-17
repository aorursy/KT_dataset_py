# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



import surprise

ratings=pd.read_csv('/kaggle/input/goodbooks-10k/ratings.csv')

ratings.drop_duplicates(inplace=True)

print('we have',ratings.shape[0], 'ratings')

print('the number of unique users :', len(ratings.user_id.unique()))

print('the number of unique books :', len(ratings.book_id.unique()))

print("mean of user rated books.", ratings.user_id.value_counts().mean())



print(ratings)
showrat = ratings.groupby('book_id')['rating'].count()



hist = plt.hist(showrat.values, bins=100)

plt.xlabel('Number of Ratings Per Book')

#plt.gca().invert_xaxis()

plt.show()
from sklearn.model_selection import train_test_split



#swapping columns

ratings=ratings[['user_id','book_id','rating']] 



Train,Test = train_test_split(ratings, test_size=0.25 )

reader = surprise.Reader(rating_scale=(1,5)) 

#into surprise:

train = surprise.Dataset.load_from_df(Train, reader)

test = surprise.Dataset.load_from_df(Test,reader)

kSplit = surprise.model_selection.split.KFold(n_splits=10, shuffle=True) # split data into folds. 



sim_options = sim_options = {'name': 'cosine',

               'user_based': False  # compute  similarities between items

               }

collabKNN = surprise.KNNBasic(k=40,sim_options=sim_options)

rmseKNN = []

for trainset, testset in kSplit.split(train): #iterate through the folds.

    collabKNN.fit(trainset)

    predictionsKNN = collabKNN.test(testset)

    rmseKNN.append(surprise.accuracy.rmse(predictionsKNN,verbose=True))#get root means squared error

    
#need dataframe to show data

df = pd.DataFrame(predictionsKNN, columns=['uid', 'iid', 'rui', 'est', 'details'])  

#add error column

df['err'] = abs(df.est - df.rui)

best_predictions = df.sort_values(by='err')[:10]

print(best_predictions)

#i implement slopeOne algorithm

slopeOne = surprise.prediction_algorithms.slope_one.SlopeOne()

rmseSlope = []

for trainset, testset in kSplit.split(train): #iterate through the folds.

    slopeOne.fit(trainset)

    predictionsSlope = slopeOne.test(testset)

    rmseSlope.append(surprise.accuracy.rmse(predictionsSlope,verbose=True))#get root means squared error
df = pd.DataFrame(predictionsKNN, columns=['uid', 'iid', 'rui', 'est', 'details'])  

#add error column

df['err'] = abs(df.est - df.rui)

best_predictions = df.sort_values(by='err')[:10]

print(best_predictions)