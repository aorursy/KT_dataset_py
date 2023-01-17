# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

os.chdir("../input")

print(os.getcwd())



# Any results you write to the current directory are saved as output.
# training data

train_data = pd.read_csv("train.csv")

train_columns = train_data.columns

#print(train_columns)

# testing data

test_X  = pd.read_csv("test.csv")

test_cols = test_X.columns

print(test_X.head())

print(test_cols)

dummy  = pd.read_csv("dummy_submission.csv")

os.chdir("../working")

#dummy.to_csv("dummy.csv",index = False)
#X_train[['userId','movieId']]

X_train_tot = np.concatenate((train_data.userId[:,None],train_data.movieId[:,None]),axis = 1)

Y_train_tot = np.array(train_data.rating)

mu = np.mean(Y_train_tot)

# test data

X_test = np.concatenate((test_X.userId[:,None],test_X.movieId[:,None]),axis = 1)

Y_test = mu*np.ones((len(X_test)))



# splitting into train and validation data 

l = len(X_train_tot)

X_train = X_train_tot[:int(0.8*l)]

Y_train = Y_train_tot[:int(0.8*l)]

X_valid = X_train_tot[int(0.8*l):]

Y_valid = Y_train_tot[int(0.8*l):]





#print(dummy)
# model 1 =====>>>>> rui = bu + bi + mu <<<<=========

b_users = np.zeros(10000);

b_movies = np.zeros(10000);

Y_test = np.zeros(len(X_test));

learning_rate = 0.01;

no_iterations = 5;

for iter in range(no_iterations):

    for i in range(int(0.8*l)):

        user  = X_train[i][0]

        movie = X_train[i][1]

        d_term = Y_train[i] - mu - b_users[user] - b_movies[movie]

        b_users[user]+= learning_rate*d_term

        b_movies[movie]+= learning_rate*d_term

        

# for testing

for i in range(len(X_test)):

    user = X_test[i][0];

    movie = X_test[i][1];

    if(user > 9999 and movie > 9999):

        Y_test[i] = mu;

    elif(user > 9999):

        Y_test[i] = mu + b_movies[movie];

        if(Y_test[i] > 5):

            Y_test[i] = 5

        if(Y_test[i] < 0):

            Y_test[i] = 0

    elif(movie > 9999):

        Y_test[i] = mu + b_users[user];

        if(Y_test[i] > 5):

            Y_test[i] = 5

        if(Y_test[i] < 0):

            Y_test[i] = 0

    else:

        Y_test[i] = mu + b_movies[movie] + b_users[user]

        if(Y_test[i] > 5):

            Y_test[i] = 5

        if(Y_test[i] < 0):

            Y_test[i] = 0

        

        

        

        
print(Y_test)

dummy['Prediction'] = Y_test

dummy.to_csv("dummy.csv",index = False)