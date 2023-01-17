# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from sklearn.preprocessing import Imputer 



import warnings

warnings.filterwarnings('ignore')



import matplotlib.pyplot as plt

%matplotlib inline

from matplotlib import pylab, mlab, pyplot

plt = pyplot

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
movies = pd.read_csv('../input/tmdb_5000_movies.csv')

credits = pd.read_csv('../input/tmdb_5000_credits.csv')



del credits['title']

df = pd.concat([movies, credits], axis=1)



newCols = ['id','title','release_date','popularity','vote_count',

           'budget','revenue','genres','keywords','cast','crew','tagline', 'runtime', 'production_companies', 

           'production_countries', 'status','vote_average']



df2 = df[newCols]



my_imputer = Imputer()



temp=df2

X2 = my_imputer.fit_transform(df2[['runtime']])

df2['runtime'] = X2



temp = df2['genres']

df2['genres']

for j in range(len(df2['genres'])):

    if "Science Fiction" in df2['genres'][j]:

        df2['genres'][j]=df2['genres'][j].replace("Science Fiction", "Sciencefiction")



for j in range(len(df2['genres'])):

    if "TV Movie" in df2['genres'][j]:

        df2['genres'][j]=df2['genres'][j].replace("TV Movie", "TVMovie")

    

genreArray = []*len(df2['genres'])

for j in range(len(df2['genres'])):

    string = df2['genres'][j]

    newArray = []*10

    for i in [1,5,9,13]:

         if i < len(string.split()):

              newArray.append(string.split()[i][:-1])

    genreArray.append(newArray)

    

df2['num_genres']=genreArray

test = map(int, df2['num_genres'])



def find_in_sublists(lst, value):

    for sub_i, sublist in enumerate(lst):

        try:

            return (sub_i, sublist.index(value))

        except ValueError:

            pass



    raise ValueError('%s is not in lists' % value)



temp = df2['num_genres']

test = []*len(temp)

for i in range(len(temp)):

    test.append(list(map(int, temp[i])))

    

df2['genreArray_Int'] = test





df2.head(10)
votes_roundedup = rounded_up = np.ceil(df2['vote_average'])

list(map(int,rounded_up))



df2['votes_roundedup']=votes_roundedup



df2.head()

df2.info()
newCols1 = ['id','title','release_date','popularity','vote_count',

           'budget','revenue','genres','keywords','cast','crew','tagline', 'runtime', 'production_companies', 

           'production_countries', 'status', 'num_genres', 'genreArray_Int','vote_average', 'votes_roundedup']

df3 = df2[newCols1]
df3.head()
df3.info()
david = df3['genreArray_Int'][1]



david[1]


from scipy import special, optimize

from pylab import *

import matplotlib



x= df3['vote_average']

y= df3['budget']

pylab.xlabel('cost')

pylab.ylabel('signups')

pylab.plot(x, y, 'o')

pylab.plot(x, y, 'k-')

pylab.show()

sns.regplot(df2['vote_average'],df2['budget'])



# df['profit'] = df['revenue'] - df['budget']

# df['succes'] = df['profit'] >= 0

#for i in range(10):

    #if df.loc[i,['vote_average']] < i and df.loc[i,'vote_average'] >= (i-1):

     #   df['succes2'] = i 

# print(df.head())



from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn import linear_model

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error





from sklearn.metrics import mean_squared_error, r2_score



vote_average = ['vote_average']

training = df[vote_average]



budget= ['budget']

target = df['budget']



X = training.values

y = target.values



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)



# Create linear regression object

regr = linear_model.LinearRegression()



# Train the model using the training sets

regr.fit(X_train, y_train)



# Make predictions using the testing set

y_pred_lr = regr.predict(X_test)



vis = regr.predict(100000000)

#print(vis)



print(regr.coef_, regr.intercept_)

print(r2_score(y_test, y_pred_lr))



f = plt.figure(figsize=(10,5))

plt.scatter(X_test[:,0], y_test, label="Real score");

plt.scatter(X_test[:,0], y_pred_lr, c='r',label="Predicted score");

plt.xlabel("budget");

plt.ylabel('vote_average');

plt.legend(loc=2);



from sklearn.ensemble import RandomForestRegressor

# Create linear regression object

rf = RandomForestRegressor(1)



# Train the model using the training sets

rf.fit(X_train, y_train)



# Make predictions using the testing set

y_pred_rf = rf.predict(X_test)

#print(y_pred_rf)

f = plt.figure(figsize=(10,5))

plt.scatter(X_test[:,0], y_test, s=50,label="Real Price");

plt.scatter(X_test[:,0], y_pred_rf,s=100, c='r',label="Predited Price");

plt.xlabel("budget");

plt.ylabel("vote_average");

plt.legend(loc=2);



hoi = rf.predict(100000000)

#print(hoi)



from sklearn.metrics import mean_squared_error



error_lr = mean_squared_error(y_test,y_pred_lr)

error_rf = mean_squared_error(y_test,y_pred_rf)



#print(error_lr)

#print(error_rf)



from sklearn.metrics import mean_squared_error, r2_score



vote_count = ['vote_count']

training = df[vote_count]



revenue= ['revenue']

target = df['revenue']



X = training.values

y = target.values



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)



# Create linear regression object

regr = linear_model.LinearRegression()



# Train the model using the training sets

regr.fit(X_train, y_train)



# Make predictions using the testing set

y_pred_lr = regr.predict(X_test)



vis = regr.predict(100000000)

#print(vis)



print(regr.coef_, regr.intercept_)

print(r2_score(y_test, y_pred_lr))



f = plt.figure(figsize=(10,5))

plt.scatter(X_test[:,0], y_test, label="Real score");

plt.scatter(X_test[:,0], y_pred_lr, c='r',label="Predicted score");

plt.xlabel("revenue");

plt.ylabel('vote_count');

plt.legend(loc=2);



from sklearn.ensemble import RandomForestRegressor

# Create linear regression object

rf = RandomForestRegressor(1)



# Train the model using the training sets

rf.fit(X_train, y_train)



# Make predictions using the testing set

y_pred_rf = rf.predict(X_test)

#print(y_pred_rf)

f = plt.figure(figsize=(10,5))

plt.scatter(X_test[:,0], y_test, s=50,label="Real Price");

plt.scatter(X_test[:,0], y_pred_rf,s=100, c='r',label="Predited Price");

plt.xlabel("revenue");

plt.ylabel("vote_count");

plt.legend(loc=2);



hoi = rf.predict(100000000)

#print(hoi)



from sklearn.metrics import mean_squared_error



error_lr = mean_squared_error(y_test,y_pred_lr)

error_rf = mean_squared_error(y_test,y_pred_rf)



#print(error_lr)

#print(error_rf)

from sklearn.metrics import mean_squared_error, r2_score



vote_average = ['vote_average']

training = df[vote_average]



revenue= ['revenue']

target = df['revenue']



X = training.values

y = target.values



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)



# Create linear regression object

regr = linear_model.LinearRegression()



# Train the model using the training sets

regr.fit(X_train, y_train)



# Make predictions using the testing set

y_pred_lr = regr.predict(X_test)



vis = regr.predict(100000000)

#print(vis)



print(regr.coef_, regr.intercept_)

print(r2_score(y_test, y_pred_lr))



f = plt.figure(figsize=(10,5))

plt.scatter(X_test[:,0], y_test, label="Real score");

plt.scatter(X_test[:,0], y_pred_lr, c='r',label="Predicted score");

plt.xlabel("revenue");

plt.ylabel('vote_average');

plt.legend(loc=2);



from sklearn.ensemble import RandomForestRegressor

# Create linear regression object

rf = RandomForestRegressor(1)



# Train the model using the training sets

rf.fit(X_train, y_train)



# Make predictions using the testing set

y_pred_rf = rf.predict(X_test)

#print(y_pred_rf)

f = plt.figure(figsize=(10,5))

plt.scatter(X_test[:,0], y_test, s=50,label="Real Price");

plt.scatter(X_test[:,0], y_pred_rf,s=100, c='r',label="Predited Price");

plt.xlabel("revenue");

plt.ylabel("vote_average");

plt.legend(loc=2);



hoi = rf.predict(100000000)

#print(hoi)



from sklearn.metrics import mean_squared_error



error_lr = mean_squared_error(y_test,y_pred_lr)

error_rf = mean_squared_error(y_test,y_pred_rf)



#print(error_lr)

#print(error_rf)

from sklearn.metrics import mean_squared_error, r2_score



vote_average = ['vote_average']

training = df[vote_average]



vote_count= ['vote_count']

target = df['vote_count']



X = training.values

y = target.values



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)



# Create linear regression object

regr = linear_model.LinearRegression()



# Train the model using the training sets

regr.fit(X_train, y_train)



# Make predictions using the testing set

y_pred_lr = regr.predict(X_test)



vis = regr.predict(100000000)

#print(vis)



print(regr.coef_, regr.intercept_)

print(r2_score(y_test, y_pred_lr))



f = plt.figure(figsize=(10,5))

plt.scatter(X_test[:,0], y_test, label="Real score");

plt.scatter(X_test[:,0], y_pred_lr, c='r',label="Predicted score");

plt.xlabel("vote_count");

plt.ylabel('vote_average');

plt.legend(loc=2);



from sklearn.ensemble import RandomForestRegressor

# Create linear regression object

rf = RandomForestRegressor(1)



# Train the model using the training sets

rf.fit(X_train, y_train)



# Make predictions using the testing set

y_pred_rf = rf.predict(X_test)

#print(y_pred_rf)

f = plt.figure(figsize=(10,5))

plt.scatter(X_test[:,0], y_test, s=50,label="Real Price");

plt.scatter(X_test[:,0], y_pred_rf,s=100, c='r',label="Predited Price");

plt.xlabel("vote_count");

plt.ylabel("vote_average");

plt.legend(loc=2);



hoi = rf.predict(100000000)

#print(hoi)



from sklearn.metrics import mean_squared_error



error_lr = mean_squared_error(y_test,y_pred_lr)

error_rf = mean_squared_error(y_test,y_pred_rf)



#print(error_lr)

#print(error_rf)

# df['profit'] = df['revenue'] - df['budget']

# df['succes'] = df['profit'] >= 0

#for i in range(10):

    #if df.loc[i,['vote_average']] < i and df.loc[i,'vote_average'] >= (i-1):

     #   df['succes2'] = i 

# print(df.head())



from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn import linear_model

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error





from sklearn.metrics import mean_squared_error, r2_score



vote_count = ['vote_count']

training = df[vote_count]



budget= ['budget']

target = df['budget']



X = training.values

y = target.values



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)



# Create linear regression object

regr = linear_model.LinearRegression()



# Train the model using the training sets

regr.fit(X_train, y_train)



# Make predictions using the testing set

y_pred_lr = regr.predict(X_test)



vis = regr.predict(100000000)

#print(vis)



print(regr.coef_, regr.intercept_)

print(r2_score(y_test, y_pred_lr))



f = plt.figure(figsize=(10,5))

plt.scatter(X_test[:,0], y_test, label="Real score");

plt.scatter(X_test[:,0], y_pred_lr, c='r',label="Predicted score");

plt.xlabel("budget");

plt.ylabel('vote_count');

plt.legend(loc=2);



from sklearn.ensemble import RandomForestRegressor

# Create linear regression object

rf = RandomForestRegressor(1)



# Train the model using the training sets

rf.fit(X_train, y_train)



# Make predictions using the testing set

y_pred_rf = rf.predict(X_test)

#print(y_pred_rf)

f = plt.figure(figsize=(10,5))

plt.scatter(X_test[:,0], y_test, s=50,label="Real Price");

plt.scatter(X_test[:,0], y_pred_rf,s=100, c='r',label="Predited Price");

plt.xlabel("budget");

plt.ylabel("vote_count");

plt.legend(loc=2);



hoi = rf.predict(100000000)

#print(hoi)



from sklearn.metrics import mean_squared_error



error_lr = mean_squared_error(y_test,y_pred_lr)

error_rf = mean_squared_error(y_test,y_pred_rf)



#print(error_lr)

#print(error_rf)

from sklearn.metrics import mean_squared_error, r2_score



budget = ['budget']

training = df[budget]



revenue= ['revenue']

target = df['revenue']



X = training.values

y = target.values



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)



# Create linear regression object

regr = linear_model.LinearRegression()



# Train the model using the training sets

regr.fit(X_train, y_train)



# Make predictions using the testing set

y_pred_lr = regr.predict(X_test)



vis = regr.predict(100000000)

#print(vis)



print(regr.coef_, regr.intercept_)

print(r2_score(y_test, y_pred_lr))



f = plt.figure(figsize=(10,5))

plt.scatter(X_test[:,0], y_test, label="Real score");

plt.scatter(X_test[:,0], y_pred_lr, c='r',label="Predicted score");

plt.xlabel("budget");

plt.ylabel('revenue');

plt.legend(loc=2);



from sklearn.ensemble import RandomForestRegressor

# Create linear regression object

rf = RandomForestRegressor(1)



# Train the model using the training sets

rf.fit(X_train, y_train)



# Make predictions using the testing set

y_pred_rf = rf.predict(X_test)

#print(y_pred_rf)

f = plt.figure(figsize=(10,5))

plt.scatter(X_test[:,0], y_test, s=50,label="Real Price");

plt.scatter(X_test[:,0], y_pred_rf,s=100, c='r',label="Predited Price");

plt.xlabel("budget");

plt.ylabel("revenue");

plt.legend(loc=2);



hoi = rf.predict(100000000)

#print(hoi)



from sklearn.metrics import mean_squared_error



error_lr = mean_squared_error(y_test,y_pred_lr)

error_rf = mean_squared_error(y_test,y_pred_rf)



#print(error_lr)

#print(error_rf)
