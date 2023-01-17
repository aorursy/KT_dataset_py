# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



import matplotlib.pyplot as plt

%matplotlib inline



import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
movies = pd.read_csv('../input/tmdb_5000_movies.csv')

credits = pd.read_csv('../input/tmdb_5000_credits.csv')



movies.head()

#list(movies)
movies.loc[:, movies.isnull().any()].head()

#movies.info()
movies.describe().round()
del credits['title']

df = pd.concat([movies, credits], axis=1)
pd.set_option('display.max_columns', None)



df.head()

df.loc[:, df.isnull().any()].head()
df.info()
df.describe().round()
list(df)



newCols = ['id','title','release_date','popularity','vote_average','vote_count',

           'budget','revenue','genres','keywords','cast','crew','tagline', 'runtime', 'production_companies', 

           'production_countries', 'status']



df2 = df[newCols]



#tagline, runtime, production_companies, production_countries, status,
df2.head()
df2.describe().round()
#If we normalize both columns, this graph might get nicer



for i in ['vote_average','vote_count','budget','revenue']:

    for j in ['vote_average','vote_count','budget','revenue']:

        df2.plot.scatter(i,j)

    
df2.plot.scatter('vote_average','vote_count')

df2.plot.scatter('vote_average','budget')

df2.plot.scatter('vote_average','revenue')



df2.plot.scatter('vote_count','budget')

df2.plot.scatter('vote_count','revenue')

df2.plot.scatter('revenue','budget')



df2.plot.scatter('runtime','vote_count')

df2.plot.scatter('runtime','budget')

df2.plot.scatter('runtime','revenue')

df2.plot.scatter('runtime','vote_average')



y = df2['vote_average'].values

X = df2[['budget','runtime','revenue']].values

df2.loc[:, df2.isnull().any()].head()
from sklearn.preprocessing import Imputer

my_imputer = Imputer()



temp=df2

X2 = my_imputer.fit_transform(df2[['runtime']])
#len(df2['budget'])

df2['runtime'] = X2
from sklearn.model_selection import train_test_split



train, test = train_test_split(df2, test_size=0.3)
from sklearn import linear_model

x_train = train[['budget','revenue','runtime']]

y_train = train[['vote_average']]



x_test = test[['budget','revenue','runtime']]



#Create linear regression object

linear = linear_model.LinearRegression()



#Train the model using the training sets and check score

linear.fit(x_train, y_train)

linear.score(x_train, y_train)



#Equation coefficient and Intercept

print('Coefficient: \n', linear.coef_)

print('Intercept: \n', linear.intercept_)



#Predict Output

predicted = linear.predict(x_test)

y_test = test[['vote_average']]
#plt.scatter(y_test,x_test['budget'])



import matplotlib.pyplot as plt



x = range(10)

y = range(300000000)

fig = plt.figure()

ax1 = fig.add_subplot(111)



ax1.scatter(y_test,x_test['budget'], s=10, c='b', marker = "s", label="Real averages")

ax1.scatter(predicted, x_test['budget'], s=10, c='r', marker="o", label="predicted averages")
plt.scatter(y_test,x_test['budget'])

plt.scatter(predicted, x_test['budget'])
plt.scatter(y_test,x_test['revenue'])

plt.scatter(predicted, x_test['revenue'])
plt.scatter(y_test,x_test['runtime'])

plt.scatter(predicted, x_test['runtime'])
from sklearn import tree



x_train = train[['budget','revenue','runtime']]

y_train = train[['vote_average']]



x_test = test[['budget','revenue','runtime']]

y_test = test[['vote_average']]



#Create tree object

model = tree.DecisionTreeRegressor()

model.fit(x_train,y_train)

model.score(x_train,y_train)



#Predict output

predicted= model.predict(x_test)
plt.scatter(y_test,x_test['revenue'])

plt.scatter(predicted, x_test['revenue'])
plt.scatter(predicted, x_test['budget'])
plt.scatter(y_test, x_test['budget'])
plt.scatter(predicted, x_test['runtime'])

plt.scatter(y_test, x_test['runtime'])
#The science fiction genre is causing trouble in splitting the strings, so we change it to sciencefiction:

import warnings

warnings.filterwarnings('ignore')



temp = df2['genres']

df2['genres']

for j in range(len(df2['genres'])):

    if "Science Fiction" in df2['genres'][j]:

        df2['genres'][j]=df2['genres'][j].replace("Science Fiction", "Sciencefiction")

        

df2['genres'][3285]
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
df2.head()
df2['num_genres']=genreArray

df2
test = map(int, df2['num_genres'])
def find_in_sublists(lst, value):

    for sub_i, sublist in enumerate(lst):

        try:

            return (sub_i, sublist.index(value))

        except ValueError:

            pass



    raise ValueError('%s is not in lists' % value)
find_in_sublists(df2['num_genres'], '{"id"')
df3 = df2.as_matrix()

df3[3285]
for count in enumerate(temp):

    map(int, temp[count])
temp = df2['num_genres']

test = []*len(temp)

for i in range(len(temp)):

    test.append(list(map(int, temp[i])))
df2['genreArray_Int'] = test
type(df2['genreArray_Int'][1][0])