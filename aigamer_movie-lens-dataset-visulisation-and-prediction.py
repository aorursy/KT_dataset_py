#data analysis libraries 

import numpy as np

import pandas as pd



#visualization libraries

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

#ignore warnings

import warnings

warnings.filterwarnings('ignore')
#import train and test CSV files

movie = pd.read_csv("../input/movies.csv")

ratings = pd.read_csv("../input/ratings.csv")

links = pd.read_csv("../input/links.csv")

tags = pd.read_csv("../input/tags.csv")

#take a look at the training data

print(movie.shape)

print(ratings.shape)

print(links.shape)

print(tags.shape,end="\n\n")

m=pd.read_csv("../input/movies.csv")

movie.head(3)

links.head(3)

ratings.head(3)

tags.head(3)



#get a list of the features within the dataset

print("Movie : ", movie.columns,end="\n\n")

print("Rating : ", ratings.columns,end="\n\n")

print("Links : ", links.columns,end="\n\n")

print("Tags : ", tags.columns,end="\n\n")



movie.info()

ratings.info()

tags.info()
# Droping the timestamp column from ratings and tags file

ratings.drop(columns='timestamp',inplace=True)

tags.drop(columns='timestamp',inplace=True)
#Extracting the year from the Title

movie['Year'] = movie['title'].str.extract('.*\((.*)\).*',expand = False)
#Ploting a Graph with No.of Movies each Year corresponding to its Year

plt.plot(movie.groupby('Year').title.count())

plt.show()

a=movie.groupby('Year').title.count()

print('Max No.of Movies Relesed =',a.max())

for i in a.index:

    if a[i] == a.max():

        print('Year =',i)

a.describe()
# Seperate the Geners Column and Encoding them with One-Hot-Encoding Method.

genres=[]

for i in range(len(movie.genres)):

    for x in movie.genres[i].split('|'):

        if x not in genres:

            genres.append(x)  



len(genres)

for x in genres:

    movie[x] = 0

for i in range(len(movie.genres)):

    for x in movie.genres[i].split('|'):

        movie[x][i]=1

movie
movie.drop(columns='genres',inplace=True)

movie.sort_index(inplace=True)
x={}

for i in movie.columns[4:23]:

    x[i]=movie[i].value_counts()[1]

    print("{}    \t\t\t\t{}".format(i,x[i]))



plt.bar(height=x.values(),x=x.keys())

plt.show()
#Add a Column `rating` in movie DF and assign them with the Mean Movie Rating for that Movie.

x=ratings.groupby('movieId').rating.mean()

movie = pd.merge(movie,x,how='outer',on='movieId')

movie['rating'].fillna('0',inplace=True)
# Now Lets group all the ratings with respect to movieId and count the no of Users

x = ratings.groupby('movieId',as_index=False).userId.count()

x.sort_values('userId',ascending=False,inplace=True)

y = pd.merge(movie,x,how='outer',on='movieId')



y.drop(columns=[i for i in movie.columns[2:23]],inplace=True)



y.sort_values(['userId','rating'],ascending=False)
#find the user with highest no.of. movie ratings and that users mean rating. 

x = ratings.groupby('userId',as_index=False).movieId.count()

y = ratings.groupby('userId',as_index=False).rating.mean()

x = pd.merge(x,y,how='outer',on='userId')
x.describe()
x.sort_values('movieId',ascending=False)
for i in movie.columns[3:]:

    movie[i] = movie[i].astype(int)
#importing necessary packages for model prediction and evaluation

import sklearn

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split
#split the data into features and results

X = movie[movie.columns[3:23]]

y = movie[movie.columns[-1]]
#spliting the data into Train Test and Validation sets

X_train , X_test , y_train , y_test = train_test_split(X ,y ,test_size=0.20 ,random_state = 1 ,shuffle = True)
model = RandomForestRegressor(n_estimators=560,random_state=42)

model.fit(X_train,y_train)

print(mean_absolute_error(model.predict(X_train),y_train))
preds = model.predict(X_test)

preds
print(mean_absolute_error(y_test,preds))