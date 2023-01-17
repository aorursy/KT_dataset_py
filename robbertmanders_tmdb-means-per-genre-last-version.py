import json



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from sklearn.preprocessing import Imputer

from sklearn.decomposition import PCA # Principal Component Analysis module

from sklearn.cluster import KMeans # KMeans clustering 





import warnings

warnings.filterwarnings('ignore')



import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output



movies = pd.read_csv('../input/tmdb_5000_movies.csv')

credits = pd.read_csv('../input/tmdb_5000_credits.csv')



movies.head()
credits.head()
(credits['title']==movies['title']).describe()
del credits['title']

del credits['movie_id']

movie_df = pd.concat([movies, credits], axis=1)
movie_df.head()
newCols = ['id','title','release_date','popularity','vote_average','vote_count',

           'budget','revenue','genres','keywords','cast','crew','tagline', 'runtime', 'production_companies', 

           'production_countries', 'status']



df2 = movie_df[newCols]

df2.head()
df2.describe().round()
my_imputer = Imputer()



temp=df2

X2 = my_imputer.fit_transform(df2[['runtime']])

df2['runtime'] = X2

df2.describe().round()
del df2['id']
#df2['vote_classes'] = pd.cut(df2['vote_average'],10, labels=["1", "2","3","4","5","6","7","8","9","10"])

df2['vote_classes'] = pd.cut(df2['vote_average'],4, labels=["low", "medium-low","medium-high","high"])
fig = plt.figure(figsize = (10,15))

ax = fig.gca()



#fig, axes = plt.subplots(nrows=3, ncols=2)

#fig.tight_layout() # Or equivalently,  "plt.tight_layout()"



#fig.subplots_adjust(hspace=0.1)

df2.hist(ax=ax)

#df2.hist(ax=ax)
def load_tmdb_movies(path):

    df = pd.read_csv(path)

    df['release_date'] = pd.to_datetime(df['release_date']).apply(lambda x: x.date())

    json_columns = ['genres', 'keywords', 'production_countries', 'production_companies', 'spoken_languages']

    for column in json_columns:

        df[column] = df[column].apply(json.loads)

    return df



def load_tmdb_credits(path):

    df = pd.read_csv(path)

    json_columns = ['cast', 'crew']

    for column in json_columns:

        df[column] = df[column].apply(json.loads)

    return df



def pipe_flatten_names(keywords):

    return '|'.join([x['name'] for x in keywords])



credits = load_tmdb_credits("../input/tmdb_5000_credits.csv")

movies = load_tmdb_movies("../input/tmdb_5000_movies.csv")



del credits['title']

df = pd.concat([movies, credits], axis=1)



df['genres'] = df['genres'].apply(pipe_flatten_names)



liste_genres = set()

for s in df['genres'].str.split('|'):

    liste_genres = set().union(s, liste_genres)

liste_genres = list(liste_genres)

liste_genres.remove('')
df_reduced = df[['title','vote_average','release_date','runtime','budget','revenue']].reset_index(drop=True)



for genre in liste_genres:

    df_reduced[genre] = df['genres'].str.contains(genre).apply(lambda x:1 if x else 0)

df_reduced[:5]



df_reduced.head()
plt.rc('font', weight='bold')

f, ax = plt.subplots(figsize=(5,5))

genre_count = []

for genre in liste_genres:

    genre_count.append([genre, df_reduced[genre].values.sum()])

genre_count.sort(key = lambda x:x[1], reverse = True)

labels, sizes = zip(*genre_count)

labels_selected = [n if v > sum(sizes) * 0.01 else '' for n, v in genre_count]

ax.pie(sizes, labels=labels_selected,

      autopct = lambda x:'{:2.0f}%'.format(x) if x>1 else '',

      shadow = False, startangle=0)

ax.axis('equal')

plt.tight_layout()
mean_per_genre = pd.DataFrame(liste_genres)



#Mean votes average

newArray = []*len(liste_genres)

for genre in liste_genres:

    newArray.append(df_reduced.groupby(genre, as_index=True)['vote_average'].mean())

newArray2 = []*len(liste_genres)

for i in range(len(liste_genres)):

    newArray2.append(newArray[i][1])



mean_per_genre['mean_votes_average']=newArray2



#Mean budget

newArray = []*len(liste_genres)

for genre in liste_genres:

    newArray.append(df_reduced.groupby(genre, as_index=True)['budget'].mean())

newArray2 = []*len(liste_genres)

for i in range(len(liste_genres)):

    newArray2.append(newArray[i][1])



mean_per_genre['mean_budget']=newArray2



#Mean revenue 

newArray = []*len(liste_genres)

for genre in liste_genres:

    newArray.append(df_reduced.groupby(genre, as_index=True)['revenue'].mean())

newArray2 = []*len(liste_genres)

for i in range(len(liste_genres)):

    newArray2.append(newArray[i][1])



mean_per_genre['mean_revenue']=newArray2



mean_per_genre['profit'] = mean_per_genre['mean_revenue']-mean_per_genre['mean_budget']



mean_per_genre    
mean_per_genre.sort_values('mean_votes_average', ascending=False).head()

mean_per_genre.sort_values('mean_budget', ascending=False).head()
mean_per_genre.sort_values('mean_revenue', ascending=False).head()
mean_per_genre.sort_values('profit', ascending=False).head()
from datetime import datetime



t = df_reduced['release_date']

t = pd.to_datetime(t)

t = t.dt.year

df_reduced['release_year'] = t



df_list = []*len(liste_genres)

for genre in liste_genres:

    df_list.append(df_reduced.groupby([genre,'release_year']).mean().reset_index())



df_per_genre = []*len(liste_genres)

for i in range(len(df_list)):

    df_per_genre.append(df_list[i][df_list[i].ix[:,0] == 1])
# Budget

columns = range(1988,2018)

budget_genre = pd.DataFrame( columns = columns)



for genre in liste_genres:

    temp=(df_per_genre[liste_genres.index(genre)].pivot_table(index = genre, columns = 'release_year', values = 'budget', aggfunc = np.mean))

    temp = temp[temp.columns[-30:]].loc[1]

    budget_genre.loc[liste_genres.index(genre)]=temp

budget_genre['genre']=liste_genres



# Revenue 



columns = range(1988,2018)

revenue_genre = pd.DataFrame( columns = columns)



for genre in liste_genres:

    temp=(df_per_genre[liste_genres.index(genre)].pivot_table(index = genre, columns = 'release_year', values = 'revenue', aggfunc = np.mean))

    temp = temp[temp.columns[-30:]].loc[1]

    revenue_genre.loc[liste_genres.index(genre)]=temp

revenue_genre['genre']=liste_genres



# Vote average 

columns = range(1988,2018)

vote_avg_genre = pd.DataFrame( columns = columns)



for genre in liste_genres:

    temp=(df_per_genre[liste_genres.index(genre)].pivot_table(index = genre, columns = 'release_year', values = 'vote_average', aggfunc = np.mean))

    temp = temp[temp.columns[-30:]].loc[1]

    vote_avg_genre.loc[liste_genres.index(genre)]=temp

vote_avg_genre['genre']=liste_genres



#vote_avg_genre.index = vote_avg_genre['genre']
budget_genre.index = budget_genre['genre']

budget_genre
revenue_genre.index = revenue_genre['genre']

revenue_genre

vote_avg_genre.index = vote_avg_genre['genre']

vote_avg_genre
fig, ax = plt.subplots(figsize=(9,9))

cmap = sns.cubehelix_palette(start = 1.5, rot = 1.5, as_cmap = True)

sns.heatmap(budget_genre.ix[:,0:30], xticklabels=3, cmap=cmap, linewidths=0.05)


fig, ax = plt.subplots(figsize=(9,9))

cmap = sns.cubehelix_palette(start = 1.5, rot = 1.5, as_cmap = True)

sns.heatmap(revenue_genre.ix[:,0:30], xticklabels=3, cmap=cmap, linewidths=0.05)
fig, ax = plt.subplots(figsize=(9,9))

cmap = sns.cubehelix_palette(start = 1.5, rot = 1.5, as_cmap = True)

sns.heatmap(vote_avg_genre.ix[:,0:30], xticklabels=3, cmap=cmap, linewidths=0.05)
temp = budget_genre

temp[2013]=temp[2013].replace(2.550000e+08, 0)
fig, ax = plt.subplots(figsize=(9,9))

cmap = sns.cubehelix_palette(start = 1.5, rot = 1.5, as_cmap = True)

sns.heatmap(temp.ix[:,0:30], xticklabels=3, cmap=cmap, linewidths=0.05)
from datetime import datetime



t = df_reduced['release_date']

t = pd.to_datetime(t)

t = t.dt.month

df_reduced['release_year'] = t



df_list = []*len(liste_genres)

for genre in liste_genres:

    df_list.append(df_reduced.groupby([genre,'release_year']).mean().reset_index())



df_per_genre = []*len(liste_genres)

for i in range(len(df_list)):

    df_per_genre.append(df_list[i][df_list[i].ix[:,0] == 1])
# Budget

columns = range(1,13)

budget_genre = pd.DataFrame( columns = columns)



for genre in liste_genres:

    temp=(df_per_genre[liste_genres.index(genre)].pivot_table(index = genre, columns = 'release_year', values = 'budget', aggfunc = np.mean))

    temp = temp[temp.columns[-30:]].loc[1]

    budget_genre.loc[liste_genres.index(genre)]=temp

budget_genre['genre']=liste_genres



# Revenue 



columns = range(1,13)

revenue_genre = pd.DataFrame( columns = columns)



for genre in liste_genres:

    temp=(df_per_genre[liste_genres.index(genre)].pivot_table(index = genre, columns = 'release_year', values = 'revenue', aggfunc = np.mean))

    temp = temp[temp.columns[-30:]].loc[1]

    revenue_genre.loc[liste_genres.index(genre)]=temp

revenue_genre['genre']=liste_genres



# Vote average 

columns = range(1,13)

vote_avg_genre = pd.DataFrame( columns = columns)



for genre in liste_genres:

    temp=(df_per_genre[liste_genres.index(genre)].pivot_table(index = genre, columns = 'release_year', values = 'vote_average', aggfunc = np.mean))

    temp = temp[temp.columns[-30:]].loc[1]

    vote_avg_genre.loc[liste_genres.index(genre)]=temp

vote_avg_genre['genre']=liste_genres
budget_genre.index = budget_genre['genre']

budget_genre
revenue_genre.index = revenue_genre['genre']

revenue_genre
vote_avg_genre.index = vote_avg_genre['genre']

vote_avg_genre
fig, ax = plt.subplots(figsize=(9,9))

cmap = sns.cubehelix_palette(start = 1.5, rot = 1.5, as_cmap = True)

sns.heatmap(budget_genre.ix[:,0:12], xticklabels=1, cmap=cmap, linewidths=0.05)
fig, ax = plt.subplots(figsize=(9,9))

cmap = sns.cubehelix_palette(start = 1.5, rot = 1.5, as_cmap = True)

sns.heatmap(revenue_genre.ix[:,0:12], xticklabels=1, cmap=cmap, linewidths=0.05)
fig, ax = plt.subplots(figsize=(9,9))

cmap = sns.cubehelix_palette(start = 1.5, rot = 1.5, as_cmap = True)

sns.heatmap(vote_avg_genre.ix[:,0:12], xticklabels=1, cmap=cmap, linewidths=0.05)
revenue_genre = revenue_genre.drop('Animation')
fig, ax = plt.subplots(figsize=(9,9))

cmap = sns.cubehelix_palette(start = 1.5, rot = 1.5, as_cmap = True)

sns.heatmap(revenue_genre.ix[:,0:12], xticklabels=1, cmap=cmap, linewidths=0.05)
num_list = ['budget','popularity','revenue','runtime','vote_average','vote_count']

movie_num = df2[num_list]

movie_num.head()
f, ax = plt.subplots(figsize=(12,10))

plt.title('Pearson Correlation of Movie Features')

sns.heatmap(movie_num.astype(float).corr(), linewidths=0.25, vmax=1.0, square=True,

           cmap="YlGnBu", linecolor='black', annot=True)
num_list = ['budget','popularity','revenue','runtime','vote_average','vote_count']

movie_num = df2[num_list]

movie_num.head()
training_list = ['budget','popularity','revenue','runtime','vote_count']

training = movie_num[training_list]

target = movie_num['vote_average']
X = training.values

y = target.values
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.33, random_state=42)
from sklearn import linear_model

# Create linear regression object

regr = linear_model.LinearRegression()



# Train the model using the training sets

regr.fit(X_train, y_train)



# Make predictions using the testing set

y_pred_lr = regr.predict(X_test)
f = plt.figure(figsize=(10,5))

plt.scatter(X_test[:,1], y_test, s=50,label="Real vote_average");

plt.scatter(X_test[:,1], y_pred_lr,s=100, c='r',label="Predicted vote_average");

plt.ylabel("vote_average");

plt.legend(loc=2);
from sklearn.ensemble import RandomForestRegressor

# Create linear regression object

rf = RandomForestRegressor(1)



# Train the model using the training sets

rf.fit(X_train, y_train)



# Make predictions using the testing set

y_pred_rf = rf.predict(X_test)
f = plt.figure(figsize=(10,5))

plt.scatter(X_test[:,1], y_test, s=50,label="Real vote_average");

plt.scatter(X_test[:,1], y_pred_rf,s=100, c='r',label="Predited vote_average");

plt.ylabel("vote_average");

plt.legend(loc=2);
from sklearn.metrics import mean_squared_error



error_lr = mean_squared_error(y_test,y_pred_lr)

error_rf = mean_squared_error(y_test,y_pred_rf)



print(error_lr)

print(error_rf)
f = plt.figure(figsize=(10,5))

plt.bar(range(2),[error_lr,error_rf])

plt.xlabel("Classifiers");

plt.ylabel("Mean Squared Error of the vote_average");

plt.xticks(range(2),['Linear Regression','Random Forest'])

plt.legend(loc=2);