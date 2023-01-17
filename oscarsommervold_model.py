#imports

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import PolynomialFeatures

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression, Lasso, LinearRegression

from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, log_loss, mean_squared_error

import seaborn as sns

import matplotlib.pyplot as plt
#les inn data

rank_df = pd.read_csv('../input/sample-data/rangering.csv')

user_df = pd.read_csv('../input/sample-data/bruker.csv')

movie_df = pd.read_csv('../input/sample-data/film.csv')

userrank_df = rank_df.merge(user_df,left_on = 'BrukerID', right_on = 'BrukerID')

final_df = userrank_df.merge(movie_df,left_on = 'FilmID', right_on = 'FilmID')
#Exploring user activity

average_numRating= len(final_df['Rangering'])/len(user_df['BrukerID'])

print('The average number of ratings per user is: ' ,average_numRating)

print('\nThe most common rating is : ', final_df['Rangering'].median())

print('\nThe average rating is: ', final_df['Rangering'].mean())

#With a rating scale of 1-5, our dataset is very skewed towards positive reviews. 



user_activity = final_df['BrukerID'].value_counts()

print('\n most active and inactive users \n' ,user_activity)

#All users have rated at least 10 movies. All users in the dataset have rated a significant number of titles.

#Exploring user data

age_freq = user_df['Alder'].value_counts(normalize = True)

print(age_freq)

# 76% of the users are in the agerange 18 - 44, and 60% in the range 25-44

gender_freq = user_df['Kjonn'].value_counts(normalize = True)

print('\n',gender_freq)

#74% of the users are Male, meaning the dataset is very skewed towards male preferences in ratings. 

jobs = ["annet eller ikke spesifisert","akademisk/pedagog","kunstner","administrativt","student","kundeservice",

        "helse","ledende","bonde","advokat","programmerer","pensjonert","salg/markedsføring","forsker","selvstendig næringsdrivende",

        "tekniker/ingeniør","håndverker","arbeidsledig","forfatter"]



df = pd.DataFrame({'Jobs':jobs

                   , 'Counts':user_df['Jobb'].value_counts().sort_index().values})

ax = df.plot.barh(x='Jobs', y='Counts',figsize = (10,5))

#finding and visualizing occupation distribution among users

#De mest representerte er: Annet eller ikke spesifisert, student, ledende og tekniker/ingeniør

#De minst representerte er: håndverker, advokat og pensjonert.

# Hjemmeværende og elev er ikke representert i det hele tatt

#Exploring ranking habits by occupation

rating_count_by_job = final_df['Jobb'].value_counts().sort_index()

average_rating_by_job = []

for i in rating_count_by_job.index:

    average_rating_by_job.insert(i,final_df.loc[final_df['Jobb'] == i, 'Rangering'].sum()/rating_count_by_job[i])



df = pd.DataFrame({'Jobs':jobs,'Average Rating':average_rating_by_job})

ax = df.plot.barh(x='Jobs', y='Average Rating',figsize = (10,5),xlabel = "Job Description",ylabel = "Average Rating", xlim = {2,4})

#Advokater, ingeniører og håndverkere rangerer filmer de ser lavere enn gjennomsnittet

#Pensjonister og arbeidsledige rangerer det meste de ser høyere enn de andre gruppene.

#Det er ikke utenkelig at disse tallene er sterkt påvirket av at det er få håndverkere, advokater og pensjonister i datasettet våres
#Exploring movie popularity

popularity = final_df['Tittel'].value_counts()

print(popularity)

#There are some titles with very few ratings that should probably 

#not be included in the datasets when selecting a model

print('\nThere are', rank_df['FilmID'].nunique(),'Unique movies rated in the dataset')

#Identifying titles with less than five total ratings.

#Might be useful for filtering titles with insuficcient number of ratings later on

least_popular = (popularity <= 5).where(lambda x : x == True).dropna()

print('\n Number of titles with less than 5 ratings:', len(least_popular))

#Exploring genre popularity and correlation with other genres

genres = ['Action', 'Adventure', 'Animation',"Children's",'Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']

genre_distribution = []

for gen in genres:

    genre_distribution.append(movie_df[gen].value_counts(normalize = True)[1])

    

df = pd.DataFrame({'Genres':genres

                   , 'occurence frequency':genre_distribution})

ax = df.plot.bar(x='Genres', y='occurence frequency', rot=0,figsize = (20,5))

#Drama and Comedy are by far the most frequent genres,with Drama occuring in over 40% of all movies in the dataset.

#Fantasy, Film-Noir and Western are the least frequent 
movie_df.drop(columns = ['FilmID','Tittel']).corr().style.background_gradient(cmap='coolwarm')

#The only strong correlation is between the animation and children's genre

#There is some correlations between combinations such as action-adventure, fantasy-adventure, animation-musical, film-noir-crime, musical-animation, musical-children's etc.

#Some genres such as Documentary, War and Western has no correlation to other genres and is mostly independent.
genre_ratings = []

for gen in genres:

    genre_ratings.append(final_df.loc[final_df[gen] == 1, 'Rangering'].sum()/final_df[gen].value_counts()[1])

df = pd.DataFrame({'Genres':genres, 'Average Rating':genre_ratings})

ax = df.plot.bar(x='Genres', y='Average Rating', rot=0,figsize = (20,5), ylim = {2,4.5})

#Film-Noir movies are consistently rated very highly while Horror and western-movies tend to be rated lower than average

#Most genres hover around the 3.5 rating, which is the mean rating we found earlier.
#removing all entries for movies with less than 5 ratings

filtered_final_df = final_df[~final_df.Tittel.isin(least_popular.index)]
#Basemodell

#Predicts the mean rating of the movie for all users

base_prediction = pd.DataFrame(rank_df.groupby('FilmID')['Rangering'].mean().round())

base_model  = rank_df.drop(columns = 'Tidstempel').merge(base_prediction, left_on = 'FilmID',right_on = 'FilmID')

base_model.columns = ['BrukerID','FilmID','BrukerRangering','Prediksjon']

base_model.set_index('BrukerID', inplace = True)

base_model.sort_index()
#Finding the root mean squared error of the basemodel

base_rms = np.sqrt(mean_squared_error(base_model['BrukerRangering'],base_model['Prediksjon']))

print(base_rms)
#Creating user_tables and preparing test-data

user_ratings = []

user_X = []

user_Y = []

# Looping through all unique userID's

for id in final_df['BrukerID'].value_counts().sort_index().index:

    #Getting all movie-profiles and the corresponding rating by this user

    user_rating = final_df.loc[final_df['BrukerID'] == id].drop(columns = ['Tittel','Tidstempel','Postkode'])

    user_ratings.append(user_rating)

    #Preparing testdata for current user

    user_X.append(user_rating.drop(columns = ['BrukerID','Rangering','FilmID','Kjonn','Alder','Jobb']).values)

    user_Y.append(user_rating['Rangering'].values)
models_val_rmse = []

models_gen_rmse = []

models_val_acc = []

model_predictions = []

for profile in range(len(final_df['BrukerID'].value_counts().sort_index().index)):

    #fetching and splitting testdata for current user

    X = user_X[profile]

    y = user_Y[profile]

    X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, train_size = 0.7, test_size=0.3, random_state=42)

    X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, train_size = 0.7, test_size=0.3, random_state=42)

    #Building a model for each user

    lr = LinearRegression()

    #fit model with the available movie-profiles rated

    lr.fit(X_train,y_train)

    #Collecting root mean squared error on validation and test datasets

    models_val_rmse.append(np.sqrt(mean_squared_error(y_val, np.clip(lr.predict(X_val).round(),1,5))))

    models_gen_rmse.append(np.sqrt(mean_squared_error(y_test, np.clip(lr.predict(X_test).round(),1,5))))

    models_val_acc.append(accuracy_score(y_val,np.clip(lr.predict(X_val).round(),1,5)))

    #predict rating for every movie

    prediction = movie_df[['FilmID','Tittel']].sort_values(by = 'FilmID')

    prediction['Prediksjon'] = pd.DataFrame(np.clip(lr.predict(movie_df.drop(columns = ['Tittel','FilmID'])).round(),1,5))

    val = user_ratings[profile]

    val = val[['FilmID','Rangering']]

    prediction_merge = pd.merge(prediction,val,how = 'outer')

    prediction_merge['Rangering'] = prediction_merge['Rangering'].fillna(0)

    model_predictions.append(prediction_merge)
#Mean RMSE across all models

print('The mean RMSE for validation data is:' ,pd.DataFrame(models_val_rmse).mean()[0])

print('\nThe estimated generalization error of the model is: ',pd.DataFrame(models_gen_rmse).mean()[0])
#Computing RMSE between known rating and predicted rating across all models

models_pred_rmse = []

for prediction in model_predictions:

    p = prediction[prediction['Rangering'] != 0]

    models_pred_rmse.append(np.sqrt(mean_squared_error(p['Rangering'],p['Prediksjon'])))

print('The mean RMSE for all predicted ratings with known rating is: ', pd.DataFrame(models_pred_rmse).mean()[0])
#Looking at prediction table for users.

model_number = 100

mod = model_predictions[model_number]

mod = mod[mod['Rangering'] != 0]

print('this is the model for userID :' ,final_df['BrukerID'].value_counts().sort_index().index[model_number], ' (only showing predictions for movies the user rated )')

#user_ratings[10].sort_values(by = 'FilmID')

print('The rmse for this model is: ' ,models_pred_rmse[model_number])

mod
def build(mID,vector):

    a = []

    for i in mID:

        if i in vector.index:

            a.append(vector.loc[i,'Rangering'] - vector.Rangering.mean())

        else:

            a.append(0)

    return a
filtered_movies = movie_df[movie_df.Tittel.isin(final_df['Tittel'])]

col_matrix = filtered_movies[['FilmID','Tittel']]

for user in final_df['BrukerID'].value_counts().sort_index().index:

    a = final_df[final_df.BrukerID == user][['FilmID','Rangering']]

    a.set_index('FilmID', inplace = True)

    col_matrix[user] = build(col_matrix.FilmID.values,a)

col_matrix.set_index('FilmID', inplace = True)

col_matrix
#Finding the most correlated users

cor_matrix = pd.get_dummies(col_matrix.drop(columns = ['Tittel']))

cor_val = cor_matrix.corr()
matrix_c = col_matrix.copy().drop(columns = 'Tittel')

for vector in col_matrix.drop(columns = 'Tittel').columns:

    col_matrix[vector] = col_matrix.loc[:,cor_val[vector].sort_values(kind = "quicksort",ascending = False)[:50].index].replace(0,np.nan).mean(numeric_only = True,axis = 1)
col_matrix.fillna(0,inplace = True)
for vector in col_matrix.drop(columns = 'Tittel').columns:

    col_matrix[vector]+=rank_df[rank_df.BrukerID == vector]['Rangering'].mean()

    col_matrix[vector] = np.clip(col_matrix[vector].round(),1,5)
col_matrix
model_overall = []

for vector in col_matrix.drop(columns = 'Tittel').columns:

    #compare predicted ratings to acutal ratings

    compare = pd.merge(rank_df.loc[rank_df.BrukerID == vector].drop(columns = ['BrukerID','Tidstempel']),col_matrix[vector], on = 'FilmID', how = 'outer').fillna(0)

    known_ratings = compare[compare.Rangering != 0]

    model_overall.append(np.sqrt(mean_squared_error(known_ratings['Rangering'],known_ratings[vector])))
print('The mean RMSE for all predicted ratings with known rating is: ', pd.DataFrame(model_overall).mean()[0])
rank_df = pd.read_csv('../input/cleandata/rangering_cleaned.csv')

user_df = pd.read_csv('../input/cleandata/bruker_cleaned.csv')

movie_df = pd.read_csv('../input/cleandata/film_cleaned.csv')

userrank_df = rank_df.merge(user_df,left_on = 'BrukerID', right_on = 'BrukerID')

final_df = userrank_df.merge(movie_df,left_on = 'FilmID', right_on = 'FilmID')
models_val_rmse = []

models_gen_rmse = []

model_predictions = []

for (i,j) in enumerate(user_df.BrukerID):

    X = final_df.loc[final_df['BrukerID'] == j].drop(columns = ['BrukerID','Tittel','Tidstempel','Postkode','Rangering','Postkode','FilmID','Kjonn','Jobb','Alder']).values

    y = final_df.loc[final_df['BrukerID'] == j].Rangering.values

    

    X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, train_size = 0.8, test_size=0.2, random_state=42)

    X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, train_size = 0.8, test_size=0.2, random_state=42)

    

    lr = LinearRegression()

    lr.fit(X_train,y_train)

    models_val_rmse.append(np.sqrt(mean_squared_error(y_val, np.clip(lr.predict(X_val).round(),1,5))))

    models_gen_rmse.append(np.sqrt(mean_squared_error(y_test, np.clip(lr.predict(X_test).round(),1,5))))

    prediction = movie_df[['FilmID','Tittel']].sort_values(by = 'FilmID')

    prediction['Prediksjon'] = pd.DataFrame(np.clip(lr.predict(movie_df.drop(columns = ['Tittel','FilmID'])).round(),1,5))

    val = userrank_df.loc[userrank_df.BrukerID == j][['FilmID','Rangering']]

    prediction_merge = pd.merge(prediction,val,how = 'outer')

    prediction_merge['Rangering'] = prediction_merge['Rangering'].fillna(0)

    model_predictions.append(prediction_merge)
#Mean RMSE across all models

print('The mean RMSE for validation data is:' ,pd.DataFrame(models_val_rmse).mean()[0])

print('\nThe estimated generalization error of the model is: ',pd.DataFrame(models_gen_rmse).mean()[0])
cbf_matrix = movie_df[['FilmID','Tittel']]

for (i,uid) in enumerate(user_df.BrukerID):

    cbf_matrix[uid] = model_predictions[i].Prediksjon

cbf_matrix.set_index('FilmID',inplace = True)  
cbf_matrix
matrix = pd.read_csv('../input/rating-vectors/user_rating_vector.csv',index_col = 'FilmID')

matrix
matrix_cor = pd.read_csv('../input/user-cor-matrix/user_correlation_matrix.csv', index_col = 0)

matrix_cor
rank = pd.read_csv('../input/cleandata/rangering_cleaned.csv')

rank
matrix_c = matrix.copy().drop(columns = 'Tittel')

for vector in matrix.drop(columns = 'Tittel').columns:

    matrix[vector] = matrix_c.loc[:,matrix_cor[vector].sort_values(kind = "quicksort",ascending = False)[:100].index.astype(str)].replace(0,np.nan).mean(numeric_only = True,axis = 1)
matrix
matrix.fillna(0,inplace = True)
for vector in matrix.drop(columns = 'Tittel').columns:

    matrix[vector]+=rank[rank.BrukerID == pd.to_numeric(vector)]['Rangering'].mean()

    matrix[vector] = np.clip(matrix[vector].round(),1,5)
model_overall = []

for vector in matrix.drop(columns = 'Tittel').columns:

    #compare predicted ratings to acutal ratings

    compare = pd.merge(rank.loc[rank.BrukerID == pd.to_numeric(vector)].drop(columns = ['BrukerID','Tidstempel']),matrix[vector], on = 'FilmID', how = 'outer').fillna(0)

    known_ratings = compare[compare.Rangering != 0]

    model_overall.append(np.sqrt(mean_squared_error(known_ratings['Rangering'],known_ratings[vector])))
print('The mean RMSE for all predicted ratings with known rating is: ', pd.DataFrame(model_overall).mean()[0])
comp = final_df.copy()

comp['Kjonn'].replace(['F','M'],[0,1],inplace=True)
comp.Kjonn.value_counts()
X = comp.drop(columns = ['BrukerID','FilmID','Rangering','Tidstempel','Postkode','Tittel','Jobb','Alder']).values

y = comp.Rangering.values

X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, train_size = 0.8, test_size=0.2, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, train_size = 0.8, test_size=0.2, random_state=42)

    

lr = LinearRegression()

lr.fit(X_train,y_train)

val = np.sqrt(mean_squared_error(y_val, np.clip(lr.predict(X_val).round(),1,5)))

gen = np.sqrt(mean_squared_error(y_test, np.clip(lr.predict(X_test).round(),1,5)))
gen