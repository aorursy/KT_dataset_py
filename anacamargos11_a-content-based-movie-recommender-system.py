# Importing the relevant packages

from IPython.display import Image

import os

import numpy as np

import pandas as pd

import ast

import time 

import random

from datetime import datetime

from sklearn import ensemble, metrics, linear_model

from sklearn.preprocessing import LabelEncoder, LabelBinarizer, MinMaxScaler

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.decomposition import PCA

from sklearn.svm import SVR
Image("../input/svdimage/SVD.png",width=500)
Image("../input/svrimage/SVR.png",width=450)
Metadata = pd.read_csv('/kaggle/input/the-movies-dataset/movies_metadata.csv')

Nmovies = Metadata['id'].shape[0]

print(Nmovies)

Metadata.head(5)
# treat or drop mal-formatted data rows

Metadata['id'][35587] = '22'

Metadata['id'][29503] = '12'

Metadata['id'][19730] = '1'

Metadata['budget'][35587] = '0'

Metadata['budget'][29503] = '0'

Metadata['budget'][19730] = '0'

Metadata['popularity'][35587] = '2.185485'

Metadata['popularity'][29503] = '1.931659'

Metadata['popularity'][19730] = '0.065736'

Metadata['revenue'][35587] = '0'

Metadata['revenue'][29503] = '0'

Metadata['revenue'][19730] = '0'

Metadata['runtime'][35587] = '0'

Metadata['runtime'][29503] = '0'

Metadata['runtime'][19730] = '0'

Metadata['adult'][35587] = 'False'

Metadata['adult'][29503] = 'False'

Metadata['adult'][19730] = 'False'

Metadata['original_language'][35587] = 'en'

Metadata['original_language'][29503] = 'ja'

Metadata['original_language'][19730] = 'en'

Metadata['genres'][35587] = float('nan')

Metadata['genres'][29503] = float('nan')

Metadata['genres'][19730] = float('nan')

Metadata['production_companies'][35587] = "[{'name': 'Odyssey Media', 'id': 17161}, {'name': 'Pulser Productions', 'id': 18012}, {'name': 'Rogue State', 'id': 18013}, {'name': 'The Cartel', 'id': 23822}]"

Metadata['production_companies'][29503] = "[{'name': 'Aniplex', 'id': 2883}, {'name': 'GoHands', 'id': 7759}, {'name': 'BROSTA TV', 'id': 7760}, {'name': 'Mardock Scramble Production Committee', 'id': 7761}, {'name': 'Sentai Filmworks', 'id': 33751}]"

Metadata['production_companies'][19730] = "[{'name': 'Carousel Productions', 'id': 11176}, {'name': 'Vision View Entertainment', 'id': 11602}, {'name': 'Telescene Film Group Productions', 'id': 29812}]"

Metadata['production_countries'][35587] = "[{'iso_3166_1': 'CA', 'name': 'Canada'}]"

Metadata['production_countries'][29503] = "[{'iso_3166_1': 'US', 'name': 'United States of America'}, {'iso_3166_1': 'JP', 'name': 'Japan'}]"

Metadata['production_countries'][19730] = "[{'iso_3166_1': 'CA', 'name': 'Canada'}, {'iso_3166_1': 'LU', 'name': 'Luxembourg'}, {'iso_3166_1': 'GB', 'name': 'United Kingdom'}, {'iso_3166_1': 'US', 'name': 'United States of America'}]"

Metadata['spoken_languages'][35587] = "[{'iso_639_1': 'en', 'name': 'English'}]"

Metadata['spoken_languages'][29503] = "[{'iso_639_1': 'ja', 'name': '日本語'}]"

Metadata['spoken_languages'][19730] = "[{'iso_639_1': 'en', 'name': 'English'}]"

Metadata['release_date'][35587] = '2014-01-01'

Metadata['release_date'][29503] = '2012-09-29'

Metadata['release_date'][19730] = '1997-08-20'

Metadata['original_title'][35587] = 'Avalanche Sharks'

Metadata['original_title'][29503] = 'Mardock Scramble: The Third Exhaust'

Metadata['original_title'][19730] = 'Midnight Man'

Metadata['overview'][35587] = ' Avalanche Sharks tells the story of a bikini contest that turns into a horrifying affair when it is hit by a shark avalanche.'

Metadata['overview'][29503] = ' Rune Balot goes to a casino connected to the October corporation to try to wrap up her case once and for all.'

Metadata['overview'][19730] = ' - Written by Ørnås'

Metadata['tagline'][35587] = 'Beware Of Frost Bites'

Metadata['tagline'][29503] = float('nan')

Metadata['tagline'][19730] = float('nan')
RatingsDataFrame = pd.read_csv('/kaggle/input/the-movies-dataset/ratings.csv')

Nusers = RatingsDataFrame['userId'].nunique()

print(Nusers)

RatingsDataFrame.head(10)
MovieFeatures = pd.DataFrame()

# keep the numerical columns untouched for now

MovieFeatures[0] = Metadata['id']

MovieFeatures[1] = Metadata['budget']

MovieFeatures[2] = Metadata['popularity']

MovieFeatures[3] = Metadata['revenue']

MovieFeatures[4] = Metadata['runtime']

MovieFeatures[5] = Metadata['vote_average']

MovieFeatures[6] = Metadata['vote_count']
# extracting the year from the release date feature

release_date = np.array(Metadata['release_date'])

for i in range(0,Nmovies):

    if (type(release_date[i])==str and \

        len(release_date[i]) == 10):

        year = datetime.strptime(release_date[i], '%Y-%m-%d')

        year = year.year

        release_date[i] = year

    else:

        release_date[i] = float('nan')



for i in range(1,31): 

    if (type(release_date[-i])==str and \

        len(release_date[-i]) == 10): 

        year = datetime.strptime(release_date[-i], '%Y-%m-%d') 

        year = year.year 

        release_date[-i] = year 

    else: 

        release_date[-i] = float('nan') 



MovieFeatures[7] = release_date
Metadata['genres'][0]
# this function extracts words out of dictionaries and formats them

def word_extractor(row_of_words):

    words_joined = []

    if (type(row_of_words)!=str or type(ast.literal_eval(row_of_words))!=list):

        words_joined = ['']

    else:

    # extract words from the dictionaries

        word_list = ast.literal_eval(row_of_words)

        for w in range(0,len(word_list)):

            word_list[w] = word_list[w]['name']

            word_list[w] = word_list[w].replace(" ","")

        words_joined.append(' '.join(word_list))    

    return words_joined



# applying the word_extractor function to dict. features

genres = []

for m in range(0,Nmovies):

    genres.append(word_extractor(Metadata['genres'][m]))



genres = np.array(genres)

genres = genres.ravel()

print(genres[0])
# one-hot encoding binary features

le = LabelEncoder()

lb = LabelBinarizer()

original_language = le.fit_transform(Metadata['original_language'].fillna('0'))

original_language = lb.fit_transform(original_language)

for i in range(8,8+original_language.shape[1]):

    MovieFeatures[i] = original_language[:,i-8]

    

# finally, vectorizing the features.

count_vectorizer = CountVectorizer()

tfid = TfidfVectorizer(stop_words={'english','french','spanish','german'},\

                  max_features=200)



genres = count_vectorizer.fit_transform(genres)

original_title = tfid.fit_transform(Metadata['original_title'])

overview = tfid.fit_transform(Metadata['overview'].values.astype('U'))

tagline = tfid.fit_transform(Metadata['tagline'].values.astype('U'))
genres[0].todense()
print("finally, vectorizing the features... \n")

count_vectorizer = CountVectorizer()

tfid = TfidfVectorizer(stop_words={'english','french','spanish','german'},\

                  max_features=200)



original_title = tfid.fit_transform(Metadata['original_title'])

overview = tfid.fit_transform(Metadata['overview'].values.astype('U'))

tagline = tfid.fit_transform(Metadata['tagline'].values.astype('U'))



# this function records the processed data in the MovieFeatures DataFrame

def record_new_data(new_data):

  size = MovieFeatures.shape[1]

  for i in range(size,size+new_data.shape[1]):

    MovieFeatures[i] = new_data.toarray()[:,i-size]



record_new_data(genres)

record_new_data(original_title)

record_new_data(overview)

record_new_data(tagline)



genres = genres.toarray()

print("Getting rid of NaN values... \n")

Metadata_mean = Metadata.mean(skipna=True,numeric_only=True)

MovieFeatures[3] = MovieFeatures[3].fillna(Metadata_mean['revenue'])

MovieFeatures[4] = MovieFeatures[4].fillna(Metadata_mean['runtime'])

MovieFeatures[5] = MovieFeatures[5].fillna(Metadata_mean['vote_average'])

MovieFeatures[6] = MovieFeatures[6].fillna(Metadata_mean['vote_count'])

MovieFeatures = MovieFeatures.fillna('0')
print("Running PCA on Movie Features... \n")

features = np.array(MovieFeatures)

scaler = MinMaxScaler(feature_range=[0, 1])

features[:,3:7] = scaler.fit_transform(features[:, 3:7])

ncomp = 2

pca = PCA(n_components=ncomp)

pca_features = pca.fit_transform(features[:,1:-1]) 

PCAfeatures = np.zeros((pca_features.shape[0],pca_features.shape[1]+1))

PCAfeatures[:,1:pca_features.shape[1]+1] = pca_features

PCAfeatures[:,0] = MovieFeatures[0]

PCA_df = pd.DataFrame(PCAfeatures)

pca_variance = pca.explained_variance_ratio_.sum()
print(pca_variance)
MovieFeatures.head(5)
PCA_df.head(5)
def user_dataframe(active_user):

  user_df = RatingsDataFrame.groupby('userId').get_group(active_user)

  user_df = PCA_df.merge(user_df,left_on=0,right_on='movieId')

  return user_df
user_dataframe(11)
# Step 1: this function splits a user dataframe into training and test data

def test_split(active_user):

  percentage = 0.85

  user_df = user_dataframe(active_user)

  X = user_df.iloc[:,1:3]

  y = user_df.iloc[:,5]

  x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.15)

  return x_train, x_test, y_train, y_test      
# this function finds the best hyperparameters for an svr user model

def svr_tuning(active_user):

  parameters = {'C':[0.1, 1, 10],'epsilon':[0.1,0.2,0.5],'gamma':['auto','scale']}

  x_train, x_test, y_train, y_test = test_split(active_user)

  svr = SVR(gamma='scale')

  svr = GridSearchCV(svr, parameters, random_state=0)

  search = svr.fit(x_train, y_train)

  return search.best_params_



# this function finds the best hyperparameters for an gbr user model

def GBR_tuning(active_user):

  parameters = {'n_estimators':[100,300,500],'learning_rate':[0.001,0.01,0.1,1],\

    'loss':['ls','lad','huber','quantile']}

  x_train, x_test, y_train, y_test = test_split(active_user)

  GBR = ensemble.GradientBoostingRegressor()

  clf = GridSearchCV(GBR, parameters)

  search = clf.fit(x_train, y_train)

  return search.best_params_
# this function creates an svr model for an active user

def training(active_user):

  x_train, x_test, y_train, y_test = test_split(active_user)

  svr = SVR(gamma='auto', epsilon=0.2, C=0.1)

  LR = linear_model.LinearRegression()

  GBR = ensemble.GradientBoostingRegressor(learning_rate=0.001,loss='ls',\

    n_estimators=100)

  svr.fit(x_train, y_train)

  predicted = svr.predict(x_test)

  model_rmse = np.sqrt(metrics.mean_squared_error(y_test,predicted))

  return svr, [x_test, y_test], model_rmse
# this function returns the N largest elements of a list

def Nmaxelements(list1, N): 

    final_list = []  

    for i in range(0, N):  

        max1 = 0          

        for j in range(len(list1)):      

            if list1[j] > max1: 

                max1 = list1[j];                  

        list1.remove(max1); 

        final_list.append(max1)          

    return final_list 



# this function creates an svr model for an active user

def recommendations(active_user,n_recom):

  svr, testdata, model_rmse = training(active_user)  

  recommend = svr.predict(pca_features)

  recommend_max = Nmaxelements(recommend.tolist(),n_recom)

  suggestions = []

  genres_array = np.zeros(20)

  for i in range(0,len(recommend)): 

     if recommend[i] in recommend_max:

       suggestions.append(Metadata['original_title'][i])

       suggestions.append(word_extractor(Metadata['genres'][i]))

       genres_array = genres_array + genres[i]

  return genres_array, suggestions, model_rmse
# this function computes a (N movies watched)-weighted accuracy avg for N users

def accuracy(N):

  counter = 0

  R2 = 0

  error_rsme = 0

  for i in range(1,N):

    active_user = random.randint(1,Nusers)

    user_df = user_dataframe(active_user)

    Nmovies_rated = user_df.shape[0]

    if (Nmovies_rated > 10):

      svr, testdata, model_rmse = training(active_user)

      predicted = svr.predict(testdata[0])

      rmse = np.sqrt(metrics.mean_squared_error(testdata[1],predicted))

      error_rsme = error_rsme + Nmovies_rated*rmse

      test_score = svr.score(testdata[0], testdata[1])

      R2 = R2 + Nmovies_rated*test_score

      counter = counter + Nmovies_rated

  error_rsme = error_rsme/counter

  R2 = R2/counter

  return error_rsme, R2
# this function computes a diversity index for an active user

def diversity(active_user,n_recom):

  genres_counter = 0

  Ngenres = 20

  user_df = user_dataframe(active_user)

  Nmovies_rated = user_df.shape[0]

  if (Nmovies_rated > 1):

    genres_array, suggestions, model_rmse = recommendations(active_user,n_recom)

    for i in range(0,genres_array.size):

      if (genres_array[i] > 0):

        genres_counter = genres_counter + 1

  genres_counter = genres_counter/Ngenres

  return genres_counter



# this function computes a diversity avg index for N users

def diversity_avg(N,n_recom):

  diversity_total = 0

  for user in range(1,N):

    diversity_total = diversity_total + diversity(user,n_recom)

  diversity_total = diversity_total/N

  return diversity_total
# this function measures the algorithmic runtime and accuracy for N user models

def model_performance(N):

  t1 = time.clock()

  error_rsme, R2 = accuracy(N)

  t2 = time.clock()

  t  = t2-t1

  return error_rsme, R2, t
recommendations(2,5)
recommendations(3,5)
recommendations(11,5)
recommendations(13,5)
performance = model_performance(100)
performance[0]
performance[1]
performance[2]
Image("../input/noveltyhist/novelty_hist2.png",width=750)