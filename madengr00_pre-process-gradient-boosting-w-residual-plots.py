import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from datetime import date, datetime

from sklearn.metrics import mean_squared_error

from sklearn import ensemble

from sklearn import datasets

from sklearn.utils import shuffle

import os

print(os.listdir("../input"))
# Read the csv files into a pandas DataFrame

# Our goal is to predict the movie revenue without adding additional data to the dataset

train = pd.read_csv("../input/tmdb-box-office-prediction/train.csv")

test = pd.read_csv("../input/tmdb-box-office-prediction/test.csv")

train.head(2)
#Good catch & correction found here

#https://www.kaggle.com/jiegeng94/simple-tmdb-prediction-with-gradient-boosting

#The year values have only two digits and the years before 1969 are denoted as ones of 2000's. Make it correct. 

def expand_release_date(df):

    df.release_date = pd.to_datetime(df.release_date)



    df['year'] = df.release_date.dt.year

    df['year'] = df.year.apply(lambda x: x-100 if x > 2020 else x)

    

    df['month'] = df.release_date.dt.month

    df['day'] = df.release_date.dt.dayofweek

    df['quarter'] = df.release_date.dt.quarter

    

    return df



train = expand_release_date(train)

test = expand_release_date(test)
# Import Cast Table with Counts of Appearances

file = "../input/weighttables/Cast_Data.csv"

castData = pd.read_csv(file)

castData.rename(columns = {"Num":"numtimesDS"}, inplace = True)

castData.sort_values(by='numtimesDS', ascending=False, inplace = True)

# calculation to create the actor wt

castData['actorWt'] = castData['numtimesDS']/castData['numtimesDS'].max().astype(np.float64)

castData.head()
# Import Production Company Table with Counts of Appearances

file = "../input/prodcotable/ProdCo_Data.csv"

prodcoData = pd.read_csv(file)

prodcoData.rename(columns = {"Prod_Co":"numtimesDS"}, inplace = True)

prodcoData.sort_values(by='numtimesDS', ascending=False, inplace = True)



prodcoData['prodcoWt'] = prodcoData['numtimesDS']/prodcoData['numtimesDS'].max().astype(np.float64)

prodcoData.head()
# Import Keyword Table with Counts of Appearances

file = "../input/weighttables/Keyword_Data.csv"

keywordData = pd.read_csv(file)

keywordData.rename(columns = {"# of Uses in DS":"numtimesDS"}, inplace = True)

keywordData.sort_values(by='numtimesDS', ascending=False, inplace = True)



keywordData['keywordWt'] = keywordData['numtimesDS']/keywordData['numtimesDS'].max().astype(np.float64)

keywordData.head()
# Import Crew Table with Counts of Appearances

file = "../input/weighttables/Crew_Data_clean.csv"

crewData = pd.read_csv(file)

crewData.rename(columns = {"Num":"numtimesDS"}, inplace = True)

crewData.sort_values(by='numtimesDS', ascending=False, inplace = True)



crewData['crewWt'] = crewData['numtimesDS']/crewData['numtimesDS'].max().astype(np.float64)

crewData.head()
# In order to apply weights to the field with JSON data, we first flatten, then convert to list, then iterate to produce a wt. 

# We are assuming that bigger "stars" appear in the dataset more times. 

#https://www.kaggle.com/rajuspartan/exploratory-data-analysis-with-reusable-functions

#Flatening JSON columns

def get_dictionary(s):

    try:

        d = eval(s)

    except:

        d = {}

    return d

train["castList"] = train.cast.map(lambda x: sorted([d['id'] for d in get_dictionary(x)])).map(lambda x: ','.join(map(str, x)))

test["castList"] = test.cast.map(lambda x: sorted([d['id'] for d in get_dictionary(x)])).map(lambda x: ','.join(map(str, x)))



train["keywordList"] = train.Keywords.map(lambda x: sorted([d['id'] for d in get_dictionary(x)])).map(lambda x: ','.join(map(str, x)))

test["keywordList"] = test.Keywords.map(lambda x: sorted([d['id'] for d in get_dictionary(x)])).map(lambda x: ','.join(map(str, x)))



train["prodcoList"] = train.production_companies.map(lambda x: sorted([d['id'] for d in get_dictionary(x)])).map(lambda x: ','.join(map(str, x)))

test["prodcoList"] = test.production_companies.map(lambda x: sorted([d['id'] for d in get_dictionary(x)])).map(lambda x: ','.join(map(str, x)))



train["crewList"] = train.crew.map(lambda x: sorted([d['id'] for d in get_dictionary(x)])).map(lambda x: ','.join(map(str, x)))

test["crewList"] = test.crew.map(lambda x: sorted([d['id'] for d in get_dictionary(x)])).map(lambda x: ','.join(map(str, x)))



train['castList'].head()
#The castList is a string, convert this to a list for easy function application

#Convert string to list, separate on commas

train['castList'] = train['castList'].apply(lambda x: x[1:-1].split(','))

test['castList'] = test['castList'].apply(lambda x: x[1:-1].split(','))



train['keywordList'] = train['keywordList'].apply(lambda x: x[1:-1].split(','))

test['keywordList'] = test['keywordList'].apply(lambda x: x[1:-1].split(','))



train['prodcoList'] = train['prodcoList'].apply(lambda x: x[1:-1].split(','))

test['prodcoList'] = test['prodcoList'].apply(lambda x: x[1:-1].split(','))



train['crewList'] = train['crewList'].apply(lambda x: x[1:-1].split(','))

test['crewList'] = test['crewList'].apply(lambda x: x[1:-1].split(','))



train['castList'].head()
#create sum of weights column

def weight(data):

    wt = 0

    for row in data:

        for x in row:

            wt = wt + int(x)

            

    return wt



train['castWt'] = train['castList'].apply(weight)

test['castWt'] = test['castList'].apply(weight)



train['keywordWt'] = train['keywordList'].apply(weight)

test['keywordWt'] = test['keywordList'].apply(weight)  



train['prodcoWt'] = train['prodcoList'].apply(weight)

test['prodcoWt'] = test['prodcoList'].apply(weight)



train['crewWt'] = train['crewList'].apply(weight)

test['crewWt'] = test['crewList'].apply(weight)



#Add columns for Team weights

train['teamWt'] = train['castWt']+train['crewWt']+train['prodcoWt']

test['teamWt'] = test['castWt']+test['crewWt']+test['prodcoWt']



train.head(2)
def proc_json_len(string):

    try:

        data = eval(string)

        return len(data)

    except:

        return 0



train['count_genre'] = train.genres.apply(proc_json_len)

train['count_country'] = train.production_countries.apply(proc_json_len)

train['count_company'] = train.production_companies.apply(proc_json_len)

train['count_splang'] = train.spoken_languages.apply(proc_json_len)

train['count_cast'] = train.cast.apply(proc_json_len)

train['count_crew'] = train.crew.apply(proc_json_len)

train['count_staff'] = train.count_cast + train.count_crew

train['count_keyword'] = train.Keywords.apply(proc_json_len)

test['count_genre'] = test.genres.apply(proc_json_len)

test['count_country'] = test.production_countries.apply(proc_json_len)

test['count_company'] = test.production_companies.apply(proc_json_len)

test['count_splang'] = test.spoken_languages.apply(proc_json_len)

test['count_cast'] = test.cast.apply(proc_json_len)

test['count_crew'] = test.crew.apply(proc_json_len)

test['count_staff'] = test.count_cast + test.count_crew

test['count_keyword'] = test.Keywords.apply(proc_json_len)

train.head()
#Collection or not?

train['belongs_to_collection'] = train['belongs_to_collection'].notna()

test['belongs_to_collection'] = test['belongs_to_collection'].notna()

test.head(2)
new_genres_train = pd.DataFrame(train['genres'])

new_genres_test = pd.DataFrame(test['genres'])

new_splang_train = pd.DataFrame(train['spoken_languages'])

new_splang_test = pd.DataFrame(test['spoken_languages'])

new_genres_train.head()
def get_dictionary(s):

    try:

        d = eval(s)

    except:

        d = {}

    return d

#https://www.kaggle.com/rajuspartan/exploratory-data-analysis-with-reusable-functions

#Flatening JSON columns

new_genres_train.genres = new_genres_train.genres.map(lambda x: sorted([d['name'] for d in get_dictionary(x)])).map(lambda x: ','.join(map(str, x)))

new_genres_test.genres = new_genres_test.genres.map(lambda x: sorted([d['name'] for d in get_dictionary(x)])).map(lambda x: ','.join(map(str, x)))



new_splang_train.spoken_languages = new_splang_train.spoken_languages.map(lambda x: sorted([d['iso_639_1'] for d in get_dictionary(x)])).map(lambda x: ','.join(map(str, x)))

new_splang_test.spoken_languages = new_splang_test.spoken_languages.map(lambda x: sorted([d['iso_639_1'] for d in get_dictionary(x)])).map(lambda x: ','.join(map(str, x)))

new_splang_train.head()

#We used this approach from Stack Overflow

#https://stackoverflow.com/questions/50394099/separate-columns-based-on-genre

#featurize the genre column

new_genres_train = new_genres_train['genres'].str.get_dummies(',')

new_genres_test = new_genres_test['genres'].str.get_dummies(',')

print(new_genres_test)
#featurize the spoken language column

# Is english in the spoken language?

def proc_json_len2(string):

        if ('en' in string):

            return 1

        else:

            return 0

  

new_splang_train['inEnglish'] = new_splang_train['spoken_languages'].apply(proc_json_len2)

new_splang_test['inEnglish'] = new_splang_test['spoken_languages'].apply(proc_json_len2)

new_splang_train.columns

new_splang_train.head()

#add genres back to data (join)

train = pd.concat([train, new_genres_train], axis = 1, sort = False)

test = pd.concat([test, new_genres_test], axis = 1, sort = False)
#add spoken languages back to data (join)

train = pd.concat([train, new_splang_train], axis = 1, sort = False)

test = pd.concat([test, new_splang_test], axis = 1, sort = False)

train.head(2)
train.columns
#Select subset of columns

train = train[['id','belongs_to_collection','budget', 'original_language', 'popularity', 'status','year', 'month', 'Action', 'Adventure', 'Animation', 'Comedy',

       'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'Foreign',

       'History', 'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction',

       'TV Movie', 'Thriller', 'War', 'Western', 'count_genre',

       'count_country', 'count_company', 'count_splang', 'count_cast',

       'count_crew', 'count_staff', 'count_keyword', "castWt",'prodcoWt','keywordWt', 'crewWt','teamWt','inEnglish', "revenue"]]

test = test[['id','belongs_to_collection','budget', 'original_language', 'popularity', 'status','year', 'month', 'Action', 'Adventure', 'Animation', 'Comedy',

       'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'Foreign',

       'History', 'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction',

       'Thriller', 'War', 'Western', 'count_genre',

       'count_country', 'count_company', 'count_splang', 'count_cast',

       'count_crew', 'count_staff', 'count_keyword', "castWt",'prodcoWt','keywordWt', 'crewWt','teamWt','inEnglish']]



train.head()
data = train.copy()

data2 = test.copy()

data_binary_encoded = pd.get_dummies(data, columns=["belongs_to_collection", "status"])

data2_binary_encoded = pd.get_dummies(data2, columns=["belongs_to_collection", "status"])

#Select subset of columns

train = data_binary_encoded[['id','belongs_to_collection_True', 'budget', 'original_language', 'popularity', 'status_Released', 'year', 'month', 'Action', 'Adventure', 'Animation', 'Comedy',

       'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'Foreign',

       'History', 'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction', 'Thriller', 'War', 'Western', 'count_genre',

       'count_country', 'count_company', 'count_splang', 'count_cast',

       'count_crew', 'count_staff', 'count_keyword', "castWt",'prodcoWt','keywordWt', 'crewWt','teamWt','inEnglish', 'revenue']]

test = data2_binary_encoded[['id','belongs_to_collection_True', 'budget', 'original_language', 'popularity', 'status_Released', 'year', 'month', 'Action', 'Adventure', 'Animation', 'Comedy',

       'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'Foreign',

       'History', 'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction','Thriller', 'War', 'Western', 'count_genre',

       'count_country', 'count_company', 'count_splang', 'count_cast',

       'count_crew', 'count_staff', 'count_keyword', "castWt", 'prodcoWt','keywordWt', 'crewWt','teamWt','inEnglish']]



train.head()
train['language_enc'] = pd.factorize(train['original_language'])[0]

test['language_enc'] = pd.factorize(test['original_language'])[0]

#This is just a list of all of the languages listed

catenc = pd.factorize(train['original_language'])



train.head()
X = train[['id','belongs_to_collection_True', 'budget', 'language_enc', 'popularity','year', 'month', 'count_genre',

       'count_country', 'count_company', 'count_splang', 'count_cast',

       'count_crew', 'count_staff', 'count_keyword', "castWt",'prodcoWt','keywordWt', 'crewWt','teamWt','inEnglish']]

y = train['revenue'].values.reshape(-1,1)

print(X.shape, y.shape)
#looking at a subset of attributes only

train2 = train[['belongs_to_collection_True', 'budget', 'popularity', 'year','month', 'revenue']]

train2.head()
#explore datasets

import seaborn as sns; sns.set(style="ticks", color_codes=True)

pairplots = sns.pairplot(train2, diag_kind = 'kde', hue = "month",palette = "Accent", plot_kws = {'alpha': 0.6, 's': 80, 'edgecolor': 'k'},

             height = 4)
# Use train_test_split to create training and testing data from our "train" dataset

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

print(X_train.shape, y_train.shape)

print(X_test.shape, y_test.shape)
from sklearn.linear_model import LinearRegression

#create the linear regression object

model = LinearRegression(fit_intercept = True)

#train the model

model.fit(X_train, y_train)

training_score = model.score(X_train, y_train)

print(f"R2 Score: {training_score}")
# Plot the Residuals for the Training and Testing data

#Residuals are the difference between the true values of y and the predicted values of y.

#make predictions using the testing set

prediction = model.predict(X_test)

#plot residuals

plt.scatter(model.predict(X_train), model.predict(X_train) - y_train, c="black", label="Training Data")

plt.scatter(prediction, prediction - y_test, c="grey", label="Testing Data")

plt.legend()

plt.hlines(y=0, xmin=y_test.min(), xmax=y_test.max())

plt.title("Residual Plot")

plt.show()



MSE = mean_squared_error(y_test, prediction)

r2 = model.score(X_test, y_test)

### END SOLUTION

print(f"MSE: {MSE}, R2: {r2}")
# Fit regression model

params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,

          'learning_rate': .01, 'loss': 'ls'} 

clf = ensemble.GradientBoostingRegressor(**params)

predictions2 = clf.fit(X_train,y_train)

training_score = clf.score(X_train, y_train)

print(f"Training Score: {training_score}")
# Plot the Residuals for the Training and Testing data

### BEGIN SOLUTION

predictions2 = np.expand_dims(clf.predict(X_test), axis = 1)

plt.scatter((np.expand_dims(clf.predict(X_train), axis = 1)), (np.expand_dims(clf.predict(X_train), axis = 1)) - y_train, c="black", label="Training Data")

plt.scatter(predictions2, predictions2 - y_test, c="grey", label="Testing Data")

plt.legend()

plt.hlines(y=0, xmin=y_test.min(), xmax=y_test.max())

plt.title("Residual Plot")

plt.show()



MSE = mean_squared_error(y_test, predictions2)

r2 = clf.score(X_test, y_test)

print(f"MSE: {MSE}, R2: {r2}")
#Predictions for the test data

revenue_predictions = clf.predict(X_test)

gbr_predictions = pd.DataFrame(revenue_predictions, columns = ['revenue'])

gbr_predictions.head()
test2 = pd.concat([test, gbr_predictions], axis = 1, join_axes = [test.index])

#look at top values only

test2 = test2[['belongs_to_collection_True', 'budget', 'popularity', 'year','month', 'revenue']]

test2.head()
#explore datasets

import seaborn as sns; sns.set(style="ticks", color_codes=True)

pairplots = sns.pairplot(test2, diag_kind = 'kde', hue = "month",palette = "Accent", plot_kws = {'alpha': 0.6, 's': 80, 'edgecolor': 'k'},

             height = 4)