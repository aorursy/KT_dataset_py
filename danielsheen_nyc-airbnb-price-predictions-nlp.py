#from pandas import read_csv

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from pandas.plotting import scatter_matrix

from matplotlib import pyplot
df = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

df.head()
columns = df.columns

columns, len(columns)
df.shape
# checking data types

df.info()
# checking # of NaN in each column

if(df.isnull().values.any() == True):

    for name in columns:

        print('%s: %i' %(name, df[name].isnull().sum()) )
df.drop(columns = ['id', 'host_id', 'host_name', 'last_review'], inplace = True)

df.head()
df_1 = df[df['availability_365'] > 0]

df_1
df_1['reviews_per_month'] = df['reviews_per_month'].fillna(0)

df_1
df = df_1.copy()

df['name'] = df_1['name'].fillna('to') # we are using 'to' bc it is a stopword -> explained later
# checking our NaN values again

df.isnull().values.any() 
df.drop(columns = ['latitude', 'longitude'], inplace = True)

df.head()
df['room_type'].unique()
df['neighbourhood_group'].unique()
df['neighbourhood'].unique()
# taking columns with an object type only

object_cols = df.select_dtypes(include = [object])



# dropping the name column bc we do not want to perform encoding on it

object_cols.drop(columns = ['name'], inplace = True)

object_cols.head()
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()

object_cols = encoder.fit_transform(object_cols)
# creating a function to remove stop words to decrease our runtime

from nltk.corpus import stopwords 

from nltk.tokenize import word_tokenize 

def stop_words(text):

    stop_words = set(stopwords.words('english'))

    word_tokens = word_tokenize(text)

    filtered_sentence = [w for w in word_tokens if not w in stop_words]

    return filtered_sentence
from langdetect import detect

from langdetect.lang_detect_exception import LangDetectException



description = df['name'].astype(str)



errors = []

not_english = []

english = 0

other = 0

for title in description:

    try:

        #print(title)

        if detect(title) == 'en':

            english += 1

        else:

            not_english += [title]

            other += 1

    except LangDetectException:

        other += 1

        errors += [title]

english, other
not_english
def grammar(string):

    # add whatever else you think you might have to account for

    result = str(string)

    result = result.replace('/', ' ')

    result = result.replace('*', ' ')

    result = result.replace('&', ' ')

    result = result.replace('>', ' ')

    result = result.replace('<', ' ')

    result = result.replace('-', ' ')

    result = result.replace('...', ' ')

    result = result.replace('@', ' ')

    result = result.replace('#', ' ')

    result = result.replace('-', ' ')

    result = result.replace('$', ' ')

    result = result.replace('%', ' ')

    result = result.replace('+', ' ')

    result = result.replace('=', ' ')

    

    return result
import langid

desc = description.apply(grammar)

langid.set_languages(['en', 'es', 'zh'])

not_en = []

not_en_index = []

i = 0

for title in desc:

    if langid.classify(title)[0] != 'en':

        not_en += [title]

        not_en_index += [desc.index[i]]

    i += 1

    

len(not_en)
not_en
for i in desc.index:

    if desc[i] in not_en:

        desc[i] = ''
description = desc.apply(stop_words)

description 
# creating a function to convert a list of strings to single string

def to_single_string(list_of_strings):

    result = ''

    for string in list_of_strings:

        result += ' ' +string

    return result



# applying above function

description = description.apply(to_single_string)

description
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer



sentiment_analyzer = SentimentIntensityAnalyzer()



def sentiment_score(string):

        result = sentiment_analyzer.polarity_scores(string)

        return result
sentiment = description.apply(sentiment_score)

sentiment
# method 1

def compound_score(sent):

    return sent.get('compound')



sentiment_M1 = sentiment.apply(compound_score)

sentiment_M1
# method 2

def polarity(sent):

    compound = sent.get('compound')

    if(compound >= 0.05):

        return 1

    elif(compound <= -0.05):

        return -1

    return 0



sentiment_M2 = sentiment.apply(polarity)

sentiment_M2
df.columns
# creating temporary df

temporary = pd.DataFrame()

temporary['location'] = df['neighbourhood']

temporary['sent_M1'] = sentiment_M1.to_frame()

temporary['sent_M2'] = sentiment_M2.to_frame()

temporary['name'] = description.to_frame()

temporary
# removing rows that are in not_en, with not_en_index (which was obtained in the same block of code as not_en)

temporary.drop(index = not_en_index, inplace = True)

temporary
neighborhood_sent = temporary.groupby(['location']).mean()

neighborhood_sent
def polarity_range(score):

    if(score >= 0.05):

        return 1

    elif(score <= -0.05):

        return -1

    return 0





# obtaining our different sentiment scores

nhood_sent_M1 = neighborhood_sent['sent_M1']

nhood_sent_M2 = neighborhood_sent['sent_M2']



# applying our function to sent_M2 to turn it into -1, 0 and +1 only

nhood_sent_M2 = nhood_sent_M2.apply(polarity_range)
nhood_sent_M1
sent_m1 = sentiment_M1.copy()

sent_m2 = sentiment_M2.copy()



for index in not_en_index:

    sent_m1[index] = nhood_sent_M1[df['neighbourhood'][index]]

    sent_m2[index] = nhood_sent_M2[df['neighbourhood'][index]]
df.head()
df.drop(columns = ['number_of_reviews',	'reviews_per_month'], inplace = True)

df.head()
df.drop(columns = ['name', 'neighbourhood_group', 'neighbourhood', 'room_type'], inplace = True)

df.head()
# creating a copy of df for later

df_copy = df.copy()

#################################

y = df['price']

df.drop(columns = ['price'], inplace = True)
df_sent_m1 = df.copy()

df_sent_m2 = df.copy()



df_sent_m1['sentiment'] = sent_m1

df_sent_m2['sentiment'] = sent_m2





# adding onto our copy of df

df_copy['sent_m1'] = sent_m1

df_copy['sent_m2'] = sent_m2
corr_matrix_m1 = df_copy.corr()

corr_matrix_m1["price"].sort_values(ascending = False)
object_cols
# we will change df_sent_m1 and _m2 into sparse matrices as this data type get processed better

from scipy import sparse

#from scipy.sparse import hstack



X_m1 = sparse.hstack([object_cols, sparse.csr_matrix(df_sent_m1.to_numpy())])

X_m2 = sparse.hstack([object_cols, sparse.csr_matrix(df_sent_m2.to_numpy())])
from sklearn.model_selection import train_test_split

X_train_m1, X_test_m1, y_train_m1, y_test_m1 = train_test_split(X_m1, y, test_size = 0.2, random_state = 42)

X_train_m2, X_test_m2, y_train_m2, y_test_m2 = train_test_split(X_m2, y, test_size = 0.2, random_state = 42)



y_train_m1 = y_train_m1.tolist()

y_train_m2 = y_train_m2.tolist()

y_test_m1 = y_test_m1.tolist()

y_test_m2 = y_test_m2.tolist()
temp_train_m1 = X_train_m1[:, X_train_m1.shape[1]-4:].toarray()

temp_train_m2 = X_train_m2[:, X_train_m2.shape[1]-4: X_test_m2.shape[1]-1].toarray()



temp_test_m1 = X_test_m1[:, X_test_m1.shape[1]-4:].toarray()

temp_test_m2 = X_test_m2[:, X_test_m2.shape[1]-4: X_test_m2.shape[1]-1].toarray()



from sklearn.preprocessing import StandardScaler

scaler_m1 = StandardScaler().fit(temp_train_m1)

scaler_m2 = StandardScaler().fit(temp_train_m2)



temp_train_m1 = scaler_m1.transform(temp_train_m1)

temp_train_m2 = scaler_m2.transform(temp_train_m2)



temp_test_m1 = scaler_m1.transform(temp_test_m1)

temp_test_m2 = scaler_m2.transform(temp_test_m2)
X_train_m1[:,X_train_m1.shape[1]-4:] = sparse.csr_matrix(temp_train_m1)

X_train_m2[:,X_train_m2.shape[1]-4: X_test_m2.shape[1]-1] = sparse.csr_matrix(temp_train_m2)



X_test_m1[:,X_test_m1.shape[1]-4: ] = sparse.csr_matrix(temp_test_m1)

X_test_m2[:,X_test_m2.shape[1]-4: X_test_m2.shape[1]-1] = sparse.csr_matrix(temp_test_m2)
from sklearn.svm import SVR

from sklearn.linear_model import Lasso

from sklearn.linear_model import Ridge

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from keras.models import Sequential

from keras.layers import Dense

from keras import regularizers

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import KFold

from keras.initializers import RandomNormal



#### FOR NEURAL NETWORK ####

initializer = RandomNormal(mean=0., stddev=1.)



def neural_net():

    model = Sequential()

    model.add(Dense(int(2/3 * X_test_m1.shape[1]), kernel_initializer = initializer, activation = 'relu', input_dim = X_test_m1.shape[1]))

    model.add(Dense(int(4/9 * X_test_m1.shape[1]), kernel_initializer = initializer, activation = 'relu'))

    model.add(Dense(1, kernel_initializer = 'normal', activation = 'relu'))

    model.compile(optimizer = 'SGD', loss = 'mse', metrics = ['mae'])

    return model

############################    



models = []

models += [['SVM', SVR(kernel = 'linear')]]

models += [['Lasso', Lasso(alpha = 0.9, normalize = False, selection = 'cyclic')]]

models += [['Ridge', Ridge(alpha = 0.9, normalize = False, solver = 'auto')]]

models += [['Linear', LinearRegression(normalize = False)]]

models += [['Random Forests', RandomForestClassifier(n_estimators = 100, max_features = X_test_m1.shape[1], random_state = 42, max_depth = 9)]]



# for the k fold cross validation

kfold = KFold(n_splits = 10, random_state = 1, shuffle = True)
from sklearn.model_selection import cross_val_score



result_m1 =[]

names = []



for name, model in models:

    cv_score = -1 * cross_val_score(model, X_train_m1, y_train_m1, cv = kfold, scoring = 'neg_mean_absolute_error')

    result_m1 +=[cv_score]

    names += [name]

    print('%s: %f (%f)' % (name,cv_score.mean(), cv_score.std()))
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold



estimator = KerasRegressor(build_fn = neural_net, epochs = 1000, batch_size = 2000, verbose=0)

results = -1 * cross_val_score(estimator, X_train_m1, y_train_m1, cv = kfold, scoring = 'neg_mean_absolute_error')

print("Neural net: %f (%f) MSE" % (results.mean(), results.std()))
result_m2 =[]

names = []



for name, model in models:

    cv_score = -1 * cross_val_score(model, X_train_m2, y_train_m2, cv = kfold, scoring = 'neg_mean_absolute_error')

    result_m2 +=[cv_score]

    names += [name]

    print('%s: %f (%f)' % (name,cv_score.mean(), cv_score.std()))
estimator = KerasRegressor(build_fn = neural_net, epochs = 1000, batch_size = 2000, verbose=0)

results = -1 * cross_val_score(estimator, X_train_m2, y_train_m2, cv = kfold, scoring = 'neg_mean_absolute_error')

print("Neural net: %f (%f) MSE" % (results.mean(), results.std()))
model = SVR(kernel = 'linear').fit(X_train_m2, y_train_m2)



# obtaining predictions

predictions = model.predict(X_test_m2)
from sklearn.metrics import mean_absolute_error

from sklearn.metrics import r2_score

print("Final mean squared error: %f      R^2 value: %f" %(mean_absolute_error(y_test_m2, predictions), r2_score(y_test_m2, predictions)))