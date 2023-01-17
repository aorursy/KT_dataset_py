# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Import useful libraries



import time

import re

import string

from numpy import mean



from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.preprocessing import FunctionTransformer

from sklearn.impute import SimpleImputer

from sklearn.metrics import mean_squared_log_error, make_scorer, mean_squared_error

from sklearn.model_selection import train_test_split



from sklearn.linear_model import LinearRegression

from sklearn.naive_bayes import MultinomialNB

from lightgbm import LGBMRegressor

from xgboost import XGBRegressor

from sklearn.model_selection import RandomizedSearchCV



import warnings

warnings.filterwarnings('ignore')
# Read dataset



train_data = pd.read_excel('/kaggle/input/predict-book-prices/train.xlsx')

test_data = pd.read_excel('/kaggle/input/predict-book-prices/test.xlsx')

sample_submission = pd.read_excel('/kaggle/input/predict-book-prices/sample_submission.xlsx')

train_data.columns = train_data.columns.str.lower().str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

test_data.columns = test_data.columns.str.lower().str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
print('Train Data Shape: ', train_data.shape)

print('Test Data Shape: ', test_data.shape)

train_data.head()
train_data.nunique()
# Merge the training set and test set



pd.set_option('display.max_rows', 200)

train_data['type'] = 'train'

test_data['type'] = 'test'

master_data = pd.concat([train_data, test_data])

unique_titles = pd.DataFrame(master_data.title.unique()).reset_index()

unique_titles.columns = ['id', 'title']

master_data = master_data.merge(unique_titles, on = 'title', how = 'left')

#master_data = master_data.sort_values(by = ['id'], ascending = [True])

master_data.head()
# Reviews handling



master_data['reviews'] = master_data['reviews'].apply(lambda x: x.split(' ')[0])

master_data['reviews'] = master_data['reviews'].astype(np.float16)
# Ratings handling



master_data['ratings'] = master_data['ratings'].apply(lambda x: x.split(' ')[0])

master_data['ratings'] = master_data['ratings'].apply(lambda x: int(x.replace(',', '')))
# Publication year and age of editions



master_data['year'] = master_data['edition'].str[-4:]



# Random publication year for some books

master_data['year'] = master_data['year'].apply(lambda x: re.sub("[^0-9]", 'NA', x))

master_data['year'] = master_data['year'].apply(lambda x: x.replace('NA', '0'))

master_data['year'] = master_data['year'].astype(np.int16)



master_data['age'] = 2019 - master_data['year']



master_data.loc[(master_data['year'] == 0), 'year'] = np.NaN

avg_age = master_data['age'].mean()

master_data.loc[(master_data['year'].isnull()), 'age'] = avg_age



master_data.head()
# loading stop words from nltk library



import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))



def nlp_preprocessing(total_text):

    if type(total_text) is not int:

        string = ""

        for word in total_text.split():

        # if the word is a not a stop word then retain that word from the data

            if not word in stop_words:

                string += word + " "

        

    return string
# text processing - remove stop words



start_time = time.process_time()

for column in ['title', 'author', 'edition', 'synopsis', 'genre', 'bookcategory']:

    master_data[column] = master_data[column].apply(lambda x: nlp_preprocessing(x))

    master_data[column] = master_data[column].str.lower()

    master_data[column] = master_data[column].astype(str).apply(lambda x : re.sub("[^A-Za-z]"," ",x))

    master_data[column] = master_data[column].apply(lambda x:''.join([i for i in x if i not in string.punctuation]))

print('Time took for preprocessing the text :',time.process_time() - start_time, "seconds")
# Both genre and bookcategory columns have categorized structure



master_data['genre'] = master_data['genre'].map(master_data['genre'].value_counts())

master_data['bookcategory'] = master_data['bookcategory'].map(master_data['bookcategory'].value_counts())
# Create features for Binding and Imported versions 



master_data['binding'] = master_data['edition'].apply(lambda x: np.where('paperback' in x, -1, 1))

master_data['imported'] = master_data['edition'].apply(lambda x: np.where('import' in x, 1, -1))

master_data['synopsis'] = master_data['synopsis'] + " " + master_data['title']

master_data = master_data.drop(['title'], axis = 1)
# Separate train and test data



train_data = master_data.loc[master_data['type'] == 'train']

test_data = master_data.loc[master_data['type'] == 'test']



train_data = train_data.drop(['id', 'type', 'author', 'edition'], axis = 1)

test_data = test_data.drop(['id', 'price', 'type', 'author', 'edition'], axis = 1)

train_data.head()
X = train_data.drop(['price'],axis = 1)

y = train_data['price']

y = np.log1p(y)



# Split the data



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 22)
get_numeric_data = FunctionTransformer(lambda x: x[['reviews','ratings','genre', 'bookcategory','binding','year']], validate = False)

get_text_data = FunctionTransformer(lambda x: x['synopsis'],validate = False)
numeric_pipeline = Pipeline([('selector', get_numeric_data),])

text_pipeline = Pipeline([('selector', get_text_data), ('vectorizer', CountVectorizer()),])
params = {

    'clf__n_estimators' : [100, 200],

    #'clf__max_depth' : [3,4,5,6,7],

    'clf__learning_rate': [0.01, 0.1, 0.01],

    'clf__reg_lambda': list(np.arange(0.1, 0.9, 0.1)),

    #'clf__colsample_bytree' : list(np.arange(0.1,0.8,0.1)),

    'clf__importance_type': ['gain', 'weight', 'cover', 'total_gain', 'total_cover'],

    'clf__booster': ['gbtree', 'gblinear', 'dart']

}
pipeline = Pipeline([('union', FeatureUnion([('numeric', numeric_pipeline), ('text', text_pipeline)])),

               ('clf', LGBMRegressor(verbosity = 1, objective = 'regression'))])
def get_score(y_val, y_pred):

    return np.sqrt(mean_squared_log_error(y_pred, y_val))

    

criteria = make_scorer(get_score, greater_is_better = False)

grid = RandomizedSearchCV(pipeline, param_distributions = params, n_iter = 15, cv = 5, scoring = criteria)

grid.fit(X_train, y_train)

print(grid.best_params_)

print('Best Score: ', grid.best_score_)
y_preds = grid.predict(X_test)

print('Mean Squared Error: ', mean_squared_error(y_preds, y_test))



print('Validation set score: ', 1 - np.sqrt(np.square(np.log10(y_preds +1) - np.log10(y_test +1)).mean()))
grid.fit(X, y)

Preds = np.expm1(grid.predict(test_data))

submission = pd.DataFrame({'Price': Preds})

submission['Price'] = Preds

submission.head(10)
submission.to_excel('submission_v2.xlsx', index = False)