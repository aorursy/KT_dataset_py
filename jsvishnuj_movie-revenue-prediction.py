# importing the essential libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import ast

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
train_data = pd.read_csv('/kaggle/input/tmdb-box-office-prediction/train.csv')

train_data.head()
# let us see the types of data in the training set

train_data.dtypes
training_data = train_data.drop(['id', 'belongs_to_collection', 'homepage', 'imdb_id', 'original_title',

                                'overview', 'poster_path', 'release_date', 'tagline', 'title',

                                'Keywords', 'cast', 'crew'], axis = 1)

training_data.head()
# checking the missing values

training_data.isnull().sum()
# filling the missing values

training_data = training_data.fillna('0')

training_data.isnull().sum()
def feature_engineering(series):

    # Feature engineering for genres

    string_list = []

    for i in series:

        string = []

        if (i != '0'):

            o = ast.literal_eval(i)

            for i in o:

                for j in i.items():

                    if (j[0] == 'name'):

                        string.append(j[1])

        string_list.append(' + '.join(string))

    return LabelEncoder().fit_transform(string_list)
# Feature Engineering

training_data.index = train_data['id']

training_data['genres'] = feature_engineering(training_data['genres'])

training_data['production_companies'] = feature_engineering(training_data['production_companies'])

training_data['production_countries'] = feature_engineering(training_data['production_countries'])

training_data['spoken_languages'] = feature_engineering(training_data['spoken_languages'])

training_data['original_language'] = LabelEncoder().fit_transform(training_data['original_language'])

training_data['status'] = LabelEncoder().fit_transform(training_data['status'])
training_data.head()
sns.heatmap(training_data.corr())
plt.plot(training_data['revenue'], training_data['budget'], 'o', label = 'revenue VS budget')

plt.legend()
plt.plot(training_data['revenue'], training_data['popularity'], 'o', label = 'revenue VS popularity')

plt.legend()
X = training_data.drop(['revenue'], axis = 1)

y = training_data['revenue']



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 101)
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)



validations = model.predict(X_val)



print(np.sqrt(mean_squared_error(validations, y_val)))
test_data = pd.read_csv('/kaggle/input/tmdb-box-office-prediction/test.csv')

test_data.head()
testing_data = test_data.drop(['id', 'belongs_to_collection', 'homepage', 'imdb_id', 'original_title',

                                'overview', 'poster_path', 'release_date', 'tagline', 'title',

                                'Keywords', 'cast', 'crew'], axis = 1)
# filling the missing values

testing_data = testing_data.fillna('0')

testing_data.isnull().sum()
# Feature Engineering

testing_data.index = test_data['id']

testing_data['genres'] = feature_engineering(testing_data['genres'])

testing_data['production_companies'] = feature_engineering(testing_data['production_companies'])

testing_data['production_countries'] = feature_engineering(testing_data['production_countries'])

testing_data['spoken_languages'] = feature_engineering(testing_data['spoken_languages'])

testing_data['original_language'] = LabelEncoder().fit_transform(testing_data['original_language'])

testing_data['status'] = LabelEncoder().fit_transform(testing_data['status'])
testing_data.head()
predictions = model.predict(testing_data)
# Now creating a dataset and submitting

submission = pd.DataFrame({'id' : test_data['id'], 'revenue' : predictions})

submission.head()
submission.to_csv('submission.csv', index = False)