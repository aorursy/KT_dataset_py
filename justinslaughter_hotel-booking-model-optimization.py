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
#Let's take a look at the data

df = pd.read_csv('/kaggle/input/hotel-booking-demand/hotel_bookings.csv')



df.head()
df.reservation_status.unique()
df[['is_canceled','reservation_status']].head(20)
import pandas as pd

import numpy as np

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.feature_extraction.text import CountVectorizer

from lightgbm import LGBMClassifier

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.base import BaseEstimator,TransformerMixin

from sklearn.preprocessing import FunctionTransformer



from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel, f_classif



def pipe_frame(df,classifier):

    #drop target, leaky features, and country column because it gave me trouble

    X = df.drop(['is_canceled', 'reservation_status', 'reservation_status_date','country'],axis=1) 

    

    #I'm predicting whether or not booking will be canceled

    y = df.is_canceled

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)



    #Get numerical and categorical columns

    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns

    categorical_cols = X_train.select_dtypes(include=['object']).columns



    #Insert 0 in numerical columns when empty

    numerical_transformer = SimpleImputer(strategy='constant')



    #Fill in with most frequent value, one hot encode, and ignore unknowns

    categorical_transformer = Pipeline(steps=[

        ('imputer', SimpleImputer(strategy='most_frequent')),

        ('onehot', OneHotEncoder(handle_unknown='ignore'))

    ])



    #Combine transformers together

    preprocessor = ColumnTransformer(

        transformers=[

            ('num', numerical_transformer, numerical_cols),

            ('cat', categorical_transformer, categorical_cols)

        ])

    

    #After preprocessing, select 10 best features using f_classif, then use provided classifier to predict target

    pipeline = Pipeline(steps=[

                            ('preprocessor', preprocessor),

                            ('skb',SelectKBest(f_classif, k=10)),

                            ('classifier', classifier)

                                ])



    #Fit pipeline to training data

    pipeline.fit(X_train,y_train)

    

    #Score pipeline using validation data

    score = pipeline.score(X_valid, y_valid)

    

    return score
#Will use process_time from time to get timepoints. Subtracting start time from stop time gives time process took in seconds

from time import process_time 



#Read Data as Data Frame

df = pd.read_csv('/kaggle/input/hotel-booking-demand/hotel_bookings.csv')



#Convert the months column from Strings to Numbers instead of letting an encoder randomly assign labels

def months_toint(input_df):

    #Have to explicitly make a copy or df will be modified

    copy = input_df.copy()

    #Dictionary to use to convert months

    month_map = {'January': 1,'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,

             'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}

    copy.loc[:,'arrival_date_month'] = copy.arrival_date_month.map(month_map)

    return copy



classifiers = {'lgb': LGBMClassifier(random_state=0),

               'forest': RandomForestClassifier(random_state=0),

               'adaboost': AdaBoostClassifier(random_state=0),

               'gradboost': GradientBoostingClassifier(random_state=0)

              }

for name, classifier in classifiers.items(): #Loop through classifiers and score them

    start = process_time()

    score = pipe_frame(months_toint(df), classifier)

    stop = process_time()

    print('{}: '.format(name),score, ', Elapsed Time: {}'.format(stop-start))
classifiers = {}

#This time we're looping through options for number of leaves

for num_leaves in [16,32,64,128, 256]:

    classifiers[str(num_leaves)] = LGBMClassifier(num_leaves= num_leaves, random_state=0)



for name, classifier in classifiers.items():

    start = process_time()

    score = pipe_frame(months_toint(df), classifier)

    stop = process_time()

    print('Number of Leaves {}: '.format(name),score, ', Elapsed Time: {}'.format(stop-start))
classifiers = {}

#This time we're looping through options for number of estimators

for n_estimators  in [50,100,200,400, 800]:

    classifiers[str(n_estimators)] = LGBMClassifier(n_estimators= n_estimators, random_state=0)



for name, classifier in classifiers.items():

    start = process_time()

    score = pipe_frame(months_toint(df), classifier)

    stop = process_time()

    print('Number of Estimators {}: '.format(name),score, ', Elapsed Time: {}'.format(stop-start))
classifiers = {}

#Last loop wasn't specific enough, will loop through values around 200 to see if I can get better results

for n_estimators  in [100,150,200,250,300]:

    classifiers[str(n_estimators)] = LGBMClassifier(n_estimators= n_estimators, random_state=0)



for name, classifier in classifiers.items():

    start = process_time()

    score = pipe_frame(months_toint(df), classifier)

    stop = process_time()

    print('Number of Estimators {}: '.format(name),score, ', Elapsed Time: {}'.format(stop-start))
classifier = LGBMClassifier(num_leaves=128, n_estimators=250, nrandom_state=0)



start = process_time()

score = pipe_frame(months_toint(df), classifier)

stop = process_time()

print('Current Model Score: {}'.format(score), ', Elapsed Time: {}'.format(stop-start))