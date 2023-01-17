# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import datetime

import numpy as np

import matplotlib.pyplot as plt

import os

import pandas as pd

import seaborn as sns



from sklearn.preprocessing import MinMaxScaler
train = pd.read_csv('../input/tmdb-box-office-prediction/train.csv')

test = pd.read_csv('../input/tmdb-box-office-prediction/test.csv')

dataset = train.copy()

dataset1 = test.copy()
dataset.loc[dataset['id'] == 16, 'revenue'] = 192864

dataset.loc[dataset['id'] == 313, 'revenue'] = 12000000       

dataset.loc[dataset['id'] == 451, 'revenue'] = 12000000

dataset.loc[dataset['id'] == 1865, 'revenue'] = 25000000 

dataset.loc[dataset['id'] == 2491, 'revenue'] = 6800000

dataset.loc[dataset['id'] == 90, 'budget'] = 30000000                

dataset.loc[dataset['id'] == 118, 'budget'] = 60000000  

dataset.loc[dataset['id'] == 149, 'budget'] = 18000000  

dataset.loc[dataset['id'] == 464, 'budget'] = 20000000       

dataset.loc[dataset['id'] == 470, 'budget'] = 13000000      

dataset.loc[dataset['id'] == 513, 'budget'] = 930000          

dataset.loc[dataset['id'] == 797, 'budget'] = 8000000       

dataset.loc[dataset['id'] == 819, 'budget'] = 90000000       

dataset.loc[dataset['id'] == 850, 'budget'] = 90000000  

dataset.loc[dataset['id'] == 1112, 'budget'] = 7500000  

dataset.loc[dataset['id'] == 1131, 'budget'] = 4300000      

dataset.loc[dataset['id'] == 1359, 'budget'] = 10000000      

dataset.loc[dataset['id'] == 1542, 'budget'] = 1500000          

dataset.loc[dataset['id'] == 1542, 'budget'] = 15800000      

dataset.loc[dataset['id'] == 1571, 'budget'] = 4000000        

dataset.loc[dataset['id'] == 1714, 'budget'] = 46000000       

dataset.loc[dataset['id'] == 1721, 'budget'] = 17500000            

dataset.loc[dataset['id'] == 2268, 'budget'] = 17500000      

dataset.loc[dataset['id'] == 2602, 'budget'] = 31000000

dataset.loc[dataset['id'] == 2612, 'budget'] = 15000000

dataset.loc[dataset['id'] == 2696, 'budget'] = 10000000

dataset.loc[dataset['id'] == 2801, 'budget'] = 10000000

dataset.loc[dataset['id'] == 3889, 'budget'] = 15000000       

dataset.loc[dataset['id'] == 6733, 'budget'] = 5000000     

dataset.loc[dataset['id'] == 3197, 'budget'] = 8000000     

dataset.loc[dataset['id'] == 6683, 'budget'] = 50000000     

dataset.loc[dataset['id'] == 5704, 'budget'] = 4300000     

dataset.loc[dataset['id'] == 6109, 'budget'] = 281756      

dataset.loc[dataset['id'] == 7242, 'budget'] = 10000000     

dataset.loc[dataset['id'] == 7021, 'budget'] = 17540562

dataset.loc[dataset['id'] == 5591, 'budget'] = 4000000      

dataset.loc[dataset['id'] == 4282, 'budget'] = 20000000

dataset.loc[dataset['id'] == 391, 'runtime'] = 86 

dataset.loc[dataset['id'] == 592, 'runtime'] = 90 

dataset.loc[dataset['id'] == 925, 'runtime'] = 95 

dataset.loc[dataset['id'] == 978, 'runtime'] = 93 

dataset.loc[dataset['id'] == 1256, 'runtime'] = 92 

dataset.loc[dataset['id'] == 1542, 'runtime'] = 93

dataset.loc[dataset['id'] == 1875, 'runtime'] = 86 

dataset.loc[dataset['id'] == 2151, 'runtime'] = 108

dataset.loc[dataset['id'] == 2499, 'runtime'] = 108 

dataset.loc[dataset['id'] == 2646, 'runtime'] = 98

dataset.loc[dataset['id'] == 2786, 'runtime'] = 111

dataset.loc[dataset['id'] == 2866, 'runtime'] = 96

dataset.loc[dataset['id'] == 4074, 'runtime'] = 103 

dataset.loc[dataset['id'] == 4222, 'runtime'] = 93

dataset.loc[dataset['id'] == 4431, 'runtime'] = 100 

dataset.loc[dataset['id'] == 5520, 'runtime'] = 86 

dataset.loc[dataset['id'] == 5845, 'runtime'] = 83 

dataset.loc[dataset['id'] == 5849, 'runtime'] = 140

dataset.loc[dataset['id'] == 6210, 'runtime'] = 104

dataset.loc[dataset['id'] == 6804, 'runtime'] = 145 

dataset.loc[dataset['id'] == 7321, 'runtime'] = 87

dataset.dropna()

dataset.loc[dataset.release_date.isnull(), 'release_date'] = '05/01/2000'



dataset['release_year'] = dataset.release_date.str.extract('\S+/\S+/(\S+)', expand=False).astype(np.int16)

dataset['release_month'] = dataset.release_date.str.extract('(\S+)/\S+/\S+', expand=False).astype(np.int16)

dataset['release_day'] = dataset.release_date.str.extract('\S+/(\S+)/\S+', expand=False).astype(np.int16)



dataset.loc[(21 <= dataset.release_year) & (dataset.release_year <= 99), 'release_year'] += 1900

dataset.loc[dataset.release_year < 21, 'release_year'] += 2000



dataset['release_date'] = pd.to_datetime(dataset.release_day.astype(str) + '-' + 

                                         dataset.release_month.astype(str) + '-' + 

                                         dataset.release_year.astype(str))



dataset['release_weekday'] = dataset.release_date.dt.weekday + 1

dataset['release_quarter'] = dataset.release_date.dt.quarter

ol_count = dataset['original_language'].value_counts()

for lang, count in ol_count.loc[ol_count > 80].iteritems():

    feature = 'ol_' + lang

    dataset[feature] = 0

    dataset.loc[dataset.original_language == lang, feature] = 1

threshold = 80

for feature in ['genres', 'production_companies', 'production_countries', 'spoken_languages']:

    dataset.loc[dataset[feature].isnull(), feature] = '{}'

    dataset[feature] = dataset[feature].apply(lambda x: sorted([d['name'] for d in eval(x)]))

    dataset['num_of_' + feature] = dataset[feature].apply(lambda x: len(x))

    dataset[feature] = dataset[feature].apply(lambda x: ','.join(map(str, x)))

    

    tmp = dataset[feature].str.get_dummies(sep=',')

    tmp = tmp.loc[:, tmp.sum() > threshold]

    dataset = pd.concat([dataset, tmp], axis=1)

    

dataset['has_budget'] = 1

dataset.loc[dataset.budget == 0, 'has_budget'] = 0

dataset['has_collection'] = 1

dataset.loc[dataset.belongs_to_collection.isnull(), 'has_collection'] = 0

dataset['has_homepage'] = 1

dataset.loc[dataset.homepage.isnull(), 'has_homepage'] = 0

dataset['has_tagline'] = 1

dataset.loc[dataset.tagline.isnull(), 'has_tagline'] = 0



for feature in ['Keywords', 'cast', 'crew']:

    dataset.loc[dataset[feature].isnull(), feature] = '{}'

    dataset['num_of_' + feature] = dataset[feature].apply(lambda x: len([d['name'] for d in eval(x)]))

    

scaler = MinMaxScaler()

numeric_features = ['runtime', 'budget', 'popularity', 'release_year', 

                    'release_month', 'release_day', 'release_quarter', 

                    'num_of_Keywords', 'num_of_cast', 'num_of_crew']

for feature in numeric_features:

    if feature == 'budget':

        dataset.loc[dataset[feature] == 0, feature] = np.nanmedian(dataset[feature].loc[dataset[feature] != 0])

        dataset[feature] = np.log2(dataset[feature] + 1)

    dataset.loc[dataset[feature].isnull(), feature] = np.nanmedian(dataset[feature])

    dataset[feature] = scaler.fit_transform(dataset[feature].values.reshape(-1, 1))

    

dataset = dataset.drop(['id', 'belongs_to_collection', 'genres', 'homepage', 'imdb_id', 

                        'original_language', 'original_title', 'overview', 'poster_path', 

                        'production_companies', 'production_countries', 'release_date', 

                        'spoken_languages', 'status', 'tagline', 'title', 'Keywords', 'cast', 'crew'], axis=1)
dataset1.loc[dataset1['id'] == 16, 'revenue'] = 192864

dataset1.loc[dataset1['id'] == 313, 'revenue'] = 12000000       

dataset1.loc[dataset1['id'] == 451, 'revenue'] = 12000000

dataset1.loc[dataset1['id'] == 1865, 'revenue'] = 25000000 

dataset1.loc[dataset1['id'] == 2491, 'revenue'] = 6800000

dataset1.loc[dataset1['id'] == 90, 'budget'] = 30000000                

dataset1.loc[dataset1['id'] == 118, 'budget'] = 60000000  

dataset1.loc[dataset1['id'] == 149, 'budget'] = 18000000  

dataset1.loc[dataset1['id'] == 464, 'budget'] = 20000000       

dataset1.loc[dataset1['id'] == 470, 'budget'] = 13000000      

dataset1.loc[dataset1['id'] == 513, 'budget'] = 930000          

dataset1.loc[dataset1['id'] == 797, 'budget'] = 8000000       

dataset1.loc[dataset1['id'] == 819, 'budget'] = 90000000       

dataset1.loc[dataset1['id'] == 850, 'budget'] = 90000000  

dataset1.loc[dataset1['id'] == 1112, 'budget'] = 7500000  

dataset1.loc[dataset1['id'] == 1131, 'budget'] = 4300000      

dataset1.loc[dataset1['id'] == 1359, 'budget'] = 10000000      

dataset1.loc[dataset1['id'] == 1542, 'budget'] = 1500000          

dataset1.loc[dataset1['id'] == 1542, 'budget'] = 15800000      

dataset1.loc[dataset1['id'] == 1571, 'budget'] = 4000000        

dataset1.loc[dataset1['id'] == 1714, 'budget'] = 46000000       

dataset1.loc[dataset1['id'] == 1721, 'budget'] = 17500000            

dataset1.loc[dataset1['id'] == 2268, 'budget'] = 17500000      

dataset1.loc[dataset1['id'] == 2602, 'budget'] = 31000000

dataset1.loc[dataset1['id'] == 2612, 'budget'] = 15000000

dataset1.loc[dataset1['id'] == 2696, 'budget'] = 10000000

dataset1.loc[dataset1['id'] == 2801, 'budget'] = 10000000

dataset1.loc[dataset1['id'] == 3889, 'budget'] = 15000000       

dataset1.loc[dataset1['id'] == 6733, 'budget'] = 5000000     

dataset1.loc[dataset1['id'] == 3197, 'budget'] = 8000000     

dataset1.loc[dataset1['id'] == 6683, 'budget'] = 50000000     

dataset1.loc[dataset1['id'] == 5704, 'budget'] = 4300000     

dataset1.loc[dataset1['id'] == 6109, 'budget'] = 281756      

dataset1.loc[dataset1['id'] == 7242, 'budget'] = 10000000     

dataset1.loc[dataset1['id'] == 7021, 'budget'] = 17540562

dataset1.loc[dataset1['id'] == 5591, 'budget'] = 4000000      

dataset1.loc[dataset1['id'] == 4282, 'budget'] = 20000000

dataset1.loc[dataset1['id'] == 391, 'runtime'] = 86 

dataset1.loc[dataset1['id'] == 592, 'runtime'] = 90 

dataset1.loc[dataset1['id'] == 925, 'runtime'] = 95 

dataset1.loc[dataset1['id'] == 978, 'runtime'] = 93 

dataset1.loc[dataset1['id'] == 1256, 'runtime'] = 92 

dataset1.loc[dataset1['id'] == 1542, 'runtime'] = 93

dataset1.loc[dataset1['id'] == 1875, 'runtime'] = 86 

dataset1.loc[dataset1['id'] == 2151, 'runtime'] = 108

dataset1.loc[dataset1['id'] == 2499, 'runtime'] = 108 

dataset1.loc[dataset1['id'] == 2646, 'runtime'] = 98

dataset1.loc[dataset1['id'] == 2786, 'runtime'] = 111

dataset1.loc[dataset1['id'] == 2866, 'runtime'] = 96

dataset1.loc[dataset1['id'] == 4074, 'runtime'] = 103 

dataset1.loc[dataset1['id'] == 4222, 'runtime'] = 93

dataset1.loc[dataset1['id'] == 4431, 'runtime'] = 100 

dataset1.loc[dataset1['id'] == 5520, 'runtime'] = 86 

dataset1.loc[dataset1['id'] == 5845, 'runtime'] = 83 

dataset1.loc[dataset1['id'] == 5849, 'runtime'] = 140

dataset1.loc[dataset1['id'] == 6210, 'runtime'] = 104

dataset1.loc[dataset1['id'] == 6804, 'runtime'] = 145 

dataset1.loc[dataset1['id'] == 7321, 'runtime'] = 87

dataset1.dropna()

dataset1.loc[dataset1.release_date.isnull(), 'release_date'] = '05/01/2000'



dataset1['release_year'] = dataset1.release_date.str.extract('\S+/\S+/(\S+)', expand=False).astype(np.int16)

dataset1['release_month'] = dataset1.release_date.str.extract('(\S+)/\S+/\S+', expand=False).astype(np.int16)

dataset1['release_day'] = dataset1.release_date.str.extract('\S+/(\S+)/\S+', expand=False).astype(np.int16)



dataset1.loc[(21 <= dataset1.release_year) & (dataset1.release_year <= 99), 'release_year'] += 1900

dataset1.loc[dataset1.release_year < 21, 'release_year'] += 2000



dataset1['release_date'] = pd.to_datetime(dataset1.release_day.astype(str) + '-' + 

                                         dataset1.release_month.astype(str) + '-' + 

                                         dataset1.release_year.astype(str))



dataset1['release_weekday'] = dataset1.release_date.dt.weekday + 1

dataset1['release_quarter'] = dataset1.release_date.dt.quarter

ol_count = dataset1['original_language'].value_counts()

for lang, count in ol_count.loc[ol_count > 80].iteritems():

    feature = 'ol_' + lang

    dataset1[feature] = 0

    dataset1.loc[dataset1.original_language == lang, feature] = 1

threshold = 80

for feature in ['genres', 'production_companies', 'production_countries', 'spoken_languages']:

    dataset1.loc[dataset1[feature].isnull(), feature] = '{}'

    dataset1[feature] = dataset1[feature].apply(lambda x: sorted([d['name'] for d in eval(x)]))

    dataset1['num_of_' + feature] = dataset1[feature].apply(lambda x: len(x))

    dataset1[feature] = dataset1[feature].apply(lambda x: ','.join(map(str, x)))

    

    tmp = dataset1[feature].str.get_dummies(sep=',')

    tmp = tmp.loc[:, tmp.sum() > threshold]

    dataset1 = pd.concat([dataset1, tmp], axis=1)

    

dataset1['has_budget'] = 1

dataset1.loc[dataset1.budget == 0, 'has_budget'] = 0

dataset1['has_collection'] = 1

dataset1.loc[dataset1.belongs_to_collection.isnull(), 'has_collection'] = 0

dataset1['has_homepage'] = 1

dataset1.loc[dataset1.homepage.isnull(), 'has_homepage'] = 0

dataset1['has_tagline'] = 1

dataset1.loc[dataset1.tagline.isnull(), 'has_tagline'] = 0



for feature in ['Keywords', 'cast', 'crew']:

    dataset1.loc[dataset1[feature].isnull(), feature] = '{}'

    dataset1['num_of_' + feature] = dataset1[feature].apply(lambda x: len([d['name'] for d in eval(x)]))

    

scaler = MinMaxScaler()

numeric_features = ['runtime', 'budget', 'popularity', 'release_year', 

                    'release_month', 'release_day', 'release_quarter', 

                    'num_of_Keywords', 'num_of_cast', 'num_of_crew']

for feature in numeric_features:

    if feature == 'budget':

        dataset1.loc[dataset1[feature] == 0, feature] = np.nanmedian(dataset1[feature].loc[dataset1[feature] != 0])

        dataset1[feature] = np.log2(dataset1[feature] + 1)

    dataset1.loc[dataset1[feature].isnull(), feature] = np.nanmedian(dataset1[feature])

    dataset1[feature] = scaler.fit_transform(dataset1[feature].values.reshape(-1, 1))

    

dataset1 = dataset1.drop(['id', 'belongs_to_collection', 'genres', 'homepage', 'imdb_id', 

                        'original_language', 'original_title', 'overview', 'poster_path', 

                        'production_companies', 'production_countries', 'release_date', 

                        'spoken_languages', 'status', 'tagline', 'title', 'Keywords', 'cast', 'crew'], axis=1)
# First we separate into input and output components

X_dataset = dataset.drop('revenue', axis=1)

Y_dataset = dataset['revenue']

X = X_dataset.values

y = Y_dataset.values

np.set_printoptions(suppress=True)

pd.DataFrame(X).head()

pd.DataFrame(y).head()
X_test = dataset1[X_dataset.columns]
X_dataset.shape, X_test.shape
# KNN Regression 

from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn import metrics

from sklearn import preprocessing

from sklearn.neighbors import KNeighborsRegressor



kfold=KFold(n_splits=10, random_state=7)



model=KNeighborsRegressor()

scoring = "neg_mean_squared_error"



results=cross_val_score(model, X, y, cv=kfold, scoring=scoring)



print(f'KNN Regression - MSE {results.mean():.3f} std {results.std():.3f}')
model.fit(X,y)

predictions = model.predict(X_test)



output = pd.DataFrame({'id': test.id, 'revenue': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")