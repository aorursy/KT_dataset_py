#importing all the required libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
import os

os.chdir('../input/bengaluru-house-price-data')

os.getcwd()
#loading the dataset

bdf = pd.read_csv('Bengaluru_House_Data.csv')
bdf.head()
#shape of the dataframe

bdf.shape
bdf.groupby('area_type').area_type.count()
bdf.groupby('availability').availability.count()
#renaming 'size' column

bdf.rename(columns = {'size':'house_type'}, inplace = True)
bdf.groupby('house_type').house_type.count()
#dropping 'society' and 'availability' columns

bdf = bdf.drop(['society', 'availability'], axis = 1)
bdf.head()
#checking for missing values

bdf.isnull().sum()
#removing 'house_type' and adding cleaned 'house_type_clean' column

bdf['house_type_clean'] = bdf.house_type.map(lambda x: float(str(x).split(' ')[0]))

bdf = bdf.drop('house_type', axis = 1)
#dropping rows will null values for 'house_type_clean

bdf = bdf.dropna(axis = 0, subset = ['house_type_clean'])

bdf.house_type_clean.value_counts()
bdf.head()
bdf.house_type_clean.isnull().sum()
bdf.house_type_clean.unique()
bdf.house_type_clean = bdf.house_type_clean.astype('int64')

bdf.bath = bdf.bath.dropna().astype('int64')
#getting median values of 'bath' according to the corresponding value of 'house_type_clean' and storing it in a dictionary

bath_fill_by_type = {}

for i in list(bdf['house_type_clean'].unique()):

  a=bdf.loc[bdf['house_type_clean'] == i, ['bath']].dropna().median()

  bath_fill_by_type[i] = int(a.values)
bdf['bath_clean'] = bdf['bath'].fillna(bdf['house_type_clean'].apply(lambda x: bath_fill_by_type.get(x)))

bdf['bath_clean'] = bdf['bath_clean'].astype('int64')
bdf = bdf.drop('bath', axis = 1)

bdf.head()
bdf.balcony.isnull().value_counts()
#getting median values of 'bath' according to the corresponding value of 'house_type_clean' and storing it in a dictionary

balcony_fill_by_type = {}

for i in list(bdf['house_type_clean'].unique()):

  a=bdf.loc[bdf['house_type_clean'] == i, ['balcony']].dropna().median()

  balcony_fill_by_type[i] = a.values[0]
for k in [16, 18, 19]:

    del balcony_fill_by_type[k]
bdf.balcony.isnull().value_counts()
bdf['balcony_clean'] = bdf['balcony'].fillna(bdf['house_type_clean'].apply(lambda x: bath_fill_by_type.get(x)))

bdf['balcony_clean'] = bdf['balcony_clean'].astype('int64')
bdf.balcony_clean.isnull().value_counts()
bdf = bdf.drop(['balcony'], axis = 'columns')

bdf.head()
bdf.isnull().sum()
bdf = bdf[bdf['location'].notna()]
#after removing and imputing missing the null entries

bdf.isnull().sum()
#now exploring the 'total_sqft' column

bdf.total_sqft.unique()
#defining a function to return all the non unifrom values

def unif(bdf):

  try:

     float(bdf)

  except:

    return False

  return True
#return a dataframe(use '~' for negation)

bdf[~bdf['total_sqft'].apply(unif)]
#writing a function to return average of ranges

def conv_to(bdf):

  a = bdf.split('-')

  if len(a) == 2:

    return (float(a[0]) + float(a[1]))/2

  try:

    return float(bdf)

  except:

    return None
#transforming all the non uniform values

bdf['total_sqft'] = bdf['total_sqft'].apply(conv_to)
bdf[~bdf['total_sqft'].apply(unif)]
#now lets create a new feature for price per square feet

bdf['price_per_sqft'] = (bdf['price'] * 100000)/bdf['total_sqft']

bdf.head()
#exploring location column

bdf['location'] = bdf['location'].apply(lambda x: x.strip())

bdf.location.nunique()
loc_stats = bdf.groupby('location')['location'].count().sort_values(ascending = False)

loc_stats
loc_stats_below_ten = loc_stats[loc_stats <= 10]

loc_stats_below_ten
#naming all the less frequent locations in the dataset as other

bdf['location'] = bdf['location'].apply(lambda x: 'other' if x in loc_stats_below_ten else x)
bdf.location.value_counts()
#now checking the size of the house per bedroom and detecting outliers and removing them

bdf[bdf['total_sqft']/bdf['house_type_clean'] < 300].head()
bdf = bdf[~(bdf['total_sqft']/bdf['house_type_clean'] < 300)]
bdf.shape
#now similarly try to detect outliers based on price/sqft

bdf.price_per_sqft.describe()
#this function removes the outliers based on price_per_sqft value grouped by location

def remove_ppsft_outliers(bdf):

  df_out = pd.DataFrame()

  for k, v in bdf.groupby('location'):

    m = np.mean(v.price_per_sqft)

    st = np.std(v.price_per_sqft)

    sub_df = v[(v.price_per_sqft < (m+st)) & (v.price_per_sqft > (m-st))]

    df_out = pd.concat([df_out, sub_df], ignore_index = True)

  return df_out
bdf = remove_ppsft_outliers(bdf)
bdf.shape
#the following plot shows more anomolies and outliers in the dataset; same sqft area shows 3bk to be cheaper than 2bhk

def plot_scatter_plot(bdf, location):

  bhk2 = bdf[(bdf['location'] == location) & (bdf['house_type_clean'] == 2)]

  bhk3 = bdf[(bdf['location'] == location) & (bdf['house_type_clean'] == 3)]

  plt.figure(figsize=(15,10))

  plt.scatter(bhk2.total_sqft, bhk2.price_per_sqft, color = 'blue', label = '2BHK')

  plt.scatter(bhk3.total_sqft, bhk3.price_per_sqft, color = 'green', label = '3BHK', marker = '+')

  plt.xlabel('Total Sqft')

  plt.ylabel('Price per Sqft')

  plt.title(location)



plot_scatter_plot(bdf, "Hebbal")
 #Here we observe that 3 BHK cost that same as 2 BHK in 'Hebbal' location hence removing such outliers is necessary

def remove_bhk_outliers(df):

    exclude_indices = np.array([])

    

    for location, location_df in df.groupby('location'):

        bhk_stats = {}

        

        for bhk, bhk_df in location_df.groupby('house_type_clean'):

            bhk_stats[bhk] = {

                'mean': np.mean(bhk_df.price_per_sqft),

                'std': np.std(bhk_df.price_per_sqft),

                'count': bhk_df.shape[0]

            }

        

        for bhk, bhk_df in location_df.groupby('house_type_clean'):

            stats = bhk_stats.get(bhk-1)

            if stats and stats['count']>5:

                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)

    

    return df.drop(exclude_indices, axis='index')



bdf = remove_bhk_outliers(bdf)

bdf.shape
#a major chunk of the green data points overlapping with the blue data points are now gone

plot_scatter_plot(bdf, "Hebbal")
#now lets examine bath_clean column closely

bdf[bdf.bath_clean > 10]
#lets look at cases when the number of bathrooms exceed the number of room by more than 2(anomoly), so lets remove these

bdf[bdf.bath_clean > bdf.house_type_clean + 2]
bdf = bdf[bdf.bath_clean < bdf.house_type_clean + 2]
#lets also remove the column 'price_per_sqft'

bdf = bdf.drop('price_per_sqft', axis = 'columns') 
bdf.head()
bdf.area_type.isnull().sum()
pd.get_dummies(bdf.area_type)
#lets convert areat_type column to category and label encode it

bdf['area_type'] = bdf['area_type'].astype('category')

bdf['area_type_clean'] = bdf['area_type'].cat.codes
bdf.area_type
bdf.area_type_clean
bdf = bdf.drop('area_type', axis = 'columns')

bdf.head()
#now lets use one hot encoding by getting dummies for the location column

loc_dum = pd.get_dummies(bdf.location)
loc_dum
#now lets concatenate the original df columns with the dummies

bdf_new = pd.concat([bdf.drop('location', axis = 'columns'), loc_dum.drop('other', axis = 'columns')], axis = 'columns')

bdf_new.head()
bdf_new.shape
X = bdf_new.drop('price', axis = 'columns')

X.head()
y = bdf_new.price

y.head()
#using 20% of data for test, 80% for training

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state =20)
#using linear regression to model the prediction

from sklearn.linear_model import LinearRegression



lr_bdf = LinearRegression()

lr_bdf.fit(X_train, y_train)
lr_bdf.score(X_test, y_test)
from sklearn.model_selection import ShuffleSplit

from sklearn.model_selection import cross_val_score



cv = ShuffleSplit(n_splits = 5, test_size = 0.2, random_state = 0)

cross_val_score(LinearRegression(), X, y, cv = cv)
from sklearn.linear_model import Lasso

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import GridSearchCV



def find_best_model(X,y):

    models = {

        'linear_regression': {

            'model': LinearRegression(),

            'parameters': {

                'normalize': [True,False]

            }

        },

        

        'lasso': {

            'model': Lasso(),

            'parameters': {

                'alpha': [1,2],

                'selection': ['random', 'cyclic']

            }

        },

        

        'decision_tree': {

            'model': DecisionTreeRegressor(),

            'parameters': {

                'criterion': ['mse', 'friedman_mse'],

                'splitter': ['best', 'random']

            }

        }

    }

    

    scores = []

    cv_X_y = ShuffleSplit(n_splits=5, test_size=0.20, random_state=0)

    

    for model_name, model_params in models.items():

        gs = GridSearchCV(model_params['model'], model_params['parameters'], cv=cv_X_y, return_train_score=False)

        gs.fit(X,y)

        scores.append({

            'model': model_name,

            'best_parameters': gs.best_params_,

            'accuracy': gs.best_score_

        })

        

    return pd.DataFrame(scores, columns=['model', 'best_parameters', 'accuracy'])



find_best_model(X, y)
X.columns
# For finding the appropriate location

np.where(X.columns=='2nd Phase Judicial Layout')[0][0]
# Creating a fuction to predict values

def prediction(location, house_type_clean, bath_clean, balcony_clean, total_sqft, area_type):

    

    dict_area_type = {'Built-up Area': 0, 'Carpet Area':1 , 'Plot Area': 2, 'Super Built-up Area': 3}

    area_type_clean = dict_area_type.get(area_type)

    loc_index = int(np.where(X.columns==location)[0][0])

                   

    x = np.zeros(len(X.columns))

    x[0] = total_sqft

    x[1] = house_type_clean

    x[2] = bath_clean

    x[3] = balcony_clean

    x[4] = area_type_clean

    

    if loc_index >= 0:

        x[loc_index] = 1

        

    return lr_bdf.predict([x])[0]
#Prediction 1: Input in the form : Location, BHK, Bath, Balcony, Sqft, area_type, availability.

prediction(location = '1st Block Jayanagar', house_type_clean = 2, bath_clean = 2, balcony_clean = 2, total_sqft = 1000, area_type = 'Built-up Area')
# Prediction 3: Input in the form : Location, BHK, Bath, Balcony, Sqft, area_type, availability.

prediction('1st Phase JP Nagar', 2, 3, 2, 2000, 'Plot Area')