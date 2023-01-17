# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
ls
## Reading the data

training_data = pd.read_csv('/kaggle/input/airbnb-price-prediction/train.csv')

training_data.head(3)
print(training_data.shape)
training_data.dtypes
# unique Id

len(set(training_data.id))
## remove id column

training_data.drop('id', axis=1, inplace=True)
## Missing values

missPer = training_data.apply(lambda x: sum(x.isnull())/training_data.shape[0]*100, axis=0)

missPer
# separating numeric and categorical data

colnames = training_data.columns

numcolnames = training_data._get_numeric_data().columns

cat_data = training_data[list(set(colnames) - set(numcolnames))]

cat_data.columns
## numerical data exploration

numeric_data = training_data[numcolnames]
# Skews

skew = numeric_data.skew()

skew
# varience

var = numeric_data.var()

var
# correlation

cor = numeric_data.corr()

cor
# For vif

numeric_data.drop('cleaning_fee', axis=1, inplace=True)

numeric_data.drop('log_price', axis=1, inplace=True)

numeric_data.dropna(inplace=True)
# For each X, calculate VIF and save in dataframe

from statsmodels.stats.outliers_influence import variance_inflation_factor



vif = pd.DataFrame()

vif["VIF Factor"] = [variance_inflation_factor(numeric_data.values, i) for i in range(numeric_data.shape[1])]

vif["features"] = numeric_data.columns

vif
###### excluding the variables

# From the correlation values, we can drop few variables which has no correlation

training_data.drop(['latitude', 'longitude', 'number_of_reviews',

                    'review_scores_rating'], axis=1, inplace=True)

# we have city we can drop Zipcode

training_data.drop(['zipcode'], axis=1, inplace=True)

# first_review,last_review, host_since are dates and 21% of the data are missing

training_data.drop(['first_review', 'last_review', 'host_since'],

                   axis=1, inplace=True)

# description, name, thumbnail_url are almost unique

training_data.drop(['description', 'name', 'thumbnail_url'],

                   axis=1, inplace=True)

# host details are not giving much relation with the target variable

training_data.drop(['host_has_profile_pic', 'host_response_rate'],

                   axis=1, inplace=True)
## =============================================================================

## Feature engineering

## =============================================================================

## deriving new features with amenities
amenities_list = []

for i in range(0, training_data.shape[0]):

    am = training_data['amenities'][i].split(',')

    for j in am:

        amenities_list.append(j.replace('"', '').replace('}', '').replace('{', ''))

amenities_set = set(amenities_list)

len(amenities_set)
amenities_dict = {}

for am in set(amenities_list):

    #print(str(am) +' : ' +str(amenities_list.count(am)))

    amenities_dict.update({str(am) : amenities_list.count(am)})

sorted(amenities_dict.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
# Keep only top 20

# 'Wireless Internet', 71265),

#  ('Kitchen', 67526),

#  ('Heating', 67073),

#  ('Essentials', 64005),

#  ('Smoke detector', 61727),

#  ('Air conditioning', 55210),

#  ('TV', 52458),

#  ('Shampoo', 49465),

#  ('Hangers', 49173),

#  ('Carbon monoxide detector', 47190),

#  ('Internet', 44648),

#  ('Laptop friendly workspace', 43703),

#  ('Hair dryer', 43330),

#  ('Washer', 43169),

#  ('Dryer', 42711),

#  ('Iron', 41687),

#  ('Family/kid friendly', 37026),

#  ('Fire extinguisher', 30724),

#  ('First aid kit', 27532),

#  ('translation missing: en.hosting_amenity_50', 25291),

#  ('Cable TV', 24253),

#  ('Free parking on premises', 23639),

#  ('translation missing: en.hosting_amenity_49', 20427),

#  ('24-hour check-in', 19015),

#  ('Lock on bedroom door', 17983),

#  ('Buzzer/wireless intercom', 17033),

#  ('Safety card', 11513),
amenities_array = []

for i in range(0, training_data.shape[0]):

    array = np.zeros(shape=(len(amenities_set)))

    am = training_data['amenities'][i].split(',')

    for j in am:

        item = j.replace('"', '').replace('}', '').replace('{', '')

        res = list(amenities_dict.keys()).index(item)

        array[res] = 1

    amenities_array.append(array.tolist())



amenities_df = pd.DataFrame(amenities_array, columns=amenities_dict.keys())
amenities_df.apply(pd.Series.value_counts)
amenities_df = amenities_df[['Wireless Internet','Kitchen','Heating','Essentials','Smoke detector','Air conditioning','TV','Shampoo','Hangers',

'Carbon monoxide detector']]

#                              ,'Internet','Laptop friendly workspace', 'Hair dryer','Washer','Dryer','Iron',

# 'Family/kid friendly', 'Fire extinguisher', 'First aid kit','translation missing: en.hosting_amenity_50']]

#                              ,'Cable TV',

# 'Free parking on premises']]
amenities_df.shape
amenities_df = amenities_df.astype('category')
training_data.drop(['amenities'], axis=1, inplace=True)

training_data = pd.concat([training_data, amenities_df], axis=1)
training_data.head()
from fastai.imports import *

#from fastai.structured import *

from fastai.tabular import * 
colnames = training_data.columns

numcolnames = training_data._get_numeric_data().columns

cat_data = training_data[list(set(colnames) - set(numcolnames))]

cat_data.columns
numcolnames[1:]
dep_var = 'log_price'



cat_names = cat_data.columns.tolist()



cont_names = numcolnames[1:].tolist()



# Transformations

procs = [FillMissing, Categorify, Normalize]
#Start index for creating a validation set from train_data

start_indx = len(training_data) - int(len(training_data) * 0.2)



#End index for creating a validation set from train_data

end_indx = len(training_data)
#TabularList for Validation

val = (TabularList.from_df(training_data.iloc[start_indx:end_indx].copy(), cat_names=cat_names, cont_names=cont_names))
# Train Data Bunch

data = (TabularList.from_df(training_data, path='.', cat_names=cat_names, cont_names=cont_names, procs=procs)

                        .split_by_idx(list(range(start_indx,end_indx)))

                        .label_from_df(cols = dep_var)

                        .databunch())



data.show_batch(rows=10)
# Create deep learning model

learn = tabular_learner(data, layers=[1000, 200,15], metrics= [rmse,r2_score], emb_drop=0.1, callback_fns=ShowGraph)



# select the appropriate learning rate

learn.lr_find(start_lr = 1e-05,end_lr = 1e+05, num_it = 100)



# we typically find the point where the slope is steepest

learn.recorder.plot()



# Fit the model based on selected learning rate

learn.fit_one_cycle(15)



# Analyse our model

learn.model

learn.recorder.plot_losses()
training_data.iloc[0,:].values
learn.predict(training_data.iloc[0,:])[0]