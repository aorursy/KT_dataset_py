# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import pickle

import numpy as np

import datetime as dt

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/citizensdata/Citizens"))



path = "../input/citizensdata/Citizens"

# Any results you write to the current directory are saved as output.
#import datetime

#Read in the data

test = open(path+'/datathon_propattributes.obj', 'rb')

test = pickle.load(test)



 



pd.set_option('display.max_rows', 200)

#test.dtypes
test.info()
test.head()
pd.set_option("display.colheader_justify", 'left')

test2=pd.read_excel(path+'/FileLayout.xlsx')

pd.set_option('display.max_rows', 72)

pd.set_option('max_colwidth', 500)

test2 
for col in test.columns:

    print(test[col].value_counts(dropna=False).head(3))

    print()

#     break
checkdate = '2018-10-01'

checkdate = pd.to_datetime(checkdate)

testset = test[test['transaction_dt'] >= checkdate]

trainset = test[test['transaction_dt'] < checkdate]
trainset.info()
test['sale_amt'].describe()
testset['sale_amt'].describe()
print("Total Skewness: %f" % test['sale_amt'].skew())

print("Total Kurtosis: %f" % test['sale_amt'].kurt())
cols = [ 'year_built',

       'effective_year_built', 'bedrooms', 'total_rooms',

       'total_baths_calculated', 'air_conditioning', 'basement_cd',

       'condition', 'construction_type', 'fireplace_num', 'garage_type',

       'heating_type', 'construction_quality', 'roof_cover', 'roof_type',

       'stories_cd', 'style', 'geocode_latitude', 'geocode_longitude',

       'avm_final_value0', 'avm_std_deviation0', 'avm_final_value1',

       'avm_std_deviation1', 'avm_final_value2', 'avm_std_deviation2',

       'avm_final_value3', 'avm_std_deviation3', 'avm_final_value4',

       'avm_std_deviation4', 'first_mtg_amt', 'distressed_sale_flg']

my_part = test[cols]
my_part.info()
df_train = test

sns.distplot(df_train['sale_amt'].apply(np.log));
sns.distplot(df_train['sale_amt']);
#scatter plot feature/saleprice

# var = ['avm_final_value4','sale_amt']

# data = test[var]

# data.plot.scatter(x='avm_final_value4', y='sale_amt');
# my_part.columns
# avm = test[ test['avm_final_value0']>100]

# avm = avm[ avm['avm_final_value0'].notna()]

# avm = avm[ avm['avm_final_value1']>100]

# avm = avm[ avm['avm_final_value1'].notna()]

# avm = avm[ avm['avm_final_value2']>100]

# avm = avm[ avm['avm_final_value2'].notna()]

# avm = avm[ avm['avm_final_value3']>100]

# avm = avm[ avm['avm_final_value3'].notna()]

# avm = avm[ avm['avm_final_value4']>100]

# avm = avm[ avm['avm_final_value4'].notna()]

# avm = avm[avm['prop_unit_type']=='APT']
# sns.set()

# cols = [ 'avm_final_value0', 'avm_final_value1', 'avm_final_value2', 

#        'avm_final_value3', 'avm_final_value4'

#        ,'sale_amt' ]



# sns.pairplot(avm[cols].apply(np.log))

# plt.show();
#missing data

# total = df_train.isnull().sum().sort_values(ascending=False)

# percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)

# missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

# missing_data.head(20)
# my_part.info()
# room = test[ test['bedrooms']>=1]

# room = room[ room['total_rooms']>=1]

# room = room[ room['total_baths_calculated']>=1]

# room = room[ room['fireplace_num']>=1]

# room = room[ room['sale_amt']<400000]
# sns.set()

# cols = [ 'bedrooms', 'total_rooms', 'total_baths_calculated', 

#        'fireplace_num', 

#        'sale_amt' ]



# sns.pairplot(room[cols])

# plt.show();