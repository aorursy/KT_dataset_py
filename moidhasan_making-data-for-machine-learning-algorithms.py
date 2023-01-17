import pandas as pd

import numpy as np

import copy

%matplotlib inline
df_flights = pd.read_csv('../input/flights.csv')



df_flights.head()
#This will select numerical columns and show them

df_flights._get_numeric_data().columns
# This will give the information about pandas dataframe(non-null entries) and schema

print(df_flights.info())
#this is a boxplot which gives you distribution of dep_time column on Origin column classes

df_flights.boxplot('dep_time','origin',rot = 30,figsize=(5,6))
#Copy the dataframe and its metadata into another dataframe

cat_df_flights = df_flights.select_dtypes(include=['object']).copy()
cat_df_flights.head()
#this is used to find t=out the number of rows which has null value in them

print(cat_df_flights.isnull().values.sum())
#To know exactly which columns have nullvalues in them

print(cat_df_flights.isnull().sum())
#This is called missing value imputation

cat_df_flights = cat_df_flights.fillna(cat_df_flights['tailnum'].value_counts().index[0])
#Checking whether null-values are left

print(cat_df_flights.isnull().sum())
#This is frequency count of a given categorical variable

print(cat_df_flights['carrier'].value_counts())
#This will give the count of rows

cat_df_flights.count()
#Number of distinct classes in the courier column

print(cat_df_flights['carrier'].value_counts().count())
#Plotting the carrier column as a Barplot, as it is evident by plot AS has maximum number of counts.

%matplotlib inline

import seaborn as sns

import matplotlib.pyplot as plt

carrier_count = cat_df_flights['carrier'].value_counts()

sns.set(style="darkgrid")

sns.barplot(carrier_count.index, carrier_count.values, alpha=0.9)

plt.title('Frequency Distribution of Carriers')

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Carrier', fontsize=12)

plt.show()
#Pie-chart of the above column, not a good way of seeing the distributions of classes in a column

labels = cat_df_flights['carrier'].astype('category').cat.categories.tolist()

counts = cat_df_flights['carrier'].value_counts()

sizes = [counts[var_cat] for var_cat in labels]

fig1, ax1 = plt.subplots()

ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True) #autopct is show the % on plot

ax1.axis('equal')

plt.show()
#Creating a mapping for carrier columns classes

replace_map = {'carrier': {'AA': 1, 'AS': 2, 'B6': 3, 'DL': 4,

                                  'F9': 5, 'HA': 6, 'OO': 7 , 'UA': 8 , 'US': 9,'VX': 10,'WN': 11}}
labels = cat_df_flights['carrier'].astype('category').cat.categories.tolist()

labels
##Creating a mapping for carrier columns classes by another way

replace_map_comp = {'carrier' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}

print(replace_map_comp)

replace_map_comp.values()
replace_map_comp['carrier'].values()
cat_df_flights_replace = cat_df_flights.copy()
cat_df_flights.head()
#Using the above mapping replacing the values in carrier column

cat_df_flights_replace.replace(replace_map_comp, inplace=True)
print(cat_df_flights_replace.head())
#type-casting the columns

cat_df_flights_lc = cat_df_flights.copy()

cat_df_flights_lc['carrier'] = cat_df_flights_lc['carrier'].astype('category')

cat_df_flights_lc['origin'] = cat_df_flights_lc['origin'].astype('category')                                                              



print(cat_df_flights_lc.dtypes)
#DataFrame with object dtype columns

import time

%timeit cat_df_flights.groupby(['origin','carrier']).count()
##DataFrame with category dtype columns lesser time than object dtype columns

%timeit cat_df_flights_lc.groupby(['origin','carrier']).count()
cat_df_flights_lc['carrier'].head()
#alphabetically labeled from 0 to 10

cat_df_flights_lc['carrier'] = cat_df_flights_lc['carrier'].cat.codes

cat_df_flights_lc.head()
cat_df_flights_specific = cat_df_flights.copy()
#Another way of replacing the String values with Int values

cat_df_flights_specific['US_code'] = np.where(cat_df_flights_specific['carrier'].str.contains('US'), 1, 0)

cat_df_flights_specific.head()

cat_df_flights_sklearn = cat_df_flights.copy()
from sklearn.preprocessing import LabelEncoder
lb_make = LabelEncoder()
#Label encoder is another way of converting the string values of categorical column into INT values, only problem is that it would not make sense for nominal variables

cat_df_flights_sklearn['carrier_code'] = lb_make.fit_transform(cat_df_flights['carrier'])
cat_df_flights_sklearn.head()
#To deal with nominal varaibles one-hot encoding can be used in which a column in conerted into N-columns where N is number of distinct classes in that column.

#This creates problem if the variable/feature has high cardinality(many number of distinct classes)

cat_df_flights_onehot = cat_df_flights.copy()

cat_df_flights_onehot = pd.get_dummies(cat_df_flights_onehot, columns=['carrier'], prefix = ['carrier'])



print(cat_df_flights_onehot.head())
cat_df_flights_onehot_sklearn = cat_df_flights.copy()



from sklearn.preprocessing import LabelBinarizer



lb = LabelBinarizer()

lb_results = lb.fit_transform(cat_df_flights_onehot_sklearn['carrier'])

lb_results_df = pd.DataFrame(lb_results, columns=lb.classes_)



print(lb_results_df.head())
result_df = pd.concat([cat_df_flights_onehot_sklearn, lb_results_df], axis=1)



print(result_df.head())
#One way of Preprocessing variables which has range as values like Age Range or Income-Range

dummy_df_age = pd.DataFrame({'age': ['0-20', '20-40', '40-60','60-80']})

dummy_df_age['start'], dummy_df_age['end'] = zip(*dummy_df_age['age'].map(lambda x: x.split('-')))



dummy_df_age.head()
def split_mean(x):

    split_list = x.split('-')

    mean = (float(split_list[0])+float(split_list[1]))/2

    return mean
dummy_df_age['age_mean'] = dummy_df_age['age'].apply(lambda x: split_mean(x))



dummy_df_age.head()
##one hot encoding and label encoder can also be implemented in Spark, for when if number of dat-points or instances are more than 2 Million.

#import os

#from __future__ import print_function

#import findspark

#findspark.init()

#import pyspark



#sc = pyspark.SparkContext.getOrCreate()

#sqlContext = pyspark.HiveContext(sc)



#from pyspark.sql import functions as F

#sc

#spark_flights = sqlContext.read.format("csv").option('header',True).load('../input/flights.csv',inferSchema=True)

#spark_flights.show(3)
#spark_flights.printSchema()
#carrier_df = spark_flights.select("carrier")

#carrier_df.show(5)
#from pyspark.ml.feature import StringIndexer

#carr_indexer = StringIndexer(inputCol="carrier",outputCol="carrier_index")

#carr_indexed = carr_indexer.fit(carrier_df).transform(carrier_df)



#carr_indexed.show(7)
#carrier_df_onehot = spark_flights.select("carrier")



#from pyspark.ml.feature import OneHotEncoder, StringIndexer



#stringIndexer = StringIndexer(inputCol="carrier", outputCol="carrier_index")

#model = stringIndexer.fit(carrier_df_onehot)

#indexed = model.transform(carrier_df_onehot)

#encoder = OneHotEncoder(dropLast=False, inputCol="carrier_index", outputCol="carrier_vec")

#encoded = encoder.transform(indexed)



#encoded.show(7)