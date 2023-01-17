import pandas as pd

import numpy as np



# Define the headers since the data does not have any

headers = ["symboling", "normalized_losses", "make", "fuel_type", "aspiration",

           "num_doors", "body_style", "drive_wheels", "engine_location",

           "wheel_base", "length", "width", "height", "curb_weight",

           "engine_type", "num_cylinders", "engine_size", "fuel_system",

           "bore", "stroke", "compression_ratio", "horsepower", "peak_rpm",

           "city_mpg", "highway_mpg", "price"]



df = pd.read_csv("../input/automobile-dataset/Automobile_data.csv")

 

# Read in the CSV file.

#df = pd.read_csv('./Datasets/auto.data', names=headers)

df.head()
# # Convert "?" to NaN

df = pd.read_csv("../input/automobile-dataset/Automobile_data.csv", na_values="?")

df.head()
# Dataset shape

df.shape
# Check if a complete columns has "null" value.

# All the following columns have "Null" Value

df[df.isnull().all(axis=1)].head()
# Check 'null' values in "rows" --- default is (axis=0)

df.isnull().count()
# Delete any "row" has at least one "null" value.

df = df.dropna(how='any') # default is (axis=0)

print(df.shape)

df.head()
# The final check we want to do is see what data types we have:

df.dtypes
obj_df = df.select_dtypes(include=['object']).copy()

obj_df.head()
print('Data before encoding: ')

print(obj_df["num-of-doors"].value_counts(),'\n')

print(obj_df["num-of-cylinders"].value_counts())
try:

    obj_df = obj_df.replace({'num_doors': {'two':2, 'four':4}})

    obj_df = obj_df.replace({'num_cylinders': {'four':4, 'six':6, 'five':5, 'eight':8, 'two':2, 'three':3, 'twelve':12}})

except:

    pass



print('After Encoding: ')

print(obj_df["num-of-doors"].value_counts(),'\n')

print(obj_df["num-of-cylinders"].value_counts())



obj_df.head()
# Convert features to "categorial" data-type

obj_df['fuel-type'] = obj_df['fuel-type'].astype('category')



# Check feature befor encoding

print(obj_df['fuel-type'].dtypes)

print('Before:\n', obj_df['fuel-type'].value_counts())



# Encode: (alphabetically labeled from 0)

obj_df['fuel-type'] = obj_df['fuel-type'].cat.codes



# After encoding

print('After:\n', obj_df['fuel-type'].value_counts())

obj_df.head()
from sklearn.preprocessing import LabelEncoder



# Before encoding

print('Before:\n', obj_df['aspiration'].value_counts())



# Encode data

l_encoder = LabelEncoder()

obj_df['aspiration'] = l_encoder.fit_transform(obj_df['aspiration'])



# After encoding

print('After:\n', obj_df['aspiration'].value_counts())

obj_df.head()
import category_encoders as ce



# Before encoding

print('Before:\n', obj_df['drive-wheels'].value_counts())



encoder = ce.OrdinalEncoder(cols=['drive-wheels'], return_df=True)

obj_df = encoder.fit_transform(obj_df)



# After encoding

print('\nAfter:\n', obj_df['drive-wheels'].value_counts())

obj_df.head()
# Before encoding

print('Before:\n', obj_df['engine-type'].value_counts())



# Run Encoder

obj_df = pd.get_dummies(obj_df, columns=['engine-type'])



# After encoding

obj_df.head()
import category_encoders as ce



# Before encoding

print('Before:\n', obj_df['body-style'].value_counts())



encoder = ce.OneHotEncoder(cols=['body-style'], use_cat_names=True, return_df=True)

obj_df = encoder.fit_transform(obj_df)



# After encoding

obj_df.head()
df_bin = df.select_dtypes(include=['object']).copy()



import category_encoders as ce



encoder = ce.BinaryEncoder(cols=['body-style'], return_df=True)

df_bin = encoder.fit_transform(df_bin)



df_bin.head()
df_hash = df.select_dtypes(include=['object']).copy()



import category_encoders as ce



# default "n_components=8"

encoder = ce.HashingEncoder(cols=['body-style'], n_components=4, return_df=True, hash_method='sha256')

df_hash = encoder.fit_transform(df_hash)



df_hash.head()
df_target = df.copy()

df_target.head()
import category_encoders as ce



encoder = ce.TargetEncoder(cols=['make'], return_df=True)



df_target = encoder.fit_transform(df_target, df_target['engine-size'])

# when apply to test dataset

# df_target = encoder.transform(df_test)



df_target.head()