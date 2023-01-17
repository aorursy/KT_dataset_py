#let's create a dataframe

import pandas as pd

df = pd.DataFrame ({'country' : ['India','U.S','Australia','India','Australia','India','U.S'],

                    'Age' : [44,34,28,27,30,42,25],

                    'Salary' : [72000,44000,35000,27000,32000,56000,45000],

                    'Purchased' : ['yes','no','yes','yes','no','yes','no']

                    })
#Let's check our dataframe

print(df)
#check the datatypes

df.dtypes
df['country'].unique() #check unique
from sklearn.preprocessing import LabelEncoder #import the LabelEncoder from sklrean library

le= LabelEncoder()    #create the instance of LabelEncoder



df['country_temp'] = le.fit_transform(df['country'])   #apply LabelEncoding of country column
df['country_temp']
#we will use get_dummies to do One Hot encoding

pd.get_dummies(df['country'])
#Dropping the first column

pd.get_dummies(df['country'],drop_first=True)
#create 1 more column occupation here

df['occupation'] = ['Self-employeed','Freelancer','Family-business','Data-scientist','Data -Analyst ','Manager','Daily-wage-worker']

print(df['occupation'])
# we will use BinaryEncoder from category_encoders library to do binary encoding

import category_encoders as ce

encoder = ce.BinaryEncoder(cols = ['occupation'])

df_binary = encoder.fit_transform(df)

print(df_binary)