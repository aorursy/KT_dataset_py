import pandas as pd

url = "https://raw.githubusercontent.com/dataoptimal/posts/master/data%20cleaning%20with%20python%20and%20pandas/property%20data.csv"

df2 = pd.read_csv(url)

df2
# Looking at the OWN_OCCUPIED column

print (df2.isna().sum())
# Making a list of missing value types

missing_values = ["n.a.","?","NA","n/a", "na", "--"]

url = "https://raw.githubusercontent.com/dataoptimal/posts/master/data%20cleaning%20with%20python%20and%20pandas/property%20data.csv"

df2 = pd.read_csv(url, na_values = missing_values)

df2
print (df2.isna().sum())
# Replace using median  

df2['NUM_BEDROOMS'].fillna(df2['NUM_BEDROOMS'].median(), inplace=True)

 

# Replace using mode  

df2['OWN_OCCUPIED'].fillna(df2['OWN_OCCUPIED'].mode()[0], inplace=True)

 

# Replace using mode  

df2['NUM_BATH'].fillna(df2['NUM_BATH'].mode()[0], inplace=True)

 

# Replace using mean

df2['SQ_FT'].fillna(df2['SQ_FT'].mean(), inplace=True)

   

# Find and replace

df2.loc[4,'PID'] = 100005000 

df2
# Find and replace

df2.loc[6,'NUM_BATH'] = df2['NUM_BATH'].mode()[0]

df2
import numpy as np

X  = np.array([[ -1., -10.,  -100.],

                    [ 0.,  0.,  0.],

                    [ 1.,  10., 100.]])

print(X)

print("mean",np.mean(X,axis=0))  #the mean of each column. axis =0 for column mean                                                                 

print("std", np.std(X,axis=0))   #the standard deviation of each column
from sklearn import preprocessing

scaler = preprocessing.StandardScaler()

scaler.fit(X) 

X_new = scaler.transform(X)                           

print(X_new)

print("mean",np.mean(X_new, axis=0))                

print("std", np.std(X_new, axis=0))
from sklearn import preprocessing

scaler = preprocessing.MinMaxScaler()

scaler.fit(X) 

X_new = scaler.transform(X)                           

print(X_new)

print("mean",np.mean(X_new, axis=0))                

print("std", np.std(X_new, axis=0))
import pandas as pd

import numpy as np

df = pd.DataFrame([['M', 'O-','medium','VW','often'],

                   ['M', 'O-', 'high','Ford','often'],

                   ['F', 'O+', 'high','BMW','occasional'],

                   ['F','AB','low','VW','occasional'],

                   ['M','A','low','Ford','never'],

                  ['F', 'B+','NA','Fiat','often']]) #create a data frame 

df.columns = ['gender', 'blood_type', 'edu_level','car','pub_going'] # add columns name to data frame  

df
from sklearn.preprocessing import OrdinalEncoder 

encoder = OrdinalEncoder()

X = df.values

print(X)

X_new = encoder.fit_transform(X)

print(X_new)
from sklearn.preprocessing import OneHotEncoder 

encoder = OneHotEncoder()

X = df.values

print(X)

X_new = encoder.fit_transform(X.reshape(-1, 1)).toarray()

print(X_new)
from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder(categories = [['NA', 'low', 'medium', 'high']]) # create an encoder with order

X_edu = encoder.fit_transform(df['edu_level'].values.reshape(-1, 1)) # fit encoder with data and transfer data 

print("edu_level", X_edu)
from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder(categories = [['never','occasional','often']]) # create an encoder with order

X_pub = encoder.fit_transform(df['pub_going'].values.reshape(-1, 1)) # fit encoder with data and transfer data 

print("pub_going", X_pub)
encoder = OneHotEncoder() # create an encoder

X_gender = encoder.fit_transform(df['gender'].values.reshape(-1, 1)).toarray() # fit encoder with data and transfer data 

print("gender",X_gender)

X_blood = encoder.fit_transform(df['blood_type'].values.reshape(-1, 1)).toarray() # fit encoder with data and transfer data 

print("blood_type",X_blood)

X_car = encoder.fit_transform(df['car'].values.reshape(-1, 1)).toarray() # fit encoder with data and transfer data 

print("car_color",X_car)
X_Encode= np.concatenate((X_gender,X_blood, X_edu, X_car, X_pub),axis=1)

df_Encode = pd.DataFrame(X_Encode)

#df_Encode.columns = ['gender_M', 'gender_F', 'blood_O-',  'blood_O+', 'blood_AB', 'blood_B+', 'edu_level'] # add columns name to data frame  

df_Encode