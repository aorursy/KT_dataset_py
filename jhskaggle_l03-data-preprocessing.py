import pandas as pd
df = pd.read_csv("../input/lecture02/data-cleansing-example.csv" )
df
df.iloc[:,1:2]
df.iloc[:,2:3]
df.iloc[:,3:4]
df.iloc[:,4:5]
df.iloc[:,5:6]
url = "https://raw.githubusercontent.com/dataoptimal/posts/master/data%20cleaning%20with%20python%20and%20pandas/property%20data.csv"
df2 = pd.read_csv(url)
df2
# Looking at the ST_NUM column
print (df2['ST_NUM'])
print (df2['ST_NUM'].isna()) 
# Looking at the NUM_BEDROOMS column
print (df2['NUM_BEDROOMS'])
print (df2['NUM_BEDROOMS'].isna())
# Making a list of missing value types
missing_values = ["n.a.","?","NA","n/a", "na", "--"]
url = "https://raw.githubusercontent.com/dataoptimal/posts/master/data%20cleaning%20with%20python%20and%20pandas/property%20data.csv"
df2 = pd.read_csv(url, na_values = missing_values) 
# Looking at the NUM_BEDROOMS column
print (df2['NUM_BEDROOMS'])
print (df2['NUM_BEDROOMS'].isna())
# Looking at the OWN_OCCUPIED column
print (df2['OWN_OCCUPIED'])
print (df2['OWN_OCCUPIED'].isna())
print (df2.isna().sum())
# Making a list of missing value types
missing_values = ["n.a.","?","NA","n/a", "na", "--"]
url = "https://raw.githubusercontent.com/dataoptimal/posts/master/data%20cleaning%20with%20python%20and%20pandas/property%20data.csv"
df2 = pd.read_csv(url, na_values = missing_values) 
df2
df2.dropna(inplace=True) #inplace is a Bool variable, default False. If True, fill in-place
#df2.dropna(axis=1,inplace=True)
df2
# Making a list of missing value types
missing_values = ["n.a.","?","NA","n/a", "na", "--"]
url = "https://raw.githubusercontent.com/dataoptimal/posts/master/data%20cleaning%20with%20python%20and%20pandas/property%20data.csv"
df2 = pd.read_csv(url, na_values = missing_values) 
# Replace missing values with a number
print(df2['ST_NUM'])
df2['ST_NUM'].fillna(125, inplace=True) 
print(df2['ST_NUM'])
# Making a list of missing value types
missing_values = ["n.a.","?","NA","n/a", "na", "--"]
url = "https://raw.githubusercontent.com/dataoptimal/posts/master/data%20cleaning%20with%20python%20and%20pandas/property%20data.csv"
df2 = pd.read_csv(url, na_values = missing_values) 
df2['NUM_BEDROOMS']
# Replace using median 
df2['NUM_BEDROOMS'].fillna(df2['NUM_BEDROOMS'].median(), inplace=True)
print(df2['NUM_BEDROOMS'])

# Making a list of missing value types
missing_values = ["n.a.","?","NA","n/a", "na", "--"]
url = "https://raw.githubusercontent.com/dataoptimal/posts/master/data%20cleaning%20with%20python%20and%20pandas/property%20data.csv"
df2 = pd.read_csv(url, na_values = missing_values) 
df2['OWN_OCCUPIED']
# Replace using mode  
df2['OWN_OCCUPIED'].fillna(df2['OWN_OCCUPIED'].mode()[0], inplace=True)
df2['OWN_OCCUPIED']
# Making a list of missing value types
missing_values = ["n.a.","?","NA","n/a", "na", "--"]
url = "https://raw.githubusercontent.com/dataoptimal/posts/master/data%20cleaning%20with%20python%20and%20pandas/property%20data.csv"
df2 = pd.read_csv(url, na_values = missing_values) 
df2 
df2.loc[4,'PID'] = 100005000 
df2
from sklearn import preprocessing
import numpy as np
# A feature is a column. (1,2,3,4,5) must be represented into a column, not a row!
X = np.array([[ 1.],
                    [ 2.],
                    [ 3.],
                    [ 4.],
                    [ 5.]])
scaler = preprocessing.MinMaxScaler() # create a scaler
X_new = scaler.fit_transform(X) # fit and transform data
X_new
# A feature is a column. (1,2,3,4,5) must be represented into a column, not a row!
X = np.array([[ 1.],
                    [ 2.],
                    [ 3.],
                    [ 4.],
                    [ 100.]])
scaler = preprocessing.MinMaxScaler() # create a scaler
X_new =  scaler.fit_transform(X) # fit and transform data
X_new
from sklearn import preprocessing
import numpy as np
X = np.array([[ 1.],
                    [ 2.],
                    [ 3.],
                    [ 4.],
                    [ 5.]])
print("mean",np.mean(X,axis=0))                                 
print("std", np.std(X,axis=0))   
scaler = preprocessing.StandardScaler() # create a scaler
X_new =  scaler.fit_transform(X) # fit and transform data
print(X_new)
print("mean",np.mean(X_new,axis=0))                                 
print("std", np.std(X_new,axis=0)) 
import pandas as pd
#import numpy as np
df = pd.DataFrame([['M', 'O-','medium'],
                   ['M', 'O-', 'high'],
                   ['F', 'O+', 'high'],
                   ['F','AB','low'],
                  ['F', 'B+','NA']]) #create a data frame 
df.columns = ['gender', 'blood_type', 'edu_level'] # add columns name to data frame  
df
from sklearn.preprocessing import OrdinalEncoder
encoder =  OrdinalEncoder() # create an encoder
X = df.values # X are feature values
X_new = encoder.fit_transform(X) # fit encoder with data and transfer data 
X_new
import pandas as pd
import numpy as np
df = pd.DataFrame([['M', 'O-','medium'],
                   ['M', 'O-', 'high'],
                   ['F', 'O+', 'high'],
                   ['F','AB','low'],
                  ['F', 'B+','NA']]) #create a data frame 
df.columns = ['gender', 'blood_type', 'edu_level'] # add columns name to data frame  
df
from sklearn.preprocessing import OrdinalEncoder
encoder = OrdinalEncoder(categories = [['NA', 'low', 'medium', 'high']]) # create an encoder with order
X_edu = encoder.fit_transform(df['edu_level'].values.reshape(-1, 1)) # fit encoder with data and transfer data 
print("edu_level", X_edu) 
 
import pandas as pd
import numpy as np
df = pd.DataFrame([['M', 'O-','medium'],
                   ['M', 'O-', 'high'],
                   ['F', 'O+', 'high'],
                   ['F','AB','low'],
                  ['F', 'B+','NA']]) #create a data frame 
df.columns = ['gender', 'blood_type', 'edu_level'] # add columns name to data frame  
df
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder() # create an encoder
X_gender = encoder.fit_transform(df['gender'].values.reshape(-1, 1)).toarray() # fit encoder with data and transfer data 
print("gender",X_gender)
X_blood = encoder.fit_transform(df['blood_type'].values.reshape(-1, 1)).toarray() # fit encoder with data and transfer data 
print("blood_type",X_blood)
X_Encode= np.concatenate((X_gender,X_blood, X_edu),axis=1)
df_Encode = pd.DataFrame(X_Encode)
df_Encode.columns = ['gender_F', 'gender_M',  'blood_B+', 'blood_AB',   'blood_O+', 'blood_O-',   'edu_level'] # add columns name to data frame  
df_Encode