# Code you have previously used to load data

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from learntools.core import *

import scipy.sparse as sp

# path to file you will use for predictions

test_data_path = '../input/test.csv'



# read test data file using pandas

test_data = pd.read_csv(test_data_path)







# Path of the file to read. We changed the directory structure to simplify submitting to a competition

iowa_file_path = '../input/train.csv'



home_data = pd.read_csv(iowa_file_path)



# Create target object and call it y

y = home_data.SalePrice



X=home_data.select_dtypes(exclude=['object'])

Xc=home_data.select_dtypes(exclude=['number'])

X.dropna(1,inplace=True)

Xc.dropna(1,inplace=True)

X.drop(columns=['SalePrice'], inplace=True)



features1 = []

features2 = []

for col in X:

    features1.append(col)

for col in Xc:

    features2.append(col)    

test_X2 = test_data[features2]

#test_X2.fillna('NA', axis=1, inplace=True)

#X2 = sp.hstack(test_X2.apply(lambda col: tf.transform(col)))

df_all_rows = pd.concat([Xc, test_X2])

df_all_rows=df_all_rows.reset_index(drop=True)

X_all= pd.get_dummies(df_all_rows)

train = X_all.iloc[:1460,:]

X2 = X_all.iloc[1460:,:]

X2 = X2.reset_index(drop=True)

#X2.fillna(0 , axis=1, inplace=True)



data = sp.hstack([train,X])

#train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Split into validation and training data





# Specify Model

#iowa_model = RandomForestRegressor(random_state=1)

xgb_model = XGBRegressor(random_state=1, n_estimators =500, learning_rate=0.05)



# Fit Model

#iowa_model.fit(X,y)

xgb_model.fit(data,y)


# create test_X which comes from test_data but includes only the columns you used for prediction.

# The list of columns is stored in a variable called features

test_X1 = test_data[features1]

test_X1.fillna(0, axis=1, inplace=True)



test_X = sp.hstack([X2,test_X1])



# make predictions which we will submit. 

test_preds = xgb_model.predict(test_X) 



# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



output = pd.DataFrame({'Id': test_data.Id,

                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)