import numpy as np 

import pandas as pd

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from xgboost.sklearn import XGBRegressor

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import train_test_split
#reading test and train dataset files

train = pd.read_csv('../input/blackfriday/train.csv')

test = pd.read_csv('../input/blackfriday/test.csv')
#creating dataframe for the required output

submission = pd.DataFrame()

submission['User_ID'] = test['User_ID']

submission['Product_ID'] = test['Product_ID']
#label encoding User ID

enc = LabelEncoder()

train['User_ID'] = enc.fit_transform(train['User_ID'])

test['User_ID'] = enc.transform(test['User_ID'])
# erasing P00 from product_ID and standardizing the number as it could't be label encoded because test has different products than train

train['Product_ID'] = train['Product_ID'].str.replace('P00', '')

test['Product_ID'] = test['Product_ID'].str.replace('P00', '')

scaler = StandardScaler()

train['Product_ID'] = scaler.fit_transform(train['Product_ID'].to_numpy().reshape(-1, 1))

test['Product_ID'] = scaler.transform(test['Product_ID'].to_numpy().reshape(-1, 1))
#taking the average of age range and treating it a numeric variable

train['Age'] = train['Age'].map({'0-17': 15,

                               '18-25': 21,

                               '26-35': 30,

                               '36-45': 40,

                               '46-50': 48,

                               '51-55': 53,

                               '55+': 60})

test['Age'] = test['Age'].map({'0-17': 15,

                               '18-25': 21,

                               '26-35': 30,

                               '36-45': 40,

                               '46-50': 48,

                               '51-55': 53,

                               '55+': 60})



#4+ to 4 and considering number of years as a numeric variable

train['Stay_In_Current_City_Years'] = train['Stay_In_Current_City_Years'].map({'0': 0,

                                                                               '1': 1,

                                                                                '2': 2,

                                                                                '3': 3,

                                                                                '4+': 4})

test['Stay_In_Current_City_Years'] = test['Stay_In_Current_City_Years'].map({'0': 0,

                                                                               '1': 1,

                                                                                '2': 2,

                                                                                '3': 3,

                                                                                '4+': 4})



#label encoding following column

cat_col = ['Gender', 'City_Category']

encoder = LabelEncoder()

for col in cat_col:

    train[col] = encoder.fit_transform(train[col])

    test[col] = encoder.transform(test[col])

#standardizing following columns    

num_col = ['Age', 'Occupation', 'Stay_In_Current_City_Years', 'Product_Category_1', 

           'Product_Category_2', 'Product_Category_3']

scaler = StandardScaler()

for col in num_col:

    train[col] = scaler.fit_transform(train[col].to_numpy().reshape(-1, 1))

    test[col] = scaler.transform(test[col].to_numpy().reshape(-1, 1))



#seperating the dependant variable  

X = train.drop(['Purchase'], axis=1)

y = train[['Purchase']]

X_test = test



#train validation split in 80:20 ratio

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42, shuffle=True)        
#It will take 1 min to run

xgb_reg = XGBRegressor(learning_rate=1)

xgb_reg.fit(X_train, y_train)

y_pred = xgb_reg.predict(X_val)

rmse = np.sqrt(mean_squared_error(y_pred, y_val))

print(rmse)

r2 = r2_score(y_val, y_pred)

print("R2 Score:", r2)
xgb_reg.fit(X, y)

predict = xgb_reg.predict(X_test)

submission['Purchase'] = predict

submission.to_csv('checkscore.csv',index=False)