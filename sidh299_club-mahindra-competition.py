# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



import matplotlib.pyplot as plt # This is used for plotting

from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV

from sklearn.metrics import mean_squared_error, make_scorer
# Read the training data set

train = pd.read_csv("../input/train.csv")
train.head()
train.info()
#Size of the data set

train_rows, train_cols = train.shape
# Read the test data set

test = pd.read_csv("../input/test.csv")
#Merge the two data sets

test.info()

target = train.amount_spent_per_room_night_scaled

test_reservation_id = test.reservation_id

df = pd.concat([train, test], sort = False)
df.booking_type_code.value_counts()
#Drop the reservation id

df = df.drop(['reservation_id'], axis=1)
#Convert checking date into other variables

df['booking_date'] = pd.to_datetime(df['booking_date'], format='%d/%m/%y')

df['booking_day'] = df['booking_date'].dt.day

df['booking_month'] = df['booking_date'].dt.month

df['booking_year'] = df['booking_date'].dt.year

df['booking_day_of_week'] = df['booking_date'].dt.day_name()



df['checkin_date'] = pd.to_datetime(df['checkin_date'], format='%d/%m/%y')

df['checkin_day'] = df['checkin_date'].dt.day

df['checkin_month'] = df['checkin_date'].dt.month

df['checkin_year'] = df['checkin_date'].dt.year

df['checkin_day_of_week'] = df['checkin_date'].dt.day_name()



df['checkout_date'] = pd.to_datetime(df['checkout_date'], format='%d/%m/%y')

df['checkout_day'] = df['checkout_date'].dt.day

df['checkout_month'] = df['checkout_date'].dt.month

df['checkout_year'] = df['checkout_date'].dt.year

df['checkout_day_of_week'] = df['checkout_date'].dt.day_name()



df['duration_of_stay'] = (df['checkout_date'] - df['checkin_date']).dt.days



df.drop(['booking_date', 'checkin_date', 'checkout_date'], axis = 1, inplace = True)

df.state_code_residence = df.state_code_residence.fillna(100)

df.season_holidayed_code = df.season_holidayed_code.fillna(5)

#print("NAs for categorical features in train : " + str(df.isnull().values.sum()))
# Convert some numerical columns to object

num_to_object_features = ['channel_code', 'main_product_code', 'resort_region_code', 'resort_type_code', 

                          'room_type_booked_code', 'persontravellingid', 'state_code_residence',

                          'state_code_resort', 'total_pax', 'season_holidayed_code',

                          'booking_type_code']



for feature in num_to_object_features:

    df[feature] = df[feature].astype(object)

    

df = df.replace({"member_age_buckets" : {"A" : 1, "B" : 2, "C" : 3, "D" : 4, "E" : 5, "F" : 6, "G" : 7, 

                                         "H" : 8, "I" : 9, "J" : 10}})
df['guest_nights'] = df['numberofadults'] * df['roomnights']

df['same_state'] = np.where(df['state_code_residence'] == df['state_code_resort'], 1, 0)
# Find most important features relative to target

print("Find most important features relative to target")

corr = df.corr()

corr.sort_values(["amount_spent_per_room_night_scaled"], ascending = False, inplace = True)

print(corr.amount_spent_per_room_night_scaled)
df['numberofadults-s2'] = df['numberofadults'] ** 2

df['numberofadults-s3'] = df['numberofadults'] ** 3

df['numberofadults-sq'] = np.sqrt(df['numberofadults'])



df['duration_of_stay-s2'] = df['duration_of_stay'] ** 2

df['duration_of_stay-s3'] = df['duration_of_stay'] ** 3

df['duration_of_stay-sq'] = np.sqrt(df['duration_of_stay'])



df['guest_nights-s2'] = df['guest_nights'] ** 2

df['guest_nights-s3'] = df['guest_nights'] ** 3

df['guest_nights-sq'] = np.sqrt(df['guest_nights'])



df['booking_year-s2'] = df['booking_year'] ** 2

df['booking_year-s3'] = df['booking_year'] ** 3

df['booking_year-sq'] = np.sqrt(df['booking_year'])



df['roomnights-s2'] = df['roomnights'] ** 2

df['roomnights-s3'] = df['roomnights'] ** 3

df['roomnights-sq'] = np.sqrt(df['roomnights'])
#Lets check the correlation matrix

def plot_corr(df,size=10):

    '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.



    Input:

        df: pandas DataFrame

        size: vertical and horizontal size of the plot'''



    corr = df.corr()

    fig, ax = plt.subplots(figsize=(size, size))

    ax.matshow(corr)

    plt.xticks(range(len(corr.columns)), corr.columns);

    plt.yticks(range(len(corr.columns)), corr.columns);

    plt.xticks(rotation=90)

    

plot_corr(df, 12)
print(f"Sum of null values in each feature:\n{35 * '-'}")

print(f"{df.isnull().sum()}")
df.drop(['amount_spent_per_room_night_scaled'], axis = 1, inplace = True)
# Differentiate numerical features (minus the target) and categorical features

categorical_features = df.select_dtypes(include = ["object"]).columns

numerical_features = df.select_dtypes(exclude = ["object"]).columns

df_num = df[numerical_features]

df_cat = df[categorical_features]
from scipy.stats import skew



skewness = df_num.apply(lambda x: skew(x))

skewness = skewness[abs(skewness) > 0.5]

print(str(skewness.shape[0]) + " skewed numerical features to log transform")

skewed_features = skewness.index

df_num[skewed_features] = np.log1p(df_num[skewed_features])
print("NAs for numerical features in train : " + str(df_num.isnull().values.sum()))
from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()

for feature in categorical_features:

    df_cat[feature] = le.fit_transform(df_cat[feature])





#df_cat = pd.get_dummies(df_cat)
df = pd.concat([df_num, df_cat], axis = 1)

X_train = df.iloc[:train_rows,:]

y_train = target

X_test = df.iloc[train_rows:,:]
# train and valid sets from train

from sklearn.model_selection import train_test_split



X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.3, random_state = 0)

print("X_train : " + str(X_train.shape))

print("X_valid : " + str(X_valid.shape))

print("y_train : " + str(y_train.shape))

print("y_valid : " + str(y_valid.shape))
# Standardize numerical features

stdSc = StandardScaler()

X_train.loc[:, numerical_features] = stdSc.fit_transform(X_train.loc[:, numerical_features])

X_valid.loc[:, numerical_features] = stdSc.fit_transform(X_valid.loc[:, numerical_features])

X_test.loc[:, numerical_features] = stdSc.transform(X_test.loc[:, numerical_features])
# Define error measure for official scoring : RMSE

scorer = make_scorer(mean_squared_error, greater_is_better = False)



def rmse_cv_train(model):

    rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring = scorer, cv = 10))

    return(rmse)



def rmse_cv_valid(model):

    rmse= np.sqrt(-cross_val_score(model, X_valid, y_valid, scoring = scorer, cv = 10))

    return(rmse)
X_valid.duration_of_stay.value_counts()
# Try linear regression 

'''lr = LinearRegression()

lr.fit(X_train, y_train)



# Look at predictions on training and validation set

print("RMSE on Training set :", rmse_cv_train(lr).mean())

#print("RMSE on Validation set :", rmse_cv_valid(lr).mean())

y_test_pred_lr = lr.predict(X_test)

'''
'''from sklearn.linear_model import RidgeCV



ridge = RidgeCV(alphas = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60])

ridge.fit(X_train,y_train)

alpha = ridge.alpha_

print("Best alpha :", alpha)





y_test_pred_ridge = ridge.predict(X_test)

'''
'''from sklearn.linear_model import LassoCV



lasso = LassoCV(alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 

                          0.3, 0.6, 1], 

                max_iter = 5000, cv = 10)

lasso.fit(X_train,y_train)

alpha = lasso.alpha_

print("Best alpha :", alpha)





y_tes_pred_lasso = lasso.predict(X_test)

'''
import xgboost as xgb



model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1) #the params were tuned using xgb.cv

model_xgb.fit(X_train, y_train)

y_test_pred_xgb = model_xgb.predict(X_test)
'''y_test_pred = y_test_pred_ridge

submission = pd.DataFrame({'reservation_id': test_reservation_id,'amount_spent_per_room_night_scaled': y_test_pred})

submission.to_csv('submit.csv', sep=',', index=False)

'''

submission = pd.DataFrame({'reservation_id': test_reservation_id,'amount_spent_per_room_night_scaled': y_test_pred_xgb})

submission.to_csv('submit_lxgb.csv', sep=',', index=False)



print("Its done")