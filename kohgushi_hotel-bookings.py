from datetime import datetime as dt



from matplotlib import pyplot as plt

import lightgbm as lgbm

import numpy as np

import pandas as pd

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import accuracy_score

import seaborn as sns



%matplotlib inline

sns.set()
def load_hotel_bookings():

    filename = '/kaggle/input/hotel-booking-demand/hotel_bookings.csv'

    drop_col = ['reservation_status', 'reservation_status_date']

    df = pd.read_csv(filename)

    df.drop(drop_col, axis=1, inplace=True)

    df['arrival_date_month'] = pd.to_datetime(df['arrival_date_month'], format='%B').dt.month

    return df
hotel_bookings = load_hotel_bookings()
hotel_bookings.isnull().sum()
TARGET = 'is_canceled'

X = hotel_bookings.drop(TARGET, axis=1)

y = hotel_bookings[TARGET]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.8, random_state=42, stratify=y)
def add_features(df, room_as_requested=True, total_people=True, stays_in_nights=True, arrival_date_weekday=True):

    category_cols = ["hotel", 'arrival_date_week_number',"meal", "country", "market_segment", "distribution_channel", "is_repeated_guest", "reserved_room_type", "assigned_room_type",  "deposit_type", "agent", "company", "customer_type"]

    date_cols = ["arrival_date_year", "arrival_date_month", "arrival_date_day_of_month"]

    numerical_cols = ["lead_time", "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children", "babies", "previous_cancellations", "previous_bookings_not_canceled", "booking_changes", "days_in_waiting_list", "adr", "required_car_parking_spaces", "total_of_special_requests"]



    if room_as_requested:

        df['room_as_requested'] = df['reserved_room_type'] == df['assigned_room_type']

        category_cols.append('room_as_requested')

    

    if total_people:

        df['total_people'] = df['adults'] + df['children'] + df['babies']

        numerical_cols.append('total_people')

    

    if stays_in_nights:

        df['stays_in_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']

        numerical_cols.append('stays_in_nights')

    

    if arrival_date_weekday:

        date_cols = ["arrival_date_year", "arrival_date_month", "arrival_date_day_of_month"]

        temp_df = df[date_cols].copy()

        temp_df.columns = ['year', 'month', 'day']

        df['arrival_date_weekday'] = pd.to_datetime(temp_df, infer_datetime_format=True).dt.weekday

        category_cols.append('arrival_date_weekday')

        

    return df, category_cols, date_cols, numerical_cols
def one_hot_encoding(df, enc, mode='train'):

    category_cols = ["hotel", 'arrival_date_week_number', 'arrival_date_weekday', "meal", "country",  "market_segment", "distribution_channel", "is_repeated_guest", "reserved_room_type", "assigned_room_type",  'room_as_requested', "deposit_type", "agent", "company", "customer_type"]

    cat_df = df[category_cols].copy()

    if mode == 'train':

        ret_ndarray = enc.fit_transform(cat_df)

    else:

        ret_ndarray = enc.transform(cat_df)

    return ret_ndarray, enc
def impute(df):

    df.fillna({'country':'unknown', 'agent':0, 'company':0, 'children':0}, inplace=True)

    return df
X_train = impute(X_train)

X_train, category_cols, date_cols, numerical_cols = add_features(X_train.copy())

enc = OneHotEncoder(sparse=False, handle_unknown='ignore')

X_train_cat, enc = one_hot_encoding(X_train.copy(), enc, mode='train')

X_train_num = X_train[numerical_cols].values

X_train_prepared = np.concatenate([X_train_num, X_train_cat], axis=1)
lgbm_clf = lgbm.LGBMClassifier()

# lgbm_clf.fit(X_train_prepared, y_train)
param_grid = {"max_depth": [10, 25, 50, 75],

              "learning_rate" : [0.001,0.01,0.05,0.1],

              "num_leaves": [100,300,900,1200],

              "n_estimators": [100,200,500]

             }
grid_result = GridSearchCV(estimator = lgbm_clf,

                           param_grid = param_grid,

                           scoring = 'accuracy',

                           cv = 5,

                           verbose=3,

                           return_train_score = True,

                           n_jobs = -1)



grid_result.fit(X_train_prepared, y_train)
print(grid_result.best_params_)

print(grid_result.best_score_)
best_clf = grid_result.best_estimator_
X_test = impute(X_test)

X_test, category_cols, date_cols, numerical_cols = add_features(X_test.copy())

# enc = OneHotEncoder(sparse=False)

X_test_cat, enc = one_hot_encoding(X_test, enc, mode='test')

X_test_num = X_test[numerical_cols].values

X_test_prepared = np.concatenate([X_test_num, X_test_cat], axis=1)
pred = best_clf.predict(X_test_prepared)
accuracy_score(y_test, pred)