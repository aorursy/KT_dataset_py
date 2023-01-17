%load_ext autoreload

%autoreload 2

%matplotlib inline
import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from math import *

import os

import requests

import re

from sklearn.preprocessing import LabelEncoder





pd.set_option('display.max_rows', 1000)

pd.set_option('display.max_columns', 1000)





DATA_FOLDER = '../input/'





def read_csv(name, index_col='sale_id'):

    df = pd.read_csv(os.path.join(DATA_FOLDER, name), low_memory=False, index_col=index_col)

    return df

!ls ../input/

df_train = read_csv('train.csv')
def look(df):

    display(df.describe())

    display(df.head())

    display(df.dtypes)

    

look(df_train)
!pip install uszipcode

from uszipcode import SearchEngine





def add_zipinfo(df):

    search = SearchEngine(simple_zipcode=True) # set simple_zipcode=False to use rich info database

    zips = df['zip_code'].unique()

    cols = ['major_city', 'lat', 'lng', 'radius_in_miles', 'population',  

           'population_density', 'land_area_in_sqmi', 'water_area_in_sqmi',

            'housing_units', 'occupied_housing_units', 'median_home_value', 'median_household_income',

        'bounds_west', 'bounds_east', 'bounds_north', 'bounds_south']

    rows = []



    for z in zips:

        zipcode = search.by_zipcode(z)

        d = zipcode.to_dict()

        row = [z, ]

        for c in cols:

            val = d[c]

            if val is None:

                val = ''

            row.append(val)

        rows.append(row)



    cols = ['zip_code'] + cols



    zipped = pd.DataFrame(rows, columns=cols)

    return pd.merge(df, zipped, on='zip_code')
def string_to_category(df, columns):

    assert columns is not None

    df[columns] = df[columns].apply(LabelEncoder().fit_transform)

    

def onehot_encoder(df, columns):

    assert columns is not None

    

    res = pd.get_dummies(df[columns], dummy_na=True)

    res = df.drop(columns, axis=1).join(res)

    return res



def apartment_number(s):

    n = ''

    for c in s:

        if c.isdigit():

            n += c

    if not n:

        n = '1'

    return int(n) / 10



def add_datepart(df, fldname, drop=False):

    fld = df[fldname]

    if not np.issubdtype(fld.dtype, np.datetime64):

        df[fldname] = fld = pd.to_datetime(fld, format='%m/%d/%y')

    targ_pre = re.sub('[Dd]ate$', '', fldname)

    for n in ('Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',

            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start'):

        df[targ_pre+n] = getattr(fld.dt,n.lower())

    df[targ_pre+'Elapsed'] = fld.astype(np.int64) // 10**9

    if drop: df.drop(fldname, axis=1, inplace=True)



def add_dateparts(df):

    date_columns = df.select_dtypes(include=[np.datetime64]).columns.tolist()

    for c in date_columns:

        add_datepart(df, c, drop=True)



def preprocess(df, dates, split_dates=False,

               columns_todiscard=None,

               columns_tocategorize=None,

               columns_tooneshot=None,

               columns_tonumeric=None,

               sort_column=None,

               fillna_with_median=False,

               apply_apartment=False, ):

    assert dates is not None

   

    df = df.copy()

        

    if sort_column is not None:

        df.sort_values(sort_column, inplace=True)

    

    if apply_apartment:

         df['apartment_number'] = df['apartment_number'].apply(apartment_number)

    

    for d in dates:

        df[d] = pd.to_datetime(df[d], format='%m/%d/%y')

        

    if columns_todiscard is not None:

        df.drop(columns_todiscard, axis=1, inplace=True, errors='ignore')

    

    if columns_tocategorize is not None:

        string_to_category(df, columns_tocategorize)

        

    if columns_tooneshot is not None:

        df = onehot_encoder(df, columns_tooneshot)

        

    if columns_tonumeric is not None:

        for c in columns_tonumeric:

            df[c] = pd.to_numeric(df[c], errors='coerce')

            

    if split_dates:

        add_dateparts(df)

        

    if fillna_with_median:

        df.fillna(df.median(), inplace=True)



    return df



def do_preprocess(df,

                columns_tocategorize=None,

                columns_tooneshot=None,

                columns_todiscard=None,

                columns_tonumeric=None,

                apply_apartment=False,

                add_zip=False):

    if add_zip:

        df = add_zipinfo(df)

    df = preprocess(df, dates=['sale_date'], split_dates=True, 

                  columns_tocategorize=columns_tocategorize,

                 columns_tooneshot=columns_tooneshot,

                 columns_todiscard=columns_todiscard,

                 columns_tonumeric=columns_tonumeric,

                 sort_column='sale_date',

                 fillna_with_median=True,

                 apply_apartment=apply_apartment)

    return df



df_train = read_csv('train.csv', index_col='sale_id')

columns_tocategorize = ['neighborhood', 'building_class_category', 'major_city', ]

columns_tooneshot = ['tax_class_at_present', 'building_class_at_present', 'building_class_at_time_of_sale', ]

columns_todiscard = ['ease-ment', 'address', 'apartment_number', ]

columns_tonumeric = ['land_square_feet', 'gross_square_feet', 'lat',

 'lng',

 'radius_in_miles',

 'population',

 'population_density',

 'land_area_in_sqmi',

 'water_area_in_sqmi',

 'housing_units',

 'occupied_housing_units',

 'median_home_value',

 'median_household_income',

 'bounds_west',

 'bounds_east',

 'bounds_north',

 'bounds_south']



df_train = do_preprocess(df_train,

                         columns_tocategorize=columns_tocategorize,

                        columns_tooneshot=columns_tooneshot,

                        columns_todiscard=columns_todiscard,

                        columns_tonumeric=columns_tonumeric,

                        apply_apartment=False,

                        add_zip=True)



display(df_train.select_dtypes(exclude=[np.int64, np.uint8, np.float64, np.bool, ]).columns.tolist())

display(f'df_train.shape={df_train.shape}')

display(df_train.head())
df_valid = read_csv('valid.csv')

df_test = read_csv('test.csv')



display(f'df_valid.shape = {df_valid.shape}')

display(f'df_test.shape = {df_test.shape}')
def split_xy(df, y):

    return df.drop(y, axis=1), df[y]



def split_train_valid(df, y_column, validation_size):

    train_size = df.shape[0] - validation_size

    train = df[:train_size]

    valid = df[train_size:]

    return split_xy(train, y_column) + split_xy(valid, y_column)



validation_size = df_valid.shape[0]

display(f'validation_size = {validation_size}')

df_train_x, df_train_y, df_valid_x, df_valid_y = split_train_valid(df_train, 'sale_price', validation_size)



display(f'df_train_x.shape = {df_train_x.shape}')

display(f'df_train_y.shape = {df_train_y.shape}')

display(f'df_valid_x.shape = {df_valid_x.shape}')

display(f'df_valid_y.shape = {df_valid_y.shape}')
from sklearn.ensemble import RandomForestRegressor



def print_score(m, x_train, y_train, x_valid, y_valid):

    res = [m.score(x_train, y_train), m.score(x_valid, y_valid)]

    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)

    print(res)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)

m.fit(df_train_x, df_train_y)

print_score(m, df_train_x, df_train_y, df_valid_x, df_valid_y)
def tunning(columns_tocategorize=None,

            columns_tooneshot=None,

            columns_todiscard=None,

            columns_tonumeric=None, ):

    

    df_train = read_csv('train.csv', index_col='sale_id')

    

    df_train = do_preprocess(df_train,

                             columns_tocategorize=columns_tocategorize,

                            columns_tooneshot=columns_tooneshot,

                            columns_todiscard=columns_todiscard,

                            columns_tonumeric=columns_tonumeric,

                            apply_apartment=True,

                            add_zip=True)

    

    display(f'df_train.shape = {df_train.shape}')

    display(f'df_train.columns = {df_train.columns.tolist()}')

    

    

    df_valid = read_csv('valid.csv')

    validation_size = df_valid.shape[0]

    display(f'validation_size = {validation_size}')



    df_train_x, df_train_y, df_valid_x, df_valid_y = split_train_valid(df_train, 'sale_price', validation_size)

    display(f'df_train_x.shape = {df_train_x.shape}')

    display(f'df_train_y.shape = {df_train_y.shape}')

    display(f'df_valid_x.shape = {df_valid_x.shape}')

    display(f'df_valid_y.shape = {df_valid_y.shape}')



    m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)

    m.fit(df_train_x, df_train_y)

    

    print_score(m, df_train_x, df_train_y, df_valid_x, df_valid_y)

    return

columns_tocategorize = ['neighborhood', 'building_class_category', 'tax_class_at_present', 'major_city',

                        'building_class_at_present', 'building_class_at_time_of_sale',

                        'ease-ment', 'address', ]

columns_tooneshot = None

columns_todiscard = None

columns_tonumeric = ['land_square_feet', 'gross_square_feet', 'lat',

 'lng',

 'radius_in_miles',

 'population',

 'population_density',

 'land_area_in_sqmi',

 'water_area_in_sqmi',

 'housing_units',

 'occupied_housing_units',

 'median_home_value',

 'median_household_income',

 'bounds_west',

 'bounds_east',

 'bounds_north',

 'bounds_south']



tunning(columns_tocategorize=columns_tocategorize,

        columns_tooneshot=columns_tooneshot,

        columns_todiscard=columns_todiscard,

        columns_tonumeric=columns_tonumeric,)
def validate_solution():

    sub = pd.read_csv('submission.csv')

    valid = pd.read_csv('../input/valid.csv')

    test = pd.read_csv('../input/test.csv')

    

    assert(sub.shape == (21589, 2))

    

    expected_saleids = set(valid['sale_id'].values).union(test['sale_id'].values)

    got_saleids = set(sub['sale_id'].values)



    display(len(set(sub.index.values)))

    display(sub.index.nunique())

    assert (expected_saleids == got_saleids), f'Expected sale_id size to be {len(expected_saleids)} but it is {len(got_saleids)}'

    assert(sub['sale_price'].min() > 0.0)

    print('Solution is ready to go!!!')
# from sklearn.preprocessing import Imputer



def get_submission_csv():

    train = read_csv('train.csv', index_col='sale_id')

    valid = read_csv('valid.csv', index_col=None)

    test = read_csv('test.csv', index_col=None)

    

    

    columns_tocategorize = ['neighborhood', 'building_class_category', 'tax_class_at_present', 'major_city',

                        'building_class_at_present', 'building_class_at_time_of_sale',

                        'ease-ment', 'address', ]

    columns_tooneshot = None

    columns_todiscard = None

    columns_tonumeric = ['land_square_feet', 'gross_square_feet', 'lat',

     'lng',

     'radius_in_miles',

     'population',

     'population_density',

     'land_area_in_sqmi',

     'water_area_in_sqmi',

     'housing_units',

     'occupied_housing_units',

     'median_home_value',

     'median_household_income',

     'bounds_west',

     'bounds_east',

     'bounds_north',

     'bounds_south']

    

    train = do_preprocess(train,

                             columns_tocategorize=columns_tocategorize,

                            columns_tooneshot=columns_tooneshot,

                            columns_todiscard=columns_todiscard,

                            columns_tonumeric=columns_tonumeric,

                            apply_apartment=True,

                            add_zip=True)

    display(f'train.index.nunique() = {train.index.nunique()}')



    

    valid = do_preprocess(valid,

                             columns_tocategorize=columns_tocategorize,

                            columns_tooneshot=columns_tooneshot,

                            columns_todiscard=columns_todiscard,

                            columns_tonumeric=columns_tonumeric,

                            apply_apartment=True,

                            add_zip=True)

    display(f'valid.index.nunique() = {valid.index.nunique()}')

    

    test = do_preprocess(test,

                             columns_tocategorize=columns_tocategorize,

                            columns_tooneshot=columns_tooneshot,

                            columns_todiscard=columns_todiscard,

                            columns_tonumeric=columns_tonumeric,

                            apply_apartment=True,

                            add_zip=True)

    display(f'test.index.nunique() = {test.index.nunique()}')



    

    df_all = pd.concat([valid, test, ])

    display(f'train.shape = {train.shape}')

    display(f'df_all.shape = {df_all.shape}')

    display(f'df_all.columns = {df_all.columns.tolist()}')

    display(df_all.head())



    m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)

    df_train_x, df_train_y = split_xy(train, 'sale_price')

    m.fit(df_train_x, df_train_y)



    y = m.predict(df_all.drop('sale_id', axis=1))

    x = df_all['sale_id']

    df_ans = pd.DataFrame()

    df_ans['sale_id'] = x

    df_ans['sale_price'] = y

    df_ans.sort_values('sale_id', inplace=True)

    display(df_ans.head())

    display(df_ans.shape)



    df_ans.to_csv('submission.csv', index=False)

    validate_solution()



    

get_submission_csv()