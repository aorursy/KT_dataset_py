import pandas as pd

import numpy as np

    

data_frame = pd.read_csv('../input/craigslist-carstrucks-data/craigslistVehicles.csv' , sep=',')

data_frame = data_frame[data_frame['price'].between(5000, 300000)]

prices = data_frame['price']

data_frame = data_frame.drop(columns=['url', 'city', 'city_url', 'price', 'condition', 'cylinders', 'title_status', 'VIN', 'drive', 'size', 'image_url', 'desc'])
print(data_frame.dtypes)

print(data_frame.columns)
import math

def normalize_str(string):

#     print(string)

    if not isinstance(string, str):

        return string

    normal_string = ''.join(e for e in string if e.isalnum())

    return normal_string



string_columns = ['manufacturer', 'make', 'fuel', 'transmission', 'type', 'paint_color']

number_columns = ['long', 'lat', 'year', 'odometer']

for str_col in string_columns:

    data_frame[str_col] = data_frame[str_col].apply(normalize_str)

from sklearn.impute import SimpleImputer

cat_imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

data_frame[string_columns] = cat_imp.fit_transform(data_frame[string_columns])



num_imp = SimpleImputer(missing_values=np.nan, strategy='median')

data_frame[number_columns] = cat_imp.fit_transform(data_frame[number_columns])
cat_imp = SimpleImputer(missing_values='', strategy='most_frequent')

data_frame[string_columns] = cat_imp.fit_transform(data_frame[string_columns])
unique_makes = data_frame['make'].unique()

print(len(unique_makes))

unique_manufacturers = data_frame['manufacturer'].unique()

print(len(unique_manufacturers))

print()

print(data_frame.isnull().sum())
prices.to_csv('prices.csv', index=None, header=False, sep=';')

data_frame.to_csv('dataset_imputated.csv', index=None, header=True, sep=';')

print(prices)

print(data_frame)

print('done')