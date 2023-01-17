# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import io



retail_raw = pd.read_csv('../input/retail-raw/retail_raw_reduced_data_quality.csv')

print(retail_raw.dtypes)



#Descriptive Statistics - Part 1

# Kolom city

length_city =len(retail_raw['city'])

print('Length kolom city:', length_city)



# Tugas Praktek: Kolom product_id

length_product_id = len(retail_raw['product_id'])

print('Length kolom product_id:', length_product_id)



# Count kolom city

count_city= retail_raw['city'].count()

print('Count kolom count_city:', count_city)



# Tugas praktek: count kolom product_id

count_product_id = retail_raw['product_id'].count()

print('Count kolom product_id:', count_product_id)



# Missing value pada kolom city

number_of_missing_values_city = length_city - count_city

float_of_missing_values_city = float(number_of_missing_values_city/length_city)

pct_of_missing_values_city = '{0:.1f}%'.format(float_of_missing_values_city * 100)

print('Persentase missing value kolom city:', pct_of_missing_values_city)



# Tugas praktek: Missing value pada kolom product_id

number_of_missing_values_product_id = length_product_id - count_product_id

float_of_missing_values_product_id = float(number_of_missing_values_product_id/length_product_id)

pct_of_missing_values_product_id = '{0:.1f}%'.format(float_of_missing_values_product_id * 100)

print('Persentase missing value kolom product_id:', pct_of_missing_values_product_id)



# Deskriptif statistics kolom quantity

print('Kolom quantity')

print('Minimum value: ', retail_raw['quantity'].min())

print('Maximum value: ', retail_raw['quantity'].max())

print('Mean value: ', retail_raw['quantity'].mean())

print('Mode value: ', retail_raw['quantity'].mode())

print('Median value: ', retail_raw['quantity'].median())

print('Standard Deviation value: ', retail_raw['quantity'].std())



# Tugas praktek: Deskriptif statistics kolom item_price

print('item_price')

print('Minimum value: ', retail_raw['item_price'].min())

print('Maximum value: ', retail_raw['item_price'].max())

print('Mean value: ', retail_raw['item_price'].mean())

print('Mode value: ', retail_raw['item_price'].mode())

print('Median value: ', retail_raw['item_price'].median())

print('Standard Deviation value: ', retail_raw['item_price'].std())



# Deskriptif statistics kolom quantity

print('Kolom quantity')

print('Minimum value: ', retail_raw['quantity'].min())

print('Maximum value: ', retail_raw['quantity'].max())

print('Mean value: ', retail_raw['quantity'].mean())

print('Mode value: ', retail_raw['quantity'].mode())

print('Median value: ', retail_raw['quantity'].median())

print('Standard Deviation value: ', retail_raw['quantity'].std())



# Tugas praktek: Deskriptif statistics kolom item_price

print('')

print('Kolom item_price')

print('Minimum value: ', retail_raw['item_price'].min())

print('Maximum value: ', retail_raw['item_price'].max())

print('Mean value: ', retail_raw['item_price'].mean())

print('Median value: ', retail_raw['item_price'].median())

print('Standard Deviation value: ', retail_raw['item_price'].std())



# Quantile statistics kolom quantity

print('Kolom quantity:')

print(retail_raw['quantity'].quantile([0.25, 0.5, 0.75]))



# Tugas praktek: Quantile statistics kolom item_price

print('')

print('Kolom item_price:')

print(retail_raw['item_price'].quantile([0.25, 0.5, 0.75]))



print('Korelasi quantity dengan item_price')

print(retail_raw[['quantity', 'item_price']].corr())



import pandas_profiling 

pd = pandas_profiling.ProfileReport(retail_raw)
print('>>>>> CLEANSING DATA <<<<')

# Check kolom yang memiliki missing data

print('Check kolom yang memiliki missing data:')

print(retail_raw.isnull().any())



# Filling the missing value (imputasi)x

print('\nFilling the missing value (imputasi):')

print(retail_raw['quantity'].fillna(retail_raw['quantity'].mean()))



# Drop missing value

print('\nDrop missing value:')

print(retail_raw['quantity'].dropna())



#PRAKTEK KERJA

print(retail_raw['item_price'].fillna(retail_raw['item_price'].mean()))



#OUTLIERS

# Q1, Q3, dan IQR

Q1 = retail_raw['quantity'].quantile(0.25)

Q3 = retail_raw['quantity'].quantile(0.75)

IQR = Q3 - Q1



# Check ukuran (baris dan kolom) sebelum data yang outliers dibuang

print('Shape awal: ', retail_raw.shape)



# Removing outliers

retail_raw = retail_raw[~((retail_raw['quantity'] < (Q1 - 1.5 *IQR)) | (retail_raw['quantity'] > (Q3 +1.5 *IQR)))]



# Check ukuran (baris dan kolom) setelah data yang outliers dibuang

print('Shape akhir: ', retail_raw.shape)



#TUGAS PRAKTEK

# Q1, Q3, dan IQR

Q1 = retail_raw['item_price'].quantile(0.25)

Q3 = retail_raw['item_price'].quantile(0.75)

IQR = Q3 - Q1



# Check ukuran (baris dan kolom) sebelum data yang outliers dibuang

print('Shape awal: ', retail_raw.shape)



# Removing outliers

retail_raw = retail_raw[~((retail_raw['item_price'] < (Q1 - 1.5 * IQR)) | (retail_raw['item_price'] > (Q3 + 1.5 *IQR)))]



# Check ukuran (baris dan kolom) setelah data yang outliers dibuang

print('Shape akhir: ', retail_raw.shape)



# Check ukuran (baris dan kolom) sebelum data duplikasi dibuang

print('Shape awal: ', retail_raw.shape)



# Buang data yang terduplikasi

retail_raw.drop_duplicates(inplace=True)



# Check ukuran (baris dan kolom) setelah data duplikasi dibuang

print('Shape akhir: ', retail_raw.shape)


