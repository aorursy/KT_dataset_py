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
import zipfile



zf = zipfile.ZipFile('../input/bosch-production-line-performance/train_numeric.csv.zip') 

train_numeric_chunks = pd.read_csv(zf.open('train_numeric.csv'), iterator=True, chunksize=100000)



zf = zipfile.ZipFile('../input/bosch-production-line-performance/train_categorical.csv.zip') 

train_categorical_chunks = pd.read_csv(zf.open('train_categorical.csv'), iterator=True, chunksize=100000, low_memory=False)



zf = zipfile.ZipFile('../input/bosch-production-line-performance/train_date.csv.zip') 

train_date_chunks = pd.read_csv(zf.open('train_date.csv'), iterator=True, chunksize=100000)



pd.options.display.max_columns = None

pd.options.display.max_rows = None
'''

total_number_of_lines_numeric = 0

total_number_of_lines_categorical = 0

total_number_of_lines_date = 0



for chunk in train_numeric_chunks:

    total_number_of_lines_numeric += chunk.shape[0]



for chunk in train_categorical_chunks:

    total_number_of_lines_categorical += chunk.shape[0]

    

for chunk in train_date_chunks:

    total_number_of_lines_date += chunk.shape[0]

    

print("Number of lines in numeric data is {}".format(total_number_of_lines_numeric))

print("Number of lines in categorical data is {}".format(total_number_of_lines_categorical))

print("Number of lines in date data is {}".format(total_number_of_lines_date))

'''



total_number_of_lines_numeric = 1183747

total_number_of_lines_categorical = 1183747

total_number_of_lines_date = 1183747
def get_numeric_frame():

    for data_frame in train_numeric_chunks:

        yield data_frame

        

def get_categorical_frame():

    for data_frame in train_categorical_chunks:

        yield data_frame

        

def get_date_frame():

    for data_frame in train_date_chunks:

        yield data_frame



get_df_numeric = get_numeric_frame()

get_df_categorical = get_categorical_frame()

get_df_date = get_date_frame()
df_numeric = next(get_df_numeric)

df_categorical = next(get_df_categorical)

df_date = next(get_df_date)
df_numeric.info()
print("numeric has {} columns".format(len(df_numeric.columns)))

#df_numeric.columns.tolist()
#df_numeric

df_numeric.head()
df_numeric.describe()
df_categorical.info()
print("categorical has {} columns".format(len(df_categorical.columns)))

#df_categorical.columns.tolist()
#df_categorical

df_categorical.head()
df_categorical.describe()
df_date.info()
print("date has {} columns".format(len(df_date.columns)))

#df_date.columns.tolist()
#df_date

df_date.head()
df_date.describe()
fail_parts_numeric = df_numeric[df_numeric.Response == 1]

fail_parts_categorical = df_categorical[df_numeric.Response == 1]

fail_parts_date = df_date[df_numeric.Response == 1]

working_parts = df_numeric[df_numeric.Response == 0]
print("Number of fails in first 100000 parts: {}".format(len(fail_parts_numeric)))

print("Number of working parts in first 100000 parts: {}".format(len(working_parts)))

print("Fail/Working ratio: {}".format(len(fail_parts_numeric)/len(working_parts)))
df_numeric['Response'].value_counts().sort_index().plot.bar()
#fail_parts_numeric

fail_parts_numeric.head()
#fail_parts_categorical

fail_parts_categorical.head()
#fail_parts_date

fail_parts_date.head()