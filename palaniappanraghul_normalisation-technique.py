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
from sklearn import preprocessing

# Get dataset

df = pd.read_csv("/kaggle/input/california-housing-train/california_housing_train.csv", sep=",")

df
# Normalize total_bedrooms column

x_array = np.array(df['total_bedrooms'])

x_array
normalized_X = preprocessing.normalize([x_array])

normalized_X
# Get column names first

names = df.columns

# Create the Scaler object

scaler = preprocessing.StandardScaler()

# Fit your data on the scaler object

scaled_df = scaler.fit_transform(df)

scaled_df
scaled_df = pd.DataFrame(scaled_df, columns=names)

scaled_df
# Create range for your new columns

lat_range = zip(range(32, 44), range(33, 45))

new_df = pd.DataFrame()

# Iterate and create new columns, with the 0 and 1 encoding

for r in lat_range:

        new_df["latitude_%d_to_%d" % r] = df["latitude"].apply(

            lambda l: 1.0 if l >= r[0] and l < r[1] else 0.0)

new_df