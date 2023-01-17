# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy.spatial.distance import pdist

from scipy.spatial.distance import squareform



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.


# save filepath to variable for easier access

melbourne_file_path = '../input/melbourne-housing-market/MELBOURNE_HOUSE_PRICES_LESS.csv'

# read the data and store data in DataFrame titled melbourne_data

melbourne_data = pd.read_csv(melbourne_file_path)
#To select the first column ‘Suburb’,

melbourne_data['Suburb']
melbourne_data.Suburb
#To select multiple columns

melbourne_data_four = melbourne_data[['Suburb', 'Address', 'Rooms', 'Type']]
#assign all your columns

cols = ['Suburb', 'Address', 'Rooms', 'Type']

melbourne_data_four = melbourne_data[cols]

#To select columns using select_dtypes method

melbourne_data.get_dtype_counts()
#To select only the float columns

melbourne_data.select_dtypes(include = ['float'])
#filter method to select columns based on the column names or index labels

melbourne_data.filter(like = 'Room')


melbourne_data.iloc[2]
melbourne_data.set_index('Rooms', inplace=True)

melbourne_data.loc[2]
#To select rows with different index positions

melbourne_data.iloc[[1, 4, 7]]
#You can use slicing to select multiple rows

melbourne_data.iloc[1:4]



#The above operation selects rows 2, 3 and 4
melbourne_data.iloc[:, [3, 4, 6]]
melbourne_data.loc[:, ['Price', 'Method', 'Date']]
melbourne_data_num = melbourne_data.loc[1:5,['Price','Distance']]

sf = squareform(pdist(melbourne_data_num))
pdist(melbourne_data_num, metric = 'mahalanobis')
pd.DataFrame(

squareform(pdist(melbourne_data_num)),

columns = ['1', '2', '3', '4', '5'],

index = ['1', '2', '3', '4', '5'])
import seaborn as sns

sns.heatmap(sf)