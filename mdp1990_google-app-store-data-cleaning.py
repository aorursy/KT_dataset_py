# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt
#crating a Dataframe for the .csv file 
data = pd.read_csv("/kaggle/input/google-play-store-apps/googleplaystore.csv")
data.dropna(inplace=True)
def remove_postfix_and_prefix(column):
    column=column.str.replace('+','')
    column=column.str.replace(',','')
    column=column.str.replace('$','')
    column=column.str.replace("''","")
    return column

def change_datatype(column):
    column=column.astype('float')
    return column

def change_to_date_time(column):
    column=pd.to_datetime(column)
    return column
data['Rating'] = data['Rating'].fillna(data['Rating'].median())
data['Current Ver'] = data['Current Ver'].replace('Varies with device',np.nan)
data['Current Ver'] = data['Current Ver'].fillna(data['Current Ver'].mode()[0])

# Removing NaN values
data = data[pd.notnull(data['Last Updated'])]
data = data[pd.notnull(data['Content Rating'])]

# This is to be anomaly record.
i = data[data['Category'] == '1.9'].index
data.loc[i]
# Drop the anomaly record
data = data.drop(i)

data['Installs']=remove_postfix_and_prefix(data['Installs'])
data['Price']=remove_postfix_and_prefix(data['Price'])


data['Price']=change_datatype(data['Price'])
data['Installs']=change_datatype(data['Installs'])
data['Reviews']=change_datatype(data['Reviews'])

data['Last Updated']=change_to_date_time(data['Last Updated'])
data