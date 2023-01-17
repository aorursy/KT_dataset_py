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

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split    

from sklearn.metrics import mean_absolute_error



# Read SCV files

train_csv = "/kaggle/input/house-prices-advanced-regression-techniques/train.csv"

test_csv = "/kaggle/input/house-prices-advanced-regression-techniques/test.csv"

train_data = pd.read_csv(train_csv)

test_data = pd.read_csv(test_csv)



# Select fratures and X, y

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

train_X = train_data[features]

train_y = train_data.SalePrice

test_X = test_data[features]

test_index = test_data.Id



def get_predictions(train_X, test_X, train_y):



    # RFR model

    model = RandomForestRegressor(random_state=1)

    model.fit(train_X, train_y)



    # Predict

    predictions = model.predict(test_X)



    return predictions



# Get Predictions

predictions = get_predictions(train_X, test_X, train_y)



# Mearge mats

result = np.vstack([test_index, predictions])



# Debug...

print(len(predictions))

print(type(predictions))

print(result.T)
# import the modules we'll need

from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv()

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



# create a random sample dataframe

df = pd.DataFrame(result.T)



# create a link to download the dataframe

create_download_link(df)