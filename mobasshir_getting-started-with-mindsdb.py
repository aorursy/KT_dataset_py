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
!wget https://s3.eu-west-2.amazonaws.com/mindsdb-example-data/home_rentals.csv
for dirname, _, filenames in os.walk('/kaggle/working'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
!pip install mindsdb
import mindsdb
!mkdir models


# Instantiate a mindsdb Predictor
mdb = mindsdb.Predictor(name='real_estate_model',root_folder="/kaggle/working/models",log_level=50)

# We tell the Predictor what column or key we want to learn and from what data
mdb.learn(from_data="/kaggle/working/home_rentals.csv", to_predict='rental_price')

# mdb = mindsdb.Predictor(name='real_estate_model')

# Predict a single data point
result = mdb.predict(when={'number_of_rooms': 2,'number_of_bathrooms':1, 'sqft': 1190})
print('The predicted price is ${price} with {conf} confidence'.format(price=result[0]['rental_price'], conf=result[0]['rental_price_confidence']))
# mindsdb.explanation()
models = mdb.get_models()
models
mdb.get_model_data(model_name='real_estate_model')
mdb.export_model(model_name='real_estate_model')
mdb.analyse_dataset(from_data="/kaggle/working/home_rentals.csv")

