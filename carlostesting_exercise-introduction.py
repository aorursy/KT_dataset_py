# Set up code checking

import os

if not os.path.exists("../input/train.csv"):

    os.symlink("../input/home-data-for-ml-course/train.csv", "../input/train.csv")  

    os.symlink("../input/home-data-for-ml-course/test.csv", "../input/test.csv")  

from learntools.core import binder

binder.bind(globals())

from learntools.ml_intermediate.ex1 import *
import pandas as pd

from sklearn.model_selection import train_test_split



# Read the data

X_full = pd.read_csv('../input/train.csv', index_col='Id')

y = X_full.SalePrice  # TARGET



features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']  #Selected features

X = X_full[features].copy()

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)  # TRAIN + VALIDATION



X_test_full = pd.read_csv('../input/test.csv', index_col='Id')

X_test = X_test_full[features].copy()
from sklearn.ensemble import RandomForestRegressor

model_3 = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0)

models = [model_3]



from sklearn.metrics import mean_absolute_error



# Function for comparing different models

def score_model(model, X_t=X_train, X_v=X_valid, y_t=y_train, y_v=y_valid):

    model.fit(X_t, y_t)

    preds = model.predict(X_v)

    return mean_absolute_error(y_v, preds)



for i in range(0, len(models)):

    mae = score_model(models[i])

    print("Model %d MAE: %d" % (i+1, mae))
my_model = RandomForestRegressor(n_estimators=200, criterion='mae', min_samples_split=5, random_state=1)

my_model.fit(X, y)  # FIT with the train+validation



preds_test = my_model.predict(X_test)  # PREDICT

output = pd.DataFrame({'Id': X_test.index, 'SalePrice': preds_test}) # Save to file in the proper format to be submitted

output.to_csv('submission.csv', index=False)
