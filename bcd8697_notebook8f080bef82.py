from google.colab import files
files.upload() #upload kaggle.json 
# Let's make sure the kaggle.json file is present.
!ls -lha kaggle.json
# Next, install the Kaggle API client.
!pip install -q kaggle
# The Kaggle API client expects this file to be in ~/.kaggle,
# so move it there.
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/

# This permissions change avoids a warning on Kaggle tool startup.
!chmod 600 ~/.kaggle/kaggle.json
# Copy the stackoverflow data set locally.
!kaggle competitions download -c house-prices-advanced-regression-techniques
#installing
!pip install catboost
# importing
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
# Opening tables with pandas
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")
df_train
df_test
# data check
df_train.info()
# splitting
y_train = df_train["SalePrice"]
X_train = df_train.drop("SalePrice", axis = 1)
X_train
X_train.info()
X_test = df_test
#Let's fill in the missing values with means
X_train = X_train.fillna(method='ffill')
X_train = X_train.fillna(method='bfill')

X_test = X_test.fillna(method='ffill')
X_test = X_test.fillna(method='bfill')

X_train
X_train.info()
list_of_cat_features = list(X_train.select_dtypes(include=['object']).columns)
list_of_cat_features
# creating model for CatBoostRegressor
model = CatBoostRegressor(iterations=1300,
                          learning_rate=0.7,
                          depth=7,
                          cat_features = list_of_cat_features,
                          one_hot_max_size=26)
# Fit model
model.fit(X_train, y_train)
preds = model.predict(X_test)
preds
sample = pd.read_csv("sample_submission.csv")

sample = sample.drop("SalePrice", axis = 1)
sample["SalePrice"] = preds
sample
sample.to_csv('sample_10.csv', index=False)
