# import some libraries
import pandas as pd
import numpy as np

# load data
trainset = pd.read_csv('../input/train.csv')
testset = pd.read_csv('../input/test.csv')

# create target vector
y = trainset.iloc[:, 1].values

# drop some columns such as ID and target
trainset = trainset.iloc[:,2:]
testset = testset.iloc[:,1:]
# create numpy arrays
X = trainset.iloc[:, :].values
X_Test = testset.iloc[:, :].values
# Applying LDA for dimensionality reduction
# Here is where the data leakage occurs
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA()
X = lda.fit_transform(X, y.astype(int))
X_Test = lda.transform(X_Test)
# create out train/test split and prepare for LightGBM model training
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=42)

import lightgbm as lgb
train = lgb.Dataset(X_train, label=np.log1p(y_train))
test = lgb.Dataset(X_test, label=np.log1p(y_test))
# Train LightGBM model with simple default parameters
params = {
    'boosting_type': 'gbdt',
    'metric': 'rmse',
    'zero_as_missing':True
    }
regressor = lgb.train(params, 
                      train, 
                      3000, 
                      valid_sets=[test], 
                      early_stopping_rounds=100, 
                      verbose_eval=100)