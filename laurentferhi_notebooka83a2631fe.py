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
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()

import pickle as pk
from time import time

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
practice = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
sub = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
label = data.pop('SalePrice')
def pre_process(df_train, df_test):
    # Convert all df to str
    df_train = df_train.applymap(lambda x: str(x))
    df_test = df_test.applymap(lambda x: str(x))
    # replace NaN by 0
    df_train.fillna('0', inplace=True) # if fill with int(0), label encoder fails
    df_test.fillna('0', inplace=True)
    # Drop Id
    df_train.drop('Id', axis=1, inplace=True)
    df_test.drop('Id', axis=1, inplace=True)
    # Label encode each non numerical features
    dict_encoder = {}
    for col in list(df_train.select_dtypes(include='object')):
        le = LabelEncoder()
        le.fit(list(df_train[col])+list(df_test[col]))
        df_train[col] = le.transform(df_train[col])
        df_test[col] = le.transform(df_test[col])
        dict_encoder[col] = le
    # Re-convert all df to float
    df_train = df_train.applymap(lambda x: float(x))
    df_test = df_test.applymap(lambda x: float(x))
    return df_train, df_test, dict_encoder
data, practice, dict_encoder = pre_process(data, practice)
enc = dict_encoder['MSZoning']
print('MSZoning'+' classes:',list(enc.classes_))

enc = dict_encoder['Fence']
print('Fence'+' classes:',list(enc.classes_))
X = np.array(data)
y = np.array(label)
print('X shape:',X.shape)
print('y shape:',y.shape)
# Generate train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Size of sets
print("Train set:",X_train.shape)
print("Train set:",y_train.shape)
print("Test set:",X_test.shape)
print("Test set:",y_test.shape)
# Ignore runtime warnings
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# cross-validator : ShuffleSplit 
ss = ShuffleSplit(n_splits = 10, test_size = 0.2, random_state = 42) # To avoid over-fitting

# Functions to be used in the pipeline
skb = SelectKBest(f_regression)

### Import regressor ###
reg = RandomForestRegressor()

# definition of the pipeline
pipeline = Pipeline(steps = [
    ("SKB",skb),
    ("RDFR",reg)
])   

# parameters to tune 
param_grid = {
    "SKB__k":[5,"all"],
    "RDFR__n_estimators":[100,200],
    "RDFR__max_depth":[None],
    "RDFR__min_samples_split":[2],
    "RDFR__min_samples_leaf":[2]
} 

# exhaustive search over specified parameter
grid = GridSearchCV(pipeline, param_grid, verbose = 1, cv = ss)
# training classifier
print (" > training classifier:")
t0 = time()
grid.fit(X_train, y_train.ravel())
print ("training time: ", round(time()-t0, 3), "s")

# best classifier using the cross-validator and the Stratified Shuffle Split 
reg = grid.best_estimator_

# predicition with the classifier
t0 = time()
pred = reg.predict(X_test)
print ("testing time: ", round(time()-t0, 3), "s")

# print grid parameters
print ("\n > Best grid search:")
print (grid.best_params_)

# dump classifier in a pickle file
print ("\n > Regressor dumped")
with open("housing_RFR_best_reg.pkl", 'wb') as file:
    pk.dump(reg, file)
RMSE = np.sqrt(mean_squared_error(y_test, pred))
R2 = r2_score(y_test, pred)
print('RMSE:',RMSE,'\nr2:',R2)
RFR_results = pd.DataFrame({
    'y_test':y_test,
    'pred':pred
})
sns.lmplot(data=RFR_results, x="y_test", y="pred")
# Submit predictions
submission = pd.DataFrame({
    'Id': pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')['Id'],
    'SalePrice': reg.predict(practice)
})
submission.to_csv('RFR_submission.csv', index=False)