# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline


# Input data files are available in the "../input/" directory.
train = pd.read_csv('../input/train.csv')
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
train.head()

train.describe()
train.dtypes

y_train = train['SalePrice']
sns.distplot(y_train)
(train.isnull().sum().sort_values(ascending = False))
# Make imputations
def imputate_df(data):
    """
    Function to make imputations based on means
    Returns a dataframe with imputations
    """
    from sklearn.preprocessing import Imputer
    my_imputer = Imputer()
    data_with_imputed_values = my_imputer.fit_transform(data)
    return pd.DataFrame(data_with_imputed_values, columns=data.columns)
# Drop some cloumsn and One hot encoding
X_train = train.drop(['Id','SalePrice'],axis=1) # drop id and target 
X_train = pd.get_dummies(X_train) # one-hot encoding
X_train = imputate_df(X_train) # make imputations
X_train.head()
# check for nulls
X_train.isnull().sum().sum()




from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=50, criterion='mse', max_depth=None, 
                                 min_samples_split=2, min_samples_leaf=2, 
                                 min_weight_fraction_leaf=0.0, max_features='auto', 
                                 max_leaf_nodes=None, min_impurity_decrease=0.0,
                                 min_impurity_split=None, bootstrap=True, oob_score=False, 
                                 n_jobs=-1, random_state=100, verbose=1, warm_start=False)
rf_results = rf_model.fit(X_train, y_train)


rf_model.score(X_train,y_train)
test = pd.read_csv('../input/test.csv')
test.head()
# check for nans
test.isnull().sum().sum()
# Drop some cloumsn and One hot encoding
X_test = test.drop(['Id'],axis=1) # drop id and target 
X_test = pd.get_dummies(X_test) # one-hot encoding
X_test = imputate_df(X_test) # make imputations
X_test.head()
# check columns
print(len(X_train.columns))
print(len(X_test.columns))
#Missing columns
missing_columns = list(X_train.columns.difference(X_test.columns))
print(missing_columns)
len(missing_columns)

for col in missing_columns:
    X_test[col] = 0
X_test[missing_columns].head()
SalePrice = rf_model.predict(X_test)
Id = pd.Series(test.index) + 1461
my_submission = pd.DataFrame({'Id': Id,'SalePrice':SalePrice})
my_submission.tail()
#Submit
my_submission.to_csv('submission.csv', index=False)
