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
from sklearn.model_selection import train_test_split



# Read the data

X1 = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv', index_col='TransactionID')

X2 = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv', index_col='TransactionID')



X = X1.merge(X2, how='left', left_index=True, right_index=True)

del X1, X2



low_cardinality_cols = [cname for cname in X.columns if X[cname].dtype == "object" and

                        X[cname].nunique() < 10]



# Select numeric columns

numeric_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]



# Keep selected columns only

my_cols = low_cardinality_cols + numeric_cols

X = X[my_cols].copy()



# Drop rows where the target is NaN

X.dropna(axis=0, subset=['isFraud'], inplace=True)



# One-hot encode the data

X = pd.get_dummies(X, drop_first=True)



# Convert the target to int

X['isFraud'] = X['isFraud'].apply(lambda x: int(x))



# Can't just drop NaN rows because the test set might contain NaN values

X.fillna(X.mean(), inplace=True)
def make_hour_feature(df, tname='TransactionDT'):

    """

    Creates an hour of the day feature, encoded as 0-23. 

    

    Parameters:

    -----------

    df : pd.DataFrame

        df to manipulate.

    tname : str

        Name of the time column in df.

    """

    hours = df[tname] / (3600)        

    encoded_hours = np.floor(hours) % 24

    return encoded_hours



X['hours'] = make_hour_feature(X)
y = X['isFraud'].copy()

X.drop(['isFraud'], axis=1, inplace=True)

model_cols = X.columns



X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)



# Feature Scaling

#from sklearn.preprocessing import StandardScaler

#sc_X = StandardScaler()

#X_train = sc_X.fit_transform(X_train)

#X_test = sc_X.transform(X_test)
from xgboost import XGBClassifier

xgb_model = XGBClassifier()

model = xgb_model

model.fit(X_train, y_train)



del X_train, y_train, X_valid, y_valid
#prob_predictions = model.predict_proba(X_test)[:,1]

#rounded_predictions = np.around(prob_predictions, decimals=0)
#from sklearn.metrics import confusion_matrix



#print(confusion_matrix(y_test, rounded_predictions))

#print(model.score(X_test,y_test))
# Read the test data

X_test1 = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv', index_col='TransactionID')

X_test2 = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv', index_col='TransactionID')

X_test = X_test1.merge(X_test2, how='left', left_index=True, right_index=True)

del X_test1, X_test2

my_cols.remove('isFraud')

X_test = X_test[my_cols].copy()
# One-hot encode the data

X_test = pd.get_dummies(X_test, drop_first=True)



X_test, X = X_test.align(X, axis=1, join='right')





# Filter for model columns

#test_cols = model_cols.drop('hours')

#X = X[test_cols]



# Can't just drop NaN rows because the test set might contain NaN values

X_test.fillna(X_test.mean(), inplace=True)



X_test['hours'] = make_hour_feature(X)
prob_predictions = model.predict_proba(X_test)[:,1]

#rounded_predictions = np.around(prob_predictions, decimals=1)
submission = pd.DataFrame(index=X_test.index, data={'isFraud': prob_predictions})

submission.to_csv('/kaggle/working/submission.csv')