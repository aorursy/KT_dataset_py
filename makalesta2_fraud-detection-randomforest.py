# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd 

from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score

from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score

import matplotlib.pyplot as plt

from sklearn.preprocessing import scale 

from sklearn import model_selection

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import BaggingRegressor



from warnings import filterwarnings

filterwarnings('ignore')

train_identity=pd.read_csv('../input/train_identity.csv', index_col='TransactionID')
train_identity.shape
train_transaction=pd.read_csv('../input/train_transaction.csv', index_col='TransactionID')
train_transaction.shape
test_transaction = pd.read_csv('../input/test_transaction.csv', index_col='TransactionID')

test_identity = pd.read_csv('../input/test_identity.csv', index_col='TransactionID')

sample_submission = pd.read_csv('../input/sample_submission.csv', index_col='TransactionID')
train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)

test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)
y_train_full = train['isFraud'].copy()

X_train_full = train.drop('isFraud', axis=1)

X_test = test.copy()
del train_transaction, train_identity, test, test_transaction, test_identity
X_train_full = X_train_full.fillna(-999)

X_test = X_test.fillna(-999)
from sklearn import preprocessing



for f in X_train_full.columns:

    if X_train_full[f].dtype=='object' or X_test[f].dtype=='object': 

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(X_train_full[f].values) + list(X_test[f].values))

        X_train_full[f] = lbl.transform(list(X_train_full[f].values))

        X_test[f] = lbl.transform(list(X_test[f].values)) 
X_train, X_val, Y_train, Y_val = train_test_split(X_train_full, y_train_full, 

                                                    test_size=0.15, 

                                                    random_state=42)
del X_train_full, y_train_full
from sklearn.ensemble import RandomForestRegressor

rf_tuned = RandomForestRegressor(max_depth  = 45,max_features = 30, n_estimators =500, n_jobs=-1, min_samples_leaf=200)
%time rf_tuned.fit(X_train, Y_train)
print("Square root of error squares",np.sqrt(mean_squared_error(Y_val,rf_tuned.predict(X_val))))

print("Roc Auc Score:",roc_auc_score(Y_val,rf_tuned.predict(X_val)))
sample_submission['isFraud'] = rf_tuned.predict(X_test)

sample_submission.to_csv('fraud-detection-submission.csv')
import matplotlib.pyplot as plt



Importance = pd.DataFrame(index=X_train.columns)

Importance['Importance'] = rf_tuned.feature_importances_*100

Importance.loc[Importance['Importance'] > 1.5].sort_values('Importance').head(40).plot(kind='barh', figsize=(14, 28),color="r", title='Feature Importance')

plt.xlabel("Variable Severity Levels")

plt.show()