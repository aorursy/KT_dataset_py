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
data = pd.read_csv(os.path.join(dirname, filename))

data.head(5)
data.columns = ['country', 'year', 'sex', 'age', 'suicides_no', 'population',

       'suicides_100kpop', 'country-year', 'HDI_for_year',

       'gdp_for_year_dollars', 'gdp_per_capita_dollars', 'generation']
del data['country-year']

del data['HDI_for_year']
data['gdp_for_year_dollars'] = data['gdp_for_year_dollars'].str.replace(',','').astype(int)
no_categorical_column_data = data
del no_categorical_column_data['sex']

del no_categorical_column_data['age']

del no_categorical_column_data['generation']
no_categorical_column_data
data['risk'] = data.suicides_100kpop.copy()

data['risk'] = np.where(data.risk < data.suicides_100kpop.mean(), 0, 1)
#alfabetik sırasıyla

from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()



no_categorical_column_data.country = le.fit_transform(no_categorical_column_data.country) 

no_categorical_column_data.country.unique()
#Risk sütunu 0 ez az riskli, 1 en fazla riskli olarak ayarlandı.

data.describe()
X = np.asarray(no_categorical_column_data)

y = np.asarray(data['risk'])
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X)

X_scaled = scaler.transform(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=4)



print('Train set:', X_train.shape, y_train.shape)

print('Test set:', X_test.shape, y_test.shape)
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix
dt = DecisionTreeClassifier(random_state=42)

dt = dt.fit(X_train, y_train)
dt.tree_.node_count, dt.tree_.max_depth
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



def measure_error(y_true, y_pred, label):

    return pd.Series({'accuracy':accuracy_score(y_true, y_pred),

                      'precision': precision_score(y_true, y_pred),

                      'recall': recall_score(y_true, y_pred),

                      'f1': f1_score(y_true, y_pred)},

                      name=label)
# The error on the training and test data sets

y_train_pred = dt.predict(X_train)

y_test_pred = dt.predict(X_test)



train_test_full_error = pd.concat([measure_error(y_train, y_train_pred, 'train'),

                              measure_error(y_test, y_test_pred, 'test')],

                              axis=1)



train_test_full_error
from sklearn.model_selection import GridSearchCV



param_grid = {'max_depth':range(1, dt.tree_.max_depth+1, 2),

              'max_features': range(1, len(dt.feature_importances_)+1)}



GR = GridSearchCV(DecisionTreeClassifier(random_state=42),

                  param_grid=param_grid,

                  scoring='accuracy',

                  n_jobs=-1)



GR = GR.fit(X_train, y_train)
GR.best_estimator_.tree_.node_count, GR.best_estimator_.tree_.max_depth
y_train_pred_gr = GR.predict(X_train)

y_test_pred_gr = GR.predict(X_test)



train_test_gr_error = pd.concat([measure_error(y_train, y_train_pred_gr, 'train'),

                                 measure_error(y_test, y_test_pred_gr, 'test')],

                                axis=1)
train_test_gr_error

#what's wroong?