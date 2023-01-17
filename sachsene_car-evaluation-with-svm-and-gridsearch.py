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
import pandas as pd

pd.options.display.max_colwidth = 80



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OrdinalEncoder



from sklearn.svm import SVC # SVM model with kernels

from sklearn.model_selection import GridSearchCV



from sklearn.metrics import accuracy_score



from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report



import warnings

warnings.filterwarnings('ignore')
data = '/kaggle/input/car-evaluation-data-set/car_evaluation.csv'



header_list = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class value']



cars = pd.read_csv(data, names=header_list, index_col=None)
cars.head()
cars.describe()
cars.info(), cars.shape
for column in cars.columns:

    print(cars[column].value_counts(), '\n') 
a = cars.loc[cars['doors'] == '2', ['lug_boot']]

b = cars.loc[cars['doors'] == '3', ['lug_boot']]

c = cars.loc[cars['doors'] == '4', ['lug_boot']]

d = cars.loc[cars['doors'] == '5more', ['lug_boot']]



print(a['lug_boot'].value_counts(), '\n\n', b['lug_boot'].value_counts(), '\n\n', 

      c['lug_boot'].value_counts(), '\n\n', d['lug_boot'].value_counts())
X = cars.drop(['class value'], axis=1)

y = cars['class value']



X, y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 42)
X_train.shape, X_test.shape
y_train.shape, y_test.shape
columns_encode = []

columns_encode.append(header_list)

columns_encode
ordinal_encoder = OrdinalEncoder()



X_train = ordinal_encoder.fit_transform(X_train, columns_encode)

X_test = ordinal_encoder.transform(X_test)
X_train, X_train.shape
y_train, y_train.shape
param_grid = [{'kernel': ['poly'], 'C' : [3, 5, 7, 9, 10]},

             {'kernel' : ['rbf'], 'C' : [3, 5, 7, 9, 10], 'gamma' : [2, 4, 6, 8]}]



svm = SVC()
grid_search = GridSearchCV(svm, param_grid, return_train_score=True)



grid_search.fit(X_train, y_train)
grid_search.best_params_
grid_search.best_estimator_
svm_y_pred = grid_search.predict(X_test)



accuracy_score(y_test, svm_y_pred)
svm_y_pred_train = grid_search.predict(X_train)



accuracy_score(y_train, svm_y_pred_train)
confusion_matrix(y_test, svm_y_pred)
print(classification_report(y_test, svm_y_pred))