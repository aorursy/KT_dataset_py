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
data = pd.read_csv('/kaggle/input/used-car-dataset-ford-and-mercedes/vw.csv')
data
data.fuelType.value_counts()
data.model.value_counts()
data.transmission.value_counts()
# label them

auto = {'Manual':0,'Semi-Auto':1, 'Automatic':2}
data['transmission'] = data['transmission'].map(auto)
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
data['model'] = lb.fit_transform(data['model'])
fuel = {'Diesel':0, 'Petrol': 1, 'Hybrid':2, 'Other':3}
data['fuelType'] = data['fuelType'].map(fuel)
data
import h2o
from h2o.automl import H2OAutoML
h2o.init()
train_data = h2o.H2OFrame(data)
train_data['fuelType'] = train_data['fuelType'].asfactor()
train_data
ml = H2OAutoML(max_models = 10, seed=10, exclude_algos=['DeepLearning','StackedEnsemble'], verbosity = 'info', nfolds=0, balance_classes=True, max_after_balance_size=0.3)
X = data.drop(columns='fuelType')
Y = data['fuelType']
x = list(X.columns) 
y = 'fuelType'
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
training = pd.concat([X_train, y_train], axis=1)
testing = pd.concat([X_test, y_test], axis=1)
training = h2o.H2OFrame(training)
testing = h2o.H2OFrame(testing)

training['fuelType'] = training['fuelType'].asfactor()
testing['fuelType'] = testing['fuelType'].asfactor()
ml.train(x = x, y= y, training_frame = training, validation_frame = testing)
ml.leaderboard
pred = ml.leader.predict(testing)
ml.leader.model_performance(testing)
model_id = list(ml.leaderboard['model_id'].as_data_frame().iloc[:,0])
model_id
out = h2o.get_model([mod for mod in model_id if 'XGBoost' in mod][0])
out
params = out.convert_H2OXGBoostParams_2_XGBoostParams()
Y
from xgboost import XGBClassifier
model = XGBClassifier(params)
model.fit(X_train, y_train)

pred = model.predict(X_test)
np.mean(pred == y_test)
pred = model.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(pred, y_test))
data
from sklearn.utils.class_weight import compute_class_weight
class_weigth = list(compute_class_weight('balanced',
                                        np.unique(data['fuelType']),
                                        data['fuelType']))
class_weigth
weight = np.ones(X_train.shape[0], dtype='float')
weight

for i,v in enumerate(y_train):
    weight[i] = class_weigth[v]
# for i,v in enumerate(y_train):
#     print(weight[i])
weight
model.fit(X_train, y_train, sample_weight= weight)
pred = model.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(pred, y_test))