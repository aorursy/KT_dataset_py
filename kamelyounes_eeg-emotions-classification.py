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
df = pd.read_csv('../input/eeg-brainwave-dataset-feeling-emotions/emotions.csv')
df.head(10)
print(df.shape)
df.isnull().sum().sum()
df.info()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


X = df.drop(columns=['label'])
y = df['label']

scaler = StandardScaler()
X = scaler.fit_transform(X)

enc = LabelEncoder()
y = enc.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X
y
X_train.shape
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


parameters_svc = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
grid_search_svc = GridSearchCV(SVC(), parameters_svc, n_jobs=-1)
grid_search_svc.fit(X_train, y_train)


grid_search_svc.best_score_
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV


xgb_clf = xgb.XGBClassifier()

parameters_xgb =    {
            'max_depth': [3, 4, 5],
            'learning_rate': [0.1, 0.2, 0.3],
            'n_estimators': [50, 100, 150],
            'gamma': [0, 0.1, 0.2],
            'min_child_weight': [0, 0.5, 1],
            'max_delta_step': [0],
            'subsample': [0.7, 0.8, 0.9, 1],
            'colsample_bytree': [0.6, 0.8, 1],
            'colsample_bylevel': [1],
            'reg_alpha': [0, 1e-2, 1, 1e1],
            'reg_lambda': [0, 1e-2, 1, 1e1],
            'base_score': [0.5]
            }

search_xgb = RandomizedSearchCV(xgb_clf, parameters_xgb, n_jobs=-1)
search_xgb.fit(X_train, y_train)




search_xgb.best_score_
from keras.models import Sequential

model = Sequential()
from keras.layers import Dense

model.add(Dense(32, activation='relu', input_shape=(1,2548)))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation="softmax"))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
from keras.utils import to_categorical

X_train_ann, X_val_ann, y_train_ann, y_val_ann = train_test_split(X_train, y_train, test_size=0.33, random_state=42)

y_train_ann = to_categorical(y_train_ann, 3)
y_val_ann = to_categorical(y_val_ann, 3)

history = model.fit(X_train_ann, y_train_ann, epochs=100)

model.evaluate(X_val_ann, y_val_ann)
best_xgb = search_xgb.best_estimator_
best_xgb
best_xgb.score(X_test, y_test)
X_test_np = np.array([X_test[98]])
best_xgb.predict(X_test_np)