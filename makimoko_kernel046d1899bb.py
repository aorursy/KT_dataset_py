# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # visualization library



#warnings.filterwarnings('ignore')

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
TRAIN_PATH = os.path.join("..", "input", "train.csv")

TEST_PATH = os.path.join("..", "input", "test.csv")



train = pd.read_csv(TRAIN_PATH)

test = pd.read_csv(TEST_PATH)



print(f"training set shape : {train.shape}")

print(f"testing set shape : {test.shape}")
train.head()
train.info()
train.isna().sum()
train.trip_duration.min()
train.trip_duration.max()
train.describe()
plt.subplots(figsize=(18,6))

plt.title("Visualisation des outliers")

train.boxplot();
col_diff = list(set(train.columns).difference(set(test.columns)))

col_diff
#Calcule de la distance entre drop et pickup

train['dist'] = np.sqrt((train['pickup_latitude']-train['dropoff_latitude'])**2

                        + (train['pickup_longitude']-train['dropoff_longitude'])**2)

test['dist'] = np.sqrt((test['pickup_latitude']-test['dropoff_latitude'])**2

                        + (test['pickup_longitude']-test['dropoff_longitude'])**2)
#il n'est pas nécessaire d'avoir 0 passager, nous allons les enlever

train = train[train['passenger_count']>= 1]
# La durée du voyage est comprise entre 1 sec. et 3526282 sec.

# Nous laisserons tomber les valeurs inférieures à 1 min 30 (90sec) et supérieures à 166 min (10 000 sec).

train = train[train['trip_duration']>= 1.5 ]

train = train[train['trip_duration']<= 10000 ]
# Nous allons laisser tomber la longitude et la latitude (On drop ce qui ressemble à des valeurs aberrantes)

train = train.loc[train['pickup_longitude']> -90]

train = train.loc[train['pickup_latitude']< 47.5]



train = train.loc[train['dropoff_longitude']> -90]

train = train.loc[train['dropoff_latitude']> 34]
#convertir le string en datetime pour avoir que l'heure 

train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'], format='%Y-%m-%d %H:%M:%S')

test['pickup_datetime'] = pd.to_datetime(test['pickup_datetime'], format='%Y-%m-%d %H:%M:%S')
#ajout dans une nouvelle colonne

train['hour'] = train.loc[:,'pickup_datetime'].dt.hour;

test['hour'] = test.loc[:,'pickup_datetime'].dt.hour;
#Ajout dans une nouvelle colonne à train 

X_train = train[["passenger_count","vendor_id", "pickup_longitude", "pickup_latitude", "dropoff_longitude","dropoff_latitude", "dist", "hour" ]]

y_train = train["trip_duration"]  # This is our target

X_train = train[["passenger_count","vendor_id", "pickup_longitude", "pickup_latitude", "dropoff_longitude","dropoff_latitude", "dist","hour" ]]
# importer la lib pour cross valider le model

from sklearn.model_selection import cross_val_score



# importer la lib pour la regression de Random Forest

from sklearn.ensemble import RandomForestRegressor



# importer la lib pour la regression de Random Forest

from sklearn.linear_model import SGDRegressor



from sklearn.linear_model import LinearRegression



from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, random_state=42)

sgd = SGDRegressor()

sgd.fit(X_train, y_train)
b2o = RandomForestRegressor(n_estimators=19, min_samples_split=2, min_samples_leaf=4, max_features='auto', max_depth=80, bootstrap=True)

b2o.fit(X_train, y_train)
cv = ShuffleSplit(n_splits=4, test_size=0.8, random_state=42)

cv_scores = cross_val_score(b2o, X_train, y_train, cv=cv, scoring= 'neg_mean_squared_log_error')
cv_scores
for i in range(len(cv_scores)):

    cv_scores[i] = np.sqrt(abs(cv_scores[i]))

print(np.mean(cv_scores))
## Prediction
test.head()
X_fit = test[["vendor_id", "passenger_count","pickup_longitude", "pickup_latitude","dropoff_longitude","dropoff_latitude","dist","hour"]]

prediction = b2o.predict(X_fit)

prediction
my_submission = pd.DataFrame({'id': test.id, 'trip_duration': prediction})

my_submission.head()
my_submission.to_csv('submission.csv', index=False)