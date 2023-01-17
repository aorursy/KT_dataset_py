import pandas as pd

import numpy as np

import matplotlib.pyplot as plt 



print('Hello Capstone Project Course')
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv("/kaggle/input/data-collisions/Data-Collisions.csv")
collisions = data[['SPEEDING','SEVERITYCODE','LIGHTCOND','WEATHER']].copy()

collisions.head()
collisions['SPEEDING'].unique()
collisions['SPEEDING'].fillna(0, inplace = True)

collisions['SPEEDING'].replace('Y',1, inplace = True)

collisions['SPEEDING'].unique()
collisions['SEVERITYCODE'].unique()
collisions['LIGHTCOND'].unique()
collisions.replace({'LIGHTCOND' : {'Daylight' : 2, 'Dark - Street Lights On':1, 'Dark - No Street Lights': 0,

       'Dusk': 1, 'Dawn':1}}, inplace = True)                   

print(collisions['LIGHTCOND'].unique())
collisions.replace({'LIGHTCOND' : {'Dark - Street Lights Off':0,

       'Dark - Unknown Lighting':1, 'Unknown' :3 , 'Other' : 3 }}, inplace = True) 

collisions['LIGHTCOND'].fillna(3, inplace = True)

print(collisions['LIGHTCOND'].unique())
collisions['WEATHER'].unique()
collisions.replace({'WEATHER' : {'Overcast':1, 'Raining':2, 'Clear':0, 'Snowing':2,

       'Fog/Smog/Smoke':2, 'Sleet/Hail/Freezing Rain':2, 'Blowing Sand/Dirt':2,

       'Severe Crosswind':2}}, inplace = True)

collisions['WEATHER'].head()
collisions.replace({'WEATHER' : {'Partly Cloudy':1, 'Unknown' :3, 'Other' :3}}, inplace = True)



collisions['WEATHER'].fillna(3, inplace = True)

print(collisions['WEATHER'].unique())

collisions['WEATHER'].head()
collisions.head()
import sklearn.model_selection as model_selection

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier as DTC

from sklearn import metrics



feature_columns = ['SPEEDING','LIGHTCOND','WEATHER']

X = collisions[feature_columns]

y = collisions['SEVERITYCODE']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

print("Completed")
tree = DTC(random_state = 1)

tree = tree.fit(X_train,y_train)

y_pred = tree.predict(X_test)

print("Completed")
from sklearn.metrics import confusion_matrix as Conf_Mat

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print('\n')

Conf_Mat(y_test, y_pred)