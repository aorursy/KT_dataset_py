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
df1 = pd.read_table("/kaggle/input/gas-sensors-for-home-activity-monitoring-data-set/HT_Sensor_dataset/HT_Sensor_dataset.dat")
df1.head()
df1['id time']

x = []
for i in df1['id time']:
    x.append(i.split())
df2 = pd.DataFrame(x,columns=['id','time', 'R1' ,'R2','R3','R4','R5','R6','R7','R8','Temp' ,'Humidity'])
df2.head()
df2.info()
df2['id'] = df2['id'].astype(int)
df2['time'] = df2['time'].astype(float)
df2['R1'] = df2['R1'].astype(float)
df2['R2'] = df2['R2'].astype(float)
df2['R3'] = df2['R3'].astype(float)
df2['R4'] = df2['R4'].astype(float)
df2['R5'] = df2['R5'].astype(float)
df2['R6'] = df2['R6'].astype(float)
df2['R7'] = df2['R7'].astype(float)
df2['R8'] = df2['R8'].astype(float)
df2['Temp'] = df2['Temp'].astype(float)
df2['Humidity'] = df2['Humidity'].astype(float)
df2.info()
from tsfresh.feature_extraction import extract_features
from tsfresh.feature_extraction import settings

import numpy as np
import pandas as pd
settings.ComprehensiveFCParameters, settings.EfficientFCParameters, settings.MinimalFCParameters
settings_minimal = settings.MinimalFCParameters() 
settings_minimal
X_tsfresh = extract_features(df2, column_id="id", default_fc_parameters=settings_minimal)
X_tsfresh.head()
Meta_data = pd.read_table("/kaggle/input/gas-sensors-for-home-activity-monitoring-data-set/HT_Sensor_metadata.dat")
Meta_data.head()
Meta_data.columns 
categories = []
for filename in Meta_data["Unnamed: 2"]:
    if filename == 'banana':
        categories.append(1)
    elif filename == 'wine':
         categories.append(2)
        
    else:
        categories.append(0)

categories = pd.DataFrame(categories)
categories.columns = ["Target"]
Meta_data = Meta_data.drop(["id","date","dt","Unnamed: 2"],axis=1)
data = pd.concat([X_tsfresh,Meta_data,categories],axis=1)
data.head()
data = data.dropna()
from sklearn.model_selection import train_test_split
X = data.drop('Target',1)
y = data.Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=30)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
classifier = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))