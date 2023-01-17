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
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv('/kaggle/input/flight-delays/flights.csv', low_memory = False)
dataset = dataset.iloc[0:100000,:]
dataset.shape
dataset.info()
dataset.DIVERTED.value_counts()
import seaborn as sns
plt.figure(figsize = (10,10))
sns.heatmap(dataset.corr(), vmin = -1, vmax = 1, linewidth = 1,cmap = "coolwarm", linecolor = "black")
dataset.drop(['YEAR','FLIGHT_NUMBER','AIRLINE','DISTANCE','TAIL_NUMBER','TAXI_OUT','SCHEDULED_TIME','DEPARTURE_TIME','WHEELS_OFF',
             'ELAPSED_TIME','AIR_TIME','WHEELS_ON','DAY_OF_WEEK','TAXI_IN','ARRIVAL_TIME','CANCELLATION_REASON'], axis = 1, inplace = True)
dataset.isna().sum()
dataset.fillna(dataset.mean(), inplace = True)
dataset.isna().sum()
def is_delayed(x):
    if x > 15: return 1
    return 0
dataset['RESULT'] = dataset['ARRIVAL_DELAY'].apply(is_delayed)
dataset['RESULT'].value_counts()
dataset.drop(['ORIGIN_AIRPORT','DESTINATION_AIRPORT','ARRIVAL_DELAY'], axis = 1, inplace = True)
from sklearn import model_selection, linear_model
x = dataset.drop(['RESULT'], axis = 1)
y = dataset['RESULT']
x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y,test_size = 0.3, random_state = 42)
from sklearn.preprocessing import StandardScaler
scaling = StandardScaler()
x_train = scaling.fit_transform(x_train)
x_test = scaling.transform(x_test)
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(x_train,y_train)
from sklearn.metrics import roc_auc_score
y_pred = clf.predict(x_test)
roc_auc_score(y_pred,y_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_pred, y_test)
