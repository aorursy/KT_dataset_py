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

print(os.listdir("../input"))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np 

import warnings

# data processing 

import pandas as pd 



# visualisation 

import matplotlib.pyplot as plt 
data = pd.read_csv("/kaggle/input/new-data-1/data_2.csv") 



#print (data.head)
data.head()
#data=data.drop(['diagnosis','id'], axis = 1)



data=data.drop(['diagnosis','id','radius_mean','texture_mean','perimeter_mean','area_mean','area_se','area_worst'], axis = 1)



#data=data.drop(['diagnosis','id','concavity_mean','area_se','texture_se','fractal_dimension_mean','concave points_worst','perimeter_se','radius_se','fractal_dimension_worst','symmetry_se','concavity_se','compactness_se','smoothness_worst','concave points_mean','concave points_se','compactness_mean','fractal_dimension_se','smoothness_mean','texture_worst','smoothness_se'], axis = 1)





#'concavity_worst','area_mean',,'compactness_worst','area_se','texture_se'

#data=data.drop(['diagnosis','id','concavity_mean','symmetry_mean','radius_se','compactness_se','compactness_worst','radius_worst','texture_worst','perimeter_worst','concavity_worst'], axis = 1)

y = data.new.values 

x_data = data.drop(['new'], axis = 1) 
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(x_data, y, test_size = 0.3, random_state = 101)
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)



warnings.filterwarnings('ignore')
predictions = logreg.predict(X_test)
from sklearn.metrics import classification_report



print(classification_report(y_test, predictions))