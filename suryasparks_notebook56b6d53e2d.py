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
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

dataset =pd.read_csv('../input/house-price-prediction-challenge/train.csv')
dataset=dataset.drop(columns='ADDRESS')

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
x=np.array(x)
y=np.array(y)



from sklearn.preprocessing import LabelEncoder
le =LabelEncoder()
x[:,0]=le.fit_transform(x[:,0])
x[:,4]=le.fit_transform(x[:,4])


from sklearn.ensemble import RandomForestRegressor
regressor =RandomForestRegressor(n_estimators =10,random_state=0)
regressor.fit(x,y)

test_set =pd.read_csv('../input/house-price-prediction-challenge/test.csv')
test_set =test_set.drop(columns='ADDRESS')
w=test_set.iloc[:,:].values
w=np.array(w)
from sklearn.preprocessing import LabelEncoder
le =LabelEncoder()
w[:,0]=le.fit_transform(w[:,0])
w[:,4]=le.fit_transform(w[:,4])
print(regressor.predict(w))






