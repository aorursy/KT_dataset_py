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
from sklearn.linear_model import LinearRegression
raw_data = pd.read_csv("../input/insurance/insurance.csv")
raw_data.shape
raw_data.describe()
raw_data.info()
raw_data.isnull().sum()
raw_data.head()
data = pd.get_dummies(raw_data,columns=['sex','smoker','region'],drop_first=True)
data.head()
data.hist('charges')
data.charges = np.log(data.charges)
data.hist('charges');
data.corr()
import seaborn as sns
# correlation plot
corr = data.corr()
sns.heatmap(corr, cmap = 'Wistia', annot= True);
X = data.copy()
del X['charges']
X.shape
#X = X.drop(['region_northwest','region_southeast','region_southwest'],axis=1)
#X.shape
y=data['charges']
y.shape
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
X_train.shape,X_test.shape,y_train.shape,y_test.shape
model = LinearRegression()
model.fit(X_train,y_train)
model.score(X_train,y_train)
pred = model.predict(X_test)
sse=np.sum((y_test-pred)**2)
sse
sst = np.sum((pred-np.mean(y_train))**2)
sst
r2=1-(sse/sst)
r2
