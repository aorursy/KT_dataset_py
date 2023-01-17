# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import numpy as np
import  pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore')
data = pd.read_csv('../input/train_revenue.csv')
data.head()
data.shape
data.isnull().sum()
data.info()
data = pd.get_dummies(data,columns= ['City Group', 'Type',],drop_first=True)
data.head()
from sklearn.preprocessing import LabelEncoder
encode = LabelEncoder()
data.City = encode.fit_transform(data.City)
data.head()
from datetime import datetime
data['Open Date']=pd.to_datetime(data['Open Date'])
data.info()
data.describe()
data.drop("Open Date",axis=1,inplace=True)
data.info()
f,ax = plt.subplots(figsize=(20,20))
sns.heatmap(data.corr(), annot = True,linewidths=.2, fmt='.1f', ax=ax)
X = data.loc[:,data.columns!="revenue"]
y = data.revenue
rcParams['figure.figsize']=10,5
color = np.array(['blue','red'])
fig = plt.figure()
ax = fig.add_axes([.1,.1,1,1])
ax.set_xlabel('City')
ax.set_ylabel('revenue')
plt.scatter(x=data.City,y=data.revenue,)
rcParams['figure.figsize']=50,50
data.revenue.plot(kind='barh')


X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=10)

model = LinearRegression()
model.fit(X_train,y_train)
y_predict = model.predict(X_test)
r2_score(y_test,y_predict)
y_test
y_predict

