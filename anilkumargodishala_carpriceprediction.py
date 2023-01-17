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
dataset =pd.read_csv('../input/cardata/car_data.csv')
dataset.head()
dataset.head(20)
dataset.info()
dataset.describe()
dataset['transmission'].unique()
dataset['transmission'].value_counts()
x=dataset.iloc[:,[1,3,4,6]].values
y=dataset.iloc[:,2].values
x
y
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
le1=LabelEncoder()
x[:,2]=le.fit_transform(x[:,2])
x[:,2]
#le1=LabelEncoder()
x[:,3]=le1.fit_transform(x[:,3])
le1
x[:,3]
x.shape
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.2,random_state=0)
x_train[0,:]
from sklearn.ensemble import RandomForestRegressor
rfr =RandomForestRegressor(n_estimators=300,random_state=0)
rfr.fit(x_train,y_train)
#training accuracy
accuracy_train=rfr.score(x_train,y_train)
accuracy_train
accuracy=rfr.score(x_test,y_test)
accuracy
accuracy*100
dataset['fuel'].value_counts()
new_data=[2017,7000,"Petrol","Manual"]
new_data[2]=le.transform([new_data[2]])[0]
new_data[3]=le1.transform([new_data[3]])[0]

new_data
rfr.predict([new_data])
import pickle
pickle.dump(rfr,open('anilcarprediction.pkl','wb'))
pickle.dump(le,open('le','wb'))
pickle.dump(le1,open('le1','wb'))
