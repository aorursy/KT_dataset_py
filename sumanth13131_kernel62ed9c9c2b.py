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
import matplotlib.pyplot as plt
%matplotlib inline
train_data=pd.read_csv('/kaggle/input/mobile-price-range-prediction-is2020/train_data.csv')
test_data=pd.read_csv('/kaggle/input/mobile-price-range-prediction-is2020/test_data.csv')
train_data.head()
train_data = train_data.drop(['id'],axis=1)
train_data.head()
test_id=test_data.iloc[:,0]
test_data = test_data.drop(['id'],axis=1)
test_data.head()
test_id.head()
col_train=set(train_data.columns)
col_test=set(test_data.columns)
col_train-col_test
train_data.describe()
train_data.corr()
import seaborn as sns
plt.figure(figsize=(20,7))
sns.heatmap(train_data.corr())
plt.figure(figsize=(20,10))
train_data.boxplot(column=['ram'])
min(train_data['ram'])
from sklearn.model_selection import train_test_split,cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
x_train,x_test,y_train,y_test=train_test_split(train_data.drop(columns=['price_range']),train_data['price_range'],random_state=101)
x_train.shape,y_train.shape
x_test.shape,y_test.shape
from sklearn.metrics import mean_squared_error as msqe
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
n=[]
mse=[]
scores=[]
for i in range(2,50):
    n.append(i)
    classifier=KNeighborsClassifier(n_neighbors=i)
    m=classifier.fit(x_train,y_train)
    y_pred=cross_val_predict(m,x_test,y_test,cv=10)
    scores.append(m.score(x_test,y_test))
    mse.append(msqe(y_test,y_pred))
y_pred,y_test
plt.plot(n,mse)
plt.plot(n,scores)
max(scores),'--->',score.index(max(scores))

mse.index(min(mse))
max(score),score.index(max(score))
classifier=KNeighborsClassifier(n_neighbors=n[6])
classifier.fit(x_train,y_train)
test_pred=classifier.predict(test_data)
test_pred=pd.DataFrame({'price_range': test_pred})
ids=pd.DataFrame({'id':test_id})
result=pd.concat([ids,test_pred],axis=1)
result.to_csv('result_mse4.csv',index=False)
classifier=KNeighborsClassifier(n_neighbors=n[7])
classifier.fit(x_train,y_train)
test_pred=classifier.predict(test_data)
test_pred=pd.DataFrame({'price_range': test_pred})
ids=pd.DataFrame({'id':test_id})
result=pd.concat([ids,test_pred],axis=1)
result.to_csv('result_scores4.csv',index=False)
