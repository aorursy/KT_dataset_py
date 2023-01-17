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
train=pd.read_csv('/kaggle/input/mobile-price-range-prediction-is2020-v2/train_data.csv')
test=pd.read_csv('/kaggle/input/mobile-price-range-prediction-is2020-v2/test_data.csv')
price_sub=pd.read_csv('/kaggle/input/mobile-price-range-prediction-is2020-v2/sample_submission.csv')
train.head()
test.head()
train.info()
test.info()
test.isnull().sum()
train.describe()
import seaborn as sns
sns.pointplot(y="talk_time",x="price_range",data=train)
sns.pointplot(y="battery_power",x="price_range",data=train)
sns.pointplot(y="ram",x="price_range",data=train)
sns.pointplot(y="int_memory",x="price_range",data=train)
sns.pointplot(y="mobile_wt",x="price_range",data=train)
sns.pointplot(y="dual_sim",x="price_range",data=train)
sns.pointplot(y="four_g",x="price_range",data=train)
sns.pointplot(y="three_g",x="price_range",data=train)
train_clean=train.drop(columns=['id'])
x_train=train_clean.drop(columns=['price_range'])
y_train=train_clean['price_range']
data_test=test.drop(columns=['id'])
data_test.head()
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
train['fc'].hist(alpha=0.5,color='blue',label='Front camera')

plt.legend()
plt.xlabel('MegaPixels')
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))

train['pc'].hist(alpha=0.5,color='red',label='Primary camera')
plt.legend()
plt.xlabel('MegaPixels')
plt.figure(figsize=(10,6))
train['fc'].hist(alpha=0.5,color='blue',label='Front camera')
train['pc'].hist(alpha=0.5,color='red',label='Primary camera')
plt.legend()
plt.xlabel('MegaPixels')


sns.pointplot(y="dual_sim",x="price_range",data=train)
sns.pointplot(y="four_g",x="price_range",data=train)
data_test.head()
from sklearn.preprocessing import StandardScaler
x_train_scales=StandardScaler().fit_transform(x_train)
pd.DataFrame(x_train_scales).head()
x_train.head()
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
knn = KNeighborsClassifier(n_neighbors=10).fit(x_train,y_train)
scores=cross_val_score(knn,x_train_scales,y_train,cv=5)
print(scores)
print(scores.mean())
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
dtc = DecisionTreeClassifier().fit(x_train,y_train)
scores=cross_val_score(dtc,x_train_scales,y_train,cv=5)
print(scores)
print(scores.mean())
price_pred_dtc=dtc.predict(data_test)
print(price_pred_dtc)
price_pred_knn=knn.predict(data_test)
print(price_pred_knn)
data={'id':price_sub['id'],'price_range':price_pred_knn}
result_knn=pd.DataFrame(data)
result_knn.to_csv('/kaggle/working/result_knn.csv',index=False)

data={'id':price_sub['id'],'price_range':price_pred_dtc}
result_dtc=pd.DataFrame(data)
result_dtc.to_csv('/kaggle/working/result_dtc.csv',index=False)
