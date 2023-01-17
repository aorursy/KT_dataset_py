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
import warnings
warnings.filterwarnings("ignore")
train = pd.read_csv('/kaggle/input/mobile-price-range-prediction-is2020-v2/train_data.csv')
test = pd.read_csv('/kaggle/input/mobile-price-range-prediction-is2020-v2/test_data.csv')
sample_submission=pd.read_csv('/kaggle/input/mobile-price-range-prediction-is2020-v2/sample_submission.csv')
train.head()
test.head()
print(train.shape,test.shape)
train.info()
test.info()
train.isnull().sum()
test.isnull().sum()
train['price_range'].unique
import matplotlib.pyplot as plt
import seaborn as sb
plt.hist(train['battery_power'])
plt.show()
sb.countplot(train['price_range'])
plt.show()
sb.boxplot(train['price_range'],train['clock_speed'])
sb.countplot(train['dual_sim'])
plt.show()
plt.hist(train['fc'])
plt.show()

sb.boxplot(train['four_g'],train['price_range'])
plt.hist(train['int_memory'])
plt.show()
plt.scatter(train['price_range'],train['int_memory'])
plt.show()
train['n_cores'].unique()
sb.boxplot(train['n_cores'],train['price_range'])
plt.show()
train. loc[(train['price_range']==0)&(train['n_cores']==8)]['n_cores'].count()
corr = train. corr()
sb.heatmap(corr,cmap='YlGnBu',vmin=-1,vmax = 1)

test.columns
x_train = train.drop(columns=['price_range','id'])
y_train = train['price_range']
x_test = test.drop(columns=['id'])
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

scaler = StandardScaler()
X = train.drop('price_range',axis=1)
y = train['price_range']

scaler.fit(X)
X_transformed = scaler.transform(X)

X_train,X_test,y_train,y_test = train_test_split(X_transformed,y,test_size=0.3)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score as score
model = LogisticRegression()
model.fit(X_train,y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
print("Train data Accuracy:"+str(score(y_train_pred,y_train)*100))
print("Test data Accuracy:"+str(score(y_test_pred,y_test)*100))
scaler = StandardScaler()
X = train[['battery_power','bluetooth','dual_sim','four_g','px_height','px_width','ram','touch_screen','wifi','fc']]
y = train['price_range']

scaler.fit(X)
X_transform = scaler.transform(X)

X_train,X_test,y_train,y_test = train_test_split(X_transform,y,test_size=0.3)
from sklearn.metrics import confusion_matrix as cm
model = LogisticRegression()
model.fit(X_train,y_train)
y_train_predict = model.predict(X_train)
y_test_predict = model.predict(X_test)
print("Train data Accuracy="+str(score(y_train_predict,y_train)*100))
print("Test data Accuracy="+str(score(y_test_predict,y_test)*100))
print("\nConfusion Matrix=\n%s"%cm(y_test_predict,y_test))
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier()
model.fit(X_train,y_train)
y_train_predict = model.predict(X_train)
y_test_predict = model.predict(X_test)
print("Train Data Accuracy="+str(score(y_train_predict,y_train)*100))
print("Test Data Accuracy="+str(score(y_test_predict,y_test)*100))
print("\nConfusion Matrix=\n%s"%cm(y_test_predict,y_test))
from sklearn.svm import SVC

model = SVC()
model.fit(X_train,y_train)
y_train_predict = model.predict(X_train)
y_test_predict = model.predict(X_test)


print("Train data Accuracy="+str(score(y_train_predict,y_train)*100))
print("Test data Accuracy="+str(score(y_test_predict,y_test)*100))
print("\nConfusion Matrix=\n%s"%cm(y_test_predict,y_test))
from sklearn.tree import DecisionTreeClassifier 
model = DecisionTreeClassifier()
model.fit(X_train,y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)


print("Train data Accuracy="+str(score(y_train_predict,y_train)*100))
print("Test data Accuracy="+str(score(y_test_predict,y_test)*100))
print("\nConfusion Matrix=\n%s"%cm(y_test_predict,y_test))
data={'id':sample_submission['id'],'price_range':sample_submission['price_range']}
r=pd.DataFrame(data)
r.to_csv('/kaggle/working/result_svc.csv',index=False)
predicted_data=pd.read_csv('/kaggle/working/result_svc.csv')
print(predicted_data) 