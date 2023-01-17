import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
from scipy import stats

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report,confusion_matrix
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
data = pd.read_csv('../input/iris-classifier-with-knn/Iris.csv')

data.shape
data.columns
data.head()
data.tail()
data.describe()
data.isnull().sum()
sns.boxplot(y=data['SepalLengthCm'])
sns.boxplot(y=data['SepalWidthCm'])
sns.boxplot(y=data['PetalLengthCm'])
sns.boxplot(y=data['PetalWidthCm'])
data['Species'].value_counts()
data.shape
features = data.drop('Species', axis=1)

target = data['Species']
z = np.abs(stats.zscore(features))

print(z)
threshold = 3

print(np.where(z>threshold))
data_new = features[z>threshold]

print(data_new)
sns.pairplot(data, hue='Species')
scale = StandardScaler()
scale.fit(features)
scaled_features=scale.transform(features)
data_new = pd.DataFrame(scaled_features)

data_new.head(3)
x_train, x_test, y_train, y_test = train_test_split(data_new, target, test_size=0.25, random_state=45)
x_train.shape
x_train.head()
x_test.head()
y_train.head()
y_test.head()
model = KNeighborsClassifier(n_neighbors=1)
model.fit(x_train, y_train)
pred = model.predict(x_test)
pred
confusion_matrix(y_test, pred)
print(classification_report(y_test, pred))
error_rate = []

for i in range(1,40):

    model = KNeighborsClassifier(n_neighbors=i)

    model.fit(x_train,y_train)

    pred_i = model.predict(x_test)

    error_rate.append(np.mean(pred_i!=y_test))
plt.figure(figsize=(15,6))

plt.plot(range(1,40),error_rate, color='red',linestyle='dashed', marker='o',markerfacecolor='blue', markersize=8)

plt.title("Elbow Graph")

plt.xlabel("K-Value")

plt.ylabel("Error Rate")
model = KNeighborsClassifier(n_neighbors=21)



model.fit(x_train,y_train)

pred = model.predict(x_test)



print('WITH K=21')

print('\n')

print(confusion_matrix(y_test,pred))

print('\n')

print(classification_report(y_test,pred))