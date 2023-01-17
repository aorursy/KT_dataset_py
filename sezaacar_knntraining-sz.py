# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import warnings
warnings.filterwarnings('ignore') 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv')
data3 = pd.read_csv('/kaggle/input/biomechanical-features-of-orthopedic-patients/column_3C_weka.csv')
data.head()
data.describe()
type(data)
data.isnull().sum()
def discrete_univariate(dataset, discrete_feature):
    fig, axarr=plt.subplots(nrows=1,ncols=2, figsize=(8,5))
      
    dataset[discrete_feature].value_counts().plot(kind="bar",ax=axarr[0])
    dataset[discrete_feature].value_counts().plot.pie(autopct="%1.1f%%",ax=axarr[1])
        
    plt.tight_layout()
    plt.show()
discrete_univariate(dataset=data , discrete_feature="class")
discrete_univariate(dataset=data3, discrete_feature="class")
sns.pairplot(data ,hue ="class",palette="husl")
plt.show()
#%%  Normal =1  Abnormal =0
data['class'] = [1 if each == "Normal" else 0 for each in data['class']]


y = data.loc[:,'class']

x1 = data.loc[:,data.columns != 'class']
x = (x1 - np.min(x1))/(np.max(x1)-np.min(x1))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2 ,random_state=1)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)

X_train = scaler.transform(x_train)
X_test = scaler.transform(x_test)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 23) 
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)
print(" {} nn score: {} ".format(23,knn.score(x_test,y_test)))

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, prediction))
print(classification_report(y_test, prediction))
error=[]
for i in range(1,40):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    pred_i = knn.predict(x_test)
    error.append(np.mean(pred_i != y_test)) 
    
    
    
plt.figure(figsize=(20, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()