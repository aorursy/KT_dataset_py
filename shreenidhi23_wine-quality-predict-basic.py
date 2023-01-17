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
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
data.head()
data.info()
data.isnull().sum()
data.describe()
data.corr()
plt.figure(figsize=(12,12))
sns.heatmap(data=data.corr(),annot=True)
plt.show()
for i in data.columns:
    if i =="quality":
        break
    sns.boxplot("quality",i,data=data)
    plt.show()
data.corr()['quality']
data['quality'].value_counts()
bins_ = (2,6,8)
labels_ = ['bad','good']
data['quality']=pd.cut(data['quality'],bins=bins_,labels=labels_)
print(data['quality'])
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
data['quality'] = labelencoder_y.fit_transform(data['quality'])
data['quality']
from sklearn.model_selection import train_test_split, cross_val_score
#Now separate the dataset as response variable and feature variabes
X = data.drop('quality', axis = 1)
Y = data['quality']
#Train and Test splitting of data 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.fit_transform(X_test)

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
%matplotlib inline

classifier_knn = KNeighborsClassifier(weights = 'distance')
classifier_knn.fit(X_train_scaled, Y_train)
Y_pred_knn = classifier_knn.predict(X_test_scaled)
#Let's see how our model performed
print(classification_report(Y_test, Y_pred_knn))
print ("KNN ACCURACY : ",accuracy_score(Y_test, Y_pred_knn))
# Fitting classifier to the Training set

classifier_svm_linear = SVC()
classifier_svm_linear.fit(X_train_scaled, Y_train)
Y_pred_svm = classifier_svm_linear.predict(X_test_scaled)
#Let's see how our model performed
print(classification_report(Y_test, Y_pred_svm))
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train_scaled, Y_train)
Y_pred_rfc = rfc.predict(X_test_scaled)
#Let's see how our model performed
print(classification_report(Y_test, Y_pred_rfc))
print("KNN ACCURACY : ",accuracy_score(Y_test, Y_pred_knn))
print("SVM : ",accuracy_score(Y_test, Y_pred_svm))
print("RANDOM FOREST : ",accuracy_score(Y_test, Y_pred_rfc))
