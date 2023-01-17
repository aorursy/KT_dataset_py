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
#Loading the Dataset from Sklearn

from sklearn.datasets import load_breast_cancer
#loading the Breast cancer data in a variable

cancer=load_breast_cancer()
#printing the Data in Dictionary format

cancer
#Printing the keys of the Dictionary, to get enough details about the data.

cancer.keys()
#Will basically give you broad description of Breast Cancer Dataset 

print(cancer['DESCR'])
#Printing the Target column, which is either 0=Malignenet or 1=Benign

print(cancer['target'])
#Printing the Target names

print(cancer['target_names'])
#All the column names which are also known as features

print(cancer['feature_names'])
#Shape of the Data which is 569 Rows and 30 Columns

cancer['data'].shape
#Creating a Pandas DataFrame to deal with the data in a much easier way.

#While creating this DataFrame I added the additional column which is 'target' to the df_cancer using np.append().

df_cancer=pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'], ['target']))
#to print the first 5 rows of the data, also lookaround for the target column which I've added. 

df_cancer.head()
#to simply print the last 5 rows. so we can check the number of records

df_cancer.tail()
#importing the libraries to plot the Data

import matplotlib.pyplot as plt

import seaborn as sns
#taking only 5 variable out of 30 just to showcase how powerfull seaborn library actually is

sns.pairplot(df_cancer, vars = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness'])
#in above plotting we are not able to differentiate much,so we use 'hue' on target column, which will seperate the two(Malignent, Benign).

sns.pairplot(df_cancer, hue = 'target', vars = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness'])



#blue points are malignent case which are severe cases or life threatning cases.

#orange points are not very severe or life threatning
#will simply tell you how many Malignent and Benign cases we have.

#Malignent= 200~ and Benign = 350~ approx.

sns.countplot(df_cancer['target'])
#plotting a scatter plot diagram for mean area anf mean smoothness, you can plot any feature combination scatterplot.

sns.scatterplot(x = 'mean area', y = 'mean smoothness', hue = 'target', data = df_cancer)
#here we made a heatmap figure of correlation of all the columns

# 'figsize()'' is simply the size of the figure

plt.figure(figsize = (20,10))

sns.heatmap(df_cancer.corr(), annot = True)
#train test split

#here we split our data in train and test, simply we split our data like X = feature_names and y = target

#in 'X' we dropped the 'target' column

X= df_cancer.drop(['target'], axis = 1)
#printing just to verify is target column is dropped

X.head()
#here in 'y' we are only taking the 'target' column

y = df_cancer['target']

y.head()
#from sklearn library we import train_test_split which will split the data in whatever manner we want.

#'test_size' is what is the size of the test data whicg is 15% of the whole data. we had 569 rows which will get split by train = 483 and test = 86. 

#look below for better understanding, we've printed all the 4 values for X_train, X_text, y_train, y_test.

#also make sure the X is capital and y is small..

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 5)
X_train
X_test
y_train
y_test
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train_sc = sc.fit_transform(X_train)

X_test_sc = sc.transform(X_test)
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier

DT_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 19)

DT_classifier.fit(X_train, y_train)

y_pred_DT = DT_classifier.predict(X_test)

accuracy_score(y_test, y_pred_DT)
from sklearn.neighbors import KNeighborsClassifier

knn_classifier = KNeighborsClassifier(n_neighbors = 7, metric = 'minkowski', p = 2)

knn_classifier.fit(X_train, y_train)

y_pred_knn = knn_classifier.predict(X_test)

accuracy_score(y_test, y_pred_knn)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred_DT)

plt.title('Heatmap of Confusion Matrix', fontsize = 10)

sns.heatmap(cm, annot = True)

plt.show()
cm = confusion_matrix(y_test,y_pred_knn)

plt.title('Heatmap of Confusion Matrix', fontsize = 10)

sns.heatmap(cm, annot = True)

plt.show()