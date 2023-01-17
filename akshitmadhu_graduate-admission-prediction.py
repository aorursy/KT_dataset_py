# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # for visualisation



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Loading Dataset

df = pd.read_csv('../input/graduate-admissions/Admission_Predict_Ver1.1.csv')
df.head()
df = df.set_index('Serial No.')
df.head()
# Checking if number of null values in each column

df.isnull().sum()
# Shape of Data :- (Rows,columns)

df.shape
# Datatype of each column

df.dtypes
# Spliting dataframe into input features and target variable

x = df.iloc[:,:-1]

y = df.iloc[:,-1]
x.shape
y.shape
# Splitting Data for training and testing

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20)
# Tranforming data for better accuracy

from sklearn.preprocessing import MinMaxScaler

ms = MinMaxScaler()

x_train = ms.fit_transform(x_train)

x_test = ms.fit_transform(x_test)
y_train=[1 if chance > 0.50 else 0 for chance in y_train]

y_train=np.array(y_train)



y_test=[1 if chance > 0.50 else 0 for chance in y_test]

y_test=np.array(y_test)
# Training of data using machine learning algorithm

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(x_train, y_train)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,lr.predict(x_test))
print("Accuracy Score of model is {}".format(accuracy))
y_predict = lr.predict(x_test)

from sklearn.metrics import confusion_matrix

import seaborn as sns
lr_confm = confusion_matrix(y_test, y_predict)

sns.heatmap(lr_confm, annot=True, fmt='.2f',xticklabels = ["Admitted", "Rejected"] , yticklabels = ["Admitted", "Rejected"] )

plt.ylabel('Actual Class')

plt.xlabel('Predicted Class')

plt.title('Logistic Regression')

plt.show()