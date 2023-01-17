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
#import library yang dibutuhkan
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#impor dataset
dataset = pd.read_csv('/kaggle/input/salary-data-simple-linear-regression/Salary_Data.csv')
X= dataset.iloc[:, :-1].values
y= dataset.iloc[:, 1].values
#membagi data menjadi training set dan test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size =1/3,random_state=0)
#Fitting Simple Linear regression terhadap trainning set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train , y_train)
#memprediksi hasil test set
y_pred = regressor.predict(X_test)
#Visualisasi hasil training set
plt.scatter(X_train, y_train , color ='red')
plt.plot(X_train, regressor.predict(X_train),color='blue')
plt.title ('Gaji vs Pengalaman (Training set)')
plt.xlabel('Tahun bekerja')
plt.ylabel('Gaji')
plt.show
#visualisasi hasil test set
plt.scatter(X_test, y_test, color ='red')
plt.plot(X_train , regressor.predict(X_train),color='blue')
plt.title('Gaji vs Pengalaman (Test set)')
plt.xlabel('Tahun Bekerja ')
plt.ylabel('Gaji')
plt.show