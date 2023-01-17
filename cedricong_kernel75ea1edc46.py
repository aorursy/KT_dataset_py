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
data_1 = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict.csv', encoding="utf-8-sig")

data_2 = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv')
import seaborn as sns

import matplotlib.pyplot as plt
data_1.info()
data_1.drop(['Serial No.'], axis=1)
sns.set()

sns.pairplot(data=data_1)
sns.boxplot(x = 'University Rating', y = data_1.iloc[:,-1], data=data_1)
sns.boxplot(x = 'Research', y = data_1.iloc[:,-1], data=data_1)
sns.boxplot(x = 'SOP', y = data_1.iloc[:,-1], data=data_1)
sns.boxplot(x = data_1.iloc[:,5], y = data_1.iloc[:,-1], data=data_1)
scores = data_1.iloc[:,[1, 2, 6, -1]]

sns.heatmap(scores.corr(), annot=True)
scores = data_1

sns.heatmap(scores.corr(), annot=True)
from sklearn.linear_model import LinearRegression

import sklearn

from sklearn.metrics import mean_squared_error



model = LinearRegression()

X = data_1.iloc[:,1:8]

y = data_1.iloc[:, -1]



x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.05)

model.fit(x_train, y_train)





data_2_x = data_2.iloc[399:,1:8]

predictions = model.predict(data_2_x)

print(np.sqrt(mean_squared_error(data_2.iloc[399:,-1], predictions)))