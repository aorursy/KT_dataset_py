# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df1 = pd.read_csv("/kaggle/input/pima-indians-diabetes-database/diabetes.csv")
df1.head(5)

df1.describe()
df1.info()
df1.head()
df1.DiabetesPedigreeFunction =df1.DiabetesPedigreeFunction.astype('int64')
df1.DiabetesPedigreeFunction
df1.BMI = df1.BMI.astype('int64')
df1.info()
df1.corr(method='pearson')
X = ["Pregnancies","Glucose" ,"BloodPressure", "SkinThickness", "Insulin","BMI","DiabetesPedigreeFunction","Age"]
y = "Outcome"
from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=0)