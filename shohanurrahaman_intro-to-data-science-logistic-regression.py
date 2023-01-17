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
# we are going to predict diabetes
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')
display(df.head())

#selecting features and target variable
feature_cols = ['Pregnancies', 'Insulin', 'BMI', 'Age','Glucose','BloodPressure','DiabetesPedigreeFunction']
X = df[feature_cols] # Features
y = df['Outcome'] # Target variable
#print(y)
#spliting dataset into train and test 
X_train, x_test, Y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
logreg= LogisticRegression()

logreg.fit(X_train, Y_train)
y_pred = logreg.predict(x_test)

print(y_pred)