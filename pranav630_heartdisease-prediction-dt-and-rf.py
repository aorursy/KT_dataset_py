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
import matplotlib.pyplot as plt

import seaborn as sns
heart = pd.read_csv("../input/heart-disease-dataset/heart.csv")
heart.head()

print (heart.shape)
heart.describe()
print (heart.columns)
Y = heart['target']

feature_n = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','ca','thal']



X = heart[feature_n]
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import train_test_split



X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.25, random_state = 7)

heart_model_DT = DecisionTreeRegressor(random_state = 7)

heart_model_DT.fit(X_train, Y_train)



heart_pred_DT = heart_model_DT.predict(X_test)

accuracy_DT = np.mean(heart_pred_DT == Y_test)

print ("Model accuracy from DT algo is [{:.2f}]".format(accuracy_DT*100))
from sklearn.ensemble import RandomForestClassifier

heart_model_RF = RandomForestClassifier(random_state = 7)

heart_model_RF.fit(X_train, Y_train)



heart_pred_RF = heart_model_RF.predict(X_test)

accuracy_RF = np.mean(heart_pred_RF == Y_test)

print ("Model accuracy from RF algo is [{:.2f}]".format(accuracy_RF*100))