# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import Imputer

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report,confusion_matrix

import matplotlib.pyplot as plt

%matplotlib inline
dataset ="../input/data.csv"
df =pd.read_csv(dataset)

#getting all the feature colums in x

x =df.drop(["id","diagnosis"],axis=1)

#replacing missing values with mean



#col = x.columns

imputer = Imputer(missing_values = 'NaN',strategy= 'mean' , axis = 0)

imputer = imputer.fit(x)

x = imputer.transform(x)



#gettng the result column in y

y =df["diagnosis"]

#converting categorical data into indicator/dummy variable

y = pd.get_dummies(df["diagnosis"],drop_first=True)

#print(df.head())

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
rf_model = RandomForestClassifier(random_state=1)

rf_model.fit(x_train,y_train)
pre = rf_model.predict(x_test)

print (rf_model.score(x_test,y_test))

print (classification_report(y_test,pre))

print (confusion_matrix(y_test,pre))