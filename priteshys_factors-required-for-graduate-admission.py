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

import matplotlib.pyplot as plt

import pandas as pd



df = pd.read_csv('../input/Admission_Predict_Ver1.1.csv')

X = df.iloc[:, 1:2].values

y = df.iloc[:, 2].values



print('check sample \n',df.sample(2))

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)

plt.hist(df['GRE Score'])

plt.hist(df['TOEFL Score'])

plt.hist(df['CGPA'])

plt.hist(df['SOP'])



#checking out coorealtion 

df.corr()

# Fitting Naive Bayes to the Training set

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(X_train, y_train)



df_sort = df.sort_values(by=df.columns[-1],ascending=False)

df_sort.head(5)


