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
# Loading the dataset

df = pd.read_csv('/kaggle/input/students-performance-in-exams/StudentsPerformance.csv')

print(df)
# Finding the null values

df.isnull().sum()
# Splitting the input variable from dataset

x = df.drop(columns=['writing score'])

print(x)
# Splitting the output variable from dataset

y = df['writing score']

y
# Label Encoding

from sklearn import preprocessing 

label_encoder = preprocessing.LabelEncoder()  

x= x.apply(label_encoder.fit_transform)

print(x)
# Splitting the input and target variables

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)
# Training the ML model

from sklearn.linear_model import ElasticNetCV

reg = ElasticNetCV(cv=5, random_state=0).fit(x_train, y_train)

reg.score(x_train, y_train)