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
import pandas as pd

import numpy as np



df = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')

print(df.columns)



# Separating the data into the input and output variables

X = df.iloc[:, [7]].values.reshape(-1, 1)

Y = df.iloc[:, [12]].values.reshape(-1, 1)



# Splitting the datatset into training and testing data

from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=0)



# fitting the model

from sklearn.linear_model import LinearRegression



model = LinearRegression()

model.fit(x_train, y_train)



y_pred = model.predict(x_test)



opt = pd.DataFrame({'Actual': y_test.flatten(), 'Predictetd': y_pred.flatten()})

print(opt)

from sklearn.metrics import r2_score

print('\n\nR squared score : ', r2_score(y_pred, y_test))



# visualizing the model

import matplotlib.pyplot as plt



# training dataset

plt.scatter(x_train, y_train, color='b')

plt.plot(x_train, model.predict(x_train), color='k')

plt.xlabel('degree_p')

plt.ylabel('mba_p')

plt.title('Training dataset')

plt.show()



# test_dataset

plt.scatter(x_test, y_test, color='b')

plt.plot(x_test, y_pred, color='k')

plt.xlabel('degree_p')

plt.ylabel('mba_p')

plt.title('Testing dataset')

plt.show()