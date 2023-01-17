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
#reading dataset to a variable named data

data = pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
#exploring the dataset

data.head()
# checking the datatypes and null values in dataset

data.info()
# Exploring statistics of the dataset

data.describe()
# Creating a new dataframe by removing first and last column

data2 = data.iloc[:,1:-1]

data2.head()
# Encoding the status column in dataset to '1' for placed and '0' for not placed.

data2['status'] = data2['status'].replace('Placed',1).replace('Not Placed',0)
# Exploring the new dataset

data2.head()
# Creating a heatmap for correlation between various columns to check the relation and select the variables for training the model.

import seaborn as sns

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(13,13))

sns.heatmap(data2.corr(),annot=True,ax=ax)
# Taking subset of dataset and creating a new dataframe named data3 which filters out unplaced students.

data3 = data2[data2['status']==1]

# Plotting the frequency of placed students with various categories to find some insights and inference.

plt.hist(data3['gender'])

plt.xlabel('Gender')

plt.ylabel('Students Placed')

plt.show()

plt.hist(data3['hsc_b'])

plt.xlabel('Higher Secondary school board')

plt.ylabel('Students Placed')

plt.show()

plt.hist(data3['hsc_s'])

plt.xlabel('Subject/Stream')

plt.ylabel('Students Placed')

plt.show()

plt.hist(data3['degree_t'])

plt.xlabel('Degree Name')

plt.ylabel('Students Placed')

plt.show()

plt.hist(data3['workex'])

plt.xlabel('Work Experience')

plt.ylabel('Students Placed')

plt.show()

plt.hist(data3['specialisation'])

plt.xlabel('MBA Branch')

plt.ylabel('Students Placed')

plt.show()
import numpy as np

import matplotlib.pyplot as pl

import pandas as pd

import seaborn as sns
# Selecting the training and target variables

x = data2[['ssc_p','hsc_p']].values

y = data2.iloc[:,-1].values
# Splitting the training and testing data

from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size = 0.15 , random_state = 5 )
# Training the multiple regression model

from sklearn.linear_model import LinearRegression

r = LinearRegression()

r.fit(x_train , y_train)
# Predicting the target variable

y_pred = r.predict(x_test)

y_pred = y_pred.round().astype(int)
# Finding the coefficients and intercept

print('Coefficients are ' , r.coef_)

print('Intercept is '  , r.intercept_)
# Calculating R-squared value

from sklearn.metrics import r2_score

r2 = r2_score(y_test,y_pred)

print(r2*100)
# Creating confusion matrix

from sklearn import metrics

cm = metrics.confusion_matrix(y_test,y_pred)

cm
# Plotting confusion matrix on a heatmap

sns.heatmap(cm,annot=True)