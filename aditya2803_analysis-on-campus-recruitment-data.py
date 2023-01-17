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

import matplotlib.pyplot as plt

import seaborn as sn
df = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv', index_col = 'sl_no')

df.head(10)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows = 2, ncols = 2, figsize = (12, 12), sharex = True)



ax = [ax1, ax2, ax3, ax4]



var = ['ssc_p', 'hsc_p', 'degree_p', 'mba_p']





for i, axes in zip(var, ax):

    

    sn.scatterplot(x = df.index, y = df[i], ax = axes)

    

plt.show()
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows = 2, ncols = 2, figsize = (12, 12), sharex = True)



ax = [ax1, ax2, ax3, ax4]



var = ['ssc_p', 'hsc_p', 'degree_p', 'mba_p']





for i, axes in zip(var, ax):

    

    sn.set(style="whitegrid")

    sn.boxplot(df[i], ax = axes)

    

plt.show()
# Correcting Outilers in hsc_p variable using IQR Score



Q1 = df['hsc_p'].quantile(0.25)

Q3 = df['hsc_p'].quantile(0.75)

IQR = Q3 - Q1

print('IQR Score :',IQR)



df['hsc_p'] = df['hsc_p'][~((df['hsc_p'] < (Q1 - 1.5 * IQR)) |(df['hsc_p'] > (Q3 + 1.5 * IQR)))]



sn.boxplot(df['hsc_p'])

plt.title('After Correcting Outliers')

plt.show()
import missingno as miss



miss.matrix(df)
df['hsc_p'] = df['hsc_p'].fillna(np.mean(df['hsc_p']))
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows = 2, ncols = 2, figsize = (12, 12), sharex = True, sharey = True)



ax = [ax1, ax2, ax3, ax4]



var = ['ssc_p', 'hsc_p', 'degree_p', 'mba_p']



for i, axes in zip(var, ax):

    

    sn.distplot(df[i][df.status == 'Placed'], ax = axes, label = 'Placed')

    sn.distplot(df[i][df.status == 'Not Placed'], ax = axes, label = 'Not Placed')

    axes.legend()

plt.show()
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows = 2, ncols = 2, figsize = (12, 12), sharey = True)



ax = [ax1, ax2, ax3, ax4]



var = ['ssc_b', 'hsc_b', 'degree_t', 'specialisation']



for i, axes in zip(var, ax):

    

    sn.countplot(df[i], hue = df['status'], ax = axes)

    axes.legend()

plt.show()
#Independent Variables

X = df[['ssc_p', 'hsc_p', 'degree_p', 'mba_p', 'specialisation']]



#Dependent Variables

y = df[['status']]



#Assigning 1 to Mkt&HR Specialisation and 0 to Mkt&Fin Specialisation

X = X.replace({'Mkt&HR' : 1, 'Mkt&Fin' : 0})



#Assigning 1 to Placed Students and 0 to Not Placed Students

y = y.replace({'Placed' : 1, 'Not Placed' : 0})
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, np.array(y).ravel(), random_state = 0)
from sklearn.linear_model import LogisticRegression



Linear = LogisticRegression(C = 15, solver = 'liblinear').fit(X_train, y_train)



pred = Linear.predict(X_test)





from sklearn.metrics import precision_score, recall_score

from sklearn.metrics import f1_score



print('Score : ', Linear.score(X_test, y_test))

print('\nf1 Score : ', f1_score(y_test, pred))

print('\nPrecision Score : ', precision_score(y_test, pred))

print('\nRecall Score : ', recall_score(y_test, pred))
status = pd.DataFrame(y_test).rename({0 : 'Actual Status'}, axis = 1)



status['Predicted Status'] = pred

status