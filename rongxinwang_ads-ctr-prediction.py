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
#Import packages

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
#Load dataset

data = pd.read_csv('../input/facebook-ads-2/Facebook_Ads_2.csv', encoding='latin-1')

data.head()
print("The total number of users in this dataset is: ", len(data))

print("The number of users who clicked through the ads is: ", len(data[data['Clicked']==1]))

print("The number of users who did not click through the ads is: ", len(data[data['Clicked']==0]))
#Count by country

data['Country'].value_counts()
#Distribution of time spent

plt.figure(figsize=[12,5])

fig, ax = plt.subplots(2)

sns.distplot(data['Time Spent on Site'], ax = ax[0])

sns.boxplot(x='Clicked', y='Time Spent on Site', data = data, ax = ax[1])

plt.show()
plt.figure(figsize = [12,5])

sns.distplot(data[data['Clicked']==0]['Time Spent on Site'], label = 'Clicked==0')

sns.distplot(data[data['Clicked']==1]['Time Spent on Site'], label = 'Clicked==1')

plt.legend()

plt.show()
#Distribution of salary

plt.figure(figsize = (12,5))

fig, ax = plt.subplots(1,2)

sns.distplot(data['Salary'], ax = ax[0])

sns.boxplot(data = data, x = 'Clicked', y = 'Salary', ax = ax[1])

plt.show()
plt.figure(figsize = [12,5])

sns.distplot(data[data['Clicked']==0]['Salary'], label = 'Clicked==0')

sns.distplot(data[data['Clicked']==1]['Salary'], label = 'Clicked==1')

plt.legend()

plt.show()
#Drop name, email, country

data.drop(['Names','emails','Country'], axis = 1, inplace = True)

data.head()
#Check on missing data

sns.heatmap(data.isnull(), yticklabels = False, cmap = 'Blues', cbar = False)
#Split data into dependent and independent variables

X = data.drop(['Clicked'], axis = 1).values

y = data['Clicked'].values
#Scale X

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X = sc.fit_transform(X)
#Split data into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
#Import and fit logistic regression mode

from sklearn.linear_model import LogisticRegression

LR = LogisticRegression(random_state = 0)

LR.fit(X_train, y_train)
y_pred = LR.predict(X_test)
#Import and run confusion matrix & classification report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot = True, fmt = 'd')

print(classification_report(y_test, y_pred))