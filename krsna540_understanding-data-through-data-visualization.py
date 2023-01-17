# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # for data visualization

import seaborn as sns

from sklearn import preprocessing

from sklearn.preprocessing import MinMaxScaler

sns.set(style='darkgrid')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
orgData = pd.read_csv("/kaggle/input/breast-cancer-wisconsin-data/data.csv")

orgData.head()
orgData.isnull().count().count()
orgData.drop('Unnamed: 32', axis=1, inplace=True)

orgData.drop('id', axis=1, inplace=True)

orgData.head()
orgData.info()
## Plotting the bar char to identify the frequnecy of values

sns.countplot(orgData["diagnosis"])

##prinitng number of values for each type

print(orgData["diagnosis"].value_counts())
diag_binary = {'M': 1,'B': 0}

orgData["diagnosis"]= [diag_binary[item] for item in orgData["diagnosis"]]

orgData.head()
orgData.describe()
fig, ax = plt.subplots(3, 4, figsize=(20, 10))

batch1=['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave points_mean','concave points_worst','symmetry_worst','fractal_dimension_worst']

for variable, subplot in zip(batch1, ax.flatten()):

    sns.boxplot(orgData[variable], ax=subplot)

    for label in subplot.get_xticklabels():

        label.set_rotation(0)
fig, ax = plt.subplots(4, 5, figsize=(20, 10))

batch1=['symmetry_mean','fractal_dimension_mean','radius_se','texture_se','perimeter_se','area_se','smoothness_se', 'compactness_se','concavity_se','concave points_se' , 'concavity_se','concave points_se','symmetry_se','fractal_dimension_se','radius_worst','texture_worst', 'perimeter_worst','area_worst','smoothness_worst','compactness_worst','concavity_worst']

for variable, subplot in zip(batch1, ax.flatten()):

    sns.boxplot(orgData[variable], ax=subplot)

    for label in subplot.get_xticklabels():

        label.set_rotation(0)
orgData.shape
Q1 = orgData.quantile(0.25)

Q3 = orgData.quantile(0.75)

IQR = Q3 - Q1



df = orgData[~((orgData < (Q1 - 1.5 * IQR)) |(orgData > (Q3 + 1.5 * IQR))).any(axis=1)]

orgData=pd.DataFrame(df)

orgData.shape
## Applying Min Max scaller makes more sense

scaler = MinMaxScaler()

MinMaxData = pd.DataFrame(scaler.fit_transform(orgData),columns=orgData.columns.values)

MinMaxData.head()
y=MinMaxData["diagnosis"]

MinMaxData=MinMaxData.drop('diagnosis',axis = 1 )
y.head()
f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(MinMaxData.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
corr_matrix = MinMaxData.corr().abs()



#the matrix is symmetric so we need to extract upper triangle matrix without diagonal (k = 1)

sol = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

                 .stack()

                 .sort_values(ascending=False))

#first element of sol series is the pair with the bigest correlation
print(sol.head(30))
drop_list1 = ['perimeter_mean','radius_mean','compactness_mean','concavity_worst','concave points_mean','area_se','smoothness_mean','radius_se','perimeter_se','radius_worst','perimeter_worst','compactness_worst','concave points_worst','compactness_se','concave points_se','texture_worst','area_worst']

df = MinMaxData.drop(drop_list1,axis = 1 )        # do not modify x, we will use it later 

df.head()
f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score,confusion_matrix

from sklearn.metrics import accuracy_score



# split data train 70 % and test 30 %

x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=42)



#random forest classifier with n_estimators=10 (default)

clf_rf = RandomForestClassifier(random_state=43)      

clr_rf = clf_rf.fit(x_train,y_train)



ac = accuracy_score(y_test,clf_rf.predict(x_test))

print('Accuracy is: ',ac)

cm = confusion_matrix(y_test,clf_rf.predict(x_test))