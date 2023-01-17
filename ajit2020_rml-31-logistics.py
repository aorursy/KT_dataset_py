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
# Import data analysis modules

import numpy as np

import pandas as pd

import os

# to save model

import pickle

# Import visualization modules

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

Chemistry = pd.read_csv(('/kaggle/input/glass/glass.csv'))

Chemistry.head(5)
Chemistry.dtypes
Glass['Type'].unique()
#count of the target variable

sns.countplot(x='Type', data=Chemistry)
Chemistry.describe()
#We are checking correlation of values using Feature Matrix

features = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type']



mask = np.zeros_like(Glass[features].corr(), dtype=np.bool) 

mask[np.triu_indices_from(mask)] = True 

f, ax = plt.subplots(figsize=(16, 12))

plt.title('Correlation Matrix',fontsize=25)

sns.heatmap(Chemistry[features].corr(),linewidths=0.25,vmax=0.7,square=True,cmap="BuGn", 

            #"BuGn_r" to reverse 

            linecolor='b',annot=True,annot_kws={"size":8},mask=mask,cbar_kws={"shrink": .9});
# glass 1, 2, 3 are Thikc glass

# glass 5, 6, 7 are Thin glass

Chemistry['Classification'] = Chemistry.Type.map({1:0, 2:0, 3:0, 5:1, 6:1, 7:1})

Chemistry.head()
plt.scatter(Chemistry.Al, Chemistry.Classification)

plt.xlabel('Al')

plt.ylabel('Classification')
# Plot logistic regression line 

sns.regplot(x='Al', y='Classification', data=Chemistry, logistic=True, color='b')
from sklearn.model_selection import train_test_split



#Independent variable

X = Chemistry[['Al']]

#Dependent variable

y = Chemistry['Classification']

# Split data set into training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=200)
# Import model for fitting

from sklearn.linear_model import LogisticRegression



# Create instance (i.e. object) of LogisticRegression

model = LogisticRegression(class_weight='balanced')

output=model.fit(X_train, y_train)

output
from sklearn.metrics import confusion_matrix 

from sklearn.metrics import accuracy_score 

from sklearn.metrics import classification_report
y_pred = model.predict(X_test)



#Confusion matrix

results = confusion_matrix(y_test, y_pred)

print(results)



#Accuracy score

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy rate : {0:.2f} %".format(100 * accuracy))



#Classification report

report = classification_report(y_test, y_pred)

print(report)
pkl_filename = "pickle_model.pkl"

with open(pkl_filename, 'wb') as file:

    pickle.dump(model, file)



# Load from file

with open(pkl_filename, 'rb') as file:

    pickle_model = pickle.load(file)







Ypredict = pickle_model.predict(X_test)

model.fit(X_train,y_train)

y_predict = model.predict(X_test)

y_predict