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

        global file_path

        #print(os.path.join(dirname, filename))

        file_path = os.path.join(dirname, filename)

        print(file_path)



# Any results you write to the current directory are saved as output.
# Import visualization modules

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv(file_path)

#Reading and printing few values from input file

df.head(5)
#Seems to be all numbers, but we will double check to ensure date type is integer or float.

df.dtypes
#We will check whether any value in dataframe is null, if applicable, we will remove that row

missing_values = df.isnull()

sns.heatmap(data = missing_values, yticklabels=False, cbar=False, cmap='viridis')
#As graph is even in nature, let's analyse count of each glass type in our model.

df.Type.value_counts().sort_index()
#There is no Type 4 Glass, so we will remap the values to make output either 0 or 1.

df['glass_type'] = df.Type.map({1:0, 2:0, 3:0, 5:1, 6:1, 7:1})

df.head()
#We will check correlation of values using Feature Matrix

features = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'glass_type']



mask = np.zeros_like(df[features].corr(), dtype=np.bool) 

mask[np.triu_indices_from(mask)] = True 

f, ax = plt.subplots(figsize=(16, 12))

plt.title('Glass Correlation Matrix',fontsize=25)

sns.heatmap(df[features].corr(),linewidths=0.25,vmax=0.7,square=True,cmap="BuGn", 

            #"BuGn_r" to reverse 

            linecolor='b',annot=True,annot_kws={"size":8},mask=mask,cbar_kws={"shrink": .9});
#As seen, plot of Al has maximum value followed by Ba, we will take independent variables 'Al', 'Ba'

#Independent variable

X = df[['Al', 'Ba']]

#Dependent variable

y = df['glass_type']
#Split training and testing data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=200)
# Import model for fitting

from sklearn.linear_model import LogisticRegression



# Create instance (i.e. object) of LogisticRegression

model = LogisticRegression(class_weight='balanced')

output_model=model.fit(X_train, y_train)

output_model
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