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

        



import pickle

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Any results you write to the current directory are saved as output.
input_file = pd.read_csv('/kaggle/input/glass/glass.csv')

input_file.head (5)
input_file.describe()
input_file.shape
input_file.dtypes
#Step 3: Clean up data

# Use the .isnull() method to locate missing data

missing_values = input_file.isnull()

missing_values.head(5)
input_file['Type'].unique()
#Check correleation between the variables using Seaborn's pairplot. 

sns.pairplot(input_file)
sns.boxplot('Type','RI',data =input_file)
#Step 4.1: Visualize the data

# Use seaborn to conduct heatmap to identify missing data

# data -> argument refers to the data to creat heatmap

# yticklabels -> argument avoids plotting the column names

# cbar -> argument identifies if a colorbar is required or not

# cmap -> argument identifies the color of the heatmap

sns.heatmap(data = missing_values, yticklabels=False, cbar=False, cmap='viridis')
#Create an empty list called traget

target = []

for i in input_file['Type']:

    if i >= 1 and i <= 4:

        target.append('1')

    elif i >= 5 and i <= 7:

        target.append('2')

input_file['Target'] = target
input_file.head()
input_file.tail()
# Importing the dataset



#X = input_file.iloc[:,2].values

#y = input_file.iloc[:,4].values



#X = input_file[['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']]

#X = np.array(input_file.iloc[:,:-1])

#y = input_file[['Type']]

#y = np.array(input_file['Type'])



#Alternate method

# glass_type 1, 2, 3 are window glass

# glass_type 5, 6, 7 are non-window glass

#df['household'] = df.Type.map({1:0, 2:0, 3:0, 5:1, 6:1, 7:1})

#df.head()





X= input_file.iloc[:,:9]

y = input_file['Target']



print (X)

print (y)
#Step 5.2: Split data into Train and test



# Import module to split dataset

from sklearn.model_selection import train_test_split

# Split data set into training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=200)
#Step 5.3: Checking file types created



print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)



print(y_test)

#y_test.dtypes
plt.scatter(input_file.Al, input_file.Target)

plt.xlabel('Al')

plt.ylabel('Target')
#Step 6 : Run the model



# Import model for fitting

from sklearn.linear_model import LogisticRegression

#from sklearn import datasets

#from sklearn.metrics import mean_squared_error, r2_score



# Create instance (i.e. object) of LogisticRegression

#model = LogisticRegression()



#You can try following variation on above model, above is just default one

model = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, 

                           fit_intercept=True, intercept_scaling=1, class_weight=None, 

                           random_state=None, solver='lbfgs', max_iter=10000, multi_class='auto', 

                           verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)



# Fit the model using the training data

# X_train -> parameter supplies the data features

# y_train -> parameter supplies the target labels

output_model=model.fit(X_train, y_train)

output_model



# Create a seperate table to store predictions

#glass_df = X_train

#glass_df2 = y_train





# Create a seperate table to store predictions

#glass_df = X_train[['Al']]

#glass_df['household_actual'] = y_train



#y_pred = model.predict(X_test)
model.predict_proba(X_train)
model.predict(X_train)
print (X_test.shape)

print (y_test.shape)

print (X_test)

print (y_test)
print('Coefficients: \n', model.intercept_)

print('Coefficients: \n', model.coef_)

print('Coefficients: \n', model.n_iter_)
model.score(X_test, y_test)*100
from sklearn.metrics import confusion_matrix, accuracy_score

predict=model.predict(X_test)

print(confusion_matrix(y_test,predict))

print(accuracy_score(y_test,predict))
#Plot the Logistic graph

sns.regplot(x='Al', y='y_test', data=output_model, logistic=True, color='b')            