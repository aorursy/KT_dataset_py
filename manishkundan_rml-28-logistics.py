import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



for dirname, _, filenames in os.walk('/kaggle/input/glass/glass.csv'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
os.path.isfile('/kaggle/input/glass/glass.csv')
Input=pd.read_csv('/kaggle/input/glass/glass.csv')

Input.head(5)
Input.dtypes
Input.describe()
missing_values = Input.isnull()

missing_values.head(5)
sns.heatmap(data = missing_values, yticklabels=False, cbar=False, cmap='viridis')
Input.Type.value_counts().sort_index()
Input.shape
import seaborn as sns

sns.pairplot(Input)

#Comparision of different graphs

mask = np.zeros_like(Input.corr(), dtype=np.bool) 

mask[np.triu_indices_from(mask)] = True 

f, ax = plt.subplots(figsize=(16, 12))

plt.title('Pearson Correlation Matrix',fontsize=25)

sns.heatmap(Input.corr(),linewidths=0.25,vmax=0.7,square=True,cmap="BuGn", 

            #"BuGn_r" to reverse 

            linecolor='w',annot=True,annot_kws={"size":8},mask=mask,cbar_kws={"shrink": .9});
Input['Output'] = Input.Type.map({1:0, 2:0, 3:0, 5:1, 6:1, 7:1})

#Output is for good and bad classification

Input.head()
plt.scatter(Input.Al, Input.Output)

plt.xlabel('Al')

plt.ylabel('Output')
sns.regplot(x='Al', y='Output', data=Input, logistic=True, color='b')
X = Input[['Al']]

#Dependent variable

Y = Input['Type']
#Split training and testing data

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=200)
print(X.shape)

#print(X_test.head())

#print(Y_train.head())

print(Y.shape)
# Run the model

# Import model for fitting

from sklearn.linear_model import LogisticRegression

# Create instance (i.e. object) of LogisticRegression

#model = LogisticRegression()



#You can try follwoing variation on above model, above is just default one

model = LogisticRegression()

# Fit the model using the training data

# X_train -> parameter supplies the data features

# Y_train -> parameter supplies the target labels

output_model=model.fit(X,Y)

#output =X_test

#output['vehicleTypeId'] = Y_test

output_model
from sklearn import linear_model

import pickle
model = LogisticRegression()

output_model=model.fit(X_train, Y_train)

output_model

pkl_filename = "pickle_model.pkl"

with open(pkl_filename, 'wb') as file:

    pickle.dump(model, file)



# Load from file

with open(pkl_filename, 'rb') as file:

    pickle_model = pickle.load(file)



# Calculate the accuracy score and predict target values

score = pickle_model.score(X_test, Y_test)

print("Test score: {0:.2f} %".format(100 * score))

Ypredict = pickle_model.predict(X_test)
from sklearn.metrics import confusion_matrix 

from sklearn.metrics import accuracy_score 

from sklearn.metrics import classification_report
Y_pred = model.predict(X_test)



#Confusion matrix

results = confusion_matrix(Y_test, Y_pred)

print(results)



#Accuracy score

accuracy = accuracy_score(Y_test, Y_pred)

print("Accuracy rate : {0:.2f} %".format(100 * accuracy))



#Classification report

report = classification_report(Y_test, Y_pred)

print(report)
df = pd.DataFrame({'Actual': Y_test, 'Predicted': Ypredict.flatten()})

df




from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Ypredict))  

print('Mean Squared Error:', metrics.mean_squared_error(Y_test, Ypredict))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, Ypredict)))



import matplotlib.pyplot as plt
ax = plt.axes()

ax.scatter(X, Y)

plt.title("Input Data and regression line ") 

ax.plot(X_test, Ypredict, color ='Red')

ax.set_xlabel('X')

ax.set_ylabel('Y')

ax.axis('tight')

plt.show()
from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score



predictions = model.predict(X_test)

plt.style.use('fivethirtyeight') 

  

## plotting residual errors in training data 

plt.scatter(model.predict(X_train), model.predict(X_train) - Y_train, 

            color = "green", s = 1, label = 'Train data' ,linewidth = 5) 

  

## plotting residual errors in test data 

plt.scatter(model.predict(X_test), model.predict(X_test) - Y_test, 

            color = "blue", s = 1, label = 'Test data' ,linewidth = 4) 

  

## plotting line for zero residual error 

plt.hlines(y = 0, xmin = 0, xmax = 4, linewidth = 2) 

  

## plotting legend 

plt.legend(loc = 'upper right') 

  

## plot title 

plt.title("Residual errors") 

  

## function to show plot 

plt.show() 
