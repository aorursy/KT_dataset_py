# from sklearn.datasets import fetch_mldata

from sklearn.preprocessing import StandardScaler

from sklearn import metrics

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
glass = pd.read_csv('../input/glass.csv',sep='\,', 

                  names=["Id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "glasstype"])

glass.head()
print(glass.shape)

print('\n',glass.glasstype.head())

print(glass.glasstype.shape)

print(np.unique(glass.glasstype))
y=glass.glasstype

x=glass.drop('glasstype',axis=1)

y



#spliting the dataset

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)

x_train.head()
y_train.head()
x_train.shape

x_test.head()
# #Standardize the data:

# 

# scaler = StandardScaler()

# scaler.fit(x_train)

# # Apply transform to both the training set and the test set.

# x_train = scaler.transform(x_train)

# x_test = scaler.transform(x_test)
#Fit the model:

model = LogisticRegression(solver = 'lbfgs')

model.fit(x_train, y_train)



# use the model to make predictions with the test data

y_pred = model.predict(x_test)

# how did our model perform?

count_misclassified = (y_test != y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))

#### plot the grapth 

from sklearn.linear_model import LinearRegression as lm

model=lm().fit(x_train,y_train)

predictions = model.predict(x_test)

import matplotlib.pyplot as plt

plt.scatter(y_test,predictions)

plt.xlabel('True values')

plt.ylabel('Predictions')

plt.show()
#plotting the graph between predi

fig, ax = plt.subplots()

ax.scatter(y_test, predictions)

ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)

ax.set_xlabel('Measured')

ax.set_ylabel('Predicted')

plt.show()