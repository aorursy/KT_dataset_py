# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/glass/glass.csv'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
os.path.isfile('/kaggle/input/glass/glass.csv')
File=pd.read_csv('/kaggle/input/glass/glass.csv')

File.head(5)
File.describe()
File.dtypes
missing_values = File.isnull()

missing_values.head(5)
File.shape
import seaborn as sns


sns.pairplot(File)
import matplotlib.pyplot as plt
mask = np.zeros_like(File.corr(), dtype=np.bool) 

mask[np.triu_indices_from(mask)] = True 

f, ax = plt.subplots(figsize=(16, 12))

plt.title('Pearson Correlation Matrix',fontsize=25)

sns.heatmap(File.corr(),linewidths=0.25,vmax=0.7,square=True,cmap="BuGn", 

            #"BuGn_r" to reverse 

            linecolor='w',annot=True,annot_kws={"size":8},mask=mask,cbar_kws={"shrink": .9});


x = File[['Al']]

y = File['Type']

#Step 5.2: Split data into Train and test



# Import module to split dataset

from sklearn.model_selection import train_test_split

# Split data set into training and test sets

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=100)
print(x.shape)

#print(x_test.head())

#print(y_train.head())

print(y.shape)
#Step 6 : Run the model



# Import model for fitting

from sklearn.linear_model import LogisticRegression

# Create instance (i.e. object) of LogisticRegression

#model = LogisticRegression()



#You can try follwoing variation on above model, above is just default one

model = LogisticRegression()

# Fit the model using the training data

# X_train -> parameter supplies the data features

# y_train -> parameter supplies the target labels

output_model=model.fit(x,y)

#output =x_test

#output['vehicleTypeId'] = y_test

output_model
from sklearn import linear_model




import pickle



#Step 7.0 Save the model in pickle

#Save to file in the current working directory

pkl_filename = "pickle_model.pkl"

with open(pkl_filename, 'wb') as file:

    pickle.dump(model, file)



# Load from file

with open(pkl_filename, 'rb') as file:

    pickle_model = pickle.load(file)



# Calculate the accuracy score and predict target values

score = pickle_model.score(x_test, y_test)

#print(score)

print("Test score: {0:.2f} %".format(100 * score))

Ypredict = pickle_model.predict(x_test)
model.predict(x_train)
model.predict(x_train)[0:15]
sns.regplot(x='Al', y='Type', data=File, logistic=False)
sns.regplot(x='Al', y='Type', data=File, logistic=True, color='b')
print (x_test.shape)

print (y_test.shape)

print (x_test)

print (y_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': Ypredict.flatten()})

df
plt.figure(figsize=(9,9))

sns.heatmap(File, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');

plt.ylabel('Actual label');

plt.xlabel('Predicted label');

all_sample_title = 'Accuracy Score: {0}'.format(score)

plt.title(all_sample_title, size = 15);




from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, Ypredict))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, Ypredict))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, Ypredict)))




import matplotlib.pyplot as plt



ax = plt.axes()

ax.scatter(x, y)

plt.title("Input Data and regression line ") 

ax.plot(x_test, Ypredict, color ='Red')

ax.set_xlabel('x')

ax.set_ylabel('y')

ax.axis('tight')

plt.show()
from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score



predictions = model.predict(x_test)

#print("",classification_report(y_test, predictions))

#print("confusion_matrix",confusion_matrix(y_test, predictions))

#print("accuracy_score",accuracy_score(y_test, predictions))

##**Accuracy is a classification metric. You can't use it with a regression. See the documentation for info on the various metrics.

#For regression problems you can use: R2 Score, MSE (Mean Squared Error), RMSE (Root Mean Squared Error).

#print("Score",score(y_test, X_test))

#score(self, X, y, sample_weight=None)

## setting plot style 

plt.style.use('fivethirtyeight') 

  

## plotting residual errors in training data 

plt.scatter(model.predict(x_train), model.predict(x_train) - y_train, 

            color = "green", s = 1, label = 'Train data' ,linewidth = 5) 

  

## plotting residual errors in test data 

plt.scatter(model.predict(x_test), model.predict(x_test) - y_test, 

            color = "blue", s = 1, label = 'Test data' ,linewidth = 4) 

  

## plotting line for zero residual error 

plt.hlines(y = 0, xmin = 0, xmax = 4, linewidth = 2) 

  

## plotting legend 

plt.legend(loc = 'upper right') 

  

## plot title 

plt.title("Residual errors") 

  

## function to show plot 

plt.show() 
from sklearn.ensemble import RandomForestRegressor

rf_regressor=RandomForestRegressor(n_estimators=28, random_state=0)

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import explained_variance_score

rf_regressor.fit(x_train,y_train)

rf_regressor.score(x_test,y_test)

rf_pred=rf_regressor.predict(x_test)

rf_score=rf_regressor.score(x_test,y_test)

expl_rf=explained_variance_score(rf_pred,y_test)
print("Random Forest regression Model Score is",round(rf_regressor.score(x_test,y_test)*100))




# Split data into 'X' features and 'y' target label sets

X1 = File[['RI','Na','Mg','Al','Si','K','Ca']]

y1 = File['Type']



from sklearn.model_selection import train_test_split

# Split data set into training and test sets

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.25, random_state=100)
#Step 6 : Run the model



# Import model for fitting

from sklearn.linear_model import LogisticRegression

# Create instance (i.e. object) of LogisticRegression

#model = LogisticRegression()



#You can try follwoing variation on above model, above is just default one

model = LogisticRegression()

# Fit the model using the training data

# X_train -> parameter supplies the data features

# y_train -> parameter supplies the target labels

output_model=model.fit(x,y)

#output =x_test

#output['vehicleTypeId'] = y_test

output_model
from sklearn import linear_model
import pickle
model = LogisticRegression()

output_model=model.fit(X1_train, y1_train)

output_model

pkl_filename = "pickle_model.pkl"

with open(pkl_filename, 'wb') as file:

    pickle.dump(model, file)



# Load from file

with open(pkl_filename, 'rb') as file:

    pickle_model = pickle.load(file)



# Calculate the accuracy score and predict target values

score = pickle_model.score(X1_test, y1_test)

print("Test score: {0:.2f} %".format(100 * score))

Ypredict = pickle_model.predict(X1_test)
df = pd.DataFrame({'Actual': y1_test, 'Predicted': Ypredict.flatten()})

df