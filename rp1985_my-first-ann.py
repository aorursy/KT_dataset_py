#Importing the libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

import os

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

print(os.listdir("../input"))



warnings.filterwarnings('ignore') 



# Any results you write to the current directory are saved as output.
#Importing the data

data = pd.read_csv('../input/Churn_Modelling.csv')
data.head()
#Removing the unnecessary columns

data = data.drop(labels = ['RowNumber', 'CustomerId', 'Surname'], axis = 1)
#Checking for NULL values in the dataset

data.isnull().sum()
#Doing some Exploratory Data Analysis on our new Data

data.head()
#Checking for demographic exits

geoExit = data[['Geography', 'Exited']].groupby(['Geography'], as_index = False).sum()

plt.figure(figsize = (15,5))

ax = sns.barplot(x = 'Geography', y = 'Exited', data = geoExit, color = 'LightSeaGreen')

plt.show()
#Checking for Gender wise exits

genderExit = data[['Gender', 'Exited']].groupby(['Gender'], as_index = False).sum()

plt.figure(figsize = (10,5))

ax = sns.barplot(x = 'Gender', y = 'Exited', data = genderExit, color = 'teal')

plt.show()
#Checking for Age wise exits

#We will create a new column, AgeGroup and seggregate the ages into different buckets

data['Age Group'] = pd.cut(data['Age'], np.arange(15, 90, 10))

ageExit = data[['Age Group', 'Exited']].groupby(['Age Group'], as_index = False).sum()

plt.figure(figsize = (15,8))

sns.barplot(x = 'Age Group', y = 'Exited', data = ageExit, color = 'seagreen')

plt.show()
#Checking density plots for the Age wise exits

plt.figure(figsize=(15,8))

ax = sns.kdeplot(data['Age'][data.Exited == 1], color = 'darkturquoise', shade = True, bw = 0.6)

sns.kdeplot(data['Age'][data.Exited == 0], color = 'lightcoral', shade = True, bw = 0.6)

plt.legend(['Exited', 'Remained'])

plt.title('Density plot of Exited vs Remained for Age Group')

ax.set(xlabel =  'Age')

plt.xlim(10, 100)

plt.show()
#Checking the distribution of the feautres

dist = data.drop('Exited', axis = 1)

dist.hist(figsize = (20,10), bins = 20, xlabelsize = 5, ylabelsize = 5, color = 'lightblue')
#Encoding the categorical variables

data = pd.get_dummies(data, columns = ['Geography', 'Gender'])
data.head()
#Dropping one of the columns created during encoding of categorical variables

data = data.drop(labels = ['Geography_Spain', 'Gender_Male'], axis = 1)
sns.pairplot(data, hue = 'Exited', palette = 'husl')
#Checking for Collinearity between the dependent variables

corr = data.drop('Exited', axis = 1).corr()
plt.figure(figsize = (10,10))

plt.suptitle('Correlation between the diffrent variables')

sns.heatmap(corr, cmap = 'PuBu', annot = True, linewidths = 0.1)
#Splitting the dataset

X = data.drop(labels = ['Exited','Age Group'], axis = 1)

Y = data['Exited']
#Checking how the independent variables look like

X.head()
#Converting the dataframe and dataseries into an np.array

X = X.values

Y = Y.values

print(type(X), type(Y))
#Standardizing the data, but before that, we must split the data into training and test sets

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.8, random_state = 6)
#Now the standardization - since ew have variables ranging in different large values, this is required.

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
#Building the ANN.

#Importing Keras library packages

import keras

from keras.models import Sequential #To initialize the ANN

from keras.layers import Dense #To define the no of layers in the ANN
#Initialize the ANN

classifier = Sequential()



#Adding the input layer and specifying the first hidden layer

classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))



#Adding the second hidden layer

classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))



#Adding the output layer

classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))



#Compiling the ANN using Stochastic Gradient Descent optimizer, specifying Binary CE as my loss and using Accuracy as my measuring metric

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#Training the model, I have specified the batch_size after which my weights are adjusted, and the total no. of epochs

classifier.fit(X_train, Y_train, batch_size = 25, nb_epoch = 200)
#Predictions on the Test Set

Y_pred = classifier.predict(X_test)
#Converting the probabilities predicted above into TRUE or FALSE in order to get binary output

Y_pred = (Y_pred > 0.5)
#Checking for accuracy using Confusion_Matrix

cm = confusion_matrix(Y_test, Y_pred)



#My Confusion_Matrix accuracy

print(cm.trace()/ cm.sum())