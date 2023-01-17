# Importing the libraries

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# Importing the dataset

df = pd.read_csv('../input/telecom_churn.csv')
df.head()
# Checking missing values

df.isna().sum()
# Checking for duplicates

df.duplicated().sum()
df.columns
# Let's have a look at whole dataset

round(df.describe().drop(['Area code'], axis=1, inplace=False),1)
# Deployment of customers with International plan

fig1, ax1 = plt.subplots(figsize=(10,5))

ax1.pie(df['International plan'].value_counts(), explode = (0.2,0), labels=['Yes', 'No'], autopct='%1.1f%%',

        shadow=True, startangle=90)

plt.legend()

plt.title('Deployment of customers with (Yes) / without (No) International plan')
# Deployment of customers with Voice mail plan

fig1, ax1 = plt.subplots(figsize=(10,5))

ax1.pie(df['Voice mail plan'].value_counts(), explode = (0.2,0), labels=['No', 'Yes'], autopct='%1.1f%%',

        shadow=True, startangle=90)

plt.legend()

plt.title('Deployment of customers with (Yes) / without (No) Voice mail plan')
# Deployment of customers who left the company

fig1, ax1 = plt.subplots(figsize=(10,5))

ax1.pie(df['Churn'].value_counts(), explode = (0.2,0), labels=['False', 'True'], autopct='%1.1f%%',

        shadow=True, startangle=90)

plt.legend()

plt.title('Churn deployment')
# Deployment of customers by day calls

fig1, ax1 = plt.subplots(figsize=(10,5))

sns.distplot(df['Total day calls'], bins=10)

plt.title('Deployment of customers by day calls')

plt.ylabel
# Deployment of customers by evening calls# 

fig1, ax1 = plt.subplots(figsize=(10,5))

sns.distplot(df['Total eve calls'], bins=10)

plt.title('Deployment of customers by evening calls')

plt.ylabel
# Deployment of customers by night calls

fig1, ax1 = plt.subplots(figsize=(10,5))

sns.distplot(df['Total night calls'], bins=10)

plt.title('Deployment of customers by night calls')

plt.ylabel
# Histograms of Numerical Columns

dataset = df.drop(columns = ['State','Area code', 'International plan','Voice mail plan', 'Churn'])

fig = plt.figure(figsize=(15, 12))

plt.suptitle('Histograms of Numerical Columns', fontsize=20)

for i in range(1, dataset.shape[1] + 1):

    plt.subplot(5, 3, i)

    f = plt.gca()

    f.axes.get_yaxis().set_visible(False)

    f.set_title(dataset.columns.values[i - 1])

    plt.hist(dataset.iloc[:, i - 1], bins=10)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# Taking care of categorical data

from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

categorical_cols = ['State', 'International plan', 'Voice mail plan', 'Churn']

df[categorical_cols] = df[categorical_cols].apply(lambda col: labelencoder.fit_transform(col.astype(str)))

df.head(10).transpose()
# Let's see the correlation with 'Churn' column

round(df[df.columns[:]].corr()['Churn'][:],3)
# Let's drop the columns with correlation less then 0.03 - State, Account length and Area code

df.drop(['State', 'Account length', 'Area code'], axis=1, inplace=True)
# Creating DV and IV sets

X = df.drop('Churn', axis=1)

y = df['Churn']



# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=123)
sum_met = pd.DataFrame(index = ['accuracy'], columns = ['Random_Forest', 'Naive_Bayes', 'Keras_ANN'])
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 50, criterion = 'entropy', random_state = 123)

classifier.fit(X_train, y_train)
# Predicting the Test set results

y_pred = classifier.predict(X_test)
# Making the Confusion Matrix and Classification report

from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, y_pred)



# Saving the accuracy for Random Forest

cl = classification_report(y_test, y_pred)

list_of_words = cl.split()

sum_met.at['accuracy','Random_Forest'] = list_of_words[list_of_words.index('accuracy') + 1]



# Printing the results

print('Confusion matrix:\n',cm)

print('\n','Classification report:\n\n',classification_report(y_test, y_pred))
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(X_train, y_train) 
# Predicting the Test set results

y_pred = classifier.predict(X_test)
# Making the Confusion Matrix and Classification report

from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, y_pred)



# Saving the accuracy for Naive Bayes

cl = classification_report(y_test, y_pred)

list_of_words = cl.split()

sum_met.at['accuracy','Naive_Bayes'] = list_of_words[list_of_words.index('accuracy') + 1]



# Printing the results

print('Confusion matrix:\n',cm)

print('\n','Classification report:\n\n',classification_report(y_test, y_pred))
# Importing the Keras libraries and packages

from keras import backend as K

K.tensorflow_backend._get_available_gpus() 

import keras

from keras.models import Sequential

from keras.layers import Dense



# Initialising the ANN

classifier = Sequential()



# Adding the input layer and the first hidden layer

classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu', input_dim = 16))



# Adding the second hidden layer

classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu'))



# Adding the output layer

classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))



# Compiling the ANN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



# Fitting the ANN to the Training set

classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
# Predicting the Test set results

y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)
# Making the Confusion Matrix and Classification report

from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, y_pred)



# Saving the accuracy for Naive Bayes

cl = classification_report(y_test, y_pred)

list_of_words = cl.split()

sum_met.at['accuracy','Keras_ANN'] = list_of_words[list_of_words.index('accuracy') + 1]



# Printing the results

print('Confusion matrix:\n',cm)

print('\n','Classification report:\n\n',classification_report(y_test, y_pred))
sum_met