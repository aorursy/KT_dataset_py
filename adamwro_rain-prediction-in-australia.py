# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# load data

df = pd.read_csv('../input/weatheraus/weatherAUS.csv')
# check the head of dataset

df.head()
# Check shape of dataset:

df.shape
# Check the datatypes

df.dtypes
# First - we check the distribution of the target value

counts = df['RainTomorrow'].value_counts()

print(counts)
# We check the exact ratio of 'Yes' samples

print(np.sum(counts))

print(counts[1]/np.sum(counts))
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

tmp = df.select_dtypes(include=numerics)

tmp["RainTomorrow"]= df["RainTomorrow"]
# check distributions of first 4 numerical values against target:

sns.pairplot(tmp, vars = tmp.columns[:4],hue="RainTomorrow")

plt.show()
# check distributions of first numerical values against target (cols 4-8):

sns.pairplot(tmp, vars = tmp.columns[4:8],hue="RainTomorrow")

plt.show()
# check distributions of first numerical values against target (cols 8-12):

sns.pairplot(tmp, vars = tmp.columns[8:12],hue="RainTomorrow")

plt.show()
# check distributions of first numerical values against target (cols 4-8):

sns.pairplot(tmp, vars = tmp.columns[12:16],hue="RainTomorrow")

plt.show()
# Just for curiosity we check the RISK_MM - but according to the data description we should drop this data to not oto overfit

# Below note from dataset description:

# "Note: You should exclude the variable Risk-MM when training a binary classification model. 

# Not excluding it will leak the answers to your model and reduce its predictability.""

sns.pairplot(tmp, vars = tmp.columns[16:17],hue="RainTomorrow")

plt.show()
# We should not use strict date in our model - instead we will engineer a feature by extracting the month.

# We assume that it makes sense that in some months rain is more likely to happen

df['Month'] = pd.to_datetime(df['Date']).dt.month



# We check the target distribution across our new feature

sns.countplot(x = 'Month', hue =  'RainTomorrow', orient = 'h', data = df)
# Now check  the location

# Set the plot size to make it more readable

plt.figure(figsize=(20, 10))

sns.countplot(y = 'Location', hue =  'RainTomorrow', orient = 'h', data = df)
len(df['Location'].unique())
df['Location'].value_counts()
sns.countplot(y = 'WindGustDir', hue =  'RainTomorrow', orient = 'h', data = df)
sns.countplot(y = 'WindDir9am', hue =  'RainTomorrow', orient = 'h', data = df)
sns.countplot(y = 'WindDir3pm', hue =  'RainTomorrow', orient = 'h', data = df)
sns.countplot(y = 'RainToday', hue =  'RainTomorrow', orient = 'h', data = df)
# We drop the Date to not overfit the model to particular date and place:

df.drop(['Date'], axis=1, inplace = True)



# And Risk-MM according to the data descripton:

# "Note: You should exclude the variable Risk-MM when training a binary classification model. 

# Not excluding it will leak the answers to your model and reduce its predictability.""

df.drop(['RISK_MM'], axis=1, inplace = True)
# check % of missing data in columns

df.isnull().sum()/df.shape[0]*100
# Evaporation, Sunshine Cloud 9 am and Cloud 3pm have a lot of missing data (above 30%)- we remove them:

df.drop(['Evaporation', 'Sunshine', 'Cloud9am', 'Cloud3pm'], axis=1, inplace = True)

df.isnull().sum()/df.shape[0]*100
df.shape
#replace Na in numerical columns with mean for columns with Na ratio higher than 3%:

df['WindGustSpeed'].fillna(np.mean(df['WindGustSpeed'].dropna().values), inplace = True)

df['Pressure9am'].fillna(np.mean(df['Pressure9am'].dropna().values), inplace = True)

df['Pressure3pm'].fillna(np.mean(df['Pressure3pm'].dropna().values), inplace = True)
# replace categorical values with the 'Unknown' value for columns with Na ratio higher than 3%:

df['WindGustDir']= df['WindGustDir'].fillna('Unknown')

df['WindDir9am']= df['WindDir9am'].fillna('Unknown')

df.isnull().sum()/df.shape[0]*100
df.dropna(inplace = True)

df.isnull().sum()/df.shape[0]*100
# First - we check the distribution of the target value

counts = df['RainTomorrow'].value_counts()

print(counts)
# We check the exact ratio of 'Yes' samples

print(np.sum(counts))

print(counts[1]/np.sum(counts))
# build temporary dataset:

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

tmp2 = df.select_dtypes(include=numerics)

tmp2["RainTomorrow"]= df["RainTomorrow"]

# check columns:

tmp2.columns
# check distributions of first 3 numerical values against target (to comapre with the previous plots on original data - we take 3 colums

# because we removed evaporation because a lot of Na:

sns.pairplot(tmp2, vars = tmp2.columns[:4],hue="RainTomorrow")

plt.show()
sns.pairplot(tmp2, vars = tmp2.columns[4:8],hue="RainTomorrow")

plt.show()
sns.pairplot(tmp2, vars = tmp2.columns[8:12],hue="RainTomorrow")

plt.show()
# checkt the types after data removal:

df.dtypes
plt.figure(figsize=(20, 10))

sns.countplot(y = 'Location', hue =  'RainTomorrow', orient = 'v', data = df)
sns.countplot(x = 'WindGustDir', hue =  'RainTomorrow', orient = 'h', data = df)
sns.countplot(x = 'WindDir9am', hue =  'RainTomorrow', orient = 'h', data = df)
sns.countplot(x = 'WindDir3pm', hue =  'RainTomorrow', orient = 'h', data = df)
sns.countplot(x = 'RainToday', hue =  'RainTomorrow', orient = 'h', data = df)
# replace the string labels with 0 and 1 numbers:

df['RainToday'].replace({'No':0,'Yes':1},inplace = True)

df['RainTomorrow'].replace({'No':0,'Yes':1},inplace = True)



# encode categorical values

categorical = ['WindGustDir','WindDir9am','WindDir3pm','Location']

df = pd.get_dummies(df,columns = categorical,drop_first=True)

df.shape
df.select_dtypes(include=numerics).describe()
from scipy import stats



skew_var = ['Humidity3pm', 'Humidity9am', 'Rainfall', 'WindSpeed3pm', 'WindSpeed9am']

tmp3 = df[skew_var]



for c in tmp3.columns:

    r = stats.boxcox(df[c] + 1)

    tmp3[c] = r[0]



sns.pairplot(tmp3)

plt.show
df[skew_var] = tmp3

df.shape
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x = df.drop(labels = ['RainTomorrow'],axis = 1)

x.columns
y = df['RainTomorrow']
x = sc.fit_transform(x)
x.shape
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.4,random_state = 40)

x_test,x_validation,y_test,y_validation = train_test_split(x_test,y_test,test_size = 0.5,random_state = 40)
import keras

from keras.models import Sequential

from keras.layers import Dense
classifier = Sequential()
classifier.add(Dense(units = 30,kernel_initializer='uniform',activation = 'relu',input_dim = 109))

classifier.add(Dense(units = 30,kernel_initializer='uniform',activation = 'relu'))

classifier.add(Dense(units = 30,kernel_initializer='uniform',activation = 'relu'))

classifier.add(Dense(units = 1,activation='sigmoid',kernel_initializer='uniform'))



from keras.utils import plot_model

plot_model(classifier, show_shapes=True, to_file='model.png')
classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])
classifier.fit(x_train,y_train,epochs = 100,batch_size=10)
y_pred = classifier.predict_classes(x_test)

y_train_pred = classifier.predict_classes(x_train)

y_validation_pred = classifier.predict_classes(x_validation)
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

print('Training Accuracy ---->',accuracy_score(y_train,y_train_pred))

print('Testing Accuracy  ---->',accuracy_score(y_test,y_pred))

print('Validation Accuracy  ---->',accuracy_score(y_validation,y_validation_pred))
print(classification_report(y_train,y_train_pred))
print(confusion_matrix(y_train,y_train_pred))
print(classification_report(y_validation,y_validation_pred))
print(confusion_matrix(y_validation,y_validation_pred))