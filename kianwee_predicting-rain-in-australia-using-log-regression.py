# Importing required packages

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
# Creating a function to print 

def overview():

    data =pd.read_csv('../input/weather-dataset-rattle-package/weatherAUS.csv')

    print("First 5 lines of data:\n")

    print(data.head())

    print("\n\n\n")

    print("There are {} rows and {} columns".format(data.shape[0], data.shape[1]))

    print("\n\n\n")

    print("Data types:\n")

    print(data.dtypes)

    print("\n\n\n")

    print("% of missing values per column:\n")

    print(data.isnull().mean().round(2)*100)

    print("Statistical summary:\n")

    print(data.describe())

    return data

    

data = overview()
data.drop(columns = "RISK_MM", inplace = True)
sns.countplot(data = data, x = "RainTomorrow")
sns.countplot(data = data, x = "RainToday")
sns.distplot(data['MinTemp'])
sns.distplot(data['MaxTemp'])
# Create a function to separate out numerical and categorical data 

    ## Using this function to ensure that all non-numerical in a numerical column 

    ## and non-categorical in a categorical column is annotated

def cat_variable(df):

    return list(df.select_dtypes(include = ['category', 'object']))

    

def num_variable(df):

    return list(df.select_dtypes(exclude = ['category', 'object']))



categorical_variable = cat_variable(data)

numerical_variable = num_variable(data)
sns.pairplot(data[numerical_variable], kind='scatter', diag_kind='hist', palette='Rainbow')

plt.show()
# Dealing with outliers



q = data[numerical_variable].quantile(0.99)

data_new = data[data[numerical_variable]< q]
data = data.dropna(subset=['Rainfall', 'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Temp9am','Temp3pm','RainToday'])

cat_variable = ['WindGustDir', 'WindDir9am']

num_variable = ['MinTemp', 'MaxTemp', 'Evaporation', 'Sunshine', 'WindGustSpeed', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm']

imp_cat = SimpleImputer(missing_values = np.nan, strategy='most_frequent')

data[cat_variable] = imp_cat.fit_transform(data[cat_variable])

imp_num = SimpleImputer(missing_values = np.nan, strategy='median')

data[num_variable] = imp_num.fit_transform(data[num_variable])
le = LabelEncoder()

 

# Implementing LE on WindGustDir

le.fit(data.WindGustDir.drop_duplicates()) 

data.WindGustDir = le.transform(data.WindGustDir)



# Implementing LE on WindDir9am

le.fit(data.WindDir9am.drop_duplicates()) 

data.WindDir9am = le.transform(data.WindDir9am)



# Implementing LE on WindDir3pm

le.fit(data.WindDir3pm.drop_duplicates()) 

data.WindDir3pm = le.transform(data.WindDir3pm)



# Implementing LE on RainToday

le.fit(data.RainToday.drop_duplicates()) 

data.RainToday = le.transform(data.RainToday)



# Implementing LE on RainTomorrow

le.fit(data.RainTomorrow.drop_duplicates()) 

data.RainTomorrow = le.transform(data.RainTomorrow)
plt.figure(figsize=(15,10))

 

corrMatrix = data.corr()

sns.heatmap(corrMatrix, annot=True)

plt.show()
# Assigning X and y

X = data.drop(['RainTomorrow', 'Date', 'Location'], axis=1)



y = data['RainTomorrow']



# Implementing train and test splits

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
# Looking into the shape of training and test dataset

print(X_train.shape)

print(X_test.shape)
# instantiate the model

logreg = LogisticRegression(solver='liblinear', random_state=0)





# fit the model

logreg.fit(X_train, y_train)
y_pred_test = logreg.predict(X_test)

print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred_test)))