#Import the packages

import pandas as pd

import numpy as np

import requests

import csv

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt 

import seaborn as sns 

from scipy import stats

from scipy.stats import norm, skew

import sklearn.metrics as metrics

import warnings  

warnings.filterwarnings('ignore')
#download dataset

df = pd.read_excel('../input/covid19/dataset.xlsx', encoding='utf8')
#Rename some important coluns to a simplier column name

df.rename(columns={"Patient age quantile": "age_quantile", 

                   "SARS-Cov-2 exam result": "cov_2_result",

                   "Patient addmited to regular ward (1=yes, 0=no)": "regular_ward", 

                   "Patient addmited to semi-intensive unit (1=yes, 0=no)": "semi_intensive",

                   "Patient addmited to intensive care unit (1=yes, 0=no)": "UTI"}, inplace= True)

# Total missing values for each feature

df.isnull().sum()
# Total number of missing values

df.isnull().sum().sum()
#Let´s see the covid-19 result 

df.cov_2_result.value_counts().plot(kind="bar")
#create a dataset with only the positive results 

positive= df.loc[(df.cov_2_result == "positive")]

positive.info()
positive.age_quantile.value_counts(ascending= False).plot(kind= "bar")
#mean age for patients in intensive care unit

positive.groupby(["UTI"])["age_quantile"].mean().plot(kind="bar", alpha= 0.5)

plt.title("Intensive care unit mean age")

#mean age of patients addmited to semi-intensive care unit

positive.groupby(["semi_intensive"])["age_quantile"].mean().plot(kind="bar", alpha= 0.5)

plt.title("Semi-intensive care unit mean age")
#mean age of patients addmited to regular ward

positive.groupby(["regular_ward"])["age_quantile"].mean().plot(kind="bar", alpha= 0.5)

plt.title("regular ward mean age")
#Feature Engineering

#missing data

total = df.isnull().sum().sort_values(ascending=False)

percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(25)
#mean of the hemoglobin for positive and negative covid-19 result

df.groupby(["cov_2_result"])["Hemoglobin"].mean().plot(kind="bar") 
df2= df.dropna(subset=["Hemoglobin"], how="any")
df2.info()
total = df2.isnull().sum().sort_values(ascending=False)

percent = (df2.isnull().sum()/df2.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(25)

#Drop all the columns with more than 20% missing values

df3 = df2.drop((missing_data[missing_data['Percent'] >= 0.60]).index,1)
df3.info()
#check missing data

total = df3.isnull().sum().sort_values(ascending=False)

percent = (df3.isnull().sum()/df3.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(15)

df3 = df3.drop((missing_data[missing_data['Percent'] >= 0.20]).index,1)
df3.info()
#check missing data

total = df3.isnull().sum().sort_values(ascending=False)

percent = (df3.isnull().sum()/df3.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(25)
df3= df3.dropna(subset=["Neutrophils"], how="any")
#check missing data

total = df3.isnull().sum().sort_values(ascending=False)

percent = (df3.isnull().sum()/df3.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(25)
df3= df3.dropna(subset=["Proteina C reativa mg/dL"], how="any")
#check missing data

total = df3.isnull().sum().sort_values(ascending=False)

percent = (df3.isnull().sum()/df3.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(25)

df3 = df3.drop((missing_data[missing_data['Percent'] >= 0.004]).index,1)
df3.info()
#check missing data

total = df3.isnull().sum().sort_values(ascending=False)

percent = (df3.isnull().sum()/df3.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(25)
df3.head(15)
#mean Neutrophils for postive and negative covid-19 results

df3.groupby(["cov_2_result"])["Neutrophils"].mean().plot(kind="bar")
#mean Proteina C reativa mg/dL for postive and negative covid-19 results

df3.groupby(["cov_2_result"])["Proteina C reativa mg/dL"].mean().plot(kind="bar")
#mean Mean corpuscular volume (MCV) for postive and negative covid-19 results

df3.groupby(["cov_2_result"])["Mean corpuscular volume (MCV)"].mean().plot(kind="bar")
#mean Leukocytes for postive and negative covid-19 results

df3.groupby(["cov_2_result"])["Leukocytes"].mean().plot(kind="bar")
#mean Eosinophils for postive and negative covid-19 results

df3.groupby(["cov_2_result"])["Eosinophils"].mean().plot(kind="bar")
#mean Basophils for postive and negative covid-19 results

df3.groupby(["cov_2_result"])["Basophils"].mean().plot(kind="bar")
#mean Hematocrit for postive and negative covid-19 results

df3.groupby(["cov_2_result"])["Hematocrit"].mean().plot(kind="bar")
#Variables selection

X= df3.drop(["cov_2_result"], axis=1)

Id= df3.loc[:, "Patient ID"]

y = df3.loc[:, "cov_2_result"]

X= X.drop(["Patient ID"], axis=1)



#Transform y into a Dataframe

y= pd.DataFrame(y)
X.info()
# Encoding categorical data

y= pd.get_dummies(y, prefix=["covid"], columns=["cov_2_result"])

y= y.drop(["covid_negative"], axis=1)
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
# Importing the Keras libraries and packages

import keras

from keras.models import Sequential

from keras.layers import Dense

# Initialising the ANN

classifier = Sequential()
# Adding the input layer and the first hidden layer

classifier.add(Dense(output_dim = 10, init = 'uniform', activation = 'relu', input_dim = 19))

# Adding the second hidden layer

classifier.add(Dense(output_dim = 10, init = 'uniform', activation = 'relu'))

# Adding the output layer

classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set

classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Predicting the Test set results

y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)
print(y_pred)
# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)
from sklearn.metrics import accuracy_score

print("accuracy score: ", accuracy_score(y_test, y_pred))
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
type(y_test)
#examine the class distribution of the testig set

#convert dataframe into series

y_test= y_test.iloc[:,0]

type(y_test)
y_test.value_counts()
#calculate the percentage of ones

y_test.mean()
#calculate the percentage of zeros

1- y_test.mean()

#calculate null accuracy(for bonary classification problems coded 1/0)

max(y_test.mean(), 1- y_test.mean())

#calculate null accuracy(for multi-classfication problem)

y_test.value_counts().head(1)/len(y_test)