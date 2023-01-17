import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

import pylab as pl

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.svm import SVC

from sklearn import metrics

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.model_selection import train_test_split
#Disabling warning

warnings.simplefilter("ignore")
#Import data

data = pd.read_csv('../input/employee_reviews.csv', index_col=0)
#Droppig columns

data.drop(columns=['location', 'dates', 'link', 'advice-to-mgmt', 'summary', 'helpful-count', 'job-title'], inplace=True)
#Renaming columns

data.rename(columns={'overall-ratings':'overall', 'work-balance-stars':'workBalance', 'culture-values-stars':'cultureValue', 'carrer-opportunities-stars':'opportunities', 'senior-mangemnet-stars':'management','comp-benefit-stars':'benefits'}, inplace=True)
#Data Transformations

data['workBalance'] = np.where(data['workBalance']=='none', 0, data['workBalance'])

data['cultureValue'] = np.where(data['cultureValue']=='none', 0, data['cultureValue'])

data['opportunities'] = np.where(data['opportunities']=='none', 0, data['opportunities'])

data['benefits'] = np.where(data['benefits']=='none', 0, data['benefits'])

data['management'] = np.where(data['management']=='none', 0, data['management'])

data['workBalance'] = pd.to_numeric(data['workBalance'])

data['cultureValue'] = pd.to_numeric(data['cultureValue'])

data['opportunities'] = pd.to_numeric(data['opportunities'])

data['benefits'] = pd.to_numeric(data['benefits'])

data['management'] = pd.to_numeric(data['management'])
#Peek at data

print(data.shape)

print(data.describe())
data.head(5)
#Comparison of total count of pros/cons company wise

sns.set(context='notebook', style='whitegrid')

pl.figure(figsize =(15,5))

data.groupby(['company']).pros.count().plot('barh')

pl.ylabel('Company', fontsize=15)

pl.xlabel('Total (pros & cons)', fontsize=15)

pl.title('Companies pros & cons comparison', fontsize=15)

plt.show()
#Companies Overall Ratings

pl.figure(figsize =(10,5))

sns.boxplot(x="overall", y="company", data=data, whis="range", palette="vlag")

plt.xlabel('Overall Rating', fontsize=15)

plt.ylabel('Company', fontsize=15)

plt.title('Companies Overall Ratings', fontsize=15)

plt.show()
#Companies Work Balance Ratings

pl.figure(figsize =(10,5))

sns.boxplot(x="workBalance", y="company", data=data, whis="range", palette="vlag")

plt.xlabel('Work Balance Rating', fontsize=15)

plt.ylabel('Company', fontsize=15)

plt.title('Companies Work Balance Ratings', fontsize=15)

plt.show()
#Companies Culture Values Ratings

pl.figure(figsize =(10,5))

sns.boxplot(x="cultureValue", y="company", data=data, whis="range", palette="vlag")

plt.xlabel('Culture Values Rating', fontsize=15)

plt.ylabel('Company', fontsize=15)

plt.title('Companies Culture Values Ratings', fontsize=15)

plt.show()
#Companies Careerr Opportunities Stars

pl.figure(figsize =(10,5))

sns.boxplot(x="opportunities", y="company", data=data, whis="range", palette="vlag")

plt.xlabel('Career Opportunities Stars', fontsize=15)

plt.ylabel('Company', fontsize=15)

plt.title('Companies Career Opportunities Stars', fontsize=15)

plt.show()
#Companies Benefits Stars

pl.figure(figsize =(10,5))

sns.boxplot(x="benefits", y="company", data=data, whis="range", palette="vlag")

plt.xlabel('Company Benefits Stars', fontsize=15)

plt.ylabel('Company', fontsize=15)

plt.title('Companies Benefits Stars', fontsize=15)

plt.show()
#Companies Senior Management Stars

pl.figure(figsize =(10,5))

sns.boxplot(x="management", y="company", data=data, whis="range", palette="vlag")

plt.xlabel('Senior Management Stars', fontsize=15)

plt.ylabel('Company', fontsize=15)

plt.title('Companies Senior Management Stars', fontsize=15)

plt.show()
#Classifying and Adding new column of satisfied/Unsatisfied employees to data; 

#If ratings of Features(workBalance, cultureValue, opportunities, benefit, management) are greater than their means: 

#'Satisfied->1' else 'Unsatisfied->0'

data['remarks']=np.where((data['workBalance']>data['workBalance'].mean())&(data['cultureValue']>data['cultureValue'].mean())&(data['opportunities']>data['opportunities'].mean())&(data['benefits']>data['benefits'].mean())&(data['management']>data['management'].mean()), 1, 0)
print('Satisfied Employees:', (data['remarks']==1).sum())

print('Unsatisfied Employees:', (data['remarks']==0).sum())
#Creating satisfied/Unsatisfied arrays for barplots

satisfied = np.array(data[data['remarks']==1].groupby('company').remarks.count())

unsatisfied = np.array(data[data['remarks']==0].groupby('company').remarks.count())
#Plotting satisfied/Unsatisfied employees company wise

pl.figure(figsize =(20,10))

N = 6

ind = np.arange(N) 

width = 0.35       

plt.bar(ind, satisfied, width, label='Satisfied Employees')

plt.bar(ind + width, unsatisfied, width, label='UnSatisfied Employees')



plt.ylabel('Total Count', fontsize=15)

plt.xlabel('Company', fontsize=15)

plt.title('Satisfied/Unsatisfied Employees by Companies', fontsize=15)



plt.xticks(ind + width / 2, ('amazon', 'apple', 'facebook', 'google', 'microsoft', 'netflix'))

plt.legend(loc='best')

plt.show()
# Get all the columns from the dataframe.

columns = data.columns.tolist()

# Filtering unrequired columns

columns = [c for c in columns if c not in ["remarks", "pros", "cons", "company"]]

# Storing the variable we'll be predicting on.

target = "remarks"
#Defining features and labels

X = data[columns] #Features

y = data[target] #Labels
# Splitting dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)



#Printing the shapes of both sets.

print("Training FeaturesSet:", X_train.shape)

print("Training Labels:", y_train.shape)

print("Testing FeaturesSet:", X_test.shape)

print("Testing Labels:", y_test.shape)
#Initializing the model class.

model = SGDClassifier(max_iter = 100)

#Fitting the model to the training data.

model.fit(X_train, y_train)

#Generating our predictions for the test set.

predictions = model.predict(X_test)

#Computing the Model Accuracy

print("SGD Accuracy:",round(metrics.accuracy_score(y_test, predictions), 2))

#Computing the error.

print("Mean Absolute Error:", round(mean_absolute_error(predictions, y_test), 2))

#Computing classification Report

print("Classification Report:\n", classification_report(y_test, predictions))

#Plotting confusion matrix

print("Confusion Matrix:")

df = pd.DataFrame(

    confusion_matrix(y_test, predictions),

    index = [['actual', 'actual'], ['0','1']],

    columns = [['predicted', 'predicted'], ['0', '1']])

print(df)
#Initializing the model with some parameters.

model = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, random_state=1)

#Fitting the model to the data.

model.fit(X_train, y_train)

#Generating our predictions for the test set.

predictions = model.predict(X_test)

#Computing the Model Accuracy

print("Random Forrest Accuracy:",metrics.accuracy_score(y_test, predictions))

#Computing the error.

print("Mean Absoulte Error:", mean_absolute_error(predictions, y_test))

#Computing classification Report

print("Classification Report:\n", classification_report(y_test, predictions))

#Plotting confusion matrix

print("Confusion Matrix:")

df = pd.DataFrame(

    confusion_matrix(y_test, predictions),

    index = [['actual', 'actual'], ['0','1']],

    columns = [['predicted', 'predicted'], ['0', '1']])

print(df)
#Initializing the model with some parameters.

model = SVC(gamma='auto')

#Fitting the model to the data.

model.fit(X_train, y_train)

#Generating predictions for the test set.

predictions = model.predict(X_test)

#Computing the Model Accuracy

print("SVM Accuracy:",metrics.accuracy_score(y_test, predictions))

#Computing the error.

print("Mean Absoulte Error:", mean_absolute_error(predictions, y_test))

#Computing classification Report

print("Classification Report:\n", classification_report(y_test, predictions))

#Plotting confusion matrix

print("Confusion Matrix:")

df = pd.DataFrame(

    confusion_matrix(y_test, predictions),

    index = [['actual', 'actual'], ['0','1']],

    columns = [['predicted', 'predicted'], ['0', '1']])

print(df)