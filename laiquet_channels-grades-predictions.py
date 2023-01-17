import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import pylab as pl

from sklearn import preprocessing

from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from sklearn import linear_model

import warnings
#Disabling warnings

warnings.simplefilter("ignore")
#importing data and replacing missing set with NaNs

missing=["-- "," --","--"]

data = pd.read_csv('../input/data.csv', na_values=missing)
#Displaying columns of data

data.columns
#Finding empty cells

data.isna().sum()
#Checking data for any other inconsistent values

data
#Typecasting to int

pd.to_numeric(data['Video Uploads'])

pd.to_numeric(data['Subscribers'])



#Replacing NaNs with mean value of respective columns (Video Uploads,Subscribers,Video views) in the respective cells

data['Video Uploads']=data['Video Uploads'].fillna(round(data['Video Uploads'].mean(), 0))

data['Subscribers']=data['Subscribers'].fillna(round(data['Subscribers'].mean(), 0))

data['Video views']=data['Video views'].fillna(round(data['Video views'].mean(), 0))
#Displaying shape and description of data

print(data.shape)

data.describe()
#Peek at Transformed dataset

data.head(20)
#Grades Distribution with barplot

plotD = data.groupby(['Grade']).Grade.count()

pl.figure(figsize =(25,5))

pl.ylabel('Grades', fontsize=15)

pl.xlabel('Total count', fontsize=15)

plot = plotD.plot('barh')

plot.tick_params(axis='both', which='major', labelsize=15)
#Grades Distribution with piechart - Â is represented as '\xa0' so it willl not get rendered in the legend plot.

fig1, ax1 = plt.subplots(figsize=(10,5))

ax1.pie(plotD.values)

total=sum(plotD.values)

plt.legend(

    loc='upper left',

    labels=['%s, %1.1f%%' % (

        l, (float(s) / total) * 100) for l, s in zip(plotD.index, plotD.values)],

    prop={'size': 13},

    bbox_to_anchor=(0.2, 1),

    bbox_transform=fig1.transFigure

)
#Transforming data - Grades & Ranks into numerical values

data['Grade']=data['Grade'].replace('A++ ', 6)

data['Grade']=data['Grade'].replace('A+ ', 5)

data['Grade']=data['Grade'].replace('A ', 4)

data['Grade']=data['Grade'].replace('A- ', 3)

data['Grade']=data['Grade'].replace('B+ ', 2)

data['Grade']=data['Grade'].replace('\xa0 ', 1)

pd.to_numeric(data['Grade'])

data['Rank']=data['Rank'].replace(data['Rank'].values, range(1,5001))
#Pairplot of parameters

sns.pairplot(data, kind="reg")
#Correlation matrix & Heatmap

pl.figure(figsize =(10,5))

corrmat = data.corr()

sns.heatmap(corrmat, annot=True, fmt='.1f', vmin=0, vmax=1, square=True);
#Separating Labels and featureSet columns

columns = data.columns.tolist()

columns = [c for c in columns if c not in ['Grade','Channel name']]

target = 'Grade'



X = data[columns]

y = data[target]
#Splitting data into training and testing sets and further normalizing training and testing FeatureSets data for better classifier's results

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

X_train_scaled = preprocessing.scale(X_train)

X_test_scaled = preprocessing.scale(X_test)



print("Training FeatureSet:", X_train.shape)

print("Training Labels:", y_train.shape)

print("Testing FeatureSet:", X_test.shape)

print("Testing Labels:", y_test.shape)
#Initializing the model with some parameters.

model = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)

#Fitting the model to the data.

model.fit(X_train_scaled, y_train)

#Generating predictions for the test set.

predictions = model.predict(X_test_scaled)

#Computing the Model Accuracy

print("SGD Accuracy:",metrics.accuracy_score(y_test, predictions))

#Computing the error.

print("Mean Absoulte Error:", mean_absolute_error(predictions, y_test))

#Computing classification Report

print("Classification Report:\n", classification_report(y_test, predictions))

#Plotting confusion matrix

print("Confusion Matrix:")

df = pd.DataFrame(

    confusion_matrix(y_test, predictions),

    index = [['actual', 'actual', 'actual', 'actual', 'actual', 'actual'], ['1','2','3','4','5','6']],

    columns = [['predicted', 'predicted', 'predicted', 'predicted', 'predicted', 'predicted'], ['1','2','3','4','5','6']])

print(df)
#Initializing the model with some parameters.

model = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, random_state=1)

#Fitting the model to the data.

model.fit(X_train_scaled, y_train)

#Generating predictions for the test set.

predictions = model.predict(X_test_scaled)

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

    index = [['actual', 'actual', 'actual', 'actual', 'actual', 'actual'], ['1','2','3','4','5','6']],

    columns = [['predicted', 'predicted', 'predicted', 'predicted', 'predicted', 'predicted'], ['1','2','3','4','5','6']])

print(df)
#Initializing the model with some parameters.

model = SVC(gamma='auto')

#Fitting the model to the data.

model.fit(X_train_scaled, y_train)

#Generating predictions for the test set.

predictions = model.predict(X_test_scaled)

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

    index = [['actual', 'actual', 'actual', 'actual', 'actual', 'actual'], ['1','2','3','4','5','6']],

    columns = [['predicted', 'predicted', 'predicted', 'predicted', 'predicted', 'predicted'], ['1','2','3','4','5','6']])

print(df)