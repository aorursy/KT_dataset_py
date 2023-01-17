import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib

import seaborn as sns

import pylab as pl

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.model_selection import train_test_split

import warnings
#Disabling warnings

warnings.simplefilter("ignore")
#Importing data

data = pd.read_csv("../input/weatherAUS.csv")
#Data shape & description

print(data.shape)

print(data.describe())
#Peeking at data

data.head(10)
#Checking for nulls

data.isna().sum().sort_values(ascending=False)
#Dropping unrequired columns and columns with more than 50K empty cells  

data.drop(columns={"Sunshine", "Evaporation", "Cloud3pm", "Cloud9am", "Date", "Location"}, inplace=True)
#Dropping rows with any empty cell

data = data.dropna(how="any")
#Data Transformations

encode = LabelEncoder()

encode.fit(['E','ENE','ESE','N','NA','NE','NNE','NNW','NW','S','SE','SSE','SSW','SW','W','WNW','WSW'])

data['WindGustDir'] = encode.transform(data['WindGustDir'])



encode.fit(['E','ENE','ESE','N','NA','NE','NNE','NNW','NW','S','SE','SSE','SSW','SW','W','WNW','WSW'])

data['WindDir9am'] = encode.transform(data['WindDir9am'])



encode.fit(['E','ENE','ESE','N','NA','NE','NNE','NNW','NW','S','SE','SSE','SSW','SW','W','WNW','WSW'])

data['WindDir3pm'] = encode.transform(data['WindDir3pm'])



encode.fit(['No','Yes'])

data['RainToday'] = encode.transform(data['RainToday'])



encode.fit(['No','Yes'])

data['RainTomorrow'] = encode.transform(data['RainTomorrow'])
data.head(10)
#Plotting graphs of different features w.r.t predictions of rain

#Graphs are plotted against features data w.r.t to [RainTomorrow] if No or Yes

#The main purpose of these graphs are to analyze relationship b/w features real values & mean and how different are features from each other with [RainTomorrow] if No or Yes 

fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(20, 40))

matplotlib.style.use('bmh')



dataSpec1 = data[data['RainTomorrow']==0].groupby('MinTemp')

dataSpec2 = data[data['RainTomorrow']==1].groupby('MinTemp')

mtmean = data['MinTemp'].rolling(1000).mean()

plot1 = dataSpec1['MinTemp'].head().plot(ax=axes[0][0],sharex=True, label='Min Temp for Rain')

plot2 = dataSpec2['MinTemp'].head().plot(ax=axes[0][0],sharex=True, label='Min Temp for no Rain')

plot3 = mtmean.plot(ax=axes[0][0],sharex=True, label='Mean Min Temp')

plot1.legend(loc="upper right")

axes[0][0].set_ylabel('Min Temperature [$^o$C]')



dataSpec1 = data[data['RainTomorrow']==0].groupby('MaxTemp')

dataSpec2 = data[data['RainTomorrow']==1].groupby('MaxTemp')

mtmean = data['MaxTemp'].rolling(1000).mean()

plot1 = dataSpec1['MaxTemp'].head().plot(ax=axes[0][1],sharex=True, label='Max Temp for Rain')

plot2 = dataSpec2['MaxTemp'].head().plot(ax=axes[0][1],sharex=True, label='Max Temp for no Rain')

plot3 = mtmean.plot(ax=axes[0][1],sharex=True, label='Mean Max Temp')

plot1.legend(loc="upper right")

axes[0][1].set_ylabel('Max Temperature [$^o$C]')



dataSpec1 = data[data['RainTomorrow']==0].groupby('WindGustSpeed')

dataSpec2 = data[data['RainTomorrow']==1].groupby('WindGustSpeed')

mtmean = data['WindGustSpeed'].rolling(1000).mean()

plot1 = dataSpec1['WindGustSpeed'].head().plot(ax=axes[1][0],sharex=True, label='Wind Speed for Rain')

plot2 = dataSpec2['WindGustSpeed'].head().plot(ax=axes[1][0],sharex=True, label='Wind Speed for no Rain')

plot3 = mtmean.plot(ax=axes[1][0],sharex=True, label='Mean Wind Gust Speed')

plot1.legend(loc="upper right")

axes[1][0].set_ylabel('Wind Speed [km/h]')



dataSpec1 = data[data['RainTomorrow']==0].groupby('WindSpeed9am')

dataSpec2 = data[data['RainTomorrow']==1].groupby('WindSpeed9am')

mtmean = data['WindSpeed9am'].rolling(1000).mean()

plot1 = dataSpec1['WindSpeed9am'].head().plot(ax=axes[1][1],sharex=True, label='Wind Speed@9am for Rain')

plot2 = dataSpec2['WindSpeed9am'].head().plot(ax=axes[1][1],sharex=True, label='Wind Speed@9am for no Rain')

plot3 = mtmean.plot(ax=axes[1][1],sharex=True, label='Mean Wind Speed@9am')

plot1.legend(loc="upper right")

axes[1][1].set_ylabel('Wind Speed@9am [km/h]')



dataSpec1 = data[data['RainTomorrow']==0].groupby('WindSpeed3pm')

dataSpec2 = data[data['RainTomorrow']==1].groupby('WindSpeed3pm')

mtmean = data['WindSpeed3pm'].rolling(1000).mean()

plot1 = dataSpec1['WindSpeed3pm'].head().plot(ax=axes[2][0],sharex=True, label='Wind Speed@3pm for Rain')

plot2 = dataSpec2['WindSpeed3pm'].head().plot(ax=axes[2][0],sharex=True, label='Wind Speed@3pm for no Rain')

plot3 = mtmean.plot(ax=axes[2][0],sharex=True, label='Mean Wind Speed@3pm')

plot1.legend(loc="upper right")

axes[2][0].set_ylabel('Wind Speed@3pm [km/h]')



dataSpec1 = data[data['RainTomorrow']==0].groupby('Humidity9am')

dataSpec2 = data[data['RainTomorrow']==1].groupby('Humidity9am')

mtmean = data['Humidity9am'].rolling(1000).mean()

plot1 = dataSpec1['Humidity9am'].head().plot(ax=axes[2][1],sharex=True, label='Humidity@9am for Rain')

plot2 = dataSpec2['Humidity9am'].head().plot(ax=axes[2][1],sharex=True, label='Humidity@9am for no Rain')

plot3 = mtmean.plot(ax=axes[2][1],sharex=True, label='Mean Humidity@9am')

plot1.legend(loc="upper right")

axes[2][1].set_ylabel('Humidity@9am [%]')



dataSpec1 = data[data['RainTomorrow']==0].groupby('Humidity3pm')

dataSpec2 = data[data['RainTomorrow']==1].groupby('Humidity3pm')

mtmean = data['Humidity3pm'].rolling(1000).mean()

plot1 = dataSpec1['Humidity3pm'].head().plot(ax=axes[3][0],sharex=True, label='Humidity@3pm for Rain')

plot2 = dataSpec2['Humidity3pm'].head().plot(ax=axes[3][0],sharex=True, label='Humidity@3pm for no Rain')

plot3 = mtmean.plot(ax=axes[3][0],sharex=True, label='Mean Humidity@3pm')

plot1.legend(loc="upper right")

axes[3][0].set_ylabel('Humidity@3pm [%]')



dataSpec1 = data[data['RainTomorrow']==0].groupby('Pressure9am')

dataSpec2 = data[data['RainTomorrow']==1].groupby('Pressure9am')

mtmean = data['Pressure9am'].rolling(1000).mean()

plot1 = dataSpec1['Pressure9am'].head().plot(ax=axes[3][1],sharex=True, label='Pressure@9am for Rain')

plot2 = dataSpec2['Pressure9am'].head().plot(ax=axes[3][1],sharex=True, label='Pressure@9am for no Rain')

plot3 = mtmean.plot(ax=axes[3][1],sharex=True, label='Mean Pressure@9am')

plot1.legend(loc="upper right")

axes[3][1].set_ylabel('Pressure@9am [hpa]')



dataSpec1 = data[data['RainTomorrow']==0].groupby('Pressure3pm')

dataSpec2 = data[data['RainTomorrow']==1].groupby('Pressure3pm')

mtmean = data['Pressure3pm'].rolling(1000).mean()

plot1 = dataSpec1['Pressure3pm'].head().plot(ax=axes[4][0],sharex=True, label='Pressure@3pm for Rain')

plot2 = dataSpec2['Pressure3pm'].head().plot(ax=axes[4][0],sharex=True, label='Pressure@3pm for no Rain')

plot3 = mtmean.plot(ax=axes[4][0],sharex=True, label='Mean Pressure@3pm')

plot1.legend(loc="upper right")

axes[4][0].set_ylabel('Pressure@3pm [hpa]')



dataSpec1 = data[data['RainTomorrow']==0].groupby('Temp9am')

dataSpec2 = data[data['RainTomorrow']==1].groupby('Temp9am')

mtmean = data['Temp9am'].rolling(1000).mean()

plot1 = dataSpec1['Temp9am'].head().plot(ax=axes[4][1],sharex=True, label='Temp@9am for Rain')

plot2 = dataSpec2['Temp9am'].head().plot(ax=axes[4][1],sharex=True, label='Temp@9am for no Rain')

plot3 = mtmean.plot(ax=axes[4][1],sharex=True, label='Mean Temp@9am')

plot1.legend(loc="upper right")

axes[4][1].set_ylabel('Temp@9am [$^o$C]')



dataSpec1 = data[data['RainTomorrow']==0].groupby('Temp3pm')

dataSpec2 = data[data['RainTomorrow']==1].groupby('Temp3pm')

mtmean = data['Temp3pm'].rolling(1000).mean()

plot1 = dataSpec1['Temp3pm'].head().plot(ax=axes[5][0],sharex=True, label='Temp@3pm for Rain')

plot2 = dataSpec2['Temp3pm'].head().plot(ax=axes[5][0],sharex=True, label='Temp@3pm for no Rain')

plot3 = mtmean.plot(ax=axes[5][0],sharex=True, label='Mean Temp@3pm')

plot1.legend(loc="upper right")

axes[5][0].set_ylabel('Temp@3pm [$^o$C]')



dataSpec1 = data[data['RainTomorrow']==0].groupby('RISK_MM')

dataSpec2 = data[data['RainTomorrow']==1].groupby('RISK_MM')

mtmean = data['RISK_MM'].rolling(1000).mean()

plot1 = dataSpec1['RISK_MM'].head().plot(ax=axes[5][1],sharex=True, label='RISK_MM for Rain')

plot2 = dataSpec2['RISK_MM'].head().plot(ax=axes[5][1],sharex=True, label='RISK_MM for no Rain')

plot3 = mtmean.plot(ax=axes[5][1],sharex=True, label='Mean RISK_MM')

plot1.legend(loc="upper right")

axes[5][1].set_ylabel('Amount of Rain')
#Correlation matrix & Heatmap - Finding correlation

pl.figure(figsize =(15,15))

corrmat = data.corr()

sns.heatmap(corrmat, annot=True, fmt='.2f', vmin=0, vmax=1, square=True);

plt.show()
#dropping uncorrelated columns

data=data.drop(columns=['MinTemp','WindGustDir', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm', 'Temp9am', 'RISK_MM'])

data.head(5)
#Labels and featureSet columns

columns = data.columns.tolist()

columns = [c for c in columns if c not in ['RainToday', 'RainTomorrow']]

target1 = 'RainToday'

target2 = 'RainTomorrow'



X = data[columns]

y1 = data[target1]

y2 = data[target2]
#Splitting data into training and testing sets for RainToday & RainTomorrow

X1_train, X1_test, y1_train, y1_test = train_test_split(X, y1, test_size=0.25, random_state=1)

X2_train, X2_test, y2_train, y2_test = train_test_split(X, y2, test_size=0.25, random_state=1)



print("Training FeatureSet (RainToday):", X1_train.shape)

print("Training Labels (RainToday):", y1_train.shape)

print("Testing FeatureSet (RainToday):", X1_test.shape)

print("Testing Labels (RainToday):", y1_test.shape)

print("\n")

print("Training FeatureSet (RainTomorrow):", X2_train.shape)

print("Training Labels (RainTomorrow):", y2_train.shape)

print("Testing FeatureSet (RainTomorrow):", X2_test.shape)

print("Testing Labels (RainTomorrow):", y2_test.shape)
#Initializing the model with some parameters for predicting rain today.

model = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, random_state=1)

#Fitting the model to the data.

model.fit(X1_train, y1_train)

#Generating predictions for the test set.

predictions = model.predict(X1_test)

#Computing the Model Accuracy

print("Random Forrest Accuracy (Rain Today):",round(metrics.accuracy_score(y1_test, predictions),2)*100)

#Computing the error.

print("Mean Absoulte Error:", round(mean_absolute_error(predictions, y1_test),2)*100)

#Computing classification Report

print("Classification Report:\n", classification_report(y1_test, predictions))

#Plotting confusion matrix

print("Confusion Matrix:")

df = pd.DataFrame(

    confusion_matrix(y1_test, predictions),

    index = [['actual', 'actual'], ['0','1']],

    columns = [['predicted', 'predicted'], ['0', '1']])

print(df)
#Initializing the model with some parameters for predicting rain tomorrow.

model = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, random_state=1)

#Fitting the model to the data.

model.fit(X2_train, y2_train)

#Generating predictions for the test set.

predictions = model.predict(X2_test)

#Computing the Model Accuracy

print("Random Forrest Accuracy (Rain Tomorrow):",round(metrics.accuracy_score(y2_test, predictions),2)*100)

#Computing the error.

print("Mean Absoulte Error:", round(mean_absolute_error(predictions, y2_test),2)*100)

#Computing classification Report

print("Classification Report:\n", classification_report(y2_test, predictions))

#Plotting confusion matrix

print("Confusion Matrix:")

df = pd.DataFrame(

    confusion_matrix(y2_test, predictions),

    index = [['actual', 'actual'], ['0','1']],

    columns = [['predicted', 'predicted'], ['0', '1']])

print(df)