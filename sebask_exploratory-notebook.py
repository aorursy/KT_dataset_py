# Libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt

import seaborn as sns



# Load Data

df = pd.read_csv('../input/No-show-Issue-Comma-300k.csv')



# Clean the data into a fully numeric form

df['Gender'] = df.Gender.apply(lambda x: 0 if(x=='F') else 1)

df['Status'] = df.Status.apply(lambda x: 0 if(x=='No-Show') else 1)

#df['DayOfTheWeek'] = df.DayOfTheWeek.apply(lambda x: weekdays.index(x))

df['ApointmentData'] = pd.to_datetime(df['ApointmentData'],format='%Y-%m-%dT%H:%M:%SZ')

df['AppointmentRegistration'] = pd.to_datetime(df['AppointmentRegistration'],format='%Y-%m-%dT%H:%M:%SZ')

# df['TimeBetween'] = df['ApointmentData'] - df['AppointmentRegistration']



#Show a sample of the cleaned data

df.info()

df.head()
# Delete columns the model wont be able to process

df2 = df.copy()

del df2['AppointmentRegistration']

del df2['ApointmentData']

del df2['DayOfTheWeek']

print(df2.columns)



#Create training and testing sets

train, test = train_test_split(df2, train_size=0.8)



# Extract output variable

testY = test['Status'].copy()

del test['Status']

trainY = train['Status'].copy()

del train['Status']



# Flatten into a 1-D array

trainY = np.ravel(trainY)

testY = np.ravel(testY)



# Create random forest model

model = RandomForestClassifier(n_estimators=200)

model.fit(train, trainY)



# Evaluate model

expected = testY

predicted = model.predict(test)

print(metrics.classification_report(expected, predicted))

print(metrics.confusion_matrix(expected, predicted))



# Let the model tell us the important features and plot it

importances = model.feature_importances_

indices = np.argsort(importances)[::-1]

std = np.std([tree.feature_importances_ for tree in model.estimators_],

             axis=0)

plt.figure()

plt.title("Feature importances")

plt.bar(range(train.shape[1]), importances[indices],

       color="royalblue", yerr=std[indices], align="center")

plt.xticks(range(train.shape[1]), indices)

plt.xlim([-1, train.shape[1]])

plt.show()
# View correlation matrix

corr = df2.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(10, 220, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3,

            square=True, xticklabels=True, yticklabels=True,

            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)

plt.title('Correlation Matrix for Appointment Data')

# Keep the top 5 features, delete the rest

df3 = df2[['Sms_Reminder','Age','Tuberculosis','Gender','Alcoolism','Status']].copy()



#Create training and testing sets

train, test = train_test_split(df3, train_size=0.8)



# Extract output variable

testY = test['Status'].copy()

del test['Status']

trainY = train['Status'].copy()

del train['Status']



# Flatten into a 1-D array

trainY = np.ravel(trainY)

testY = np.ravel(testY)



# Create random forest model

model = RandomForestClassifier(n_estimators=200)

model.fit(train, trainY)



# Evaluate model

expected = testY

predicted = model.predict(test)

print(metrics.classification_report(expected, predicted))

print(metrics.confusion_matrix(expected, predicted))



# Let the model tell us the important features and plot it

importances = model.feature_importances_

indices = np.argsort(importances)[::-1]

std = np.std([tree.feature_importances_ for tree in model.estimators_],

             axis=0)

plt.figure()

plt.title("Feature importances")

plt.bar(range(train.shape[1]), importances[indices],

       color="royalblue", yerr=std[indices], align="center")

plt.xticks(range(train.shape[1]), indices)

plt.xlim([-1, train.shape[1]])

plt.show()