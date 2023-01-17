# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/test"))



# Any results you write to the current directory are saved as output.
# import modules

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from matplotlib import pyplot as plt

import pandas as pd

import numpy as np

import random

import time

# Read in csv file

adoption = pd.read_csv('../input/train/train.csv')

# Replace names and description with length of string for input

adoption['Name'] = adoption['Name'].str.len()

adoption['Description'] = adoption['Description'].str.len()

# Replace NaN with 0

adoption = adoption.fillna(0)

#Exploring data, plot number animals per age group

age = list(adoption.Age)

print (min(age))

age = [age.count(i) for i in range(min(age), max(age))]

plt.plot(age)

plt.show()

print (age[:5])

#Exploring data, plot number animals per adoption speed group

speed = list(adoption.AdoptionSpeed)

speed = [speed.count(i) for i in range(0,5)]

print(speed)

plt.bar(range(len(speed)),speed)

plt.show()
#Exploring data, plot number animals per number of photos 

photo = list(adoption.PhotoAmt)

photo = [photo.count(i) for i in range(int(min(photo)), int(max(photo)))]

plt.bar(range(len(photo)),photo)

plt.show()
# Replace with different numbers

#adoption.Type = adoption.Type.map({1:0,2:1})

# Make breed binairy, pure or not pure

breed1 = list(adoption.Breed1)

breed2 = list(adoption.Breed2)

breed = []

for i in range(len(breed1)):

    if int(breed1[i]) == 307:

        breed += [0]

    else:

        if breed2[i] != 0 and breed2[i]!=breed1[i]:

            breed += [0]

        else:

            breed += [1]
# converting panda data to lists in python

breed1 = list(adoption.Breed1)

breed2 = list(adoption.Breed2)

age = list(adoption.Age)

types = list(adoption.Type)

color = list(adoption.Color1)

color2 = list(adoption.Color2)

color3 = list(adoption.Color3)

maturity = list(adoption.MaturitySize)

photoamt = list(adoption.PhotoAmt)

speed = list(adoption.AdoptionSpeed)

gender = list(adoption.Gender)

description = list(adoption.Description)

fur = list(adoption.FurLength)

vaccinated = list(adoption.Vaccinated)

dewormed = list(adoption.Dewormed)

sterilized = list(adoption.Sterilized)

health = list(adoption.Health)

fee = list(adoption.Fee)

#health=[0 for i in health]

# Combining the features to be used by the machine learning algorithms

features = np.array([age, types, color, color2,color3,maturity, photoamt,gender, description,fur, vaccinated,dewormed,sterilized,health,fee,  breed1])

np.save('features.npy', features)

labels = np.array(speed)

np.save('labels.npy', labels)
#features = features.transpose()

features = np.load('features.npy').transpose()

labels = np.load('labels.npy').transpose()

print(features.shape)
# Split the data randomly into train and validation data

traindata, testdata, trainlabels, testlabels = train_test_split(features, labels,test_size = 0.20,random_state = 0)
# Make a random forest classifier and train it

# Here the hyperparameters can easily be changed

rf = RandomForestClassifier(random_state=0)

rf.set_params(criterion = 'entropy',max_features = 'log2',max_depth = 30, n_estimators=100)

rf.fit(traindata, trainlabels)
# To see the contribution of each feature

names = ['age', 'types', 'color', 'color2', 'color3', 'maturity', 'photoamt', 'gender', 'description', 'fur', 'vaccinated', 'dewormed', 'sterilized', 'health', 'fee',  'breed1']

contribution = rf.feature_importances_

sorting = np.argsort(contribution)[::-1]

print (len(set(breed1)))
# printing the contributions of all features

for i in sorting:

    print(names[i]+'\t'+ str(contribution[i]))
# A visual representation of the importance of features

plt.bar(range(len(names)), [contribution[i]for i in sorting])
# Predict on the validation set and create a confusion matrix

# Also make a confusion matrix that shows how a perfect score would look like 

prediction = rf.predict(testdata)

mat = np.zeros((5,5))

for i in range(len(prediction)):

    mat[testlabels[i],prediction[i]] += 1

print(mat)

plt.pcolor(mat[::-1],cmap='seismic')

plt.show()

mat2 = np.zeros((5,5))

for i in range(len(prediction)):

    mat2[testlabels[i],testlabels[i]] += 1

plt.pcolor(mat2[::-1],cmap='seismic')

plt.show()
# Experimenting with other ways of getting the results.

print("Confusion Matrix:")

print(confusion_matrix(testlabels, prediction))

print()

print("Classification Report")

print(classification_report(testlabels, prediction))

accuracy = rf.score(testdata, testlabels)

print(accuracy)
# Experiment with gradient boosting

from sklearn.ensemble import GradientBoostingClassifier



gb = GradientBoostingClassifier(n_estimators=50, learning_rate = 0.5, max_features=4, max_depth = 3, random_state = 0)

gb.fit(traindata, trainlabels)

predictions = gb.predict(testdata)



accuracy = gb.score(testdata, testlabels)

print(accuracy)

print("Confusion Matrix:")

print(confusion_matrix(testlabels, predictions))

print()

print("Classification Report")

print(classification_report(testlabels, predictions))


# Preperring the test dataset

# Read in csv file

adoption = pd.read_csv('../input/test/test.csv')

# Replace names and description with length of string for input

adoption['Name'] = adoption['Name'].str.len()

adoption['Description'] = adoption['Description'].str.len()

# Replace NaN with 0

adoption = adoption.fillna(0)

breed1 = list(adoption.Breed1)

breed2 = list(adoption.Breed2)

age = list(adoption.Age)

types = list(adoption.Type)

color = list(adoption.Color1)

color2 = list(adoption.Color2)

color3 = list(adoption.Color3)

maturity = list(adoption.MaturitySize)

photoamt = list(adoption.PhotoAmt)

#speed = list(adoption.AdoptionSpeed)

gender = list(adoption.Gender)

description = list(adoption.Description)

fur = list(adoption.FurLength)

vaccinated = list(adoption.Vaccinated)

dewormed = list(adoption.Dewormed)

sterilized = list(adoption.Sterilized)

health = list(adoption.Health)

fee = list(adoption.Fee)

#health=[0 for i in health]

# Creating the test features

test_features = np.array([age, types, color, color2,color3,maturity, photoamt,gender, description,fur, vaccinated,dewormed,sterilized,health,fee,  breed1])

test_features=test_features.transpose()
# Training on all the trainingset before making predictions on the testset

gb = GradientBoostingClassifier(n_estimators=50, learning_rate = 0.5, max_features=4, max_depth = 2, random_state = 0)

gb.fit(features, labels)

predictions = gb.predict(test_features)
# Experimenting with adding randomforest predictions as extra feature to the gradient boosting algorithm

rf = RandomForestClassifier(random_state=0)

rf.set_params(criterion = 'entropy',max_features = 'log2',max_depth = 8, n_estimators=100)

rf.fit(features, labels)



rf_predictions = rf.predict(features)

# Also write the results for test score.

read = open('../input/test/sample_submission.csv').read().split('\n')[:-1]

write = open('submission.csv', 'w')

write.write(read[0]+'\n')

for line in range(1,len(read)):

    newline = read[line].split(',')

    write.write(newline[0]+','+ str(predictions[line-1])+'\n')

write.close()
# Adding the "extra" RF feature

new_features = np.concatenate((features, rf_predictions.reshape(len(features),1)),1)

print (new_features)



# spliting the data into training and validation sets

traindata, testdata, trainlabels, testlabels = train_test_split(new_features, labels,test_size = 0.20,random_state = 0)



# training the fgradient boosting algorithm

gb = GradientBoostingClassifier(n_estimators=30, learning_rate = 0.5, max_features=3, max_depth = 3, random_state = 0)

gb.fit(traindata, trainlabels)

predictions = gb.predict(testdata)



# Test if it does betet with the extra RF predicition feature

accuracy = gb.score(testdata, testlabels)

print(accuracy)

print("Confusion Matrix:")

print(confusion_matrix(testlabels, predictions))

print()

print("Classification Report")

print(classification_report(testlabels, predictions))

# Test it on the final testset with training on complete trainingdata

gb = GradientBoostingClassifier(n_estimators=30, learning_rate = 0.5, max_features=3, max_depth = 3, random_state = 0)

gb.fit(new_features, labels)

predictions = gb.predict(test_features)



read = open('../input/test/sample_submission.csv').read().split('\n')[:-1]

write = open('submission.csv', 'w')

write.write(read[0]+'\n')

for line in range(1,len(read)):

    newline = read[line].split(',')

    write.write(newline[0]+','+ str(predictions[line-1])+'\n')

write.close()
