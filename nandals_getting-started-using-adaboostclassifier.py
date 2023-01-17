# Loading the required python modules

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
# Reading Training Data
trainDataCsvFilepath = '../input/train/train.csv'
trainDataFrame = pd.read_csv(trainDataCsvFilepath)

# Reading Test Data
testDataCsvFilepath = '../input/test/test.csv'
testDataFrame = pd.read_csv(testDataCsvFilepath)
# Features List which are being targeted
featuresList = ['Age','Health','Vaccinated','Dewormed','Sterilized','PhotoAmt','Gender','Breed1','Breed2','Color1','Color2','Fee','MaturitySize']
x = trainDataFrame[featuresList]
y = trainDataFrame.AdoptionSpeed

# DataFrame for testing the Prediction Model
test_x = testDataFrame[featuresList]
#Prediction using AdaBoostClassifier
clf = AdaBoostClassifier()

# Load Training Dataset into the Classifier
clf.fit(x, y)
pred = pd.DataFrame()
pred['PetID'] = testDataFrame['PetID']

# Generate Prediction on Test Dataset using the Trained Model
pred['AdoptionSpeed'] = clf.predict(test_x)

# Saving Predicitions to submission.csv file
pred.set_index('PetID').to_csv("submission.csv", index=True)