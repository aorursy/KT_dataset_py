#importing all libraries for analysis,visualisations,model fitting and model evaluation

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from scipy.stats import norm
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


# Loading the training file from the folder.

dataset = pd.read_csv("../input/bda-2019-ml-test/Train_Mask.csv")
dataset.head()
#checking the type of the variable and basic statistics

dataset.info()
dataset.describe()
#Setting the dependent and independent variable

X = dataset[['timeindex', 'currentBack', 'motorTempBack', 'positionBack','refPositionBack', 'refVelocityBack', 'trackingDeviationBack','velocityBack', 'currentFront', 'motorTempFront', 'positionFront','refPositionFront', 'refVelocityFront', 'trackingDeviationFront','velocityFront']]
y = dataset[['flag']]
#splitting the data into train and test with a test size of 20%

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20,random_state=0)
#Color palette
myPal = ["#1E90FF", "#FFD700", "#00EEEE", "#668B8B", "#EAEAEA", "#FF3030"]
sns.set_palette(myPal)
sns.palplot(sns.color_palette())
#basic visualisations for categorical variables

l= ['flag',  'motorTempBack', 'motorTempFront']
for i in l:
    plt.figure()
    sns.countplot(x=i, data=dataset)
#distribution of continuous variables

l= [dataset['timeindex'], dataset['positionBack'], dataset['refPositionBack'], dataset['refVelocityBack'], dataset['trackingDeviationBack'],dataset['velocityBack'],dataset['velocityFront']]
for i in l:
    plt.figure(figsize=(11,6))
    sns.distplot(i, fit=norm, kde=False)
#fitting the model

classifier = RandomForestClassifier(n_estimators=100,oob_score = 'TRUE',criterion='entropy',max_features='auto',random_state=30,n_jobs=-1)
classifier.fit(X_train, y_train)
#predicting with X_test and checking the f1_score

y_pred = classifier.predict(X_test)
f1_score(y_pred,y_test)
#classification report of the model

print(classification_report(y_test,y_pred))
#Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
#using 10-fold cross validation 
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
#opening the test file from the folder
Test_Data = pd.read_csv("../input/bda-2019-ml-test/Test_Mask_Dataset.csv")
#making a column anomaly flag with the predicted test_data values
Test_Data['AnomalyFlag'] = classifier.predict(Test_Data)
#opening the sample submission file and replacing it with our predicted values
Submission = pd.read_csv("../input/bda-2019-ml-test/Sample Submission.csv")
Submission['flag'] = Test_Data['AnomalyFlag']
Submission.to_csv("Sample Submission.csv",index=False)
