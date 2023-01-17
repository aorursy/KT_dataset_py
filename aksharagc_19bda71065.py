#Calling necessary libraries, pandas for reading, os for interacting with operating system, f1 and cross validation score for checking and also importing train and test split.
import pandas as pd 
import os
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
#reading train data and test data
folder="C:/Users/Akshara/Downloads/bda-2019-ml-test/"
File=os.listdir(folder)
File
# x is all the columns except time index and flag and y is just the flag column. So i separated dependent in variable y and independent in variable x.
dataset = pd.read_csv(folder+File[2])
dataset.head()
# fetching all the datatypes of the columns
dataset.info()
#checking for missing values
dataset.isnull().sum()
#checking for the statistics summary of the dataset
dataset.describe()
# x contains all the independent variables and y contains just one column, that is flag
X = dataset[['timeindex', 'currentBack', 'motorTempBack', 'positionBack','refPositionBack', 'refVelocityBack', 'trackingDeviationBack','velocityBack', 'currentFront', 'motorTempFront', 'positionFront','refPositionFront', 'refVelocityFront', 'trackingDeviationFront','velocityFront']]
y = dataset[['flag']]
# splitting the data into train and test, with the test size being 30% of the train data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)
#importing random forest classifier with an number of estimatores as 10, random state as 30.
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10,oob_score = 'TRUE',criterion='entropy',max_features='auto',random_state=30,n_jobs=-1)
classifier.fit(X_train, y_train)
# assigning a variable to prediction
y_pred = classifier.predict(X_test)
#importing classification report for checking precision, recall and average score
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
#calling f1 score
f1_score(y_pred,y_test)
#importing cross val score and estimating the accuracy
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
#checking f1 score
from sklearn.metrics import f1_score
#calling the test data to perform 
Test_Data = pd.read_csv(folder+File[1])
#predicting the anomaly flag 
Test_Data['Anomaly_Flag'] = classifier.predict(Test_Data)
# submitting the sample and checking the accuracy
Sample_Submission = pd.read_csv(folder+File[0])
Sample_Submission['flag'] = Test_Data['Anomaly_Flag']
Sample_Submission.to_csv(folder+File[0],index=False)

