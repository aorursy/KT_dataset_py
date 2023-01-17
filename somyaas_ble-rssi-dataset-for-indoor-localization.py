#Classify and predict the location of the user to receive a good signal strength.

#Building 2 different model and comparing them
#The RSSI dataset has 13 beacon readings which tells the strength of the signal, RSSI readings are represented in negative, 

#if it is –50 it represents the user is close to the ibeacon and hence the strength of signal is much strong than the value –95

#which represents the user isn't in the close proximity of the beacon and hence the strength of the signal is weak. 

#BLE RSSI labeled dataset is used to train different classifiers and analyze the performance.
#loading the csv file

beacon='../input/ble-rssi-dataset/iBeacon_RSSI_Labeled.csv'
#installing pandas packages

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

import statistics
# reading the csv file and assigning it to 'beacon' variable

beacon = pd.read_csv(beacon,decimal=",")
# display all columns in beacon

features_names=beacon.columns

features_names
#checking data types of all columns

beacon.dtypes
#checking dimension of beacon

beacon.shape
#creating a dataframe of beacon

df=pd.DataFrame(beacon)
beacon['location'].isnull().value_counts()
beacon['date'].isnull().value_counts()
beacon['b3001'].isnull().value_counts()
beacon['b3002'].isnull().value_counts()
beacon['b3003'].isnull().value_counts()
beacon['b3004'].isnull().value_counts()
beacon['b3005'].isnull().value_counts()
beacon['b3006'].isnull().value_counts()
beacon['b3007'].isnull().value_counts()
beacon['b3008'].isnull().value_counts()
beacon['b3009'].isnull().value_counts()
beacon['b3010'].isnull().value_counts()
beacon['b3011'].isnull().value_counts()
beacon['b3012'].isnull().value_counts()
beacon['b3013'].isnull().value_counts()
#calculating basic statistics

beacon.describe()
# mapping all locations column in form of histogram



loc=beacon.iloc[:,0]

loc.hist(figsize=(30,20))
# storing ibeacon readings in 'values' variable

values=beacon.iloc[:,2:]
# plotting 'values' in the form of histogram

values.hist(figsize=(20,15))
# creating a mask for ibeacon3001 values above -90 and ploting a bar graph against the locations of users

mask1=beacon['b3001']>-90

a=beacon.loc[mask1,'location'].value_counts()

a1=a.plot(kind='bar')

a1.grid()

a
# creating a mask for ibeacon3002 values above -90 and ploting a bar graph against the locations of users

mask2=beacon['b3002']>-90

a=beacon.loc[mask2,'location'].value_counts()

a1=a.plot(kind='bar',color='orange')

a1.grid()

a1.set_title('strength of beacon b3002 at particular location')

a1.set_xlabel('User location')

a1.set_ylabel('frequency')
# creating a mask for ibeacon3003 values above -90 and ploting a bar graph against the locations of users

mask3=beacon['b3003']>-90

a=beacon.loc[mask3,'location'].value_counts()

a1=a.plot(kind='bar',color='pink')

a1.grid()

a1.set_title('strength of beacon b3003 at particular location')

a1.set_xlabel('User location')

a1.set_ylabel('frequency')
# creating a mask for ibeacon3004 values above -90 and ploting a bar graph against the locations of users

mask4=beacon['b3004']>-90

a=beacon.loc[mask4,'location'].value_counts()

a1=a.plot(kind='bar')

a1.grid()

a1.set_title('strength of beacon b3004 at particular location')

a1.set_xlabel('User location')

a1.set_ylabel('frequency')
# creating a mask for ibeacon3005 values above -90 and ploting a bar graph against the locations of users

mask5=beacon['b3005']>-90

a=beacon.loc[mask5,'location'].value_counts()

a1=a.plot(kind='bar',color='green')

a1.grid()

a1.set_title('strength of beacon b3005 at particular location')

a1.set_xlabel('User location')

a1.set_ylabel('frequency')
# creating a mask for ibeacon3006 values above -90 and ploting a bar graph against the locations of users

mask6=beacon['b3006']>-90

a=beacon.loc[mask6,'location'].value_counts()

a1=a.plot(kind='bar',color='red')

a1.grid()

a1.set_title('strength of beacon b3006 at particular location')

a1.set_xlabel('User location')

a1.set_ylabel('frequency')
# creating a mask for ibeacon3007 values above -90 and ploting a bar graph against the locations of users

mask7=beacon['b3007']>-90

a=beacon.loc[mask7,'location'].value_counts()

a1=a.plot(kind='bar')

a1.grid()

a

a1.set_title('strength of beacon at particular location')

a1.set_xlabel('User location')

a1.set_ylabel('frequency')
# creating a mask for ibeacon3008 values above -90 and ploting a bar graph against the locations of users

mask8=beacon['b3008']>-90

a=beacon.loc[mask8,'location'].value_counts()

a1=a.plot(kind='bar')

a1.grid()
# creating a mask for ibeacon3009 values above -90 and ploting a bar graph against the locations of users

mask9=beacon['b3009']>-90

a=beacon.loc[mask9,'location'].value_counts()

a1=a.plot(kind='bar')

a1.grid()

a1.set_title('strength of beacon at particular location')

a1.set_xlabel('User location')

a1.set_ylabel('frequency')
# creating a mask for ibeacon3010 values above -90 and ploting a bar graph against the locations of users

mask10=beacon['b3010']>-90

a=beacon.loc[mask10,'location'].value_counts()

a1=a.plot(kind='bar')

a1.grid()

a1.set_title('strength of beacon at particular location')

a1.set_xlabel('User location')

a1.set_ylabel('frequency')
# creating a mask for ibeacon3011 values above -90 and ploting a bar graph against the locations of users

mask11=beacon['b3011']>-90

a=beacon.loc[mask11,'location'].value_counts()

a1=a.plot(kind='bar')

a1.grid()

a1.set_title('strength of beacon at particular location')

a1.set_xlabel('User location')

a1.set_ylabel('frequency')
# creating a mask for ibeacon3012 values above -90 and ploting a bar graph against the locations of users

mask12=beacon['b3012']>-90

a=beacon.loc[mask12,'location'].value_counts()

a1=a.plot(kind='bar')

a1.grid()

a1.set_title('strength of beacon at particular location')

a1.set_xlabel('User location')

a1.set_ylabel('frequency')
# creating a mask for ibeacon3013 values above -90 and ploting a bar graph against the locations of users

mask13=beacon['b3013']>-90

a=beacon.loc[mask13,'location'].value_counts()

a1=a.plot(kind='bar')

a1.grid()

a1.set_title('strength of beacon at particular location')

a1.set_xlabel('User location')

a1.set_ylabel('frequency')
# checking how every beacon is correlated with one another using 'corr' function

values=beacon.iloc[:,2:]

a=values.corr()

a
#plotting correlation coefficients vs ibeacons



fig=a.plot()

fig.set_title('Correlation between beacons')

fig.set_xlabel('Beacon')

fig.set_ylabel('Correlation coefficient')
# calculating score of input features to see importance while making predictions using ExtraTreesClassifier() method

# X stores ibeacon readings, y contains location column



X = beacon.iloc[:,2:15]

y = beacon.iloc[:,0]



from sklearn.ensemble import ExtraTreesClassifier



model = ExtraTreesClassifier()

model.fit(X,y)



print(model.feature_importances_)



# plotting graph of feature importances

feat_importances = pd.Series(model.feature_importances_, index=X.columns)

feat_importances.nlargest(10).plot(kind='barh')

plt.show()
#storing important features in x variablea and location(target variable) in y



x=beacon.iloc[:,3:8]

y=beacon.iloc[:,0]
#importing train_test_split package and  splitting data into test and training



from sklearn. model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.18,random_state=6296)
#display X_train

X_train
# checking dimensioncof X_train

X_train.shape
# display X_test

X_test
#checking dimension of X_test

X_test.shape
# display y_train

y_train
# checking dimension of y_train

y_train.shape
# display y_test

y_test
# checking dimension of y_test

y_test.shape
#applying decision tree model with gini index parameter and sample splits to be 50 and then fitting in the model



from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(criterion="gini",max_depth=35,min_samples_split=50)

fit = clf.fit(X_train, y_train)
#predicting the testing dataset



y_pre = fit.predict(X_test)

y_pre
#checking dimension of predicted

y_pre.shape
# importing confusion matrix package to see performance of our model



from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pre)

cm
# printing report giving details



from sklearn.metrics import classification_report

print(classification_report(y_test,y_pre))

#checking accuracy of predicted values

from sklearn import metrics 

print('Accuracy Score:', metrics.accuracy_score(y_test,y_pre))

# droping certain features and storing in a variable

features_names=beacon.columns.drop(['location','date','b3001','b3007','b3008','b3009','b3010','b3011','b3012','b3013'])
#plotting decision tree for visualisation 

from sklearn import tree

with open('beacon.dot', 'w') as f:

    f = tree.export_graphviz(clf, out_file=f,

feature_names=features_names,filled=True, rounded=True,

special_characters=True)
######## k fold cross validation
# importing packages

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import cross_val_score

#Decision tree model using K-fold CV; checking the best value of k and plotting for better visualization



score = []

ary = []

for x in range (2 , 27):

    ary = cross_val_score(clf, X_train, y_train, cv=x)

    score.append(max(ary[:]))

    

x = np.arange(2,27,1);

y_1 = score

plt.plot(x,y_1)

plt.xlabel('K-fold')

plt.ylabel('Score')
# K-Fold Cross Validation of Decision tree model to check how accurate our model is predicting

#Performance Analysis of K-Fold Cross Validation



kfold = KFold(n_splits=2,random_state=7)

score_arr = cross_val_score(clf, X, y, cv=kfold)



print('Decision tree model mean score (K-fold cross validation): ',score_arr.mean())

predicted = cross_val_predict(clf, X_test, y_test)

print(classification_report(y_test, predicted, digits=3))
#########################    KNN    #########################
# importing KNN model, importing package and setting parameters



from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(10,weights='distance', p=2)

fit = clf.fit(X_train, y_train)
#predicting values



y_pre = fit.predict(X_test)

y_pre
#checking dimension of predicted value array

y_pre.shape
# getting confusion matrix to see performance of our model



from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pre)

print(cm)
#printing report to see details



from sklearn.metrics import classification_report

print(classification_report(y_test,y_pre))
# calculating accuracy of the model



from sklearn import metrics 

print('Accuracy Score:', metrics.accuracy_score(y_test,y_pre))

#KNN model using K-fold CV, checking best k value

score = []

ary = []

for x in range (2 , 27):

    ary = cross_val_score(clf, X_train, y_train, cv=x)

    score.append(max(ary[:]))

    

x = np.arange(2,27,1);

y_1 = score

plt.plot(x,y_1)

plt.xlabel('K-fold')

plt.ylabel('Score')
#K-Fold Cross Validation of Decision tree model

#Performance Analysis of K-Fold Cross Validation

kfold = KFold(n_splits=2,random_state=7)

score_arr = cross_val_score(clf, X, y, cv=kfold)



print('KNN model mean score (K-fold cross validation): ',score_arr.mean())

predicted = cross_val_predict(clf, X_test, y_test)

print(classification_report(y_test, predicted, digits=3))
#we can see from the graph that the beacon3002 to b3006 have the highest value of ‘feature_importances_’, 

#hence these features are the best features as compared to others and can be further used to train and fit the model.

#After preparing and cleaning the data we explored and analyzed interesting relationships between the features of the given dataset,

#we found that beacon3002, beacon3003, beacon3003, beacon3004, beacon3005 and beacon3006 were most effective at their places and 

#giving the strong signals to the users at a particular location and that near these ibeacons most users are in proximity.

#After that, we trained our data and built 2 models to predict the location of the user based on strong readings from ibeacon.

#Decision tree gave an accurate prediction for 20.77% from the model I created, on the other hand, the KNN model predicted the values

#accurately to a 39.01%, which is approximately 16% more accurate than the decision tree.

#As the KNN model relies on the actual values and predict based on the datapoints which are close to a particular point

#we are predicting, also has capability to predict the target variable with more precision and in simpler manner 

#by calculating ‘distance’ between points makes it better and reliable model than the decision tree.

#And as the KNN method is cautious about the local input pattern and based on that makes prediction while the decision tree

#where it is sensitive to noise and can vary if minor changes are made in the model.

#Hence, in this case of dataset, I recommend K-nearest neighbor data model to classify the user’s location based on the given

#RSSI values of the ibeacon.