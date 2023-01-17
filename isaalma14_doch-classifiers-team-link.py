#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 17:41:28 2018

@author: machinelearning
"""

# Importing the libraries
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import time

### for python 3.6:
import _pickle as cPickle
### for python 2.7:
#import cPickle

timestart = time.time()


# Importing the dataset
data = pd.read_csv('../input/doch-dataset-modif-team-link/TPSDD.csv', dtype={'Project_ID':str,
 'School_ID': str, 'Teacher_ID': str, 'Project_Type': str, 
 'Project_Subject_Category_Tree': str, 'Project_Subject_Subcategory_Tree': str,
 'Project_Grade_Level_Category': str, 'Project_Resource_Category': str, 
 'Project_Cost': np.float64, 'Project_Expiration_Date': str, 
 'Is_teachers_first_project': np.int32, 
 'Teacher_Prefix': str, 'School_Name': str, 
 'School_Metro_Type': str, 'School_State':str, 'School_Zip': str, 
 'School_City': str, 'School_County': str, 'School_District': str, 
 'School_Percentage_Free_Lunch': np.int32, 'Donor_ID': str, 
 'Donation_Included_Optional_Donation': str, 'Donation_Amount': np.float64,
 'Donor_Cart_Sequence': np.int32, 'Donation_Received_Date':str, 
 'Type_of_Donor': str, 
 'Donor_City': str, 'Donor_State': str, 
 'Donor_Is_Teacher': str, 'Donor_Zip': str},
 parse_dates=['Project_Expiration_Date', 'Donation_Received_Date'])

#All the column names (30):
#'Project_ID','School_ID','Teacher_ID','Project_Type',
#'Project_Subject_Category_Tree','Project_Subject_Subcategory_Tree',
#'Project_Grade_Level_Category','Project_Resource_Category','Project_Cost',
#'Project_Expiration_Date','Is_teachers_first_project',
#'Teacher_Prefix','School_Name','School_Metro_Type','School_State',
#'School_Zip','School_City','School_County','School_District',
#'School_Percentage_Free_Lunch','Donor_ID',
#'Donation_Included_Optional_Donation','Donation_Amount','Donor_Cart_Sequence',
#'Donation_Received_Date','Type_of_Donor',
#'Donor_City','Donor_State','Donor_Is_Teacher','Donor_Zip'

data = data.iloc[np.random.permutation(len(data))]
data = data.reset_index(drop=True)

#iddataset = data[['Project_ID','School_ID','Teacher_ID','Donor_ID']]
data = data.drop(['Project_ID','School_ID','Teacher_ID',
                        'Donor_ID'], axis=1)

data['Time_Bf_Exp'] = data['Project_Expiration_Date']-data['Donation_Received_Date']
data['Time_Bf_Exp'] = (data['Time_Bf_Exp'] / np.timedelta64(1, 'D')).astype(int)

# List of the chosen features 1:
flist1 = ['School_City','School_County','School_Zip','School_State',
         'School_District','Donor_State','Donor_City',
         'School_Metro_Type','Teacher_Prefix','Project_Grade_Level_Category',
         'Project_Resource_Category',#'Project_Subject_Subcategory_Tree',
         #'School_Name',
         'Project_Cost','Donor_Cart_Sequence','School_Percentage_Free_Lunch',
         'Donation_Amount','Time_Bf_Exp']

# The chosen label 1:
lab1 = ['Project_Subject_Category_Tree']
dataset = data[flist1+lab1]
l=len(dataset.columns)-1
X = dataset.iloc[:,[x for x in range(0,l,1)]].values
y = dataset.iloc[:, l].values


labelencoder_X_0 = LabelEncoder()
X[:, 0] = labelencoder_X_0.fit_transform(X[:, 0])
np.save('classesX_0.npy', labelencoder_X_0.classes_)
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
np.save('classesX_1.npy', labelencoder_X_1.classes_)
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
np.save('classesX_2.npy', labelencoder_X_2.classes_)
labelencoder_X_3 = LabelEncoder()
X[:, 3] = labelencoder_X_3.fit_transform(X[:, 3])
np.save('classesX_3.npy', labelencoder_X_3.classes_)
labelencoder_X_4 = LabelEncoder()
X[:, 4] = labelencoder_X_4.fit_transform(X[:, 4])
np.save('classesX_4.npy', labelencoder_X_4.classes_)
labelencoder_X_5 = LabelEncoder()
X[:, 5] = labelencoder_X_5.fit_transform(X[:, 5])
np.save('classesX_5.npy', labelencoder_X_5.classes_)
labelencoder_X_6 = LabelEncoder()
X[:, 6] = labelencoder_X_6.fit_transform(X[:, 6])
np.save('classesX_6.npy', labelencoder_X_6.classes_)
labelencoder_X_7 = LabelEncoder()
X[:, 7] = labelencoder_X_7.fit_transform(X[:, 7])
np.save('classesX_7.npy', labelencoder_X_7.classes_)
labelencoder_X_8 = LabelEncoder()
X[:, 8] = labelencoder_X_8.fit_transform(X[:, 8])
np.save('classesX_8.npy', labelencoder_X_8.classes_)
labelencoder_X_9 = LabelEncoder()
X[:, 9] = labelencoder_X_9.fit_transform(X[:, 9])
np.save('classesX_9.npy', labelencoder_X_9.classes_)
labelencoder_X_10 = LabelEncoder()
X[:, 10] = labelencoder_X_10.fit_transform(X[:, 10])
np.save('classesX_10.npy', labelencoder_X_10.classes_)

# List of the chosen features 2:
flist2 = ['School_City','School_County','School_Zip','School_State',
         'School_District','Donor_State','Donor_City',
         'School_Metro_Type','Teacher_Prefix','Project_Subject_Category_Tree',
         'Project_Grade_Level_Category',#'Project_Subject_Subcategory_Tree',
         #'School_Name',
         'Project_Cost','Donor_Cart_Sequence','School_Percentage_Free_Lunch',
         'Donation_Amount','Time_Bf_Exp']
# The chosen label 2:
lab2 = ['Project_Resource_Category']
dataset2 = data[flist2+lab2]
l2=len(dataset2.columns)-1
X2 = dataset2.iloc[:,[x for x in range(0,l2,1)]].values
y2 = dataset2.iloc[:, l2].values

labelencoder_X2_0 = LabelEncoder()
X2[:, 0] = labelencoder_X2_0.fit_transform(X2[:, 0])
np.save('classesX2_0.npy', labelencoder_X2_0.classes_)
labelencoder_X2_1 = LabelEncoder()
X2[:, 1] = labelencoder_X2_1.fit_transform(X2[:, 1])
np.save('classesX2_1.npy', labelencoder_X2_1.classes_)
labelencoder_X2_2 = LabelEncoder()
X2[:, 2] = labelencoder_X2_2.fit_transform(X2[:, 2])
np.save('classesX2_2.npy', labelencoder_X2_2.classes_)
labelencoder_X2_3 = LabelEncoder()
X2[:, 3] = labelencoder_X2_3.fit_transform(X2[:, 3])
np.save('classesX2_3.npy', labelencoder_X2_3.classes_)
labelencoder_X2_4 = LabelEncoder()
X2[:, 4] = labelencoder_X2_4.fit_transform(X2[:, 4])
np.save('classesX2_4.npy', labelencoder_X2_4.classes_)
labelencoder_X2_5 = LabelEncoder()
X2[:, 5] = labelencoder_X2_5.fit_transform(X2[:, 5])
np.save('classesX2_5.npy', labelencoder_X2_5.classes_)
labelencoder_X2_6 = LabelEncoder()
X2[:, 6] = labelencoder_X2_6.fit_transform(X2[:, 6])
np.save('classesX2_6.npy', labelencoder_X2_6.classes_)
labelencoder_X2_7 = LabelEncoder()
X2[:, 7] = labelencoder_X2_7.fit_transform(X2[:, 7])
np.save('classesX2_7.npy', labelencoder_X2_7.classes_)
labelencoder_X2_8 = LabelEncoder()
X2[:, 8] = labelencoder_X2_8.fit_transform(X2[:, 8])
np.save('classesX2_8.npy', labelencoder_X2_8.classes_)
labelencoder_X2_9 = LabelEncoder()
X2[:, 9] = labelencoder_X2_9.fit_transform(X2[:, 9])
np.save('classesX2_9.npy', labelencoder_X2_9.classes_)
labelencoder_X2_10 = LabelEncoder()
X2[:, 10] = labelencoder_X2_10.fit_transform(X2[:, 10])
np.save('classesX2_10.npy', labelencoder_X2_10.classes_)
labelencoder_X2_11 = LabelEncoder()


# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size = 0.25, random_state = 0)


# Fitting Random Forest Classification to the Training set
print('Training the classifiers...')
classifier = RandomForestClassifier(n_estimators = 14,
                                    min_samples_leaf=5, random_state = 0)
classifier.fit(X_train, y_train)

classifier2 = RandomForestClassifier(n_estimators = 14,
                                    min_samples_leaf=5, random_state = 0)
classifier2.fit(X2_train, y2_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y2_pred = classifier2.predict(X2_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
cm2 = confusion_matrix(y2_test, y2_pred)
print(cm2)

# Feature Importance
feature_imp = pd.DataFrame(classifier.feature_importances_,
                           index = flist1,
                           columns=['importance']).sort_values('importance',
                                   ascending=False)
with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
    print(feature_imp)

feature_imp2 = pd.DataFrame(classifier2.feature_importances_,
                           index = flist2,
                           columns=['importance']).sort_values('importance',
                                   ascending=False)
with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
    print(feature_imp2)

# Results:
print(classification_report(y_test, y_pred))
print('\n')
print(classification_report(y2_test, y2_pred))
print('\n')

 
# Save the classifiers

#with open('DoCh_classifier_Subject.pkl', 'wb') as fid:
#    cPickle.dump(classifier, fid)    

#with open('DoCh_classifier_Resource.pkl', 'wb') as fid:
#    cPickle.dump(classifier2, fid)

timeend = time.time()
print('\n----------------------------------------------')
print(' Run time of the script:')
print('  %d minutes ' % ((timeend - timestart)/60))
print('----THE END----\n ')
