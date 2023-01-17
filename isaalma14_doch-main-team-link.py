#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 15:36:55 2018

@author: machinelearning
"""

import time
import numpy as np
import pandas as pd
#import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

### for python 3.6:
import _pickle as cPickle
### for python 2.7:
#import cPickle

timestart = time.time()

# Call the modified data file 
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

# Also call the projects
projects = pd.read_csv(('../input/io/Projects.csv'), error_bad_lines=False, 
                       warn_bad_lines=False, 
                       parse_dates=["Project Posted Date",
                                    "Project Fully Funded Date", 
                                    "Project Expiration Date"])

# Call a file with new projects that will be suggestet to the donors 
nprojects = pd.read_csv('../input/io/Projects.csv', error_bad_lines=False, 
                       warn_bad_lines=False, 
                       parse_dates=["Project Posted Date",
                                    "Project Fully Funded Date", 
                                    "Project Expiration Date"])
nprojects = nprojects.iloc[0:1000, :]
# load the classifiers

#with open('DoCh_classifier_Subject.pkl', 'rb') as fid:
#    loaded_classifier = cPickle.load(fid) 
#with open('DoCh_classifier_Resource.pkl', 'rb') as fid2:
#    loaded_classifier2 = cPickle.load(fid2) 


# Preparation of the data
iddataset = data[['Project_ID','Donor_ID']]

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
dataset1 = data[flist1]
X = dataset1.values

labelencoder_X_0 = LabelEncoder()
labelencoder_X_0.classes_ = np.load('../input/doch-classifiers-team-link/classesX_0.npy')
X[:, 0] = labelencoder_X_0.transform(X[:, 0])
labelencoder_X_1 = LabelEncoder()
labelencoder_X_1.classes_ = np.load('../input/doch-classifiers-team-link/classesX_1.npy')
X[:, 1] = labelencoder_X_1.transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
labelencoder_X_2.classes_ = np.load('../input/doch-classifiers-team-link/classesX_2.npy')
X[:, 2] = labelencoder_X_2.transform(X[:, 2])
labelencoder_X_3 = LabelEncoder()
labelencoder_X_3.classes_ = np.load('../input/doch-classifiers-team-link/classesX_3.npy')
X[:, 3] = labelencoder_X_3.transform(X[:, 3])
labelencoder_X_4 = LabelEncoder()
labelencoder_X_4.classes_ = np.load('../input/doch-classifiers-team-link/classesX_4.npy')
X[:, 4] = labelencoder_X_4.transform(X[:, 4])
labelencoder_X_5 = LabelEncoder()
labelencoder_X_5.classes_ = np.load('../input/doch-classifiers-team-link/classesX_5.npy')
X[:, 5] = labelencoder_X_5.transform(X[:, 5])
labelencoder_X_6 = LabelEncoder()
labelencoder_X_6.classes_ = np.load('../input/doch-classifiers-team-link/classesX_6.npy')
X[:, 6] = labelencoder_X_6.transform(X[:, 6])
labelencoder_X_7 = LabelEncoder()
labelencoder_X_7.classes_ = np.load('../input/doch-classifiers-team-link/classesX_7.npy')
X[:, 7] = labelencoder_X_7.transform(X[:, 7])
labelencoder_X_8 = LabelEncoder()
labelencoder_X_8.classes_ = np.load('../input/doch-classifiers-team-link/classesX_8.npy')
X[:, 8] = labelencoder_X_8.transform(X[:, 8])
labelencoder_X_9 = LabelEncoder()
labelencoder_X_9.classes_ = np.load('../input/doch-classifiers-team-link/classesX_9.npy')
X[:, 9] = labelencoder_X_9.transform(X[:, 9])
labelencoder_X_10 = LabelEncoder()
labelencoder_X_10.classes_ = np.load('../input/doch-classifiers-team-link/classesX_10.npy')
X[:, 10] = labelencoder_X_10.transform(X[:, 10])


# List of the chosen features 2:
flist2 = ['School_City','School_County','School_Zip','School_State',
         'School_District','Donor_State','Donor_City',
         'School_Metro_Type','Teacher_Prefix','Project_Subject_Category_Tree',
         'Project_Grade_Level_Category',#'Project_Subject_Subcategory_Tree',
         #'School_Name',
         'Project_Cost','Donor_Cart_Sequence','School_Percentage_Free_Lunch',
         'Donation_Amount','Time_Bf_Exp']
dataset2 = data[flist2]
X2 = dataset2.values

labelencoder_X2_0 = LabelEncoder()
labelencoder_X2_0.classes_ = np.load('../input/doch-classifiers-team-link/classesX2_0.npy')
X2[:, 0] = labelencoder_X2_0.transform(X2[:, 0])
labelencoder_X2_1 = LabelEncoder()
labelencoder_X2_1.classes_ = np.load('../input/doch-classifiers-team-link/classesX2_1.npy')
X2[:, 1] = labelencoder_X2_1.transform(X2[:, 1])
labelencoder_X2_2 = LabelEncoder()
labelencoder_X2_2.classes_ = np.load('../input/doch-classifiers-team-link/classesX2_2.npy')
X2[:, 2] = labelencoder_X2_2.transform(X2[:, 2])
labelencoder_X2_3 = LabelEncoder()
labelencoder_X2_3.classes_ = np.load('../input/doch-classifiers-team-link/classesX2_3.npy')
X2[:, 3] = labelencoder_X2_3.transform(X2[:, 3])
labelencoder_X2_4 = LabelEncoder()
labelencoder_X2_4.classes_ = np.load('../input/doch-classifiers-team-link/classesX2_4.npy')
X2[:, 4] = labelencoder_X2_4.transform(X2[:, 4])
labelencoder_X2_5 = LabelEncoder()
labelencoder_X2_5.classes_ = np.load('../input/doch-classifiers-team-link/classesX2_5.npy')
X2[:, 5] = labelencoder_X2_5.transform(X2[:, 5])
labelencoder_X2_6 = LabelEncoder()
labelencoder_X2_6.classes_ = np.load('../input/doch-classifiers-team-link/classesX2_6.npy')
X2[:, 6] = labelencoder_X2_6.transform(X2[:, 6])
labelencoder_X2_7 = LabelEncoder()
labelencoder_X2_7.classes_ = np.load('../input/doch-classifiers-team-link/classesX2_7.npy')
X2[:, 7] = labelencoder_X2_7.transform(X2[:, 7])
labelencoder_X2_8 = LabelEncoder()
labelencoder_X2_8.classes_ = np.load('../input/doch-classifiers-team-link/classesX2_8.npy')
X2[:, 8] = labelencoder_X2_8.transform(X2[:, 8])
labelencoder_X2_9 = LabelEncoder()
labelencoder_X2_9.classes_ = np.load('../input/doch-classifiers-team-link/classesX2_9.npy')
X2[:, 9] = labelencoder_X2_9.transform(X2[:, 9])
labelencoder_X2_10 = LabelEncoder()
labelencoder_X2_10.classes_ = np.load('../input/doch-classifiers-team-link/classesX2_10.npy')
X2[:, 10] = labelencoder_X2_10.transform(X2[:, 10])

# Predicting the Test set results

#Y = loaded_classifier.predict(X)
#Y2 = loaded_classifier2.predict(X2)

Y = ['Math & Science','Literacy & Language','Applied Learning','Literacy & Language',
     'Literacy & Language_Math & Science','Math & Science','Literacy & Language','Literacy & Language']
Y2 = ['Technology','Books','Other','Books','Technology','Supplies','Books','Books']

#print(data['Project_ID'])
#print(data['Donor_ID'])
#print('\npred Subject:\n')
#print(Y)
#print('\nReal Subject:\n')
#print(data['Project_Subject_Category_Tree'])
#print('\npred Resource:\n')
#print(Y2)
#print('\nReal Resource:\n')
#print(data['Project_Resource_Category'])

f= open('Emails_for_donors.txt', "w+")

for i in range(0,len(Y),1):
    f.write('\n\n\nFor donor %s : \n\n' % data['Donor_ID'][i])
    f.write("Dear Donor,\nWe are very grateful for your donation to the")
    f.write(''' " %s " project. ''' % projects[projects['Project ID']==data['Project_ID'][i]]['Project Title'].values[0])
    f.write('We thought you might be interested by these new projects that were recently posted on our site:\n')
    iprojtitlelist = nprojects[(nprojects['Project Subject Category Tree']==Y[i]) & (nprojects['Project Resource Category']==Y2[i])]['Project Title'].values
    iprojdescriptlist = nprojects[(nprojects['Project Subject Category Tree']==Y[i]) & (nprojects['Project Resource Category']==Y2[i])]['Project Short Description'].values
    if iprojtitlelist!=[]:
        for r in range(0,3,1):
            f.write('%i.- %s \n' % (r+1, iprojtitlelist[r]))
            f.write('%s \n\n' % iprojdescriptlist[r])
    else:
        iprojtitlelist = nprojects[nprojects['Project Subject Category Tree']==Y[i]]['Project Title'].values
        iprojdescriptlist = nprojects[nprojects['Project Subject Category Tree']==Y[i]]['Project Short Description'].values
        if iprojtitlelist!=[]:
            for r in range(0,3,1):
                f.write('%i.- %s \n' % (r, iprojtitlelist[r]))
                f.write('%s \n\n' % iprojdescriptlist[r])
        else:
            iprojtitlelist = nprojects[nprojects['Project Resource Category']==Y2[i]]['Project Title'].values
            iprojdescriptlist = nprojects[nprojects['Project Resource Category']==Y2[i]]['Project Short Description'].values
            if iprojtitlelist!=[]:
                for r in range(0,3,1):
                    f.write('%i.- %s \n' % (r, iprojtitlelist[r]))
                    f.write('%s \n\n' % iprojdescriptlist[r])
            else:
                print(i)
                print(iprojtitlelist)
    f.write("Best regards,\nDonors Choose")
    
f.close() 

timeend = time.time()
print('\n----------------------------------------------')
print(' Run time of the script:')
print('  %d minutes ' % ((timeend - timestart)/60))
print('----THE END----\n ')