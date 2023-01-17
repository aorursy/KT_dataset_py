#The following codes leverages Machine Learning techniques to

#analyse a training set of telematics data and classify whether a driver is driving dangerously



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import gc #garbage collector



#this command enables auto prediction of commands

%config IPCompleter.greedy=True
#read in the first set of training data.

df_data0=pd.read_csv('../input/features-dataset0/part-00000-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv')



#load the training set of labels

df_Labels=pd.read_csv('../input/labels-dataset/part-00000-e9445087-aa0a-433b-a7f6-7f4c19d78ad6-c000.csv')
#check to see if there are any null values in the df_data0 dataset.

#if so then drop these rows of data, as they could affect the performance of the model training

if (df_data0.isnull().sum().sum()!=0):

    df_data0.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
#check if the Labels dataset have all unique BookingIDs.

#If there are duplicated BookingIDs, then drop these rows

#as they will confuse training of the classification algo.



if ((len(df_Labels))!=(len(df_Labels['bookingID'].unique()))):

    duplicateRows_Labels=df_Labels[df_Labels.duplicated(['bookingID'],keep=False)]

    for x in range(len(duplicateRows_Labels)):

        df_Labels.drop(duplicateRows_Labels.index[x],inplace=True)
#confirm that these duplicated erroneous rows in Labels dataset have been dropped

duplicateRows_Labels=df_Labels[df_Labels.duplicated(['bookingID'],keep=False)]
#take a peek to ensure no more duplicate labels

duplicateRows_Labels
#we merge the Labels dataset into the Features dataset, using bookingID as the key to merge both datasets

df_Combined_Dataset=pd.merge(df_data0, df_Labels, on='bookingID',

         left_index=True, right_index=False, sort=False)
#show the column headings in the merged dataset

df_Combined_Dataset.columns
#take a peek at the Combined Dataset

df_Combined_Dataset.head()
#We split the combined dataset into its constituent independent and dependent variables

#BookingID is deemed not relevant in the safety analysis and is not included as indepdendent variables list

iv=df_Combined_Dataset[['Accuracy', 'Bearing', 'acceleration_x', 'acceleration_y','acceleration_z', 'gyro_x', 'gyro_y', 'gyro_z', 'second', 'Speed']]

dv=df_Combined_Dataset[['label']]
#Perform feature scaling to normalise all variabls to comparable scales so that 

#the analysis will not be skewed by certain variables taking on large values.

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

iv[['Accuracy', 'Bearing', 'acceleration_x', 'acceleration_y','acceleration_z', 'gyro_x', 'gyro_y', 'gyro_z', 'second', 'Speed']] = sc.fit_transform(iv[['Accuracy', 'Bearing', 'acceleration_x', 'acceleration_y','acceleration_z', 'gyro_x', 'gyro_y', 'gyro_z', 'second', 'Speed']])
#Use Logistic Regression classification technique

#Apply Recursive Feature Elimination (RFE) method for automatic feature selection to remove unimportant features

from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import RFE



log_reg=LogisticRegression(random_state=1)

#log_reg.fit(iv_train,dv_train)

#log_reg.predict(iv_test)



# create the RFE model for the svm classifier 

# and select attributes

rfe = RFE(log_reg, 7)

rfe = rfe.fit(iv, dv)



# print summaries for the selection of attributes

print(rfe.support_)

print(rfe.ranking_)
idx=iv.columns
idx
#reduced list of features should be

#'Accuracy', 'Bearing', 'acceleration_y', 'acceleration_z', 'gyro_x', 'second', 'Speed'

reduced_features=[]

reduced_features_withKey=['bookingID']



for i in range(len(rfe.ranking_)):

    if (rfe.ranking_[i]==1):

        reduced_features.append(idx[i])

        reduced_features_withKey.append(idx[i])
reduced_features
reduced_features_withKey
#Now that we have completed features selection, we will create the full training dataset

#using the reduced list of features



#read in the training data subsets based on reduced features.

df_data0=pd.read_csv('../input/features-dataset0/part-00000-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv',usecols=reduced_features_withKey)

df_data1=pd.read_csv('../input/features-dataset1/part-00001-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv',usecols=reduced_features_withKey)

df_data2=pd.read_csv('../input/features-dataset2/part-00002-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv',usecols=reduced_features_withKey)

df_data3=pd.read_csv('../input/features-dataset3/part-00003-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv',usecols=reduced_features_withKey)

df_data4=pd.read_csv('../input/features-dataset4/part-00004-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv',usecols=reduced_features_withKey)

df_data5=pd.read_csv('../input/features-dataset5/part-00005-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv',usecols=reduced_features_withKey)

df_data6=pd.read_csv('../input/features-dataset6/part-00006-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv',usecols=reduced_features_withKey)

df_data7=pd.read_csv('../input/features-dataset7/part-00007-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv',usecols=reduced_features_withKey)

df_data8=pd.read_csv('../input/features-dataset8/part-00008-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv',usecols=reduced_features_withKey)

df_data9=pd.read_csv('../input/features-dataset9/part-00009-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv',usecols=reduced_features_withKey)



#combine the reduced feature data subsets into the full training dataset

df_ReducedFeatures_Dataset=pd.concat([df_data0,df_data1,df_data2,df_data3,df_data4,df_data5,df_data6,df_data7,df_data8,df_data9],axis=0)



#release memory from holding all the data subsets.

del [[df_data0, df_data1, df_data2, df_data3, df_data4, df_data5, df_data6, df_data7, df_data8, df_data9]]

gc.collect()

df_data0=pd.DataFrame()

df_data1=pd.DataFrame()

df_data2=pd.DataFrame()

df_data3=pd.DataFrame()

df_data4=pd.DataFrame()

df_data5=pd.DataFrame()

df_data6=pd.DataFrame()

df_data7=pd.DataFrame()

df_data8=pd.DataFrame()

df_data9=pd.DataFrame()
#check to see if there are any null values in the df_data0 dataset.

#if so then drop these rows of data, as they could affect the performance of the model training

if (df_ReducedFeatures_Dataset.isnull().sum().sum()!=0):

    df_ReducedFeatures_Dataset.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
#take a peek at the full dataset of reduced features

df_ReducedFeatures_Dataset.head()
len(df_ReducedFeatures_Dataset)
#we merge the Labels dataset into the reduced features dataset, using bookingID as the key to merge both datasets

df_Combined_Dataset=pd.merge(df_ReducedFeatures_Dataset, df_Labels, on='bookingID',

         left_index=True, right_index=False, sort=False)



#release memory from holding datasets df_Labels and df_ReducedFeatures_Dataset

del [[df_Labels, df_ReducedFeatures_Dataset]]

gc.collect()

df_Labels=pd.DataFrame()

df_ReducedFeatures_Dataset=pd.DataFrame()
#show the column headings in the merged dataset

df_Combined_Dataset.columns
df_Combined_Dataset.head()
#We split the combined dataset into its constituent independent and dependent variables

#BookingID is deemed not relevant in the safety analysis and is not included as indepdendent variables list

iv=df_Combined_Dataset[reduced_features]

dv=df_Combined_Dataset[['label']]
#take a peek at the iv dataset BEFORE feature scaling

iv.head()
#Perform feature scaling to normalise all variabls to comparable scales so that 

#the analysis will not be skewed by certain variables taking on large values.

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

iv[reduced_features] = sc.fit_transform(iv[reduced_features])
#take a peek at the iv dataset AFTER feature scaling

iv.head()
from sklearn.model_selection import train_test_split



df1=df_Combined_Dataset['bookingID'] #extract the bookingID column into a temporary dataframe

iv_withBookingID=pd.concat([df1,iv], axis=1) #concatenate the bookingID into the iv set called iv_withBookingID



#split the iv set into training and test data with 80/20 split

iv_train_withBookingID,iv_test_withBookingID,dv_train,dv_test=train_test_split(iv_withBookingID,dv,test_size=0.2,random_state=0)



#release memory from holding df1 dataset.

del [[df1]]

gc.collect()

df1=pd.DataFrame()
#take a peek at the iv training set (with bookingID column)

iv_train_withBookingID.head()
#take a peek at the iv test set (with bookingID column)

iv_test_withBookingID.head()
#extract the bookingID from iv train and iv test sets,

#in preparation for sending to classification algo for training / prediction

iv_train=iv_train_withBookingID[reduced_features]

iv_test=iv_test_withBookingID[reduced_features]
#take a peek at the iv train set, to ensure bookingID column has been successfully removed.

iv_train.head()
#take a peek at the iv test set, to ensure bookingID column has been successfully removed.

iv_test.head()
#Perform Logistic Regression classification technique

from sklearn.linear_model import LogisticRegression

log_reg=LogisticRegression(random_state=1)

log_reg.fit(iv_train,dv_train)

log_reg.predict(iv_test)
#Create dataframe to store analysis results, with 3 columns - bookingID, actual labels from test set, predicted labels

Train_results=pd.DataFrame()



#store bookingID values from the test dataset into results dataframe

Train_results['bookingID']=pd.Series(range(len(iv_test_withBookingID)))

Train_results['bookingID']=iv_test_withBookingID['bookingID'].reset_index(drop=True) #if you do not put drop=True, then the actual index values will show
#store actual labels from the test dataset into results dataframe

Train_results['DangerousDriver_Actual']=dv_test['label'].reset_index(drop=True)
#store predicted labels of the test dataset into results dataframe

Train_results['DangerousDriver_Predicted']=pd.DataFrame(log_reg.predict(iv_test)).reset_index(drop=True)
#take a peek at the Train_results table to see that everything is in order

Train_results.head()
#Write the results dataframe to a csv file

Train_results.to_csv('Train_results.csv', index=False)
#apply Confusion Matrix to the Log Regression model to measure the accuracy of the predictions

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(dv_test,log_reg.predict(iv_test))

cm
TN=cm[0][0]

FP=cm[0][1]

FN=cm[1][0]

TP=cm[1][1]
#Log Regression model accuracy

LR_accuracy = round((TP+TN)/(TP+FN+TN+FP),2)



#Log Regression model precision

LR_precision = round(TP/(TP+FP),2)



#Log Regression model sensitivity

LR_sensitivity = round(TP/(TP+FN),2)



#Log Regression model F-score

LR_Fscore= 2*LR_precision*LR_sensitivity / (LR_precision+LR_sensitivity)
#display the accuracy figure of this Log Regression model

LR_accuracy
#release memory from holding all the data subsets.

del [[iv_test,Train_results,df_Combined_Dataset, iv, dv, iv_withBookingID, iv_train_withBookingID,iv_test_withBookingID,dv_train,dv_test]]



gc.collect()

iv_test=pd.DataFrame()

Train_results=pd.DataFrame()

df_Combined_Dataset=pd.DataFrame()

iv=pd.DataFrame()

dv=pd.DataFrame()

iv_withBookingID=pd.DataFrame()

iv_train_withBookingID=pd.DataFrame()

iv_test_withBookingID=pd.DataFrame()

dv_train=pd.DataFrame()

dv_test=pd.DataFrame()
#*********************************

#END of CODES

#*********************************
#*********************************

#Codes For Prediction on other Hold-out GRAB_TEST Data

#*********************************
#read in the test dataset.

#df_test=pd.read_csv('..\\XXXX.csv')



#GRAB_TEST_iv=df_test[reduced_features]

#GRAB_TEST_dv=df_test['label']
#Perform feature scaling on actual test set to normalise all variabls to comparable scales so that 

#the analysis will not be skewed by certain variables taking on large values.



#from sklearn.preprocessing import StandardScaler

#sc = StandardScaler()

#GRAB_TEST_iv[reduced_features] = sc.fit_transform(GRAB_TEST_iv[reduced_features])
#from sklearn.model_selection import train_test_split



#df1=df_test['bookingID'] #extract the bookingID column into a temporary dataframe

#GRAB_TEST_iv_withBookingID=pd.concat([df1,GRAB_TEST_iv], axis=1) #concatenate the bookingID into the GRAB_TEST_iv, call it GRAB_TEST_iv_withBookingID



#split the iv set into training and test data with 80/20 split

#GRAB_TEST_iv_train_withBookingID,GRAB_TEST_iv_test_withBookingID,GRAB_TEST_dv_train,GRAB_TEST_dv_test=train_test_split(GRAB_TEST_iv_withBookingID,GRAB_TEST_dv,test_size=0.2,random_state=0)
#extract the bookingID from GRAB_TEST_iv train and GRAB_TEST_iv test sets,

#in preparation for sending to classification algo for training / prediction

#GRAB_TEST_iv_train=GRAB_TEST_iv_train_withBookingID[reduced_features]

#GRAB_TEST_iv_test=GRAB_TEST_iv_test_withBookingID[reduced_features]
#Perform Logistic Regression classification technique

#from sklearn.linear_model import LogisticRegression

#log_reg=LogisticRegression(random_state=1)

#log_reg.fit(GRAB_TEST_iv_train,GRAB_TEST_dv_train)

#log_reg.predict(GRAB_TEST_iv_test)
#Create dataframe to store analysis results, with 3 columns - bookingID, actual labels from test set, predicted labels

#GRAB_TEST_results=pd.DataFrame()



#store bookingID values from the test dataset into results dataframe

#GRAB_TEST_results['bookingID']=pd.Series(range(len(GRAB_TEST_iv_test_withBookingID)))

#GRAB_TEST_results['bookingID']=GRAB_TEST_iv_test_withBookingID['bookingID'].reset_index(drop=True) #if you do not put drop=True, then the actual index values will show
#store actual labels from the test dataset into results dataframe

#GRAB_TEST_results['DangerousDriver_Actual']=GRAB_TEST_dv_test['label'].reset_index(drop=True)
#store predicted labels of the test dataset into results dataframe

#GRAB_TEST_results['DangerousDriver_Predicted']=pd.DataFrame(log_reg.predict(GRAB_TEST_iv_test)).reset_index(drop=True)
#*********************************

#END of CODES for GRAB_TEST data predictions on Hold-out Test Data

#*********************************