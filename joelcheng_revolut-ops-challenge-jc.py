# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np #linear algebra

import pandas as pd #data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #data visualisation

import statsmodels.formula.api as sm #running stats functions

from datetime import datetime #date and time convertor

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
##Importing New Dataset## 

#Loads the dataset as a CSV file, "data"

data = pd.read_csv("../input/JC_Revolut Database CSV_Final.csv", low_memory = False)
#Converting 'created_at' into datetime format; changes object to string

data['created_at'] =  pd.to_datetime(data['created_at'])
##Definition of New Feature## 

#Defines Feature data as only results whose status is 'clear' or 'consider'

data = data[(data.result == 'clear') | (data.result == 'consider')] 
#Checking the shape of the dataset to determine total number of KYC attempts

data.shape
#Defines Feature docpasses as only doc results whose status is 'clear'

docpasses = data[(data.result == 'clear')]
#Determine total number of successful document passes



docpasses.shape
#Defines Feature docpasses as only face results whose status is 'clear'

facepasses = data[(data.result2 == 'clear')]
#Determine total number of successful facial passes



facepasses.shape
#Defines Feature kycpasses as attempts with both doc and face results as 'clear'

kycpasses = data[(data.result2 == 'clear') & (data.result == 'clear')] 
#Determine total number of successful KYC passes



kycpasses.shape
#Creating a copy of the main dataset for Unique IDs

data_unq=data.copy(deep=True)
#Dropping duplicate User IDs but keeping the first instance, dataset 'data_unq'

data_unq.drop_duplicates(subset ="user_id", inplace = True)

#Create a new column 'target' that gives 'True' if both conditons are met - i.e. KYC Passes

data_unq['target'] = (data_unq.result2 == 'clear') & (data_unq.result == 'clear')
#Determining the total number of unique User IDs

data_unq.shape
#Determining count of KYC fails and passes; False=fail, Pass=True

from collections import Counter



Counter(data_unq.target)
#Creating new dataset for unique KYC fails

kycfails_unq=data_unq.copy (deep=True)



#Filtering for unique KYC fails

kycfails_unq = kycfails_unq[(kycfails_unq.result2 == 'consider') | (kycfails_unq.result == 'consider')] 
plt.hist(kycfails_unq.created_at,500)
#Deep dive into Oct 2017



oct17=kycfails_unq[(kycfails_unq['created_at'] > '2017-10-01') & (kycfails_unq['created_at'] < '2017-10-31')]
plt.hist(oct17.created_at,50)


plt.hist(kycfails_unq.created_at, 50, alpha=0.5)

plt.hist(data_unq.created_at, 50, alpha=0.5)
#Define attempts from June to October

JtO_failsunq = kycfails_unq[(kycfails_unq['created_at'] > '2017-06-01') & (kycfails_unq['created_at'] < '2017-10-31')]

JtO = data_unq[(data_unq['created_at'] > '2017-06-01') & (data_unq['created_at'] < '2017-10-31')]

len(JtO_failsunq)/len(JtO)*100
jun17_failsunq = kycfails_unq[(kycfails_unq['created_at'] > '2017-06-01') & (kycfails_unq['created_at'] < '2017-06-30')]

jun17 = data_unq[(data_unq['created_at'] > '2017-06-01') & (data_unq['created_at'] < '2017-06-30')]
len(jun17_failsunq)/len(jun17)*100
aug17_failsunq = kycfails_unq[(kycfails_unq['created_at'] > '2017-08-01') & (kycfails_unq['created_at'] < '2017-08-31')]
aug17 = data_unq[(data_unq['created_at'] > '2017-08-01') & (data_unq['created_at'] < '2017-08-31')]
len(aug17_failsunq)/len(aug17)*100
oct17_failsunq = kycfails_unq[(kycfails_unq['created_at'] > '2017-10-01') & (kycfails_unq['created_at'] < '2017-10-31')]
oct17 = data_unq[(data_unq['created_at'] > '2017-10-01') & (data_unq['created_at'] < '2017-10-31')]
len(oct17_failsunq)/len(oct17)*100
Counter(data_unq.image_quality_result)
Counter(data_unq.police_record_result)
Counter(data_unq.compromised_document_result)
Counter(data_unq.facial_image_integrity_result)
Counter(data_unq.face_comparison_result)
Counter(data_unq.facial_visual_authenticity_result)
#Creating new dataset for when image quality = clear in KYC fails

kycfails_unq_imgclear=kycfails_unq.copy (deep=True)
kycfails_unq_imgclear=kycfails_unq_imgclear[(kycfails_unq_imgclear.image_quality_result == 'clear')]
len(kycfails_unq_imgclear)
Counter(kycfails_unq_imgclear.face_detection_result)
Counter(kycfails_unq_imgclear.colour_picture_result)
Counter(kycfails_unq_imgclear.visual_authenticity_result)
Counter(kycfails_unq_imgclear.data_validation_result)
Counter(kycfails_unq_imgclear.data_consistency_result)
#Creating new dataset for when image quality = unidentified or facial image intergrity = consider in KYC fails

kycfails_unq_imgp=kycfails_unq.copy (deep=True)


kycfails_unq_imgp=kycfails_unq_imgp[(kycfails_unq_imgp.image_quality_result=='unidentified')|(kycfails_unq_imgp.facial_image_integrity_result=='consider')]
plt.hist(kycfails_unq.created_at, 50, alpha=0.5)

plt.hist(kycfails_unq_imgp.created_at, 50, alpha=0.5)