# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



#Import Scikit Learn Library

from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report

from statistics import mode



#Used for running SMOTE

import re

from xgboost import XGBClassifier

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/final.csv')

df.head()
#Checking the datatypes of our data before modeling

df.dtypes
#Converting all possible dates from object to datetime format

df['RECEIVED_DATE'] =  pd.to_datetime(df['RECEIVED_DATE'])

df['DECISION_DATE'] =  pd.to_datetime(df['DECISION_DATE'])

df['BEGIN_DATE'] =  pd.to_datetime(df['BEGIN_DATE'])

df['END_DATE'] =  pd.to_datetime(df['END_DATE'])



#Changing binary valued columns to categorical data (unordered)

df['FULL_TIME_POSITION'] = df['FULL_TIME_POSITION'].astype('category')

df['AGENT_REPRESENTING_EMPLOYER'] = df['AGENT_REPRESENTING_EMPLOYER'].astype('category')

df['H-1B_DEPENDENT'] = df['H-1B_DEPENDENT'].astype('category')

df['VISA_CLASS'] = df['VISA_CLASS'].astype('category')
#Examining all other remaining object dtypes

df.select_dtypes(object)
#Viewing all unique state values

df['EMPLOYER_STATE'].unique()
#Create a list of states by region

region_east = ['CONNECTICUT', 'MAINE', 'MASSACHUSETTS', 'NEW HAMPSHIRE', 'RHOSE ISLAND', 'VERMONT' 'NEW JERSEY', 'NEW YORK', 'PENNSYLVANIA']

region_midwest = ['ILLINOIS', 'INDIANA', 'MICHIGAN', 'OHIO', 'WISCONSIN', 'IOWA', 'KANSAS', 'MINNESOTA', 'MISSOURI', 'NEBRASKA', 'NORTH DAKOTA', 'SOUTH DAKOTA']

region_south = ['DELAWARE', 'FLORIDA', 'GEORGIA', 'MARYLAND', 'NORTH CAROLINA', 'SOUTH CAROLINA', 'VIRGINIA', 'DISTRICT OF COLUMBIA', 'WEST VIRGINIA', 'ALABAMA', 'KENTUCKY', 'MISSISSIPPI', 'TENNESSEE', 'ARKANSAS', 'LOUISIANA', 'OKLAHOMA', 'TEXAS']

region_west = ['ARIZONA', 'COLORADO', 'IDAHO', 'MONTANA', 'NEVADA', 'NEW MEXICO', 'UTAH', 'WYOMING', 'ALASKA', 'CALIFORNIA', 'HAWAII', 'OREGON', 'WASHINGTON']



#Create a new column EMPLOYER_REGION and select all the values from EMPLOYER_STATE column based on region

df['EMPLOYER_REGION'] = (

    np.select(

        condlist=[df['EMPLOYER_STATE'].isin(region_east), df['EMPLOYER_STATE'].isin(region_west), df['EMPLOYER_STATE'].isin(region_midwest), df['EMPLOYER_STATE'].isin(region_south)], 

        choicelist=['East Coast', 'West Coast', 'Mid-West Region', 'South Region']))



#Dropping all the other values that do not belong to these four regions

dropRegion = df[df['EMPLOYER_REGION'] == '0'].index

df.drop(dropRegion, inplace=True)
#Changing the data type of EMPLOYER_REGION to category

df['EMPLOYER_REGION'].astype('category')
#All the terms that might be related to an academic institution

terms = ['UNIVERSITY', 'university', 'CITY', 'STATE', 'COLLEGE']

q = r'\b(?:{})\b'.format('|'.join(map(re.escape, terms)))



#Create an empty column by the name of 'EMPLOYER_TYPE'

df['EMPLOYER_TYPE'] = np.nan

df.EMPLOYER_TYPE[df['EMPLOYER_NAME'].str.contains(q)] = 'University/College'



#Replacing all other as Private Company Category

df['EMPLOYER_TYPE']= df.EMPLOYER_TYPE.replace(np.nan, 'Private Company', regex=True)



#Changing the data type of this column 

df['EMPLOYER_TYPE'] = df['EMPLOYER_TYPE'].astype('category')
#Importing beautifulsoup

import requests

from bs4 import BeautifulSoup



#Creating a function that extracts a list of job titles by domain into a list

def extractJobs(link):

    html = requests.get(link).text

    bs = BeautifulSoup(html)

    possible_links = bs.find_all('a')

    title_list = []

    for link in possible_links[11:-28]:

        if link.string != 'Select Location':

            title_list.append(link.string.upper())

    return title_list
#Create a new column named JOB_CATEGORY

df['JOB_CATEGORY'] = np.nan
computer_link = 'https://www.careerbuilder.com/browse/category/computer-occupations'

computer_terms = extractJobs(computer_link)

computer_filter = r'\b(?:{})\b'.format('|'.join(map(re.escape, computer_terms)))



df.JOB_CATEGORY[df['JOB_TITLE'].str.contains(computer_filter)] = 'Software/Computer'
design_link = 'https://www.careerbuilder.com/browse/category/art-and-design-workers'

design_terms = extractJobs(design_link)

design_filter = r'\b(?:{})\b'.format('|'.join(map(re.escape, design_terms)))



df.JOB_CATEGORY[df['JOB_TITLE'].str.contains(design_filter)] = 'Arts and Design'
math_link = 'https://www.careerbuilder.com/browse/category/mathematical-science-occupations'

math_terms = extractJobs(math_link)

math_filter = r'\b(?:{})\b'.format('|'.join(map(re.escape, math_terms)))



df.JOB_CATEGORY[df['JOB_TITLE'].str.contains(math_filter)] = 'Mathematical Sciences'
teaching_link = 'https://www.careerbuilder.com/browse/category/postsecondary-teachers'

teaching_terms = extractJobs(teaching_link)

teaching_filter = r'\b(?:{})\b'.format('|'.join(map(re.escape, teaching_terms)))



df.JOB_CATEGORY[df['JOB_TITLE'].str.contains(teaching_filter)] = 'Teaching'
sales_link = 'https://www.careerbuilder.com/browse/category/sales-representatives-services'

sales_terms = extractJobs(sales_link)

sales_filter = r'\b(?:{})\b'.format('|'.join(map(re.escape, sales_terms)))



df.JOB_CATEGORY[df['JOB_TITLE'].str.contains(sales_filter)] = 'Sales'
eng_link = 'https://www.careerbuilder.com/browse/category/engineers'

eng_terms = extractJobs(eng_link)

eng_filter = r'\b(?:{})\b'.format('|'.join(map(re.escape, eng_terms)))



df.JOB_CATEGORY[df['JOB_TITLE'].str.contains(eng_filter)] = 'Engineering and Hardware'
comm_link = 'https://www.careerbuilder.com/browse/category/counselors-social-workers-and-other-community-and-social-service-specialists'

comm_terms = extractJobs(comm_link)

comm_filter = r'\b(?:{})\b'.format('|'.join(map(re.escape, comm_terms)))



df.JOB_CATEGORY[df['JOB_TITLE'].str.contains(comm_filter)] = 'Community and Social Services'
health_link = 'https://www.careerbuilder.com/browse/category/health-diagnosing-and-treating-practitioners'

health_terms = extractJobs(health_link)

health_filter = r'\b(?:{})\b'.format('|'.join(map(re.escape, health_terms)))



df.JOB_CATEGORY[df['JOB_TITLE'].str.contains(health_filter)] = 'Healthcare'
biz_link = 'https://www.careerbuilder.com/browse/category/business-operations-specialists'

biz_terms = extractJobs(biz_link)

biz_filter = r'\b(?:{})\b'.format('|'.join(map(re.escape, biz_terms)))



df.JOB_CATEGORY[df['JOB_TITLE'].str.contains(biz_filter)] = 'Business/Management'
law_link = 'https://www.careerbuilder.com/browse/category/lawyers-judges-and-related-workers'

law_terms = extractJobs(law_link)

law_filter = r'\b(?:{})\b'.format('|'.join(map(re.escape, law_terms)))



df.JOB_CATEGORY[df['JOB_TITLE'].str.contains(law_filter)] = 'Legal'
df['JOB_CATEGORY']= df.JOB_CATEGORY.replace(np.nan, 'Other', regex=True)
df['JOB_CATEGORY'] = df['JOB_CATEGORY'].astype('category')

df['EMPLOYER_REGION'] =df['EMPLOYER_REGION'].astype('category')



#Drop older columns that are no longer pertinent

df = df.drop(['SOC_TITLE', 'EMPLOYER_NAME', 'EMPLOYER_STATE', 'JOB_TITLE'], axis=1)
import datetime as dt

df['DECISION_TIME'] = df['DECISION_DATE'] - df['RECEIVED_DATE']

df['DECISION_TIME'] = df['DECISION_TIME'].dt.days
df['TIME_EMPLOYED'] = df['END_DATE'] - df['BEGIN_DATE']

df['TIME_EMPLOYED'] = df['TIME_EMPLOYED'].dt.days
#Dropping our old columns since we created new ones 

df = df.drop(['BEGIN_DATE', 'END_DATE', 'DECISION_DATE', 'RECEIVED_DATE'], axis=1)
df.select_dtypes('category')
#Visa Class

df = pd.get_dummies(df, columns=['VISA_CLASS', 'EMPLOYER_REGION', 'JOB_CATEGORY', 'EMPLOYER_TYPE'])



#Fulltime Position

df['FULL_TIME_POSITION'].replace({'Y': 1, 'N': 0}, inplace=True)

df['FULL_TIME_POSITION'] = df['FULL_TIME_POSITION'].astype(int)



#AGENT_REPRESENTING_EMPLOYER

df['AGENT_REPRESENTING_EMPLOYER'].replace({'Y': 1, 'N': 0}, inplace=True)

df['AGENT_REPRESENTING_EMPLOYER'] = df['AGENT_REPRESENTING_EMPLOYER'].astype(int)



#H-1B_DEPENDENT

df['H-1B_DEPENDENT'].replace({'Y': 1, 'N': 0}, inplace=True)

df['H-1B_DEPENDENT'] = df['H-1B_DEPENDENT'].astype(int)
df = df.drop(df[df.CASE_STATUS == 'Certified - Withdrawn'].index)

df = df.drop(df[df.CASE_STATUS == 'Withdrawn'].index)

# df['CASE_STATUS'] = df['CASE_STATUS'].replace({'Certified': 1, 'Denied': 0}, inplace=True)

df['CASE_STATUS']
df['CASE_STATUS'] = df['CASE_STATUS'].astype('category')

df['CASE_STATUS'].replace({'Certified': 1, 'Denied': 0}, inplace=True)
#Viewing the distribution of our target variable

sns.countplot(x='CASE_STATUS', data=df, palette='hls')

plt.show()
df['CASE_STATUS'].value_counts()
#Finding the class ratio to detect imbalance:

count_certified = len(df[df['CASE_STATUS']== 1])

count_denied = len(df[df['CASE_STATUS']== 0])



#Calculating percentage of certified

pct_certified = count_certified/(count_certified+count_denied)

print("percentage of no certified is", pct_certified*100)



#Calculating percentage of denied

pct_denied = count_denied/(count_certified+count_denied)

print("percentage of denied", pct_denied*100)
df.info()
#Traning-test split before oversampling with SMOTE



#All columns except target

X = df.loc[:, df.columns != 'CASE_STATUS']



#Target variable

y = df.loc[:, df.columns == 'CASE_STATUS']



#Split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)



# Describe info about train and test set 

print("Number transactions X_train dataset: ", X_train.shape) 

print("Number transactions y_train dataset: ", y_train.shape) 

print("Number transactions X_test dataset: ", X_test.shape) 

print("Number transactions y_test dataset: ", y_test.shape) 
import numpy as np

#Now train the model without handling the imbalanced class distribution

# logistic regression object 

lr = LogisticRegression() 

  

# train the model on train set 

lr.fit(X_train, y_train) 

  

predictions = lr.predict(X_test) 

  

# print classification report 

print(classification_report(y_test, predictions)) 
print("Before OverSampling, counts of label '1': {}".format(y_train['CASE_STATUS'].value_counts()[1])) 

print("Before OverSampling, counts of label '0': {} \n".format(y_train['CASE_STATUS'].value_counts()[0])) 



# Importing SMOTE module from imblearn library 

from imblearn.over_sampling import SMOTE 

sm = SMOTE(random_state = 2) 

X_train_res, y_train_res = sm.fit_sample(X_train, y_train) 

  

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape)) 

print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape)) 

  

print("After OverSampling, counts of label '1': {}".format(y_train_res['CASE_STATUS'].value_counts()[1])) 

print("After OverSampling, counts of label '0': {}".format(y_train_res['CASE_STATUS'].value_counts()[0]))
lr1 = LogisticRegression() 

lr1.fit(X_train_res, y_train_res) 

predictions = lr1.predict(X_test) 

  

# print classification report 

print(classification_report(y_test, predictions))
from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

logit_roc_auc = roc_auc_score(y_test, predictions)

fpr, tpr, thresholds = roc_curve(y_test, lr1.predict_proba(X_test)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.savefig('Log_ROC')

plt.show()