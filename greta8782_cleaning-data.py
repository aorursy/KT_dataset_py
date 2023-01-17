# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
samplesubmission = pd.read_csv('/kaggle/input/widsdatathon2020/samplesubmission.csv', header=[0], na_filter = False, low_memory=False)

samplesubmission
train = pd.read_csv('/kaggle/input/widsdatathon2020/training_v2.csv', header=[0], na_filter = False, low_memory=False)





def train_numeric():

    na_values = ('np.nan', 'NA', 'NaN' 'inf')

    train2 = train.replace(na_values, np.nan)

    train2_booleans = train2.loc[:, train2.nunique() <= 2]

    train2_booleans.columns

    train2[['hospital_death', 'elective_surgery', 'readmission_status',

       'apache_post_operative', 'arf_apache', 'gcs_unable_apache',

       'intubated_apache', 'ventilated_apache', 'aids', 'cirrhosis',

       'diabetes_mellitus', 'hepatic_failure', 'immunosuppression', 'leukemia',

       'lymphoma', 'solid_tumor_with_metastasis']] = train[['hospital_death', 'elective_surgery', 'readmission_status',

       'apache_post_operative', 'arf_apache', 'gcs_unable_apache',

       'intubated_apache', 'ventilated_apache', 'aids', 'cirrhosis',

       'diabetes_mellitus', 'hepatic_failure', 'immunosuppression', 'leukemia',

       'lymphoma', 'solid_tumor_with_metastasis']].astype(bool)

    #get numerical data

    cols = train2.columns

    num_cols = train2._get_numeric_data().columns

    #for categorical columns

    categorical = list(set(cols) - set(num_cols))

    numeric = list(set(cols) - set(categorical))

    numeric = train2[numeric]

    numeric.columns.tolist()

    low_categorical_uniques = train2.loc[:, train2.nunique() < 15] 

    low_cat_uniques = list(set(low_categorical_uniques) - set(num_cols))

    convert_to_bool = pd.get_dummies(train2[['gcs_verbal_apache',

     'icu_admit_source',

     'icu_type',

     'icu_stay_type',

     'ethnicity',

     'gender',

     'apache_2_bodysystem',

     'gcs_eyes_apache',

     'gcs_motor_apache',

     'apache_3j_bodysystem']])

    numeric = numeric.join(convert_to_bool)

    numeric[['aids',

     'arf_apache',

     'solid_tumor_with_metastasis',

     'patient_id',

     'diabetes_mellitus',

     'leukemia',

     'intubated_apache',

     'cirrhosis',

     'ventilated_apache',

     'lymphoma',

     'immunosuppression',

     'hospital_id',

     'hepatic_failure',

     'readmission_status',

     'hospital_death',

     'elective_surgery',

     'gcs_unable_apache',

     'pre_icu_los_days',

     'icu_id',

     'apache_post_operative']] = (numeric[['aids',

     'arf_apache',

     'solid_tumor_with_metastasis',

     'patient_id',

     'diabetes_mellitus',

     'leukemia',

     'intubated_apache',

     'cirrhosis',

     'ventilated_apache',

     'lymphoma',

     'immunosuppression',

     'hospital_id',

     'hepatic_failure',

     'readmission_status',

     'hospital_death',

     'elective_surgery',

     'gcs_unable_apache',

     'pre_icu_los_days',

     'icu_id',

     'apache_post_operative']]=='TRUE').astype(int)

    #remove columns that contain more than 90% missing data

    column_with_nan = numeric.columns[numeric.isnull().any()]

    numeric_shape = numeric.shape

    for column in column_with_nan:

        if numeric[column].isnull().sum()*100.0/numeric_shape[0] > 90:

            numeric.drop(column,1, inplace = True)



    #keep rows that contain 90% values 

    numeric = numeric.dropna(thresh=90/100*len(numeric.columns))



    #get numerical data

    #cols = numeric.columns

    #num_cols = numeric._get_numeric_data().columns

    return numeric   

train_numeric()

    




def test_numeric():

    test = pd.read_csv('/kaggle/input/widsdatathon2020/unlabeled.csv', header=[0], na_filter = False, low_memory=False)

    na_values = ('np.nan', 'NA', 'NaN' 'inf')

    test = test.replace(na_values, np.nan)

    test_booleans = test.loc[:, test.nunique() <= 2]

    test_booleans.columns

    test[['hospital_death', 'elective_surgery', 'readmission_status',

       'apache_post_operative', 'arf_apache', 'gcs_unable_apache',

       'intubated_apache', 'ventilated_apache', 'aids', 'cirrhosis',

       'diabetes_mellitus', 'hepatic_failure', 'immunosuppression', 'leukemia',

       'lymphoma', 'solid_tumor_with_metastasis']] = train[['hospital_death', 'elective_surgery', 'readmission_status',

       'apache_post_operative', 'arf_apache', 'gcs_unable_apache',

       'intubated_apache', 'ventilated_apache', 'aids', 'cirrhosis',

       'diabetes_mellitus', 'hepatic_failure', 'immunosuppression', 'leukemia',

       'lymphoma', 'solid_tumor_with_metastasis']].astype(bool)

    #get numerical data

    cols = test.columns

    num_cols = test._get_numeric_data().columns

    #for categorical columns

    categorical = list(set(cols) - set(num_cols))

    numeric = list(set(cols) - set(categorical))

    numeric = test[numeric]

    numeric.columns.tolist()

    low_categorical_uniques = test.loc[:, test.nunique() < 15] 

    low_cat_uniques = list(set(low_categorical_uniques) - set(num_cols))

    convert_to_bool = pd.get_dummies(test[['gcs_verbal_apache',

     'icu_admit_source',

     'icu_type',

     'icu_stay_type',

     'ethnicity',

     'gender',

     'apache_2_bodysystem',

     'gcs_eyes_apache',

     'gcs_motor_apache',

     'apache_3j_bodysystem']])

    numeric = numeric.join(convert_to_bool)

    numeric[['aids',

     'arf_apache',

     'solid_tumor_with_metastasis',

     'patient_id',

     'diabetes_mellitus',

     'leukemia',

     'intubated_apache',

     'cirrhosis',

     'ventilated_apache',

     'lymphoma',

     'immunosuppression',

     'hospital_id',

     'hepatic_failure',

     'readmission_status',

     'hospital_death',

     'encounter_id',

     'elective_surgery',

     'gcs_unable_apache',

     'pre_icu_los_days',

     'icu_id',

     'apache_post_operative']] = (numeric[['aids',

     'arf_apache',

     'solid_tumor_with_metastasis',

     'patient_id',

     'diabetes_mellitus',

     'leukemia',

     'intubated_apache',

     'cirrhosis',

     'ventilated_apache',

     'lymphoma',

     'immunosuppression',

     'hospital_id',

     'hepatic_failure',

     'readmission_status',

     'hospital_death',

     'encounter_id',

     'elective_surgery',

     'gcs_unable_apache',

     'pre_icu_los_days',

     'icu_id',

     'apache_post_operative']]=='TRUE').astype(int)

    #remove columns that contain more than 90% missing data

    column_with_nan = numeric.columns[numeric.isnull().any()]

    numeric_shape = numeric.shape

    for column in column_with_nan:

        if numeric[column].isnull().sum()*100.0/numeric_shape[0] > 90:

            numeric.drop(column,1, inplace = True)



    #keep rows that contain 90% values 

    numeric = numeric.dropna(thresh=90/100*len(numeric.columns))



    #get numerical data

    #cols = numeric.columns

    #num_cols = numeric._get_numeric_data().columns

    return numeric   

test_numeric()
train = pd.read_csv('/kaggle/input/widsdatathon2020/training_v2.csv', header=[0], na_filter = False, low_memory=False)

na_values = ('np.nan', 'NA', 'NaN' 'inf')

train2 = train.replace(na_values, np.nan)
train2_booleans = train2.loc[:, train2.nunique() <= 2]

train2_booleans.columns

train2[['hospital_death', 'elective_surgery', 'readmission_status',

       'apache_post_operative', 'arf_apache', 'gcs_unable_apache',

       'intubated_apache', 'ventilated_apache', 'aids', 'cirrhosis',

       'diabetes_mellitus', 'hepatic_failure', 'immunosuppression', 'leukemia',

       'lymphoma', 'solid_tumor_with_metastasis']] = train[['hospital_death', 'elective_surgery', 'readmission_status',

       'apache_post_operative', 'arf_apache', 'gcs_unable_apache',

       'intubated_apache', 'ventilated_apache', 'aids', 'cirrhosis',

       'diabetes_mellitus', 'hepatic_failure', 'immunosuppression', 'leukemia',

       'lymphoma', 'solid_tumor_with_metastasis']].astype(bool)

train2
#get numerical data

cols = train2.columns

num_cols = train2._get_numeric_data().columns

num_cols
#for categorical columns

categorical = list(set(cols) - set(num_cols))

categorical[0:5]
numeric = list(set(cols) - set(categorical))

numeric = train2[numeric]

numeric.columns.tolist()

numeric
low_categorical_uniques = train2.loc[:, train2.nunique() < 15] 

low_categorical_uniques.columns[0:5]
low_cat_uniques = list(set(low_categorical_uniques) - set(num_cols))

low_cat_uniques
convert_to_bool = pd.get_dummies(train2[['gcs_verbal_apache',

 'icu_admit_source',

 'icu_type',

 'icu_stay_type',

 'ethnicity',

 'gender',

 'apache_2_bodysystem',

 'gcs_eyes_apache',

 'gcs_motor_apache',

 'apache_3j_bodysystem']])

numeric = numeric.join(convert_to_bool)

numeric
numeric[['aids',

 'arf_apache',

 'solid_tumor_with_metastasis',

 'patient_id',

 'diabetes_mellitus',

 'leukemia',

 'intubated_apache',

 'cirrhosis',

 'ventilated_apache',

 'lymphoma',

 'immunosuppression',

 'hospital_id',

 'hepatic_failure',

 'readmission_status',

 'hospital_death',

 'encounter_id',

 'elective_surgery',

 'gcs_unable_apache',

 'pre_icu_los_days',

 'icu_id',

 'apache_post_operative']] = (numeric[['aids',

 'arf_apache',

 'solid_tumor_with_metastasis',

 'patient_id',

 'diabetes_mellitus',

 'leukemia',

 'intubated_apache',

 'cirrhosis',

 'ventilated_apache',

 'lymphoma',

 'immunosuppression',

 'hospital_id',

 'hepatic_failure',

 'readmission_status',

 'hospital_death',

 'encounter_id',

 'elective_surgery',

 'gcs_unable_apache',

 'pre_icu_los_days',

 'icu_id',

 'apache_post_operative']]=='TRUE').astype(int)



numeric
#remove columns that contain more than 90% missing data

column_with_nan = numeric.columns[numeric.isnull().any()]

numeric_shape = numeric.shape

for column in column_with_nan:

    if numeric[column].isnull().sum()*100.0/train2_shape[0] > 90:

        numeric.drop(column,1, inplace = True)



#keep rows that contain 90% values 



numeric = numeric.dropna(thresh=90/100*len(numeric.columns))



numeric['encounter_id'] = train['encounter_id']

numeric['encounter_id']
#get numerical data

cols = numeric.columns

num_cols = numeric._get_numeric_data().columns

num_cols
from sklearn import preprocessing

X = numeric.values

y = train['hospital_death']

from sklearn.model_selection import train_test_split

X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)

train_test_split

y_trainset.head()

from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier(criterion="entropy", max_depth = 8)

dtree # it shows the default parameters

dtree.fit(X_trainset,y_trainset)

predTree = dtree.predict(X_testset)

print (predTree [0:5])

print (y_testset [0:5])

from sklearn import metrics

import matplotlib.pyplot as plt

print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))
testnumeric = test_numeric()

testnumeric['encounter_id'] = test['encounter_id']
from sklearn.model_selection import train_test_split

X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)

train_test_split

y_trainset.head()

from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier(criterion="entropy", max_depth = 8)

dtree.fit(X_trainset,y_trainset)

predTree = dtree.predict(testnumeric)

print (predTree [0:5])

print (y_testset [0:5])



testnumeric['hospital_death'] = y_testset

testnumeric['hospital_death'].describe()
answercols = ['encounter_id', 'hospital_death']

answer = pd.DataFrame(testnumeric, columns = answercols)

answer = answer.apply(pd.to_numeric)

answer
answer = answer.to_csv('datadivas3.csv', encoding = 'utf-8', index = False, header = True)