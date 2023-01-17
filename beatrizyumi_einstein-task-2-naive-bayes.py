# Importing libraries



# overall libraries

import pandas as pd

import numpy as np

import warnings

warnings.filterwarnings('ignore')

from collections import OrderedDict

from IPython.core.pylabtools import figsize

import re



# plotting libraries

import seaborn as sns

sns.set_style('white')

import matplotlib.pyplot as plt

import matplotlib.lines as mlines

%matplotlib inline



# sklearn libraries

import sklearn

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import scale

from sklearn.model_selection import train_test_split

from sklearn.datasets import make_moons

from sklearn.metrics import log_loss, roc_auc_score, roc_curve, auc, precision_recall_curve, confusion_matrix, average_precision_score

from sklearn.naive_bayes import GaussianNB

from sklearn.multiclass import OneVsRestClassifier

from sklearn.svm import LinearSVC

from sklearn.preprocessing import LabelBinarizer
# setting the display so you can see all the columns and all the rows



pd.set_option("max_columns", None)

pd.set_option("max_rows", None)
# creating the DataFrame



df = pd.read_excel('../input/covid19/dataset.xlsx', encoding='utf8')
# Checking how the df imported



df.head()
# Checking the unique values for the SARS-Cov-2 exam result



df['SARS-Cov-2 exam result'].unique()
# Replacing negative to 0 an positive to 1 and then checking if it worked



df['SARS-Cov-2 exam result'] = df['SARS-Cov-2 exam result'].replace({'negative': 0, 'positive': 1})

df['SARS-Cov-2 exam result'].unique()
# checking the categorical variables



df.select_dtypes(include = ['object']).columns
# replacing the values to make them numerical, I am doing them by hand to make sure all the exams make sense.

# This is possible because there aren't many categorical variables.

# To do this, I checked all the variables unique values and created an unique dictionary



df.loc[:,'Respiratory Syncytial Virus':'Parainfluenza 2'] = df.loc[:,'Respiratory Syncytial Virus':'Parainfluenza 2'].replace({'not_detected':0, 'detected':1})

df.loc[:,'Influenza B, rapid test':'Strepto A'] = df.loc[:,'Influenza B, rapid test':'Strepto A'].replace({'negative':0, 'positive':1})

df['Urine - Esterase'] = df['Urine - Esterase'].replace({'absent':0})

df['Urine - Aspect'] = df['Urine - Aspect'].replace({'clear':0, 'cloudy':2, 'altered_coloring':3, 'lightly_cloudy':1})

df['Urine - pH'] = df['Urine - pH'].replace({'6.5':6.5, '6.0':6.0,'5.0':5.0, '7.0':7.0, '5':5, '5.5':5.5,

       '7.5':7.5, '6':6, '8.0':8.0})

df['Urine - Hemoglobin'] = df['Urine - Hemoglobin'].replace({'absent':0, 'present':1})

df.loc[:,'Urine - Bile pigments':'Urine - Nitrite'] = df.loc[:,'Urine - Bile pigments':'Urine - Nitrite'].replace({'absent':0})

df.loc[:,'Urine - Urobilinogen':'Urine - Protein'] = df.loc[:,'Urine - Urobilinogen':'Urine - Protein'].replace({'absent':0, 'normal':1})

df['Urine - Hemoglobin'] = df['Urine - Hemoglobin'].replace({'absent':0, 'present':1, 'not_done':np.nan})

df['Urine - Leukocytes'] = df['Urine - Leukocytes'].replace({'38000':38000, '5942000':5942000, '32000':32000, '22000':22000,'<1000': 900, '3000': 3000,'16000':16000, '7000':7000, '5300':5300, '1000':1000, '4000':4000, '5000':5000, '10600':106000, '6000':6000, '2500':2500, '2600':2600, '23000':23000, '124000':124000, '8000':8000, '29000':29000, '2000':2000,'624000':642000, '40000':40000, '3310000':3310000, '229000':229000, '19000':19000, '28000':28000, '10000':10000,'4600':4600, '77000':77000, '43000':43000})

df['Urine - Crystals'] = df['Urine - Crystals'].replace({'Ausentes':0, 'Urato Amorfo --+':1, 'Oxalato de Cálcio +++':3,'Oxalato de Cálcio -++':2, 'Urato Amorfo +++':4})

df.loc[:,'Urine - Hyaline cylinders':'Urine - Yeasts'] = df.loc[:,'Urine - Hyaline cylinders':'Urine - Yeasts'].replace({'absent':0})

df['Urine - Color'] = df['Urine - Color'].replace({'light_yellow':0, 'yellow':1, 'orange':2, 'citrus_yellow':1})

df = df.replace('not_done', np.NaN)

df = df.replace('Não Realizado', np.NaN)
# Dropping the patient ID column



df = df.drop('Patient ID', axis = 1)
# checking if all of the categorical variables were treated



df.select_dtypes(include = ['object']).columns
# checking how the data is distribuited in the dataframe



df.info()
# let's see what are the two columns that are working with int



df.select_dtypes(include = ['int64']).columns
# let's create a rank of missing values



null_count = df.isnull().sum().sort_values(ascending=False)

null_percentage = null_count / len(df)

null_rank = pd.DataFrame(data=[null_count, null_percentage],index=['null_count', 'null_ratio']).T

null_rank
# dropping columns that don't have any content in it



df = df.drop(['Mycoplasma pneumoniae','Urine - Nitrite', 'Urine - Sugar', 'Partial thromboplastin time (PTT) ', 'Prothrombin time (PT), Activity', 'D-Dimer'], axis = 1)
# let's see the min and max values of the variables to fill their missing values



df.describe().round(2)
# filling missing values with 0



df[['Urine - Leukocytes', 'Urine - pH']] = df[['Urine - Leukocytes', 'Urine - pH']].fillna(0)
# filling missing values with -1



df[['Patient age quantile', 'SARS-Cov-2 exam result', 'Respiratory Syncytial Virus', 'Influenza A', 'Influenza B', 'Parainfluenza 1', 'CoronavirusNL63', 'Rhinovirus/Enterovirus', 'Coronavirus HKU1', 'Parainfluenza 3', 'Chlamydophila pneumoniae', 'Adenovirus', 'Parainfluenza 4', 'Coronavirus229E', 'CoronavirusOC43', 'Inf A H1N1 2009', 'Bordetella pertussis', 'Metapneumovirus', 'Parainfluenza 2', 'Influenza B, rapid test', 'Influenza A, rapid test', 'Strepto A', 'Fio2 (venous blood gas analysis)','Myeloblasts', 'Urine - Esterase', 'Urine - Hemoglobin', 'Urine - Bile pigments', 'Urine - Ketone Bodies', 'Urine - Protein', 'Urine - Crystals', 'Urine - Hyaline cylinders', 'Urine - Granular cylinders', 'Urine - Yeasts', 'Urine - Color']] = df[['Patient age quantile', 'SARS-Cov-2 exam result', 'Respiratory Syncytial Virus', 'Influenza A', 'Influenza B', 'Parainfluenza 1', 'CoronavirusNL63', 'Rhinovirus/Enterovirus', 'Coronavirus HKU1', 'Parainfluenza 3', 'Chlamydophila pneumoniae', 'Adenovirus', 'Parainfluenza 4', 'Coronavirus229E', 'CoronavirusOC43', 'Inf A H1N1 2009', 'Bordetella pertussis', 'Metapneumovirus', 'Parainfluenza 2', 'Influenza B, rapid test', 'Influenza A, rapid test', 'Strepto A', 'Fio2 (venous blood gas analysis)','Myeloblasts', 'Urine - Esterase', 'Urine - Hemoglobin', 'Urine - Bile pigments', 'Urine - Ketone Bodies', 'Urine - Protein', 'Urine - Crystals', 'Urine - Hyaline cylinders', 'Urine - Granular cylinders', 'Urine - Yeasts', 'Urine - Color']].fillna(-1)
# filling all the other missing values with 99



df = df.fillna(99)
# let's see if there is still any missing values left



null_count = df.isnull().sum().sort_values(ascending=False)

null_percentage = null_count / len(df)

null_rank = pd.DataFrame(data=[null_count, null_percentage],index=['null_count', 'null_ratio']).T

null_rank
# let's now see the description of the dataframe again, because i am pretty sure we will have to apply some sort of normalization technique on it



df.describe()
# creating a scaler and using it, disconsidering the target column



scaler = MinMaxScaler()

addmits = pd.DataFrame(df[['Patient addmited to regular ward (1=yes, 0=no)','Patient addmited to semi-intensive unit (1=yes, 0=no)', 'Patient addmited to intensive care unit (1=yes, 0=no)']], columns = ['Patient addmited to regular ward (1=yes, 0=no)', 'Patient addmited to semi-intensive unit (1=yes, 0=no)', 'Patient addmited to intensive care unit (1=yes, 0=no)'])

df_scaled = pd.DataFrame(scaler.fit_transform(df.drop(['Patient addmited to regular ward (1=yes, 0=no)','Patient addmited to semi-intensive unit (1=yes, 0=no)', 'Patient addmited to intensive care unit (1=yes, 0=no)'], axis = 1)), columns = (df.drop(['Patient addmited to regular ward (1=yes, 0=no)','Patient addmited to semi-intensive unit (1=yes, 0=no)', 'Patient addmited to intensive care unit (1=yes, 0=no)'], axis = 1).columns))
# concatenating all the columns again



df_total = pd.concat([addmits, df_scaled], axis = 1)
# checking if the concatening worked



df_total.head()
# doing a correlation rank to see how the exams work with the result of the exam



df_total.corr()[['Patient addmited to regular ward (1=yes, 0=no)', 'Patient addmited to semi-intensive unit (1=yes, 0=no)', 'Patient addmited to intensive care unit (1=yes, 0=no)']].sort_values(by = 'Patient addmited to intensive care unit (1=yes, 0=no)', ascending=False)
# renaming the columns with backslash



df_total = df_total.rename(columns={"Meancorpuscularhemoglobinconcentration\xa0MCHC": "Meancorpuscularhemoglobinconcentrationxa0MCHC", "Gammaglutamyltransferase\xa0": "Gammaglutamyltransferasexa0", "Ionizedcalcium\xa0": "Ionizedcalciumxa0", "Creatinephosphokinase\xa0CPK\xa0" : "Creatinephosphokinasexa0CPKxa0"})
# Let's remove all special characters and spaces from the column names

# We will also make them lowercase



df_total.columns=df_total.columns.str.replace(r'\(|\)|:|,|;|\.|’|”|“|\?|%|>|<|(|)','')

df_total.columns=df_total.columns.str.replace(r'/','')

df_total.columns=df_total.columns.str.replace(' ','')

df_total.columns=df_total.columns.str.replace('"','')

df_total.columns=df_total.columns.str.replace('\-','')

df_total.columns=df_total.columns.str.replace('\=','')

df_total.columns=df_total.columns.str.replace('\#','')

df_total.columns=df_total.columns.str.lower()
# let's get a list of all the columns so we can start working on our model



list(df_total.columns)
# removing all the lines that don't give out a positive result for the sars-cov2 exam



df_total = df_total[df_total['sarscov2examresult'] != 0]
# the first column will remain with the 1 value, the other two will be replaced with 2 and 3



df_total['patientaddmitedtosemiintensiveunit1yes0no'] = df_total['patientaddmitedtosemiintensiveunit1yes0no'].replace({1:2})

df_total['patientaddmitedtointensivecareunit1yes0no'] = df_total['patientaddmitedtointensivecareunit1yes0no'].replace({1:3})
# creating one single column that sums up all the directions to where the patients where sent



df_total['patient'] = df_total.apply(lambda row: row.patientaddmitedtoregularward1yes0no + row.patientaddmitedtosemiintensiveunit1yes0no + row.patientaddmitedtointensivecareunit1yes0no, axis=1)

df_total.head()
# dropping the first three columns



df_total = df_total.drop(['patientaddmitedtoregularward1yes0no', 'patientaddmitedtosemiintensiveunit1yes0no', 'patientaddmitedtointensivecareunit1yes0no'], axis = 1)
# Creating X and y



X = df_total.drop(['patient'], axis = 1)

y = df_total['patient']
# let's split the X and y into a test and train set



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state = 42)
# Creating a Gaussian Naive Bayes Classifier



gnb = GaussianNB().fit(X_train, y_train) 

gnb_predictions = gnb.predict(X_test)
# Calculating the score of the model



accuracy = gnb.score(X_test, y_test) 

print(accuracy)
# Calculating the ROC AUC score of the model



def multiclass_roc_auc_score(y_test, y_pred, average="macro"):

    lb = LabelBinarizer()

    lb.fit(y_test)

    y_test = lb.transform(y_test)

    y_pred = lb.transform(y_pred)

    return roc_auc_score(y_test, y_pred, average=average)



multiclass_roc_auc_score(y_test, gnb_predictions)