# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
path = '/kaggle/input/janatahack-healthcare-analytics-part-2/'
dict_df = pd.read_csv(path + 'train_data_dict.csv')
dict_df
df_train = pd.read_csv(path + 'train.csv')
df_train.head()
df_train.Stay = [ '11-20' if i == 'Nov-20' else i for i in df_train.Stay.values]
df_train.Stay = [ '101-200' if i == 'More than 100 Days' else i for i in df_train.Stay.values]
df_train.Stay.unique()
#Finding the datatypes
df_train.dtypes
print('Length of the training data / Number of rows = ', len(df_train))
print('Length of data/ Rows with unique patient id = ',len(df_train.patientid.unique()))
#Finding the null values
len(df_train[df_train.isnull().any(axis=1)])
df_train[df_train.isnull().any(axis=1)].head()
df_train['Stay_min'] = [int(i.split('-')[0]) for i in df_train.Stay.values]
df_train['Stay_max'] = [int(i.split('-')[1]) for i in df_train.Stay.values]
df_train.head()
#Plotting histograms for the numerical variables/ features
numerical = [
  'Available Extra Rooms in Hospital' ,'Bed Grade' ,'Visitors with Patient' , 'Admission_Deposit'
]
df_train[numerical].hist(bins=15, figsize=(15, 6), layout=(2, 4))
#selecting only the columns with data type as object
df_train.select_dtypes(include=['object']).columns
import seaborn as sns
plt.figure(figsize=(20,5))
sns.set(style="darkgrid")
sns.barplot(df_train['Age'].value_counts().index, df_train['Age'].value_counts().values, alpha=0.9)
plt.title('Frequency Distribution of Age')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Age', fontsize=12)
plt.show()
sns.set(style="darkgrid")
sns.barplot(df_train['Type of Admission'].value_counts().index, df_train['Type of Admission'].value_counts().values, alpha=0.9)
plt.title('Frequency Distribution of Type of Admission')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Type of Admission', fontsize=12)
plt.show()
plt.figure(figsize=(20,5))
sns.set(style="darkgrid")
sns.barplot(df_train['Stay'].value_counts().index, df_train['Stay'].value_counts().values, alpha=0.9)
plt.title('Frequency Distribution of Stay')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Stay', fontsize=12)
plt.show()
plt.figure(figsize=(20,5))
sns.set(style="darkgrid")
sns.barplot(df_train['Stay_min'].value_counts().index, df_train['Stay_min'].value_counts().values, alpha=0.9)
plt.title('Frequency Distribution of Stay')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Stay', fontsize=12)
plt.show()
plt.figure(figsize=(20,5))
sns.set(style="darkgrid")
sns.barplot(df_train['Stay_max'].value_counts().index, df_train['Stay_max'].value_counts().values, alpha=0.9)
plt.title('Frequency Distribution of Stay')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Stay', fontsize=12)
plt.show()

import seaborn as sns
sns.set(style="darkgrid")
sns.barplot(df_train['Severity of Illness'].value_counts().index, df_train['Severity of Illness'].value_counts().values, alpha=0.9)
plt.title('Frequency Distribution of Severity of Illness')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Severity of Illness', fontsize=12)
plt.show()
plt.figure(figsize=(20,5))
sns.set(style="darkgrid")
sns.barplot(df_train['Department'].value_counts().index, df_train['Department'].value_counts().values, alpha=0.9)
plt.title('Frequency Distribution of Department Patients')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Department', fontsize=12)
plt.show()
categorical = ['Hospital_type_code', 'Hospital_region_code', 'Department', 'Ward_Type',
       'Ward_Facility_Code', 'Type of Admission', 'Severity of Illness', 'Age',
       'Stay']
fig, ax = plt.subplots(2, 4, figsize=(20, 10))
for variable, subplot in zip(categorical, ax.flatten()):
    sns.countplot(df_train[variable], ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)
from sklearn.preprocessing import LabelEncoder
lb_make = LabelEncoder()
df_train["Hospital_type_code"] = lb_make.fit_transform(df_train["Hospital_type_code"])
df_train["Hospital_region_code"] = lb_make.fit_transform(df_train["Hospital_region_code"])
df_train["Department"] = lb_make.fit_transform(df_train["Department"])
df_train["Ward_Type"] = lb_make.fit_transform(df_train["Ward_Type"])
df_train["Ward_Facility_Code"] = lb_make.fit_transform(df_train["Ward_Facility_Code"])
df_train["Type of Admission"] = lb_make.fit_transform(df_train["Type of Admission"])
df_train["Severity of Illness"] = lb_make.fit_transform(df_train["Severity of Illness"])
df_train["Age"] = lb_make.fit_transform(df_train["Age"])
df_train["Stay"] = lb_make.fit_transform(df_train["Stay"])

df_train.head()
df_train.City_Code_Patient.unique() #null values present
fig, ax = plt.subplots(3, 3, figsize=(15, 10))
for var, subplot in zip(categorical, ax.flatten()):
    sns.boxplot(x=var, y='Stay', data=df_train, ax=subplot)
fig, ax = plt.subplots(3, 3, figsize=(15, 10))
for var, subplot in zip(categorical, ax.flatten()):
    sns.boxplot(x=var, y='Stay_min', data=df_train, ax=subplot)
fig, ax = plt.subplots(3, 3, figsize=(15, 10))
for var, subplot in zip(categorical, ax.flatten()):
    sns.boxplot(x=var, y='Stay_max', data=df_train, ax=subplot)
df_train.columns
df_train.columns[df_train.isna().any()].tolist() #don't include in training meanwhile
X = df_train[['Hospital_code', 'Hospital_type_code', 'City_Code_Hospital',
       'Hospital_region_code', 'Available Extra Rooms in Hospital',
       'Department', 'Ward_Type', 'Ward_Facility_Code', 'Type of Admission',
       'Severity of Illness', 'Visitors with Patient', 'Age',
       'Admission_Deposit']]
y = df_train[['Stay_min']].values
y = y.ravel()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X_train)
X__ = pd.DataFrame(pca.transform(X_train))
X__.head()
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors= len(set(y)))
clf.fit(pd.DataFrame(pca.transform(X_train)),y_train)
#Predict the response for test dataset
y_pred = clf.predict(pd.DataFrame(pca.transform(X_train)))

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_train, y_pred))
print( )

y_pred = clf.predict(pd.DataFrame(pca.transform(X_test)))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
test_path = '/kaggle/input/janatahack-healthcare-analytics-part-2/'
test = pd.read_csv(test_path+'test.csv')

test["Hospital_type_code"] = lb_make.fit_transform(test["Hospital_type_code"])
test["Hospital_region_code"] = lb_make.fit_transform(test["Hospital_region_code"])
test["Department"] = lb_make.fit_transform(test["Department"])
test["Ward_Type"] = lb_make.fit_transform(test["Ward_Type"])
test["Ward_Facility_Code"] = lb_make.fit_transform(test["Ward_Facility_Code"])
test["Type of Admission"] = lb_make.fit_transform(test["Type of Admission"])
test["Severity of Illness"] = lb_make.fit_transform(test["Severity of Illness"])
test["Age"] = lb_make.fit_transform(test["Age"])

t_ = test[['Hospital_code', 'Hospital_type_code', 'City_Code_Hospital',
       'Hospital_region_code', 'Available Extra Rooms in Hospital',
       'Department', 'Ward_Type', 'Ward_Facility_Code', 'Type of Admission',
       'Severity of Illness', 'Visitors with Patient', 'Age',
       'Admission_Deposit']]
y_pred = clf.predict(pd.DataFrame(pca.transform(t_)))
submission = pd.DataFrame()
submission['case_id'] = test['case_id'].values
list_ = list()
for i in y_pred:
    if i == 101:
        list_.append("More than 100 Days")
    else:
        if i == 0:
            u = (str(i) + '-' + str(i+10))
            list_.append(u)
        else:
            u = (str(i) + '-' + str(i+9))
            list_.append(u)
        
print(len(list_) , "    " , len(y_pred))
submission['Stay'] = list_
print(submission['Stay'].unique())
submission.head()
submission.to_csv("submission.csv")

