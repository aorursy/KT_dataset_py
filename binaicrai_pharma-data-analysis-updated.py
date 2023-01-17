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



import warnings

warnings.simplefilter('ignore')

warnings.filterwarnings('ignore')



import seaborn as sns

import matplotlib as p

import matplotlib.pyplot as plt

%matplotlib inline



import copy

import math

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
pharma_data_original = pd.read_csv('https://raw.githubusercontent.com/dphi-official/Datasets/master/pharma_data/Training_set_advc.csv')

pharma_data_original.head()
test_new = pd.read_csv('https://raw.githubusercontent.com/dphi-official/Datasets/master/pharma_data/Testing_set_advc.csv')

test_new.head()
pharma_data_original.isna().sum()
test_new.isna().sum()
# Checking % of NA On train data - column wise

# pd.set_option('display.max_rows', 500)

NA_col_train = pd.DataFrame(pharma_data_original.isna().sum(), columns = ['NA_Count'])

NA_col_train['% of NA'] = (NA_col_train.NA_Count/len(pharma_data_original))*100

NA_col_train.sort_values(by = ['% of NA'], ascending = False, na_position = 'first').head(10)
# Extracting only the null value records to column 'A' and checking the counts and turned out to be the same

nacheck = pharma_data_original[pharma_data_original['A'].isnull()]

nacheck.isna().sum()
cols = pharma_data_original.columns

num_cols = pharma_data_original._get_numeric_data().columns

cat_cols = list(set(cols) - set(num_cols))
plt.figure(figsize = (14,10))

sns.heatmap(pharma_data_original[num_cols].corr(), vmin=pharma_data_original[num_cols].values.min(), vmax=1, 

            annot=True, annot_kws={"size":10}, square = False)

plt.show()
print('Train - Unique values before cleaning: ', pharma_data_original['Treated_with_drugs'].nunique())

pharma_data_original['Treated_with_drugs'] = pharma_data_original['Treated_with_drugs'].str.strip()

pharma_data_original['Treated_with_drugs'] = pharma_data_original['Treated_with_drugs'].str.upper()

print('Train - Unique values after cleaning: ', pharma_data_original['Treated_with_drugs'].nunique())
print('Test - Unique values before cleaning: ', test_new['Treated_with_drugs'].nunique())

test_new['Treated_with_drugs'] = test_new['Treated_with_drugs'].str.strip()

test_new['Treated_with_drugs'] = test_new['Treated_with_drugs'].str.upper()

test_new['Treated_with_drugs'].nunique()

print('Test - Unique values after cleaning: ', test_new['Treated_with_drugs'].nunique())
pharma_data_original['Patient_Smoker'].unique()
print('Train - Unique values before cleaning: ', pharma_data_original['Patient_Smoker'].unique())

pharma_data_original['Patient_Smoker'] = pharma_data_original['Patient_Smoker'].str.strip()

pharma_data_original['Patient_Smoker'] = pharma_data_original['Patient_Smoker'].str.upper()

pharma_data_original['Patient_Smoker'] = pharma_data_original['Patient_Smoker'].replace({'YESS': 'YES'})

print('Train - Unique values after cleaning: ', pharma_data_original['Patient_Smoker'].unique())
test_new['Patient_Smoker'].unique()
print('Test - Unique values before cleaning: ', pharma_data_original['Patient_Smoker'].unique())

test_new['Patient_Smoker'] = test_new['Patient_Smoker'].str.strip()

test_new['Patient_Smoker'] = test_new['Patient_Smoker'].str.upper()

test_new['Patient_Smoker'] = test_new['Patient_Smoker'].replace({'YESS': 'YES'})

test_new['Patient_Smoker'].unique()

print('Test - Unique values after cleaning: ', test_new['Patient_Smoker'].unique())
pharma_data_original['New_ID'] = pharma_data_original['ID_Patient_Care_Situation'].groupby(pharma_data_original['Patient_ID']).transform('count')

pharma_data_original.head()
test_new['New_ID'] = test_new['ID_Patient_Care_Situation'].groupby(test_new['Patient_ID']).transform('count')

test_new.head()
for i in pharma_data_original.columns:

    if pharma_data_original[i].nunique() == 1:

        print('With only 1 unique value: ', i)

    if pharma_data_original[i].nunique() == pharma_data_original.shape[0]:

        print('With all unique value: ', i)
for i in test_new.columns:

    if test_new[i].nunique() == 1:

        print('With only 1 unique value: ', i)

    if test_new[i].nunique() == test_new.shape[0]:

        print('With all unique value: ', i)
print('Train shape: ', pharma_data_original.shape)

print('Test shape: ', test_new.shape)
pharma_data_original = pharma_data_original.drop_duplicates()

test_new = test_new.drop_duplicates()

print('Train shape: ', pharma_data_original.shape)

print('Test shape: ', test_new.shape)
sns.catplot('Patient_Age', data= pharma_data_original, kind='count', alpha=0.7, height=4, aspect= 6)



# Get current axis on current figure

ax = plt.gca()



# Max value to be set

y_max = pharma_data_original['Patient_Age'].value_counts().max() 



# Iterate through the list of axes' patches

for p in ax.patches:

    ax.text(p.get_x() + p.get_width()/5., p.get_height(),'%d' % int(p.get_height()),

            fontsize=13, color='blue', ha='center', va='bottom')

plt.title('Frequency plot of Patient_Age for train', fontsize = 20, color = 'black')

plt.show()
sns.catplot('Patient_Age', data= test_new, kind='count', alpha=0.7, height=4, aspect= 6)



# Get current axis on current figure

ax = plt.gca()



# Max value to be set

y_max = test_new['Patient_Age'].value_counts().max() 



# Iterate through the list of axes' patches

for p in ax.patches:

    ax.text(p.get_x() + p.get_width()/5., p.get_height(),'%d' % int(p.get_height()),

            fontsize=13, color='blue', ha='center', va='bottom')

plt.title('Frequency plot of Patient_Age for test', fontsize = 20, color = 'black')

plt.show()
fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(18,8))



label1 = ['Patient_Age']



# Box plot for Patient_Age for train

bplot1 = ax1.boxplot(pharma_data_original['Patient_Age'],

                     vert=True,  # vertical box alignment

                     patch_artist=True,  # fill with color

                     labels = label1)  # will be used to label x-ticks

ax1.set_title('Box plot for Patient_Age for train')



# Box plot for Patient_Age for test

bplot2 = ax2.boxplot(test_new['Patient_Age'],

                     vert=True,  # vertical box alignment

                     patch_artist=True,  # fill with color

                     labels = label1)  # will be used to label x-ticks

ax1.set_title('Box plot for Patient_Age for train')
pharma_data_original = pharma_data_original[pharma_data_original['Patient_Age']<100]
fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(9,8))



label1 = ['Patient_Age']



# Box plot for Patient_Age for train

bplot1 = ax1.boxplot(pharma_data_original['Patient_Age'],

                     vert=True,  # vertical box alignment

                     patch_artist=True,  # fill with color

                     labels = label1)  # will be used to label x-ticks

ax1.set_title('Box plot for Patient_Age for train')
bin_labels = [1,2,3,4,5,6,7,8,9,10]

pharma_data_original['Age_band'] = pd.qcut(pharma_data_original['Patient_Age'], q=10, labels = bin_labels)

test_new['Age_band'] = pd.qcut(test_new['Patient_Age'], q=10, labels = bin_labels)
sns.catplot('Age_band', data= pharma_data_original, kind='count', alpha=0.7, height=4, aspect= 6)



# Get current axis on current figure

ax = plt.gca()



# Max value to be set

y_max = pharma_data_original['Age_band'].value_counts().max() 



# Iterate through the list of axes' patches

for p in ax.patches:

    ax.text(p.get_x() + p.get_width()/5., p.get_height(),'%d' % int(p.get_height()),

            fontsize=13, color='blue', ha='center', va='bottom')

plt.title('Frequency plot of Age_band for train', fontsize = 20, color = 'black')

plt.show()
sns.catplot('Age_band', data= test_new, kind='count', alpha=0.7, height=4, aspect= 6)



# Get current axis on current figure

ax = plt.gca()



# Max value to be set

y_max = test_new['Age_band'].value_counts().max() 



# Iterate through the list of axes' patches

for p in ax.patches:

    ax.text(p.get_x() + p.get_width()/5., p.get_height(),'%d' % int(p.get_height()),

            fontsize=13, color='blue', ha='center', va='bottom')

plt.title('Frequency plot of Age_band for test', fontsize = 20, color = 'black')

plt.show()
# Imputing 'Treated_with_drugs' with mode



def impute_with_mode(x):

    max_x = x.value_counts()

    mode = max_x[max_x == max_x.max()].index[0]

    x[x.isna()] = mode

    return x



pharma_data_original['Treated_with_drugs'] = pharma_data_original[['Treated_with_drugs']].apply(lambda x: impute_with_mode(x))
pharma_data = pharma_data_original.copy()

test = test_new.copy()
from fancyimpute import IterativeImputer



# Initialize IterativeImputer

mice_imputer = IterativeImputer()



# Impute using fit_tranform

pharma_data[['A','B','C','D','E','F','Z']] = mice_imputer.fit_transform(pharma_data[['A','B','C','D','E','F','Z']])

pharma_data['Number_of_prev_cond'] = pharma_data.apply(lambda row: (row['A']+row['B']+row['C']+row['D']+row['D']+row['D']+row['Z'])

                                                       if np.isnan(row['Number_of_prev_cond']) else row['Number_of_prev_cond'], axis=1)

pharma_data[['A','B','C','D','E','F','Z','Number_of_prev_cond']]=pharma_data[['A','B','C','D','E','F','Z','Number_of_prev_cond']].round()
pharma_data.isna().sum()
pharma_data2 = pharma_data_original[pharma_data_original['A'].notna()].copy()

test2 = test_new.copy()
pharma_data2.isna().sum()
print('Dataset1: ', pharma_data.shape)

print('Dataset2: ', pharma_data2.shape)
for i in pharma_data_original.columns:

    print(i)

    print(pharma_data_original[i].nunique())
for i in test_new.columns:

    print(i)

    print(test_new[i].nunique())
col_train = pharma_data_original.columns

col_test = test_new.columns
# Taking 32 as the threshold to include Treated_with_drugs column as well

# Another way would be to use any threshold below that and add the colum later

l1 = []

for i in col_train:

    if pharma_data_original[i].nunique() <= 32:

        l1.append(i)



l1.remove('Survived_1_year')
l2 = []

for i in col_test:

    if test_new[i].nunique() <= 32:

        l2.append(i)
# Checking the columns in train and test are same or not

df = pd.DataFrame(l1, columns = ['train'])

df['test'] = pd.DataFrame(l2)

df
# Dataset1

pharma_data[l1] = pharma_data[l1].apply(lambda x: x.astype('category'), axis=0)

test[l1] = test[l1].apply(lambda x: x.astype('category'), axis=0)

print('Dataset1: train dtypes:')

print('======================================')

print(pharma_data[l1].dtypes)

print('train shape: ', pharma_data.shape)

print('======================================')

print('test dtypes:')

print(test[l1].dtypes)

print('test shape: ', test.shape)
# Dataset1

pharma_data2[l1] = pharma_data2[l1].apply(lambda x: x.astype('category'), axis=0)

test2[l1] = test2[l1].apply(lambda x: x.astype('category'), axis=0)

print('Dataset2: train dtypes:')

print('======================================')

print(pharma_data2[l1].dtypes)

print('train shape: ', pharma_data2.shape)

print('======================================')

print('test dtypes:')

print(test2[l1].dtypes)

print('test shape: ', test2.shape)
l1
cols_to_drop = ['Patient_ID', 'ID_Patient_Care_Situation', 'Patient_mental_condition', 'Patient_Age']
pharma_data = pharma_data.drop(cols_to_drop, axis=1)

test = test.drop(cols_to_drop, axis=1)

pharma_data2 = pharma_data2.drop(cols_to_drop, axis=1)

test2 = test2.drop(cols_to_drop, axis=1)

print('Dataset1: ', pharma_data.shape, ' ', test.shape)

print('Dataset2: ', pharma_data2.shape, ' ', test2.shape)
# This code to be used if any columns to be dropped is in l1

cat_columns = list(set(l1) - set(cols_to_drop))

cat_columns
# For MICE imputed datasets

X = pharma_data.drop(['Survived_1_year'], axis = 1)

y = pharma_data['Survived_1_year']



X_num = len(X)

combined_dataset = pd.concat(objs=[X, test], axis=0)

combined_dataset = pd.get_dummies(combined_dataset, columns=cat_columns, drop_first=True)

X = copy.copy(combined_dataset[:X_num])

test = copy.copy(combined_dataset[X_num:])
# For datasets with null value rows dropped

X2 = pharma_data2.drop(['Survived_1_year'], axis = 1)

y2 = pharma_data2['Survived_1_year']



X_num = len(X2)

combined_dataset = pd.concat(objs=[X2, test2], axis=0)

combined_dataset = pd.get_dummies(combined_dataset, columns=cat_columns, drop_first=True)

X2 = copy.copy(combined_dataset[:X_num])

test2 = copy.copy(combined_dataset[X_num:])
pharma_data.to_csv("train_with_mice_imputation", index=False)

test.to_csv("test_cleaned", index=False)

pharma_data2.to_csv("train_with_null_rows_dropped", index=False)

test2.to_csv("test_cleaned2", index=False)
# For MICE imputed data

X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.3, random_state = 50)

# For null rows dropped data

X_train2, X_val2, y_train2, y_val2 = train_test_split(X2,y2,test_size=0.3, random_state = 50)
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression



grid={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]} # l1 Lasso l2 Ridge

logreg=LogisticRegression()

logreg_cv=GridSearchCV(logreg,grid,cv=5)

logreg_cv.fit(X_train,y_train)

logreg_cv.best_params_
# Predict (train)

y_train_pred = logreg_cv.predict(X_train)



# Model evaluation (train)

f1 = f1_score(y_train, y_train_pred)

acc = accuracy_score(y_train, y_train_pred)

cm = confusion_matrix(y_train, y_train_pred)

print('Dataset1: Logreg - train')

print('-------------------------')

print('f1_score: ', f1)

print('accuracy_score: ', acc)

print('Confusion Matrix: ')

print(cm)
# Predict (val)

y_val_pred = logreg_cv.predict(X_val)



# Model evaluation (train)

f1 = f1_score(y_val, y_val_pred)

acc = accuracy_score(y_val, y_val_pred)

cm = confusion_matrix(y_val, y_val_pred)

print('Dataset1: Logreg - val')

print('-----------------------')

print('f1_score: ', f1)

print('accuracy_score: ', acc)

print('Confusion Matrix: ')

print(cm)
submission_lrcv = pd.DataFrame()

submission_lrcv['Survived_1_year'] = logreg_cv.predict(test)

submission_lrcv.head()
submission_lrcv.to_csv("submission_lrcv.csv",index=False)
logreg2=LogisticRegression()

logreg_cv2=GridSearchCV(logreg2,grid,cv=5)

logreg_cv2.fit(X_train2,y_train2)

logreg_cv2.best_params_
# Predict (train)

y_train_pred = logreg_cv2.predict(X_train2)



# Model evaluation (train)

f1 = f1_score(y_train2, y_train_pred)

acc = accuracy_score(y_train2, y_train_pred)

cm = confusion_matrix(y_train2, y_train_pred)

print('Dataset2: Logreg - train')

print('-------------------------')

print('f1_score: ', f1)

print('accuracy_score: ', acc)

print('Confusion Matrix: ')

print(cm)
# Predict (val)

y_val_pred = logreg_cv2.predict(X_val2)



# Model evaluation (train)

f1 = f1_score(y_val2, y_val_pred)

acc = accuracy_score(y_val2, y_val_pred)

cm = confusion_matrix(y_val2, y_val_pred)

print('Dataset2: Logreg - val')

print('-----------------------')

print('f1_score: ', f1)

print('accuracy_score: ', acc)

print('Confusion Matrix: ')

print(cm)
submission_lrcv2 = pd.DataFrame()

submission_lrcv2['Survived_1_year'] = logreg_cv2.predict(test2)

submission_lrcv2.head()
submission_lrcv2.to_csv("submission_lrcv2.csv",index=False)
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=50)

dt.fit(X_train, y_train)
# Predict (train)

y_train_pred = dt.predict(X_train)



# Model evaluation (train)

f1 = f1_score(y_train, y_train_pred)

acc = accuracy_score(y_train, y_train_pred)

cm = confusion_matrix(y_train, y_train_pred)

print('Dataset1: DT - train')

print('---------------------')

print('f1_score: ', f1)

print('accuracy_score: ', acc)

print('Confusion Matrix: ')

print(cm)
# Predict (val)

y_val_pred = dt.predict(X_val)



# Model evaluation (train)

f1 = f1_score(y_val, y_val_pred)

acc = accuracy_score(y_val, y_val_pred)

cm = confusion_matrix(y_val, y_val_pred)

print('Dataset1: DT - val')

print('-------------------')

print('f1_score: ', f1)

print('accuracy_score: ', acc)

print('Confusion Matrix: ')

print(cm)
submission_dt = pd.DataFrame()

submission_dt['Survived_1_year'] = dt.predict(test)

submission_dt.head()
submission_dt.to_csv("submission_dt.csv",index=False)
dt2 = DecisionTreeClassifier(random_state=50)

dt2.fit(X_train2, y_train2)
# Predict (train)

y_train_pred = dt2.predict(X_train2)



# Model evaluation (train)

f1 = f1_score(y_train2, y_train_pred)

acc = accuracy_score(y_train2, y_train_pred)

cm = confusion_matrix(y_train2, y_train_pred)

print('Dataset2: DT - train')

print('---------------------')

print('f1_score: ', f1)

print('accuracy_score: ', acc)

print('Confusion Matrix: ')

print(cm)
# Predict (val)

y_val_pred = dt2.predict(X_val2)



# Model evaluation (train)

f1 = f1_score(y_val2, y_val_pred)

acc = accuracy_score(y_val2, y_val_pred)

cm = confusion_matrix(y_val2, y_val_pred)

print('Dataset2: DT - val')

print('-------------------')

print('f1_score: ', f1)

print('accuracy_score: ', acc)

print('Confusion Matrix: ')

print(cm)
submission_dt2 = pd.DataFrame()

submission_dt2['Survived_1_year'] = dt2.predict(test2)

submission_dt2.head()
submission_dt2.to_csv("submission_dt2.csv",index=False)
from sklearn.model_selection import GridSearchCV

params = {'max_leaf_nodes': list(range(2, 200,10)), 

          'min_samples_split': [3, 4, 5],

          'max_depth': list(range(4,10)),

          'criterion' :['gini', 'entropy']}
%%time

dtgs = DecisionTreeClassifier(random_state=50)

cv_dt = GridSearchCV(estimator=dtgs, param_grid=params, scoring = 'f1', cv= 5)

cv_dt.fit(X_train, y_train)

cv_dt.best_params_
# Predict (train)

y_train_pred = cv_dt.predict(X_train)



# Model evaluation (train)

f1 = f1_score(y_train, y_train_pred)

acc = accuracy_score(y_train, y_train_pred)

cm = confusion_matrix(y_train, y_train_pred)

print('Dataset1: gscv DT - train')

print('--------------------------')

print('f1_score: ', f1)

print('accuracy_score: ', acc)

print('Confusion Matrix: ')

print(cm)
# Predict (val)

y_val_pred = cv_dt.predict(X_val)



# Model evaluation (train)

f1 = f1_score(y_val, y_val_pred)

acc = accuracy_score(y_val, y_val_pred)

cm = confusion_matrix(y_val, y_val_pred)

print('Dataset1: gscv DT - val')

print('------------------------')

print('f1_score: ', f1)

print('accuracy_score: ', acc)

print('Confusion Matrix: ')

print(cm)
submission_dtgscv = pd.DataFrame()

submission_dtgscv['Survived_1_year'] = cv_dt.predict(test)

submission_dtgscv.head()
submission_dtgscv.to_csv("submission_dtgscv.csv",index=False)
%%time

dtgs2 = DecisionTreeClassifier(random_state=50)

cv_dt2 = GridSearchCV(estimator=dtgs2, param_grid=params, scoring = 'f1', cv= 5)

cv_dt2.fit(X_train2, y_train2)

cv_dt2.best_params_
# Predict (train)

y_train_pred = cv_dt2.predict(X_train2)



# Model evaluation (train)

f1 = f1_score(y_train2, y_train_pred)

acc = accuracy_score(y_train2, y_train_pred)

cm = confusion_matrix(y_train2, y_train_pred)

print('Dataset2: gscv DT - train')

print('--------------------------')

print('f1_score: ', f1)

print('accuracy_score: ', acc)

print('Confusion Matrix: ')

print(cm)
# Predict (val)

y_val_pred = cv_dt2.predict(X_val2)



# Model evaluation (train)

f1 = f1_score(y_val2, y_val_pred)

acc = accuracy_score(y_val2, y_val_pred)

cm = confusion_matrix(y_val2, y_val_pred)

print('Dataset2: gscv DT - val')

print('------------------------')

print('f1_score: ', f1)

print('accuracy_score: ', acc)

print('Confusion Matrix: ')

print(cm)
submission_dtgscv2 = pd.DataFrame()

submission_dtgscv2['Survived_1_year'] = cv_dt2.predict(test2)

submission_dtgscv2.head()
submission_dtgscv2.to_csv("submission_dtgscv2.csv",index=False)
from sklearn.ensemble import RandomForestClassifier

rf0 = RandomForestClassifier()
param_grid = { 

     'n_estimators': [55,60,65,70,80,100],

     'max_features': ['auto', 'sqrt', 'log2'],

     'max_depth' : [12,13,14,15,16],

     'min_samples_leaf': [1,2,3,4],

     'criterion' :['gini', 'entropy']

}
%%time

cv_rf = GridSearchCV(estimator=rf0, param_grid=param_grid, scoring='f1', cv= 5, verbose = 200)

cv_rf.fit(X_train, y_train)
cv_rf.best_params_
# Predict (train)

y_train_pred = cv_rf.predict(X_train)



# Model evaluation (train)

f1 = f1_score(y_train, y_train_pred)

acc = accuracy_score(y_train, y_train_pred)

cm = confusion_matrix(y_train, y_train_pred)

print('Dataset1: gscv RF - train')

print('--------------------------')

print('f1_score: ', f1)

print('accuracy_score: ', acc)

print('Confusion Matrix: ')

print(cm)
# Predict (val)

y_val_pred = cv_rf.predict(X_val)



# Model evaluation (train)

f1 = f1_score(y_val, y_val_pred)

acc = accuracy_score(y_val, y_val_pred)

cm = confusion_matrix(y_val, y_val_pred)

print('Dataset1: gscv RF - val')

print('-------------------------')

print('f1_score: ', f1)

print('accuracy_score: ', acc)

print('Confusion Matrix: ')

print(cm)
submission_rfgscv = pd.DataFrame()

submission_rfgscv['Survived_1_year'] = cv_rf.predict(test)

submission_rfgscv.head()
submission_rfgscv.to_csv("submission_rfgscv.csv",index=False)
%%time

cv_rf2 = GridSearchCV(estimator=rf0, param_grid=param_grid, scoring='f1', cv= 5, verbose = 200)

cv_rf2.fit(X_train2, y_train2)
cv_rf2.best_params_ 
# Predict (train)

y_train_pred = cv_rf2.predict(X_train2)



# Model evaluation (train)

f1 = f1_score(y_train2, y_train_pred)

acc = accuracy_score(y_train2, y_train_pred)

cm = confusion_matrix(y_train2, y_train_pred)

print('Dataset2: gscv RF - train')

print('---------------------------')

print('f1_score: ', f1)

print('accuracy_score: ', acc)

print('Confusion Matrix: ')

print(cm)
# Predict (val)

y_val_pred = cv_rf2.predict(X_val2)



# Model evaluation (train)

f1 = f1_score(y_val2, y_val_pred)

acc = accuracy_score(y_val2, y_val_pred)

cm = confusion_matrix(y_val2, y_val_pred)

print('Dataset2: gscv RF - val')

print('-------------------------')

print('f1_score: ', f1)

print('accuracy_score: ', acc)

print('Confusion Matrix: ')

print(cm)
submission_rfgscv2 = pd.DataFrame()

submission_rfgscv2['Survived_1_year'] = cv_rf2.predict(test2)

submission_rfgscv2.head()
submission_rfgscv2.to_csv("submission_rfgscv2.csv",index=False)
import xgboost as xgb

from xgboost.sklearn import XGBClassifier
param_test1 = {'learning_rate': [0.09, 0.1, 0.2, 0.3, 0.4, 0.5], 'max_depth': [4,5,6,8], 'min_child_weight':[1,2], 'n_estimators': [30,40,45,50], 'subsample':[0.6, 0.8]}
from sklearn.model_selection import GridSearchCV

xgb1 = XGBClassifier(seed=27)

xgb_gs1 = GridSearchCV(estimator = xgb1, param_grid = param_test1, scoring='f1',n_jobs=-1,cv=5)
%%time

eval_set = [(X_train, y_train)]

xgb_gs1.fit(X_train, y_train, early_stopping_rounds=10, eval_metric=["error", "logloss"], eval_set=eval_set, verbose=True)
xgb_gs1.best_params_
# Predict (train)

y_train_pred = xgb_gs1.predict(X_train)



# Model evaluation (train)

f1 = f1_score(y_train, y_train_pred)

acc = accuracy_score(y_train, y_train_pred)

cm = confusion_matrix(y_train, y_train_pred)

print('Dataset1: gscv XGB - train')

print('---------------------------')

print('f1_score: ', f1)

print('accuracy_score: ', acc)

print('Confusion Matrix: ')

print(cm) 
# Predict (val)

y_val_pred = xgb_gs1.predict(X_val)



# Model evaluation (train)

f1 = f1_score(y_val, y_val_pred)

acc = accuracy_score(y_val, y_val_pred)

cm = confusion_matrix(y_val, y_val_pred)

print('Dataset1: gscv XGB - val')

print('-------------------------')

print('f1_score: ', f1)

print('accuracy_score: ', acc)

print('Confusion Matrix: ')

print(cm)
submission_xgbgscv = pd.DataFrame()

submission_xgbgscv['Survived_1_year'] = xgb_gs1.predict(test)

submission_xgbgscv.head()
submission_xgbgscv.to_csv("submission_xgbgscv.csv",index=False)
xgb_clf = xgb.XGBClassifier(**xgb_gs1.best_params_, seed = 10)



# Learn the model with training data

xgb_clf.fit(X_train, y_train)



eval_set = [(X_train, y_train)]

xgb_clf.fit(X_train, y_train, early_stopping_rounds=10, eval_metric=["error", "logloss"], eval_set=eval_set, verbose=True)
# Plot the top 100 important features

imp_feat_xgb=pd.Series(xgb_clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)

imp_feat_xgb[:100].plot(kind='bar',title='Top 50 Important features as per XGBoost', figsize=(20,10))

plt.ylabel('Feature Importance Score')

plt.subplots_adjust(bottom=0.25)

plt.savefig('FeatureImportance.png')

plt.show()
# Selecting features with importance based on the threshold of 0.01

from sklearn.feature_selection import SelectFromModel



threshold = 0.01



sfm = SelectFromModel(xgb_clf, threshold=threshold)

sfm.fit(X_train, y_train)

X_train_t = sfm.transform(X_train)
X_val_t = sfm.transform(X_val)

test_t = sfm.transform(test)

print('Train shape: ', X_train_t.shape)

print('Val shape: ', X_val_t.shape)

print('Test shape: ', test_t.shape)
xgb_clf2 = xgb.XGBClassifier(**xgb_gs1.best_params_, seed = 10)



# Learn the model on transformed data

xgb_clf2.fit(X_train_t, y_train)



eval_set = [(X_train_t, y_train)]

xgb_clf2.fit(X_train_t, y_train, early_stopping_rounds=10, eval_metric=["error", "logloss"], eval_set=eval_set, verbose=True)
# Predict (train)

y_train_pred = xgb_clf2.predict(X_train_t)



# Model evaluation (train)

f1 = f1_score(y_train, y_train_pred)

acc = accuracy_score(y_train, y_train_pred)

cm = confusion_matrix(y_train, y_train_pred)

print('Dataset1: gscv-fea.imp. XGB - train')

print('-----------------------------------')

print('f1_score: ', f1)

print('accuracy_score: ', acc)

print('Confusion Matrix: ')

print(cm) 
# Predict (val)

y_val_pred = xgb_clf2.predict(X_val_t)



# Model evaluation (train)

f1 = f1_score(y_val, y_val_pred)

acc = accuracy_score(y_val, y_val_pred)

cm = confusion_matrix(y_val, y_val_pred)

print('Dataset1: gscv-fea.imp. XGB - val')

print('----------------------------------')

print('f1_score: ', f1)

print('accuracy_score: ', acc)

print('Confusion Matrix: ')

print(cm)
submission_xgbgscvfi = pd.DataFrame()

submission_xgbgscvfi['Survived_1_year'] = xgb_clf2.predict(test_t)

submission_xgbgscvfi.head()
submission_xgbgscvfi.to_csv("submission_xgbgscvfi.csv",index=False)
threshold = 0.015



sfm = SelectFromModel(xgb_clf, threshold=threshold)

sfm.fit(X_train, y_train)

X_train_t = sfm.transform(X_train)
X_val_t = sfm.transform(X_val)

test_t = sfm.transform(test)

print('Train shape: ', X_train_t.shape)

print('Val shape: ', X_val_t.shape)

print('Test shape: ', test_t.shape)
xgb_clf2 = xgb.XGBClassifier(**xgb_gs1.best_params_, seed = 10)



# Learn the model on transformed data

xgb_clf2.fit(X_train_t, y_train)



eval_set = [(X_train_t, y_train)]

xgb_clf2.fit(X_train_t, y_train, early_stopping_rounds=10, eval_metric=["error", "logloss"], eval_set=eval_set, verbose=True)
# Predict (train)

y_train_pred = xgb_clf2.predict(X_train_t)



# Model evaluation (train)

f1 = f1_score(y_train, y_train_pred)

acc = accuracy_score(y_train, y_train_pred)

cm = confusion_matrix(y_train, y_train_pred)

print('Dataset1: gscv-fea.imp. XGB - train')

print('-----------------------------------')

print('f1_score: ', f1)

print('accuracy_score: ', acc)

print('Confusion Matrix: ')

print(cm) 
# Predict (val)

y_val_pred = xgb_clf2.predict(X_val_t)



# Model evaluation (train)

f1 = f1_score(y_val, y_val_pred)

acc = accuracy_score(y_val, y_val_pred)

cm = confusion_matrix(y_val, y_val_pred)

print('Dataset1: gscv-fea.imp. XGB - val')

print('----------------------------------')

print('f1_score: ', f1)

print('accuracy_score: ', acc)

print('Confusion Matrix: ')

print(cm)
submission_xgbgscvfi015 = pd.DataFrame()

submission_xgbgscvfi015['Survived_1_year'] = xgb_clf2.predict(test_t)

submission_xgbgscvfi015.head()
submission_xgbgscvfi015.to_csv("submission_xgbgscvfi015.csv",index=False)
xgb2 = XGBClassifier(seed=27)

xgb_gs2 = GridSearchCV(estimator = xgb2, param_grid = param_test1, scoring='f1',n_jobs=-1,cv=5)
%%time

eval_set = [(X_train2, y_train2)]

xgb_gs2.fit(X_train2, y_train2, early_stopping_rounds=10, eval_metric=["error", "logloss"], eval_set=eval_set, verbose=True)
xgb_gs2.best_params_
# Predict (train)

y_train_pred = xgb_gs2.predict(X_train2)



# Model evaluation (train)

f1 = f1_score(y_train2, y_train_pred)

acc = accuracy_score(y_train2, y_train_pred)

cm = confusion_matrix(y_train2, y_train_pred)

print('Dataset2: gscv XGB - train')

print('---------------------------')

print('f1_score: ', f1)

print('accuracy_score: ', acc)

print('Confusion Matrix: ')

print(cm) 
# Predict (val)

y_val_pred = xgb_gs2.predict(X_val2)



# Model evaluation (train)

f1 = f1_score(y_val2, y_val_pred)

acc = accuracy_score(y_val2, y_val_pred)

cm = confusion_matrix(y_val2, y_val_pred)

print('Dataset2: gscv XGB - val')

print('-------------------------')

print('f1_score: ', f1)

print('accuracy_score: ', acc)

print('Confusion Matrix: ')

print(cm)
submission_xgbgscv2 = pd.DataFrame()

submission_xgbgscv2['Survived_1_year'] = xgb_gs2.predict(test2)

submission_xgbgscv2.head()
submission_xgbgscv2.to_csv("submission_xgbgscv2.csv",index=False)
xgb_clf3 = xgb.XGBClassifier(**xgb_gs2.best_params_, seed = 10)



# Learn the model with training data

xgb_clf3.fit(X_train2, y_train2)



eval_set = [(X_train2, y_train2)]

xgb_clf3.fit(X_train2, y_train2, early_stopping_rounds=10, eval_metric=["error", "logloss"], eval_set=eval_set, verbose=True)
# Plot the top 100 important features

imp_feat_xgb=pd.Series(xgb_clf3.feature_importances_, index=X_train2.columns).sort_values(ascending=False)

imp_feat_xgb[:100].plot(kind='bar',title='Dataset2: Top 50 Important features as per XGBoost', figsize=(20,10))

plt.ylabel('Feature Importance Score')

plt.subplots_adjust(bottom=0.25)

plt.savefig('FeatureImportance.png')

plt.show()
# Selecting features with importance based on the threshold of 0.01

from sklearn.feature_selection import SelectFromModel



threshold = 0.01



sfm3 = SelectFromModel(xgb_clf3, threshold=threshold)

sfm3.fit(X_train2, y_train2)

X_train_t2 = sfm3.transform(X_train2)
X_val_t2 = sfm3.transform(X_val2)

test_t2 = sfm3.transform(test2)

print('Train shape: ', X_train_t2.shape)

print('Val shape: ', X_val_t2.shape)

print('Test shape: ', test_t2.shape)
xgb_clf4 = xgb.XGBClassifier(**xgb_gs2.best_params_, seed = 10)



# Learn the model on transformed data

xgb_clf4.fit(X_train_t2, y_train2)



eval_set = [(X_train_t2, y_train2)]

xgb_clf4.fit(X_train_t2, y_train2, early_stopping_rounds=10, eval_metric=["error", "logloss"], eval_set=eval_set, verbose=True)
# Predict (train)

y_train_pred = xgb_clf4.predict(X_train_t2)



# Model evaluation (train)

f1 = f1_score(y_train2, y_train_pred)

acc = accuracy_score(y_train2, y_train_pred)

cm = confusion_matrix(y_train2, y_train_pred)

print('Dataset2: gscv-fea.imp. XGB - train')

print('-----------------------------------')

print('f1_score: ', f1)

print('accuracy_score: ', acc)

print('Confusion Matrix: ')

print(cm) 
# Predict (val)

y_val_pred = xgb_clf4.predict(X_val_t2)



# Model evaluation (train)

f1 = f1_score(y_val2, y_val_pred)

acc = accuracy_score(y_val2, y_val_pred)

cm = confusion_matrix(y_val2, y_val_pred)

print('Dataset2: gscv-fea.imp. XGB - val')

print('----------------------------------')

print('f1_score: ', f1)

print('accuracy_score: ', acc)

print('Confusion Matrix: ')

print(cm)
submission_xgbgscvfi2 = pd.DataFrame()

submission_xgbgscvfi2['Survived_1_year'] = xgb_clf4.predict(test_t2)

submission_xgbgscvfi2.head()
submission_xgbgscvfi2.to_csv("submission_xgbgscvfi2.csv",index=False)
import lightgbm as lgb

from sklearn.model_selection import RandomizedSearchCV

clf = lgb.LGBMClassifier(silent=True, random_state = 123, early_stopping_rounds=5, metric='f1', n_jobs=4)
from scipy.stats import randint as sp_randint

from scipy.stats import uniform as sp_uniform

params ={'cat_smooth' : sp_randint(1, 100), 'min_data_per_group': sp_randint(1,1000), 'max_cat_threshold': sp_randint(1,100),

        'learning_rate': [0.07,0.08,0.81,0.82,0.83,0.84,0.85,0.86,0.87,0.88,0.89, 0.09,0.1], 'num_iterations': sp_randint(1000,3000),

        'scale_pos_weight': sp_randint(1,15),'colsample_bytree': sp_uniform(loc=0.4, scale=0.6), 'num_leaves': sp_randint(500, 5000),  

        'min_child_samples': sp_randint(100,500), 'min_child_weight': [1e-2, 1e-1, 1, 1e1], 'max_bin': sp_randint(100, 1500), 'max_depth': sp_randint(1, 15), 

        'min_data_in_leaf': sp_randint(500,3500), 'reg_lambda': sp_randint(1, 30), 'boosting': ['goss', 'dart']}
fit_params={"early_stopping_rounds":5, 

            "eval_metric" : 'auc', 

            "eval_set" : [(X_train, y_train),(X_val,y_val)],

            'eval_names': ['train','valid'],

            'verbose': 300,

            'categorical_feature': 'auto'}
gs = RandomizedSearchCV( estimator=clf, param_distributions=params, scoring='f1',cv=5, refit=True,random_state=135,verbose=True)
gs.fit(X_train, y_train, **fit_params)

print('Best score reached: {} with params: {} '.format(gs.best_score_, gs.best_params_))
chk1_params = {**gs.best_params_, 'scoring':'f1'}

chk1_params
lgbm_train1 = lgb.Dataset(X_train, y_train, categorical_feature=cat_columns)

lgbm_val1 = lgb.Dataset(X_val, y_val, reference = lgbm_train1)
model_lgbm_chk1 = lgb.train(chk1_params,lgbm_train1,

                num_boost_round=1000,

                valid_sets=[lgbm_train1, lgbm_val1],

                feature_name=['f' + str(i + 1) for i in range(X_train.shape[-1])],

                categorical_feature= [150], 

                verbose_eval=100)
# Predict (train)

y_train_pred = model_lgbm_chk1.predict(X_train, type = 'response')

y_train_pred = np.absolute(y_train_pred)

y_train_pred = y_train_pred.round()



# Model evaluation (train)

f1 = f1_score(y_train, y_train_pred)

acc = accuracy_score(y_train, y_train_pred)

cm = confusion_matrix(y_train, y_train_pred)

print('Dataset1: gscv LGBM1 - train')

print('-----------------------------')

print('f1_score: ', f1)

print('accuracy_score: ', acc)

print('Confusion Matrix: ')

print(cm) 
# Predict (val)

y_val_pred = model_lgbm_chk1.predict(X_val)

y_val_pred = np.absolute(y_val_pred)

y_val_pred = y_val_pred.round()



# Model evaluation (train)

f1 = f1_score(y_val, y_val_pred)

acc = accuracy_score(y_val, y_val_pred)

cm = confusion_matrix(y_val, y_val_pred)

print('Dataset1: gscv LGBM1 - val')

print('---------------------------')

print('f1_score: ', f1)

print('accuracy_score: ', acc)

print('Confusion Matrix: ')

print(cm)
submission_cvlgbm1 = pd.DataFrame()

submission_cvlgbm1['Survived_1_year'] = model_lgbm_chk1.predict(test)

submission_cvlgbm1['Survived_1_year'] = submission_cvlgbm1['Survived_1_year'].round().astype(int)

submission_cvlgbm1.head()
submission_cvlgbm1.to_csv("submission_cvlgbm1.csv",index=False) 
clfnew = lgb.LGBMClassifier(silent=True, random_state = 135, early_stopping_rounds=5, metric='f1', n_jobs=4)
fit_params={"early_stopping_rounds":5, 

            "eval_metric" : 'auc', 

            "eval_set" : [(X_train_t, y_train),(X_val_t,y_val)],

            'eval_names': ['train','valid'],

            'verbose': 300,

            'categorical_feature': 'auto'}
gs = RandomizedSearchCV( estimator=clfnew, param_distributions=params, scoring='f1',cv=5, refit=True,random_state=135,verbose=True)
gs.fit(X_train_t, y_train, **fit_params)

print('Best score reached: {} with params: {} '.format(gs.best_score_, gs.best_params_))
chk2_params = {**gs.best_params_, 'scoring':'f1'}

chk2_params
lgbm_train2 = lgb.Dataset(X_train_t, y_train, categorical_feature=cat_columns)

lgbm_val2 = lgb.Dataset(X_val_t, y_val, reference = lgbm_train2)
model_lgbm_chk2 = lgb.train(chk2_params,

                lgbm_train2,

                num_boost_round=1000,

                valid_sets=[lgbm_train2, lgbm_val2],

                feature_name=['f' + str(i + 1) for i in range(X_train_t.shape[-1])],

                categorical_feature= [150],

                verbose_eval=100)
# Predict (train)

y_train_pred = model_lgbm_chk2.predict(X_train_t, pred_contrib=False)

y_train_pred = np.absolute(y_train_pred.round())

# Model evaluation (train)

f1 = f1_score(y_train, y_train_pred)

acc = accuracy_score(y_train, y_train_pred)

cm = confusion_matrix(y_train, y_train_pred)

print('Dataset1: gscv LGBM2 - train')

print('-----------------------------')

print('f1_score: ', f1)

print('accuracy_score: ', acc)

print('Confusion Matrix: ')

print(cm) 
# Predict (val)

y_val_pred = model_lgbm_chk2.predict(X_val_t)

y_val_pred = np.absolute(y_val_pred.round())

# Model evaluation (train)

f1 = f1_score(y_val, y_val_pred)

acc = accuracy_score(y_val, y_val_pred)

cm = confusion_matrix(y_val, y_val_pred)

print('Dataset1: gscv LGBM2 - val')

print('---------------------------')

print('f1_score: ', f1)

print('accuracy_score: ', acc)

print('Confusion Matrix: ')

print(cm)
submission_cvlgbm2 = pd.DataFrame()

submission_cvlgbm2['Survived_1_year'] = model_lgbm_chk2.predict(test_t)

submission_cvlgbm2['Survived_1_year'] = submission_cvlgbm2['Survived_1_year'].round().astype(int)

submission_cvlgbm2.head()
submission_cvlgbm2.to_csv("submission_cvlgbm2.csv",index=False) 
import lightgbm as lgb

from sklearn.model_selection import RandomizedSearchCV

clf2 = lgb.LGBMClassifier(silent=True, random_state = 123, early_stopping_rounds=5, metric='f1', n_jobs=4)
params2 ={'cat_smooth' : sp_randint(1, 100), 'min_data_per_group': sp_randint(1,1000), 'max_cat_threshold': sp_randint(1,100),

        'learning_rate': [0.07,0.08,0.81,0.82,0.83,0.84,0.85,0.86,0.87,0.88,0.89, 0.09,0.1], 'num_iterations': sp_randint(1000,3000),

        'scale_pos_weight': sp_randint(1,15),'colsample_bytree': sp_uniform(loc=0.4, scale=0.6), 'num_leaves': sp_randint(500, 5000),  

        'min_child_samples': sp_randint(100,500), 'min_child_weight': [1e-2, 1e-1, 1, 1e1], 'max_bin': sp_randint(100, 1500), 'max_depth': sp_randint(1, 15), 

        'min_data_in_leaf': sp_randint(500,3500), 'reg_lambda': sp_randint(1, 30), 'boosting': ['goss', 'dart']}
fit_params={"early_stopping_rounds":5, 

            "eval_metric" : 'auc', 

            "eval_set" : [(X_train2, y_train2),(X_val2,y_val2)],

            'eval_names': ['train','valid'],

            'verbose': 300,

            'categorical_feature': 'auto'}
gs2 = RandomizedSearchCV( estimator=clf2, param_distributions=params2, scoring='f1',cv=5, refit=True,random_state=123,verbose=True)
gs2.fit(X_train2, y_train2, **fit_params)

print('Best score reached: {} with params: {} '.format(gs2.best_score_, gs2.best_params_))
chk1_params = {**gs2.best_params_, 'scoring':'f1'}

chk1_params
lgbm_train3 = lgb.Dataset(X_train2, y_train2, categorical_feature=cat_columns)

lgbm_val3 = lgb.Dataset(X_val2, y_val2, reference = lgbm_train3)
model_lgbm_chk3 = lgb.train(chk1_params,lgbm_train3,

                num_boost_round=1000,

                valid_sets=[lgbm_train3, lgbm_val3],

                feature_name=['f' + str(i + 1) for i in range(X_train2.shape[-1])],

                categorical_feature= [150], 

                verbose_eval=100)
# Predict (train)

y_train_pred = model_lgbm_chk3.predict(X_train2, type = 'response')

y_train_pred = np.absolute(y_train_pred)

y_train_pred = y_train_pred.round()



# Model evaluation (train)

f1 = f1_score(y_train2, y_train_pred)

acc = accuracy_score(y_train2, y_train_pred)

cm = confusion_matrix(y_train2, y_train_pred)

print('Dataset2: gscv LGBM3 - train')

print('-----------------------------')

print('f1_score: ', f1)

print('accuracy_score: ', acc)

print('Confusion Matrix: ')

print(cm) 
# Predict (val)

y_val_pred = model_lgbm_chk3.predict(X_val2)

y_val_pred = np.absolute(y_val_pred)

y_val_pred = y_val_pred.round()



# Model evaluation (train)

f1 = f1_score(y_val2, y_val_pred)

acc = accuracy_score(y_val2, y_val_pred)

cm = confusion_matrix(y_val2, y_val_pred)

print('Dataset2: gscv LGBM3 - val')

print('---------------------------')

print('f1_score: ', f1)

print('accuracy_score: ', acc)

print('Confusion Matrix: ')

print(cm)
submission_cvlgbm3 = pd.DataFrame()

submission_cvlgbm3['Survived_1_year'] = model_lgbm_chk3.predict(test2)

submission_cvlgbm3['Survived_1_year'] = submission_cvlgbm3['Survived_1_year'].round().astype(int)

submission_cvlgbm3.head()
submission_cvlgbm3.to_csv("submission_cvlgbm3.csv",index=False) 
clf2new = lgb.LGBMClassifier(silent=True, random_state = 123, early_stopping_rounds=5, metric='f1', n_jobs=4)
fit_params={"early_stopping_rounds":5, 

            "eval_metric" : 'auc', 

            "eval_set" : [(X_train_t2, y_train2),(X_val_t2,y_val2)],

            'eval_names': ['train','valid'],

            'verbose': 300,

            'categorical_feature': 'auto'}
gs2 = RandomizedSearchCV( estimator=clf2new, param_distributions=params2, scoring='f1',cv=5, refit=True,random_state=123,verbose=True)
gs2.fit(X_train_t2, y_train2, **fit_params)

print('Best score reached: {} with params: {} '.format(gs2.best_score_, gs2.best_params_))
chk2_params = {**gs2.best_params_, 'scoring':'f1'}

chk2_params
lgbm_train4 = lgb.Dataset(X_train_t2, y_train2, categorical_feature=cat_columns)

lgbm_val4 = lgb.Dataset(X_val_t2, y_val2, reference = lgbm_train4)
model_lgbm_chk4 = lgb.train(chk2_params,

                lgbm_train4,

                num_boost_round=1000,

                valid_sets=[lgbm_train4, lgbm_val4],

                feature_name=['f' + str(i + 1) for i in range(X_train_t2.shape[-1])],

                categorical_feature= [150],

                verbose_eval=100)
# Predict (train)

y_train_pred = model_lgbm_chk4.predict(X_train_t2, pred_contrib=False)

y_train_pred = np.absolute(y_train_pred.round())

# Model evaluation (train)

f1 = f1_score(y_train2, y_train_pred.round())

acc = accuracy_score(y_train2, y_train_pred.round())

cm = confusion_matrix(y_train2, y_train_pred.round())

print('Dataset1: gscv LGBM4 - train')

print('-----------------------------')

print('f1_score: ', f1)

print('accuracy_score: ', acc)

print('Confusion Matrix: ')

print(cm) 
# Predict (val)

y_val_pred = model_lgbm_chk4.predict(X_val_t2)

y_val_pred = np.absolute(y_val_pred.round())

# Model evaluation (train)

f1 = f1_score(y_val2, y_val_pred.round())

acc = accuracy_score(y_val2, y_val_pred.round())

cm = confusion_matrix(y_val2, y_val_pred.round())

print('Dataset1: gscv LGBM4 - val')

print('---------------------------')

print('f1_score: ', f1)

print('accuracy_score: ', acc)

print('Confusion Matrix: ')

print(cm)
submission_cvlgbm4 = pd.DataFrame()

submission_cvlgbm4['Survived_1_year'] = model_lgbm_chk4.predict(test_t2)

submission_cvlgbm4['Survived_1_year'] = submission_cvlgbm4['Survived_1_year'].round().astype(int)

submission_cvlgbm4.head()
submission_cvlgbm4.to_csv("submission_cvlgbm4.csv",index=False) 
from sklearn.feature_selection import RFE #importing RFE class from sklearn library



rfe_xgb1= RFE(estimator= xgb_clf2 , step = 1) # with Random Forest



# Fit the function for ranking the features

fit = rfe_xgb1.fit(X_train_t, y_train)



print("Num Features: %d" % fit.n_features_)

print("Selected Features: %s" % fit.support_)

print("Feature Ranking: %s" % fit.ranking_)
# Transforming the data

X_train_t_rfe = rfe_xgb1.transform(X_train_t)

X_val_t_rfe = rfe_xgb1.transform(X_val_t)

test_t_rfe = rfe_xgb1.transform(test_t)

# Fitting our baseline model with the transformed data

rfe_xgb_model = xgb_clf2.fit(X_train_t_rfe, y_train)
# Predict (train)

y_train_pred = rfe_xgb_model.predict(X_train_t_rfe)



# Model evaluation (train)

f1 = f1_score(y_train, y_train_pred)

acc = accuracy_score(y_train, y_train_pred)

cm = confusion_matrix(y_train, y_train_pred)

print('Dataset1: RFE on xgb_clf2 - train')

print('----------------------------------')

print('f1_score: ', f1)

print('accuracy_score: ', acc)

print('Confusion Matrix: ')

print(cm) 
# Predict (val)

y_val_pred = rfe_xgb_model.predict(X_val_t_rfe)



# Model evaluation (train)

f1 = f1_score(y_val, y_val_pred)

acc = accuracy_score(y_val, y_val_pred)

cm = confusion_matrix(y_val, y_val_pred)

print('Dataset1: RFE on xgb_cl2 - val')

print('-------------------------------')

print('f1_score: ', f1)

print('accuracy_score: ', acc)

print('Confusion Matrix: ')

print(cm)
submission_rfexgb = pd.DataFrame()

submission_rfexgb['Survived_1_year'] = rfe_xgb_model.predict(test_t_rfe)

submission_rfexgb['Survived_1_year'] = submission_rfexgb['Survived_1_year'].round().astype(int)

submission_rfexgb.head()
submission_rfexgb.to_csv("submission_rfexgb.csv",index=False) 
X.head()
from sklearn import preprocessing

Xscaled = X.copy()

Xscaled[['Patient_Body_Mass_Index']] = preprocessing.scale(Xscaled[['Patient_Body_Mass_Index']])

Xscaled.head()
testscaled = test.copy()

testscaled[['Patient_Body_Mass_Index']] = preprocessing.scale(testscaled[['Patient_Body_Mass_Index']])

testscaled.head()
# For scaled data

X_trains, X_vals, y_trains, y_vals = train_test_split(Xscaled,y,test_size=0.3, random_state = 50)
xgbs = XGBClassifier(seed=27)

xgbs_gs = GridSearchCV(estimator = xgbs, param_grid = param_test1, scoring='f1',n_jobs=-1,cv=5)
%%time

eval_set = [(X_trains, y_trains)]

xgbs_gs.fit(X_trains, y_trains, early_stopping_rounds=10, eval_metric=["error", "logloss"], eval_set=eval_set, verbose=True)
xgbs_gs.best_params_
# Predict (train)

y_train_pred = xgbs_gs.predict(X_trains)



# Model evaluation (train)

f1 = f1_score(y_trains, y_train_pred)

acc = accuracy_score(y_trains, y_train_pred)

cm = confusion_matrix(y_trains, y_train_pred)

print('Dataset1: gscv XGB - train-scaled')

print('----------------------------------')

print('f1_score: ', f1)

print('accuracy_score: ', acc)

print('Confusion Matrix: ')

print(cm) 
# Predict (val)

y_val_pred = xgbs_gs.predict(X_vals)



# Model evaluation (train)

f1 = f1_score(y_vals, y_val_pred)

acc = accuracy_score(y_vals, y_val_pred)

cm = confusion_matrix(y_vals, y_val_pred)

print('Dataset2: gscv XGB - val=scaled')

print('--------------------------------')

print('f1_score: ', f1)

print('accuracy_score: ', acc)

print('Confusion Matrix: ')

print(cm)
submission_xgbs = pd.DataFrame()

submission_xgbs['Survived_1_year'] = xgbs_gs.predict(testscaled)

submission_xgbs['Survived_1_year'] = submission_xgbs['Survived_1_year'].round().astype(int)

submission_xgbs.head()
submission_xgbs.to_csv("submission_xgbs.csv",index=False) 