# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd

import seaborn as sns

from matplotlib import pyplot as plt

%matplotlib inline# data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("/kaggle/input/train-and-test-attrition-data-set/train.csv")

test = pd.read_csv("/kaggle/input/train-and-test-attrition-data-set/test.csv")
print(train.shape)

train.head()
print(test.shape)

test.head()
print(test.isna().sum())#no-null value
print(test.isna().sum())
ytrain=train["Attrition"]

train1=train.drop(["Attrition"],axis=1)

train1.head()
list(train.select_dtypes(['object']).columns)

list(test.select_dtypes(['object']).columns)
train.Attrition.value_counts(normalize=False)
print(train[train.Attrition==0]["MonthlyIncome"].mean())

print(train[train.Attrition==1]["MonthlyIncome"].mean())
train.groupby('Attrition').mean()
for dept in train['Department'].unique():

    print('\n', dept, ':')

    print(train[train['Department']==dept]['JobRole'].value_counts())#for analyzing the job role in each department
sns.axes_style('whitegrid')

sns.catplot("Department", data=train, aspect=1, kind='count', hue='Attrition', palette=['C1', 'C0']).set_ylabels('Number of Employees')
sns.distplot(train['MonthlyIncome'])
sns.axes_style('whitegrid')

sns.catplot("OverTime", data=train, aspect=1, kind='count', hue='Attrition', palette=['C1', 'C0']).set_ylabels('Number of Employees')
plt.figure(figsize=(8,6))

sns.axes_style('whitegrid')

sns.catplot("JobRole", data=train, aspect=4, kind='count', hue='Attrition', palette=['C1', 'C0']).set_ylabels('Number of Employees')
plt.figure(figsize=(8,6))

sns.axes_style('whitegrid')

sns.catplot("Gender", data=train, aspect=4, kind='count', hue='Attrition', palette=['C1', 'C0']).set_ylabels('Number of Employees')
plt.figure(figsize=(8,6))

sns.axes_style('whitegrid')

sns.catplot("MaritalStatus", data=train, aspect=1, kind='count', hue='Attrition', palette=['C1', 'C0']).set_ylabels('Number of Employees')
sns.axes_style('whitegrid')

sns.catplot("BusinessTravel", data=train, aspect=1, kind='count', hue='Attrition', palette=['C1', 'C0']).set_ylabels('Number of Employees')
sns.axes_style('whitegrid')

sns.catplot("TotalWorkingYears", data=train, aspect=3, kind='count', hue='Attrition', palette=['C1', 'C0']).set_ylabels('Number of Employees')
sns.axes_style('whitegrid')

sns.catplot("EducationField", data=train, aspect=3, kind='count', hue='Attrition', palette=['C1', 'C0']).set_ylabels('Number of Employees')
ytrain.value_counts().plot(kind='bar', color=('C0','C1')).set_title('Attrition')
corr=train.corr().abs()
plt.figure(figsize=(15, 15))

sns.heatmap(corr, annot=True, cmap="RdYlGn", annot_kws={"size":10})

plt.figure(figsize=(15,6))

plt.style.use('seaborn-colorblind')

plt.grid(True,alpha=0.5)

sns.kdeplot(train.loc[train['Attrition'] == 0, 'Age'], label = 'Active Employee')

sns.kdeplot(train.loc[train['Attrition'] == 1, 'Age'], label = 'Ex-Employees')

plt.xlim(left=train.Age.min(), right=train.Age.max())

plt.xlabel('Age (years)')

plt.ylabel('Density')

plt.title('Age Distribution in Percent by Attrition Status');
plt.figure(figsize=(15,6))

plt.style.use('seaborn-colorblind')

plt.grid(True, alpha=0.5)

sns.kdeplot(train.loc[train['Attrition'] == 0, 'DistanceFromHome'], label = 'Active Employee')

sns.kdeplot(train.loc[train['Attrition'] == 1, 'DistanceFromHome'], label = 'Ex-Employees')

plt.xlim(left=train['DistanceFromHome'].min(), right=train['DistanceFromHome'].max())

plt.xlabel('distance')

plt.ylabel('Density')

plt.title('distance from home distribution by Attrition Status');
def column_index(train, query_cols):

    cols = train.columns.values

    sidx = np.argsort(cols)

    return sidx[np.searchsorted(cols,query_cols,sorter=sidx)]

column_index(train, ['Age', 'DistanceFromHome','MonthlyIncome','NumCompaniesWorked',

 'PercentSalaryHike',

 'PerformanceRating',

 'StockOptionLevel',

 'TotalWorkingYears',

 'TrainingTimesLastYear',

 'YearsAtCompany',

 'YearsInCurrentRole',

 'YearsSinceLastPromotion',

 'YearsWithCurrManager'])

##finding the index of the numerical coulmns in train data set I have defined this function.
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler() 

scaled_values = scaler.fit_transform(train.iloc[:,[ 1,  5, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26]]) 

train.iloc[:,[ 1,  5, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26]] = scaled_values
train.head()# numerical columns have been scaled.
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()
le_count = 0

for col in train.columns.values:

    if train[col].dtype == 'object':

        if len(list(train[col].unique())) <= 2:

            le.fit(train[col])

            train[col] = le.transform(train[col])

            le_count += 1

print('{} columns were label encoded.'.format(le_count))

        
train = pd.get_dummies(train, drop_first=True)  

print(train.shape)

train.head()
column_index(test, ['Age', 'DistanceFromHome','MonthlyIncome','NumCompaniesWorked',

 'PercentSalaryHike',

 'PerformanceRating',

 'StockOptionLevel',

 'TotalWorkingYears',

 'TrainingTimesLastYear',

 'YearsAtCompany',

 'YearsInCurrentRole',

 'YearsSinceLastPromotion',

 'YearsWithCurrManager'])
scaled_values1 = scaler.fit_transform(test.iloc[:,[ 1,  4, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25]]) 

test.iloc[:,[ 1,  4, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25]] = scaled_values1
le_count = 0

for col in test.columns.values:

    if test[col].dtype == 'object':

        if len(list(test[col].unique())) <= 2:

            le.fit(test[col])

            test[col] = le.transform(test[col])

            le_count += 1

print('{} columns were label encoded.'.format(le_count))

        
test = pd.get_dummies(test, drop_first=True)  

print(test.shape)

test.head()
train1=train.copy()

train1=train1.drop('Attrition',axis=1)  

y_train1=train['Attrition']
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score

from sklearn.metrics import classification_report
log= LogisticRegression(solver='lbfgs',C=0.5,max_iter=10000)
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score
kfold = KFold(n_splits=10,random_state=7)

scoring ='accuracy'

results =cross_val_score(log, train1, y_train1, cv=kfold,scoring=scoring)

print(results)

print("10-fold cross validation average accuracy: %.3f" % (results.mean()))
log.fit(train1,y_train1)

log.score(train1,y_train1)
importance = log.coef_[0]
for i,v in enumerate(importance):

	print('Feature: %0d, Score: %.5f' % (i,v))

# plot feature importance

plt.figure(figsize=(10, 16))

plt.bar([x for x in range(len(importance))], importance)

plt.show()
for i,v in enumerate(importance):

    if v>0:

        print('Feature: %0d, +ve Score: %.5f' % (i,v))
for i,v in enumerate(importance):

    if v<=0:

        print('Feature: %0d, +ve Score: %.5f' % (i,v))
train2=train1.iloc[:,[0,2,6,8,10,11,13,15,16,17,19,21,23,24,25,26,28,31,32,33,39,41]]#[0,1,2,3,4,5,6,9,10,11,13,15,16,22,24,26]

train2.head()
log.fit(train2,y_train1)

log.score(train2,y_train1)
test2=test.iloc[:,[0,2,6,8,10,11,13,15,16,17,19,21,23,24,25,26,28,31,32,33,39,41]]#[0,1,2,3,4,5,6,9,10,11,13,15,16,22,24,26]

test.head()
list(train2.columns)==list(test2.columns)
pred=log.predict_proba(test2)

df=pd.DataFrame(pred[:,1]).to_csv('prediction_log.csv')