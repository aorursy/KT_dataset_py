#import all dependencies

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline 

sns.set()
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
perform = pd.read_csv('/kaggle/input/students-performance-in-exams/StudentsPerformance.csv')

perform.head()
perform.info()
perform.describe()
perform.isnull().sum()
def box_chart(feature):

    female = perform[perform['gender']==1][feature].value_counts()

    male = perform[perform['gender']==0][feature].value_counts()

    df = pd.DataFrame([male,female])

    df.index = ['male','female']

    df.plot(kind='bar',stacked=True, figsize=(10,5))
def cross_tab_(feature):

    print(pd.crosstab(perform[feature],perform['gender']).apply(lambda x: x*100/x.sum(), axis=1))
perform['gender'] = perform['gender'].apply(lambda x: 1 if x == 'female' else 0)
perform.head()
box_chart('test preparation course')

cross_tab_('test preparation course')
box_chart('parental level of education')

cross_tab = pd.crosstab(perform['parental level of education'],perform['gender']).apply(lambda x: x*100/x.sum(), axis=1)

print(cross_tab)
box_chart('race/ethnicity')

cross_tab_('race/ethnicity')
box_chart('lunch')

cross_tab_('lunch')
perform['math score'].value_counts()
plt.figure(figsize=(20,10))

sns.countplot('math score', hue='gender', data=perform)
plt.figure(figsize=(20,10))

sns.countplot('math score', hue='gender', data=perform)

plt.xlim(30,45)
plt.figure(figsize=(20,10))

sns.countplot('math score', hue='gender', data=perform)

plt.xlim(45,55)
plt.figure(figsize=(20,10))

sns.countplot('math score', hue='gender', data=perform)

plt.xlim(55,65)
plt.figure(figsize=(20,10))

sns.countplot('math score', hue='gender', data=perform)

plt.xlim(65,75)
plt.figure(figsize=(20,10))

sns.countplot('math score', hue='gender', data=perform)

plt.xlim(75,85)
plt.figure(figsize=(20,10))

sns.countplot('reading score', hue='gender', data=perform)
plt.figure(figsize=(20,10))

sns.countplot('writing score', hue='gender', data=perform)
plt.figure(figsize=(8,6))

sns.heatmap(perform.corr(), annot = True)
perform.head()
parent_ed_map = {'some college': 0.43, "associate's degree": 0.47, 'high school': 0.7,

                 'some high school': 0.5, "bachelor's degree": 0.46, "master's degree": 0.3}

perform['parental level of education'] = perform['parental level of education'].map(parent_ed_map)
perform.head()
perform['lunch'] = perform['lunch'].apply(lambda x: 0 if x == 'free/reduced' else 1)
perform.head()
perform['test preparation course'] = perform['test preparation course'].apply(lambda x: 1 if x == 'completed' else 0)
perform.head()
group_map = {'group A': 0.2, 'group B': 0.55, 'group C': 0.6, 'group D': 0.33, 'group E': 0.3}

perform['race/ethnicity'] = perform['race/ethnicity'].map(group_map)
perform.head()
perform['math score'] = perform['math score'].apply(lambda x: x/100)

perform['reading score'] = perform['reading score'].apply(lambda x: x/100)

perform['writing score'] = perform['writing score'].apply(lambda x: x/100)
perform.head()
from sklearn.model_selection import train_test_split
X = perform.drop(['gender', 'reading score'], axis=1)

y = perform['gender']
X.head()
y.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)
print(model.coef_, model.intercept_)
coeff_df = pd.DataFrame(model.coef_, X.columns,columns=['Coefficent'])

coeff_df
predictions = model.predict(X_test)

plt.scatter(y_test, predictions)
copy_pred = (predictions > 0.6)*1
sns.distplot((y_test-predictions), bins = 50)
TP = sum((y_test == copy_pred) & (copy_pred == 1))
FP = sum((copy_pred == 1) & (y_test != copy_pred))
precision = TP/(TP+FP)

print(precision)