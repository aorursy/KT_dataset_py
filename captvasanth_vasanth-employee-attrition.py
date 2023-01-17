# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn import preprocessing

import matplotlib.pyplot as plt 

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

import seaborn as sns







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df= pd.read_csv("../input/HR-Employee-Attrition.csv")
df.head()
df.info().sum()
print(list(df.columns))
df.drop(columns='Attrition').dtypes

print(df['Attrition'].dtype)

df.isna().sum()
df.duplicated().sum()
num_cols = df.select_dtypes(include = np.number)
a = num_cols[num_cols.columns].hist(bins=15, figsize=(15,35), layout=(9,3),color = 'blue',alpha=0.6)

cat_col = df.select_dtypes(exclude=np.number)

cat_col
fig, ax = plt.subplots(4, 2, figsize=(15, 15))

for variable, subplot in zip(cat_col, ax.flatten()):

    sns.countplot(df[variable], ax=subplot,palette = 'plasma')

    for label in subplot.get_xticklabels():

        label.set_rotation(90)

plt.tight_layout()
df[['StandardHours','EmployeeCount']].describe()
df[['StandardHours','EmployeeCount']].corr()
corr = df.drop(columns=['StandardHours','EmployeeCount']).corr()
cols = ['Age', 'BusinessTravel', 'Department',

       'DistanceFromHome', 'Education', 'EducationField', 'EmployeeCount',

        'EnvironmentSatisfaction', 'Gender', 

       'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction',

       'MaritalStatus', 'NumCompaniesWorked',

       'Over18', 'OverTime', 'PercentSalaryHike', 'PerformanceRating',

       'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel',

       'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',

       'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',

       'YearsWithCurrManager']

for col in cols:

    pd.crosstab(df[col],df.Attrition).plot(kind='bar',color = ('green','red'),figsize=(10,5))
#onehot encoding
df.columns.shape
cat_col.columns.shape
num_cols.columns.shape
cat_col_encoded = pd.get_dummies(cat_col)
cat_col_encoded.head()
df = pd.concat([num_cols,cat_col_encoded],sort=False,axis=1)

df.head()
X = df.drop(columns='Attrition')
y = df['Attrition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

logreg = LogisticRegression()

logreg.fit(X_train, y_train)
train_Pred = logreg.predict(X_train)
metrics.confusion_matrix(y_train,train_Pred)
metrics.accuracy_score(y_train,train_Pred)
test_Pred = logreg.predict(X_test)
metrics.accuracy_score(y_test,test_Pred)
from sklearn.metrics import classification_report

print(classification_report(y_test, test_Pred))
###The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives. The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.



#The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. The recall is intuitively the ability of the classifier to find all the positive samples.



#The support is the number of occurrences of each class in y_test.