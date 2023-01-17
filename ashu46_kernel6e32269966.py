# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import csv
import seaborn as sns
import random
df = pd.read_excel("../input/Data_cardio.xlsx" )
df.head()
df.dtypes
def gen_to_cat(x):
    if x== 'M':
        return 1
    if x== 'F':
        return 0
def mob_to_cat(x):
    if x== 'Car':
        return 1
    if x== 'No car':
        return 0
df['Gender'] = df['Gender'].apply(gen_to_cat)
df['Mobility'] = df['Mobility'].apply(mob_to_cat)
df['Age'] = df.Age.astype(str)
df['Distance'] = df.Distance.astype(str)
df['Age'] = df.Age.str.replace(',', '.')
df['Distance'] = df.Distance.str.replace(',', '.')
df['Age'] = df.Age.astype(float)
df['Distance'] = df.Distance.astype(float)
df.head()
df.dtypes
df.describe()
df.Age.hist()
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
df.Distance.hist()
plt.title('Histogram of Distance')
plt.xlabel('Distance')
plt.ylabel('Frequency')
sns.countplot(x = 'Gender', data = df, palette = 'hls')
sns.countplot(x = 'Mobility', data = df, palette = 'hls')
sns.countplot(x = 'Participation', data = df, palette = 'hls')
df.groupby('Participation').mean()
%matplotlib inline
pd.crosstab(df.Mobility,df.Participation).plot(kind='bar')
plt.title('Acceptance Frequency for Mobility')
plt.xlabel('Mobility')
plt.ylabel('Frequency of Mobility')
%matplotlib inline
pd.crosstab(df.Gender,df.Participation).plot(kind='bar')
plt.title('Acceptance Frequency for Gender')
plt.xlabel('Gender')
plt.ylabel('Frequency of Gender')
# plt.savefig('purchase_fre_job')
df.isnull().sum()
corr_pearson = df.corr('pearson')
corr_pearson
corr_spearman = df.corr('spearman')
corr_spearman
sns.countplot(x = 'Participation', data = df, palette = 'hls')
X = df.iloc[:,2:5] # without variable 'Gender'
X.head()
y = df['Participation']
y.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25, random_state=0)
X_train.shape, y_train.shape
X_test.shape, y_test.shape
XG = df.iloc[:,1:5] # with variable 'Gender'
XG.head()
XG_train, XG_test, yG_train, yG_test = train_test_split(XG, y,test_size=0.25, random_state=0)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
plt.figure(figsize = (4,4))
sns.heatmap(confusion_matrix, annot= True, fmt = ".3f", linewidths= .5, square=True, cmap= 'Blues_r')
plt.ylabel('Actual Lable')
plt.xlabel('Predicted Lable')
print('Accuracy of logistic regression classifier on test set with LIBLINEAR solver: {:.2f}'.format(classifier.score(X_test, y_test)))
classifier = LogisticRegression(random_state=0)
classifier.fit(XG_train, yG_train)
yG_pred = classifier.predict(XG_test)
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(yG_test, yG_pred)
print(confusion_matrix)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(classifier.score(XG_test, yG_test)))
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
