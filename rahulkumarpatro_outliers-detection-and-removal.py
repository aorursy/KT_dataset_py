# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df=pd.read_csv("/kaggle/input/titanic/train.csv")
df.head()
df.isnull().sum()
df.describe()
df.dtypes
df.info()
df.columns
df.select_dtypes(include='number').nunique()
uniques=df.select_dtypes(exclude='number').nunique()
uniques
df.Fare.describe()
fig = df.Fare.hist(bins=50)
fig.set_title('Fare Distribution')
fig.set_xlabel('Fare')
fig.set_ylabel('Number of Passengers')
## another way of visualising outliers is using boxplots and whiskers,
# which provides the quantiles (box) and inter-quantile range (whiskers),
# with the outliers sitting outside the error bars (whiskers).

# All the dots in the plot below are outliers according to the quantiles + 1.5 IQR rule
sns.boxplot(x=df['Fare'])
## let's look at the values of the quantiles so we can
# calculate the upper and lower boundaries for the outliers

# 25%, 50% and 75% in the output below indicate the
# 25th quantile, median and 75th quantile respectively
df.Fare.describe()
# Let's calculate the upper and lower boundaries
# to identify outliers according
# to interquantile proximity rule
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)
IQR = df.Fare.quantile(0.75) - df.Fare.quantile(0.25)

Lower_fence = df.Fare.quantile(0.25) - (IQR * 1.5)
Upper_fence = df.Fare.quantile(0.75) + (IQR * 1.5)

Upper_fence, Lower_fence, IQR
IQR = df.Fare.quantile(0.75) - df.Fare.quantile(0.25)

Lower_fence = df.Fare.quantile(0.25) - (IQR * 3)
Upper_fence = df.Fare.quantile(0.75) + (IQR * 3)

Upper_fence, Lower_fence, IQR
#  lets look at the actual number of passengers on the upper Fare ranges

print('total passengers: {}'.format(df.shape[0]))

print('passengers that paid more than 65: {}'.format(
    df[df.Fare > 65].shape[0]))

print('passengers that paid more than 100: {}'.format(
    df[df.Fare > 100].shape[0]))
df.shape
df[df.Fare>65].shape
df[df.Fare>100].shape
total= np.float(df.shape[0])

print('Total passengers: {}'.format(total))

print('passengers that paid more than 65: {}'.format(
    df[df.Fare > 65].shape[0]/total))

print('passengers that paid more than 100: {}'.format(
    df[df.Fare > 100].shape[0]/total))
high_fare_df = df[df.Fare>100]
high_fare_df
#Some extreme Outliers
df[df.Fare>300]
data = df.copy()

# replace outliers in Fare
# using the boundary of the interquantile range method
data.loc[data.Fare > 100, 'Fare'] = 100
data[data.Fare>100]
sns.boxplot(x=df['Age'])
df.Age.describe()
Upper_boundary = df.Age.mean() + 3* df.Age.std()
Lower_boundary = df.Age.mean() - 3* df.Age.std()

Upper_boundary, Lower_boundary

IQR = df.Age.quantile(0.75) - df.Age.quantile(0.25)

Lower_fence = df.Age.quantile(0.25) - (IQR * 1.5)
Upper_fence = df.Age.quantile(0.75) + (IQR * 1.5)

Upper_fence, Lower_fence, IQR
IQR = df.Age.quantile(0.75) - df.Age.quantile(0.25)

Lower_fence = df.Age.quantile(0.25) - (IQR * 3)
Upper_fence = df.Age.quantile(0.75) + (IQR * 3)

Upper_fence, Lower_fence, IQR
data1 = data.dropna(subset=['Age'])

total_passengers = np.float(data.shape[0])

print('passengers older than 73 (Gaussian approach): {}'.format(
    data[data.Age > 73].shape[0] / total_passengers))

print('passengers older than 65 (IQR): {}'.format(
    data[data.Age > 65].shape[0] / total_passengers))

print('passengers older than 91 (IQR, extreme): {}'.format(
    data[data.Age >= 91].shape[0] / total_passengers))
data1[data1.Age>65]
data1[data1.Age>73]
data.loc[data.Fare > 100, 'Fare'] = 100
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    df[['Age', 'Fare']].fillna(0),
    df.Survived,
    test_size=0.3,
    random_state=0)

X_train.shape, X_test.shape
X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(
    data[['Age', 'Fare']].fillna(0),
    data.Survived,
    test_size=0.3,
    random_state=0)
logit = LogisticRegression(random_state=44)

# train model
logit.fit(X_train, y_train)

# make predicion on test set
pred = logit.predict_proba(X_test)

print('LogReg Accuracy: {}'.format(logit.score(X_test, y_test)))
print('LogReg roc-auc: {}'.format(roc_auc_score(y_test, pred[:, 1])))
logit = LogisticRegression(random_state=44)

# train model
logit.fit(X_train_clean, y_train_clean)

# make predicion on test set
pred = logit.predict_proba(X_test_clean)

print('LogReg Accuracy: {}'.format(logit.score(X_test, y_test)))

# call model
rf = RandomForestClassifier(n_estimators=200, random_state=39)

# train model
rf.fit(X_train, y_train)

# make predictions
pred = rf.predict_proba(X_test)

print('Random Forests Accuracy: {}'.format(rf.score(X_test, y_test)))
print('Random Forests roc-auc: {}'.format(roc_auc_score(y_test, pred[:, 1])))
# call model
rf = RandomForestClassifier(n_estimators=200, random_state=39)

# train model
rf.fit(X_train_clean, y_train_clean)

# make predictions
pred = rf.predict_proba(X_test_clean)

print('Random Forests Accuracy: {}'.format(rf.score(X_test_clean, y_test_clean)))
print('Random Forests roc-auc: {}'.format(roc_auc_score(y_test_clean, pred[:,1])))