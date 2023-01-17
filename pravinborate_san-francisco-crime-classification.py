# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import zipfile



sns.set()

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import datetime
z = zipfile.ZipFile('/kaggle/input/sf-crime/test.csv.zip')

print(z.namelist())
test = pd.read_csv(z.open('test.csv'))
z = zipfile.ZipFile('/kaggle/input/sf-crime/sampleSubmission.csv.zip')

print(z.namelist())
sampleSubmission = pd.read_csv(z.open('sampleSubmission.csv'))
z = zipfile.ZipFile('/kaggle/input/sf-crime/train.csv.zip')

print(z.namelist())
train = pd.read_csv(z.open('train.csv'))
test.head()
train.head()
y = train['Category']
train['year'] = train['Dates'].apply(lambda x : x.split()[0].split('-')[0])
train['Week'] = train['Dates'].apply(lambda x : x.split()[0].split('-')[1])
train['Hours'] = train['Dates'].apply(lambda x : x.split()[1].split(':')[0])
train.isnull().sum()
sns.heatmap(train.isnull())
train.PdDistrict.value_counts().plot(kind='bar', figsize=(8,10))

plt.show()
train['DayOfWeek'].value_counts().plot(kind='bar', figsize=(8,10))

plt.show()
train['Category'].value_counts().plot(kind='bar', figsize=(8,10))

plt.xlabel('Category of crime')

plt.ylabel('Count')

plt.show()
target = train['Category'].unique()

print(target)
data_dict = {}

count = 1

for data in target:

    data_dict[data] = count

    count+=1

train["Category"] = train["Category"].replace(data_dict)
dayofweeks = train['DayOfWeek'].unique()

print(dayofweeks)
data_week_dict = {

    "Monday":1,

    "Tuesday":2,

    "Wednesday":3,

    "Thursday":4,

    "Friday":5,

    "Saturday":6,

    "Sunday":7

}
test.head()
train['DayOfWeek'] = train['DayOfWeek'].replace(data_week_dict)

test['DayOfWeek'] = test['DayOfWeek'].replace(data_week_dict)
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()
train['PdDistrict'] = labelencoder.fit_transform(train['PdDistrict'])

test['PdDistrict'] = labelencoder.fit_transform(test['PdDistrict'])
train_data_columns = train.columns

print(train_data_columns)

test_data_columns = test.columns

print(test_data_columns)
corr = train.corr()

print(corr['Category'])
corrMat = train[['Category','DayOfWeek','PdDistrict','X','Y']].corr()
mask = np.array(corrMat)

# print(mask)

mask[np.triu_indices_from(mask)] = False

# print(mask)
fig, ax = plt.subplots(figsize=(11, 9))

fig.set_size_inches(20,10)

sns.heatmap(corrMat,mask=mask,vmax=.3,square=True,annot=True)

plt.show()
skew = train.skew()

print(skew)
feautes = ["DayOfWeek","PdDistrict","X","Y"]
X_train = train[feautes]

y_train = train['Category']

X_test = test[feautes]
def rmsle(y,y_pred,convertExp = True):

  if convertExp:

    y = np.exp(y)

    y_pred = np.exp(y_pred)

  log1 = np.nan_to_num(np.array([np.log(v+1) for v in y]))

  log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_pred]))

  calc = (log1 - log2)**2

  # print(calc)

  return np.sqrt(np.mean(calc))
from sklearn.linear_model import LinearRegression,Ridge,Lasso

from sklearn.model_selection import GridSearchCV

from sklearn import metrics

import warnings

pd.options.mode.chained_assignment = None

warnings.filterwarnings("ignore", category=DeprecationWarning)



# Initialize logistic regression model

lModel = LinearRegression()



# Train the model

yLabelsLog = np.log1p(y_train)

lModel.fit(X = X_train,y = yLabelsLog)



# Make predictions

preds = lModel.predict(X= X_train)

print ("RMSLE Value For Linear Regression: ",rmsle(np.exp(yLabelsLog),np.exp(preds),False))
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

predictions = knn.predict(X_test)
from collections import OrderedDict

data_dict_new = OrderedDict(sorted(data_dict.items()))

print(data_dict_new)
result_dataframe = pd.DataFrame({

    "Id": test["Id"]

})

for key,value in data_dict_new.items():

    result_dataframe[key] = 0

count = 0

for item in predictions:

    for key,value in data_dict.items():

        if(value == item):

            result_dataframe[key][count] = 1

    count+=1

result_dataframe.to_csv("submission_knn.csv", index=False)