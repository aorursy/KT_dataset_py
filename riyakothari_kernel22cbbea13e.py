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
import warnings

import seaborn as sns

warnings.filterwarnings("ignore")

sns.set(style = 'darkgrid')

sns.set_palette('PuBuGn_d')
train_data = pd.read_csv("../input/sf-crime/train.csv.zip")

test_data = pd.read_csv("../input/sf-crime/test.csv.zip")
train_data.head()
print(train_data.isnull().sum())

print(test_data.isnull().sum())

print(train_data.info())
train_data = train_data.drop(['Descript', 'Resolution'], axis=1)

test_data.head()

train_data.head()
def dataTransform(dataset):

    dataset['Dates'] = pd.to_datetime(dataset['Dates'])

    dataset['Date'] = dataset['Dates'].dt.date

    dataset['n_days'] = (dataset['Dates']- dataset['Dates'].min()).apply(lambda x: x.days)

    

    dataset['Year'] = dataset['Dates'].dt.year

    dataset['DayOfWeek'] = dataset['Dates'].dt.dayofweek # OVERWRITE

    dataset['WeekOfYear'] = dataset['Dates'].dt.weekofyear

    dataset['Month'] = dataset['Dates'].dt.month

#     dataset['Year'] = dataset['Dates'].dt.year

#     dataset['Month'] = dataset['Dates'].dt.month

#     dataset['Week'] = dataset['Dates'].dt.weekofyear

#     dataset['DayofWeek'] = dataset['Dates'].dt.dayofweek #OVERWRITE

    

    dataset['Hour']  =dataset['Dates'].dt.hour

    dataset['Block'] = dataset['Address'].str.contains('block', case=False)

    dataset['Block'] = dataset['Block'].map(lambda x: 1 if x==True else 0)

    dataset = dataset.drop(['Dates', 'Date', 'Address'], axis=1)

    dataset = pd.get_dummies(data=dataset, columns=[ 'PdDistrict'], drop_first = True)

    return dataset
crime_data = dataTransform(train_data)

test_data = dataTransform(test_data)

#print(crime_data.head())
crime_data.head()
sns.pairplot(crime_data[['X', 'Y']])
sns.boxplot(crime_data[['Y']])
crime_data = crime_data[crime_data['Y']<80]

sns.distplot(crime_data[['X']])
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize = (9.2, 10))

plt.barh(crime_data['Category'].unique(), crime_data['Category'].value_counts())
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

crime_data['Category'] = le.fit_transform(crime_data['Category'])

crime_data.head()
from sklearn.model_selection import train_test_split

y = crime_data['Category'].values

X = crime_data.drop(['Category'], axis=1).values

train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=1, test_size=0.10)

print(train_X, train_y)
from sklearn.tree import DecisionTreeClassifier

crime_model = DecisionTreeClassifier()

crime_model.fit(train_X, train_y)
prediction = crime_model.predict(test_X)
from sklearn.ensemble import RandomForestClassifier

crime_model2 = RandomForestClassifier(random_state=1)

crime_model2.fit(train_X, train_y)

predictions = crime_model2.predict(test_X)

print(predictions)

print("Hallelujah")

print(prediction)

from sklearn.metrics import classification_report, confusion_matrix

cm = confusion_matrix(test_y, prediction)

fig, ax = plt.subplots(figsize=(20, 20))

sns.heatmap(cm, annot=False, ax = ax); #annot=True to annotate cells

ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 

ax.set_title('Confusion Matrix'); 
print (classification_report(test_y,prediction))
print (classification_report(test_y,predictions))
keys = le.classes_

values = le.transform(le.classes_)

keys
dictionary = dict(zip(keys, values))

print(dictionary)
test_data.head()

#test_data = test_data.drop('Id', 1)
y_pred_proba = crime_model2.predict_proba(test_data)

y_pred_proba
result = pd.DataFrame(y_pred_proba, columns=keys)

result.head()
result.to_csv(path_or_buf="randomforestclassifier_predict.csv",index=True, index_label = 'Id')