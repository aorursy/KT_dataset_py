import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
train_data = pd.read_csv("/kaggle/input/sf-crime/train.csv")

test_data = pd.read_csv("/kaggle/input/sf-crime/test.csv")
train_data.head()
print (train_data.isnull().sum())

print (test_data.isnull().sum())
train_data.info()
train_data = train_data.drop(["Descript", "Resolution"], axis = 1)
def transformDataset(dataset):

    dataset['Dates'] = pd.to_datetime(dataset['Dates'])

    

    dataset['Date'] = dataset['Dates'].dt.date

    

    dataset['n_days'] = (dataset['Date'] - dataset['Date'].min()).apply(lambda x: x.days)

    

    dataset['Year'] = dataset['Dates'].dt.year

    dataset['DayOfWeek'] = dataset['Dates'].dt.dayofweek # OVERWRITE

    dataset['WeekOfYear'] = dataset['Dates'].dt.weekofyear

    dataset['Month'] = dataset['Dates'].dt.month

    

    dataset['Hour'] = dataset['Dates'].dt.hour

    

    dataset['Block'] = dataset['Address'].str.contains('block', case=False)

    dataset['Block'] = dataset['Block'].map(lambda x: 1 if  x == True else 0)



    

    dataset = dataset.drop('Dates', 1)

    dataset = dataset.drop('Date', 1)

    dataset = dataset.drop('Address', 1)

    

    dataset = pd.get_dummies(data=dataset, columns=[ 'PdDistrict'], drop_first = True)

    return dataset

    
train_data = transformDataset(train_data)
test_data  = transformDataset(test_data)
train_data.head()
sns.pairplot(train_data[["X", "Y"]])
sns.boxplot(train_data[["Y"]])
train_data = train_data[train_data["Y"] < 80]

sns.distplot(train_data[["X"]])
fig, ax = plt.subplots(figsize=(9.2, 10))

plt.barh(train_data["Category"].unique(),train_data["Category"].value_counts())
from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()

train_data["Category"] = le.fit_transform(train_data["Category"])

X = train_data.drop("Category",axis=1).values

y = train_data["Category"].values
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()

dtree.fit(X_train,y_train)
predictions = dtree.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix



cm = confusion_matrix(y_test,predictions)

fig, ax = plt.subplots(figsize=(20, 20))

sns.heatmap(cm, annot=False, ax = ax); #annot=True to annotate cells

ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 

ax.set_title('Confusion Matrix'); 
print (classification_report(y_test,predictions))
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=40,min_samples_split=100 )

rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)

print (classification_report(y_test,rfc_pred))
n_features = X.shape[1]

plt.barh(range(n_features),rfc.feature_importances_)

plt.yticks(np.arange(n_features),train_data.columns[1:])
keys = le.classes_

values = le.transform(le.classes_)

keys
dictionary = dict(zip(keys, values))

print(dictionary)
test_data.head()
test_data = test_data.drop('Id', 1)
y_pred_proba = rfc.predict_proba(test_data)

y_pred_proba
result = pd.DataFrame(y_pred_proba, columns=keys)

result.head()
result.to_csv(path_or_buf="rfc_predict_4.csv",index=True, index_label = 'Id')