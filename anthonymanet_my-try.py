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
print(os.listdir("../input/home-credit-default-risk/"))
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set_style('darkgrid')
application_train_data = pd.read_csv("/kaggle/input/home-credit-default-risk/application_train.csv")

print('Application Train Data Shape: ', application_train_data.shape)

application_train_data.head()
(application_train_data['DAYS_BIRTH']/-365).describe()
# Target Distribution

application_train_data['TARGET'].value_counts()



# Result: this is an imbalanced class problem
# Testing data features

application_test_data = pd.read_csv('../input/home-credit-default-risk/application_test.csv')

print('Application Testing Data Shape: ', application_test_data.shape)

application_test_data.head()
def missing_data(data):

    mis_data = data.isnull().sum().sort_values(ascending = False)

    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)

    return pd.concat([mis_data, percent], axis=1, keys=['Total', 'Percent'])
application_train_data.info()
(application_train_data.isna().sum() == 0).value_counts()
missing_data(application_test_data).head(10)
(application_test_data.isna().sum() == 0).value_counts()
ColSelect = missing_data(application_train_data)[missing_data(application_train_data).Percent < 60]
list1 = list(ColSelect.index)

list1.remove("TARGET")
app_train = application_train_data[list(ColSelect.index)]

app_test = application_test_data[list1]
app_train.dropna(inplace=True)
app_train.info()
app_test.select_dtypes('object')
(app_test.isna().sum()).head(10)
app_train.info()

# dtypes: float64(65), int64(41), object(16)
app_train.select_dtypes('object').head()
# for example:

app_train['CODE_GENDER'].value_counts()
cat_feats = list(app_train.select_dtypes('object').columns)
app_train_final = pd.get_dummies(app_train,columns=cat_feats,drop_first=True)

app_train_final
sns.countplot(x='NAME_CONTRACT_TYPE', hue='TARGET', data=app_train, palette='Set1')



# Target 1: client with payment difficulties

# Result: Cash loans is the most popular contract type and no difficulties for Revolving contract type
plt.figure(figsize=(10,6))

app_train['age'] = app_train['DAYS_BIRTH']/-365

app_train[app_train['TARGET']==1]['age'].hist(bins=35,color='blue',

                                                                        label='Target = 1',alpha=0.7)

app_train[app_train['TARGET']==0]['age'].hist(bins=35,color='red',

                                                                        label='Target = 0',alpha=0.4)

plt.legend()

plt.xlabel('Age')



# Target 1: client with payment difficulties
app_train_final.drop('SK_ID_CURR',axis=1,inplace=True)
app_train_corr = app_train_final.corr()['TARGET'].sort_values()
app_train_corr.head(10)

# Display Neg
app_train_corr.tail(10)

# Display Pos
# Make a new dataframe for features

train_feats = app_train_final[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_EMPLOYED', 'TARGET']]

test_feats = app_test[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_EMPLOYED']]
# THIS IS GOING TO BE A VERY LARGE PLOT

sns.pairplot(train_feats,hue='TARGET',palette='coolwarm')
# imputer for handling missing values

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import MinMaxScaler



train = train_feats.drop(columns = ['TARGET'])

test = test_feats



# Feature names

features = list(train.columns)





# Median imputation of missing values

imputer = SimpleImputer(strategy = 'median')



# Fit on the training data

imputer.fit(train)



imputer.fit(test)
# Transform both training and testing data

train = imputer.transform(train)

test = imputer.transform(test)
# Scale each feature to 0-1

scaler = MinMaxScaler(feature_range = (0, 1))



# Repeat with the scaler

scaler.fit(train)

train = scaler.transform(train)

test = scaler.transform(test)
print('Training data shape: ', train.shape)

print('Testing data shape: ', test.shape)
traindf = pd.DataFrame(train, columns = (['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']))

testdf = pd.DataFrame(test, columns = (['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']))
train_labels = train_feats['TARGET']
from sklearn.linear_model import LogisticRegression



# Make the model

# logmodel = LogisticRegression()



# Train on the training data

# logmodel.fit(traindf, train_labels)

# predict_lr = logmodel.predict_proba(testdf)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(traindf,train_feats['TARGET'],

                                                    test_size=0.30)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
error_rate = []



# Will take some time

for i in range(1,40):

    

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    pred_i = knn.predict(X_test)

    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10,6))

plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='red', markersize=10)

plt.title('Error Rate vs. K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')
# NOW WITH K=30

knn = KNeighborsClassifier(n_neighbors=6)



knn.fit(X_train,y_train)

pred = knn.predict(X_test)



print('WITH K=10')

print('\n')

print(confusion_matrix(y_test,pred))

print('\n')

print(classification_report(y_test,pred))
pred_testset = knn.predict(testdf)
pred_testset
# Submission dataframe

results = app_test[['SK_ID_CURR']]

results['TARGET'] = pred_testset

results['TARGET'].value_counts()
results.to_csv('my_submission4.csv', index = False)