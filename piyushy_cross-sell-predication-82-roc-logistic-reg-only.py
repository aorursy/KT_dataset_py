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
from matplotlib import pyplot as plt

import seaborn as sns
train_data = pd.read_csv("/kaggle/input/health-insurance-cross-sell-prediction/train.csv")

test_data = pd.read_csv("/kaggle/input/health-insurance-cross-sell-prediction/test.csv")
train_data.head()
test_data.head()
print(f"Training set has {train_data.shape[0]} examples.")

print(f"Test set has {test_data.shape[0]} examples.")
plt.style.use('seaborn')

plt.figure(figsize=(10,5))

sns.heatmap(train_data.isnull(), yticklabels = False, cmap='plasma')

plt.title('Null Values in Training Set');
train_data.isnull().sum(axis=0)
test_data.isnull().sum(axis=0)
plt.figure(figsize=(15,5))

plt.subplot(1,2,1)

sns.countplot(train_data.Response)

plt.title('Number of customers Insured');
plt.figure(figsize=(15,15))

plt.subplot(3,3,1)

sns.countplot(train_data.Gender)

plt.subplot(3,3,2)

sns.countplot(train_data.Driving_License)

plt.subplot(3,3,3)

sns.countplot(train_data.Region_Code)

plt.subplot(3,3,4)

sns.countplot(train_data.Previously_Insured)

plt.subplot(3,3,5)

sns.countplot(train_data.Vehicle_Age)

plt.subplot(3,3,6)

sns.countplot(train_data.Vehicle_Damage)

plt.subplot(3,3,7)

sns.countplot(train_data.Policy_Sales_Channel)
plt.figure(figsize=(15,15))

plt.subplot(3,3,1)

sns.countplot(x="Gender", hue="Response", data=train_data)

plt.subplot(3,3,2)

sns.countplot(x="Driving_License", hue="Response", data=train_data)

plt.subplot(3,3,3)

sns.countplot(x="Region_Code", hue="Response", data=train_data)

plt.subplot(3,3,4)

sns.countplot(x="Previously_Insured", hue="Response", data=train_data)

plt.subplot(3,3,5)

sns.countplot(x="Vehicle_Age", hue="Response", data=train_data)

plt.subplot(3,3,6)

sns.countplot(x="Vehicle_Damage", hue="Response", data=train_data)

plt.subplot(3,3,7)

sns.countplot(x="Policy_Sales_Channel", hue="Response", data=train_data)
plt.figure(figsize=(24,5))

plt.subplot(1,4,1)

train_data.Age.plot(kind='hist')

plt.title("Age Distribution")

plt.subplot(1,4,2)

train_data.Vintage.plot(kind='hist')

plt.title("Vintage Distribution")

plt.subplot(1,4,3)

train_data.Region_Code.plot(kind='hist')

plt.title("Region Code Distribution")

plt.subplot(1,4,4)

train_data.Policy_Sales_Channel.plot(kind='hist')

plt.title("Policy Sales Channel Distribution")
plt.figure(figsize=(24,5))

plt.subplot(1,4,1)

train_data.Age.hist(bins=80)

plt.title("Age Distribution")

plt.subplot(1,4,2)

train_data.Vintage.hist(bins=30)

plt.title("Vintage Distribution")

plt.subplot(1,4,3)

train_data.Region_Code.hist(bins=50)

plt.title("Region Code Distribution")

plt.subplot(1,4,4)

train_data.Policy_Sales_Channel.hist(bins=80)

plt.title("Policy Sales Channel Distribution")
plt.figure(figsize=(24,5))

plt.subplot(1,4,1)

train_data.groupby('Response').Age.hist(bins=80)

plt.title("Age Distribution")

plt.subplot(1,4,2)

train_data.groupby('Response').Vintage.hist(bins=30)

plt.title("Vintage Distribution")

plt.subplot(1,4,3)

train_data.groupby('Response').Region_Code.hist(bins=50)

plt.title("Region Code Distribution")

plt.subplot(1,4,4)

train_data.groupby('Response').Policy_Sales_Channel.hist(bins=80)

plt.title("Policy Sales Channel Distribution")
train_data.Region_Code.value_counts()
train_data.Policy_Sales_Channel.value_counts()
plt.figure(figsize=(24,15))

plt.subplot(2,2,1)

sns.boxplot(y = 'Response', x = 'Age', data = train_data, fliersize = 0, orient = 'h')

plt.subplot(2,2,2)

sns.boxplot(y = 'Response', x = 'Vintage', data = train_data, fliersize = 0, orient = 'h')

plt.subplot(2,2,3)

sns.boxplot(y = 'Response', x = 'Region_Code', data = train_data, fliersize = 0, orient = 'h')

plt.subplot(2,2,4)

sns.boxplot(y = 'Response', x = 'Policy_Sales_Channel', data = train_data, fliersize = 0, orient = 'h')
plt.figure(figsize=(24,15))

plt.subplot(3,4,1)

sns.stripplot(x='Response', y='Age', data=train_data, alpha=0.01, jitter=True);

plt.title("Age Distribution")

plt.subplot(3,4,2)

sns.stripplot(x='Response', y='Vintage', data=train_data, alpha=0.01, jitter=True);

plt.title("Vintage Distribution")

plt.subplot(3,4,3)

sns.stripplot(x='Response', y='Region_Code', data=train_data, alpha=0.01, jitter=True);

plt.title("Region Code Distribution")

plt.subplot(3,4,4)

sns.stripplot(x='Response', y='Policy_Sales_Channel', data=train_data, alpha=0.01, jitter=True);

plt.title("Policy Sales Channel Distribution")
train_data.Annual_Premium.plot(kind='hist')
train_data.Annual_Premium.hist(bins=100)
train_data.groupby('Response').Annual_Premium.hist(bins=100)
sns.boxplot(y = 'Response', x = 'Annual_Premium', data = train_data, fliersize = 0, orient = 'h')

sns.stripplot(y = 'Response', x = 'Annual_Premium', data = train_data,linewidth = 0.6, orient = 'h')
from sklearn.preprocessing import LabelEncoder



corr_check = train_data.copy()



col_ls = ['Gender', 'Vehicle_Age', 'Vehicle_Damage']



for col in col_ls:

    corr_check[col] = LabelEncoder().fit_transform(corr_check[col])
sns.heatmap(corr_check.corr(), annot=True)

plt.title('Corelation Matrix');
train_data['Gender'][train_data['Gender'] == 'Male'] = 0

train_data['Gender'][train_data['Gender'] == 'Female'] = 1



train_data['Vehicle_Age'][train_data['Vehicle_Age'] == '< 1 Year'] = 0

train_data['Vehicle_Age'][train_data['Vehicle_Age'] == '1-2 Year'] = 1

train_data['Vehicle_Age'][train_data['Vehicle_Age'] == '> 2 Years'] = 2



train_data['Vehicle_Damage'][train_data['Vehicle_Damage'] == 'No'] = 0

train_data['Vehicle_Damage'][train_data['Vehicle_Damage'] == 'Yes'] = 1
train_data.head()
test_data['Gender'][test_data['Gender'] == 'Male'] = 0

test_data['Gender'][test_data['Gender'] == 'Female'] = 1



test_data['Vehicle_Age'][test_data['Vehicle_Age'] == '< 1 Year'] = 0

test_data['Vehicle_Age'][test_data['Vehicle_Age'] == '1-2 Year'] = 1

test_data['Vehicle_Age'][test_data['Vehicle_Age'] == '> 2 Years'] = 2



test_data['Vehicle_Damage'][test_data['Vehicle_Damage'] == 'No'] = 0

test_data['Vehicle_Damage'][test_data['Vehicle_Damage'] == 'Yes'] = 1
train_data.Annual_Premium[train_data.Annual_Premium > 200000] = train_data.Annual_Premium.mean()

test_data.Annual_Premium[test_data.Annual_Premium > 200000] = train_data.Annual_Premium.mean()
train_data.columns
train_data = train_data.drop(['Driving_License'], axis = 1)

test_data = test_data.drop(['Driving_License'], axis = 1)

# train_data = train_data.drop(['id'], axis = 1)

# test_data = test_data.drop(['id'], axis = 1)

train_data.head()
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(train_data.drop(['id','Response'], axis=1), train_data['Response'], test_size = 0.3, random_state=0)
from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression()

logreg.fit(X_train, y_train)

test_pred = logreg.predict(X_test)

test_pred_proba = logreg.predict_proba(X_test)
from sklearn.metrics import roc_auc_score, accuracy_score

print(accuracy_score(y_test, test_pred))

print(roc_auc_score(y_test, test_pred_proba[:,1]))
predictions = logreg.predict(test_data.drop(['id'], axis=1))

print(test_data.shape, predictions.shape)

output = pd.DataFrame({'i': test_data.PassengerId, 'Survived': predictions})

output.to_csv('submission.csv', index=False)

print("Your submission was successfully saved!")