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
df_aus = pd.read_csv(os.path.join(dirname, filename))
df_aus.head()
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set()
def basic_info(data):

    print("Dataset shape is: ", data.shape)

    print("Dataset size is: ", data.size)

    print("Dataset columns are: ", data.columns)

    print("Dataset info is: ", data.info())

    categorical = []

    numerical = []

    for i in data.columns:

        if data[i].dtype == object:

            categorical.append(i)

        else:

            numerical.append(i)

    print("Categorical variables are:\n ", categorical)

    print("Numerical variables are:\n ", numerical)

    return categorical, numerical
categorical, numerical = basic_info(df_aus)
df_aus2 = df_aus.copy(deep = True)
df_aus2 = df_aus.drop(['RISK_MM'], axis = 1)
df_aus2['Date'] = pd.to_datetime(df_aus2['Date'])
df_aus2.head()
categorical2, numerical2 = basic_info(df_aus2)
df_aus2.isnull().sum()
plt.figure(figsize = (30,8))

plt.plot(df_aus2['Date'][:1000], df_aus2['MinTemp'][:1000], color = "#DC143C", label = 'Minimum Temperature',)

plt.plot(df_aus2['Date'][:1000], df_aus2['MaxTemp'][:1000], color = "#104E8B", label = 'Maximum Temperature')

plt.fill_between(df_aus2['Date'][:1000], df_aus2['MinTemp'][:1000], df_aus2['MaxTemp'][:1000], facecolor = "#EEE685")

plt.legend()

plt.show()
plt.figure(figsize = (30,8))

plt.plot(df_aus2['Date'][1000:2000], df_aus2['MinTemp'][1000:2000], color = "#DC143C", label = 'Minimum Temperature',)

plt.plot(df_aus2['Date'][1000:2000], df_aus2['MaxTemp'][1000:2000], color = "#104E8B", label = 'Maximum Temperature')

plt.fill_between(df_aus2['Date'][1000:2000], df_aus2['MinTemp'][1000:2000], df_aus2['MaxTemp'][1000:2000], facecolor = "#EEE685")

plt.legend()

plt.show()
categorical2
df_aus2['Location'].value_counts()
plt.figure(figsize=(50, 8))

sns.countplot(df_aus2['Location'])

plt.xticks(rotation=-45)

plt.show()
df_aus2['WindGustDir'].value_counts()
plt.figure(figsize=(30, 8))

sns.countplot(df_aus2['WindGustDir'])

plt.xticks(rotation=-45)

plt.show()
df_aus2['WindDir9am'].value_counts()
plt.figure(figsize=(30, 8))

sns.countplot(df_aus2['WindDir9am'])

plt.xticks(rotation=-45)

plt.show()
df_aus2['WindDir3pm'].value_counts()
plt.figure(figsize=(30, 8))

sns.countplot(df_aus2['WindDir3pm'])

plt.xticks(rotation=-45)

plt.show()
numerical2
numerical3 = numerical2[:]
numerical3 = numerical3[1:]  #removing date
numerical3
numerical_hist = df_aus2[numerical3]
numerical_hist
numerical_hist.hist(figsize = [20,20], bins = 50)

plt.show()
min(df_aus2['MinTemp'].value_counts().index)
max(df_aus2['MaxTemp'].value_counts().index)
max(df_aus2['Rainfall'].value_counts().index)
def making_new_df(data, columnlist):

    for i in columnlist:

        dummy = pd.get_dummies(data[i])

        #print(dummy)

        del dummy[dummy.columns[-1]]

        data = pd.concat([data, dummy], axis = 1)

    return data
df_aus3 = making_new_df(df_aus2, categorical2)
df_aus3
df_aus3 = df_aus3.drop(['Date']+categorical, axis = 1)
df_aus3.head()
df_aus3.isnull().sum()
from sklearn.model_selection import train_test_split
X = df_aus3.iloc[:, :-1]
X
Y = df_aus3.iloc[:, -1]
Y
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size = 0.20, random_state=42)
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(train_x)

imputer.fit(test_x)
train_x = imputer.transform(train_x)

test_x = imputer.transform(test_x)
from sklearn.linear_model import LogisticRegression

LR = LogisticRegression(max_iter=5000)
model_lr = LR.fit(train_x, train_y)
y_lr_predict = model_lr.predict(test_x)
LR_df = pd.DataFrame(data = {"Actual": test_y, "Predicted": y_lr_predict})
LR_df
model_lr.score(test_x, test_y)
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
model_rfr = rfc.fit(train_x, train_y)

y_rfr_predict = model_rfr.predict(test_x)
RFR_df = pd.DataFrame(data = {"Actual": test_y, "Predicted": y_rfr_predict})
RFR_df
model_rfr.score(test_x, test_y)
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
model_gnb = gnb.fit(train_x, train_y)

y_gnb_predict = model_gnb.predict(test_x)
GNB_df = pd.DataFrame(data = {"Actual":test_y, "Predicted": y_gnb_predict})
GNB_df
model_gnb.score(test_x, test_y)
print("Logistic Regression Score: ", model_lr.score(test_x, test_y))

print("Random Forest Classifier Score: ", model_rfr.score(test_x, test_y))

print("Naive Bayes Score: ", model_gnb.score(test_x, test_y))