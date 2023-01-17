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



import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

sns.set_palette('pastel')
data = pd.read_csv('../input/heart-disease-uci/heart.csv')
data.describe()
data.isnull().sum()
data[data.chol>380]
data.drop(data[data.chol>380].index, axis=0, inplace=True)
#checking the balance of targets:



data.target.value_counts()
data.describe()
data.hist(bins=30, figsize=(30,30))
plt.figure(figsize=(10,5))

sns.distplot(data.age[data.target==0])

sns.distplot(data.age[data.target==1])

plt.legend(labels=['Heart Disease','No Heart Disease'])

plt.title('Distribution of Age')

plt.xlabel('Age')

plt.show()
plt.figure(figsize=(8,5))

sns.countplot(data.sex, hue=data.target)

plt.legend(labels=['Heart Disease','No Heart Disease'])

plt.title('Count of Observations by Sex')

plt.xlabel('Sex')

plt.xticks(np.arange(2),('Female','Male'))
sex_total = data.sex.value_counts()

sex_heartdisease = data.sex[data.target==0].value_counts()



percentage_hd_sex = []



for i in range(2):

    perc_hd = sex_heartdisease[i]/sex_total[i]

    percentage_hd_sex.append(perc_hd)
plt.figure(figsize=(6,5))

sns.barplot(y=pd.Series(percentage_hd_sex),x=['Female','Male'])

plt.title('Percentage of Patients with Heart Disease, by Sex')

plt.ylabel('Percentage (%)')

plt.xlabel('Sex')
data.cp.value_counts()
plt.figure(figsize=(10,8))

sns.countplot(data.cp, hue=data.target)

plt.legend(labels=['Heart Disease','No Heart Disease'])

plt.xlabel('Chest Pain')

plt.title('Chest Pain experienced by Patients')
plt.figure(figsize=(18,6))

sns.distplot(data.trestbps[data.target==0])

sns.distplot(data.trestbps[data.target==1])

plt.legend(labels=['Heart Disease','No Heart Disease'])

plt.title('Distribution of Blood Pressure')

plt.xticks(np.arange(60, 221, step=5))

plt.xlabel('Blood Pressure')
print(data.trestbps[data.target==0].mean(), data.trestbps[data.target==1].mean())
plt.figure(figsize=(4,8))

sns.boxplot(x=data.target, y=data.trestbps)

plt.title('Boxplot displaying Blood Pressure by target')

plt.yticks(np.arange(90, 210, step=5))

plt.ylabel('Blood Pressure')

plt.xlabel('Heart Disease (Y/N)')
plt.figure(figsize=(18,6))

sns.distplot(data.chol[data.target==0])

sns.distplot(data.chol[data.target==1])

plt.legend(labels=['Heart Disease','No Heart Disease'])

plt.title('Distribution of Cholestrol')

plt.xticks(np.arange(50,650, step=50))

plt.xlabel('Cholestrol')
print('HD Cholestrol Average: ',data.chol[data.target==0].mean(), ', No HD Cholestrol Average: ',data.chol[data.target==1].mean())
data[data.chol>500]
#dropping the 564 chol case:

x = data[data.chol>500].index

data.drop(x, axis=0, inplace=True)

data.index = range(len(data))
plt.figure(figsize=(18,6))

sns.distplot(data.chol[data.target==0])

sns.distplot(data.chol[data.target==1])

plt.legend(labels=['Heart Disease','No Heart Disease'])

plt.title('Distribution of Cholestrol')

plt.xticks(np.arange(50,650, step=50))

plt.xlabel('Cholestrol')
print('HD Cholestrol Average: ',data.chol[data.target==0].mean(), ', No HD Cholestrol Average: ',data.chol[data.target==1].mean())
data.head()
fig = plt.figure(figsize=(10,5))



fig.add_subplot(131)

plt.title('Fasting Blood Sugar')

sns.countplot(data.fbs, hue=data.target)



fig.add_subplot(132)

plt.title('RECG results')

sns.countplot(data.restecg, hue=data.target)



fig.add_subplot(133)

plt.title('Ex. induced Angina')

sns.countplot(data.exang, hue=data.target)
plt.figure(figsize=(12,6))

sns.distplot(data.thalach[data.target==0])

sns.distplot(data.thalach[data.target==1])

plt.legend(labels=['Heart Disease','No Heart Disease'])

plt.title('Distribution of Max Heart Rate Achieved')

plt.xlabel('Max Heart Rate')
X = data.drop('target',axis=1)

y = data.target



from sklearn.feature_selection import chi2

F, p_values = chi2(X,y)
p_values = pd.Series(p_values, index=X.columns)

p_values.sort_values(ascending=False, inplace=True)
features = p_values[p_values < 0.05].index
features
X = data[features]

y = data.target
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model  import LogisticRegression

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import classification_report, confusion_matrix



RF = RandomForestClassifier()

lr = LogisticRegression()

NB = MultinomialNB()

models = [RF,lr,NB]
def model_test(model, X_train, X_test, y_train, y_test):

    print(model)

    model.fit(X_train, y_train)

    y_predict = model.predict(X_test)

    print(classification_report(y_test, y_predict))
for model in models:

    model_test(model,X_train,X_test,y_train,y_test)
lr.fit(X_train, y_train)

y_predict = model.predict(X_test)

cm = confusion_matrix(y_test,y_predict)

sns.heatmap(cm, annot=True)
lr.fit(X,y)