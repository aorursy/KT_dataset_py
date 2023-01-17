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
df_h = pd.read_csv(os.path.join(dirname, filename))
df_h
import matplotlib.pyplot as plt

import seaborn as sns
def basic_info(data):

    print("Dataset size is: ", data.size)

    print("Dataset shape is: ", data.shape)

    print("Dataset dimensions is: ", data.ndim)

    print("Dataset columns are: ", data.columns)

    print(data.info())

    cat, num = list(), list()

    for i in data.columns:

        if data[i].dtype == object:

            cat.append(i)

        else:

            num.append(i)

    print("Categotical columns are: ", cat)

    print("Numerical columns are: ", num)

    

    return cat, num
categorical1, numerical1 = basic_info(df_h)
df_h.isnull().sum()
corr = df_h.corr()

plt.figure(figsize=(35,35))

ax= sns.heatmap(corr, vmin = -1, vmax = 1, square = True, annot=True)

ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, horizontalalignment = 'right', fontsize=10)

plt.show()
sns.pairplot(df_h, kind = 'reg')
df_h['age'].value_counts()
plt.figure(figsize = (20,8))

sns.countplot(df_h['age'], edgecolor = '#DC143C')

plt.show()
plt.figure(figsize = (20,8))

plt.hist(df_h['age'], edgecolor = '#DC143C')

plt.show()
df_h['anaemia'].value_counts()
plt.figure(figsize=(20,9))

graph = sns.countplot(df_h['anaemia'])

i=0

for p in graph.patches:

    #print(p)

    height = p.get_height()

    graph.text(p.get_x()+p.get_width()/2., height + 0.1,

        df_h['anaemia'].value_counts()[i],ha="center")

    i += 1

#plt.legend()

plt.show()
plt.figure(figsize=(20,9))

graph = sns.countplot(df_h['anaemia'], hue = df_h['sex'])

i=0

for p in graph.patches:

    #print(p)

    height = p.get_height()

    graph.text(p.get_x()+p.get_width()/2., height + 0.1,

        height,ha="center")

    i += 1

#plt.legend()

plt.show()
plt.figure(figsize=(20,9))

graph = sns.countplot(df_h['anaemia'], hue = df_h['DEATH_EVENT'])

i=0

for p in graph.patches:

    #print(p)

    height = p.get_height()

    graph.text(p.get_x()+p.get_width()/2., height + 0.1,

        height,ha="center")

    i += 1

#plt.legend()

plt.show()
df_h['creatinine_phosphokinase'].value_counts()
plt.figure(figsize=(20,8))

plt.hist(df_h['creatinine_phosphokinase'], color = "#68228B", edgecolor = '#97FFFF')

plt.show()
df_h['diabetes'].value_counts()
plt.figure(figsize=(20,9))

graph = sns.countplot(df_h['diabetes'])

i=0

for p in graph.patches:

    #print(p)

    height = p.get_height()

    graph.text(p.get_x()+p.get_width()/2., height + 0.1,

        height,ha="center")

    i += 1

#plt.legend()

plt.show()
plt.figure(figsize=(20,9))

graph = sns.countplot(df_h['diabetes'], hue = df_h['sex'])

i=0

for p in graph.patches:

    #print(p)

    height = p.get_height()

    graph.text(p.get_x()+p.get_width()/2., height + 0.1,

        height,ha="center")

    i += 1

#plt.legend()

plt.show()
plt.figure(figsize=(20,9))

graph = sns.countplot(df_h['diabetes'], hue = df_h['DEATH_EVENT'])

i=0

for p in graph.patches:

    #print(p)

    height = p.get_height()

    graph.text(p.get_x()+p.get_width()/2., height + 0.1,

        height,ha="center")

    i += 1

#plt.legend()

plt.show()
df_h['ejection_fraction']
plt.figure(figsize=(20,8))

plt.hist(df_h['ejection_fraction'], edgecolor = 'green')

plt.show()
plt.figure(figsize=(20,9))

graph = sns.countplot(df_h['ejection_fraction'], hue = df_h['sex'])

i=0

for p in graph.patches:

    #print(p)

    height = p.get_height()

    graph.text(p.get_x()+p.get_width()/2., height + 0.1,

        height,ha="center")

    i += 1

#plt.legend()

plt.show()
df_h['high_blood_pressure']
plt.figure(figsize=(20,9))

graph = sns.countplot(df_h['high_blood_pressure'])

i=0

for p in graph.patches:

    #print(p)

    height = p.get_height()

    graph.text(p.get_x()+p.get_width()/2., height + 0.1,

        height,ha="center")

    i += 1

plt.show()
plt.figure(figsize=(20,9))

graph = sns.countplot(df_h['high_blood_pressure'], hue = df_h['sex'])

i=0

for p in graph.patches:

    #print(p)

    height = p.get_height()

    graph.text(p.get_x()+p.get_width()/2., height + 0.1,

        height,ha="center")

    i += 1

plt.show()
plt.figure(figsize=(20,9))

graph = sns.countplot(df_h['high_blood_pressure'], hue = df_h['DEATH_EVENT'])

i=0

for p in graph.patches:

    #print(p)

    height = p.get_height()

    graph.text(p.get_x()+p.get_width()/2., height + 0.1,

        height,ha="center")

    i += 1

plt.show()
df_h['serum_creatinine']
plt.figure(figsize=(20,8))

sns.distplot(df_h['serum_creatinine'])
df_h['serum_sodium'].value_counts()
plt.figure(figsize=(20,8))

sns.distplot(df_h['serum_sodium'])
df_h['sex'].value_counts()
labels = df_h['sex'].value_counts().index.tolist()

sizes = df_h['sex'].value_counts()

fig, ax=plt.subplots()

ax.pie(sizes, labels=labels, autopct="%1.1f%%", shadow=True, startangle=90)

ax.axis('equal')

plt.show()
df_h['smoking'].value_counts()
labels = df_h['smoking'].value_counts().index.tolist()

sizes = df_h['smoking'].value_counts()

fig, ax=plt.subplots()

explode = (0, 0.2)

ax.pie(sizes, explode = explode,labels=labels, autopct="%1.1f%%", shadow=True, startangle=90)

ax.axis('equal')

plt.show()
df_h['time'].value_counts()
plt.figure(figsize = (20,8))

plt.hist(df_h['time'], edgecolor = 'red')

plt.show()
# age, creatinine_phosphokinase, ejection_fraction, platelets, serum_creatinine, serum_sodium, time are the columns which we will use for predicting DEATH_EVENT.

# we will see for outliers

# subplot(nrows, ncols, index, **kwargs)

plt.figure(figsize=(20,13))

ax=plt.subplot(221)

plt.boxplot(df_h['age'])

plt.title('Age')

ax = plt.subplot(222)

plt.boxplot(df_h['creatinine_phosphokinase'])

plt.title('creatinine_phosphokinase')

ax = plt.subplot(223)

plt.boxplot(df_h['ejection_fraction'])

plt.title('ejection_fraction')

ax = plt.subplot(224)

plt.boxplot(df_h['platelets'])

plt.title('platelets')
plt.figure(figsize=(20,13))

ax = plt.subplot(221)

plt.boxplot(df_h['serum_creatinine'])

plt.title('serum_creatinine')

ax = plt.subplot(222)

plt.boxplot(df_h['serum_sodium'])

plt.title('serum_sodium')

ax = plt.subplot(223)

plt.boxplot(df_h['time'])

plt.title('time')
df_h = df_h[df_h['ejection_fraction'] < 70]
df_h['creatinine_phosphokinase'] = np.where(df_h['creatinine_phosphokinase'] > df_h['creatinine_phosphokinase'].quantile(0.95), df_h['creatinine_phosphokinase'].quantile(0.50), df_h['creatinine_phosphokinase'])
sns.boxplot(df_h['creatinine_phosphokinase'])
df_h['platelets'] = np.where(df_h['platelets'] > df_h['platelets'].quantile(0.95), df_h['platelets'].quantile(0.50), df_h['platelets'])
sns.boxplot(df_h['platelets'])
df_h['serum_creatinine'] = np.where(df_h['serum_creatinine'] > df_h['serum_creatinine'].quantile(0.95), df_h['serum_creatinine'].quantile(0.50), df_h['serum_creatinine'])
sns.boxplot(df_h['serum_creatinine'])
df_h['serum_sodium'] = np.where(df_h['serum_sodium'] > df_h['serum_sodium'].quantile(0.95), df_h['serum_sodium'].quantile(0.50), df_h['serum_sodium'])
sns.boxplot(df_h['serum_sodium'])
from sklearn.model_selection import train_test_split
X = df_h.loc[:, ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium', 'time']]

y = df_h.loc[:, 'DEATH_EVENT']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 10)
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
model_lr = lr.fit(X_train, y_train)
y_lr_predict = model_lr.predict(X_test)
LR_df = pd.DataFrame(data = {"Actual": y_test, "Predicted": y_lr_predict})
LR_df
model_lr.score(X_test, y_test)
score_lr = accuracy_score(y_test, y_lr_predict)
score_lr
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
model_rfr = rfc.fit(X_train, y_train)
y_rfr_predict = model_rfr.predict(X_test)
RFR_df = pd.DataFrame(data = {"Actual": y_test, "Predicted": y_rfr_predict})
RFR_df
model_rfr.score(X_test, y_test)
score_rfr = accuracy_score(y_test, y_rfr_predict)
score_rfr
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
model_gnb = gnb.fit(X_train, y_train)
y_gnb_predict = model_gnb.predict(X_test)
GNB_df = pd.DataFrame(data = {"Actual":y_test, "Predicted": y_gnb_predict})
GNB_df
model_gnb.score(X_test, y_test)
score_gnb = accuracy_score(y_test, y_gnb_predict)
score_gnb
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(random_state=42)
model_dtc = dtc.fit(X_train, y_train)
y_dtc_predit = model_dtc.predict(X_test)
DTC_df = pd.DataFrame(data = {'Actual': y_test, "Predicted": y_dtc_predit})
DTC_df
model_dtc.score(X_test, y_test)
score_dtc = accuracy_score(y_test, y_dtc_predit)
score_dtc
print("Logistic Regression accuracy: ", score_lr)

print("Random Forest Classifier accuracy: ", score_rfr)

print("Naive Bayes accuracy: ", score_gnb)

print("DecisionTreeClassifier accuracy", score_dtc)