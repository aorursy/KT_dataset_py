# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/pima-indians-diabetes-database/diabetes.csv")
data.head()
data.Outcome.value_counts().plot.pie(figsize = (6,6),autopct='%.1f')

plt.show()

print(data.Outcome.value_counts())
data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
def median_target(var):   

    temp = data[data[var].notnull()]

    temp = temp[[var, 'Outcome']].groupby(['Outcome'])[[var]].median().reset_index()

    return temp
data.groupby("Outcome").median()
data.loc[(data['Outcome'] == 0 ) & (data['Insulin'].isnull()), 'Insulin'] = 102.5

data.loc[(data['Outcome'] == 1 ) & (data['Insulin'].isnull()), 'Insulin'] = 169.5

data.loc[(data['Outcome'] == 0 ) & (data['Glucose'].isnull()), 'Glucose'] = 107

data.loc[(data['Outcome'] == 1 ) & (data['Glucose'].isnull()), 'Glucose'] = 140

data.loc[(data['Outcome'] == 0 ) & (data['SkinThickness'].isnull()), 'SkinThickness'] = 27

data.loc[(data['Outcome'] == 1 ) & (data['SkinThickness'].isnull()), 'SkinThickness'] = 32

data.loc[(data['Outcome'] == 0 ) & (data['BloodPressure'].isnull()), 'BloodPressure'] = 70

data.loc[(data['Outcome'] == 1 ) & (data['BloodPressure'].isnull()), 'BloodPressure'] = 74.5

data.loc[(data['Outcome'] == 0 ) & (data['BMI'].isnull()), 'BMI'] = 30.1

data.loc[(data['Outcome'] == 1 ) & (data['BMI'].isnull()), 'BMI'] = 34.3
f,ax = plt.subplots(figsize=(10, 10))

sns.heatmap(data.corr(), annot=True,  fmt= '.1f',ax=ax)

plt.show()
sns.countplot("Outcome",data = data)

plt.xlabel("Patient or not")

plt.ylabel("mean of BMI")

plt.show()

print(data.BMI.mean())
data["bmi_highlow"] = data.BMI

data.bmi_highlow = [1 if i >  31 else 0 for i in data.bmi_highlow]
plt.subplots(figsize=(10, 10))

sns.swarmplot(x="Outcome",y="Glucose",hue="bmi_highlow",data=data)

plt.show()
f,ax = plt.subplots(figsize=(10, 5))

sns.boxplot(x="Outcome",y="Glucose",hue="bmi_highlow",data=data,ax=ax)

plt.show()

sns.countplot("Outcome",data = data)

plt.show()

print("Mean of Glucose:",data.Glucose.mean())
data["glucose_highlow"] = data.Glucose

data.glucose_highlow = [1 if i >  120 else 0 for i in data.glucose_highlow]
f,ax = plt.subplots(figsize=(5, 5))

sns.heatmap(data[["Glucose","Age","Insulin"]].corr(), annot=True,  fmt= '.1f',ax=ax)

plt.show()
y = data.Outcome.values

X = data.drop("Outcome",axis = 1).values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
data = data.sample(frac=1,replace = False)

fraud_df = data.loc[data['Outcome'] == 1]

non_fraud_df = data.loc[data['Outcome'] == 0][:268]



normal_distributed_df = pd.concat([fraud_df, non_fraud_df])



df = normal_distributed_df.sample(frac=1,replace = False, random_state= 42 )
sns.countplot("Outcome",data = df)

plt.show()
yu = df.Outcome.values

Xu = df.drop("Outcome",axis = 1).values

Xu_train, Xu_test, yu_train, yu_test = train_test_split(Xu, yu, test_size=0.33, random_state=42)
from sklearn import metrics

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(Xu_train, yu_train)

yu_pred = logreg.predict(Xu_test)

print(metrics.accuracy_score(yu_test, yu_pred))
yu_pred = logreg.predict(Xu_test)

yu_true = yu_test

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(yu_true,yu_pred)

f, ax = plt.subplots(figsize=(8,8))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)

plt.xlabel("yu_pred")

plt.ylabel("yu_true")

plt.title("Confision Matrix")

plt.show()
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

score_list2=[]

for i in range(1,50):

    rt2=RandomForestClassifier(n_estimators=i,random_state=42)

    rt2.fit(Xu_train,yu_train)

    score_list2.append(rt2.score(Xu_test,yu_test))



plt.figure(figsize=(12,8))

plt.plot(range(1,50),score_list2)

plt.xlabel("Esimator values")

plt.ylabel("Acuuracy")

plt.show()
from sklearn.ensemble import RandomForestClassifier

rt2=RandomForestClassifier(n_estimators=40,random_state=42)

rt2.fit(Xu_train,yu_train)

yu_pred = rt2.predict(Xu_test)

yu_true = yu_test

print(metrics.accuracy_score(yu_test, yu_pred))

cm=confusion_matrix(yu_true,yu_pred)

f, ax = plt.subplots(figsize=(8,8))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)

plt.xlabel("yu_pred")

plt.ylabel("yu_true")

plt.title("Confision Matrix")

plt.show()