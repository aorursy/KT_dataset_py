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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # data visualization

import seaborn as sns # statistical data visualization

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import LinearSVC

from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv("../input/weather-dataset-rattle-package/weatherAUS.csv")

df.head()
categorical = [var for var in df.columns if df[var].dtype=='O']

print('There are {} categorical variables\n'.format(len(categorical)))

print('The categorical variables are :', categorical)
cat1 = [var for var in categorical if df[var].isnull().sum()!=0]

print(df[cat1].isnull().sum())
for var in categorical:

    print(var + ' conatins '+str(len(df[var].unique()))+ " labels ")
df['Date'] = pd.to_datetime(df['Date'])

df['Year'] = df['Date'].dt.year

df['Month'] = df['Date'].dt.month

df['Day'] = df['Date'].dt.day



df.drop('Date',axis=1,inplace=True)
categorical = [var for var in df.columns if df[var].dtype=='O']

print("There are {} categorical variables : ".format(len(categorical)))

print(categorical)
for var in categorical:

    df[var].fillna(df[var].mode()[0],inplace=True)
numerical = [var for var in df.columns if df[var].dtype!='O']

print(numerical)
num1 = df[numerical].isnull().sum()

num1 = num1[num1!=0]

num1
for col in num1.index:

    col_mean = df[col].mean()

    df[col].fillna(col_mean,inplace=True)
le = LabelEncoder()

new_df = df

for col in categorical:

    new_df[col] = le.fit_transform(df[col])

col_names = new_df.columns
new_df.head()
from sklearn.preprocessing import MinMaxScaler

ss = MinMaxScaler()

new_df = ss.fit_transform(new_df)

new_df = pd.DataFrame(new_df,columns = col_names )
new_df.describe()
# new_df.to_csv("weatherCleaned.csv")
correlation = new_df.corr()

plt.figure(figsize=(16,12))

plt.title('Correlation Heatmap of Rain in Australia Dataset')

ax = sns.heatmap(correlation, square=True, annot=True, fmt='.2f', linecolor='white',cmap='viridis')

ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

ax.set_yticklabels(ax.get_yticklabels(), rotation=30)           

plt.show()
y = new_df.RainTomorrow

X = new_df.drop('RainTomorrow',axis=1)
results = []
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42,shuffle=True)
gnb = GaussianNB()

gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

gnb.score(X_test,y_test)
print(accuracy_score(y_test,y_pred))

print(cross_val_score(gnb,X_train,y_train,cv=3))

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))

results.append(accuracy_score(y_test,y_pred))

sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,annot_kws={"size": 12},cmap='viridis',fmt="d")
dtc = DecisionTreeClassifier(max_depth=10, min_samples_split=2,random_state=42)

dtc.fit(X_train,y_train)

y_pred = dtc.predict(X_test)

dtc.score(X_test,y_test)
print(accuracy_score(y_test,y_pred))

print(cross_val_score(dtc,X_train,y_train,cv=3))

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))

results.append(accuracy_score(y_test,y_pred))

sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,annot_kws={"size": 12},cmap='viridis',fmt="d")
svc = LinearSVC(random_state=42)

svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)

svc.score(X_test,y_test)

print(cross_val_score(svc,X_train,y_train,cv=3))
print(accuracy_score(y_test,y_pred))

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))

results.append(accuracy_score(y_test,y_pred))

sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,annot_kws={"size": 12},cmap='viridis',fmt="d")
rfc = RandomForestClassifier(n_estimators=200,max_depth=10, random_state=42)

rfc.fit(X_train,y_train)

y_pred = rfc.predict(X_test)

rfc.score(X_test,y_test)
print(accuracy_score(y_test,y_pred))

print(cross_val_score(rfc,X_train,y_train,cv=3))

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))

results.append(accuracy_score(y_test,y_pred))

sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,annot_kws={"size": 12},cmap='viridis',fmt="d")
names = ["Naive Bayes","Decision Tree","Linear SVM","Random Forest",]

results
sns.barplot(names,results)