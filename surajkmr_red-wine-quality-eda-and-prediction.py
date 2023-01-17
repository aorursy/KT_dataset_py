import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
train_df = pd.read_csv("../input/winequality-red.csv")

train_df.sample(5)
train_df.isnull().sum()
train_df.describe()
plt.figure(figsize=(15,15))

sns.heatmap(train_df.corr(),color = "k", annot=True)
sns.countplot(x='quality', data=train_df)
plt.figure(figsize=(15,5))

sns.swarmplot(x= "quality", y="fixed acidity" , data = train_df) 

plt.title('fixed acidity and quality')
plt.figure(figsize=(15,5))

sns.boxplot(x="quality", y="fixed acidity",   data=train_df )
train_df.groupby('quality')['fixed acidity'].mean().plot.line()

plt.ylabel("fixed acidity")
plt.figure(figsize=(10,4))

sns.barplot(x="quality", y="volatile acidity",   data=train_df )
train_df.groupby('quality')['volatile acidity'].mean().plot.line()

plt.ylabel("volatile acidity")
plt.figure(figsize=(10,4))

sns.barplot(x="quality", y="sulphates",   data=train_df )
train_df.groupby('quality')['sulphates'].mean().plot.line()

plt.ylabel("sulphates")
sns.boxplot(x="quality", y="sulphates",   data=train_df )
sns.boxplot(x="quality", y="pH",   data=train_df )
train_df.groupby('quality')['pH'].mean().plot.line()

plt.ylabel("pH")
sns.lmplot(x="fixed acidity", y="pH", data=train_df)
sns.lmplot(y="fixed acidity", x="citric acid", data=train_df)
reviews = []

for i in train_df['quality']:

    if i >= 1 and i <= 3:

        reviews.append('1')

    elif i >= 4 and i <= 7:

        reviews.append('2')

    elif i >= 8 and i <= 10:

        reviews.append('3')

train_df['Reviews'] = reviews

trainX = train_df.drop(['quality', 'Reviews'] , axis = 1)

trainy = train_df['Reviews']
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

%matplotlib inline
scaler = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(trainX, trainy, test_size = 0.2, random_state = 42)

X_train = scaler.fit_transform(X_train)

X_test = scaler.fit_transform(X_test)

score = {}
rfc = RandomForestClassifier(n_estimators=100)

rfc.fit(X_train, y_train)

predicted_rfc = rfc.predict(X_test)

print(classification_report(y_test, predicted_rfc))
rfc_conf_matrix = confusion_matrix(y_test, predicted_rfc)

rfc_acc_score = accuracy_score(y_test, predicted_rfc)

print(rfc_conf_matrix)

print(rfc_acc_score*100)

score.update({'Random_forest_classifier': rfc_acc_score*100})
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(X_train, y_train)

predicted_lr = lr.predict(X_test)

print(classification_report(y_test, predicted_lr))
lr_conf_matrix = confusion_matrix(y_test, predicted_lr)

lr_acc_score = accuracy_score(y_test, predicted_lr)

print(lr_conf_matrix)

print(lr_acc_score*100)

score.update({'logistic_regressor': lr_acc_score*100})
svc =  SVC()

svc.fit(X_train, y_train)

predicted_svc = svc.predict(X_test)

print(classification_report(y_test, predicted_svc))
svc_conf_matrix = confusion_matrix(y_test, predicted_svc)

svc_acc_score = accuracy_score(y_test, predicted_svc)

print(svc_conf_matrix)

print(svc_acc_score*100)

score.update({'SVC': svc_acc_score*100})
dt = DecisionTreeClassifier()

dt.fit(X_train,y_train)

predicted_dt = dt.predict(X_test)

print(classification_report(y_test, predicted_dt))
dt_conf_matrix = confusion_matrix(y_test, predicted_dt)

dt_acc_score = accuracy_score(y_test, predicted_dt)

print(dt_conf_matrix)

print(dt_acc_score*100)

score.update({'DecisionTreeClassifier': dt_acc_score*100})
gb = GaussianNB()

gb.fit(X_train,y_train)

predicted_gb = gb.predict(X_test)

print(classification_report(y_test, predicted_gb))
gb_conf_matrix = confusion_matrix(y_test, predicted_gb)

gb_acc_score = accuracy_score(y_test, predicted_gb)

print(gb_conf_matrix)

print(gb_acc_score*100)

score.update({'GaussianNB': gb_acc_score*100})
model_acc = pd.DataFrame()

model_acc['Models'] = score.keys() 

model_acc['Accuracy'] = score.values()

model_acc
from matplotlib.pyplot import xticks

sns.lineplot(x='Models', y='Accuracy',data=model_acc)

xticks(rotation=90)