#loading required libraries

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import eli5

from sklearn.metrics import roc_auc_score

from sklearn.metrics import confusion_matrix, classification_report

from sklearn import metrics

from sklearn.model_selection import cross_val_score



import os

print(os.listdir("../input"))

#loading sensor_data

data = pd.read_csv('../input/sensor_data.csv')

data.head(5)
# checking datatypes of all the features

data.dtypes
data = data.astype({"x": int, "y": int, "z": int})

data['Timestamp'] = pd.to_datetime(data['Timestamp'])
# checking datatypes again after correction

data.dtypes
len(data)
data.isnull().sum()
print(data['joker present?'].value_counts())

sns.countplot(x='joker present?', data=data, palette="deep")

#data['joker present?'].value_counts().plot.bar()
import seaborn as sns

import matplotlib.pyplot as plot

print("**********X***************")

print(data['x'].value_counts())

print("******************************")

print("**********Y**************")

print(data['y'].value_counts())

print("******************************")

print("***********Z********")

print(data['z'].value_counts())





plot.figure(figsize=(14,14))

plot.subplot(2,2,1)

ax = sns.countplot(x='y', data=data, palette="vlag")

ax.set_xlabel("x")



plot.subplot(2,2,2)

ax = sns.countplot(x='y', data=data, palette="pastel")

ax.set_xlabel("y")



plot.subplot(2,2,3)

ax = sns.countplot(x='z', data=data, palette="Set3")

ax.set_xlabel("z")



plot.subplot(2,2,4)

ax = sns.countplot(x='number of thugs',data=data,order=pd.value_counts(data['number of thugs']).iloc[:20].index)

ax.set_xlabel("number of thugs")

df1 = data['number of thugs'].value_counts()[:20]

df1
plot.figure(figsize=(14,14))

plot.subplot(2,2,1)

ax = sns.countplot(x=data['joker present?'],hue=data['x'],data=data, palette="Set3")

ax.set_xlabel("x")



plot.subplot(2,2,2)

ax = sns.countplot(x=data['joker present?'],hue=data['y'],data=data, palette="Set3")

ax.set_xlabel("y")



plot.subplot(2,2,3)

ax = sns.countplot(x=data['joker present?'],hue=data['z'],data=data, palette="Set3")

ax.set_xlabel("z")



plot.figure(figsize=(14,14))

plot.subplot(2,2,1)

ax = sns.boxplot(x="number of citizens", data=data, palette="Set3")

ax.set_xlabel("Number of citizens")



plot.subplot(2,2,2)

ax = sns.boxplot(x="number of thugs", data=data, palette="Set3")

ax.set_xlabel("Number of thugs")
data['joker present?'].replace({'no':0, 'yes':1}, inplace=True)
data.dtypes
# checking correlation

corr = data.corr()

abs(corr['joker present?']).sort_values(ascending=False)
#data1= data



data1 = data.copy(deep=True)
data1.drop(['Timestamp'],axis=1, inplace=True)
data1.head(5)
data.head(5)
#separating target variable.

y = data1['joker present?']

data1.drop(['joker present?'],axis=1, inplace=True)
# splitting our dataset into train and test 

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data1, y, test_size = 0.3, random_state = 100)
len(y_test)
import lightgbm as lgb

model_lgb = lgb.LGBMClassifier(n_estimator=2000, 

                         learning_rate =0.08

                         )

model_lgb.fit(X_train, y_train)

eli5.explain_weights(model_lgb)
y_pred_lgbm = model_lgb.predict(X_test)
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred_lgbm)

conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])

plt.figure(figsize = (8,5))

sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")
print(classification_report(y_test,y_pred_lgbm))
print("Accuracy score with baseline lightgbm:", metrics.accuracy_score(y_test, y_pred_lgbm))

print("roc_auc score with decision tree:", roc_auc_score(y_test, y_pred_lgbm))
cv_scores = cross_val_score(model_lgb, X_train, y_train, cv=10)

cv_scores
print("Mean validation auc: {:.4f} +/- {:.4f}".format(np.mean(cv_scores), np.std(cv_scores)))
from sklearn.tree import DecisionTreeClassifier

#model_decision = DecisionTreeClassifier(min_samples_split=20, random_state=100)

model_decision = DecisionTreeClassifier(class_weight="balanced", random_state=100, max_depth=1)

model_decision.fit(X_train, y_train)
y_pred_decision = model_decision.predict(X_test)
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred_decision)

conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])

plt.figure(figsize = (8,5))

sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")
print(classification_report(y_test,y_pred_decision))
print("Accuracy score with decision tree:", metrics.accuracy_score(y_test, y_pred_decision))

print("roc_auc score with decision tree:", roc_auc_score(y_test, y_pred_decision))
cv_scores = cross_val_score(model_decision, X_train, y_train, cv=10)

cv_scores
print("Mean validation auc: {:.4f} +/- {:.4f}".format(np.mean(cv_scores), np.std(cv_scores)))
from sklearn.linear_model import LogisticRegression

logreg=LogisticRegression()

logreg.fit(X_train,y_train)

y_pred_log=logreg.predict(X_test)


from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred_log)

conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])

plt.figure(figsize = (8,5))

sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")
print(classification_report(y_test,y_pred_log))
print("Accuracy score with logistic regression:", metrics.accuracy_score(y_test, y_pred_log))

print("roc_auc score with logistic regression:", roc_auc_score(y_test, y_pred_log))
from sklearn.preprocessing import binarize

for i in range(1,5):

    cm2=0

    y_pred_prob_yes=logreg.predict_proba(X_test)

    y_pred2=binarize(y_pred_prob_yes,i/10)[:,1]

    cm2=confusion_matrix(y_test,y_pred2)

    print ('With',i/10,'threshold the Confusion Matrix is ','\n',cm2,'\n',

            'with',cm2[0,0]+cm2[1,1],'correct predictions and',cm2[1,0],'Type II errors( False Negatives)','\n\n',

          'Sensitivity: ',cm2[1,1]/(float(cm2[1,1]+cm2[1,0])),'Specificity: ',cm2[0,0]/(float(cm2[0,0]+cm2[0,1])),'\n\n\n')
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob_yes[:,1])

plt.plot(fpr,tpr)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.title('ROC curve for classifier')

plt.xlabel('False positive rate (1-Specificity)')

plt.ylabel('True positive rate (Sensitivity)')

plt.grid(True)
cv_scores = cross_val_score(logreg, X_train, y_train, cv=10)

print("Mean validation auc: {:.4f} +/- {:.4f}".format(np.mean(cv_scores), np.std(cv_scores)))
from sklearn.ensemble import RandomForestClassifier

random_classifier= RandomForestClassifier(class_weight="balanced", random_state=100)

random_classifier.fit(X_train,y_train)

y_pred_random= random_classifier.predict(X_test)
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred_random)

conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])

plt.figure(figsize = (8,5))

sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")
print(classification_report(y_test,y_pred_random))
print("Accuracy score with random forest:", metrics.accuracy_score(y_test, y_pred_random))

print("precision score with random forest:", metrics.precision_score(y_test, y_pred_random))

cv_scores = cross_val_score(random_classifier, X_train, y_train, cv=10)

print("Mean validation auc: {:.4f} +/- {:.4f}".format(np.mean(cv_scores), np.std(cv_scores)))
#loading all the datasets

data2= data

df1 = pd.read_csv('../input/bat_signal_data.csv')

df2 = pd.read_csv('../input/moon_phases_data.csv')

df3 = pd.read_csv('../input/weather_data.csv')

data2.head(5)
data3= pd.concat([data2, df1, df2, df3], ignore_index=True)

data3.head(5)
# checking the data types

data3.dtypes
data3['x'] = data3['x'].fillna(0).astype(int)

data3['y'] = data3['y'].fillna(0).astype(int)

data3['z'] = data3['z'].fillna(0).astype(int)

data3['joker present?'] = data3['joker present?'].fillna(0).astype(int)

data3['number of citizens'] = data3['number of citizens'].fillna(0).astype(int)

data3['number of thugs'] = data3['number of thugs'].fillna(0).astype(int)

data3['temperature'] = data3['temperature'].fillna(00.00000)
data3.dtypes
data3.head(5)
data3['phase'] = data3['phase'].fillna('no info phase')

data3['precip'] = data3['precip'].fillna('no info precip')

data3['sky conditions'] = data3['sky conditions'].fillna('no info sky')

data3['status'] = data3['status'].fillna('no info status')
data3.head(5)
import seaborn as sns

import matplotlib.pyplot as plot

print("**********Phase***************")

print(data3['phase'].value_counts())

print("******************************")

print("**********precip**************")

print(data3['precip'].value_counts())

print("******************************")

print("***********sky conditions********")

print(data3['sky conditions'].value_counts())

print("******************************")

print("*********status*************")

print(data3['status'].value_counts())



plot.figure(figsize=(25,35))

plot.subplot(2,2,1)

ax = data3['phase'].value_counts().plot.bar()

ax.set_xlabel("Phase")



plot.subplot(2,2,2)

ax = data3['precip'].value_counts().plot.bar()

ax.set_xlabel("precip")



plot.subplot(2,2,3)

ax = data3['sky conditions'].value_counts().plot.bar()

ax.set_xlabel("sky conditions")



plot.subplot(2,2,4)

ax = data3['status'].value_counts().plot.bar()

ax.set_xlabel("status")
#encoding categorical variables

precip=pd.get_dummies(data3['precip'])

sky_condition = pd.get_dummies(data3['sky conditions'])

status = pd.get_dummies(data3['status'])

phase = pd.get_dummies(data3['phase'])
#adding all the above dataframes into our main dataframe

data3 = pd.concat([data3, phase, status, sky_condition, precip], axis=1)
data3.head(5)
data3.drop(['phase','precip','sky conditions','status'],axis=1, inplace=True)
data3.drop(['Timestamp'], axis=1, inplace=True)
data3.head(5)
corr = data3.corr()

abs(corr['joker present?']).sort_values(ascending=False)
#splitting target column

y_new = data3['joker present?']
#dropping target feature from the main dataframe

data3.drop(['joker present?'],axis=1, inplace=True)
data3.head(5)
# splitting our dataset into train and test 

X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(data3, y_new, test_size = 0.3, random_state = 100)
#check the split of train and test shape

print('Train:',X_train_new.shape)

print('Test:',X_test_new.shape)
model = lgb.LGBMClassifier(n_estimator=2000,

                         learning_rate =0.05

                         )

model.fit(X_train_new, y_train_new)

eli5.explain_weights(model, top=30)
y_pred_lgbm_new = model.predict(X_test_new)
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test_new,y_pred_lgbm_new)

conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])

plt.figure(figsize = (8,5))

sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")
print(classification_report(y_test_new,y_pred_lgbm_new))
print("Accuracy score with baseline lightgbm:", metrics.accuracy_score(y_test_new, y_pred_lgbm_new))

print("roc_auc score with baseline lightgbm:", roc_auc_score(y_test_new, y_pred_lgbm_new))
# Feature scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train_new1 = sc.fit_transform(X_train_new)

X_test_new1 = sc.fit_transform(X_test_new)
X_train_new1
model = lgb.LGBMClassifier(n_estimator=2000,

                         learning_rate =0.05

                         )

model.fit(X_train_new1, y_train_new)

y_pred_lgbm_new1 = model.predict(X_test_new1)

print("Accuracy score:", metrics.accuracy_score(y_test_new, y_pred_lgbm_new1))
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test_new,y_pred_lgbm_new)

conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])

plt.figure(figsize = (8,5))

sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")
cv_scores = cross_val_score(model, X_train_new, y_train_new, cv=10)

print("Mean validation auc: {:.4f} +/- {:.4f}".format(np.mean(cv_scores), np.std(cv_scores)))
model_decision = DecisionTreeClassifier(class_weight="balanced", random_state=100, max_depth=1)

model_decision.fit(X_train_new, y_train_new)
y_pred_decision = model_decision.predict(X_test_new)
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test_new,y_pred_decision)

conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])

plt.figure(figsize = (8,5))

sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")
print(classification_report(y_test_new,y_pred_decision))
print("Accuracy score with decision tree:", metrics.accuracy_score(y_test_new, y_pred_decision))

print("roc_auc score with decision tree:", roc_auc_score(y_test_new, y_pred_decision))
cv_scores = cross_val_score(model_decision, X_train_new, y_train_new, cv=10)

print("Mean validation auc: {:.4f} +/- {:.4f}".format(np.mean(cv_scores), np.std(cv_scores)))
from sklearn.linear_model import LogisticRegression

logreg=LogisticRegression()

logreg.fit(X_train_new,y_train_new)

y_pred_log=logreg.predict(X_test_new)
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test_new,y_pred_log)

conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])

plt.figure(figsize = (8,5))

sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")
print(classification_report(y_test_new,y_pred_log))
print("Accuracy score with logistic regression:", metrics.accuracy_score(y_test_new, y_pred_log))

print("roc_auc score with logistic regression:", roc_auc_score(y_test_new, y_pred_log))
cv_scores = cross_val_score(logreg, X_train_new, y_train_new, cv=10)

print("Mean validation auc: {:.4f} +/- {:.4f}".format(np.mean(cv_scores), np.std(cv_scores)))
from sklearn.ensemble import RandomForestClassifier

random_classifier= RandomForestClassifier(class_weight="balanced", random_state=100)

random_classifier.fit(X_train_new,y_train_new)

y_pred_random= random_classifier.predict(X_test_new)
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test_new,y_pred_random)

conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])

plt.figure(figsize = (8,5))

sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")
print(classification_report(y_test_new,y_pred_random))
print("Accuracy score with Random Forest:", metrics.accuracy_score(y_test_new, y_pred_random))

print("roc_auc score with Random Forest:", roc_auc_score(y_test_new, y_pred_random))