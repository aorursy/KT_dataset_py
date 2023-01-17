import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import statsmodels.api as sm
#from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
data = pd.read_csv("../input/persnalloan/Bank_Personal_Loan_Modelling.csv")
data.head()
# No missing values in the data

data.info()
for column in data.columns:
    uniques = sorted(data[column].unique())
    print('{0:20s} {1:5d}\t'.format(column, len(uniques)), uniques[:5])
data.describe()
# data cleaning ideas

filter = data['ZIP Code']<90000
print(len(data[filter]))

data = data[-filter]
# data cleaning ideas

filter = data['Experience']<0
print(len(data[filter]))

data = data[-filter]
column_list = list(data.columns)
list_len = len(column_list)
fig, axes = plt.subplots(4, 4, figsize=(16,12))
fig.subplots_adjust(hspace=0.5)

for i in range(list_len):
    sns.distplot(data[column_list[i]], ax=axes.flatten()[i], kde=False, color='k')
    axes.flatten()[i].set(title='Count by '+column_list[i], xlabel='')

plt.show()    

cmap = sns.diverging_palette(250, 10, as_cmap=True)
sns.heatmap(data.corr(), cmap = cmap, annot = False)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
sns.countplot(x='Family', hue='Personal Loan', data=data, ax=ax[0])
ax[0].set_title('Count Plot by Family Size', fontsize=16)
ax[0].set_yscale('log')
sns.barplot(x='Family', y='Personal Loan', data=data, ax=ax[1]);
ax[1].set_title('Mean Acceptance Rate by Family Size', fontsize=16)
plt.tight_layout()
plt.show()
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
sns.countplot(x='Education', hue='Personal Loan', data=data, ax=ax[0])
ax[0].set_title('Count Plot by Education', fontsize=16)
ax[0].set_yscale('log')
sns.barplot(x='Education', y='Personal Loan', data=data, ax=ax[1]);
ax[1].set_title('Mean Acceptance Rate by Education', fontsize=16)
plt.tight_layout()
plt.show()
#grouped = data[['ZIP Code', 'Personal Loan']].groupby('ZIP Code').mean().reset_index()

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
sns.countplot(x='ZIP Code', hue='Personal Loan', data=data, ax=ax[0])
ax[0].set_title('Count Plot by ZIP Code', fontsize=16)
ax[0].set_yscale('log')
sns.barplot(x='ZIP Code', y='Personal Loan', data=data, ax=ax[1]);
ax[1].set_title('Mean Acceptance Rate by ZIP Code', fontsize=16)
plt.tight_layout()
plt.show()
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
sns.countplot(x='Securities Account', hue='Personal Loan', data=data, ax=ax[0])
ax[0].set_title('Count Plot by Securities Account', fontsize=16)
ax[0].set_yscale('log')
sns.barplot(x='Securities Account', y='Personal Loan', data=data, ax=ax[1]);
ax[1].set_title('Mean Acceptance Rate by Securities Account', fontsize=16)
plt.tight_layout()
plt.show()
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
sns.countplot(x='CD Account', hue='Personal Loan', data=data, ax=ax[0])
ax[0].set_title('Count Plot by CD Account', fontsize=16)
ax[0].set_yscale('log')
sns.barplot(x='CD Account', y='Personal Loan', data=data, ax=ax[1]);
ax[1].set_title('Mean Acceptance Rate by CD Account', fontsize=16)
plt.tight_layout()
plt.show()
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
sns.countplot(x='Online', hue='Personal Loan', data=data, ax=ax[0])
ax[0].set_title('Count Plot by Online', fontsize=16)
ax[0].set_yscale('log')
sns.barplot(x='Online', y='Personal Loan', data=data, ax=ax[1]);
ax[1].set_title('Mean Acceptance Rate by Online', fontsize=16)
plt.tight_layout()
plt.show()
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
sns.countplot(x='CreditCard', hue='Personal Loan', data=data, ax=ax[0])
ax[0].set_title('Count Plot by CreditCard', fontsize=16)
ax[0].set_yscale('log')
sns.barplot(x='CreditCard', y='Personal Loan', data=data, ax=ax[1]);
ax[1].set_title('Mean Acceptance Rate by CreditCard', fontsize=16)
plt.tight_layout()
plt.show()
grouped = data[['Age', 'Personal Loan']].groupby('Age').mean().reset_index()
hist_kws={'histtype': 'bar', 'edgecolor':'black', 'alpha': 0.2}

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
sns.distplot(data[data['Personal Loan'] == 0]['Age'], label='Not accepted', 
             ax=ax[0], hist_kws=hist_kws)
sns.distplot(data[data['Personal Loan'] == 1]['Age'], label='Accepted', 
             ax=ax[0], hist_kws=hist_kws)
ax[0].set_title('Count Plot by Age', fontsize=16)
ax[0].legend()
ax[1].plot(grouped['Age'], grouped['Personal Loan'], '.-')
ax[1].set_title('Mean Acceptance Rate vs. Age', fontsize=16)
ax[1].set_xlabel('Age')
ax[1].set_ylabel('Mean Acceptance Rate')
ax[1].grid(True)
plt.show()
grouped = data[['Experience', 'Personal Loan']].groupby('Experience').mean().reset_index()
hist_kws={'histtype': 'bar', 'edgecolor':'black', 'alpha': 0.2}

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
sns.distplot(data[data['Personal Loan'] == 0]['Experience'], label='Not accepted', 
             ax=ax[0], hist_kws=hist_kws)
sns.distplot(data[data['Personal Loan'] == 1]['Experience'], label='Accepted', 
             ax=ax[0], hist_kws=hist_kws)
ax[0].set_title('Count Plot by Years of Professional Experience', fontsize=16)
ax[0].legend()
ax[1].plot(grouped['Experience'], grouped['Personal Loan'], '.-')
ax[1].set_title('Mean Acceptance Rate vs. Experience', fontsize=16)
ax[1].set_xlabel('Years of Professional Experience')
ax[1].set_ylabel('Mean Acceptance Rate')
ax[1].grid(True)
plt.show()
grouped = data[['Income', 'Personal Loan']].groupby('Income').mean().reset_index()
hist_kws={'histtype': 'bar', 'edgecolor':'black', 'alpha': 0.2}

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
sns.distplot(data[data['Personal Loan'] == 0]['Income'], label='Not accepted', 
             ax=ax[0], hist_kws=hist_kws)
sns.distplot(data[data['Personal Loan'] == 1]['Income'], label='Accepted', 
             ax=ax[0], hist_kws=hist_kws)
ax[0].set_title('Count Plot by Income', fontsize=16)
ax[0].legend()
ax[1].plot(grouped['Income'], grouped['Personal Loan'], '.-')
ax[1].set_title('Mean Acceptance Rate vs. Income', fontsize=16)
ax[1].set_xlabel('Income')
ax[1].set_ylabel('Mean Acceptance Rate')
ax[1].grid(True)
plt.show()
grouped = data[['CCAvg', 'Personal Loan']].groupby('CCAvg').mean().reset_index()
hist_kws={'histtype': 'bar', 'edgecolor':'black', 'alpha': 0.2}

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
sns.distplot(data[data['Personal Loan'] == 0]['CCAvg'], label='Not accepted', 
             ax=ax[0], hist_kws=hist_kws)
sns.distplot(data[data['Personal Loan'] == 1]['CCAvg'], label='Accepted', 
             ax=ax[0], hist_kws=hist_kws)
ax[0].set_title('Count Plot by Avg. Monthly Spending on Credit Cards', fontsize=16)
ax[0].legend()
ax[1].plot(grouped['CCAvg'], grouped['Personal Loan'], '.-')
ax[1].set_title('Mean Acceptance Rate vs. Avg. Monthly Spending on Credit Cards', fontsize=16)
ax[1].set_xlabel('Avg. Monthly Spending on Credit Cards')
ax[1].set_ylabel('Mean Acceptance Rate')
ax[1].grid(True)
plt.show()
grouped = data[['Mortgage', 'Personal Loan']].groupby('Mortgage').mean().reset_index()
hist_kws={'histtype': 'bar', 'edgecolor':'black', 'alpha': 0.2}

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
sns.distplot(data[data['Personal Loan'] == 0]['Mortgage'], label='Not accepted', 
             ax=ax[0], hist_kws=hist_kws)
sns.distplot(data[data['Personal Loan'] == 1]['Mortgage'], label='Accepted', 
             ax=ax[0], hist_kws=hist_kws)
ax[0].set_title('Count Plot by Mortgage', fontsize=16)
ax[0].legend()
ax[1].plot(grouped['Mortgage'], grouped['Personal Loan'], '.-')
ax[1].set_title('Mean Acceptance Rate vs Mortgage', fontsize=16)
ax[1].set_xlabel('Mortgage')
ax[1].set_ylabel('Mean Acceptance Rate')
ax[1].grid(True)
plt.show()
data['Mortgage_Y'] = 0
data.loc[data['Mortgage']>0,'Mortgage_Y']=1

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
sns.countplot(x='Mortgage_Y', hue='Personal Loan', data=data, ax=ax[0])
ax[0].set_title('Count Plot by Mortgage_Y', fontsize=16)
#ax[0].set_yscale('log')
sns.barplot(x='Mortgage_Y', y='Personal Loan', data=data, ax=ax[1]);
ax[1].set_title('Mean Acceptance Rate by Mortgage_Y', fontsize=16)
plt.tight_layout()
plt.show()
data['Education_catog']=data['Education'].astype(str)
data_encode = pd.get_dummies(data)
data_encode.columns
# These columns are not added: ID, ZIP Code, Education, Mortgage
columns = ['Age', 'Experience', 'Income', 'Family', 'CCAvg',
       'Securities Account', 'CD Account', 'Online', 'CreditCard','Mortgage_Y', 'Education_catog_1',
       'Education_catog_2', 'Education_catog_3']
X = data_encode[columns]
y = data_encode['Personal Loan']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print(classification_report(y_train, logreg.predict(X_train)))
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.4f}'.format(logreg.score(X_test, y_test)))
confusion_mat = confusion_matrix(y_test, y_pred)
print(confusion_mat)
print(classification_report(y_test, y_pred))
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.3f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
rf = RandomForestClassifier(n_estimators = 100, random_state = 42, oob_score = True)
rf.fit(X_train, y_train.values.ravel())
rf.oob_score_
print(classification_report(y_train, rf.predict(X_train)))
y_pred = rf.predict(X_test)
confusion_mat = confusion_matrix(y_test, y_pred)
print(confusion_mat)
print(classification_report(y_test, y_pred))
importances = list(rf.feature_importances_)
features = list(X.columns)

feature_importance = [(f, i) for f, i in zip(features, importances)]
feature_importance = sorted(feature_importance, key = lambda x: x[1], reverse = True)
print("Features and Importances: ")
s = [print("{0:s}: {1:4.3f}".format(*p)) for p in feature_importance]
roc_auc = roc_auc_score(y_test, y_pred)
fpr, tpr, thresholds = roc_curve(y_test, rf.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='AUC = %0.3f' % roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest ROC')
plt.legend(loc="lower right")

plt.show()
for threshold in [0.25+i/10 for i in range(3)]:

    predicted_proba = rf.predict_proba(X_test)
    predicted = (predicted_proba [:,1] >= threshold).astype('int')

    print("Threshold is", threshold)
    print(classification_report(y_test, predicted))
len(y_train[y_train == 1])
len(y_train[y_train == 0])
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)
len(y_res[y_res==1])
rf = RandomForestClassifier(n_estimators = 100, random_state = 42, oob_score = True)
rf.fit(X_res, y_res)
rf.oob_score_
print(classification_report(y_res, rf.predict(X_res)))
y_pred = rf.predict(X_test)
confusion_mat = confusion_matrix(y_test, y_pred)
print(confusion_mat)
print(classification_report(y_test, y_pred))
importances = list(rf.feature_importances_)
features = list(X_res.columns)

feature_importance = [(f, i) for f, i in zip(features, importances)]
feature_importance = sorted(feature_importance, key = lambda x: x[1], reverse = True)
print("Features and Importances: ")
s = [print("{0:s}: {1:4.3f}".format(*p)) for p in feature_importance]
roc_auc = roc_auc_score(y_test, y_pred)
fpr, tpr, thresholds = roc_curve(y_test, rf.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='AUC = %0.3f' % roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest ROC')
plt.legend(loc="lower right")
#plt.savefig('Log_ROC')
plt.show()
