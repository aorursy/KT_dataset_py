import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

base_color = sns.color_palette()[0]
data_df = pd.read_csv('/kaggle/input/patient-survey-score-with-demographics/pxdata.csv')

data_df.head(2)
data_df.isnull().sum()
data_df.dtypes
data_df.nunique()
data_df['Perfect'].value_counts()
data_df['Rate'].value_counts()
data_df['Recommend'].value_counts()
data_df['Health'].value_counts()
data_df['Mental'].value_counts()
data_df['College'].value_counts()
data_df['White'].value_counts()
data_df['English'].value_counts()
data_df['Service'].value_counts()
data_df['Specialty'].value_counts()
data_df['Unit'].value_counts()
data_df['Source'].value_counts()
data_df['Home'].value_counts()
data_df['Age'].unique()
data_df['Stay'].unique()
data_df['Visit'].unique()
cleaned_data = data_df.copy()
cleaned_data['Date'] = pd.to_datetime(cleaned_data['Date'])
cleaned_data.dtypes
cleaned_data['Perfect'] = cleaned_data['Perfect'].apply(lambda x : True if x==1 else False)

cleaned_data['College'] = cleaned_data['College'].apply(lambda x : True if x=='Y' else False)

cleaned_data['White'] = cleaned_data['White'].apply(lambda x : True if x=='Y' else False)

cleaned_data['English'] = cleaned_data['English'].apply(lambda x : True if x=='Y' else False)
cleaned_data.dtypes
cleaned_data['Stay'] = cleaned_data['Stay'].replace({'2+':'2-3', '4+': '4-7'})
cleaned_data['Stay'].unique()
cleaned_data['Age'] = cleaned_data["Age"].apply(lambda x: '90+' if x=='80+' else x)
cleaned_data['Age'].unique()
from pandas.api.types import CategoricalDtype

age_cat = CategoricalDtype(['18-34', '35-49', '50-64', '65-79', '80-90', '90+'], ordered=True)
cleaned_data['Age'] = cleaned_data['Age'].astype(age_cat)
cleaned_data['Age'].dtype
stay_cat = CategoricalDtype(['1', '2-3', '4-7', '8+'], ordered=True)

cleaned_data['Stay'] = cleaned_data['Stay'].astype(stay_cat)
cleaned_data.dtypes
cleaned_data['Composite'].hist(bins=100)

plt.xlabel('Composite')

plt.ylabel('Count')

plt.show()
fig = plt.figure(figsize=(10,10))

ax1 = fig.add_subplot(2,2,1)

ax2 = fig.add_subplot(2,2,2)

ax3 = fig.add_subplot(2,2,3)

ax4 = fig.add_subplot(2,2,4)

sns.countplot(x='Perfect', data=cleaned_data, ax=ax1, color=base_color)

sns.countplot(x='College', data=cleaned_data, ax=ax2, color=base_color)

sns.countplot(x='White', data=cleaned_data, ax=ax3, color=base_color)

sns.countplot(x='English', data=cleaned_data, ax=ax4, color=base_color)

plt.tight_layout()

plt.show()
fig = plt.figure(figsize=(10,5))

ax1 = fig.add_subplot(1,2,1)

ax2 = fig.add_subplot(1,2,2)

sns.countplot(x='Rate', data=cleaned_data, ax=ax1, color=base_color)

sns.countplot(x='Recommend', data=cleaned_data, ax=ax2, color=base_color)

plt.tight_layout()

plt.show()
fig = plt.figure(figsize=(10,5))

ax1 = fig.add_subplot(1,2,1)

ax2 = fig.add_subplot(1,2,2)

sns.countplot(x='Health', data=cleaned_data, ax=ax1, color=base_color)

sns.countplot(x='Mental', data=cleaned_data, ax=ax2, color=base_color)

plt.tight_layout()

plt.show()
fig = plt.figure(figsize=(15,5))

ax1 = fig.add_subplot(1,3,1)

ax2 = fig.add_subplot(1,3,2)

ax3 = fig.add_subplot(1,3,3)

sns.countplot(x='Age', data=cleaned_data, ax=ax1, color=base_color)

sns.countplot(x='Sex', data=cleaned_data, ax=ax2, color=base_color)

sns.countplot(x='Home', data=cleaned_data, ax=ax3, color=base_color)

plt.tight_layout()

plt.show()
fig = plt.figure(figsize=(10,5))

ax1 = fig.add_subplot(1,2,1)

ax2 = fig.add_subplot(1,2,2)

sns.countplot(x='Source', data=cleaned_data, ax=ax1, color=base_color)

sns.countplot(x='Service', data=cleaned_data, ax=ax2, color=base_color)

plt.tight_layout()

plt.show()
fig = plt.figure(figsize=(10,5))

ax1 = fig.add_subplot(1,2,1)

ax2 = fig.add_subplot(1,2,2)

sns.countplot(x='Unit', data=cleaned_data, ax=ax1, color=base_color)

sns.countplot(x='Specialty', data=cleaned_data, ax=ax2, color=base_color)

plt.tight_layout()

plt.show()
fig = plt.figure(figsize=(10,5))

ax1 = fig.add_subplot(1,2,1)

ax2 = fig.add_subplot(1,2,2)

sns.countplot(x='Stay', data=cleaned_data, ax=ax1, color=base_color)

sns.countplot(x='Visit', data=cleaned_data, ax=ax2, color=base_color)

plt.tight_layout()

plt.show()
cleaned_data['Date'].hist(bins=100)

plt.xlabel('Date')

plt.ylabel('Count')

plt.show()
cleaned_data['Month'] = cleaned_data['Date'].apply(lambda x : x.strftime("%B"))
cleaned_data['Day_of_Week'] = cleaned_data['Date'].apply(lambda x: x.strftime("%A"))
m_cat = CategoricalDtype(['January', 'February', 'March', 'April', 'May', 'June', 'July', 

                          'August', 'September', 'October', 'November', 'December'], ordered=True)

d_w_cat = CategoricalDtype(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 

                            'Saturday', 'Sunday'], ordered=True)
cleaned_data['Month'] = cleaned_data['Month'].astype(m_cat)

cleaned_data['Day_of_Week'] = cleaned_data['Day_of_Week'].astype(d_w_cat)
cleaned_data['Month'].unique()
cleaned_data['Day_of_Week'].unique()
fig = plt.figure(figsize=(12,5))

ax1 = fig.add_subplot(1,2,1)

ax2 = fig.add_subplot(1,2,2)

ch1 = sns.countplot(x='Month', data=cleaned_data, ax=ax1, color=base_color)

ch1.set_xticklabels(labels = ch1.get_xticklabels(), rotation=45)

ch2 = sns.countplot(x='Day_of_Week', data=cleaned_data, ax=ax2, color=base_color)

ch2.set_xticklabels(labels = ch2.get_xticklabels(), rotation=45)

plt.tight_layout()

plt.show()
d_m = cleaned_data[cleaned_data['Sex']=='M']

sns.distplot(d_m['Composite'],kde=False, label='Male')

d_f = cleaned_data[cleaned_data['Sex']=='F']

sns.distplot(d_f['Composite'],kde=False, label='Female')

plt.legend(prop={'size': 12})

plt.title('Distrubtion of Composite per Gender')

plt.xlabel('Composite')

plt.ylabel('Count')

plt.show()
fig = plt.figure(figsize=(10,10))

ax1 = fig.add_subplot(2,2,1)

ax2 = fig.add_subplot(2,2,2)

ax3 = fig.add_subplot(2,2,3)

ax4 = fig.add_subplot(2,2,4)

sns.countplot(x='Perfect', data=cleaned_data, ax=ax1, hue='Sex')

sns.countplot(x='College', data=cleaned_data, ax=ax2, hue='Sex')

sns.countplot(x='White', data=cleaned_data, ax=ax3, hue='Sex')

sns.countplot(x='English', data=cleaned_data, ax=ax4, hue='Sex')

plt.tight_layout()

plt.show()
fig = plt.figure(figsize=(10,5))

ax1 = fig.add_subplot(1,2,1)

ax2 = fig.add_subplot(1,2,2)

sns.countplot(x='Rate', data=cleaned_data, ax=ax1, hue='Sex')

sns.countplot(x='Recommend', data=cleaned_data, ax=ax2, hue='Sex')

plt.tight_layout()

plt.show()
d_m = cleaned_data[cleaned_data['Visit']==0]

sns.distplot(d_m['Composite'],kde=False, label='No Visit')

d_f = cleaned_data[cleaned_data['Visit']==1]

sns.distplot(d_f['Composite'],kde=False, label='Visit')

plt.legend(prop={'size': 12})

plt.title('Distrubtion of Composite per Visit')

plt.xlabel('Composite')

plt.ylabel('Count')

plt.show()
fig = plt.figure(figsize=(10,10))

ax1 = fig.add_subplot(2,2,1)

ax2 = fig.add_subplot(2,2,2)

ax3 = fig.add_subplot(2,2,3)

ax4 = fig.add_subplot(2,2,4)

sns.countplot(x='Perfect', data=cleaned_data, ax=ax1, hue='Visit')

sns.countplot(x='College', data=cleaned_data, ax=ax2, hue='Visit')

sns.countplot(x='White', data=cleaned_data, ax=ax3, hue='Visit')

sns.countplot(x='English', data=cleaned_data, ax=ax4, hue='Visit')

plt.tight_layout()

plt.show()
fig = plt.figure(figsize=(10,5))

ax1 = fig.add_subplot(1,2,1)

ax2 = fig.add_subplot(1,2,2)

sns.countplot(x='Rate', data=cleaned_data, ax=ax1, hue='Visit')

sns.countplot(x='Recommend', data=cleaned_data, ax=ax2, hue='Visit')

plt.tight_layout()

plt.show()
fig = plt.figure(figsize=(15,5))

ax1 = fig.add_subplot(1,3,1)

ax2 = fig.add_subplot(1,3,2)

ax3 = fig.add_subplot(1,3,3)

sns.countplot(x='Age', data=cleaned_data, ax=ax1, hue='Visit')

sns.countplot(x='Sex', data=cleaned_data, ax=ax2, hue='Visit')

sns.countplot(x='Home', data=cleaned_data, ax=ax3, hue='Visit')

plt.tight_layout()

plt.show()
sns.catplot(x='Month', data=cleaned_data, kind='count', hue='Visit', aspect=2)

plt.show()
sns.catplot(x='Day_of_Week', kind='count', data=cleaned_data, hue='Visit', aspect=1.2)

plt.show()
sns.catplot(x='Day_of_Week', data=cleaned_data, y='Visit', kind='bar', aspect=1.2, ci=None, color=base_color)

plt.ylabel('Probability of Visit')

plt.show()
sns.catplot(x='Age', data=cleaned_data, y='Visit', kind='bar', aspect=1.2, ci=None, color=base_color)

plt.ylabel('Probability of Visit')

plt.show()
sns.catplot(x='Sex', data=cleaned_data, y='Visit', kind='bar', aspect=0.8, ci=None, color=base_color)

plt.ylabel('Probability of Visit')

plt.show()
import statsmodels.api as sm

female = cleaned_data[cleaned_data['Sex']=='F']

male = cleaned_data[cleaned_data['Sex']=='M']

counts = np.array([female['Visit'].sum(), 

                   male['Visit'].sum()])

nobs = np.array([female.shape[0], male.shape[0]])

zstat, pval = sm.stats.proportions_ztest(counts, nobs, alternative='smaller')

zstat, pval
sns.catplot(x='Mental', data=cleaned_data, y='Visit', kind='bar', aspect=1.2, ci=None, color=base_color)

plt.ylabel('Probability of Visit')

plt.show()
sns.catplot(x='College', data=cleaned_data, y='Visit', kind='bar', aspect=0.8, ci=None, color=base_color)

plt.ylabel('Probability of Visit')

plt.show()
collage = cleaned_data[cleaned_data['College']]

no_collage = cleaned_data[cleaned_data['College']==False]

counts = np.array([collage['Visit'].sum(), 

                   no_collage['Visit'].sum()])

nobs = np.array([collage.shape[0], no_collage.shape[0]])

zstat, pval = sm.stats.proportions_ztest(counts, nobs, alternative='smaller')

zstat, pval
sns.catplot(x='White', data=cleaned_data, y='Visit', kind='bar', aspect=0.8, ci=None, color=base_color)

plt.ylabel('Probability of Visit')

plt.show()
co = cleaned_data[cleaned_data['White']]

no = cleaned_data[cleaned_data['White']==False]

counts = np.array([co['Visit'].sum(), 

                   no['Visit'].sum()])

nobs = np.array([co.shape[0], no.shape[0]])

zstat, pval = sm.stats.proportions_ztest(counts, nobs, alternative='smaller')

zstat, pval
cleaned_data.corr()
fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot(1,1,1)

sns.heatmap(cleaned_data.corr(), ax=ax);
y = cleaned_data['Visit']

X = cleaned_data[[x for x in cleaned_data.columns if x not in ['Survey', 'Visit', 'Date']]]
X['White'] = X['White'].astype('int32')

X['College'] = X['College'].astype('int32')

X['English'] = X['English'].astype('int32')

X['Perfect'] = X['Perfect'].astype('int32')
X['Sex'] = X['Sex'].replace({'F':0, 'M':1})
X['Source'] = X['Source'].replace({'D':1,'T':0})
X = pd.get_dummies(X, columns=['Service', 'Home', 'Specialty', 'Unit'])
X.columns
X.drop(['Service_O', 'Home_Y', 'Specialty_1', 'Unit_3'], axis=1, inplace=True)
X['Month'] = X['Month'].replace({'January':0, 'February':1, 'March':2, 'April':3, 'May':4, 'June':5, 'July':6, 

                                 'August':7, 'September':8, 'October':9, 'November':10, 'December':11})

X['Day_of_Week'] = X['Day_of_Week'].replace({'Monday':0, 'Tuesday':1, 'Wednesday':2, 'Thursday':3, 'Friday':4, 

                                             'Saturday':5, 'Sunday':6})

X['Age'] = X['Age'].replace({'18-34':0, '35-49':1, '50-64':2, '65-79':3, '80-90':4, '90+':5})

X['Stay'] = X['Stay'].replace({'1':0, '2-3':1, '4-7':2, '8+':3})
X.dtypes
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, f1_score, precision_score

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.metrics import roc_auc_score, roc_curve, auc
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.7)
Performances=[]
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(random_state=0)

dtc.fit(X_train, y_train)
model={'Model': 'DecisionTree'}

model['Accuracy'] = accuracy_score(y_test, dtc.predict(X_test))

model['Precision'] = precision_score(y_test, dtc.predict(X_test))

model['F1-Score'] = f1_score(y_test, dtc.predict(X_test))

Performances.append(model)
confusion_matrix(y_test, dtc.predict(X_test))
print(classification_report(y_test, dtc.predict(X_test)))
dtc_features = pd.DataFrame()

dtc_features['Feature'] = X_train.columns.tolist()

dtc_features['Importance'] = dtc.feature_importances_

sns.catplot(y='Feature', x='Importance', data=dtc_features, kind='bar', height=15, color=base_color)

plt.show()
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100, random_state=0)

rfc.fit(X_train, y_train)
model={'Model': 'RandomForest'}

model['Accuracy'] = accuracy_score(y_test, rfc.predict(X_test))

model['Precision'] = precision_score(y_test, rfc.predict(X_test))

model['F1-Score'] = f1_score(y_test, rfc.predict(X_test))

Performances.append(model)
confusion_matrix(y_test, rfc.predict(X_test))
print(classification_report(y_test, rfc.predict(X_test)))
rfc_features = pd.DataFrame()

rfc_features['Feature'] = X_train.columns.tolist()

rfc_features['Importance'] = rfc.feature_importances_

sns.catplot(y='Feature', x='Importance', data=rfc_features, kind='bar', height=15, color=base_color)

plt.show()
from xgboost import XGBClassifier
xgb = XGBClassifier(n_estimators=100,random_state=0)

xgb.fit(X_train, y_train)
model={'Model': 'XGBoost'}

model['Accuracy'] = accuracy_score(y_test, xgb.predict(X_test))

model['Precision'] = precision_score(y_test, xgb.predict(X_test))

model['F1-Score'] = f1_score(y_test, xgb.predict(X_test))

Performances.append(model)
confusion_matrix(y_test, xgb.predict(X_test))
print(classification_report(y_test, xgb.predict(X_test)))
xgb_features = pd.DataFrame()

xgb_features['Feature'] = X_train.columns.tolist()

xgb_features['Importance'] = xgb.feature_importances_

sns.catplot(y='Feature', x='Importance', data=xgb_features, kind='bar', height=15, color=base_color)

plt.show()
from catboost import CatBoostClassifier
cb = CatBoostClassifier(n_estimators=2500, random_state=0)

cb.fit(X_train, y_train)
model={'Model': 'CatBoost'}

model['Accuracy'] = accuracy_score(y_test, cb.predict(X_test))

model['Precision'] = precision_score(y_test, cb.predict(X_test))

model['F1-Score'] = f1_score(y_test, cb.predict(X_test))

Performances.append(model)
confusion_matrix(y_test, cb.predict(X_test))
print(classification_report(y_test, cb.predict(X_test)))
cb_features = pd.DataFrame()

cb_features['Feature'] = X_train.columns.tolist()

cb_features['Importance'] = cb.feature_importances_

sns.catplot(y='Feature', x='Importance', data=cb_features, kind='bar', height=15, color=base_color)

plt.show()
pd.DataFrame(Performances)
fpr_RFC, tpr_RFC, _ = roc_curve(y_test, rfc.predict_proba(X_test)[:,1])

roc_auc_RFC = auc(fpr_RFC, tpr_RFC)

fpr_dtc, tpr_dtc, _ = roc_curve(y_test, dtc.predict_proba(X_test)[:,1])

roc_auc_dtc = auc(fpr_dtc, tpr_dtc)

fpr_XGB, tpr_XGB, _ = roc_curve(y_test, xgb.predict_proba(X_test)[:,1])

roc_auc_XGB = auc(fpr_XGB, tpr_XGB)

fpr_CAT, tpr_CAT, _ = roc_curve(y_test, cb.predict_proba(X_test)[:,1])

roc_auc_CAT = auc(fpr_CAT, tpr_CAT)
plt.figure(figsize=(8,8))

lw = 2

plt.plot(fpr_dtc, tpr_dtc, 

         lw=lw, label='ROC curve Decision Tree (area = %0.2f)' % roc_auc_dtc)

plt.plot(fpr_RFC, tpr_RFC, 

         lw=lw, label='ROC curve Random Forest (area = %0.2f)' % roc_auc_RFC)

plt.plot(fpr_XGB, tpr_XGB, 

         lw=lw, label='ROC curve XGBoost (area = %0.2f)' % roc_auc_XGB)

plt.plot(fpr_CAT, tpr_CAT, 

         lw=lw, label='ROC curve CATBoost (area = %0.2f)' % roc_auc_CAT)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Using All Features')

plt.legend(loc="lower right")

plt.show()