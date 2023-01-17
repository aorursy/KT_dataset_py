import numpy as np

import pandas as pd

import seaborn as sns

import random

import scipy.stats as stt

import warnings

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_curve,roc_auc_score

from sklearn.metrics import classification_report

from sklearn.model_selection import cross_val_score

warnings.filterwarnings('ignore')

%pylab inline
data = pd.read_csv('../input/heart.csv')
data.info()
print(data.shape)

data.head()
plt.figure(figsize=(20,7))

sns.set()

sns.countplot(data.age);
data[data.sex==1].age.hist(bins=20);

data[data.sex==0].age.hist(bins=20);
print('mens in sample -  {:.2%}'.format(data[data.sex==1].shape[0]/data.shape[0]))
sex_str = np.where(data.sex==1, 'male', 'female')

pd.crosstab(sex_str, data.target).plot(kind='barh');



print("Men have disease: ",data[(data.sex==1)&(data.target==1)].shape[0]/data[(data.sex==1)].shape[0])

print("Women have disease: ",data[(data.sex==0)&(data.target==1)].shape[0]/data[(data.sex==0)].shape[0])
fig = plt.figure(figsize=(15,7))

ax1 = fig.add_subplot(2,1,1)

ax1.set_title('have disease')

ax1.tick_params(labelbottom='off',axis='x')

sns.countplot(data[data.target==1].age);

ax2 = fig.add_subplot(2,1,2)

ax2.set_title("haven't disease")

sns.countplot(data[data.target==0].age);
pd.crosstab(data.cp, data.target).plot(kind='barh');

print('Human have pain and disease: {:.4}%'.format(data[(data.target==1) & (data.cp!=0)].shape[0]/data[data.target==1].shape[0]*100))
sns.jointplot(data.age, data.trestbps, kind='kde');
categ = pd.cut(data.age,[30,40,50,60,80])

dat_new = data[['target','trestbps']]

dat_new['age'] = categ

plt.figure(figsize=(7,7))

sns.boxplot(x='age',y='trestbps',hue='target',data=dat_new, width=0.7);
sns.boxplot(x=data.target, y=data.trestbps);
np.log(data.chol).hist();
sns.boxplot(x=data.target, y=data.chol);

fst = data[data.target==0].chol

scd = data[data.target==1].chol

stt.ttest_ind_from_stats(fst.mean(),fst.std(),fst.shape[0],\

                         scd.mean(),scd.std(),scd.shape[0],equal_var=True)
sns.pairplot(data[['age','trestbps','thalach','chol','target']],hue='target',size=2.5);
pd.crosstab(data.fbs,data.target).plot(kind='barh');
f, axes = plt.subplots(1, 5,figsize=(20,10))

sns.set(font_scale=2)

sns.boxplot(x=data.target, y=data.thalach, ax=axes[0],fliersize=4);

sns.boxplot(x=data.target, y=data.chol, ax=axes[1]);

sns.boxplot(x=data.target, y=data.age, ax=axes[2]);

sns.boxplot(x=data.target, y=data.trestbps, ax=axes[3]);

sns.boxplot(x=data.target, y=data.oldpeak, ax=axes[4]);

plt.tick_params(axis='both', which='major', labelsize=20)

plt.tight_layout()
f, axes = plt.subplots(1, 5,figsize=(40,10))

sns.set(font_scale=2)

pd.crosstab(data.slope,data.target).plot(kind='barh', ax=axes[0]);

pd.crosstab(data.exang,data.target).plot(kind='barh', ax=axes[1]);

pd.crosstab(data.thal,data.target).plot(kind='barh', ax=axes[2]);

pd.crosstab(data.restecg,data.target).plot(kind='barh', ax=axes[3]);

pd.crosstab(data.ca,data.target).plot(kind='barh', ax=axes[4]);

plt.tick_params(axis='both', which='major', labelsize=16)

plt.tight_layout()
f, axes = plt.subplots(1, 3,figsize=(40,10))

sns.set(font_scale=3)

pd.crosstab(data.slope,data.sex).plot(kind='barh', ax=axes[0]);

pd.crosstab(data.exang,data.sex).plot(kind='barh', ax=axes[1]);

pd.crosstab(data.thal,data.sex).plot(kind='barh', ax=axes[2]);

plt.tick_params(axis='both', which='major', labelsize=20)

plt.tight_layout()
sns.set(font_scale=1)

sns.countplot(data.target);
data_train, data_test, targ_train, targ_test = train_test_split(\

            data[['age','sex','cp','thalach','exang','oldpeak','slope','ca','thal']], data.target, test_size=0.2)
forest = RandomForestClassifier(criterion='gini', n_estimators=100, max_depth=4)

forest.fit(data_train, targ_train)
labels = forest.predict(data_test)

sns.set(font_scale=1.5)

auc = roc_auc_score(labels, targ_test)

fpr, tpr, thresholds = roc_curve(labels,targ_test)

print('ROC curve: AUC={0:0.2f}'.format(auc));
scores = cross_val_score(forest, data[['age','sex','cp','thalach','exang','oldpeak','slope','ca','thal']], data.target, cv=5)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(classification_report(labels,targ_test))
forest.feature_importances_
feat_imp = pd.DataFrame(forest.feature_importances_, index = ['age','sex','cp','thalach','exang','oldpeak','slope','ca','thal'])

plt.figure(figsize=(8,8));

sns.barplot(feat_imp[0],feat_imp.index);

plt.tick_params(axis='both', which='major', labelsize=16)

plt.xlabel('');