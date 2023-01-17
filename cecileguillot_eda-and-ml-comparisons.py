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

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session/'



# list of librairies used in this project

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats as st 



from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

import sklearn.metrics as metrics

from mlxtend.plotting import plot_confusion_matrix

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import AdaBoostClassifier
# Colors and style settings

sns.set_palette("Set1")

sns.set_style('ticks')
dataset = pd.read_csv('../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')
dataset.head()
print(dataset.shape)

print(dataset.columns)
s = (dataset.dtypes == 'object')

object_cols = list(s[s].index)

print('Categorial columns :', object_cols)
dataset.isnull().sum()
dataset.duplicated().sum()
plt.figure(figsize = (40,40))



_ = plt.subplot(3, 3, 1)

_ = sns.boxplot('age', data=dataset)



_ = plt.subplot(3,3,2)

_ = sns.boxplot('creatinine_phosphokinase', data=dataset)



_ = plt.subplot(3,3,3)

_ = sns.boxplot('ejection_fraction', data=dataset)



_ = plt.subplot(3,3,4)

_ = sns.boxplot('platelets', data=dataset)



_ = plt.subplot(3,3,5)

_ = sns.boxplot('serum_creatinine', data=dataset)



_ = plt.subplot(3,3,6)

_ = sns.boxplot('serum_sodium', data=dataset)



_ = plt.subplot(3,3,7)

_ = sns.boxplot('time', data=dataset)



_ = plt.tight_layout()



_ = plt.show()
outliers = dataset['ejection_fraction'] >= 70

dataset[outliers]
dataset = dataset[~outliers]
plt.figure(figsize = (40,40))



_ = plt.subplot(3, 3, 1)

_ = sns.boxplot('age', data=dataset)



_ = plt.subplot(3,3,2)

_ = sns.boxplot('creatinine_phosphokinase', data=dataset)



_ = plt.subplot(3,3,3)

_ = sns.boxplot('ejection_fraction', data=dataset)



_ = plt.subplot(3,3,4)

_ = sns.boxplot('platelets', data=dataset)



_ = plt.subplot(3,3,5)

_ = sns.boxplot('serum_creatinine', data=dataset)



_ = plt.subplot(3,3,6)

_ = sns.boxplot('serum_sodium', data=dataset)



_ = plt.subplot(3,3,7)

_ = sns.boxplot('time', data=dataset)



_ = plt.tight_layout()



_ = plt.show()
plt.figure(figsize = (15,15))



_ = plt.subplot(2,3,1)

_ = sns.countplot('sex', hue='DEATH_EVENT', data=dataset)

_ = plt.legend()



_ = plt.subplot(2,3,2)

_ = sns.countplot('anaemia', hue='DEATH_EVENT', data=dataset)

_ = plt.legend()



_ = plt.subplot(2,3,3)

_ = sns.countplot('high_blood_pressure', hue='DEATH_EVENT', data=dataset)

_ = plt.legend()



_ = plt.subplot(2,3,4)

_ = sns.countplot('smoking', hue='DEATH_EVENT', data=dataset)

_ = plt.legend()



_ = plt.subplot(2,3,5)

_ = sns.countplot('diabetes', hue='DEATH_EVENT', data=dataset)

_ = plt.legend()



_ = plt.tight_layout()



_ = plt.show()
X = 'sex'

Y = 'DEATH_EVENT'

cont = dataset[[X, Y]].pivot_table(index=X, columns=Y, aggfunc=len, margins=True, margins_name='Total')
tx = cont.loc[:,["Total"]]

ty = cont.loc[["Total"],:]

n = len(dataset)

indep = tx.dot(ty) / n



c = cont.fillna(0)

measure = (c-indep)**2/indep

xi_n = measure.sum().sum()

table = measure/xi_n



_ = plt.figure(figsize = (10,10))

_ = sns.heatmap(table.iloc[:-1,:-1],annot=c.iloc[:-1,:-1])

_ = plt.ylabel('sex')

_ = plt.xlabel('DEATH_EVENT')

_ = plt.yticks(rotation = 360)

_ = plt.title('Contingency table')



_ = plt.show()
print('xi_n : ', xi_n)



st_chi2, st_p, st_dof, st_exp = st.chi2_contingency(cont)

print('chi-squared :', st_chi2)

print('p-value :', st_p)
X = 'smoking'

Y = 'DEATH_EVENT'

cont = dataset[[X, Y]].pivot_table(index=X, columns=Y, aggfunc=len, margins=True, margins_name='Total')
tx = cont.loc[:,["Total"]]

ty = cont.loc[["Total"],:]

n = len(dataset)

indep = tx.dot(ty) / n



c = cont.fillna(0)

measure = (c-indep)**2/indep

xi_n = measure.sum().sum()

table = measure/xi_n



_ = plt.figure(figsize = (10,10))

_ = sns.heatmap(table.iloc[:-1,:-1],annot=c.iloc[:-1,:-1])

_ = plt.ylabel('smoking')

_ = plt.xlabel('DEATH_EVENT')

_ = plt.yticks(rotation = 360)

_ = plt.title('Contingency table')



_ = plt.show()
print('xi_n : ', xi_n)



st_chi2, st_p, st_dof, st_exp = st.chi2_contingency(cont)

print('chi-squared :', st_chi2)

print('p-value :', st_p)
X = 'anaemia'

Y = 'DEATH_EVENT'

cont = dataset[[X, Y]].pivot_table(index=X, columns=Y, aggfunc=len, margins=True, margins_name='Total')
tx = cont.loc[:,["Total"]]

ty = cont.loc[["Total"],:]

n = len(dataset)

indep = tx.dot(ty) / n



c = cont.fillna(0)

measure = (c-indep)**2/indep

xi_n = measure.sum().sum()

table = measure/xi_n



_ = plt.figure(figsize = (10,10))

_ = sns.heatmap(table.iloc[:-1,:-1],annot=c.iloc[:-1,:-1])

_ = plt.ylabel('anaemia')

_ = plt.xlabel('DEATH_EVENT')

_ = plt.yticks(rotation = 360)

_ = plt.title('Contingency table')



_ = plt.show()
print('xi_n : ', xi_n)



st_chi2, st_p, st_dof, st_exp = st.chi2_contingency(cont)

print('chi-squared :', st_chi2)

print('p-value :', st_p)
X = 'high_blood_pressure'

Y = 'DEATH_EVENT'

cont = dataset[[X, Y]].pivot_table(index=X, columns=Y, aggfunc=len, margins=True, margins_name='Total')
tx = cont.loc[:,["Total"]]

ty = cont.loc[["Total"],:]

n = len(dataset)

indep = tx.dot(ty) / n



c = cont.fillna(0)

measure = (c-indep)**2/indep

xi_n = measure.sum().sum()

table = measure/xi_n



_ = plt.figure(figsize = (10,10))

_ = sns.heatmap(table.iloc[:-1,:-1],annot=c.iloc[:-1,:-1])

_ = plt.ylabel('high_blood_pressure')

_ = plt.xlabel('DEATH_EVENT')

_ = plt.yticks(rotation = 360)

_ = plt.title('Contingency table')



_ = plt.show()
print('xi_n : ', xi_n)



st_chi2, st_p, st_dof, st_exp = st.chi2_contingency(cont)

print('chi-squared :', st_chi2)

print('p-value :', st_p)
plt.figure(figsize=[10,10])



_ = sns.boxplot(x='DEATH_EVENT', y='creatinine_phosphokinase', data=dataset)

_ = plt.show()
X = dataset['DEATH_EVENT'] # qualitative

Y = dataset['creatinine_phosphokinase']



def eta_squared(x,y):

    moyenne_y = y.mean()

    classes = []

    for classe in x.unique():

        yi_classe = y[x==classe]

        classes.append({'ni': len(yi_classe),

                        'moyenne_classe': yi_classe.mean()})

    SCT = sum([(yj-moyenne_y)**2 for yj in y])

    SCE = sum([c['ni']*(c['moyenne_classe']-moyenne_y)**2 for c in classes])

    return SCE/SCT

    

print('eta-squared :', eta_squared(X,Y))
plt.figure(figsize=[10,10])



_ = sns.boxplot(x='DEATH_EVENT', y='platelets', data=dataset)

_ = plt.show()
X = dataset['DEATH_EVENT'] # qualitative

Y = dataset['platelets']



print('eta-squared :', eta_squared(X,Y))
plt.figure(figsize=[10,10])



_ = sns.boxplot(x='DEATH_EVENT', y='ejection_fraction', data=dataset)

_ = plt.show()
X = dataset['DEATH_EVENT'] # qualitative

Y = dataset['ejection_fraction']



print('eta-squared :', eta_squared(X,Y))
plt.figure(figsize=[10,10])



_ = sns.boxplot(x='DEATH_EVENT', y='serum_creatinine', data=dataset)

_ = plt.show()
X = dataset['DEATH_EVENT'] # qualitative

Y = dataset['serum_creatinine']



print('eta-squared :', eta_squared(X,Y))
plt.figure(figsize=[10,10])



_ = sns.boxplot(x='DEATH_EVENT', y='serum_sodium', data=dataset)

_ = plt.show()
X = dataset['DEATH_EVENT'] # qualitative

Y = dataset['serum_sodium']



print('eta-squared :', eta_squared(X,Y))
plt.figure(figsize=[10,10])



_ = sns.boxplot(x='DEATH_EVENT', y='time', data=dataset)

_ = plt.show()
X = dataset['DEATH_EVENT'] # qualitative

Y = dataset['time']



print('eta-squared :', eta_squared(X,Y))
corrMatrix = dataset.corr()

plt.figure(figsize=(15,15))

_ = sns.heatmap(corrMatrix,square = True, cmap="coolwarm",linewidths=.5, annot=True)

_ = plt.show()
X = dataset[['age','ejection_fraction','serum_creatinine', 'serum_sodium', 'time']]

y = dataset['DEATH_EVENT']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size = 0.8, random_state=1)
rf = RandomForestClassifier(max_depth=2, random_state=42)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)



print('Accuracy : ', metrics.accuracy_score(y_pred=y_pred,y_true=y_test) * 100, '%')
cm = metrics.confusion_matrix(y_test, y_pred)

plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Reds)

_ = plt.title("Random Forest Model - Confusion Matrix")

_ = plt.xticks(range(2), ["Heart Not Failed","Heart Failed"], fontsize=16)

_ = plt.yticks(range(2), ["Heart Not Failed","Heart Failed"], fontsize=16)

_ = plt.show()
importances = rf.feature_importances_

indices = np.argsort(importances)

 

plt.figure(1)

plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), [dataset.columns[i] for i in indices])

plt.xlabel('Relative Importance')
gbc = GradientBoostingClassifier(max_depth=2, random_state=42)

gbc.fit(X_train, y_train)

y_pred = gbc.predict(X_test)



print('Accuracy : ', metrics.accuracy_score(y_pred=y_pred,y_true=y_test) * 100, '%')
cm = metrics.confusion_matrix(y_test, y_pred)

plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Reds)

plt.title("Gradient Boosting Model - Confusion Matrix")

plt.xticks(range(2), ["Heart Not Failed","Heart Failed"], fontsize=16)

plt.yticks(range(2), ["Heart Not Failed","Heart Failed"], fontsize=16)

plt.show()
importances = gbc.feature_importances_

indices = np.argsort(importances)

 

plt.figure(1)

plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), [dataset.columns[i] for i in indices])

plt.xlabel('Relative Importance')
adc = AdaBoostClassifier(n_estimators=100, random_state=42)

adc.fit(X_train, y_train)

y_pred = adc.predict(X_test)



print('Accuracy: ', metrics.accuracy_score(y_pred=y_pred,y_true=y_test) * 100, '%')
cm = metrics.confusion_matrix(y_test, y_pred)

plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Reds)

plt.title("AdaBoost Model - Confusion Matrix")

plt.xticks(range(2), ["Heart Not Failed","Heart Failed"], fontsize=16)

plt.yticks(range(2), ["Heart Not Failed","Heart Failed"], fontsize=16)

plt.show()
importances = adc.feature_importances_

indices = np.argsort(importances)

 

plt.figure(1)

plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), [dataset.columns[i] for i in indices])

plt.xlabel('Relative Importance')