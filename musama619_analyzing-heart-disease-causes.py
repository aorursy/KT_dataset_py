import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
data = pd.read_csv('../input/heart-disease-uci/heart.csv')
data.head()
data.shape
data.isnull().sum()
data.dtypes
data['sex'][data['sex'] == 0] = 'female'
data['sex'][data['sex'] == 1] = 'male'

data['cp'][data['cp'] == 1] = 'typical angina'
data['cp'][data['cp'] == 2] = 'atypical angina'
data['cp'][data['cp'] == 3] = 'non-anginal pain'
data['cp'][data['cp'] == 4] = 'asymptomatic'

data['fbs'][data['fbs'] == 0] = 'lower than 120mg/ml'
data['fbs'][data['fbs'] == 1] = 'greater than 120mg/ml'

data['restecg'][data['restecg'] == 0] = 'normal'
data['restecg'][data['restecg'] == 1] = 'ST-T wave abnormality'
data['restecg'][data['restecg'] == 2] = 'left ventricular hypertrophy'

data['exang'][data['exang'] == 0] = 'no'
data['exang'][data['exang'] == 1] = 'yes'

data['slope'][data['slope'] == 1] = 'upsloping'
data['slope'][data['slope'] == 2] = 'flat'
data['slope'][data['slope'] == 3] = 'downsloping'

data['thal'][data['thal'] == 1] = 'normal'
data['thal'][data['thal'] == 2] = 'fixed defect'
data['thal'][data['thal'] == 3] = 'reversable defect'
data['sex'] = data['sex'].astype('object')
data['cp'] = data['cp'].astype('object')
data['fbs'] = data['fbs'].astype('object')
data['restecg'] = data['restecg'].astype('object')
data['exang'] = data['exang'].astype('object')
data['slope'] = data['slope'].astype('object')
data['thal'] = data['thal'].astype('object')
data.head()
data.dtypes
sns.countplot('target', data=data)
sns.countplot('target', data=data, hue='sex', palette="Set1")
data[['target', 'sex']].groupby(['sex'], as_index=False).mean().sort_values(by='sex', ascending=False)
data['age'].hist()
sns.distplot(data['age'], color = 'red')
plt.figure(figsize=(20,10))
sns.countplot('age', hue='target', data=data)
sns.swarmplot('target', 'age', data=data)
sns.swarmplot('target', 'chol', data=data)
sns.countplot('target', hue='ca', data=data)
data.groupby(['target','ca']).size().unstack().plot(kind='bar', stacked=True, figsize=(10,8))
plt.show()
sns.countplot('target', hue='thal', data=data)
plt.figure( figsize=(20,8))
plt.scatter(x = data['target'], y = data['chol'], s = data['thalach']*100, color = 'red')
label = data['target']
label.unique()
label.value_counts()
data=data.drop(['target'], axis=1)
data.head()
label.shape
data = pd.get_dummies(data, drop_first=True)
data.head()
x = data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, label, test_size = 0.2)
x_train.shape
x_test.shape
y_train.shape
y_test.shape
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
mod1 = RandomForestClassifier()
mod1.fit(x_train, y_train)
mod2 = DecisionTreeClassifier()
mod2.fit(x_train, y_train)
pred_1 = mod1.predict(x_test)
pred_quant1 = mod1.predict_proba(x_test)[:, 1]
pred1 = mod1.predict(x_test)

pred_2 = mod2.predict(x_test)
pred_quant2 = mod2.predict_proba(x_test)[:, 1]
pred2 = mod2.predict(x_test)

score1_train=mod1.score(x_train, y_train)
print(f'Training Random Forest: {round(score1_train*100,2)}%')

score1_test=mod1.score(x_test,y_test)
print(f'Testing Random Forest: {round(score1_test*100,2)}%')
score2_train=mod2.score(x_train, y_train)
print(f'Training Decision Tree: {round(score2_train*100,2)}%')

score2_test=mod2.score(x_test,y_test)
print(f'Testing Decision Tree: {round(score2_test*100,2)}%')
from sklearn.metrics  import confusion_matrix
confusion_matrix(y_test, pred1)
sns.heatmap(confusion_matrix(y_test, pred1), annot=True)
confusion_matrix(y_test, pred2)
sns.heatmap(confusion_matrix(y_test, pred2), annot=True)
y_pred_quant1 = mod1.predict_proba(x_test)[:, 1]
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred_quant1)

fig, ax = plt.subplots()
ax.plot(fpr, tpr)
ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="-", c=".3")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])

plt.rcParams['figure.figsize'] = (15, 5)
plt.title('ROC curve for diabetes classifier', fontweight = 30)
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.show()
import eli5 
from eli5.sklearn import PermutationImportance
perm1 = PermutationImportance(mod1, random_state = 0).fit(x_test, y_test)
eli5.show_weights(perm1, feature_names = x_test.columns.tolist())
perm2 = PermutationImportance(mod2, random_state = 0).fit(x_test, y_test)
eli5.show_weights(perm2, feature_names = x_test.columns.tolist())
