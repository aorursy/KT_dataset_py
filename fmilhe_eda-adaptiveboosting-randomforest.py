import pandas as pd

import numpy as np



import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="whitegrid", color_codes=True)

%matplotlib inline
df = pd.read_csv('../input/HR_comma_sep.csv', header=0)
df.info()
left = df['left']

df = df.drop('left', axis = 1)

df = pd.concat([left, df], axis=1)

df.describe()
df.describe(include=['O'])
sns.countplot(x="left", data=df)

plt.show()
df_left_people = df.loc[df['left']==1, :]

df_left_people
#correlation matrix

k = 7 #number of variables for heatmap

corrmat = df.corr()

cols = corrmat.nlargest(k, 'left')['left'].index

cm = np.corrcoef(df[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
df[['left', 'satisfaction_level']].groupby(['left'], as_index=False).mean().sort_values(by='satisfaction_level',ascending=False)
sns.barplot(x="salary", y="left", data=df);

#Salary definetely has an impact on the decision of the employee to left his company
sns.violinplot(x="left", y="average_montly_hours", data=df)

#It seems that those who left are those who worked the more and the less
#Let's see if there is a correlation between the worked time and the salary

f, ax = plt.subplots(figsize=(15, 12))

sns.boxplot(x='salary', y='average_montly_hours', hue='left', data=df)
#In average, those who stay work around 200hours whereas those who left and were low-medium salary worked more

#and the high salary who left worked less 
sns.boxplot(x='left', y='time_spend_company', data=df)
#Time spend in the company has an impact on the decision of the employee to leave its company
sns.countplot(x='promotion_last_5years', hue='left', data=df)
#Too few promotions to take it in account, will drop fro df

df = df.drop('promotion_last_5years', axis=1)
f, ax = plt.subplots(figsize=(15, 12))

sns.barplot(x='sales', y='left', data=df)

sns.countplot(x='sales', hue='salary', data=df)
#Job category salary seems to be highly correlated to demissions
map_salaries = {'low':1, 'medium':2, 'high':3}

df['salary'] = df['salary'].map(map_salaries)
f, ax = plt.subplots(figsize=(15, 12))

sns.pointplot(x="sales", y="salary", hue="left", data=df);
#This confirm that the salary is linked to the kind of job, it was obvious
sales_dummies = pd.get_dummies(df['sales'])

df = df.drop('sales', axis=1)

df = pd.concat([df, sales_dummies], axis=1)
df.info()
from sklearn.cross_validation import train_test_split



X = df.iloc[:, 1:].values

y = df.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix



tree = DecisionTreeClassifier(max_depth=5, random_state=0)

tree.fit(X_train, y_train)



y_train_pred = tree.predict(X_train)



confmat = confusion_matrix(y_true=y_train, y_pred=y_train_pred)

f, ax = plt.subplots(figsize=(2.5, 2.5))

ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)

for i in range(confmat.shape[0]):

    for j in range(confmat.shape[1]):

        ax.text(x=j, y=i,

               s=confmat[i, j],

               va='center', ha='center')

plt.xlabel('Predicted label')

plt.ylabel('True label')

plt.show()
from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score



print ('Precision Decision tree : %.3f' %precision_score(y_true=y_train, y_pred=y_train_pred))

print ('Recall Decision tree : %.3f' %recall_score(y_true=y_train, y_pred=y_train_pred))

print ('F1 Score Decision tree : %.3f' %f1_score(y_true=y_train, y_pred=y_train_pred))

from sklearn.ensemble import RandomForestClassifier



forest = RandomForestClassifier(n_estimators=100, random_state=0)

forest.fit(X_train, y_train)



y_train_pred = forest.predict(X_train)



confmat = confusion_matrix(y_true=y_train, y_pred=y_train_pred)

f, ax = plt.subplots(figsize=(2.5, 2.5))

ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)

for i in range(confmat.shape[0]):

    for j in range(confmat.shape[1]):

        ax.text(x=j, y=i,

               s=confmat[i, j],

               va='center', ha='center')

plt.xlabel('Predicted label')

plt.ylabel('True label')

plt.show()

from sklearn.ensemble import AdaBoostClassifier



ada = AdaBoostClassifier(base_estimator=tree,

                        n_estimators=500,

                        learning_rate=0.1,

                        random_state=0)



ada.fit(X_train, y_train)



y_train_pred = ada.predict(X_train)



confmat = confusion_matrix(y_true=y_train, y_pred=y_train_pred)

f, ax = plt.subplots(figsize=(2.5, 2.5))

ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)

for i in range(confmat.shape[0]):

    for j in range(confmat.shape[1]):

        ax.text(x=j, y=i,

               s=confmat[i, j],

               va='center', ha='center')

plt.xlabel('Predicted label')

plt.ylabel('True label')

plt.show()



from sklearn.metrics import roc_auc_score



y_train_pred = ada.predict(X_train)

print('ROC AUC ADAPTIVE BOOSTING %.3f '%roc_auc_score(y_true=y_train, y_score=y_train_pred))

y_train_pred = forest.predict(X_train)

print('ROC AUC RANDOM FOREST %.3f '%roc_auc_score(y_true=y_train, y_score=y_train_pred))

y_train_pred = tree.predict(X_train)

print('ROC AUC TREE %.3f '%roc_auc_score(y_true=y_train, y_score=y_train_pred))
y_test_pred = ada.predict(X_test)

print('ROC AUC ADAPTIVE BOOSTING %.3f '%roc_auc_score(y_true=y_test, y_score=y_test_pred))

y_test_pred = forest.predict(X_test)

print('ROC AUC RANDOM FOREST %.3f '%roc_auc_score(y_true=y_test, y_score=y_test_pred))

y_test_pred = tree.predict(X_test)

print('ROC AUC TREE %.3f '%roc_auc_score(y_true=y_test, y_score=y_test_pred))