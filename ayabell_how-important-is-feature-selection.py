# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Data visualization

from matplotlib import pyplot as plt

import seaborn as sns



# Classification

from sklearn import model_selection

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier



from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2
df = pd.read_csv('../input/HR_comma_sep.csv')

df.head()
print(df.dtypes)
sns.heatmap(df.corr(),annot=True)

plt.show()
fig = plt.figure()

ax1 = fig.add_subplot(321)

ax2 = fig.add_subplot(322)

ax3 = fig.add_subplot(323)

ax4 = fig.add_subplot(324)

ax5 = fig.add_subplot(325)





sns.kdeplot(df[df.left==0].satisfaction_level,shade=True,color='blue',label='stayed',ax=ax1)

sns.kdeplot(df[df.left==1].satisfaction_level,shade=True,color='red',label='left',ax=ax1)

ax1.set(xlabel="satisfaction level")





sns.kdeplot(df[df.left==0].last_evaluation,shade=True,color='blue',label='stayed',ax=ax2)

sns.kdeplot(df[df.left==1].last_evaluation,shade=True,color='red',label='left',ax=ax2)

ax2.set(xlabel="last evaluation")



sns.kdeplot(df[df.left==0].average_montly_hours,shade=True,color='blue',label='stayed',ax=ax3)

sns.kdeplot(df[df.left==1].average_montly_hours,shade=True,color='red',label='left',ax=ax3)

ax3.set(xlabel="average monthly hours")



sns.kdeplot(df[df.left==0].time_spend_company,shade=True,color='blue',label='stayed',ax=ax4)

sns.kdeplot(df[df.left==1].time_spend_company,shade=True,color='red',label='left',ax=ax4)

ax4.set(xlabel="time spend company")



sns.kdeplot(df[df.left==0].number_project,shade=True,color='blue',label='stayed',ax=ax5)

sns.kdeplot(df[df.left==1].number_project,shade=True,color='red',label='left',ax=ax5)

ax5.set(xlabel="number_project")



plt.tight_layout()

plt.show()

print(df.salary.value_counts())

g = sns.FacetGrid(df,col='salary')

g = g.map(sns.barplot,'salary','left')

g.set_xticklabels(rotation=90)

g.add_legend()

plt.show()
print(df.Work_accident.value_counts())

g = sns.FacetGrid(df,col='Work_accident')

g = g.map(sns.barplot,'Work_accident','left')

g.set_xticklabels(rotation=90)

g.add_legend()

plt.show()
print(df.promotion_last_5years.value_counts())

g = sns.FacetGrid(df,col='promotion_last_5years')

g = g.map(sns.barplot,'promotion_last_5years','left')

g.set_xticklabels(rotation=90)

g.add_legend()

plt.show()
print(df.sales.value_counts())

g = sns.FacetGrid(df,col='sales',col_wrap=4)

g = g.map(sns.barplot,'sales','left')

g.set_xticklabels(rotation=90)

g.add_legend()

plt.show()
fig, ax = plt.subplots()

ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

ax.plot(df[df.left==1].last_evaluation, df[df.left==1].satisfaction_level, marker='o', linestyle='', ms=2.3, color='red',label='left')

ax.plot(df[df.left==0].last_evaluation, df[df.left==0].satisfaction_level, marker='o', linestyle='', ms=1.5, color='blue',label='stayed')

ax.legend()

plt.xlabel('last_evaluation')

plt.ylabel('satisfaction_level')

plt.tight_layout()

plt.show()
df['eval_cat'] = pd.cut(df.last_evaluation,bins=3,labels=['low eval','med eval','high eval'])

df['satisfaction_cat'] = pd.cut(df.satisfaction_level,bins=3,labels=['low satisfaction','med satisfaction','high satisfaction'])

#print(df.loc[df.satisfaction_cat=='med satisfaction','satisfaction_level'].describe())
g = sns.FacetGrid(df,col='eval_cat',row='satisfaction_cat',hue='left',sharex=False,sharey=False)

g = g.map(sns.kdeplot,'average_montly_hours',shade=True)

g.add_legend()

plt.show()
g = sns.FacetGrid(df,col='eval_cat',row='satisfaction_cat',hue='left',sharex=False,sharey=False)

g = g.map(sns.kdeplot,'time_spend_company',shade=True)

g.add_legend()

plt.show()
g = sns.FacetGrid(df,col='eval_cat',row='satisfaction_cat',hue='left',sharex=False,sharey=False)

g = g.map(sns.kdeplot,'number_project',shade=True)

g.add_legend()

plt.show()
g = sns.FacetGrid(df,col='eval_cat',row = 'satisfaction_cat')

g = g.map(sns.barplot,'salary','left')

g.set_xticklabels(rotation=90)

g.add_legend()

plt.show()
# Add dummy variables

df = pd.concat([df, pd.get_dummies(df['sales']).astype(int)], axis=1)

df = pd.concat([df, pd.get_dummies(df['salary']).astype(int)], axis=1)

df = pd.concat([df, pd.get_dummies(df['eval_cat']).astype(int)], axis=1)

df = pd.concat([df, pd.get_dummies(df['satisfaction_cat']).astype(int)], axis=1)

df.columns
validation_size = 0.20

seed = 7

scoring = 'accuracy'
# Select features



sub = df[['satisfaction_level', 'last_evaluation', 'number_project',

        'average_montly_hours', 'time_spend_company', 'Work_accident',

        'promotion_last_5years', 'IT', 'RandD',

        'accounting', 'hr', 'marketing', 'product_mng',

        'support', 'technical', 'high', 'low', 'medium']]



X = sub.values

Y = df['left'].values





test = SelectKBest(score_func=chi2, k=2)

fit = test.fit(X, Y)



temp = sub.columns

scores = pd.concat([pd.DataFrame(data=temp),pd.DataFrame(data=fit.scores_[:])],axis=1)

scores.columns = ['cat','score']

scores = scores.sort_values('score',ascending=False)



sns.barplot(x='score',y='cat',data=scores)

plt.show()



acc = []

for k in range(1,17):

    X = sub.values

    Y = df['left'].values



    test = SelectKBest(score_func=chi2, k=k)

    fit = test.fit(X, Y)

    X = fit.transform(X)

    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

    

    classifier = DecisionTreeClassifier()

    classifier.fit(X_train, Y_train)

    predictions = classifier.predict(X_validation)

    acc.append(accuracy_score(Y_validation, predictions))





plt.plot(acc, "o")

plt.show()
X = sub.values

Y = df['left'].values



test = SelectKBest(score_func=chi2, k=5)

fit = test.fit(X, Y)

X = fit.transform(X)

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

    

classifier = DecisionTreeClassifier()

classifier.fit(X_train, Y_train)

predictions = classifier.predict(X_validation)

print(accuracy_score(Y_validation, predictions))

print(confusion_matrix(Y_validation, predictions))

print(classification_report(Y_validation, predictions))
df['inter1'] = df['average_montly_hours']*df['low satisfaction']*df['low eval']

df['inter2'] = df['average_montly_hours']*df['med satisfaction']*df['low eval']

df['inter3'] = df['average_montly_hours']*df['high satisfaction']*df['high eval']



df['inter4'] = df['number_project']*df['low satisfaction']*df['med eval']

df['inter5'] = df['number_project']*df['low satisfaction']*df['high eval']
sub = df[['number_project', 'average_montly_hours', 'time_spend_company', 'Work_accident', 'left',

       'promotion_last_5years', 'IT', 'RandD', 'accounting', 'hr', 'management',

       'marketing', 'product_mng',  'support', 'technical', 'high',

       'low', 'medium', 'low eval', 'med eval', 'high eval',

       'low satisfaction', 'med satisfaction', 'high satisfaction', 'inter1',

       'inter2', 'inter3', 'inter4', 'inter5']]



X = sub.values

Y = df['left'].values





test = SelectKBest(score_func=chi2, k=2)

fit = test.fit(X, Y)



temp = sub.columns

scores = pd.concat([pd.DataFrame(data=temp),pd.DataFrame(data=fit.scores_[:])],axis=1)

scores.columns = ['cat','score']

scores = scores.sort_values('score',ascending=False)



sns.barplot(x='score',y='cat',data=scores)

plt.show()

acc = []

for k in range(1,29):

    X = sub.values

    Y = df['left'].values



    test = SelectKBest(score_func=chi2, k=k)

    fit = test.fit(X, Y)

    X = fit.transform(X)

    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

    

    classifier = DecisionTreeClassifier()

    classifier.fit(X_train, Y_train)

    predictions = classifier.predict(X_validation)

    acc.append(accuracy_score(Y_validation, predictions))





plt.plot(acc, "o")

plt.show()
X = sub.values

Y = df['left'].values



test = SelectKBest(score_func=chi2, k=5)

fit = test.fit(X, Y)

X = fit.transform(X)

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

    

classifier = DecisionTreeClassifier()

classifier.fit(X_train, Y_train)

predictions = classifier.predict(X_validation)

print(accuracy_score(Y_validation, predictions))

print(confusion_matrix(Y_validation, predictions))

print(classification_report(Y_validation, predictions))