import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import os



from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report



from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

print(os.listdir('../input'))
df = pd.read_csv('../input/heart.csv')
df.head()
df.info()
df.describe()
plt.figure(figsize=(15,7))



corr = df.corr()

sns.heatmap(corr, annot=True )
sns.set_style('whitegrid')



hd = df[df['target'] == 1]['age']

no_hd = df[df['target'] == 0]['age']



sns.distplot(hd, kde=False, label='Heart Disease')

sns.distplot(no_hd, kde=False, label='None')



plt.title('Age: Heart Disease VS None')

plt.legend()
g = sns.countplot(x='sex', hue='target', data=df)



g.set(xticklabels=['Female', 'Male'])



plt.legend(['None', 'Heart Disease'])

plt.title('Sex: Heart Disease VS None')
g = sns.violinplot(x='exang', y='slope', data=df)

g.set(xticklabels=['No', 'Yes'])

plt.show()
g = sns.countplot(x='fbs', hue='target', data=df)

plt.title('Heart Disease: Low FBS vs High FBS')

plt.legend(labels=['None', 'Heart Disease'])

g.set(xticklabels=['< 120', '> 120'])

plt.show()
plt.figure(figsize=(16, 4))



sns.lineplot(x='age', y='thalach', hue='target', data=df)
plt.figure(figsize=(16, 4))



sns.lineplot(x='age', y='trestbps', hue='target', data=df)
plt.figure(figsize=(16, 4))



sns.lineplot(x='age', y='chol', hue='target', data=df)
plt.figure(figsize=(16, 4))





g = sns.barplot(x='cp', y='thalach', hue='target', data=df)



g.set(xticklabels=['typical angina', 'atypical angina', 'non-anginal pain', 'asymptomatic'])



plt.show()
sns.barplot(x='exang', y='thalach', hue='target', data=df)
plt.figure(figsize=(16, 4))



sns.lineplot(x='oldpeak', y='thalach', hue='target', data=df)
g = sns.countplot(x='slope', hue='target',data=df)





g.set(xticklabels=['Up', 'Flat', 'Down',])

sns.barplot(x='target', y='thalach', data=df)
plt.figure(figsize=(16, 4))



g = sns.countplot(x='cp', hue='exang', data=df)



g.set(xticklabels=['typical angina', 'atypical angina', 'non-anginal pain', 'asymptomatic'])



plt.show()
plt.figure(figsize=(16, 4))



g = sns.countplot(x='cp', hue='target', data=df)



g.set(xticklabels=['typical angina', 'atypical angina', 'non-anginal pain', 'asymptomatic'])



plt.show()
sns.boxplot(x='exang', y='oldpeak', hue='target', data=df)
g = sns.countplot(x='slope', hue='exang',data=df)





g.set(xticklabels=['Up', 'Flat', 'Down',])

sns.countplot(x='thal', hue='target', data=df)
sns.countplot(x='ca', hue='target', data=df)
sns.violinplot(x='ca', y='age', data=df)
cp = pd.get_dummies(df['cp'], prefix='cp')

restecg = pd.get_dummies(df['restecg'], prefix='restecg')

slope = pd.get_dummies(df['slope'], prefix='slope')
df = df.drop(['cp','restecg', 'slope'], axis=1)

df = pd.concat([df, cp, restecg, slope], axis=1)
df.columns
X_train, X_test, y_train, y_test = train_test_split(df.drop('target',axis=1), 

                                                    df['target'], test_size=0.30, 

                                                    random_state=101)

lr = LogisticRegression()

lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

rf = RandomForestClassifier(n_estimators=100)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
kn = KNeighborsClassifier()

error_rate = []



for i in range(1,30):

    

    kn = KNeighborsClassifier(n_neighbors=i)

    kn.fit(X_train,y_train)

    pred_i = kn.predict(X_test)

    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10,6))

plt.plot(range(0, len(error_rate)),error_rate,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='red', markersize=10)

plt.title('Error Rate vs. K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')
kn = KNeighborsClassifier(n_neighbors=23)

kn.fit(X_test, y_test)

kn_pred = kn.predict(X_test)
dt = DecisionTreeClassifier()

dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

sv = SVC()

sv.fit(X_train, y_train)

sv_pred = sv.predict(X_test)
param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']} 
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)
grid.fit(X_train, y_train)
grid.best_params_
grid.best_estimator_
grid_pred = grid.predict(X_test)
models = {'Logistic Regression': lr_pred, 'Random Forest': rf_pred, 'K-Nearest': kn_pred, 'Decision Tree': dt_pred, 'Support vector classifier': grid_pred}
for pred in models:

    print('\n', '-'*50,'\n',pred, '\n',  '-'*50,'\n')

    print(classification_report(y_test, models[pred]))
models_avg = {}



for model in models:

    models_avg[model] = np.mean(models[model] == y_test)

    

models_avg = pd.Series(models_avg)
plt.figure(figsize=(8, 5))



models_avg.plot(kind='bar', yticks=np.arange(0, 1, .05))