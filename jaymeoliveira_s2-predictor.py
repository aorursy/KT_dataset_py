import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
patients = pd.read_csv('/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')
bin_data = ['anaemia', 'diabetes', 'high_blood_pressure','sex','smoking']
con_data = ['age', 'creatinine_phosphokinase', 'ejection_fraction','platelets','serum_creatinine','serum_sodium', 'time']

df_bin = patients[bin_data]
df_con = patients[con_data]
target = patients['DEATH_EVENT']
patients.info()
patients.describe()
import matplotlib.pyplot as plt
import seaborn as sns

sns.distplot(patients.age)
plt.show()
f = plt.figure(figsize=(12, 8))
gs = f.add_gridspec(2, 2)

ax = f.add_subplot(gs[0, 0])
sns.countplot(data=patients, x='sex', hue='DEATH_EVENT')
ax = f.add_subplot(gs[0, 1])
sns.countplot(data=patients, x='smoking', hue='DEATH_EVENT')
ax = f.add_subplot(gs[1, 0])
sns.countplot(data=patients, x='anaemia', hue='DEATH_EVENT')
ax = f.add_subplot(gs[1, 1])
sns.countplot(data=patients, x='diabetes', hue='DEATH_EVENT')

f.tight_layout()
plt.figure(figsize=(16, 12))
sns.heatmap(patients.corr(), annot=True, fmt='.2f', vmax=0.8, vmin=-0.8)
plt.show()
plt.figure(figsize=(12, 8))
g = sns.FacetGrid(patients, col="smoking", row="sex")
g.map(sns.distplot, "age")
g.add_legend()
plt.show()
patients[(patients.sex==0) & (patients.smoking==1)]
man = patients[patients.sex == 1].drop(columns=['sex'])
plt.figure(figsize=(16, 12))
sns.heatmap(man.corr(), annot=True, fmt='.2f', vmax=0.8, vmin=-0.8)
plt.show()
sns.distplot(patients.time)
plt.show()
from sklearn.preprocessing import StandardScaler

padronizador = StandardScaler()
padronizador.fit(df_con)
dados2 = padronizador.transform(df_con)
dados2 = pd.DataFrame(data = dados2, columns=df_con.columns.values)

dados_plot = pd.concat([target, dados2], axis=1)
dados_plot = pd.melt(dados_plot, id_vars='DEATH_EVENT', var_name='Caracteristicas', value_name='valores')
plt.figure(figsize=(10, 10))
sns.violinplot(x = "Caracteristicas", y = "valores", hue = "DEATH_EVENT",
               data = dados_plot, split= True)
plt.xticks(rotation = 90)

from sklearn.model_selection import train_test_split

patients.drop(columns=['DEATH_EVENT','smoking','sex','platelets'], inplace=True, errors='ignore')

X_train, X_test, Y_train, Y_test = train_test_split(patients, target, test_size=0.2)
from numpy import mean
from numpy import std

from sklearn.pipeline import Pipeline

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV

seed = 42

models = {
    'LR':make_pipeline(StandardScaler(),LogisticRegression(random_state=seed)),
    'SVC':make_pipeline(StandardScaler(),SVC(random_state=seed)),
    'KNN':KNeighborsClassifier(),
    'DT':DecisionTreeClassifier(random_state=seed),
    'AB':AdaBoostClassifier(random_state=seed),
    'ET':ExtraTreesClassifier(random_state=seed),
    'GB':GradientBoostingClassifier(random_state=seed),
    'RF':RandomForestClassifier(random_state=seed),
    'XGB':XGBClassifier(random_state=seed),
    'LGBM':LGBMClassifier(random_state=seed)
    }


def evaluate_model(model):
    cv = StratifiedKFold(shuffle=True, random_state=seed)
    #pipeline = make_pipeline(Tweet2Vec(), model)
    scores = cross_val_score(model, X_train, Y_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    return scores

#pipeline = make_pipeline(Tweet2Vec(), models['ET'])

#pipe = pipeline.fit(X_train,y_train)
#pipe.score(X_test, y_test)

results, names = list(), list()
for name, model in models.items():
    scores = evaluate_model(model)
    results.append(scores)
    names.append(name)
    print('*%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
import matplotlib.pyplot as plt

plt.boxplot(results, labels=names, showmeans=True)
plt.show()