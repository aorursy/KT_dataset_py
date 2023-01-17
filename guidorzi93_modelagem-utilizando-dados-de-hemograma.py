import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from scipy.stats import ks_2samp

from collections import OrderedDict

from operator import itemgetter



from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split, KFold

from sklearn.metrics import plot_confusion_matrix, recall_score, precision_score, plot_precision_recall_curve

from sklearn.ensemble import RandomForestClassifier

from sklearn.dummy import DummyClassifier



from imblearn.over_sampling import SMOTE
df = pd.read_excel('../input/covid19/dataset.xlsx',unidecode='ascii')

df_positive = df[df['SARS-Cov-2 exam result']=='positive']

df_negative = df[df['SARS-Cov-2 exam result']=='negative']



df = df.drop(['Patient addmited to regular ward (1=yes, 0=no)', 

                  'Patient addmited to semi-intensive unit (1=yes, 0=no)',

                  'Patient addmited to intensive care unit (1=yes, 0=no)'],axis=1)

df = df.set_index('Patient ID')
nan_analyze = df.isna().sum()/len(df)



print("Quantidade de tuplas:", len(df))

print("Percentual médio missing values:", round(nan_analyze.mean()*100,1),"%")



label_perc = []

for i in np.arange(0, len(df.columns), 10):

    label_perc.append(str(i)+"%")

plt.figure(figsize=[10,40])



plt.yticks(np.arange(len(df.columns)), nan_analyze.index.values)

plt.xticks(np.arange(0, 1.1, .1), label_perc)



plt.ylim(0,len(df.columns))



plt.barh(np.arange(len(df.columns)), nan_analyze)
df_filtered = df[~np.isnan(df['Hematocrit'])]

nan_analyze_filtered = df_filtered.isna().sum()/len(df_filtered)



print("Quantidade de tuplas:", len(df_filtered))

print("Percentual médio missing values:", round(nan_analyze_filtered.mean()*100,1),"%")





label_perc = []

for i in np.arange(0, 110, 10):

    label_perc.append(str(i)+"%")

plt.figure(figsize=[10,40])



plt.yticks(np.arange(len(df_filtered.columns)), nan_analyze_filtered.index.values)

plt.xticks(np.arange(0, 1.1, .1), label_perc)



plt.ylim(0,len(df_filtered.columns))



plt.barh(np.arange(len(df_filtered.columns)), nan_analyze_filtered)
df_filtered = df_filtered[nan_analyze_filtered[nan_analyze_filtered<=.4].index.values]



nan_analyze_filtered = df_filtered.isna().sum()/len(df_filtered)



print("Quantidade de tuplas:", len(df_filtered))

print("Percentual médio missing values:", round(nan_analyze_filtered.mean()*100,1),"%")



label_perc = []

for i in np.arange(0, 110, 10):

    label_perc.append(str(i)+"%")

plt.figure(figsize=[10,10])



plt.yticks(np.arange(len(df_filtered.columns)), nan_analyze_filtered.index.values)

plt.xticks(np.arange(0, 1.1, .1), label_perc)



plt.ylim(0,len(df_filtered.columns))



plt.barh(np.arange(len(df_filtered.columns)), nan_analyze_filtered)
features = df_filtered.dtypes[df_filtered.dtypes=='float64'].index.values



ks_list = []

pvalue_list = []

feature_list = []



for feature in features:

    

    positive = df_positive[~np.isnan(df_positive[feature])]

    negative = df_negative[~np.isnan(df_negative[feature])]

    

    if len(positive)*len(negative)>0:

        ks, pvalue = ks_2samp(positive[feature], negative[feature])

        ks_list.append(ks)

        pvalue_list.append(pvalue)

        feature_list.append(feature)

        

df_ks = pd.DataFrame(data=zip(ks_list,pvalue_list),columns=['ks', 'pvalue'],index=feature_list)

df_ks = df_ks.sort_values(by='ks',ascending=True)



df_ks['ks']

plt.figure(figsize=(8,15))

plt.yticks(np.arange(len(df_ks)), df_ks.index.values)

plt.title('Diferença entre positivo VS negativo')

plt.barh(np.arange(len(df_ks)), df_ks['ks'])
df_treated = df_filtered

cat_features = df_filtered.dtypes[df_filtered.dtypes == 'object'].index.values



for feature in cat_features:

    df_treated[feature] = df_treated[feature].fillna(df_treated[feature].mode().values[0]) 

    

df_treated = df_treated.fillna(df_treated.median())



df_treated_dummies = pd.get_dummies(df_treated, drop_first=True, dtype='bool')



columns = list(df_treated_dummies.drop(labels=['SARS-Cov-2 exam result_positive'],axis=1).columns.values)

columns.append('SARS-Cov-2 exam result_positive')



df_treated_dummies = df_treated_dummies[columns]
ax = df_treated['SARS-Cov-2 exam result'].value_counts().plot(kind='bar',

                                    figsize=(14,8))

ax.set_xticklabels(['negative', 'positive'], rotation=0, fontsize=20)
y = df_treated_dummies['SARS-Cov-2 exam result_positive']

x = df_treated_dummies.iloc[:,:-1]

x['Random'] = np.random.rand(x.shape[0])



X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)



smote = SMOTE()

X_train_random, y_train_random = smote.fit_sample(X_train, y_train)

X_train, y_train = smote.fit_sample(X_train.iloc[:, :-1], y_train)
kfold = KFold(n_splits=20, random_state=42)



param_grid = {

    'min_samples_split':[2, 4, 6],

    'min_samples_leaf':[2, 4, 6],

    'n_estimators':[10, 30, 50],

    'max_depth':[3, 5]

    }



clf_rf = RandomForestClassifier(random_state=42)

grid = GridSearchCV(estimator=clf_rf, param_grid=param_grid, cv=kfold, scoring='recall', n_jobs=-1)

grid.fit(X=X_train, y=y_train)

clf_rf = grid.best_estimator_
clf_rf.fit(X_train_random, y_train_random)

    

cols = x.columns

feature_importance = pd.DataFrame(data=clf_rf.feature_importances_, index=cols, columns=['FI'])

feature_importance = feature_importance.sort_values(by='FI', ascending=True)

    

plt.figure(figsize=(8,10))



plt.yticks(np.arange(len(feature_importance)), feature_importance.index.values)

plt.barh(np.arange(len(feature_importance)), feature_importance['FI'])



plt.show()
columns = list(feature_importance.sort_values(by='FI', ascending=False).index.values)

columns_new = columns[:columns.index('Random')]

print("Quantidade de features:", len(columns_new))



y = df_treated_dummies['SARS-Cov-2 exam result_positive']

x = df_treated_dummies[columns_new]



X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)



smote = SMOTE()

X_train, y_train = smote.fit_sample(X_train, y_train)
kfold = KFold(n_splits=20, random_state=42)



param_grid = {

    'min_samples_split':[2, 4, 6],

    'min_samples_leaf':[2, 4, 6],

    'n_estimators':[10, 30, 50],

    'max_depth':[3, 5]

    }



clf_rf = RandomForestClassifier(random_state=42)

grid = GridSearchCV(estimator=clf_rf, param_grid=param_grid, cv=kfold, scoring='recall', n_jobs=-1)

grid.fit(X=X_train, y=y_train)

clf_rf = grid.best_estimator_
clf_rf.fit(X_train, y_train)



plot_confusion_matrix(clf_rf, X_test, y_test, cmap=plt.cm.Blues, values_format='.00f')



print("Recall treino:", recall_score(y_train, clf_rf.predict(X_train)))



print("Recall validação:", cross_val_score(clf_rf, X_train, y_train, cv=20, scoring='recall', n_jobs=-1).mean())



clf_rf.fit(X_train, y_train)

print("Recall teste:", recall_score(y_test, clf_rf.predict(X_test)))