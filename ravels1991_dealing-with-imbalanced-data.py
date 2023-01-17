import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from collections import Counter





plt.style.use('ggplot')

plt.rcParams.update({'font.size': 14})



#ML Models

from sklearn.svm import SVC, LinearSVC

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, BaggingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

#Metrics 

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_auc_score



#Plot

import scikitplot as skplt



import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('../input/passenger-list-for-the-estonia-ferry-disaster/estonia-passenger-list.csv')

df.head()
#Checking nan values

df.isnull().mean() * 100
#Plot How many Survived

ax = df['Survived'].value_counts().plot(kind='bar', figsize=(12,6), color=['skyblue', 'violet'], rot=0,

                                  title='How many Survived?')
df['Sex'].value_counts().plot(kind='bar', color=['skyblue', 'violet'], rot=0, title='Numbers of Passagers by sex', figsize=(16,6));
df.groupby(['Sex', 'Survived'])['Survived'].count().unstack().plot(kind='bar', color=['skyblue', 'violet'], rot=0, title='Survived by Sex',

                                                                      figsize=(16,6));
df.groupby(['Country', 'Survived'])['Survived'].count().unstack().plot(kind='bar', color=['skyblue', 'violet'], rot=45, title='Survived by Country',

                                                                      figsize=(16,6));

print(f'How many Countrys the data have: {df["Country"].nunique()}')
df['Country'] = df['Country'].apply(lambda x: 'Other' if x not in df['Country'].value_counts()[:3].index.to_list() else x)
#Plot again

df.groupby(['Country', 'Survived'])['Survived'].count().unstack().plot(kind='bar', color=['skyblue', 'violet'], rot=45, title='Survived by Country',

                                                                      figsize=(16,6));

print(f'How many Countrys the data have: {df["Country"].nunique()}')
age_bins = [0, 10, 18, 30, 55, 100]

group_names = ['child', 'teenager', 'young adult', 'adult', 'elderly']

df['cat_age'] = pd.cut(df['Age'], age_bins, right=False, labels=group_names)
df['cat_age'].value_counts().plot(kind='bar', color='skyblue', rot=45, title='Passengers by category age', figsize=(16,6));
df.groupby(['cat_age', 'Survived'])['Survived'].count().unstack().plot(kind='bar', color=['skyblue', 'violet'], rot=45, figsize=(16, 6),

                                                                  title='Survived by category Age');
df_clean = df.drop(['PassengerId', 'Firstname', 'Lastname'], axis=1)
df_clean.head()
X = df_clean.drop(['Survived', 'cat_age'], axis=1)

X = pd.get_dummies(X, columns=['Country', 'Sex', 'Category'], drop_first=True)

y = df_clean['Survived']

X.head()
Seed = 12

knn = KNeighborsClassifier()

dt = DecisionTreeClassifier(random_state=Seed)

svc = SVC(gamma='auto', random_state=Seed)

ada = AdaBoostClassifier()

rf = RandomForestClassifier()

lr = LogisticRegression(max_iter=1000)

ls = LinearSVC()

bc = BaggingClassifier(base_estimator=DecisionTreeClassifier(),random_state=Seed)
model_list = [('KNeighborsClassifier', knn),

              ('DecisionTree', dt),

              ('SVC', svc),

              ('AdaBoost', ada),

              ('RandomForest', rf),

              ('LogisticRegression', lr),

              ('LinearSVC', ls),

              ('BaggingClassifier', bc)]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=Seed)
y_test
%%time 

for name, model in model_list:

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print(f'Model: {name} test_acc: {acc * 100:.2f}%')
predict_all_died = accuracy_score(y_test, np.zeros(len(y_test)))

print(f'Accuracy score if we predict everyone died in your test data: {predict_all_died *100:.2f}%')
predict_all_died = roc_auc_score(y_test, np.zeros(len(y_test)))

print(f'ROC AUC SCORE if we predict everyone died in your test data: {predict_all_died*100:.2f}%')
%%time 

for name, model in model_list:

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    roc_auc = roc_auc_score(y_test, y_pred)

    print(f'Model: {name} test_acc: {acc * 100:.2f}% roc_auc_test: {roc_auc * 100:.2f}%')
# Oversampling

from imblearn.over_sampling import SMOTE, ADASYN

from imblearn.over_sampling import BorderlineSMOTE

from imblearn.over_sampling import SMOTENC
smote = SMOTE()

adasyn = ADASYN()

bl = BorderlineSMOTE()

smote_nc = SMOTENC(categorical_features=[0, 1], random_state=Seed)
oversampling_list = [('SMOTE', smote),

                     ('ADASYN', adasyn),

                     ('BorderlineSMOTE', bl),

                     ('SMOTENC', smote_nc)]
#Create your validade set.

X, X_val, y, y_val = train_test_split(X, y, test_size=0.1, random_state=Seed)
#See our target

y.value_counts().plot(kind='bar', color='skyblue', rot=0, title='Data without oversampling');
# Now we will resample our data to make equal.

X_resampled, y_resampled = smote.fit_resample(X, y)

y_resampled.value_counts().plot(kind='bar', color='skyblue', rot=0, title='Data with oversampling');
%%time

results = []

for imblearn, method in oversampling_list:

    X_resampled, y_resampled = method.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=Seed)

    for name, model in model_list:

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        y_pred_val = model.predict(X_val)

        acc = round(accuracy_score(y_test, y_pred),4)

        acc_val = round(accuracy_score(y_val, y_pred_val),4)

        roc_test = round(roc_auc_score(y_test, y_pred), 4)

        roc_val = round(roc_auc_score(y_val, y_pred_val), 4)

        results.append({'Method': imblearn,'Model': name, 'test_acc': acc, 'val_acc': acc_val, 'roc_test': roc_test, 'roc_val': roc_val})
results = pd.DataFrame(results)
results.sort_values(by='roc_val', ascending=False).head(10)
#Undersampling



from imblearn.under_sampling import ClusterCentroids

from imblearn.under_sampling import RandomUnderSampler

from imblearn.under_sampling import EditedNearestNeighbours 

from imblearn.under_sampling import RepeatedEditedNearestNeighbours 

from imblearn.under_sampling import AllKNN

from imblearn.under_sampling import CondensedNearestNeighbour

from imblearn.under_sampling import OneSidedSelection

from imblearn.under_sampling import NeighbourhoodCleaningRule
cc = ClusterCentroids(random_state=Seed)

rus = RandomUnderSampler(random_state=Seed)

enn = EditedNearestNeighbours()

renn = RepeatedEditedNearestNeighbours()

allknn = AllKNN()

cnn = CondensedNearestNeighbour(random_state=Seed)

oss = OneSidedSelection(random_state=Seed)

ncr = NeighbourhoodCleaningRule()
undersampling_list = [('ClusterCentroids', cc),

                      ('RandomUnderSampler', rus),

                      ('EditedNearestNeighbours', enn),

                      ('RepeatedEditedNearestNeighbours', renn),

                      ('AllKNN', allknn),

                      ('CondensedNearestNeighbour', cnn),

                      ('OneSidedSelection', oss),

                      ('NeighbourhoodCleaningRule', ncr)]
y.value_counts().plot(kind='bar', color='skyblue', rot=0, title='Data without undersampling');
X_resampled, y_resampled = cc.fit_resample(X, y)

y_resampled.value_counts().plot(kind='bar', color='skyblue', rot=0, title='Data with undersampling');
%%time

results_under = []

for imblearn, method in undersampling_list:

    X_resampled, y_resampled = method.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=Seed)

    for name, model in model_list:

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        y_pred_val = model.predict(X_val)

        acc = round(accuracy_score(y_test, y_pred),4)

        acc_val = round(accuracy_score(y_val, y_pred_val),4)

        roc_test = round(roc_auc_score(y_test, y_pred), 4)

        roc_val = round(roc_auc_score(y_val, y_pred_val), 4)

        results_under.append({'Method': imblearn,'Model': name, 'test_acc': acc, 'val_acc': acc_val, 'roc_test': roc_test, 'roc_val': roc_val})
test_under = pd.DataFrame(results_under)

test_under.head()
test_under.sort_values(by=['roc_val', 'roc_test'], ascending=[False, False])[:10]
%%time

mixed = []

for imblearn, method in oversampling_list:

    X_resampled, y_resampled = method.fit_resample(X, y)

    for imblearn2, method2 in undersampling_list:

        X_resampled1, y_resampled1 = method2.fit_resample(X_resampled, y_resampled)

        X_train, X_test, y_train, y_test = train_test_split(X_resampled1, y_resampled1, test_size=0.2, random_state=Seed)

        for name, model in model_list:

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            y_pred_val = model.predict(X_val)

            acc = round(accuracy_score(y_test, y_pred),4)

            acc_val = round(accuracy_score(y_val, y_pred_val),4)

            roc_test = round(roc_auc_score(y_test, y_pred), 4)

            roc_val = round(roc_auc_score(y_val, y_pred_val), 4)

            mixed.append({'Method 1': imblearn,'Method 2': imblearn2, 'Model': name, 'test_acc': acc, 'val_acc': acc_val, 'roc_test': roc_test, 'roc_val': roc_val})
mixed = pd.DataFrame(mixed)

mixed.head()
mixed.sort_values(by=['roc_val', 'roc_test'], ascending=[False, False])[:10]
fig, (axes1, axes2) = plt.subplots(1, 2, figsize=(16, 6))



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=Seed)

dt.fit(X_train, y_train)

y_probas = dt.predict_proba(X_val)

y_pred = dt.predict(X_val)

acc = accuracy_score(y_val, y_pred )

skplt.metrics.plot_roc(y_val, y_probas, cmap='cool', plot_micro=False, plot_macro=False,ax= axes1)

skplt.metrics.plot_confusion_matrix(y_val, y_pred, ax=axes2)

fig.suptitle(f'Results without sampling Data and DecisionTree  Accuracy Validation Test: {acc *100:.2f}%\n', fontsize=20) 

plt.show()
fig, (axes1, axes2) = plt.subplots(1, 2, figsize=(16, 6))



X_resampled, y_resampled = smote_nc.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=Seed)

ada.fit(X_train, y_train)

y_probas = ada.predict_proba(X_val)

y_pred = ada.predict(X_val)

acc = accuracy_score(y_val, y_pred )

skplt.metrics.plot_roc(y_val, y_probas, cmap='cool', plot_micro=False, plot_macro=False,ax= axes1)

skplt.metrics.plot_confusion_matrix(y_val, y_pred, ax=axes2)

fig.suptitle(f'Results with SMOTENC and AdaBoost  Accuracy Validation Test: {acc *100:.2f}%\n', fontsize=20)

plt.show()
fig, (axes1, axes2) = plt.subplots(1, 2, figsize=(16, 6))



X_resampled, y_resampled = renn.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=Seed)

ada.fit(X_train, y_train)

y_probas = ada.predict_proba(X_val)

y_pred = ada.predict(X_val)

acc = accuracy_score(y_val, y_pred )

skplt.metrics.plot_roc(y_val, y_probas, cmap='cool', plot_micro=False, plot_macro=False,ax= axes1)

skplt.metrics.plot_confusion_matrix(y_val, y_pred, ax=axes2)

fig.suptitle(f'Results with RepeatedEditedNearestNeighbours and AdaBoost  Accuracy Validation Test: {acc *100:.2f}%\n', fontsize=20)

plt.show()


fig, (axes1, axes2) = plt.subplots(1, 2, figsize=(16, 6))



X_resampled, y_resampled = smote_nc.fit_resample(X, y)

X_resampled, y_resampled = enn.fit_resample(X_resampled, y_resampled)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=Seed)

lr.fit(X_train, y_train)

y_probas = lr.predict_proba(X_val)

y_pred = lr.predict(X_val)

acc = accuracy_score(y_val, y_pred)

skplt.metrics.plot_roc(y_val, y_probas, cmap='cool', plot_micro=False, plot_macro=False,ax= axes1)

skplt.metrics.plot_confusion_matrix(y_val, y_pred, ax=axes2)

fig.suptitle(f'Results with SMOTENC, EditedNearestNeighbours and LogisticRegression  Accuracy Validation Test: {acc *100:.2f}%\n', fontsize=20)

plt.show()