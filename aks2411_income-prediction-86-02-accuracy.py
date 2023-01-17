import numpy as np

import pandas as pd



df = pd.read_csv("../input/adult.csv")

df.head()
df.dtypes
df.isnull().sum()
df.columns.isna()
df.isin(['?']).sum()
df = df.replace('?', np.NaN)

df.head()
df = df.dropna()

df.head()
df['income'] = df['income'].map({'<=50K':0, '>50K':1})

df.income.head()
numerical_df = df.select_dtypes(exclude=['object'])

numerical_df.columns
import seaborn as sns

import matplotlib.pyplot as plt
plt.hist(df['age'], edgecolor='black')

plt.title('Age Histogram')

plt.axvline(np.mean(df['age']), color='yellow', label='average age')

plt.legend()
age50k = df[df['income']==1].age

agel50k = df[df['income']==0].age



fig, axs = plt.subplots(2, 1)



axs[0].hist(age50k, edgecolor='black')

axs[0].set_title('Distribution of Age for Income > 50K')



axs[1].hist(agel50k, edgecolor='black')

axs[1].set_title('Distribution of Age for Income <= 50K')

plt.tight_layout()
df['marital.status'].unique()
ax = sns.countplot(df['marital.status'], hue=df['income'])

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

plt.tight_layout()
df['marital.status'] = df['marital.status'].replace(['Widowed', 'Divorced', 'Separated', 'Never-married'], 'single')



df['marital.status'] = df['marital.status'].replace(['Married-spouse-absent', 'Married-civ-spouse', 'Married-AF-spouse'], 'married')
categorical_df = df.select_dtypes(include=['object'])

categorical_df.columns
sns.countplot(df['marital.status'], hue=df['income'])
from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()
ax = sns.countplot(df['income'], hue=df['race'])

ax.set_title('')
categorical_df = categorical_df.apply(enc.fit_transform)

categorical_df.head()
df = df.drop(categorical_df.columns, axis=1)

df = pd.concat([df, categorical_df], axis=1)

df.head()
sns.factorplot(data=df, x='education', y='hours.per.week', hue='income', kind='point')
sns.FacetGrid(data=df, hue='income', size=6).map(plt.scatter, 'age', 'hours.per.week').add_legend()
plt.figure(figsize=(15,12))

cor_map = df.corr()

sns.heatmap(cor_map, annot=True, fmt='.3f', cmap='YlGnBu')
from sklearn.model_selection import train_test_split



X = df.drop('income', axis=1)

y = df['income']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=24)
from sklearn.ensemble import RandomForestClassifier



clf = RandomForestClassifier(n_estimators=100, random_state=24)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score



print("Random Forests accuracy", accuracy_score(y_test, y_pred))
from sklearn.tree import DecisionTreeClassifier



dtree = DecisionTreeClassifier(criterion='gini', random_state=21, max_depth=10)



dtree.fit(X_train, y_train)

tree_pred = dtree.predict(X_test)



print("Decision Tree accuracy: ", accuracy_score(y_test, tree_pred))
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, KFold



n_estimators = np.arange(100, 1000, 100)

max_features = np.arange(1, 10, 1)

min_samples_leaf = np.arange(2, 10, 1)

kfold = KFold(n_splits = 3)

start_grid = {

    'n_estimators': n_estimators,

    'max_features': max_features,

    'min_samples_leaf': min_samples_leaf,

    }



rf = RandomForestClassifier()



test_rf = RandomizedSearchCV(estimator=rf, param_distributions=start_grid, cv=kfold)

print(start_grid)
'''

Commented out since takes a long time to run. 



------------------------------

OPTIMIZED PARAMETERS:

max_features = 3

min_samples_leaf = 5

n_estimators = 100

------------------------------





test_rf.fit(X_train, y_train)

test_rf.best_params_

'''
'''

Commented out since takes about 25 minutes to run

----------------------------------

OPTIMIZED HYPERPARAMETERS:



max_features = 3

min_samples_leaf = 3

n_estimators = 450

-----------------------------------



kfold_gs = KFold(n_splits=3)

n_estimators = np.arange(100, 500, 50)

max_features = np.arange(1, 5, 1)

min_samples_leaf = np.arange(2, 5, 1)



gs_grid = {

    'n_estimators': n_estimators,

    'max_features': max_features,

    'min_samples_leaf': min_samples_leaf

}



test_grid = GridSearchCV(estimator = rf, param_grid=gs_grid, cv=kfold_gs)

res = test_grid.fit(X_train, y_train)

print(res.best_params_)

print(res.best_score_)

'''
final_model = RandomForestClassifier(n_estimators=450, min_samples_leaf=3, max_features=3, random_state=24)

final_model.fit(X_train, y_train)
predictions = final_model.predict(X_test)

print(accuracy_score(y_test, predictions))

from sklearn.metrics import roc_curve, auc

n_estimators = np.arange(100, 1000, 100)



train_results = []

test_results = []

for n_est in n_estimators:

   rf = RandomForestClassifier(n_estimators = n_est)

   rf.fit(X_train, y_train)



   train_pred = rf.predict(X_train)



   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)

   roc_auc = auc(false_positive_rate, true_positive_rate)

   train_results.append(roc_auc)



   y_pred = rf.predict(X_test)



   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)

   roc_auc = auc(false_positive_rate, true_positive_rate)

   test_results.append(roc_auc)



from matplotlib.legend_handler import HandlerLine2D



line1, = plt.plot(n_estimators, train_results, 'b', label='Train AUC')

line2, = plt.plot(n_estimators, test_results, 'r', label='Test AUC')



plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})



plt.ylabel('AUC score')

plt.xlabel('n_estimators')

from sklearn.metrics import roc_curve, auc

max_features = np.arange(1, 10, 1)



train_results = []

test_results = []

for max_f in max_features:

   rf = RandomForestClassifier(max_features=max_f)

   rf.fit(X_train, y_train)



   train_pred = rf.predict(X_train)



   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)

   roc_auc = auc(false_positive_rate, true_positive_rate)

   train_results.append(roc_auc)



   y_pred = rf.predict(X_test)



   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)

   roc_auc = auc(false_positive_rate, true_positive_rate)

   test_results.append(roc_auc)



from matplotlib.legend_handler import HandlerLine2D



line1, = plt.plot(max_features, train_results, 'b', label='Train AUC')

line2, = plt.plot(max_features, test_results, 'r', label='Test AUC')



plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})



plt.ylabel('AUC score')

plt.xlabel('max_features')

from sklearn.metrics import roc_curve, auc

min_samples_leafs = np.arange(2, 10, 1)



train_results = []

test_results = []

for min_samples_leaf in min_samples_leafs:

   rf = RandomForestClassifier(min_samples_leaf=min_samples_leaf)

   rf.fit(X_train, y_train)



   train_pred = rf.predict(X_train)



   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)

   roc_auc = auc(false_positive_rate, true_positive_rate)

   train_results.append(roc_auc)



   y_pred = rf.predict(X_test)



   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)

   roc_auc = auc(false_positive_rate, true_positive_rate)

   test_results.append(roc_auc)



from matplotlib.legend_handler import HandlerLine2D



line1, = plt.plot(min_samples_leafs, train_results, 'b', label='Train AUC')

line2, = plt.plot(min_samples_leafs, test_results, 'r', label='Test AUC')



plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})



plt.ylabel('AUC score')

plt.xlabel('min samples leaf')


