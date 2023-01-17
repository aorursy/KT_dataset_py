import warnings

import pandas as pd

import numpy as np

import seaborn as sns



from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV

import matplotlib.pyplot as plt



from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from xgboost import XGBClassifier



from xgboost import plot_importance



from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import ExtraTreesClassifier





from sklearn.pipeline import Pipeline



from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



warnings.filterwarnings('ignore')
df = pd.read_csv('../input/bank-customer-churn-modeling/Churn_Modelling.csv')
# Remove RowNumber and CustomerId

df = df.iloc[:,2:14]



# Add Surname2 col

foo = df['Surname'].value_counts()

foo = pd.DataFrame(foo)

foo['Name'] = foo.index

foo = foo[foo['Surname'] > 1]

foo = foo.Name.unique()

foo = list(foo)

foo.sort()

len(foo)

len(df.Surname.unique())

df['Surname2'] = 0

df.loc[df['Surname'].isin(foo), ['Surname2']] = 1



# # Remove Surname col

# df = df.iloc[:,1:14]



# Rearrange cols

cols = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance',

       'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary',

       'Surname2', 'Exited']



df = df[cols]

df
# Analyze numeric cols

num_cols = ['CreditScore', 'Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary', 'Surname2', 'Exited']

df_num = df[num_cols]



sns.set_style('darkgrid')

sns.pairplot(df_num, height=1)



#...scale CreditScore, Age, Tenure, Balance, EstimatedSalary

# NumOfProducts is already categorized

#...retain binomial: HasCrCard, IsActiveMember, Surname2
lab_enc = LabelEncoder()

cols = ['Geography','Gender']



for _ in cols:

    df[_] = lab_enc.fit_transform(df[_])

df
x = df.iloc[:,0:11]

y = df.iloc[:,-1]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.2, random_state=7)
models = []

models.append(('LR', LogisticRegression(max_iter=1000)))

models.append(('CART', DecisionTreeClassifier()))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('SVC', SVC()))

models.append(('XGB', XGBClassifier()))

models
my_cv = []

my_names = []



for name, model in models:

    kfold = KFold(n_splits=10, random_state=7)

    cv = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')

    my_names.append(name)

    my_cv.append(cv)

    msg = ('%s %f (%f)' % (name, cv.mean(), cv.std()))

    print(msg)
fig = plt.figure()

fig.suptitle('Comparison of Algorithms on Unscaled Data (Baseline)')

ax = fig.add_subplot(111)

plt.boxplot(my_cv)

ax.set_xticklabels(my_names)

plt.show()

#...XGB has the highest accuracy at 83.05% with a std of 0.01850
df_all_cols = df.columns

cols = ['CreditScore','Age','Tenure','Balance','EstimatedSalary']

cols2 = ['Geography', 'Gender', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'Surname2', 'Exited']



df_cats = df[cols]

scaler = StandardScaler().fit(df_cats)

foo = pd.DataFrame(scaler.transform(df_cats))

foo.columns = ['CreditScore','Age','Tenure','Balance','EstimatedSalary']



df_scaled = pd.concat([foo, df[cols2]], axis=1)



sns.set_style('darkgrid')

sns.pairplot(df_scaled[cols], height=1)
x = df_scaled.iloc[:,0:11]

y = df_scaled.iloc[:,-1]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.2, random_state=7)
my_cv = []

my_names = []



for name, model in models:

    kfold = KFold(n_splits=10, random_state=7)

    cv = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')

    my_names.append(name)

    my_cv.append(cv)

    msg = ('%s %f (%f)' % (name, cv.mean(), cv.std()))

    print(msg)
fig = plt.figure()

fig.suptitle('Comparison of Algorithms on Scaled Data')

ax = fig.add_subplot(111)

plt.boxplot(my_cv)

ax.set_xticklabels(my_names)

plt.show()\

#...SVC's accuracy significantly jumped to 83.75% with an std of 0.026481
c_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0]

kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']

param_grid = dict(C=c_values, kernel=kernel_values)



#Note: x_train data that was used is scaled

model = XGBClassifier()

kfold = KFold(n_splits = 10, random_state=7)

grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=kfold)

grid_result = grid.fit(x_train, y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print('%f (%f) with %r' % (mean, stdev, param))

#...accuracy for tuned SVC on scaled data slightly worsened to 83.10%
min_child_weight = [1, 5, 10]

gamma = [0.5, 1, 1.5, 2, 5]

subsample = [0.6, 0.8, 1.0]

colsample_bytree =  [0.6, 0.8, 1.0]

max_depth = [3, 4, 5]

        

param_grid = dict(min_child_weight = min_child_weight, gamma = gamma, subsample = subsample, colsample_bytree = colsample_bytree, max_depth = max_depth)

model = XGBClassifier()

kfold = KFold(n_splits = 10, random_state=7)

grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=kfold)

grid_result = grid.fit(x_train, y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print('%f (%f) with %r' % (mean, stdev, param))

#...the accuracy of the tuned xgb on scaled data significantly jumped to 85.3% using colsample_bytree=0.8, gamma=5, max_depth=4, min_child_weight=1, and subsample=1
ensembles = []

ensembles.append(('ADA', AdaBoostClassifier()))

ensembles.append(('GB', GradientBoostingClassifier()))

ensembles.append(('BC', BaggingClassifier()))

ensembles.append(('ET', ExtraTreesClassifier()))
my_cv = []

my_names = []



#

for name, model in ensembles:

    kfold = KFold(n_splits=10, random_state=7)

    cv = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')

    my_names.append(name)

    my_cv.append(cv)

    msg = ('%s %f (%f)' % (name, cv.mean(), cv.std()))

    print(msg)
fig = plt.figure()

fig.suptitle('Accuracy Comparison of Algorithms Using Ensembles')

ax = fig.add_subplot(111)

plt.boxplot(my_cv)

ax.set_xticklabels(my_names)

plt.show()

#...GB resulted to 84.95 using scaled data
model = XGBClassifier(colsample_bytree=0.8, gamma=5, max_depth=4, min_child_weight=1, subsample=1)

model.fit(x_train, y_train)

predictions = model.predict(x_test)

print(accuracy_score(y_test, predictions))

print(confusion_matrix(y_test, predictions))

print(classification_report(y_test, predictions))

plot_importance(model)

#...accuracy is at 85.78%, important vars are NumOfProducts, Age, and Balance
model = GradientBoostingClassifier()

model.fit(x_train, y_train)

predictions = model.predict(x_test)

print(accuracy_score(y_test, predictions))

print(confusion_matrix(y_test, predictions))

print(classification_report(y_test, predictions))

print(model.feature_importances_)

print(x_test.columns)

#...acccuracy is highest at 85.9%