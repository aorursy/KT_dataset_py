import warnings

import pandas as pd

import numpy as np

import seaborn as sns



from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, StratifiedKFold

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
import pandas as pd

df = pd.read_excel('../input/bank-loan-modelling/Bank_Personal_Loan_Modelling.xlsx', sheet_name='Data')

df = df.loc[:,['ID', 'Age', 'Experience', 'Income', 'ZIP Code', 'Family', 'CCAvg',

       'Education', 'Mortgage',  'Securities Account',

       'CD Account', 'Online', 'CreditCard', 'Personal Loan']]



# Remove ID column

df = df.iloc[:,1:14]

df
sns.set_style('darkgrid')

sns.pairplot(df, height=1)

#...scale Income and CCAvg later
df['ZIP2'] = df['ZIP Code'].astype(str).str[0:2]

sns.set_style('darkgrid')

sns.countplot(x = 'ZIP2', data = df, palette = 'Blues_d')

plt.ylabel('Count')

plt.title('Count of First Two Digits of ZIP Code')





# # Rearrange cols

# df = df.loc[:,['Age', 'Experience', 'Income', 'Family', 'CCAvg',

#        'Education', 'Mortgage',  'Securities Account',

#        'CD Account', 'Online', 'CreditCard', 'ZIP2', 'Personal Loan']]

# df
#Remove ZIP Code and rearrange cols

df.drop(columns=['ZIP Code'], inplace=True)



df = df.loc[:,['Age', 'Experience', 'Income', 'Family', 'CCAvg', 'Education',

       'Mortgage', 'Securities Account', 'CD Account', 'Online', 'CreditCard',

       'ZIP2','Personal Loan']]

df
x = df.iloc[:,0:12]

y = df.iloc[:,-1]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.2, random_state=7)

x
models = []

models.append(('LR', LogisticRegression(max_iter=1000)))

models.append(('CART', DecisionTreeClassifier()))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('SVC', SVC()))

models
my_cv = []

my_names = []



for name, model in models:

    kfold = StratifiedKFold(n_splits=10, random_state=7)

    cv = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')

    my_names.append(name)

    my_cv.append(cv)

    msg = ('%s %f (%f)' % (name, cv.mean(), cv.std()))

    print(msg)

    #...CART has the highest acccuracy at 98.2%
fig = plt.figure()

ax = fig.add_subplot(111)

plt.boxplot(my_cv)

ax.set_xticklabels(my_names)

plt.show()
cols = ['Income','CCAvg']

cols2 = ['Age', 'Experience', 'Family', 'Education',

       'Mortgage', 'Securities Account', 'CD Account', 'Online', 'CreditCard',

       'ZIP2', 'Personal Loan']



foo = df.loc[:,['Income','CCAvg']]

scaler = StandardScaler().fit(foo)



foo = pd.DataFrame(scaler.transform(foo))

foo.columns = ['Income','CCAvg']



df_scaled = pd.concat([foo, df.loc[:,cols2]], axis=1)

df_scaled = df_scaled.loc[:,all_cols]

df_scaled = df_scaled.loc[:,['Age', 'Experience', 'Income', 'Family', 'CCAvg', 'Education','Mortgage', 'Securities Account', 'CD Account', 'Online', 'CreditCard','ZIP2','Personal Loan']]

df_scaled
x = df_scaled.iloc[:,0:12]

y = df_scaled.iloc[:,-1]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.2, random_state=7)

my_cv = []

my_names = []



for name, model in models:

    kfold = StratifiedKFold(n_splits=10, random_state=7)

    cv = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')

    my_names.append(name)

    my_cv.append(cv)

    msg = ('%s %f (%f)' % (name, cv.mean(), cv.std()))

    print(msg)

    #... CART slightly increased to 98.40%
fig = plt.figure()

fig.suptitle('Comparison of Algorithms on Scaled Data')

ax = fig.add_subplot(111)

plt.boxplot(my_cv)

ax.set_xticklabels(my_names)

plt.show()
# c_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0]

# kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']

# param_grid = dict(C=c_values, kernel=kernel_values)



parameters={'min_samples_split' : range(10,500,20),'max_depth': range(1,20,2)}



#Note: x_train data that was used is scaled

model = DecisionTreeClassifier()

kfold = StratifiedKFold(n_splits = 10, random_state=7)

grid = GridSearchCV(estimator=model, param_grid=parameters, scoring='accuracy', cv=kfold)

grid_result = grid.fit(x_train, y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print('%f (%f) with %r' % (mean, stdev, param))

#...accuracy is at 98.3% with max_depth=3, min_saples_split=70
ensembles = []

ensembles.append(('Ada', AdaBoostClassifier()))

ensembles.append(('GB', GradientBoostingClassifier()))

ensembles.append(('BC', BaggingClassifier()))

ensembles.append(('ET', ExtraTreesClassifier()))

ensembles
my_cv = []

my_names = []



for name, model in ensembles:

    kfold = StratifiedKFold(n_splits=10, random_state=7)

    cv = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')

    my_names.append(name)

    my_cv.append(cv)

    msg = ('%s %f (%f)' % (name, cv.mean(), cv.std()))

    print(msg)



#...GB has the highest at 98.6%
fig = plt.figure()

fig.suptitle('Comparison of Ensemble Methods on Scaled Data')

ax = fig.add_subplot(111)

plt.boxplot(my_cv)

ax.set_xticklabels(my_names)

plt.show()
model = GradientBoostingClassifier()

model.fit(x_train, y_train)



# note that x_test was already scaled

predictions = model.predict(x_test)

print(accuracy_score(y_test, predictions))

print(confusion_matrix(y_test, predictions))

print(classification_report(y_test, predictions))



pd.set_option('precision',5)

print(model.feature_importances_*10)

print(x_test.columns)

#...model accuracy is 97.9%

#...top predictors are Income, Education, and Family)