# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt



# allow multi-outputs

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
input = pd.read_csv("../input/insurance/insurance.csv")
input.head(5)

input.info()

input.describe()
plt.style.use('ggplot')
for col in input.loc[:,['age','bmi','children','charges']].columns:

    sns.distplot(a=input[col]);

    plt.show();
input.groupby('children').charges.agg(['median','mean']).plot(kind='bar', title='Charges by number of children - mean vs median');
plt.figure(figsize=(8,5))

plt.title('Charges by smoker status')

sns.violinplot(data=input, x='smoker', y='charges');

plt.show()



plt.figure(figsize=(8,5))

sns.scatterplot(x=input['bmi'], y=input['charges'], hue=input['smoker']);

plt.title('Charges by BMI');

plt.show();
plt.figure(figsize=(8,8))

plt.title('Charges by region')

sns.swarmplot(x=input['region'], y=input['charges']);

plt.show();



plt.figure(figsize=(8,8))

plt.title('Charges by region')

sns.boxplot(data=input, x='region', y='charges');

sns.swarmplot(data=input, x='region', y='charges', size=2, color=".3")

plt.show();



input.groupby('region').charges.agg(['mean','median']).sort_values(by='region', ascending=False).plot(kind="bar", title='Mean and median charges by region');
cnt_smoker_byRegion = input.groupby(['region', 'smoker']).agg({'smoker':'count'})

cnt_byRegion = input.groupby('region').agg({'smoker':'count'})

cnt_smoker_byRegion.div(cnt_byRegion, level='region')
plt.figure(figsize=(8,8))

sns.boxplot(data=input, x='region', y='bmi', hue='smoker');

plt.title("BMI by region and smoker status");
sns.boxplot(data=input.loc[(((input.region=='southeast') & (input.charges<42000)) | ((input.region=='northeast') & (input.charges<35000)))],

            x='smoker', y='charges', hue='region');

plt.show();



sns.boxplot(data=input.loc[(((input.region=='southeast') & (input.charges<42000)) | ((input.region=='northeast') & (input.charges<35000)))],

            x='sex', y='charges', hue='region');

plt.show();



sns.boxplot(data=input.loc[(((input.region=='southeast') & (input.charges<42000)) | ((input.region=='northeast') & (input.charges<35000)))],

            x='children', y='charges', hue='region')

plt.show();
input.loc[(((input.region=='southeast') & (input.charges<42000)) | ((input.region=='northeast') & (input.charges<35000)))].groupby(['region','children']).charges.median().plot(kind="bar");
from scipy.stats import kruskal
sw = input.loc[input.region=='southwest','charges']

se = input.loc[input.region=='southeast','charges']

ne = input.loc[input.region=='northeast','charges']

nw = input.loc[input.region=='northwest','charges']



kruskal(sw, se, ne, nw)
fg = sns.FacetGrid(data=input, row='region', col='children');

fg.map(plt.scatter, 'bmi', 'charges');

fg.add_legend();
input.groupby('sex').charges.agg(['mean','median'])
plt.figure(figsize=(8,8))

plt.title('Charges by gender')

sns.boxplot(data=input, x='sex', y='charges');

plt.show();
plt.figure(figsize=(8,8))

plt.title('Charges by gender')

sns.boxplot(data=input.loc[(((input.sex == 'female') & (input.charges < 30000)) | ((input.sex == 'male') & (input.charges < 40000))),:], x='sex', y='charges');

plt.show();
input.loc[(((input.sex == 'female') & (input.charges < 30000)) | ((input.sex == 'male') & (input.charges < 40000))),:].groupby('sex').charges.agg(['mean','median'])
sns.boxplot(data=input, x='smoker', y='charges', hue='sex')

plt.title("Charges by gender and smoker status")

plt.show();



input.groupby(['sex','smoker']).charges.median().sort_values().plot(title='Median charges by gender and smoker status');
input.groupby(['sex', 'smoker']).bmi.mean().sort_values().plot(title='Average BMI by gender and smoker status');



sns.lmplot(data=input, x='bmi', y='charges', hue='sex');

plt.title('Regression line for BMI/Charges, gender-wise')

plt.show();
from scipy.stats import mannwhitneyu
mannwhitneyu(input.loc[input.sex=='female','charges'].values,input.loc[input.sex=='male','charges'].values)
plt.figure(figsize=(8,8))

plt.title('Charges by number of children')

sns.boxplot(data=input, x='children', y='charges');

plt.show();
# split the charges by number of children

children = []

for i in range(0,6):

    children.append(input.loc[input.children==i,'charges'])
kruskal(children[0], children[1], children[2], children[3], children[4], children[5])

kruskal(children[0], children[4])

kruskal(children[2], children[5])

kruskal(children[1], children[5])
X = input.iloc[:,0:6]

y = input.iloc[:,6]
X.head()
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
LabelEncoder_X1 = LabelEncoder()

LabelEncoder_X4 = LabelEncoder()

LabelEncoder_X5 = LabelEncoder()

X.iloc[:,1] = LabelEncoder_X1.fit_transform(X.iloc[:,1])

X.iloc[:,4] = LabelEncoder_X4.fit_transform(X.iloc[:,4])

X.iloc[:,5] = LabelEncoder_X5.fit_transform(X.iloc[:,5])
from sklearn.compose import ColumnTransformer



ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [1,4,5])], remainder='passthrough')

X = ct.fit_transform(X)
X = X[:,[1,2,4,5,6,8,9,10]]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression



regressor1 = LinearRegression()

regressor1.fit(X_train, y_train)



y_pred1 = regressor1.predict(X_test)
MAE1 = mean_absolute_error(y_test, y_pred1)

MAE1
from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression
regressor2 = PolynomialFeatures(degree=3)

X_poly = regressor2.fit_transform(X_train)

regressor2.fit(X_poly, y_train)



linreg = LinearRegression()

linreg.fit(X_poly,y_train)



y_pred2 = linreg.predict(regressor2.fit_transform(X_test))
MAE2 = mean_absolute_error(y_test, y_pred2)

MAE2
from sklearn.tree import DecisionTreeRegressor



regressor3 = DecisionTreeRegressor(random_state = 0)

regressor3.fit(X_train, y_train)



y_pred3 = regressor3.predict(X_test)



MAE3 = mean_absolute_error(y_test, y_pred3)

MAE3
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators': [2, 3, 5, 10, 15, 20, 30, 50, 75, 100, 500, 1000],

              'max_leaf_nodes': [5, 10, 20, 35, 50, 100],

              'random_state': [0]}

grid_search = GridSearchCV(estimator = RandomForestRegressor(),

                           param_grid = parameters,

                           scoring = 'neg_mean_absolute_error',

                           cv = 10,

                           n_jobs = -1)

grid_search = grid_search.fit(X_train, y_train)

print(f"Best MAE: {grid_search.best_score_ * (-1)}")

print(f"Best parameters: {grid_search.best_params_}")
regressor4 = RandomForestRegressor(n_estimators=100, max_leaf_nodes=35, random_state=0)

regressor4.fit(X_train, y_train)



y_pred4 = regressor4.predict(X_test)



MAE4 = mean_absolute_error(y_test, y_pred4)

MAE4
from sklearn.svm import SVR



regressor5 = SVR(kernel = 'rbf')

regressor5.fit(X_train, y_train)



y_pred5 = regressor5.predict(X_test)



MAE5 = mean_absolute_error(y_test, y_pred5)

MAE5
from sklearn.preprocessing import StandardScaler



sc_X = StandardScaler()

sc_y = StandardScaler()

sc_X_train = sc_X.fit_transform(X_train)

sc_y_train = sc_y.fit_transform(y_train.values.reshape(-1,1))

sc_X_test = sc_X.fit_transform(X_test)
parameters = {'C': [1, 5, 10, 20, 50, 100],

              'kernel': ['rbf', 'linear', 'poly'],

              'degree': [2, 3, 4]}

grid_search = GridSearchCV(estimator = SVR(),

                           param_grid = parameters,

                           scoring = 'neg_mean_absolute_error',

                           cv = 10,

                           n_jobs = -1)

grid_search = grid_search.fit(sc_X_train, sc_y_train)

print(f"Best MAE: {grid_search.best_score_ * (-1)}")

print(f"Best parameters: {grid_search.best_params_}")
regressor5 = SVR(kernel = 'rbf', C = 1)

regressor5.fit(sc_X_train, sc_y_train)



y_pred5 = regressor5.predict(sc_X_test)

y_pred5 = sc_y.inverse_transform(y_pred5)



MAE5 = mean_absolute_error(y_test, y_pred5)

MAE5
from xgboost import XGBRegressor
parameters = {'base_score': [0.1, 0.3, 0.5, 0.7, 1, 1.5, 2, 5, 10, 20],

              'learning_rate': [0.001, 0.005, 0.01, 0.03, 0.05, 0.07, 0.1, 0.3, 0.5],

              #'booster': ['gbtree', 'linear', 'dart'],

              'n_estimators': [50, 100, 150, 200, 250, 300, 500, 750, 1000]}

              #'max_depth': [3, 5]}

grid_search = GridSearchCV(estimator = XGBRegressor(),

                           param_grid = parameters,

                           scoring = 'neg_mean_absolute_error',

                           cv = 2,

                           n_jobs = -1)

grid_search = grid_search.fit(X_train, y_train)

print(f"Best MAE: {grid_search.best_score_ * (-1)}")

print(f"Best parameters: {grid_search.best_params_}")
regressor6 = XGBRegressor(learning_rate=0.01, n_estimators=300)

regressor6.fit(X_train, y_train)

y_pred6 = regressor6.predict(X_test)

MAE6 = mean_absolute_error(y_test, y_pred6)

MAE6
summary = {'Multiple Linear': MAE1, 'Polynomial': MAE2, 'Decision Tree': MAE3,

           'Random Forest': MAE4, 'SVR': MAE5, 'XGB': MAE6}



from sklearn.metrics import r2_score



summary_R2 = {'Multiple Linear': r2_score(y_test,y_pred1), 'Polynomial': r2_score(y_test,y_pred2),

             'Decision Tree': r2_score(y_test,y_pred3), 'Random Forest': r2_score(y_test,y_pred4),

             'SVR ': r2_score(y_test,y_pred5), 'XGBoost': r2_score(y_test,y_pred6)}
f = plt.figure(figsize=(15,5))



ax = f.add_subplot(121)

plt.bar(summary.keys(), summary.values(), color='green');

plt.title("Mean absolute error by model (the lower the better)")



ax=f.add_subplot(122)

plt.plot(summary_R2.keys(), summary_R2.values(), color='cyan');

plt.title("R-Squared coefficient by model (the higher the better)")

axes = plt.gca()

axes.set_ylim([0.5,1])

plt.show();
# compare MAE to the average value of the dependent variable

round(100*MAE6/np.mean(y_test),2)

round(100*MAE6/input.charges.mean(),2)
from sklearn.cluster import KMeans

inertia = []

for i in range(1, 11):

    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)

    kmeans.fit(X)

    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia);

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('Inertia')

plt.show();
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 0)

y_kmeans = kmeans.fit_predict(X)
input.head()
XClass = input.iloc[:,[0,1,2,3,4,6]]

yClass = input.iloc[:,5]
LabelEncoder_XClass1 = LabelEncoder()

LabelEncoder_XClass4 = LabelEncoder()

LabelEncoder_yClass = LabelEncoder()

XClass.iloc[:,1] = LabelEncoder_XClass1.fit_transform(XClass.iloc[:,1])

XClass.iloc[:,4] = LabelEncoder_XClass4.fit_transform(XClass.iloc[:,4])

yClass = LabelEncoder_yClass.fit_transform(yClass)
ct_XClass = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [1,4])], remainder='passthrough')

XClass = ct_XClass.fit_transform(XClass)
XClass = XClass[:,[0,1,3,5,6,7]]
XClass_train, XClass_test, yClass_train, yClass_test = train_test_split(XClass, yClass, test_size=0.2, random_state=1)
sc_XClass = StandardScaler()

XClass_train = sc_XClass.fit_transform(XClass_train)

XClass_test = sc_XClass.transform(XClass_test)
from sklearn.metrics import accuracy_score

from sklearn.metrics import cohen_kappa_score
from sklearn.tree import DecisionTreeClassifier



classifier1 = DecisionTreeClassifier(criterion = 'entropy', random_state = 1)

classifier1.fit(XClass_train, yClass_train)

yClass_pred1 = classifier1.predict(XClass_test)
kappa1 = cohen_kappa_score(yClass_test, yClass_pred1)

kappa1

acc1 = accuracy_score(yClass_test, yClass_pred1)

acc1
from sklearn.ensemble import RandomForestClassifier
parameters = {'n_estimators': [50, 100, 150, 200, 250, 300, 500, 750, 1000], 

              'max_leaf_nodes': [5, 10, 20, 30, 50, 100, 300, 600, 800, 1000],

              'criterion': ['gini', 'entropy'],

              'max_depth': [3, 4, 6, 8, 9],

              'random_state': [0]}

grid_search = GridSearchCV(estimator = RandomForestClassifier(),

                           param_grid = parameters,

                           scoring = 'accuracy',

                           cv = 2,

                           n_jobs = -1)

grid_search = grid_search.fit(XClass_train, yClass_train)

print(f"Best Accuracy: {grid_search.best_score_}")

print(f"Best parameters: {grid_search.best_params_}")
classifier2 = RandomForestClassifier(n_estimators = 100)

classifier2.fit(XClass_train, yClass_train)

yClass_pred2 = classifier2.predict(XClass_test)

kappa2 = cohen_kappa_score(yClass_test, yClass_pred2)

kappa2

acc2 = accuracy_score(yClass_test, yClass_pred2)

acc2
from sklearn.svm import SVC
parameters = {'kernel': ['rbf'], 

              'C': [1, 3, 5, 9, 10, 20, 25, 30, 40, 50, 75, 100, 200, 500, 1000],

              'random_state': [0]}

grid_search = GridSearchCV(estimator = SVC(),

                           param_grid = parameters,

                           scoring = 'accuracy',

                           cv = 10,

                           n_jobs = -1)

grid_search = grid_search.fit(XClass_train, yClass_train)

print(f"Best Accuracy: {grid_search.best_score_}")

print(f"Best parameters: {grid_search.best_params_}")
classifier3 = SVC(kernel = 'poly', C = 1, random_state = 0)

classifier3.fit(XClass_train, yClass_train)

yClass_pred3 = classifier3.predict(XClass_test)

kappa3 = cohen_kappa_score(yClass_test, yClass_pred3)

kappa3

acc3 = accuracy_score(yClass_test, yClass_pred3)

acc3
from sklearn.linear_model import LogisticRegression

classifier4 = LogisticRegression(random_state = 0)

classifier4.fit(XClass_train, yClass_train)



# Predicting the Test set results

yClass_pred4 = classifier4.predict(XClass_test)

kappa4 = cohen_kappa_score(yClass_test, yClass_pred4)

kappa4

acc4 = accuracy_score(yClass_test, yClass_pred4)

acc4
from sklearn.naive_bayes import GaussianNB



classifier5 = GaussianNB()

classifier5.fit(XClass_train, yClass_train)

yClass_pred5 = classifier5.predict(XClass_test)

kappa5 = cohen_kappa_score(yClass_test, yClass_pred5)

kappa5

acc5 = accuracy_score(yClass_test, yClass_pred5)

acc5
from sklearn.neighbors import KNeighborsClassifier
parameters = {'n_neighbors': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 

              'p': [1, 2, 3, 5, 10, 20, 30, 50, 70, 90, 120, 150, 200]}

grid_search = GridSearchCV(estimator = KNeighborsClassifier(),

                           param_grid = parameters,

                           scoring = 'accuracy',

                           cv = 10,

                           n_jobs = -1)

grid_search = grid_search.fit(XClass_train, yClass_train)

print(f"Best Accuracy: {grid_search.best_score_}")

print(f"Best parameters: {grid_search.best_params_}")
classifier6 = KNeighborsClassifier(n_neighbors = 4, metric = 'minkowski', p = 120)

classifier6.fit(XClass_train, yClass_train)

yClass_pred6 = classifier6.predict(XClass_test)

kappa6 = cohen_kappa_score(yClass_test, yClass_pred6)

kappa6

acc6 = accuracy_score(yClass_test, yClass_pred6)

acc6
from xgboost import XGBClassifier
classifier7 = XGBClassifier(base_score=0.1, n_estimators=2600, max_depth=2)

classifier7.fit(XClass_train, yClass_train)

yClass_pred7 = classifier7.predict(XClass_test)

kappa7 = cohen_kappa_score(yClass_test, yClass_pred7)

kappa7

acc7 = accuracy_score(yClass_test, yClass_pred7)

acc7
summaryClass = {'Decision Tree': kappa1, 'Random Forest': kappa2, 'Kernel SVM': kappa3,

               'Logistic Regression': kappa4, 'Naive Bayes': kappa5, 'KNN': kappa6, 'XGBoost': kappa7}



classmodels = []

for key in summaryClass.keys():

    classmodels.append(key)
accuracies = [acc1, acc2, acc3, acc4, acc5, acc6, acc7]
plt.figure(figsize=(10,5))

sns.barplot(x=classmodels, y=accuracies);

plt.title('Model Accuracy (the higher the better)')

plt.show();