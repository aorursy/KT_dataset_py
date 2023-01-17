import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 60)

pd.set_option('display.max_colwidth', -1)

data = pd.read_csv("../input/students_data.csv")
data.head()
data.info()

# For numeric features
data.describe()
# For categorical features
data.describe(include=[object])
# Define the numerical and categotical features
categorical_features = [c for c in data.columns if data[c].dtype.name == 'object']
numerical_features = [c for c in data.columns if data[c].dtype.name != 'object']
data_describe = data.describe(include=[object])

binary_columns = [c for c in categorical_features if data_describe[c]['unique'] == 2]
nonbinary_columns = [c for c in categorical_features if data_describe[c]['unique'] > 2]
binary_columns, nonbinary_columns

# Binarization of data with two values 
# 0 - Por, 1 - Math
for c in binary_columns:
    top = data_describe[c]['top']
    top_items = data[c] == top
    data.at[top_items, c] = 0 
    data.at[np.logical_not(top_items), c] = 1

data
    
# Binarization of data with the value bigger then 2   
data_nonbinary = pd.get_dummies(data[nonbinary_columns])

data = data.drop(('Mjob'), axis=1)
data = data.drop(('Fjob'), axis=1)
data = data.drop(('reason'), axis=1)
data = data.drop(('guardian'), axis=1)
data = pd.concat((data, data_nonbinary), axis=1)
data = pd.DataFrame(data, dtype = int)

data.head()
# Plot of marks for Portogese 
data[data['Subject'] == 0]['G1'].plot(kind='kde', c='red')
data[data['Subject'] == 0]['G2'].plot(kind='kde', c='blue')
data[data['Subject'] == 0]['G3'].plot(kind='kde', c='black')

plt.show()
# Plot of marks for Math
data[data['Subject'] == 1]['G1'].plot(kind='kde', c='red')
data[data['Subject'] == 1]['G2'].plot(kind='kde', c='blue')
data[data['Subject'] == 1]['G3'].plot(kind='kde', c='black')

plt.show()
colnames = ['G3', 'sex', 'traveltime', 'schoolsup', 'famsup', 'paid', 'internet', 'romantic', 'Dalc', 'Walc']
data[colnames].corr()
import seaborn as sns

sns.heatmap(data[colnames].corr())
plt.show()
print('Reason - near home:', data[data['reason_home'] == 1]['G3'].mean())
print('Reason - good courses:', data[data['reason_course'] == 1]['G3'].mean())
print('Reason - good reputation:', data[data['reason_reputation'] == 1]['G3'].mean())
print('Reason - other reasons:', data[data['reason_other'] == 1]['G3'].mean())
cols = ['G2', 'G3']
X = data.drop(cols, axis=1)
y = data['G3'][:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=11)
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

y_pred = linear_model.predict(X_test)

error = r2_score(y_test, y_pred)
print('Coef of determination:', error)
scores = cross_val_score(linear_model, X_train, y_train, cv=5,
                        scoring='r2')
print('Average value on cross-validation:', scores.mean())
params = {'alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
lasso_model = Lasso()
clf = GridSearchCV(lasso_model, params)
clf.fit(X_train, y_train)
# Получим наши оптимально подобранные параметры
clf.best_estimator_
scores = cross_val_score(clf, X_train, y_train, cv=5,
                        scoring='r2')
print('Average value on cross-validation:', scores.mean())
lassp_model = Lasso(alpha=0.1, max_iter=1000, selection='cyclic', tol=0.0001)
lasso_model.fit(X_train, y_train)
y_pred = lasso_model.predict(X_test)

error = r2_score(y_test, y_pred)
print('Coef of determination: ', error)
k = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.3, 1.4, 1.5, 1.6, 10]
params = {'alpha': k, 'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']}
ridge_model = Ridge()
clf = GridSearchCV(ridge_model, params)
clf.fit(X_train, y_train)
clf.best_estimator_
ridge_model = Ridge(alpha=10, solver='svd', tol=0.001)
ridge_model.fit(X_train, y_train)
y_pred = ridge_model.predict(X_test)

error = r2_score(y_test, y_pred)
print('Coef of determination: ', error)
scores = cross_val_score(ridge_model, X_train, y_train, cv=5,
                        scoring='r2')
print('Average value on cross-validation:', scores.mean())
params = {'n_neighbors': range(1,20), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
knn_model = KNeighborsRegressor()
clf = GridSearchCV(knn_model, params)
clf.fit(X_train, y_train)
clf.best_estimator_
knn_model = KNeighborsRegressor(n_neighbors=18)
knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)

error = r2_score(y_test, y_pred)
print('Coef of determination: ', abs(error))
scores = cross_val_score(knn_model, X_train, y_train, cv=5,
                        scoring='r2')
print('Average value on cross-validation:', abs(scores).mean())
cols = ['G1', 'G2', 'G3']
X = data.drop(cols, axis=1)
y = data['G3'][:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=11)
linear_without_g1_model = LinearRegression()
linear_without_g1_model.fit(X_train, y_train)

y_pred = linear_without_g1_model.predict(X_test)

error = r2_score(y_test, y_pred)
print('Coef of determination:', error)
scores = cross_val_score(linear_without_g1_model, X_train, y_train, cv=5,
                        scoring='r2')
print('Average value on cross-validation:', scores.mean())
params = {'alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4]}
lasso_withou_g1_model = Lasso()
clf = GridSearchCV(lasso_withou_g1_model, params)
clf.fit(X_train, y_train)
# Получим наши оптимально подобранные параметры
clf.best_estimator_
scores = cross_val_score(clf, X_train, y_train, cv=5,
                        scoring='r2')
print('Average value on cross-validation:', scores.mean())
lasso_withou_g1_model = Lasso(alpha=0.1, max_iter=1000, selection='cyclic', tol=0.0001)
lasso_withou_g1_model.fit(X_train, y_train)
y_pred = lasso_withou_g1_model.predict(X_test)

error = r2_score(y_test, y_pred)
print('Coef of determination: ', error)
k = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.3, 1.4, 1.5, 1.6, 10]
params = {'alpha': k, 'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']}
ridge_without_g1_model = Ridge()
clf = GridSearchCV(ridge_without_g1_model, params)
clf.fit(X_train, y_train)
clf.best_estimator_
ridge_without_g1_model = Ridge(alpha=10, solver='auto', tol=0.001)
ridge_without_g1_model.fit(X_train, y_train)
y_pred = ridge_without_g1_model.predict(X_test)

error = r2_score(y_test, y_pred)
print('Coef of determination: ', error)
scores = cross_val_score(ridge_without_g1_model, X_train, y_train, cv=5,
                        scoring='r2')
print('Average value on cross-validation:', scores.mean())
params = {'n_neighbors': range(1,20), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
knn_without_g1_model = KNeighborsRegressor()
clf = GridSearchCV(knn_without_g1_model, params)
clf.fit(X_train, y_train)
clf.best_estimator_
knn_without_g1_model = KNeighborsRegressor(n_neighbors=18)
knn_without_g1_model.fit(X_train, y_train)
y_pred = knn_without_g1_model.predict(X_test)

error = r2_score(y_test, y_pred)
print('Coef of determination: ', abs(error))
scores = cross_val_score(knn_without_g1_model, X_train, y_train, cv=5,
                        scoring='r2')
print('Average value on cross-validation:', abs(scores.mean()))
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score, precision_score, confusion_matrix, f1_score, accuracy_score, classification_report
cols = ['G1', 'G2', 'G3']
X = data.drop(cols, axis=1)
y = data['G3'][:]
y = pd.Series(list(map(lambda x: 1 if x >= 8 else -1, y)))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=11)
model = KNeighborsClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

model.score(X_test, y_test)

scores = cross_val_score(model, X_train, y_train, cv=5,
                        scoring='f1')
print('Average value on cross-validation without right parametrs:', scores.mean())
params = {'n_neighbors': range(1,30), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
model = KNeighborsClassifier()
clf = GridSearchCV(model, params)
clf.fit(X_train, y_train)
clf.best_estimator_
acc = []
k = [i for i in range(2,50)]
for i in k:
    model = KNeighborsClassifier(n_neighbors=9, leaf_size=i)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    ac = f1_score(y_test, y_pred)
    acc.append(ac)
plt.plot(k, acc)
plt.xlabel('k')
plt.ylabel('f1_score')
plt.show()
model = KNeighborsClassifier(n_neighbors=9)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

model.score(X_test, y_test)

scores = cross_val_score(model, X_train, y_train, cv=5,
                        scoring='f1')
print('Average value on cross-validation with right parametrs:', scores.mean())
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred

classification_report(y_test, y_pred)

scores = cross_val_score(model, X_train, y_train, cv=5,
                        scoring='f1')
print('Average value on cross-validation:',scores.mean())
params = {'C': range(1,10), 'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
model = LogisticRegression()
clf = GridSearchCV(model, params)
clf.fit(X_train, y_train)
clf.best_estimator_
model = LogisticRegression(C=1, solver='newton-cg')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred

classification_report(y_test, y_pred)

scores = cross_val_score(model, X_train, y_train, cv=5,
                        scoring='f1')
print('Average value on cross-validation:',scores.mean())
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred

classification_report(y_test, y_pred)

scores = cross_val_score(model, X_train, y_train, cv=5,
                        scoring='f1')
print('Average value on cross-validation:',scores.mean())
params = {'min_samples_split': range(2,40), 'max_depth': range(2,20), 'criterion': ['gini', 'entropy']}
model = DecisionTreeClassifier()
clf = GridSearchCV(model, params)
clf.fit(X_train, y_train)
clf.best_estimator_

model = DecisionTreeClassifier(criterion='entropy', max_depth=3 , min_samples_split=2)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred

# classification_report(y_test, y_pred)

scores = cross_val_score(model, X_train, y_train, cv=5,
                        scoring='f1')
print('Average value on cross-validation:',scores.mean())
f1 = []
k = [i for i in range(2,40)]
for i in k:
    model = DecisionTreeClassifier(criterion='entropy', max_depth=3 , min_samples_split=2,
                                  min_samples_leaf=i)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f = f1_score(y_test, y_pred)
    f1.append(f)
plt.plot(k, f1)
plt.xlabel('k')
plt.ylabel('f1_score')
plt.show()
model = DecisionTreeClassifier(criterion='entropy', max_depth=3 , min_samples_split=2,
                              min_samples_leaf=8)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred

# classification_report(y_test, y_pred)

scores = cross_val_score(model, X_train, y_train, cv=5,
                        scoring='f1')
print('Average value on cross-validation:',scores.mean())
cols = ['G1', 'G2', 'G3']
X = data.drop(cols, axis=1)
y = data['G3'][:]

new_y = []
for el in y:
    if el >= 18 and el <= 20:
        el = 5
        
    elif el >= 14 and el <= 17:
        el = 4
        
    elif el >= 8 and el <= 13:
        el = 3
        
    else:
        el = 2
    new_y.append(el)

y = pd.Series(new_y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=11)
model = KNeighborsClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

model.score(X_test, y_test)

scores = cross_val_score(model, X_train, y_train, cv=5,
                        scoring='accuracy')
print('Average value on cross-validation without right parametrs:', scores.mean())
params = {'n_neighbors': range(1,30), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
model = KNeighborsClassifier()
clf = GridSearchCV(model, params)
clf.fit(X_train, y_train)
clf.best_estimator_
acc = []
k = [i for i in range(2,50)]
for i in k:
    model = KNeighborsClassifier(n_neighbors=24, leaf_size=i)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    ac = accuracy_score(y_test, y_pred)
    acc.append(ac)
plt.plot(k, acc)
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.show()
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred

classification_report(y_test, y_pred)

scores = cross_val_score(model, X_train, y_train, cv=5,
                        scoring='accuracy')
print('Average value on cross-validation:', scores.mean())
params = {'min_samples_split': range(2,40), 'max_depth': range(2,20), 'criterion': ['gini', 'entropy']}
model = DecisionTreeClassifier()
clf = GridSearchCV(model, params)
clf.fit(X_train, y_train)
clf.best_estimator_

f1 = []
k = [i for i in range(2,40)]
for i in k:
    model = DecisionTreeClassifier(criterion='gini', max_depth=4, min_samples_split=38,
                                  min_samples_leaf=i)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f = accuracy_score(y_test, y_pred)
    f1.append(f)
plt.plot(k, f1)
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.show()
model = DecisionTreeClassifier(criterion='gini', max_depth=4, min_samples_split=38,
                                  min_samples_leaf=12)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred

classification_report(y_test, y_pred)

scores = cross_val_score(model, X_train, y_train, cv=5,
                        scoring='accuracy')
print('Average value on cross-validation:', scores.mean())