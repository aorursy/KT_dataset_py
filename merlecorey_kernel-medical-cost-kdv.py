# imports
import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# data import
data = pd.read_csv('../input/insurance/insurance.csv')
data.head()
data.info()
data.describe(include='all')
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(data.corr(), annot=True, linewidths=.1, cmap="coolwarm", square=True)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(data.sex.drop_duplicates()) 
data.sex = le.transform(data.sex)
le.fit(data.smoker.drop_duplicates()) 
data.smoker = le.transform(data.smoker)
le.fit(data.region.drop_duplicates()) 
data.region = le.transform(data.region)

data.head()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(data.corr(), annot=True, linewidths=.1, cmap="coolwarm", square=True)
_ = data.hist(data.columns, figsize=(12, 12), bins=15)
data.bmi.median()
_ = sns.boxplot(data.charges)
fig, axes = plt.subplots(1, 2, figsize=(12,6))
data[data['smoker']==1]['charges'].hist(bins=20, alpha=0.5, label='smokers', ax=axes[0])
data[data['smoker']==0]['charges'].hist(bins=20, alpha=0.5, label='NON-smokers', ax=axes[1])
### на одном графике
fig, ax = plt.subplots(figsize=(8,6))
plt.hist([data[data['smoker']==1]['charges'],
          data[data['smoker']==0]['charges']
         ], 
         bins=20, alpha=0.5, label=['smokers','NON-smokers'])
ax.set_ylabel("Count")
ax.legend(loc='upper right')
fig, ax = plt.subplots(figsize=(8,6))
sns.boxplot(x="sex", y="charges", hue="smoker", data=data)
# импортируем модельки и прочее
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression, SGDRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn import utils
from sklearn.metrics import r2_score,mean_squared_error,f1_score,roc_auc_score 
X = data.drop(columns=['charges'])
y = data['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=146)
regressor = LinearRegression()

regressor.fit(X_train, y_train)
test_predictions = regressor.predict(X_test)

print(regressor.score(X_test,y_test))
print('test mse: ', mean_squared_error(y_test, test_predictions))
print('r2: ', r2_score(y_test, test_predictions))
plt.figure(figsize=(10, 6))
plt.bar(X.columns, regressor.coef_)
# среднее и стандартное отклонение
mean = data.mean(axis=0)
std = data.std(axis=0)
# 0 мат ожидание и 1 дисперсию
data_s = (data - mean)/std
X = data_s.drop(columns=['charges'])
y = data_s['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=146)
regressor = LinearRegression()

regressor.fit(X_train, y_train)
test_predictions = regressor.predict(X_test)

print('r2: ', regressor.score(X_test,y_test))
print('test mse: ', mean_squared_error(y_test, test_predictions))
#print('r2: ', r2_score(y_test, test_predictions))
_ = sns.catplot(data=data, y="charges", orient="h", kind="box", height=4, aspect=3)
p99= data_s.charges.quantile(0.99)
print(p99, data.charges.quantile(0.99))
data1 = data_s[data_s.charges<=p99]
X = data1.drop(columns=['charges'])
y = data1['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=146)
regressor = LinearRegression()

regressor.fit(X_train, y_train)
test_predictions = regressor.predict(X_test)

print('r2: ', regressor.score(X_test,y_test))
print('mse: ', mean_squared_error(y_test, test_predictions))
plt.figure(figsize=(10, 6))
plt.bar(X.columns, regressor.coef_)
def get_cv_scores(model):
    scores = cross_val_score(model,
                             X_train,
                             y_train,
                             cv=10,
                             scoring='r2')
    
    print('CV Mean: ', np.mean(scores))
    print('STD: ', np.std(scores))
# get cross val scores
get_cv_scores(regressor)
print('r2: ', regressor.score(X_test,y_test))
f = plt.figure(figsize=(16,6))
ax = f.add_subplot(122)
sns.distplot((y_test - test_predictions),ax=ax,color='b')
ax.axvline((y_test - test_predictions).mean(),color='k',linestyle='--')
ax.set_title('Check for Residual normality & mean: \n Residual eror');
alphas = np.linspace(1, 1000, 1000)

weights = np.empty((len(X.columns), 0))
for alpha in alphas:
    ridge_regressor = Ridge(alpha)
    ridge_regressor.fit(X_train, y_train)
    weights = np.hstack((weights, ridge_regressor.coef_.reshape(-1, 1)))
plt.figure(figsize=(8,6))
plt.plot(alphas, weights.T, linewidth=3)
plt.xlabel('regularization coef')
plt.ylabel('weight value')
plt.legend(X.columns)
plt.show()
ridge_regressor = Ridge(200)
ridge_regressor.fit(X_train, y_train)
# get cross val scores
get_cv_scores(ridge_regressor)
print('r2: ', ridge_regressor.score(X_test,y_test))
alphas = np.linspace(0.001, 1 , 100)

plt.figure(figsize=(10, 5))
weights = np.empty((len(X.columns), 0))
for alpha in alphas:
    lasso_regressor = Lasso(alpha)
    lasso_regressor.fit(X_train, y_train)
    weights = np.hstack((weights, lasso_regressor.coef_.reshape(-1, 1)))
plt.figure(figsize=(8,6))
plt.plot(alphas, weights.T, linewidth=3)
plt.xlabel('regularization coef')
plt.ylabel('weight value')
plt.legend(X.columns)
plt.grid()
plt.show()
lasso_regressor = Lasso(0.01)
lasso_regressor.fit(X_train, y_train)
# get cross val scores
get_cv_scores(lasso_regressor)
print('r2: ', lasso_regressor.score(X_test,y_test))
poly = PolynomialFeatures(2)
X_poly = poly.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.3, random_state=146)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
# get cross val scores
get_cv_scores(regressor)
print('r2: ', regressor.score(X_test,y_test))
sgd = SGDRegressor()
sgd_params = {
              'loss':['squared_loss','huber','epsilon_insensitive','squared_epsilon_insensitive'], 
              'penalty': ['l1','l2','elasticnet'], 
              'learning_rate':['constant','optimal','invscaling','adaptive'], 
              'alpha': [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
             } 
gs = GridSearchCV(sgd, sgd_params, cv=5)
gs.fit(X_train, y_train)
print('best params: ', gs.best_params_)
print('best score on CV: ', gs.best_score_)
print('r2: ', gs.score(X_test, y_test))


