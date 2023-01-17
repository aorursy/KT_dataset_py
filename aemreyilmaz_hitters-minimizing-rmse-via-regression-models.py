import warnings

warnings.simplefilter(action='ignore')



import pandas as pd

import numpy as np

from sklearn.neighbors import LocalOutlierFactor

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, ElasticNet, ElasticNetCV, Lasso, LassoCV

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import LabelEncoder
hitters = pd.read_csv('../input/hitters/Hitters.csv')

df = hitters.copy()
df.head()
df.describe().T
df.shape
df.isnull().sum()
df['Salary'].describe()
df['Year_interval'] = pd.cut(x=df['Years'], bins=[0, 3, 6, 10, 15, 19, 24])
df.head()
df.groupby(['League','Division', 'Year_interval']).agg({'Salary':'mean'})
df['Salary'] = df.groupby(['League', 'Division', 'Year_interval'])['Salary'].transform(lambda x: x.fillna(x.mean()))
df.head()
df.isnull().sum()
df.drop('Year_interval', axis = 1, inplace = True)
le = LabelEncoder()

df['League'] = le.fit_transform(df['League'])

df['Division'] = le.fit_transform(df['Division'])

df['NewLeague'] = le.fit_transform(df['NewLeague'])
df.head()
X = df.drop('Salary', axis = 1)

y = df[['Salary']]
X_train, X_test, y_train, y_test = train_test_split(X, 

                                                    y, 

                                                    test_size = 0.20, random_state = 46)
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)
y_pred_reg = reg_model.predict(X_test)

np.sqrt(mean_squared_error(y_test, y_pred_reg))
import seaborn as sns

sns.boxplot(df['Salary']);
df.sort_values('Salary', ascending = False)
df.drop(df.iloc[217:218,:].index, inplace = True)
df.drop(df.iloc[294:295, :].index, inplace = True)
X = df.drop('Salary', axis = 1)

y = df[['Salary']]

X_train, X_test, y_train, y_test = train_test_split(X, 

                                                    y, 

                                                    test_size = 0.20, random_state = 46)
reg_model.fit(X_train, y_train)
y_pred_reg = reg_model.predict(X_test)

np.sqrt(mean_squared_error(y_test, y_pred_reg))
enet = ElasticNet().fit(X_train, y_train)

y_pred_enet = enet.predict(X_test)

np.sqrt(mean_squared_error(y_test, y_pred_enet))
lasso_model = Lasso().fit(X_train, y_train)

y_pred_lass = lasso_model.predict(X_test)

np.sqrt(mean_squared_error(y_test, y_pred_lass))
ridge_model = Ridge().fit(X_train, y_train)

y_pred_ridge = enet.predict(X_test)

np.sqrt(mean_squared_error(y_test, y_pred_ridge))
alphas1 = np.linspace(0,1,1000)

alphas2 = 10**np.linspace(10,-2,100)*0.5

alphas3 = np.random.randint(0,1000,100)
enet_cv = ElasticNetCV(alphas = alphas3, cv = 10).fit(X_train, y_train)
enet_cv.alpha_
enet_tuned = ElasticNet(alpha = enet_cv.alpha_, l1_ratio = 0.999).fit(X_train, y_train)
y_pred_enett = enet_tuned.predict(X_test)

np.sqrt(mean_squared_error(y_test, y_pred_enett))
lasso_cv = LassoCV(alphas = alphas3, cv = 10).fit(X_train, y_train)

lasso_cv.alpha_
lasso_tuned = Lasso(alpha = lasso_cv.alpha_).fit(X_train, y_train)

y_pred_lassot = lasso_tuned.predict(X_test)

np.sqrt(mean_squared_error(y_test, y_pred_lassot))
ridge_cv = RidgeCV(alphas = alphas3, cv = 10).fit(X_train, y_train)

ridge_cv.alpha_
ridge_tuned = Ridge(alpha = ridge_cv.alpha_).fit(X_train, y_train)

y_pred_ridget = ridge_tuned.predict(X_test)

np.sqrt(mean_squared_error(y_test, y_pred_ridget))
clf = LocalOutlierFactor(n_neighbors = 20, contamination = 0.1)
clf.fit_predict(df)
df_scores = clf.negative_outlier_factor_
np.sort(df_scores)[:30]
threshold = np.sort(df_scores)[4]
threshold
df.drop(df[df_scores<threshold].index, inplace = True)
X = df.drop('Salary', axis = 1)

y = df[['Salary']]

X_train, X_test, y_train, y_test = train_test_split(X, 

                                                    y, 

                                                    test_size = 0.20, random_state = 46)
reg_model.fit(X_train, y_train)

y_pred_reg = reg_model.predict(X_test)

np.sqrt(mean_squared_error(y_test, y_pred_reg))
enet = ElasticNet().fit(X_train, y_train)

y_pred_enet = enet.predict(X_test)

np.sqrt(mean_squared_error(y_test, y_pred_enet))
lasso_model = Lasso().fit(X_train, y_train)

y_pred_lass = lasso_model.predict(X_test)

np.sqrt(mean_squared_error(y_test, y_pred_lass))
ridge_model = Ridge().fit(X_train, y_train)

y_pred_ridge = ridge_model.predict(X_test)

np.sqrt(mean_squared_error(y_test, y_pred_ridge))
enet_cv = ElasticNetCV(alphas = alphas3, cv = 10).fit(X_train, y_train)

enet_cv.alpha_
enet_tuned = ElasticNet(alpha = enet_cv.alpha_, l1_ratio = 0.01).fit(X_train, y_train)

y_pred_enett = enet_tuned.predict(X_test)

np.sqrt(mean_squared_error(y_test, y_pred_enett))
ridge_cv = RidgeCV(alphas = alphas1, cv = 10).fit(X_train, y_train)

ridge_cv.alpha_
ridge_tuned = Ridge(alpha = ridge_cv.alpha_).fit(X_train, y_train)

y_pred_ridget = ridge_tuned.predict(X_test)

np.sqrt(mean_squared_error(y_test, y_pred_ridget))
lasso_cv = LassoCV(alphas = alphas1, cv = 10).fit(X_train, y_train)

lasso_cv.alpha_
lasso_tuned = Lasso(alpha = lasso_cv.alpha_).fit(X_train, y_train)

y_pred_lassot = lasso_tuned.predict(X_test)

np.sqrt(mean_squared_error(y_test, y_pred_lassot))
enet_tuned = ElasticNet(alpha = 11250, l1_ratio = 0.7).fit(X_train, y_train)

y_pred_enett = enet_tuned.predict(X_test)

np.sqrt(mean_squared_error(y_test, y_pred_enett))