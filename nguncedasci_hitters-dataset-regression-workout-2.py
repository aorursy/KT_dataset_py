from warnings import filterwarnings
filterwarnings('ignore')


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt


from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error,r2_score


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.linear_model import RidgeCV, LassoCV,ElasticNetCV



Hitters=pd.read_csv("../input/hitters/Hitters.csv")
df=Hitters.copy()
df.head()
df.info()
df.isnull().sum().sum()
df[df.isnull().any(axis=1)].head(2)
df["Salary"].fillna(df["Salary"].median(), inplace=True)
df[df.isnull().any(axis=1)]
df = pd.get_dummies(df, columns = ['League', 'Division', 'NewLeague'], drop_first = True)
df.head(2)
clf= LocalOutlierFactor(n_neighbors = 20, contamination = 0.1)
clf.fit_predict(df)[0:10]
df_scores=clf.negative_outlier_factor_
np.sort(df_scores)[0:20]
sns.boxplot(df_scores);
threshold=np.sort(df_scores)[10]
threshold
df=df.loc[df_scores > threshold]
df.shape
# Regression

y=df["Salary"]
X=df.drop("Salary", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.20, 
                                                    random_state=46)
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)
y_pred=reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
#Ridge Regression

ridge_model=Ridge().fit(X_train,y_train)
y_pred= ridge_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
# Ridge_tuned(alpha1)
alphas1 = np.random.randint(0,1000,100)
alphas2 = 10**np.linspace(10,-2,100)*0.5
alphas3 = np.linspace(0,1,1000)
ridge_cv = RidgeCV(alphas = alphas1, scoring = "neg_mean_squared_error", cv = 10, normalize = True)
ridge_cv.fit(X_train,y_train)
print(ridge_cv.alpha_)
ridge_tuned=Ridge(alpha=ridge_cv.alpha_).fit(X_train,y_train)
y_pred=ridge_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
# Ridge_tuned(alpha2)
alphas1 = np.random.randint(0,1000,100)
alphas2 = 10**np.linspace(10,-2,100)*0.5
alphas3 = np.linspace(0,1,1000)
ridge_cv = RidgeCV(alphas = alphas2, scoring = "neg_mean_squared_error", cv = 10, normalize = True)
ridge_cv.fit(X_train,y_train)
print(ridge_cv.alpha_)
ridge_tuned=Ridge(alpha=ridge_cv.alpha_).fit(X_train,y_train)
y_pred=ridge_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
# Ridge_tuned(alpha3)
alphas1 = np.random.randint(0,1000,100)
alphas2 = 10**np.linspace(10,-2,100)*0.5
alphas3 = np.linspace(0,1,1000)
ridge_cv = RidgeCV(alphas = alphas3, scoring = "neg_mean_squared_error", cv = 10, normalize = True)
ridge_cv.fit(X_train,y_train)
print(ridge_cv.alpha_)
ridge_tuned=Ridge(alpha=ridge_cv.alpha_).fit(X_train,y_train)
y_pred=ridge_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
# Ridge_tuned(best alpha)
alphas4 = np.linspace(0,0.001,2)
ridge_cv = RidgeCV(alphas = alphas4, scoring = "neg_mean_squared_error", cv = 10, normalize = True)
ridge_cv.fit(X_train,y_train)
print(ridge_cv.alpha_)
ridge_tuned=Ridge(alpha=ridge_cv.alpha_).fit(X_train,y_train)
y_pred=ridge_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
#Lasso Regression
lasso_model = Lasso().fit(X_train, y_train)
y_pred=lasso_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
#Lasso Tuned(alpha1)
alphas1 = np.random.randint(0,1000,100)
alphas2 = 10**np.linspace(10,-2,100)*0.5
alphas3 = np.linspace(0,1,1000)
lasso_cv_model = LassoCV(alphas = alphas1, cv = 10).fit(X_train, y_train)
print(lasso_cv_model.alpha_)
lasso_tuned = Lasso(alpha = lasso_cv_model.alpha_).fit(X_train, y_train)
y_pred = lasso_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_pred,y_test))
#Lasso Tuned(alpha2)
lasso_cv_model = LassoCV(alphas = alphas2, cv = 10).fit(X_train, y_train)
print(lasso_cv_model.alpha_)
lasso_tuned = Lasso(alpha = lasso_cv_model.alpha_).fit(X_train, y_train)
y_pred = lasso_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_pred,y_test))
#Lasso Tuned(alpha3)
lasso_cv_model = LassoCV(alphas = alphas3, cv = 10).fit(X_train, y_train)
print(lasso_cv_model.alpha_)
lasso_tuned = Lasso(alpha = lasso_cv_model.alpha_).fit(X_train, y_train)
y_pred = lasso_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_pred,y_test))
# Lasso_tuned(best alpha)
alphas4 = np.linspace(0,1,500)
lasso_cv_model = LassoCV(alphas = alphas4, cv = 10).fit(X_train, y_train)
print(lasso_cv_model.alpha_)
lasso_tuned = Lasso(alpha = lasso_cv_model.alpha_).fit(X_train, y_train)
y_pred = lasso_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_pred,y_test))
# Elastic Net Regression
enet_model = ElasticNet().fit(X_train, y_train)
y_pred = enet_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
#Elastic Net(alpha1)
alphas1 = np.random.randint(0,1000,100)
alphas2 = 10**np.linspace(10,-2,100)*0.5
alphas3 = np.linspace(0,1,1000)
enet_cv_model = ElasticNetCV(alphas = alphas1, cv = 10).fit(X_train, y_train)
print(enet_cv_model.alpha_)
enet_tuned = ElasticNet(alpha = enet_cv_model.alpha_).fit(X_train, y_train)
y_pred = enet_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
#Elastic Net(alpha2)
enet_cv_model = ElasticNetCV(alphas = alphas2, cv = 10).fit(X_train, y_train)
print(enet_cv_model.alpha_)
enet_tuned = ElasticNet(alpha = enet_cv_model.alpha_).fit(X_train, y_train)
y_pred = enet_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
#Elastic Net(alpha3)
enet_cv_model = ElasticNetCV(alphas = alphas3, cv = 10).fit(X_train, y_train)
print(enet_cv_model.alpha_)
enet_tuned = ElasticNet(alpha = enet_cv_model.alpha_).fit(X_train, y_train)
y_pred = enet_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
enet_params = {"l1_ratio": [0,0.01,0.05,0.1,0.2,0.4,0.5,0.6,0.8,1],
               "alpha":[0.1,0.01,0.001,0.2,0.3,0.5,0.8,0.9,1,2,5,7,10,13,20,45,87,99,100,110]}
enet_model = ElasticNet().fit(X, y)
from sklearn.model_selection import GridSearchCV
gs_cv_enet = GridSearchCV(enet_model, enet_params, cv = 10).fit(X, y)
print(gs_cv_enet.best_params_)
enet_tuned = ElasticNet(**gs_cv_enet.best_params_).fit(X_train, y_train)
y_pred = enet_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
