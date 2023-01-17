import numpy as np # matris işlemleri

import pandas as pd # veri manüpilasyonları

import seaborn as sns # görselleştirme işlemleri

import matplotlib.pyplot as plt # görselleştirme işlemleri

import missingno as msno # eksik değerlerin görselleştirilmesi

from sklearn import preprocessing # değişken işlemleri

from sklearn.neighbors import LocalOutlierFactor # çok değişkenli aykırı gözlem incelemesi

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict # doğrulama işlemi

from scipy.stats import shapiro # hipotez testleri

from scipy.stats import stats # istatistiksel analizler

import statsmodels.api as sm # model doğrusallığının incelenmesi

from sklearn.linear_model import LinearRegression # Doğrusal Regresyon

from sklearn.linear_model import Ridge, RidgeCV # Ridge Regresyon

from sklearn.linear_model import Lasso, LassoCV # Lasso Regresyon

from sklearn.linear_model import ElasticNet, ElasticNetCV # ElasticNet Regresyon 

from sklearn.metrics import mean_squared_error, r2_score # rmse ve r2 değerini kullanarak değerlendirme

from sklearn.model_selection import GridSearchCV # ElasticNet Tuning
# hata göstergelerinin gizlenmesi

from warnings import filterwarnings

filterwarnings('ignore')
# veri setinin elde edilmesi

adv = pd.read_csv("../input/advertisingcsv/Advertising.csv", usecols=[1,2,3,4])

df = adv.copy()

df.head()
# değişkenlere genel bakış

df.info()
# tanımlayıcı istatistiklerin incelenmesi

df.describe().T
# TV değişkeninin yoğunluk grafiği

sns.kdeplot(df.TV, shade=True);
# Radio değişkeninin yoğunluk grafiği

sns.kdeplot(df.Radio, shade=True);
# Newspaper değişkeninin yoğunluk grafiği

sns.kdeplot(df.Newspaper, shade=True);
# Sales değişkeninin yoğunluk grafiği

sns.kdeplot(df.Sales, shade=True);
# normallik kontrolü



pvalue_TV = shapiro(df["TV"])[1]

print("p-value: %.4f" % pvalue_TV)



pvalue_Sales = shapiro(df["Sales"])[1]

print("p-value: %.4f" % pvalue_Sales)



pvalue_Sales = shapiro(df["Newspaper"])[1]

print("p-value: %.4f" % pvalue_Sales)



pvalue_Sales = shapiro(df["Radio"])[1]

print("p-value: %.4f" % pvalue_Sales)
cor = df.corr()

sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)

plt.show();
# korelasyon anlamlılığı



pvalue_corr = stats.spearmanr(df["TV"], df["Sales"])[1]

print("p-value: %.4f" % pvalue_corr)
# Boxplot kullanılarak TV değişkeninin aykırı değerlerinin incelenmesi

sns.boxplot(x=df["TV"], orient="v");
# Boxplot kullanılarak Radio değişkeninin aykırı değerlerinin incelenmesi

sns.boxplot(x=df["Radio"], orient="v");
# Boxplot kullanılarak Newspaper değişkeninin aykırı değerlerinin incelenmesi

sns.boxplot(x=df["Newspaper"], orient="v");
# Boxplot kullanılarak Sales değişkeninin aykırı değerlerinin incelenmesi

sns.boxplot(x=df["Sales"], orient="v");
# üst sınır değerinin bulunması ve df içerisinde gösterilmesi

Q1 = df.Newspaper.quantile(0.25)

Q3 = df.Newspaper.quantile(0.75)

IQR = Q3 - Q1

ust_sinir = Q3 + 1.5 * IQR

df[df["Newspaper"] > ust_sinir]
news_aykiri = df["Newspaper"] > ust_sinir

df.loc[news_aykiri, "Newspaper"] = ust_sinir

df[news_aykiri]
# Newspaper değişkeninin aykırı değer doğruluğu

sns.boxplot(x=df["Newspaper"], orient="v");
y = df["Sales"]

X = df.drop("Sales", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.20, 

                                                    random_state=46)



X_train_cons = sm.add_constant(X_train)

lm = sm.OLS(y_train, X_train_cons)

model = lm.fit()

print("p-value %.4f" % model.f_pvalue)
y = df["Sales"]

X = df.drop("Sales", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.20, 

                                                    random_state=46)

reg_model = LinearRegression()

reg_model.fit(X_train, y_train)

y_pred = reg_model.predict(X_test)

np.sqrt(mean_squared_error(y_test, y_pred))
cd_reg_rmse = np.sqrt(-cross_val_score(reg_model, 

                         X_test, 

                         y_test, 

                         cv=10, 

                         scoring="neg_mean_squared_error").mean())



cd_reg_rmse
y = df["Sales"]

X = df.drop("Sales", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.20, 

                                                    random_state=46)



ridge_model = Ridge().fit(X_train,y_train)

y_pred = ridge_model.predict(X_test)

np.sqrt(mean_squared_error(y_test, y_pred))
np.random.seed(46)

alpha_ = 10 ** np.linspace(10, -2, 100) * 0.5



ridge_cv = RidgeCV(alphas = alpha_, scoring = "neg_mean_squared_error", cv = 10, normalize = True)

ridge_cv.fit(X_train,y_train)

ridge_tuned=Ridge(alpha=ridge_cv.alpha_).fit(X_train,y_train)

y_pred=ridge_tuned.predict(X_test)

ridge_reg_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

ridge_reg_rmse
y = df["Sales"]

X = df.drop("Sales", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.20, 

                                                    random_state=46)

lasso_model = Lasso().fit(X_train, y_train)

y_pred = lasso_model.predict(X_test)

np.sqrt(mean_squared_error(y_test, y_pred))
np.random.seed(46)

alphas_ = 10 ** np.linspace(10, -2, 100) * 0.5



lasso_cv_model = LassoCV(alphas = alphas_, cv = 10).fit(X_train, y_train)

lasso_tuned = Lasso(alpha = lasso_cv_model.alpha_).fit(X_train, y_train)

y_pred = lasso_tuned.predict(X_test)

lasso_reg_rmse = np.sqrt(mean_squared_error(y_pred,y_test))

lasso_reg_rmse
y = df["Sales"]

X = df.drop("Sales", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.20, 

                                                    random_state=46)



enet_model = ElasticNet().fit(X_train, y_train)

y_pred = enet_model.predict(X_test)

np.sqrt(mean_squared_error(y_test, y_pred))
enet_params = {"l1_ratio": [0,0.01,0.05,0.1,0.2,0.4,0.5,0.6,0.8,1],

               "alpha":[0.1,0.01,0.001,0.2,0.3,0.5,0.8,0.9,1,2,5,7,10,13,20,45,99,100]}



enet_model = ElasticNet().fit(X_train, y_train)

gs_cv_enet = GridSearchCV(enet_model, enet_params, cv = 10).fit(X_train, y_train)

enet_tuned = ElasticNet(**gs_cv_enet.best_params_).fit(X_train, y_train)

y_pred = enet_tuned.predict(X_test)

enet_reg_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

enet_reg_rmse
# en küçük rmse değerini bulma

rmse_list = {"Çoklu Doğrusal Regresyon": round(cd_reg_rmse, 4),

             "Ridge Regresyon": round(ridge_reg_rmse, 4),

             "Lasso Regresyon": round(lasso_reg_rmse, 4),

             "ElasticNet Regresyon": round(enet_reg_rmse, 4)}

rmse_list
print("Sabit değer: %.4f" % lasso_model.intercept_)

print(X_train.columns)

print("Katsayılar: " + str(np.around(lasso_model.coef_, 4)))