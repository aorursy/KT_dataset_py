from warnings import filterwarnings
filterwarnings('ignore')


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt


from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


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
print("New League= A" ,df[df["NewLeague"]=="A"].agg({"Salary":"mean"}))
print("New League= N" ,df[df["NewLeague"]=="N"].agg({"Salary":"mean"}))
print("League= A" ,df[df["League"]=="A"].agg({"Salary":"mean"}))
print("League= N" ,df[df["League"]=="N"].agg({"Salary":"mean"}))
print("Division= E" ,df[df["Division"]=="E"].agg({"Salary":"mean"}))
print("Division= W" ,df[df["Division"]=="W"].agg({"Salary":"mean"}))
df.loc[(df["Salary"].isnull())& (df['Division'] == 'E'),"Salary"]=624.27
df.loc[(df["Salary"].isnull())& (df['Division'] == 'W'),"Salary"]=450.87
df.isnull().sum().sum()
# One hot Encoding
df1 = pd.get_dummies(df, columns = ['League', 'Division', 'NewLeague'], drop_first = True)
df1.head()
clf= LocalOutlierFactor(n_neighbors = 20, contamination = 0.1)
clf.fit_predict(df1)[0:10]
df1_scores=clf.negative_outlier_factor_
np.sort(df1_scores)[0:20]
sns.boxplot(df1_scores);
threshold=np.sort(df1_scores)[15]
df2=df1.loc[df1_scores > threshold]
print(df2.shape)
df2.head(2)
## Applying Standart Scaler on only X variables
dff=df2.drop(["Salary","League_N","Division_W","NewLeague_N"], axis=1)
categorical_columns=[col for col in dff.columns if dff[col].dtype=="object"]
numeric_columns=[num for num in dff.columns if num not in categorical_columns]
scaler=MinMaxScaler()
scaled_cols=scaler.fit_transform(dff[numeric_columns])
scaled_cols=pd.DataFrame(scaled_cols, columns=numeric_columns)


scaled_cols

ayrikdf=df2.loc[:, "League_N":"NewLeague_N"]
ayrikdf=pd.DataFrame(ayrikdf)
ayrikdf=ayrikdf.reset_index(drop=True)
print(ayrikdf.shape)
ayrikdf.head()
scaled_cols=pd.DataFrame(scaled_cols)
scaled_cols=scaled_cols.reset_index(drop=True)
print(scaled_cols.shape)
scaled_cols.head()
DF=pd.concat([scaled_cols,ayrikdf],axis=1)
DF.shape
# Regression
df3=df2.reset_index(drop=True)
y=df3["Salary"]
DF=DF.reset_index(drop=True)

X_train, X_test, y_train, y_test = train_test_split(DF, y, 
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
alphas4 = [59,200]
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
alphas4 = [23,30,50,100]
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
enet_params = {"l1_ratio": [0,0.001,0.002,0.003,0.005,0.01,0.03,0.05,0.1,0.2,0.4,0.5,0.6,0.8,1],
               "alpha":[21,30,40]}
enet_model = ElasticNet().fit(DF, y)
from sklearn.model_selection import GridSearchCV
gs_cv_enet = GridSearchCV(enet_model, enet_params, cv = 10).fit(DF, y)
print(gs_cv_enet.best_params_)
enet_tuned = ElasticNet(**gs_cv_enet.best_params_).fit(X_train, y_train)
y_pred = enet_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))