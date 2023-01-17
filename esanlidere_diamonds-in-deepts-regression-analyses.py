

#Libraries

import numpy as np 

import pandas as pd 



#Visualation

import matplotlib.pyplot as plt

import seaborn as sns



# Regression

from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, Lasso,LassoCV

from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score,GridSearchCV

import statsmodels.api as sm



from sklearn.preprocessing import MinMaxScaler , StandardScaler, Imputer, LabelEncoder



# Ignore warnings :

import warnings

warnings.filterwarnings('ignore')





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv("/kaggle/input/diamonds/diamonds.csv")

df.head()
df.info()
df.isnull().sum()
df.describe().T
df = df.drop(["Unnamed: 0"],axis=1)
df = df[(df[['x','y','z']] != 0).all(axis=1)]

df.describe().T
# heatmap

plt.subplots(figsize=(10, 10))

sns.heatmap(df.corr(), annot=True, linewidths=0.5,linecolor="white", cmap='BuPu', fmt= '.2f',square=True)

plt.show()
sns.pairplot(df, height=5, vars = ["price", "carat"], hue="cut", kind= "reg");
sns.jointplot(x="carat", y="price", data=df, kind="reg");
sns.jointplot(x="table", y="price", data=df, kind="reg");
sns.jointplot(x="depth", y="price", data=df, kind="reg");
(sns

    .FacetGrid(df,hue="cut",

              height=5,

              xlim=(0,10000))

    .map(sns.kdeplot, "price", shade=True)

    .add_legend()

    );
sns.catplot(x="cut",y="price", hue="color",kind="point", data=df);
sns.catplot(x = "cut", y = "price", hue= "clarity", data=df);
sns.boxplot(x="cut",y="price", data=df);
sns.boxplot(x="clarity",y="price", data=df);
sns.boxplot(x="color",y="price", data=df);
cut_le = LabelEncoder()

color_le = LabelEncoder()

clarity_le = LabelEncoder()





df['cut'] = cut_le.fit_transform(df['cut'])

df['color'] = color_le.fit_transform(df['color'])

df['clarity'] = clarity_le.fit_transform(df['clarity'])

df.head()
#Transform(0-1)

X_ = df.drop(["price"],axis = 1)

y =  df["price"]

X = (X_ - np.min(X_)) / (np.max(X_) - np.min(X_)).values
#train-test split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, random_state = 42)
lr = LinearRegression()

model = lr.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test,y_pred))

R2 = cross_val_score(model, X_train, y_train, cv = 10, scoring= "r2").mean() # average of 10 different r2 values

accuracies = cross_val_score(estimator = model, X = X_train, y = y_train, cv = 10,verbose = 1)

vte = np.sqrt(-cross_val_score(model, X_test, y_test, cv = 10,scoring = "neg_mean_squared_error")).mean() #validated test error



print("Accuracies          :", accuracies)

print("RMSE                : {}".format(rmse))

print("R2                  :  {}".format(R2))

print("Validated Test Error: {}".format(vte))
lasso_model = Lasso(alpha=0.1).fit(X_train,y_train)

y_pred = lasso_model.predict(X_test)

rmse1=np.sqrt(mean_squared_error(y_test,y_pred))

print("Non-Validated RMSE : {} ".format(rmse1))



#LassoCV

lasso_cv_model = LassoCV(alphas=None,

                        cv=10,

                        max_iter = 10000,

                        normalize = True)

lasso_cv_model.fit(X_train,y_train)

print("Optimum Alpha : {}".format(lasso_cv_model.alpha_))



#Tunned Model

lasso_tunned = Lasso(alpha = lasso_cv_model.alpha_)

lasso_tunned.fit(X_train,y_train)

y_pred = lasso_tunned.predict(X_test)

rmse2=np.sqrt(mean_squared_error(y_test,y_pred))

print("Validated RMSE : {} ".format(rmse2))

print("R2             : {} ".format(r2_score(y_test, y_pred)))
knn_model = KNeighborsRegressor().fit(X_train,y_train)

y_pred = knn_model.predict(X_test)

rmse1=np.sqrt(mean_squared_error(y_test,y_pred))

print("Non-Validated RMSE : {} ".format(rmse1))



#GridSearch - Best Params

knn_params = {"n_neighbors":np.arange(1,30,1)}

#KNN CV Model

knn = KNeighborsRegressor()

knn_cv_model = GridSearchCV(knn,knn_params,cv=10)

knn_cv_model.fit(X_train,y_train)

print("Best Params : {}".format(knn_cv_model.best_params_["n_neighbors"]))

#Tunned Model

knn_tuned = KNeighborsRegressor(n_neighbors=knn_cv_model.best_params_["n_neighbors"])

knn_tuned.fit(X_train,y_train)

rmse=np.sqrt(mean_squared_error(y_test, knn_tuned.predict(X_test)))

print("Validated RMSE : {} ".format(rmse))

print("R2             : {} ".format(r2_score(y_test, knn_tuned.predict(X_test))))
rf_model = RandomForestRegressor(n_estimators=100,random_state=42).fit(X_train,y_train)

y_pred = rf_model.predict(X_test)

rmse1 = np.sqrt(mean_squared_error(y_test,y_pred))

score = rf_model.score(X_test,y_test)

print("Non-Validated RMSE : {} ".format(rmse1))

print("Score              : {} ".format(score))

#GridSerach - Best Params

#rf_params = {"max_depth": list(range(1,10)),"max_features": [3,5,8],"n_estimators": [100,200,500,1000,2000]}

#rf_model = RandomForestRegressor(random_state=42)

#rf_cv_model = GridSearchCV(rf_model, rf_params, cv=10, n_jobs =-1)

#rf_cv_model.fit(X_train,y_train)

#print("Best Params: {} ".format(rf_cv_model.best_params_))



#Tuned Model

#rf_tuned = RandomForestRegressor()

#rf_tuned.fit(X_train,y_train)

#y_pred1 = rf_tuned.predict(X_test)

#rmse=np.sqrt(mean_squared_error(y_test,y_pred1))

#print("Validated RMSE : {} ".format(rmse))

#print("R2             : {} ".format(r2_score(y_test, y_pred)))
