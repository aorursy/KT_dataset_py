import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use("seaborn-whitegrid")

import warnings            

warnings.filterwarnings("ignore") 
y_2018 = pd.read_csv("/kaggle/input/world-happiness/2018.csv");

y_2019 = pd.read_csv("/kaggle/input/world-happiness/2019.csv");



data = pd.concat([y_2018,y_2019],sort=False)

data
data.describe().T
data.info()
data.rename(columns={

    "Overall rank": "rank",

    "Country or region": "country",

    "Score": "score",

    "GDP per capita": "gdp",

    "Social support": "social",

    "Healthy life expectancy": "healthy",

    "Freedom to make life choices": "freedom",

    "Generosity": "generosity",

    "Perceptions of corruption": "corruption"

},inplace=True)

del data["rank"]
data.columns[data.isnull().any()]
data.isnull().sum()
data[data["corruption"].isnull()]
avg_data_corruption = data[data["score"] > 6.774].mean().corruption

data.loc[data["corruption"].isnull(),["corruption"]] = avg_data_corruption

data[data["corruption"].isnull()]
df = data.copy()

df = df.select_dtypes(include=["float64","int64"])

df.head()
column_list = ["score","gdp","social","healthy","freedom","generosity","corruption"]

for col in column_list:

    sns.boxplot(x = df[col])

    plt.xlabel(col)

    plt.show()
# for corruption

df_table = df["corruption"]



Q1 = df_table.quantile(0.25)

Q3 = df_table.quantile(0.75)

IQR = Q3 - Q1



lower_bound = Q1 - 1.5*IQR

upper_bound = Q3 + 1.5*IQR

print("lower bound is " + str(lower_bound))

print("upper bound is " + str(upper_bound))

print("Q1: ", Q1)

print("Q3: ", Q3)
outliers_vector = (df_table < (lower_bound)) | (df_table > (upper_bound))

outliers_vector
outliers_vector = df_table[outliers_vector]

outliers_vector.index.values
df_table = data.copy()

df_table["corruption"].iloc[outliers_vector.index.values] = df_table["corruption"].mean()

df_table["corruption"].iloc[outliers_vector.index.values]
data = df_table
sns.jointplot(x="gdp",y="score",data=df_table,kind="reg")

plt.show()
from sklearn.linear_model import LinearRegression



X = data[["gdp"]]

X.head
y = data[["score"]]

y.head
reg = LinearRegression()

model = reg.fit(X,y)

print("intercept: ", model.intercept_)

print("coef: ", model.coef_)

print("rscore. ", model.score(X,y))
# prediction

plt.figure(figsize=(12,6))

g = sns.regplot(x=data["gdp"],y=data["score"],ci=None,scatter_kws = {'color':'r','s':9})

g.set_title("Model Equation")

g.set_ylabel("score")

g.set_xlabel("gdb")

plt.show()
# model.intercep_ + model.coef_ * 1

model.predict([[1]])
gdb_list = [[0.25],[0.50],[0.75],[1.00],[1.25],[1.50]]

model.predict(gdb_list)

for g in gdb_list:

    print("The happiness value of the country with a gdp value of ",g,": ",model.predict([g]))
def linear_reg(col,text,prdctn):

    

    sns.jointplot(x=col,y="score",data=df_table,kind="reg")

    plt.show()

    

    X = data[[col]]

    y = data[["score"]]

    reg = LinearRegression()

    model = reg.fit(X,y)

    

    # prediction

    plt.figure(figsize=(12,6))

    g = sns.regplot(x=data[col],y=data["score"],ci=None,scatter_kws = {'color':'r','s':9})

    g.set_title("Model Equation")

    g.set_ylabel("score")

    g.set_xlabel(col)

    plt.show()

    

    print(text,": ", model.predict([[prdctn]]))
linear_reg("social","The happiness value of the country whose sociability value is 2:",2)
column_list = ["score","gdp","social","healthy","freedom","generosity","corruption"]
linear_reg("healthy","The happiness value of the country whose healthiest value is 1.20:",1.20)
linear_reg("freedom","The happiness value of the country whose freedom value is 0.89:",0.89)
import statsmodels.api as sms



X = df.drop("score",axis=1)

y = df["score"]



# OLS(dependent,independent)

lm = sms.OLS(y,X)

model = lm.fit()

model.summary()
# create model with sckit learn



lm = LinearRegression()

model = lm.fit(X,y)

print("constant: ",model.intercept_)

print("coefficient: ",model.coef_)
# PREDICTION

# Score = 0.929921*gdp + 1.06504217*social + 0.94321492*healthy + 1.40426054*freedom + 0.52070628*generosity + 0.88114008*corruption



new_data = [[1],[2],[1.25],[1.75],[1.50],[0.75]]

new_data = pd.DataFrame(new_data).T

new_data
model.predict(new_data)
# calculating the amount of error



from sklearn.metrics import mean_squared_error



MSE = mean_squared_error(y,model.predict(X))

RMSE = np.sqrt(MSE)



print("MSE: ", MSE)

print("RMSE: ", RMSE)
from sklearn.model_selection import train_test_split



X = df.drop("score",axis=1)

y = df["score"]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.head()
X_test.head()
y_train.head()
y_test.head()
lm = LinearRegression()

lm.fit(X_train, y_train)

print("Training error",np.sqrt(mean_squared_error(y_train,model.predict(X_train))))

print("Test error",np.sqrt(mean_squared_error(y_test,model.predict(X_test))))
from sklearn.model_selection import cross_val_score



cross_val_score(model, X_train, y_train, cv=10, scoring="neg_mean_squared_error")
cvs_avg_mse = np.mean(-cross_val_score(model, X_train, y_train, cv=20, scoring="neg_mean_squared_error"))

cvs_avg_rmse = np.sqrt(cvs_avg_mse)



print("Cross Val Score MSE = ",cvs_avg_mse)

print("Cross Val Score RMSE = ",cvs_avg_rmse)
# Required Libraries

import numpy as np

import pandas as pd

from sklearn.linear_model import Ridge

from sklearn.metrics import mean_squared_error,r2_score

from sklearn.model_selection import train_test_split

from sklearn import model_selection

from sklearn.linear_model import RidgeCV
X = df.drop("score",axis=1)

y = df["score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



ridge_model = Ridge(alpha=0.1).fit(X_train, y_train)

ridge_model
ridge_model.coef_
ridge_model.intercept_
lambdas = 10**np.linspace(10,-2,100)*0.5 # Creates random numbers

ridge_model =  Ridge()

coefs = []



for i in lambdas:

    ridge_model.set_params(alpha=i)

    ridge_model.fit(X_train,y_train)

    coefs.append(ridge_model.coef_)

    

ax = plt.gca()

ax.plot(lambdas, coefs)

ax.set_xscale("log")
ridge_model = Ridge().fit(X_train,y_train)



y_pred = ridge_model.predict(X_train)



print("predict: ", y_pred[0:10])

print("real: ", y_train[0:10].values)
RMSE = np.mean(mean_squared_error(y_train,y_pred)) # rmse = square root of the mean of error squares

print("train error: ", RMSE)
Verified_RMSE = np.sqrt(np.mean(-cross_val_score(ridge_model, X_train, y_train, cv=20, scoring="neg_mean_squared_error")))

print("Verified_RMSE: ", Verified_RMSE)
# test error

y_pred = ridge_model.predict(X_test)

RMSE = np.mean(mean_squared_error(y_test,y_pred))

print("test error: ", RMSE)
ridge_model = Ridge(10).fit(X_train,y_train)

y_pred = ridge_model.predict(X_test)

np.sqrt(mean_squared_error(y_test,y_pred))
ridge_model = Ridge(30).fit(X_train,y_train)

y_pred = ridge_model.predict(X_test)

np.sqrt(mean_squared_error(y_test,y_pred))
ridge_model = Ridge(90).fit(X_train,y_train)

y_pred = ridge_model.predict(X_test)

np.sqrt(mean_squared_error(y_test,y_pred))
lambdas1 = 10**np.linspace(10,-2,100)

lambdas2 = np.random.randint(0,1000,100)



ridgeCV = RidgeCV(alphas = lambdas1,scoring = "neg_mean_squared_error", cv=10, normalize=True)

ridgeCV.fit(X_train,y_train)
ridgeCV.alpha_
# final model

ridge_tuned = Ridge(alpha = ridgeCV.alpha_).fit(X_train,y_train)

y_pred = ridge_tuned.predict(X_test)

np.sqrt(mean_squared_error(y_test,y_pred))
# for lambdas2

ridgeCV = RidgeCV(alphas = lambdas2,scoring = "neg_mean_squared_error", cv=10, normalize=True)

ridgeCV.fit(X_train,y_train)

ridge_tuned = Ridge(alpha = ridgeCV.alpha_).fit(X_train,y_train)

y_pred = ridge_tuned.predict(X_test)

np.sqrt(mean_squared_error(y_test,y_pred))
# Required Libraries

import numpy as np

import pandas as pd

from sklearn.linear_model import Ridge,Lasso

from sklearn.metrics import mean_squared_error,r2_score

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn import model_selection

from sklearn.linear_model import RidgeCV, LassoCV
X = df.drop("score",axis=1)

y = df["score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



lasso_model = Lasso().fit(X_train,y_train)
print("intercept: ", lasso_model.intercept_)

print("coef: ", lasso_model.coef_)
# coefficients for different lambda values



alphas = np.random.randint(0,10000,10)

lasso = Lasso()

coefs = []



for a in alphas:

    lasso.set_params(alpha=a)

    lasso.fit(X_train,y_train)

    coefs.append(lasso.coef_)
ax = plt.gca()

ax.plot(alphas,coefs)

ax.set_xscale("log")
lasso_model
lasso_model.predict(X_train)[0:5]
lasso_model.predict(X_test)[0:5]
y_pred = lasso_model.predict(X_test)

np.sqrt(mean_squared_error(y_test,y_pred))
r2_score(y_test,y_pred)
lasso_cv_model = LassoCV(cv=10,max_iter=100000).fit(X_train,y_train)

lasso_cv_model
lasso_cv_model.alpha_
lasso_tuned = Lasso().set_params(alpha= lasso_cv_model.alpha_).fit(X_train,y_train)

y_pred = lasso_tuned.predict(X_test)

np.sqrt(mean_squared_error(y_test,y_pred))
# Required Libraries

import numpy as np

import pandas as pd

from sklearn.linear_model import Ridge,Lasso,ElasticNet

from sklearn.metrics import mean_squared_error,r2_score

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn import model_selection

from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
X = df.drop("score",axis=1)

y = df["score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



enet_model = ElasticNet().fit(X_train,y_train)
enet_model.coef_
enet_model.intercept_
# prediction

enet_model.predict(X_train)[0:10]
enet_model.predict(X_test)[0:10]
y_pred = enet_model.predict(X_test)

np.sqrt(mean_squared_error(y_test,y_pred))