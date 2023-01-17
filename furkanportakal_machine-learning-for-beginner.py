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

import pandas as pd
df = pd.read_csv("/kaggle/input/Advertising.csv")
df = df.iloc[:,1:len(df)]
df.head()
df.info
import seaborn as sns
sns.jointplot(x="TV", y="sales", data=df, kind="reg");
from sklearn.linear_model import LinearRegression
X = df[["TV"]]
X.head()
y = df[["sales"]]
y.head()
reg = LinearRegression()
model = reg.fit(X,y)
model
#Let's learn the model
str(model)
dir(model)
#beta 0 (n) in basic linear regression formul(mx+n)
model.intercept_  
#beta 1 (m) in basic linear regression formul(mx+n)
model.coef_
#r2 
#Percentage of change in dependent variable explained by independent variables
model.score(X,y)
#The change in the independent variable is about 60 percent explained.
#guess
import seaborn as sns
import matplotlib.pyplot as plt

g = sns.regplot(df["TV"],df["sales"], ci=None, scatter_kws= {"color":"r","s":9})

#set table
g.set_title("Models: Sales = 7.03 + TV * 0.05")  
g.set_ylabel("Sales Number")
g.set_xlabel("TV expenditures")

plt.xlim(-10,310)
plt.ylim(bottom=0);
#Real Value 
7.03+ 0.05*165
#Predict Value
model.predict([[165]])
new_data = [[5],[150],[300],[450]]
model.predict(new_data)
#Expected 2D array, got 1D array instead
#Value Error expected : model.predict([500])
model.predict([[500]])
#predict : ~30
y.head()
y.head()
X.head()
model.predict(X)[0:6]

real_y = y[0:10]
predict_y = pd.DataFrame(model.predict(X)[0:10])
errors = pd.concat([real_y,predict_y], axis=1)
errors.columns = ["real_y","predict_y"]
errors
errors["error"] = errors["real_y"] - errors["predict_y"]
errors
errors["mean_squared"] = errors["error"]**2
errors
import numpy as np
MSE = np.mean(errors["mean_squared"])
print("MSE : ", MSE)
#Model
import numpy as np
import pandas as pd
df = pd.read_csv("/kaggle/input/Advertising.csv")
df = df.iloc[:,1:len(df)]
df.head()
X = df.drop("sales",axis=1)
y = df[["sales"]]
y.head()
X.head()
#Model : wtih Sklearn 
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
model = lm.fit(X, y)
# mx+n --> intercept = n
model.intercept_
#mx+n --> coef = m
model.coef_

2.94 + (30 * 0.04) + (10 * 0.19) -(40 * 0.001)
new_data = [[30],[10],[50]]
import pandas as pd
new_data = pd.DataFrame(new_data).T
new_data
model.predict(new_data)
from sklearn.metrics import mean_squared_error
y.head()
model.predict(X)[0:10]
MSE = mean_squared_error(y,model.predict(X))
MSE
import numpy as np
RMSE = np.sqrt(MSE)
RMSE
X.head()
y.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state =99)
#X_test.ndim
X_test.shape
X_train.shape
X_train.head()
y_train.head()
y_test.head()
lm = LinearRegression()
model = lm.fit(X_train, y_train)
#Error train value
y_predict_train = model.predict(X_train)
np.sqrt(mean_squared_error(y_train,y_predict_train))
#Error test value
y_predict_test = model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_predict_test))
from sklearn.model_selection import cross_val_score
model
# cv mse
cross_val_score(model, X_train, y_train, cv=10, scoring = "neg_mean_squared_error")
np.mean(-cross_val_score(model, X_train, y_train, cv=10, scoring = "neg_mean_squared_error"))
# cv root mse
import numpy as np
RMSE = np.sqrt(np.mean(-cross_val_score(model, X_train, y_train, cv=10, scoring = "neg_mean_squared_error")))
RMSE
P = np.sqrt(np.mean(-cross_val_score(model, X, y, cv=10, scoring = "neg_mean_squared_error")))
error =  P - RMSE 
error
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
#Data Set
df = pd.read_csv("/kaggle/input/Hitters.csv")
df = df.dropna()

dms = pd.get_dummies(df[['League','Division','NewLeague']])

y = df["Salary"]
X_ = df.drop(['Salary','League','Division','NewLeague'], axis=1).astype('float64')
X = pd.concat([X_, dms[['League_N','Division_W','NewLeague_N']]],axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                   y,
                                                   test_size=0.25,
                                                    random_state=42)
df.head()
df.shape
ridge_model = Ridge(alpha = 5).fit(X_train, y_train)
ridge_model
ridge_model.coef_
ridge_model.intercept_
# generating random numbers (from 10 to 2)
np.linspace(10,2,100)
lambdas = 10** np.linspace(10,2,100)*0.5
lambdas
ridge_model = Ridge()
factor = []

for i in lambdas:
    ridge_model.set_params(alpha=i)
    ridge_model.fit(X_train, y_train)
    factor.append(ridge_model.coef_)
ax = plt.gca()
ax.plot(lambdas,factor)
ax.set_xscale("log")
ridge_model = Ridge().fit(X_train, y_train)
y_pred = ridge_model.predict(X_train)
y_train[0:10]
y_pred[0:10]
RMSE = np.sqrt(mean_squared_error(y_train, y_pred))  
RMSE
#cv rmse
from sklearn.model_selection import cross_val_score 
np.sqrt(np.mean(-cross_val_score(ridge_model, X_train, y_train, cv=10, scoring = "neg_mean_squared_error")))
#test eror
y_pred = ridge_model.predict(X_test)
RMSE = np.sqrt(mean_squared_error(y_test, y_pred))  
RMSE
ridge_model= Ridge(alpha=1).fit(X_train, y_train)
y_pred = ridge_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
np.random.randint(0,1000,100)
lambda1 = np.random.randint(0,1000,100)
lambda2 = 10** np.linspace(10,2,100)*0.5
# pick one lambda 1 or lambda 2(you should try ) :)
ridgecv = RidgeCV(alphas = lambda1, scoring = "neg_mean_squared_error", cv=10, normalize=True )
ridgecv.fit(X_train, y_train)
ridgecv.alpha_ #(i think optimuim alpha = 2 )
#final model
ridge_tuned = Ridge(alpha=ridgecv.alpha_).fit(X_train, y_train)
y_pred = ridge_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV, LassoCV
#Data Set
df = pd.read_csv("/kaggle/input/Hitters.csv")
df = df.dropna()

dms = pd.get_dummies(df[['League','Division','NewLeague']])

y = df["Salary"]
X_ = df.drop(['Salary','League','Division','NewLeague'], axis=1).astype('float64')
X = pd.concat([X_, dms[['League_N','Division_W','NewLeague_N']]],axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                   y,
                                                   test_size=0.25,
                                                    random_state=42)
df.head()
df.shape
lasso_model = Lasso().fit(X_train, y_train)
lasso_model
lasso_model.intercept_ 
lasso_model.coef_ 
lasso = Lasso()
coefs = []
#alphas = np.random.randint(0,100000,10) #lambdas
alphas = lambdalar = 10** np.linspace(10,2,100)*0.5
for a in alphas:
    lasso.set_params(alpha = a)
    lasso.fit(X_train, y_train)
    coefs.append(lasso.coef_)
ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale("log")
lasso_model
lasso_model.predict(X_train)[0:5]
lasso_model.predict(X_test)[0:5]
y_pred = lasso_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
r2_score(y_test, y_pred)
alphas = lambdalar = 10** np.linspace(10,2,100)*0.5
lasso_cv_model = LassoCV(alphas = alphas, cv=10, max_iter = 100000).fit(X_train, y_train)
lasso_cv_model.alpha_
lasso_tuned = Lasso().set_params(alpha = lasso_cv_model.alpha_).fit(X_train, y_train)
lasso_tuned = Lasso(alpha = lasso_cv_model.alpha_).fit(X_train, y_train)
y_pred = lasso_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
pd.Series(lasso_tuned.coef_, index = X_train.columns)
#library
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
#Data Set
df = pd.read_csv("/kaggle/input/Hitters.csv")
df = df.dropna()

dms = pd.get_dummies(df[['League','Division','NewLeague']])

y = df["Salary"]
X_ = df.drop(['Salary','League','Division','NewLeague'], axis=1).astype('float64')
X = pd.concat([X_, dms[['League_N','Division_W','NewLeague_N']]],axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                   y,
                                                   test_size=0.25,
                                                    random_state=42)
enet_model = ElasticNet().fit(X_train, y_train)
enet_model.coef_
enet_model.intercept_
enet_model.predict(X_train)[0:10]
enet_model.predict(X_test)[0:10]
y_pred = enet_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred)) 
r2_score(y_test, y_pred)
enet_cv_model = ElasticNetCV(cv=10).fit(X_train, y_train)
enet_cv_model.alpha_
enet_cv_model.intercept_
enet_cv_model.coef_
enet_tuned = ElasticNet(alpha = enet_cv_model.alpha_ ).fit(X_train, y_train)
y_pred = enet_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import neighbors
from sklearn.svm import SVR
from warnings import filterwarnings
filterwarnings('ignore')
#Data Set
df = pd.read_csv("/kaggle/input/Hitters.csv")
df = df.dropna()

dms = pd.get_dummies(df[['League','Division','NewLeague']])

y = df["Salary"]
X_ = df.drop(['Salary','League','Division','NewLeague'], axis=1).astype('float64')
X = pd.concat([X_, dms[['League_N','Division_W','NewLeague_N']]],axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                   y,
                                                   test_size=0.25,
                                                    random_state=42)
X_train.head()































