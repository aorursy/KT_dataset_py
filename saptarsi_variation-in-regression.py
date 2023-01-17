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
dataset = pd.read_csv("/kaggle/input/advtlr/Advertising.csv")
print(dataset.shape)
print(dataset.head(5))
# Selecting the Second, Third and Fouth Column
X= dataset.iloc[:,1:4]
# Selecting Fouth Columnn
y=dataset.iloc[:,4]
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train.shape
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error 
regressor = LinearRegression()

nof= X_train.shape[1]
print(nof)
mse=np.empty(2**nof-1)
ind=np.empty(nof)
l=list()
bl=list()
bitem=list()
k=0

#remaining = set(X_train.columns)
from itertools import combinations
remaining = set(X_train.columns)
for j in range(1,len(remaining)+1):
    comb = combinations(remaining, j)
    tempbest=5000
   
    for i in list(comb): 
        lsti=list(i)
        print(list(i))
        l.append(lsti)
        #X_train.iloc[:,i]
        regressor.fit(X_train.loc[:,lsti], y_train)
        y_exp=regressor.predict(X_train.loc[:,list(i)])
        mse[k] = mean_squared_error(y_train,y_exp)*y_train.shape[0]/(y_train.shape[0]-len(list(i)))
        if mse[k]<tempbest:
            bitem = lsti
            tempbest=mse[k]
        k = k + 1
        
    bl.append(bitem)

#X_train[list(i)]
mse
                

tmse=np.empty(len(bl))
k1=0
bfs=list()
tempbest=5000
for m in bl:
      regressor.fit(X_train.loc[:,m], y_train)
      y_exp=regressor.predict(X_test.loc[:,m])
      tmse[k1] = mean_squared_error(y_test,y_exp)*y_test.shape[0]/(y_test.shape[0]-len(list(i)))
      if tmse[k1]<tempbest:
            bfs = m
            tempbest=tmse[k1]
      k1 = k1 + 1
print(bfs)
tmse
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


np.random.seed(42)
m = 20
X = 3 * np.random.rand(m, 1)
y = 1 + 0.5 * X + np.random.randn(m, 1) / 1.5
X_new = np.linspace(0, 3, 100).reshape(100, 1)

def plot_model(model_class, polynomial, alphas, **model_kargs):
    for alpha, style in zip(alphas, ("b-", "g--", "r:")):
        model = model_class(alpha, **model_kargs) if alpha > 0 else LinearRegression()
        if polynomial:
            model = Pipeline([
                    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
                    ("std_scaler", StandardScaler()),
                    ("regul_reg", model),
                ])
        model.fit(X, y)
        y_new_regul = model.predict(X_new)
        lw = 2 if alpha > 0 else 1
        plt.plot(X_new, y_new_regul, style, linewidth=lw, label=r"$\alpha = {}$".format(alpha))
    plt.plot(X, y, "b.", linewidth=3)
    plt.legend(loc="upper left", fontsize=15)
    plt.xlabel("$x_1$", fontsize=18)
    plt.axis([0, 3, 0, 4])

plt.figure(figsize=(8,4))
#plt.subplot(121)
plot_model(Ridge, polynomial=False, alphas=(0, 10, 100), random_state=42)
plt.ylabel("$y$", rotation=0, fontsize=18)
#plt.subplot(122)
#plot_model(Ridge, polynomial=True, alphas=(0, 10**-5, 1), random_state=42)
plt.show()
from sklearn.linear_model import Lasso

plt.figure(figsize=(8,4))
#plt.subplot(121)
plot_model(Lasso, polynomial=False, alphas=(0, 0.1, 1), random_state=42)
plt.ylabel("$y$", rotation=0, fontsize=18)
#plt.subplot(122)
#plot_model(Lasso, polynomial=True, alphas=(0, 10**-7, 1), tol=1, random_state=42)
plt.show()
from sklearn.linear_model import ElasticNet
#elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
plt.figure(figsize=(8,4))
#plt.subplot(121)
plot_model(ElasticNet, polynomial=False, alphas=(0, 0.1, 1), random_state=42)
plt.ylabel("$y$", rotation=0, fontsize=18)
#plt.subplot(122)
#plot_model(Lasso, polynomial=True, alphas=(0, 10**-7, 1), tol=1, random_state=42)
plt.show()
auto=pd.read_csv('/kaggle/input/autompg-dataset/auto-mpg.csv')
auto.head(5)
X=auto.iloc[:,[1,2,4,5,6,7]]
X.shape
#X.dropna(inplace=True)
#X.fillna(X.mean(),inplace=True)
y=auto.iloc[:,0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X.describe()
lassoreg = Lasso(alpha=0.1,normalize=True, max_iter=1e5)
lassoreg.fit(X_train,y_train)
print(lassoreg.coef_)
df=pd.DataFrame()
alpha=[0.0001,0.001,0.01,0.1,1,10]
tmse=np.empty(len(alpha))
i=0
for k in alpha:
    lassoreg = Lasso(k,normalize=True, max_iter=1e5)
    lassoreg.fit(X_train,y_train)
    #a_row = pd.DataFrame([X.columns, lassoreg.coef_])
    #row_df = pd.DataFrame([a_row])
    df[str(k)]=lassoreg.coef_.tolist()
    y_exp=lassoreg.predict(X_test)
    #tmse[k1] = mean_squared_error(y_test,y_exp)*y_test.shape[0]/(y_test.shape[0]-len(list(i)))
    tmse[i] = mean_squared_error(y_test,y_exp)*y_test.shape[0]/(y_test.shape[0])
    i = i + 1
    #df = pd.concat([row_df, df], ignore_index=False)
    #df.append(lassoreg.coef_, ignore_index=True)
df_transposed = df.T
df_transposed.columns=X.columns
#df_transposed['alpha']=alpha

df_transposed.plot.line()
val=pd.Series(tmse,index=alpha)
val.plot()
# Add title and axis names
plt.title('Test MSE and Alpha')
plt.xlabel('Alpha')
plt.ylabel('Accuracy')
plt.show() 