import numpy as np

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))  
from warnings import filterwarnings

filterwarnings('ignore')

import numpy as np

import pandas as pd

ads = pd.read_csv('../input/advertising/Advertising.csv',usecols=[1,2,3,4])

data = ads.copy()

data.head() # This is datasets
# Data's information

data.info()
# Transposition of descriptive statistics

data.describe().T
# We check for missing values

data.isnull().any()
# Correlation map between variables

data.corr()
# Let's make the same connection situation with the help of visuals

import seaborn as sns

sns.pairplot(data,kind='reg');
sns.jointplot(x='TV',y='sales',data=data,kind='reg');
import statsmodels.api as sm

x = data[['TV']] # TV Values

x[:5] # First 5
x = sm.add_constant(x) # We have to add a column of 1.0

x[:5] # Like this
y = data[['sales']] # The value of y becomes our dependent variable

y[:5]
linear = sm.OLS(y,x) # We create a model

model = linear.fit() # We fit a model

model.summary() # This is summary
model.params # Parameters of the model
model.summary().tables[1] # Location of model parameters in the table
model.fittedvalues[0:5] # Let's reach the predicted values
y[:5] # Real values in these
print("Sales = " +  str("%.2f" % model.params[0]) + " + TV" + "*" + str("%.2f" % model.params[1]))

# Mathematical representation of the established prediction model.
g = sns.regplot(data["TV"], data["sales"], ci=None, scatter_kws={'color':'g', 's':15})

g.set_title("Model Equation: Sales = 7.03 + TV*0.05")

g.set_ylabel("Sales")

g.set_xlabel("TV Spending")

import matplotlib.pyplot as plt

plt.xlim(-10,310)

plt.ylim(bottom=0);
from sklearn.linear_model import LinearRegression

x = data[['TV']]

y = data['sales']

regression = LinearRegression()

model = regression.fit(x,y)

print('Model Intercept:',model.intercept_)

print('Model Coef:',model.coef_)
model.score(x,y) # Model Square Score
model.predict(x)[:10] # Predict with sklearn
7.03 + 30 * 0.04 # b0 + x * b1
model.predict([[30]]) # Can also be done this way
MULTI = [[5],[90],[200]] # Can be made in 3 pieces

model.predict(MULTI)
from sklearn.metrics import mean_squared_error, r2_score

import statsmodels.formula.api as smf

linear = smf.ols("sales ~ TV", data)

model = linear.fit()
mse = mean_squared_error(y,model.fittedvalues) # Mean Squared Error

rmse = np.sqrt(mse) # Root-mean-square deviation

print('MSE:',mse)

print('RMSE:',rmse)
print('Predict Values:',regression.predict(x)[:10]) # Predict Values

print('Real Values',y[:10]) # Real Values
remains = pd.DataFrame({'real_values':y[:10],

                       'predict_values':regression.predict(x)[:10]})

remains
remains['error'] = remains['real_values'] - remains['predict_values'] # We subtract predicted values from real values

remains
remains['error_square'] = remains['error']**2 # Square Of Errors

remains
np.sum(remains['error_square']) # Error squares sum
np.mean(remains["error_square"]) # Error squares mean
model.resid[0:10] # These residues appear with the function
plt.plot(model.resid,color='red');
ads = pd.read_csv('../input/advertising/Advertising.csv',usecols=[1,2,3,4])

data = ads.copy()

data.head()
from sklearn.model_selection import train_test_split,cross_val_predict,cross_val_score

# Train and test library for seperating

# library for cross validation predict

# librar for cross validation score

x = data.drop('sales',axis=1) # I synchronized variables other than the Sales variable to X.

y = data['sales'] # Our sales dependent variable in y.

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30,random_state=42) #We divided it as train and test

# Test_size allowed me to divide the argument by 80% to 20%

# random_state has always split the same values ​​without producing different values each time.

trainig = data.copy() # I put it in a variable so that a data set may be needed
linear = sm.OLS(y_train,x_train)

model = linear.fit()

model.summary()
linear = LinearRegression()

model = linear.fit(x_train,y_train) # We building model
print('Model Intercept:',model.intercept_) # constant coefficient

print('Model Coef:',model.coef_) # Other independent variable coefficients
pre_value = [[30],[10],[40]]

pre_value = pd.DataFrame(pre_value).T

print('Result:',model.predict(pre_value))
print('Train Error Score:',np.sqrt(mean_squared_error(y_train,model.predict(x_train)))) # Accessing train error

print('Test Error Score:',np.sqrt(mean_squared_error(y_test,model.predict(x_test)))) # Accessing test error
data.head() # The dataset
x = data.drop(['sales'],axis=1)

y = data['sales']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=42)

linear = LinearRegression()

model = linear.fit(x,y)

print('Train Error:',np.sqrt(mean_squared_error(y_train,model.predict(x_train))))

print('Model Score:',model.score(x_train,y_train))

print('Our verified score',cross_val_score(model,x_train,y_train,cv=10,scoring='r2').mean())
hit = pd.read_csv('../input/hitters/Hitters.csv')

data = hit.copy()

data.dropna(inplace=True)

data.head()
dms = pd.get_dummies(data[['League','Division','NewLeague']])

dms.head()

# Now we get which one of this transformation to be 1
y = data['Salary']

x_ = data.drop(['Salary','League','Division','NewLeague'],axis=1).astype('float64')

x_.head() # we extracted the variables and the dependent variable that we will make dummy from our dataset
x = pd.concat([x_,dms[['League_N','Division_W','NewLeague_N']]],axis=1)

# We added the categorical variables we converted

x.head()
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30,random_state=42)

print('x_train:',x_train.shape)

print('x_test:',x_test.shape)

print('y_train:',y_train.shape)

print('y_test:',y_test.shape)
from sklearn.decomposition import PCA

from sklearn.preprocessing import scale

pca = PCA()
x_reduced_train = pca.fit_transform(scale(x_train))

x_reduced_train[0:1,:]

# Yes, our reduction process is ok
np.cumsum(np.round(pca.explained_variance_ratio_,decimals=4)*100)[:10]

# Explained variance
from sklearn.linear_model import LinearRegression

linear = LinearRegression()

pcr_model = linear.fit(x_reduced_train,y_train)

# Here is our reduced model
print('Reduced Model Intercept:',pcr_model.intercept_)

print('Reduced Model Coef:',pcr_model.coef_)
y_pred = pcr_model.predict(x_reduced_train)

print('First 5 Predict:',y_pred[:5])
from sklearn.metrics import mean_squared_error,r2_score

print('Mean Square Error:',np.sqrt(mean_squared_error(y_train,y_pred)))

print('R2 Score:',r2_score(y_train,y_pred))
PCATEST = PCA()

x_reduced_test = PCATEST.fit_transform(scale(x_test))

y_pred = pcr_model.predict(x_reduced_test)

print('Mean Square Error:',np.sqrt(mean_squared_error(y_test,y_pred)))
linear = LinearRegression()

pcr_model = linear.fit(x_reduced_train[:,0:1],y_train)

y_pred = pcr_model.predict(x_reduced_test[:,0:1])

print('Mean Square Error:',np.sqrt(mean_squared_error(y_test,y_pred)))

# One component
linear = LinearRegression()

pcr_model = linear.fit(x_reduced_train[:,0:2],y_train)

y_pred = pcr_model.predict(x_reduced_test[:,0:2])

print('Mean Square Error:',np.sqrt(mean_squared_error(y_test,y_pred)))

# Two component
from sklearn import model_selection

cv_10 = model_selection.KFold(n_splits=10,shuffle=True,random_state=42)

linear = LinearRegression()

RMSE=[]

for i in np.arange(1,x_reduced_train.shape[1]+1):

    score = np.sqrt(-1*model_selection.cross_val_score(linear,

                                                      x_reduced_train[:,:i],

                                                      y_train.ravel(),

                                                      cv=cv_10,

                                                      scoring='neg_mean_squared_error').mean())

    RMSE.append(score)



import matplotlib.pyplot as plt

plt.plot(RMSE,'-v')

plt.xlabel('Component Quantity')

plt.ylabel('RMSE')

plt.title('PCR Model Tuning');
linear = LinearRegression()

pcr_model = linear.fit(x_reduced_train[:,0:4],y_train)

y_pred = pcr_model.predict(x_reduced_train[:,0:4])

print('Mean Square Error:',np.sqrt(mean_squared_error(y_train,y_pred)))

# This is our ideal mistake
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import train_test_split,cross_val_score

import pandas as pd

import numpy as np

hit = pd.read_csv('../input/hitters/Hitters.csv')

data = hit.copy()

data = data.dropna()

dms = pd.get_dummies(data[['League','Division','NewLeague']])

y = data['Salary']

x_ = data.drop(['Salary','League','Division','NewLeague'],axis=1).astype('float64')

x = pd.concat([x_,dms[['League_N','Division_W','NewLeague_N']]],axis=1)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state= 42)
from sklearn.cross_decomposition import PLSRegression, PLSSVD

pls_model = PLSRegression(n_components=2).fit(x_train,y_train)

pls_model.coef_ # We have the coefficients of the model.
print('Predict Values(Train Set):',pls_model.predict(x_train)[:5])

y_pred = pls_model.predict(x_train)

print('Mean Square Error for Train Set:',np.sqrt(mean_squared_error(y_train,y_pred)))

print('R2 Error for Train Set:',r2_score(y_train,y_pred))
y_pred = pls_model.predict(x_test)

print('Mean Square Error for Test Set:',np.sqrt(mean_squared_error(y_test,y_pred)))
import matplotlib.pyplot as plt

from sklearn import model_selection

cv_10 = model_selection.KFold(n_splits=10,shuffle=True,random_state=42)

RMSE=[]

for i in np.arange(1,x_train.shape[1] + 1):

    pls = PLSRegression(n_components=i)

    score=np.sqrt(-1*cross_val_score(pls,x_train,y_train,cv=10,scoring='neg_mean_squared_error').mean())

    RMSE.append(score)

plt.plot(np.arange(1,x_train.shape[1]+1),np.array(RMSE),'-v',c='r')

plt.xlabel('Component Quantity')

plt.ylabel('RMSE')

plt.title('Salary');
hit = pd.read_csv('../input/hitters/Hitters.csv')

data = hit.copy()

data.dropna(inplace=True)

dms = pd.get_dummies(data[['League','Division','NewLeague']])

y = data['Salary']

x_ = data.drop(['Salary','League','Division','NewLeague'],axis=1).astype('float64')

x = pd.concat([x_,dms[['League_N','Division_W','NewLeague_N']]],axis=1)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30,random_state=42)
from sklearn.linear_model import Ridge

ridge_model = Ridge(alpha=0.1).fit(x_train,y_train)

print('Ridge Model Coef:',ridge_model.coef_)
lambdas = 10**np.linspace(10,-2,100)*0.5

ridge_model = Ridge()

coef = []

for i in lambdas:

    ridge_model.set_params(alpha=i)

    ridge_model.fit(x_train,y_train)

    coef.append(ridge_model.coef_)

ax = plt.gca()

ax.plot(lambdas,coef)

ax.set_xscale('Log')

plt.xlabel('Lambda(Alpha) Value')

plt.ylabel('Coef/Heavy')

plt.title('Ridge Coefficients as a Function of Regulation');
y_pred = ridge_model.predict(x_test)

print('Ridge Model Predict:',np.sqrt(mean_squared_error(y_test,y_pred)))
lambdas = 10**np.linspace(10,-2,100)*0.5

from sklearn.linear_model import RidgeCV

ridge_cv = RidgeCV(alphas=lambdas,scoring='neg_mean_squared_error',normalize=True)

ridge_cv.fit(x_train,y_train)
print('Ridge Alphas:',ridge_cv.alpha_)
ridge_tuned = Ridge(alpha=ridge_cv.alpha_,normalize=True).fit(x_train,y_train)

print('Verified Model Error:',np.sqrt(mean_squared_error(y_test,ridge_tuned.predict(x_test))))
hit = pd.read_csv('../input/hitters/Hitters.csv')

data = hit.copy()

data.dropna(inplace=True)

dms = pd.get_dummies(data[['League','Division','NewLeague']])

y = data['Salary']

x_ = data.drop(['Salary','League','Division','NewLeague'],axis=1).astype('float64')

x = pd.concat([x_,dms[['League_N','Division_W','NewLeague_N']]],axis=1)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=42)
from sklearn.linear_model import Lasso

lasso_model = Lasso(alpha=0.01).fit(x_train,y_train)
lambdas = 10**np.linspace(10,-2,100)*0.5

lasso = Lasso()

coef = []

for i in lambdas:

    lasso.set_params(alpha=i)

    lasso.fit(x_train,y_train)

    coef.append(lasso.coef_)

ax = plt.gca()

ax.plot(lambdas**2,coef)

ax.set_xscale('log')

plt.axis('tight')

plt.xlabel('Lambda(Alpha)')

plt.ylabel('Weights');
print('Predict Values:',lasso_model.predict(x_test))
from sklearn.linear_model import LassoCV

lasso_model_cv = LassoCV(alphas=None,

                        cv=10,

                        max_iter=1000,

                        normalize=True)

lasso_model_cv.fit(x_train,y_train)

print('Lasso Model Alpha:',lasso_model_cv.alpha_)
lasso_tuned_model = Lasso(alpha=lasso_model_cv.alpha_).fit(x_train,y_train)

print('Verified Score:',np.sqrt(mean_squared_error(y_test,lasso_tuned_model.predict(x_test))))
hit = pd.read_csv('../input/hitters/Hitters.csv')

data = hit.copy()

data.dropna(inplace=True)

dms = pd.get_dummies(data[['League','Division','NewLeague']])

y = data['Salary']

x_ = data.drop(['Salary','League','Division','NewLeague'],axis=1).astype('float64')

x = pd.concat([x_,dms[['League_N','Division_W','NewLeague_N']]],axis=1)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30,random_state=42)
from sklearn.linear_model import ElasticNet

elastic_model = ElasticNet().fit(x_train,y_train)

print('Elastic Net Coef:',elastic_model.coef_)

print('Elastic Net Intercept:',elastic_model.intercept_)
print('Elastic Net Predict Values:',elastic_model.predict(x_test))
y_pred = elastic_model.predict(x_test)

print('Simple Error Score:',np.sqrt(mean_squared_error(y_test,y_pred)))
from sklearn.linear_model import ElasticNetCV

elastic_cv_model = ElasticNetCV(cv=10,random_state=42).fit(x_train,y_train)
print('Optimum Alpha Score:',elastic_cv_model.alpha_)
elastic_tuned = ElasticNet(alpha=elastic_cv_model.alpha_).fit(x_train,y_train)

y_pred = elastic_tuned.predict(x_test)

print('Verified Score:',np.sqrt(mean_squared_error(y_test,y_pred)))
import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split,ShuffleSplit,GridSearchCV

from sklearn.metrics import mean_squared_error,r2_score

import matplotlib.pyplot as plt

from sklearn import model_selection

from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import BaggingRegressor
hit = pd.read_csv('../input/hitters/Hitters.csv')

data = hit.copy()

data.dropna(inplace=True)

dms = pd.get_dummies(data[['League','Division','NewLeague']])

y = data['Salary']

x_ = data.drop(['Salary','League','Division','NewLeague'],axis=1).astype('float64')

x = pd.concat([x_,dms[['League_N','Division_W','NewLeague_N']]],axis=1)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30,random_state=42)
knn_model = KNeighborsRegressor().fit(x_train,y_train)
y_pred = knn_model.predict(x_test)

print('KNN Model Predict Values:',np.sqrt(mean_squared_error(y_test,y_pred)))
from sklearn.model_selection import GridSearchCV

knn_parameter = {

    'n_neighbors':np.arange(1,50,1) # We do the dictionary naming in accordance with the parameter of the algorithm. not random.

}

KNN = KNeighborsRegressor()

knn_cv_model = GridSearchCV(KNN,knn_parameter,cv=10)

knn_cv_model.fit(x_train,y_train)
print('Best Parameter:',knn_cv_model.best_params_['n_neighbors'])
# We are re-modeling with our new parameter.

knn_tuned = KNeighborsRegressor(n_neighbors=knn_cv_model.best_params_['n_neighbors'])

knn_tuned.fit(x_train,y_train)

y_pred = knn_tuned.predict(x_test)

print('Verified Score:',np.sqrt(mean_squared_error(y_test,y_pred)))
hit = pd.read_csv('../input/hitters/Hitters.csv')

data = hit.copy()

data  = data.dropna()

dms = pd.get_dummies(data[['League','Division','NewLeague']])

y = data['Salary']

x_ = data.drop(['Salary','League','Division','NewLeague'],axis = 1).astype('float64')

x = pd.concat([x_,dms[['League_N','Division_W','NewLeague_N']]],axis=1)

x_train,x_test,y_train,y_test = train_test_split(x,y,

                                                test_size = 0.25,

                                                random_state=42)

x_train = pd.DataFrame(x_train['Hits'])

x_test = pd.DataFrame(x_test['Hits'])
from sklearn.svm import SVR
svr_model  = SVR('linear').fit(x_train,y_train)

print('SVR Model Predict:',svr_model.predict(x_test)[:5])

y_pred = svr_model.predict(x_train)

print('Model Equation->','y = {0} + {1} x '.format(svr_model.intercept_[0], # Model Equation

                               svr_model.coef_[0][0]))
plt.scatter(x_train,y_train)

plt.plot(x_train,y_pred,color='r');
svr_model.predict([[54]]) # Estimates are created this way
# Test Error Calculation

y_pred = svr_model.predict(x_test)

print('SVR Model Error Score:',np.sqrt(mean_squared_error(y_test,y_pred)))
svr_params = {'C':np.arange(0.1,2,3)}

svr = SVR()

svr_cv_model = GridSearchCV(svr,svr_params,cv=10).fit(x_train,y_train)

# The model is set up this way
print('Best Parametre for model:',svr_cv_model.best_params_['C'])
svr_tuned = SVR('linear',C=svr_cv_model.best_params_['C']).fit(x_train,y_train)

y_pred = svr_tuned.predict(x_test)

print('Verified Error Score:',np.sqrt(mean_squared_error(y_test,y_pred)))
np.random.seed(3)

x_sim = np.random.uniform(2,10,145)

y_sim = np.sin(x_sim) + np.random.normal(0,0.4,145)

x_outliers = np.arange(2.5,5,0.5)

y_outliers = -5*np.arange(5)

x_sim_idx = np.argsort(np.concatenate([x_sim,x_outliers]))

x_sim = np.concatenate([x_sim,x_outliers])[x_sim_idx]

y_sim = np.concatenate([y_sim,y_outliers])[x_sim_idx]
from sklearn.linear_model import LinearRegression

ols = LinearRegression()

ols.fit(np.sin(x_sim[:,np.newaxis]),y_sim)

ols_pred = ols.predict(np.sin(x_sim[:,np.newaxis]))

from sklearn.svm import SVR

eps = 0.1

svr = SVR('rbf',epsilon=eps)

svr.fit(x_sim[:,np.newaxis],y_sim)

svr_pred = svr.predict(x_sim[:,np.newaxis])
plt.scatter(x_sim,y_sim,alpha=0.5,s=26)

plt_ols = plt.plot(x_sim,ols_pred,color ='g')

plt_svr = plt.plot(x_sim,svr_pred,color='r')

plt.xlabel('independent variable')

plt.ylabel('dependent variable')

plt.ylim(-5.2,2.2)

plt.legend(['LSE','SVR']);
hit = pd.read_csv('../input/hitters/Hitters.csv')

data = hit.copy()

data  = data.dropna()

dms = pd.get_dummies(data[['League','Division','NewLeague']])

y = data['Salary']

x_ = data.drop(['Salary','League','Division','NewLeague'],axis = 1).astype('float64')

x = pd.concat([x_,dms[['League_N','Division_W','NewLeague_N']]],axis=1)

x_train,x_test,y_train,y_test = train_test_split(x,y,

                                                test_size = 0.30,

                                                random_state=42)
svr_rbf = SVR('rbf',gamma='auto').fit(x_train,y_train) # The 'rbf' here means non-linear SVR.
y_pred = svr_rbf.predict(x_test) # Estimate for x test

print('Prediction First 5 Values:',y_pred[:5])

print('Simple Error:',np.sqrt(mean_squared_error(y_test,y_pred)))
svr_params = {

    'C':[0.1,0.4,5,10,20,30,40,50]

}

svr_rbf_model = SVR('rbf')

svr_cv_model = GridSearchCV(svr_rbf_model,svr_params,cv=10)

svr_cv_model.fit(x_train,y_train)
print('Best C Parameter:',svr_cv_model.best_params_['C'])
svr_tuned = SVR('rbf',gamma='auto',C=svr_cv_model.best_params_['C']).fit(x_train,y_train)
y_pred = svr_tuned.predict(x_test)

print('Verified Score:',np.sqrt(mean_squared_error(y_test,y_pred)))
hit = pd.read_csv('../input/hitters/Hitters.csv')

data = hit.copy()

data  = data.dropna()

dms = pd.get_dummies(data[['League','Division','NewLeague']])

y = data['Salary']

x_ = data.drop(['Salary','League','Division','NewLeague'],axis = 1).astype('float64')

x = pd.concat([x_,dms[['League_N','Division_W','NewLeague_N']]],axis=1)

x_train,x_test,y_train,y_test = train_test_split(x,y,

                                                test_size = 0.25,

                                                random_state=42)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(x_train)

x_train_scaled = scaler.transform(x_train)

x_test_scaled = scaler.transform(x_test)
from sklearn.neural_network import MLPRegressor
MLP_model = MLPRegressor(hidden_layer_sizes=(100,20)).fit(x_train_scaled,y_train)

MLP_model.n_layers_ # Our number of layers
y_pred = MLP_model.predict(x_test_scaled)

print('Simple Score Error:',np.sqrt(mean_squared_error(y_test,y_pred)))
mlp_params = {

    'alpha':[0.1,0.01,0.02,0.0005],

    'hidden_layer_sizes':[(20,20),(100,50,150),(300,200,150)],

    'activation':['relu','logistic']

}

mlp_model = MLPRegressor()

mlp_cv_model = GridSearchCV(mlp_model,mlp_params,cv=10)

mlp_cv_model.fit(x_train_scaled,y_train)
print('Best Alpha Parameter:',mlp_cv_model.best_params_['alpha'])

print('Best Hidden Layer Sizes:',mlp_cv_model.best_params_['hidden_layer_sizes'])

print('Best Activation Function:',mlp_cv_model.best_params_['activation'])
mlp_tuned = MLPRegressor(alpha=mlp_cv_model.best_params_['alpha'],

                        hidden_layer_sizes=mlp_cv_model.best_params_['hidden_layer_sizes'],

                        activation=mlp_cv_model.best_params_['activation'])

mlp_tuned.fit(x_train_scaled,y_train)

y_pred = mlp_tuned.predict(x_test_scaled)

print('Verified Error Score:',np.sqrt(mean_squared_error(y_test,y_pred)))
hit = pd.read_csv('../input/hitters/Hitters.csv')

data = hit.copy()

data  = data.dropna()

dms = pd.get_dummies(data[['League','Division','NewLeague']])

y = data['Salary']

x_ = data.drop(['Salary','League','Division','NewLeague'],axis = 1).astype('float64')

x = pd.concat([x_,dms[['League_N','Division_W','NewLeague_N']]],axis=1)

x_train,x_test,y_train,y_test = train_test_split(x,y,

                                                test_size = 0.25,

                                                random_state=42)

x_train = pd.DataFrame(x_train['Hits'])

x_test = pd.DataFrame(x_test['Hits'])
cart_model = DecisionTreeRegressor(max_leaf_nodes=10) # I gave Max leaf randomly.

cart_model.fit(x_train,y_train)
x_grid = np.arange(min(np.array(x_train)),max(np.array(x_train)),0.01)

x_grid = x_grid.reshape((len(x_grid),1))

plt.scatter(x_train,y_train,color='lime')

plt.plot(x_grid,cart_model.predict(x_grid),color='Green')

plt.title('Cart Regression Tree')

plt.xlabel('Hits')

plt.ylabel('Salary')

plt.show()
y_pred = cart_model.predict(x_test)

print('Simple Predict Error:',np.sqrt(mean_squared_error(y_test,y_pred)))
params = {'min_samples_split':range(2,100),

         'max_leaf_nodes':range(2,10)}

cart_model_cv = GridSearchCV(cart_model,params,cv=10)

cart_model_cv.fit(x_train,y_train)
print('Best Parameter For Model',cart_model_cv.best_params_)
cart_tuned = DecisionTreeRegressor(max_leaf_nodes=cart_model_cv.best_params_['max_leaf_nodes'],min_samples_split=cart_model_cv.best_params_['min_samples_split'])

cart_tuned.fit(x_train,y_train)

y_pred = cart_tuned.predict(x_test)

print('Verified Error Score:',np.sqrt(mean_squared_error(y_test,y_pred)))
hit = pd.read_csv('../input/hitters/Hitters.csv')

data = hit.copy()

data  = data.dropna()

dms = pd.get_dummies(data[['League','Division','NewLeague']])

y = data['Salary']

x_ = data.drop(['Salary','League','Division','NewLeague'],axis = 1).astype('float64')

x = pd.concat([x_,dms[['League_N','Division_W','NewLeague_N']]],axis=1)

x_train,x_test,y_train,y_test = train_test_split(x,y,

                                                test_size = 0.25,

                                                random_state=42)
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(random_state=42)

rf_model.fit(x_train,y_train)

y_pred = rf_model.predict(x_test)

print('Simple Test Error:',np.sqrt(mean_squared_error(y_test,y_pred)))
rf_params = {'max_depth':list(range(1,10)),

            'max_features':[3,5,10,15],

            'n_estimators':[100,200,500,1000,2000]}



rf_model = RandomForestRegressor(random_state=42)

rf_model_cv = GridSearchCV(rf_model,rf_params,cv=10,n_jobs=-1,verbose=2) # If n_jobs -1, the number of parameters to be searched will take the processors to full performance.

rf_model_cv.fit(x_train,y_train)
print('Best Max Depth:',rf_model_cv.best_params_['max_depth'])

print('Best Max Features:',rf_model_cv.best_params_['max_features'])

print('Best N Estimators:',rf_model_cv.best_params_['n_estimators'])
rf_model_tuned = RandomForestRegressor(max_depth=rf_model_cv.best_params_['max_depth'],

                                      max_features=rf_model_cv.best_params_['max_features'],

                                      n_estimators=rf_model_cv.best_params_['n_estimators']).fit(x_train,y_train)
y_pred = rf_model_tuned.predict(x_test)

print('Verified Error Score:',np.sqrt(mean_squared_error(y_test,y_pred)))
Importance = pd.DataFrame({'Importance':rf_model_tuned.feature_importances_*100},

                         index=x_train.columns)

# We check this for which variables are more important model.



Importance.sort_values(by='Importance',axis=0,ascending=True).plot(kind='barh',color='lime')

plt.xlabel('Significance of variables');
from sklearn.ensemble import GradientBoostingRegressor

gbm_model = GradientBoostingRegressor()

gbm_model.fit(x_train,y_train)
y_pred = gbm_model.predict(x_test)

print('Simple Test Error:',np.sqrt(mean_squared_error(y_test,y_pred)))
gbm_params = {

    'learning_rate':[0.001,0.01,0.1,0.2],

    'max_depth':[3,5,8,50,100],

    'n_estimators':[200,500,1000,2000],

    'subsample':[1,0.5,0.75]

}

gbm = GradientBoostingRegressor()

gbm_cv_model = GridSearchCV(gbm,gbm_params,cv=10,n_jobs=-1,verbose=2)

gbm_cv_model.fit(x_train,y_train)
print('Best Paramter For Model:',gbm_cv_model.best_params_)
gbm_tuned = GradientBoostingRegressor(learning_rate=gbm_cv_model.best_params_['learning_rate'],

                                     max_depth=gbm_cv_model.best_params_['max_depth'],

                                     n_estimators=gbm_cv_model.best_params_['n_estimators'],

                                     subsample=gbm_cv_model.best_params_['subsample']).fit(x_train,y_train)
y_pred = gbm_tuned.predict(x_test)

print('Verified Test Error Score:',np.sqrt(mean_squared_error(y_test,y_pred)))
Importance = pd.DataFrame({'Importance':gbm_tuned.feature_importances_*100},

                         index=x_train.columns)

Importance.sort_values(by='Importance',axis=0,ascending=True).plot(kind='barh',color='Red')

plt.xlabel('Significance Level of Variables');
hit = pd.read_csv('../input/hitters/Hitters.csv')

data = hit.copy()

data  = data.dropna()

dms = pd.get_dummies(data[['League','Division','NewLeague']])

y = data['Salary']

x_ = data.drop(['Salary','League','Division','NewLeague'],axis = 1).astype('float64')

x = pd.concat([x_,dms[['League_N','Division_W','NewLeague_N']]],axis=1)

x_train,x_test,y_train,y_test = train_test_split(x,y,

                                                test_size = 0.25,

                                                random_state=42)
import xgboost as xgb
DM_train = xgb.DMatrix(data= x_train,label=y_train)

DM_test = xgb.DMatrix(data=x_test,label=y_test)
from xgboost import XGBRegressor
xgb = XGBRegressor().fit(x_train,y_train)
y_pred = xgb.predict(x_test)

print('Simple Test Error:',np.sqrt(mean_squared_error(y_test,y_pred)))
xgb_grid = {

    'colsample_bytree':[0.4,0.5,0.6,0.9,1],

    'n_estimators':[100,200,500,1000],

    'max_depth':[2,3,4,5,6],

    'learning_rate':[0.1,0.01,0.5]

}

xgb = XGBRegressor()

xgb_cv_model = GridSearchCV(xgb,param_grid=xgb_grid,cv=10,n_jobs=-1,verbose=2)

xgb_cv_model.fit(x_train,y_train)
print('XGB Best Paramter:',xgb_cv_model.best_params_)
xgb_tuned = XGBRegressor(colsample_bytree = xgb_cv_model.best_params_['colsample_bytree'],

                        learning_rate=xgb_cv_model.best_params_['learning_rate'],

                        max_depth=xgb_cv_model.best_params_['max_depth'],

                        n_estimators=xgb_cv_model.best_params_['n_estimators']).fit(x_train,y_train)
y_pred = xgb_tuned.predict(x_test)

print('Verified Test Error Score:',np.sqrt(mean_squared_error(y_test,y_pred)))
import pandas as pd

import numpy as np

import statsmodels.api as sm

import statsmodels.formula.api as smf

import seaborn as sns

from sklearn.preprocessing import scale

from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score,cross_val_predict

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report,log_loss

from sklearn.metrics import roc_auc_score,roc_curve

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn import tree

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from warnings import filterwarnings

filterwarnings('ignore')
diabetes = pd.read_csv('../input/diabetes/diabetes.csv')

data = diabetes.copy()

data.dropna(inplace=True)

data.head()
y = data['Outcome']

x = data.drop(['Outcome'],axis=1)
logistic = sm.Logit(y,x)

logistic_model = logistic.fit()

logistic_model.summary()

# We are not strangers to this output we have seen before.
from sklearn.linear_model import LogisticRegression

logistic = LogisticRegression(solver='liblinear')

logistic_model = logistic.fit(x,y)
y_pred = logistic_model.predict(x)

print(confusion_matrix(y,y_pred))
print('Classification Ratio:',accuracy_score(y,y_pred))

print('******************************')

print(classification_report(y,y_pred))
logistic_model.predict(x)[:10] # Predict Values
print('Probability Of Forecast Values:',logistic_model.predict_proba(x)[:10][:,0:2])
print('Reel Values:',y[:10])
logit_roc_auc = roc_auc_score(y,logistic_model.predict(x))

fpr,tpr,thresholds = roc_curve(y,logistic_model.predict_proba(x)[:,1])

plt.figure()

plt.plot(fpr,tpr,label='AUC(area=%0.2f)' % logit_roc_auc)

plt.plot([0,1],[0,1],'r--')

plt.xlim([0.0,1.0])

plt.ylim([0.0,1.05])

plt.xlabel('False Positive Ratio')

plt.ylabel('True Positive Ratio')

plt.title('ROC')

plt.show()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=42)

# Cross Val. We perform train test separation to calculate the score.
logistic = LogisticRegression(solver='liblinear')

logistic_model = logistic.fit(x,y)
print('Test Succes Score:',accuracy_score(y_test,logistic_model.predict(x_test)))

print('The Most Accurate Success Rate',cross_val_score(logistic_model,x_test,y_test,cv=10).mean())
diabetes = pd.read_csv('../input/diabetes/diabetes.csv')

data = diabetes.copy()

data.dropna(inplace=True)

y = data['Outcome']

x = data.drop(['Outcome'],axis=1)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30,random_state=42)
from sklearn.naive_bayes import GaussianNB

GuasNB = GaussianNB()

GuasNB_model = GuasNB.fit(x_train,y_train)
print('Guassian Naive Bayes Prediction Values:',GuasNB_model.predict(x_test)[:10])

print('************************')

print('Reel Dataset Values',y[:10])
y_pred = GuasNB_model.predict(x_test)

print('Accuracy Score:',accuracy_score(y_test,y_pred))
print('Verified Score:',cross_val_score(GuasNB,x_test,y_test,cv=10).mean())
diabetes = pd.read_csv('../input/diabetes/diabetes.csv')

data = diabetes.copy()

data.dropna(inplace=True)

y = data['Outcome']

x = data.drop(['Outcome'],axis=1)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=10,random_state=42)
KNN = KNeighborsClassifier()

KNN_model= KNN.fit(x_train,y_train)
y_pred = KNN_model.predict(x_test)

print('Simple Accuracy Score:',accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))

# We came out in more detail.
knn_params = {

    'n_neighbors':np.arange(1,50)

}

knn = KNeighborsClassifier()

knn_cv_model = GridSearchCV(knn,knn_params,cv=10)

knn_cv_model.fit(x_train,y_train)
print('Best Parameter for model:', str(knn_cv_model.best_params_['n_neighbors']))

print('Best Score for model:',knn_cv_model.best_score_)
knn_tuned = KNeighborsClassifier(n_neighbors=knn_cv_model.best_params_['n_neighbors']).fit(x_train,y_train)

print('Score Funciton Value:',knn_tuned.score(x_test,y_test))
y_pred = knn_tuned.predict(x_test)

print('Verified Score:',accuracy_score(y_test,y_pred))
diabetes = pd.read_csv('../input/diabetes/diabetes.csv')

data = diabetes.copy()

data.dropna(inplace=True)

y = data['Outcome']

x = data.drop(['Outcome'],axis=1)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30,random_state=42)
svm_model = SVC(kernel='linear').fit(x_train,y_train)
y_pred = svm_model.predict(x_test)

print('Simple Accuracy Score:',accuracy_score(y_test,y_pred))
svc_params = {

    'C':np.arange(1,10)

}

svc = SVC(kernel='linear')

svc_cv_model = GridSearchCV(svc,svc_params,cv=10,n_jobs=-1,verbose=2)

svc_cv_model.fit(x_train,y_train)
print('Best Parameter for model:',svc_cv_model.best_params_['C'])
svc_tuned = SVC(kernel='linear',C=svc_cv_model.best_params_['C']).fit(x_train,y_train)

y_pred = svc_tuned.predict(x_test)

print('Verified Score:',accuracy_score(y_test,y_pred))
diabetes = pd.read_csv('../input/diabetes/diabetes.csv')

data = diabetes.copy()

data.dropna(inplace=True)

y = data['Outcome']

x = data.drop(['Outcome'],axis=1)

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.30,random_state=42)
svc_rbf = SVC(kernel='rbf').fit(x_train,y_train)

y_pred = svc_rbf.predict(x_test)

print('Simple Accuracy Score:',accuracy_score(y_test,y_pred))
svc_params = {

    'C':[0.0001,0.001,0.1,1,5,10,50,100],

    'gamma':[0.0001,0.001,0.1,1,5,10,50,100]

}

svc = SVC(kernel='rbf')

svc_cv_model = GridSearchCV(svc,svc_params,cv=10,n_jobs=-1,verbose=2)

svc_cv_model.fit(x_train,y_train)
print('Best Parameters For model:',svc_cv_model.best_params_)
svc_tuned = SVC(kernel='rbf',

               C=svc_cv_model.best_params_['C'],

               gamma=svc_cv_model.best_params_['gamma']).fit(x_train,y_train)
y_pred = svc_tuned.predict(x_test)

print('Verified Accuracy Score:',accuracy_score(y_test,y_pred))
import pandas as pd

from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score,cross_val_predict

from sklearn.metrics import accuracy_score

diabetes = pd.read_csv('../input/diabetes/diabetes.csv')

data = diabetes.copy()

data.dropna(inplace=True)

y = data['Outcome']

x = data.drop(['Outcome'],axis=1)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30,random_state=42)
# We need a small standardization process

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(x_train)

x_train_scaled = scaler.transform(x_train)

x_test_scaled = scaler.transform(x_test)
from warnings import filterwarnings

filterwarnings('ignore')

from sklearn.neural_network import MLPClassifier

MLPC = MLPClassifier().fit(x_train_scaled,y_train)
y_pred = MLPC.predict(x_test_scaled)

print('Simple Test Error:',accuracy_score(y_test,y_pred))
mlpc_params = {

    'alpha':[0.1,0.01,0.02,0.005,0.0001,0.00001],

    'hidden_layer_sizes':[(10,10,10),

                         (100,100,100),

                         (100,100),

                         (3,5),

                         (5,3)],

    'solver':['lbgfs','adam','sgd'],

    'activation':['relu','logistic']

}



mlpc = MLPClassifier()

mlpc_cv_model = GridSearchCV(mlpc,mlpc_params,cv=10,n_jobs=-1,verbose=2)

mlpc_cv_model.fit(x_train_scaled,y_train)
print('Best Parameters for model:',mlpc_cv_model.best_params_)
mlpc_tuned = MLPClassifier(activation=mlpc_cv_model.best_params_['activation'],

                          alpha=mlpc_cv_model.best_params_['alpha'],

                          hidden_layer_sizes=mlpc_cv_model.best_params_['hidden_layer_sizes'],

                          solver=mlpc_cv_model.best_params_['solver']).fit(x_train_scaled,y_train)
y_pred = mlpc_tuned.predict(x_test_scaled)

print('Verified Test Score:',accuracy_score(y_test,y_pred))
diabetes = pd.read_csv('../input/diabetes/diabetes.csv')

data = diabetes.copy()

data.dropna(inplace=True)

y = data['Outcome']

x = data.drop(['Outcome'],axis=1)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30,random_state=42)
from sklearn.tree import DecisionTreeClassifier

cart = DecisionTreeClassifier()

cart_model = cart.fit(x_train,y_train)
!pip install astor
!pip install skompiler
from skompiler import skompile

print(skompile(cart_model.predict).to('python/code'))
y_pred = cart_model.predict(x_test)

print('Simple Test Error:',accuracy_score(y_test,y_pred))
cart_grid = {

    'max_depth':list(range(1,20)),

    'min_samples_split':list(range(2,100))

}

cart = DecisionTreeClassifier()

cart_cv_model = GridSearchCV(cart,cart_grid,n_jobs=-1,verbose=2).fit(x_train,y_train)
print('Best parameters for model:' + str(cart_cv_model.best_params_))
cart_tuned = DecisionTreeClassifier(max_depth=cart_cv_model.best_params_['max_depth'],

                                   min_samples_split=cart_cv_model.best_params_['min_samples_split']).fit(x_train,

                                                                                                         y_train)
y_pred = cart_tuned.predict(x_test)

print('Verified Test Error Score:',accuracy_score(y_test,y_pred))
diabetes = pd.read_csv('../input/diabetes/diabetes.csv')

data = diabetes.copy()

data.dropna(inplace=True)

y = data['Outcome']

x = data.drop(['Outcome'],axis=1)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30,random_state=42)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

rf_model = rf.fit(x_train,y_train)

y_pred = rf_model.predict(x_test)

print('Simple Test Error:',accuracy_score(y_test,y_pred))
rf_params = {

    'max_depth':[2,3,5,8,10],

    'max_features':[2,5,8],

    'n_estimators':[10,500,1000,2000],

    'min_samples_split':[2,5,10]

}

rf = RandomForestClassifier()

rf_cv_model = GridSearchCV(rf,rf_params,cv=10,n_jobs=-1,verbose=2).fit(x_train,y_train)
print('Best parameters for model:',rf_cv_model.best_params_)
rf_tuned = RandomForestClassifier(max_depth=rf_cv_model.best_params_['max_depth'],

                                 max_features=rf_cv_model.best_params_['max_features'],

                                 min_samples_split=rf_cv_model.best_params_['min_samples_split'],

                                 n_estimators=rf_cv_model.best_params_['n_estimators']).fit(x_train,y_train)
y_pred = rf_tuned.predict(x_test)

print('Verified Error Score:',accuracy_score(y_test,y_pred))
# Let's take a look at the importance levels of the parameters.



import matplotlib.pyplot as plt

Importance = pd.DataFrame({'Importance':rf_tuned.feature_importances_*100},

                         index=x_train.columns)

Importance.sort_values(by='Importance',axis=0,ascending=True).plot(kind='barh',color='Lime')

plt.xlabel('Importance Variable');
import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn.metrics import accuracy_score

diabetes = pd.read_csv('../input/diabetes/diabetes.csv')

data = diabetes.copy()

data.dropna(inplace=True)

y = data['Outcome']

x = data.drop(['Outcome'],axis=1)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30,random_state=42)
from sklearn.ensemble import GradientBoostingClassifier

Gradient = GradientBoostingClassifier()

gbm_model = Gradient.fit(x_train,y_train)
y_pred = gbm_model.predict(x_test)

print('Simple Test Error:',accuracy_score(y_test,y_pred))
gbm_params = {

    'learning_rate':[0.001,0.01,0.1,0.5],

    'n_estimators':[100,500,1000],

    'max_depth':[3,5,10],

    'min_samples_split':[2,5,10]

}

gbm = GradientBoostingClassifier()

gbm_cv_model = GridSearchCV(gbm,gbm_params,cv=10,n_jobs=-1,verbose=2)

gbm_cv_model.fit(x_train,y_train)
print('Best parameters of model:',gbm_cv_model.best_params_)
gbm_tuned = GradientBoostingClassifier(learning_rate=gbm_cv_model.best_params_['learning_rate'],

                                      max_depth=gbm_cv_model.best_params_['max_depth'],

                                      min_samples_split=gbm_cv_model.best_params_['min_samples_split'],

                                      n_estimators=gbm_cv_model.best_params_['n_estimators']).fit(x_train,y_train)
y_pred = gbm_tuned.predict(x_test)

print('Verified Test Error Score:',accuracy_score(y_test,y_pred))
diabetes = pd.read_csv('../input/diabetes/diabetes.csv')

data = diabetes.copy()

data.dropna(inplace=True)

y = data['Outcome']

x = data.drop(['Outcome'],axis=1)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30,random_state=42)
from xgboost import XGBClassifier

xgb_model = XGBClassifier().fit(x_train,y_train)

y_pred = xgb_model.predict(x_test)

print('Simple Test Error:',accuracy_score(y_test,y_pred))
xgb_params = {

    'n_estimators':[100,500,1000,2000],

    'subsample':[0.6,0.8,1.0],

    'max_depth':[3,4,5,6],

    'learning_rate':[0.1,0.01,0.02,0.05]

}

xgb = XGBClassifier()

xgb_cv_model = GridSearchCV(xgb,xgb_params,cv=10,n_jobs=-1,verbose=2).fit(x_train,y_train)
print('Best parameters of model:',xgb_cv_model.best_params_)
xgb_tuned = XGBClassifier(learning_rate=xgb_cv_model.best_params_['learning_rate'],

                         max_depth=xgb_cv_model.best_params_['max_depth'],

                         n_estimators=xgb_cv_model.best_params_['n_estimators'],

                         subsample=xgb_cv_model.best_params_['subsample']).fit(x_train,y_train)
y_pred = xgb_tuned.predict(x_test)

print('Verified Test Error Score:',accuracy_score(y_test,y_pred))
from warnings import filterwarnings

filterwarnings('ignore')

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import scipy as sp

from sklearn.cluster import KMeans

USA = pd.read_csv('../input/usadata/USA.csv')

data = USA.copy()

data.head()
# I'll take a few steps to fix the part that says Unnamed.

data.index = data.iloc[:,0]

data = data.iloc[:,1:5]

data.index.name = None

data.head() # Done
# We need control to dataset

data.isnull().any()

# Done
kmeans = KMeans(n_clusters=4)

k_fit = kmeans.fit(data)
print('Cluster:',k_fit.n_clusters)

print('************************************************************************************************')

print('Cluster Center:',k_fit.cluster_centers_)

print('************************************************************************************************')

print('Class Labels:',k_fit.labels_)
kmeans = KMeans(n_clusters=2)

k_fit = kmeans.fit(data)

cluster = k_fit.labels_

plt.scatter(data.iloc[:,0],data.iloc[:,1],c=cluster,cmap='viridis')

centers = k_fit.cluster_centers_

plt.scatter(centers[:,0],centers[:,1],c='black',s=200,alpha=0.5);
from mpl_toolkits.mplot3d import Axes3D

kmeans = KMeans(n_clusters=3)

k_fit = kmeans.fit(data)

clusters = k_fit.labels_

centers = k_fit.cluster_centers_

plt.rcParams['figure.figsize'] = (16,9)

fig = plt.figure()

ax = Axes3D(fig)

ax.scatter(data.iloc[:,0],data.iloc[:,1],data.iloc[:,2])

ax.scatter(centers[:,0],centers[:,1],centers[:,2],marker='*',c='#050505',s=1000);
kmeans = KMeans(n_clusters=3)

k_fit = kmeans.fit(data)

clusters = k_fit.labels_

pd.DataFrame({'States':data.index,

             'Cluster':clusters})[:10]
data['Clusters_NO'] = clusters

data.head()
from yellowbrick.cluster import KElbowVisualizer

kmeans = KMeans()

visualizer = KElbowVisualizer(kmeans,k=(2,20))

visualizer.fit(data)

visualizer.poof();
kmeans = KMeans(n_clusters=6) # Best cluster num. = 6

k_fit = kmeans.fit(data)

clusters = k_fit.labels_

pd.DataFrame({'States':data.index,

             'Clusters':clusters})[:10] # New Segmentation Datasets
import pandas as pd

import numpy as np

data = pd.read_csv('../input/usadata/USA.csv').copy()

data.index = data.iloc[:,0]

data = data.iloc[:,1:5]

data.index.name = None

data.head()
from scipy.cluster.hierarchy import linkage

hc_complete = linkage(data,'complete')

hc_average = linkage(data,'average')

hc_single = linkage(data,'single')

# We need to include these methods.
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import dendrogram

plt.figure(figsize=(15,10))

plt.title('Hierarchical Clustering - Dendogram')

plt.xlabel('Indexes')

plt.ylabel('Distance')

dendrogram(hc_complete,

          leaf_font_size=10);
from scipy.cluster.hierarchy import dendrogram

plt.figure(figsize=(15,10))

plt.title('Hierarchical Clustering - Dendogram')

plt.xlabel('Indexes')

plt.ylabel('Distance')

dendrogram(hc_complete,

          leaf_font_size=10);
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=4,affinity='euclidean',linkage='ward')

cluster.fit_predict(data)
pd.DataFrame({

    'States':data.index,

    'Clusters':cluster.fit_predict(data)

})[:10]
data['Clusters_NO'] = cluster.fit_predict(data)

data.head()
data = pd.read_csv('../input/usadata/USA.csv').copy()

data.index = data.iloc[:,0]

data = data.iloc[:,1:5]

data.index.name = None

data.head()
# To implement a PCA, we first need to standardize variables.

from sklearn.preprocessing import StandardScaler

data = StandardScaler().fit_transform(data)

data[:5]
from sklearn.decomposition import PCA

pca = PCA(n_components=2) # 2 component

pca_fit = pca.fit_transform(data)
compo_data = pd.DataFrame(data=pca_fit,columns=['1st_component','2nd_component'])

compo_data[:5]
pca.explained_variance_ratio_  # Variance explained for components

# Add these two together tells us that an average of 85% is represented.
# Visual answer to the question of how many components should we use.

pca = PCA().fit(data)

plt.plot(np.cumsum(pca.explained_variance_ratio_));
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib.colors import ListedColormap

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn.metrics import accuracy_score,confusion_matrix

from sklearn.neighbors import KNeighborsClassifier,NeighborhoodComponentsAnalysis,LocalOutlierFactor

from sklearn.decomposition import PCA

from warnings import filterwarnings

filterwarnings('ignore')
cancer = pd.read_csv('../input/cancer/cancer.csv')

data = cancer.copy()

data.drop(['Unnamed: 32','id'],axis=1,inplace=True)

data.head()
# I want to change the diagnosis variable name to target.

data = data.rename(columns= {'diagnosis':'target'})

data.head()
# How many M and how many B's are we examining them.

sns.countplot(data['target']);

print(data.target.value_counts())
# I need to convert the M and B in the target variable to 0 and 1.

# .strip () removes spaces in string expressions.

data['target'] = [1 if i.strip() == 'M'else 0 for i in data.target]
# 1 malignant so M

# 0 benign so B

data.head()
data.info()
data.describe()

# strictly standardization process is required for this data
# EDA

# Since all variables we have are numeric, we look at corr ().

ax = plt.figure(figsize=(15,8))

ax = sns.heatmap(data.corr());
# Since this is so confusing, I'll set a Threshold and cover those above it.

cor_mat = data.corr()

threshold = 0.75

filters = np.abs(cor_mat['target']) > threshold

corr_features = cor_mat.columns[filters].tolist()

ax = plt.figure(figsize=(15,8))

ax = sns.heatmap(data[corr_features].corr(),annot=True,linewidths=.3)

plt.title('Correlation Between Features w Corr Threshold 0.75');
# box plot

data_melted = pd.melt(data,id_vars='target',var_name='features',value_name='value')

ax = plt.figure(figsize=(15,8))

ax = sns.boxplot(x='features',y='value',hue='target',data=data_melted)

plt.xticks(rotation=90);

# Because the data is not standardized here, it becomes a strange table, we will use it later.
# Pair plot is one of the most effective methods used in numerical data.

# This will not look nice either, because the data needs to be standardized.

sns.pairplot(data[corr_features],diag_kind='kde',markers='+',hue='target');

# But I just used corr_features for images.
# Outliers

y = data['target']

x = data.drop(['target'],axis=1)

columns = x.columns.tolist()
clf = LocalOutlierFactor()

y_pred = clf.fit_predict(x)

x_score = clf.negative_outlier_factor_

outlier_score = pd.DataFrame()

outlier_score['score'] = x_score

outlier_score.head()
plt.figure(figsize=(8,5))

plt.scatter(x.iloc[:,0],x.iloc[:,1],color='b',s=3,label='Data Points')

plt.legend();
radius = (x_score.max() - x_score) / (x_score.max() - x_score.min())
plt.figure(figsize=(8,5))

plt.scatter(x.iloc[:,0],x.iloc[:,1],color='k',s=3,label='Data Points')

plt.scatter(x.iloc[:,0],x.iloc[:,1],s=1000*radius,edgecolors='r',facecolors='none',label='Outlier Scores')

plt.legend()

plt.show()
# We are looking at contradictory observations.

threshold = -2.5

filtre = outlier_score['score'] < threshold

outlier_index = outlier_score[filtre].index.tolist()

plt.figure(figsize=(8,5))

plt.scatter(x.iloc[outlier_index,0],x.iloc[outlier_index,1],color='b',s=50,label='Outlier')

plt.scatter(x.iloc[:,0],x.iloc[:,1],color='k',s=3,label='Data Points')

plt.scatter(x.iloc[:,0],x.iloc[:,1],s=1000*radius,edgecolors='r',facecolors='none',label='Outlier Scores')

plt.legend()

plt.show()
# Drop outliers

x = x.drop(outlier_index)

y = y.drop(outlier_index).values

# All of these, there are a lot of other variables waiting for outlier observations for columns 0 and 1.
# Train Test seperation

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)

# Standart

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)

x_test = scaler.transform(x_test)
# Let's visualize boxplot that was not visualized before.

x_train_df = pd.DataFrame(x_train,columns=columns)

x_train_df['target'] = y_train

data_melted = pd.melt(x_train_df,id_vars='target',var_name='features',value_name='value')

plt.figure(figsize=(15,8))

sns.boxplot(x='features',y='value',hue='target',data=data_melted)

plt.xticks(rotation=90)

plt.show()
knn = KNeighborsClassifier(n_neighbors=2)

knn.fit(x_train,y_train)

y_pred = knn.predict(x_test)

cm = confusion_matrix(y_test,y_pred)

acc = accuracy_score(y_test,y_pred)

score = knn.score(x_test,y_test)

print('Score:',score)

print('Confusion Matrix:',cm)

print('Basic Accuracy Score:',acc)
def knn_best_params(x_train,x_test,y_train,y_test):

    k_range = list(range(1,31))

    weight_options = ['uniform','distance']

    print()

    param_grid = dict(n_neighbors = k_range,weights=weight_options)

    knn = KNeighborsClassifier()

    grid = GridSearchCV(knn,param_grid,cv=10,scoring='accuracy')

    grid.fit(x_train,y_train)

    print('Best Training Score {} with parameters: {}'.format(grid.best_score_,grid.best_params_))

    print()

    

    knn = KNeighborsClassifier(**grid.best_params_)

    knn.fit(x_train,y_train)

    

    y_pred_test = knn.predict(x_test)

    y_pred_train = knn.predict(x_train)

    cm_test = confusion_matrix(y_test,y_pred_test)

    cm_train = confusion_matrix(y_train,y_pred_train)

    acc_test = accuracy_score(y_test,y_pred_test)

    acc_train = accuracy_score(y_train,y_pred_train)

    print('Test Score: {},Train Score: {}'.format(acc_test,acc_train))

    print()

    print('Confusion Matrix Test: {}'.format(cm_test))

    print('Confusion Matrix Train: {}'.format(cm_train))

    

    return grid
grid = knn_best_params(x_train,x_test,y_train,y_test)
scaler = StandardScaler()

x_scaled = scaler.fit_transform(x)

pca = PCA(n_components=2)

pca.fit(x_scaled)

x_reduced_pca = pca.transform(x_scaled)

pca_data = pd.DataFrame(x_reduced_pca,columns=['p1','p2'])

pca_data['target'] = y

plt.figure(figsize=(8,5))

sns.scatterplot(x='p1',y='p2',hue='target',data=pca_data)

plt.title('PCA: p1 vs p2')

plt.show()

# We reduced 30 dimensional data to 2 dimensions with PCA.
# Now we will do a chnn using 2 dimensional data.

x_train_pca,x_test_pca,y_train_pca,y_test_pca = train_test_split(x_reduced_pca,y,test_size=0.3,random_state=42)

grid_pca = knn_best_params(x_train_pca,x_test_pca,y_train_pca,y_test_pca)
# We use a visualization to see how the split is decided.

cmap_light = ListedColormap(['orange','cornflowerblue'])

cmap_bold = ListedColormap(['darkorange','darkblue'])

h = .05

X = x_reduced_pca

x_min,x_max = X[:,0].min() -1,X[:,0].max() + 1

y_min,y_max = X[:,1].min() -1 ,X[:,1].max() + 1

xx,yy = np.meshgrid(np.arange(x_min,x_max,h),

                   np.arange(y_min,y_max,h))

Z = grid_pca.predict(np.c_[xx.ravel(),yy.ravel()])

Z = Z.reshape(xx.shape)

plt.figure(figsize=(10,8))

plt.pcolormesh(xx,yy,Z,cmap=cmap_light)

plt.scatter(X[:,0],X[:,1],c=y,cmap=cmap_bold,

           edgecolors='k',s=20)

plt.xlim(xx.min(),xx.max())

plt.ylim(yy.min(),yy.max())

plt.title("%i-Class Classification (k= %i,weights = '%s')"%(len(np.unique(y)),grid_pca.best_estimator_.n_neighbors,grid_pca.best_estimator_.weights))
nca = NeighborhoodComponentsAnalysis(n_components=2,random_state=42)

nca.fit(x_scaled,y)

x_reduced_nca = nca.transform(x_scaled)

nca_data = pd.DataFrame(x_reduced_nca,columns=['p1','p2'])

nca_data['target'] = y

plt.figure(figsize=(10,8))

sns.scatterplot(x='p1',y='p2',hue='target',data=nca_data)

plt.title('NCA : p1 vs p2')

plt.show()
x_train_nca,x_test_nca,y_train_nca,y_test_nca = train_test_split(x_reduced_nca,y,test_size=0.3,random_state=42)

grid_nca = knn_best_params(x_train_nca,x_test_nca,y_train_nca,y_test_nca)