import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.pyplot import pie, axis, show

%matplotlib inline



from sklearn import metrics

from sklearn.dummy import DummyRegressor

from sklearn.metrics import mean_squared_error



import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv("https://raw.githubusercontent.com/insaid2018/Term-2/master/CaseStudy/Advertising.csv", index_col=0)

data.head()
data.shape
data.info()
data.describe()
f, axes = plt.subplots(2, 2, figsize=(7, 7), sharex=True)

sns.despine(left=True)



sns.distplot(data.sales, color="b", ax=axes[0, 0])



sns.distplot(data.TV, color="r", ax=axes[0, 1])



sns.distplot(data.radio, color="g", ax=axes[1, 0])



sns.distplot(data.newspaper, color="m", ax=axes[1, 1])
JG1 = sns.jointplot("newspaper", "sales", data=data, kind='reg')

JG2 = sns.jointplot("radio", "sales", data=data, kind='reg')

JG3 = sns.jointplot("TV", "sales", data=data, kind='reg')



#subplots migration

f = plt.figure()

for J in [JG1, JG2,JG3]:

    for A in J.fig.axes:

        f._axstack.add(f._make_key(A), A)
sns.pairplot(data, size = 2, aspect = 1.5)
sns.pairplot(data, x_vars=['TV', 'radio', 'newspaper'], y_vars='sales', size=5, aspect=1, kind='reg')
feature_cols = ['TV', 'radio', 'newspaper']

x = data[feature_cols]

x.head()
sns.heatmap(data.corr(), annot=True)
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler().fit(data)

data1 = scaler.transform(data)
data = pd.DataFrame(data1)                # Scaled data

data.head()
data.columns = ['TV','radio','newspaper','sales']

data.head()
feature_cols = ['TV', 'radio', 'newspaper']

X = data[feature_cols]

X.head()
print(type(X))

print(X.shape)
y = data.sales

y.head()
print(type(y))

print(y.shape)
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
x_train.head()
x_train.info()
print('Train cases as below')

print('x_train shape: ', x_train.shape)

print('y_train shape: ', y_train.shape)

print('\nTest cases as below')

print('x_test shape: ', x_test.shape)

print('y_test shape: ', y_test.shape)
from sklearn.linear_model import LinearRegression



linreg = LinearRegression()

linreg.fit(x_train, y_train)



y_pred_train = linreg.predict(x_train)

y_pred_test = linreg.predict(x_test)
def linear_reg(X, y, gridsearch = False):

    

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

    

    from sklearn.linear_model import LinearRegression

    linreg = LinearRegression()

    

    if not(gridsearch):

        linreg.fit(x_train, y_train) 



    else:

        from sklearn.model_selection import GridSearchCV

        parameters = {'normalize':[True,False], 'copy_X':[True, False]}

        linreg = GridSearchCV(linreg,parameters, cv = 10,refit = True)

        linreg.fit(x_train, y_train)                                                           

        print("Mean cross-validated score of the best_estimator : ", linreg.best_score_)  

        

        y_pred_test = linreg.predict(x_test)                                                   



        RMSE_test = (metrics.mean_squared_error(y_test, y_pred_test))                          

        print('RMSE for the test set is {}'.format(RMSE_test))



    return linreg
print('Intercept:',linreg.intercept_)

print('Coefficients:',linreg.coef_)
feature_cols.insert(0,'Intercept')

coef = linreg.coef_.tolist()            

coef.insert(0, linreg.intercept_)
eq1 = zip(feature_cols, coef)



for c1,c2 in eq1:

    print(c1,c2)
MAE_train = metrics.mean_absolute_error(y_train, y_pred_train)

MAE_test = metrics.mean_absolute_error(y_test, y_pred_test)



print('MAE for training set is {}'.format(MAE_train))

print('MAE for test set is {}'.format(MAE_test))
MSE_train = metrics.mean_squared_error(y_train, y_pred_train)

MSE_test = metrics.mean_squared_error(y_test, y_pred_test)



print('MSE for training set is {}'.format(MSE_train))

print('MSE for test set is {}'.format(MSE_test))
RMSE_train = np.sqrt( metrics.mean_squared_error(y_train, y_pred_train))

RMSE_test = np.sqrt(metrics.mean_squared_error(y_test, y_pred_test))



print('RMSE for training set is {}'.format(RMSE_train))

print('RMSE for test set is {}'.format(RMSE_test))
yhat = linreg.predict(x_train)

SS_Residual = sum((y_train-yhat)**2)

SS_Total = sum((y_train-np.mean(y_train))**2)

r_squared = 1 - (float(SS_Residual))/SS_Total

adjusted_r_squared = 1 - (1-r_squared)*(len(y_train)-1)/(len(y_train)-x_train.shape[1]-1)

print(r_squared, adjusted_r_squared)
yhat = linreg.predict(x_test)

SS_Residual = sum((y_test-yhat)**2)

SS_Total = sum((y_test-np.mean(y_test))**2)

r_squared = 1 - (float(SS_Residual))/SS_Total

adjusted_r_squared = 1 - (1-r_squared)*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1)

print(r_squared, adjusted_r_squared)
feature_cols = ['TV','radio']   

X_new = data[feature_cols]  

y = data.sales



linreg = linear_reg(X_new,y)
np.random.seed(123456)                                   # set a seed for reproducibility

nums = np.random.rand(len(data))

mask_suburban = (nums > 0.33) & (nums < 0.66)            # assign roughly one third of observations to each group

mask_urban = nums > 0.66

data['Area'] = 'rural'

data.loc[mask_suburban, 'Area'] = 'suburban'

data.loc[mask_urban, 'Area'] = 'urban'

data.head()
# Create three dummy variables using get_dummies



area_dummies = pd.get_dummies(data.Area, prefix='Area') 

area_dummies.head()
area_dummies = pd.get_dummies(data.Area, prefix='Area').iloc[:, 1:]

area_dummies.head()
# Concatenate the dummy variable columns onto the DataFrame (axis=0 means rows, axis=1 means columns)



data = pd.concat([data, area_dummies], axis=1)

data.head()
feature_cols = ['TV', 'radio', 'newspaper', 'Area_suburban', 'Area_urban']

X = data[feature_cols]  

y = data.sales

linreg = linear_reg(X,y)
feature_cols.insert(0,'Intercept')

coef = linreg.coef_.tolist()

coef.insert(0, linreg.intercept_)



eq1 = zip(feature_cols, coef)



for c1,c2 in eq1:

    print(c1,c2)
feature_cols = ['TV', 'radio', 'newspaper', 'Area_suburban', 'Area_urban']

X = data[feature_cols]  

y = data.sales

linreg = linear_reg(X,y,True)                     # for performing GridSearchCV
print('Intercept:',linreg.best_estimator_.intercept_)

print('Coefficients:',linreg.best_estimator_.coef_)
feature_cols.insert(0,'Intercept')

coef = linreg.best_estimator_.coef_.tolist()            

coef.insert(0, linreg.best_estimator_.intercept_) 
eq1 = zip(feature_cols, coef)



for c1,c2 in eq1:

    print(c1,c2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

y_pred_train = linreg.predict(X_train) 

y_pred_test = linreg.predict(X_test)
MAE_train = metrics.mean_absolute_error(y_train, y_pred_train)

MAE_test = metrics.mean_absolute_error(y_test, y_pred_test)



print('MAE for training set is {}'.format(MAE_train))

print('MAE for test set is {}'.format(MAE_test))
MSE_train = metrics.mean_squared_error(y_train, y_pred_train)

MSE_test = metrics.mean_squared_error(y_test, y_pred_test)



print('MSE for training set is {}'.format(MSE_train))

print('MSE for test set is {}'.format(MSE_test))
RMSE_train = np.sqrt( metrics.mean_squared_error(y_train, y_pred_train))

RMSE_test = np.sqrt(metrics.mean_squared_error(y_test, y_pred_test))



print('RMSE for training set is {}'.format(RMSE_train))

print('RMSE for test set is {}'.format(RMSE_test))
yhat = linreg.predict(X_train)

SS_Residual = sum((y_train-yhat)**2)

SS_Total = sum((y_train-np.mean(y_train))**2)

r_squared = 1 - (float(SS_Residual))/SS_Total

adjusted_r_squared = 1 - (1-r_squared)*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1)

print(r_squared, adjusted_r_squared)
yhat = linreg.predict(X_test)

SS_Residual = sum((y_test-yhat)**2)

SS_Total = sum((y_test-np.mean(y_test))**2)

r_squared = 1 - (float(SS_Residual))/SS_Total

adjusted_r_squared = 1 - (1-r_squared)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)

print(r_squared, adjusted_r_squared)