import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

# import pathlib



%matplotlib inline
path = '../input'
data = pd.read_csv(f'{path}/ex1data1.txt', names=['population','profit'])

data.head()
# x0 = np.ones(y.shape)[:,np.newaxis]

# x = data['population'].values[:,np.newaxis]

# X = np.concatenate((x0,x),1)

# y = data['profit'].values
X = np.column_stack([np.ones(len(data), dtype=np.float32),data['population'].values])

y = data['profit'].values
from pandas.plotting import scatter_matrix



axes = scatter_matrix(data, figsize=(5,5));

n = len(data.columns)

for i in range(n):

    v=axes[i,0]

    v.yaxis.label.set_rotation(0)

    v.yaxis.label.set_ha('right')

    v.set_yticks(())

    h = axes[n-1,i]

    h.xaxis.label.set_rotation(90)

    h.set_xticks(());
plt.plot(data['population'],data['profit'],'bo')

plt.xlabel('Population')

plt.ylabel('Profit')

plt.title('Population vs Profit');
data.corr()
outlier_marks = {'markerfacecolor':'r', 'marker':'s'}

plt.boxplot(data['population'],flierprops=outlier_marks, vert=False)

plt.title('Population');
from statsmodels.graphics.gofplots import qqplot



qqplot(data['profit'], line='s');
def getBinwidth(values):

    '''Function to calculate binwidth for Histograms

    ref: https://www.qimacros.com/histogram-excel/how-to-determine-histogram-bin-interval/'''

    values_range = max(values) - min(values)

    noofbins = round(len(values)**1/2)

    binwidth = values_range/noofbins

    if binwidth<5:

        return 5

    else:

        return binwidth

getBinwidth(data['population'])
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(11,5))

ax1.hist(data['population'], edgecolor='black', color='blue')

ax1.set_title('Population')

ax2.hist(data['profit'],  edgecolor='black', color='blue')

ax2.set_title('Profit');
import seaborn as sns



fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(11,5))

sns.distplot(data['population'], ax=ax[0], hist=False, kde=True,

             kde_kws={'shade': True, 'linewidth':3}, label='population')

sns.distplot(data['profit'],ax=ax[1], hist=False, kde=True, 

             kde_kws={'shade': True, 'linewidth':3}, label='profit', color='darkorange')

ax[0].set_ylabel('Density')

ax[1].set_ylabel('Density');
from scipy.stats import shapiro, normaltest, anderson



response_var = data['profit']

alpha = 0.05



# Anderson-Darling Test

result = anderson(response_var)

print(f'Anderson statistics: {result.statistic:.3f}')



for i in range(len(result.critical_values)):

    sl, cv = result.significance_level[i], result.critical_values[i]

    if result.statistic < cv:

        print(f'Data looks normal (fail to reject H0): ({sl:.3f}, {cv:.3f}) ')

    else:

        print(f'Data looks not normal (reject H0): ({sl:.3f}, {cv:.3f}) ')

        

# D'Agostino's K^2 Test

print()

stat, p_val = normaltest(response_var)

print(f"D'Agostino's K^2 Test: Statistic={stat:.3f} p-value={p_val:.7f}")

if p_val > alpha:

    print('Data looks normal (fail to reject H0)')

else:

    print('Data looks normal (fail to reject H0)')

    

# Shapiro-Wilk test

print()

stat, p_val = shapiro(response_var)

print(f'Shapiro-Wilk Test: Statistic={stat:.3f} p-value={p_val:.7f}')

if p_val > alpha:

    print('Data looks normal (fail to reject H0)')

else:

    print('Data looks normal (fail to reject H0)')
def cal_coef(X,y):

    """Function to calculate coefficients or w or theta or b0 & b1"""

    coeffs = np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,y)) #

    

    #Similar way to calculate coeffs

    # coeffs = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    return coeffs
def predict_normeq(new_x,coeffs):

    """Function to predict linear regression using normal equation"""

    

    new_y = np.dot(new_x, coeffs)

    return new_y
x_test = 6.1101



coeffs = cal_coef(X,y) # b0-intercept, b1-slope

print(f'Intercept: {coeffs[0]} Slope:{coeffs[1]}')



#add ones to x matrix

if np.isscalar(x_test):

    new_x = np.array([1, x_test])

else:

    new_x = np.column_stack([np.ones(len(x_test), dtype=np.float32),x_test])

        

normeq_preds = predict_normeq(X,coeffs)

print(f'Profit Prediction for {x_test*1000} is {predict_normeq(new_x,coeffs)*10000}')
SSE = sum((y-normeq_preds)**2) # Sum of squared error

SST = sum((y-np.mean(y))**2) # Sum of squared total

n=len(X) # Number of obeservations

q=len(coeffs) # Number of coefficients

k=len(coeffs) # Number of parameters

MSE = SSE/(n-q) # Mean Squared Error

MST = SST/(n-1) # Mean Squared Total



R_squared = 1-(SSE/SST) # R Square

Adj_Rsquared = 1-(MSE/MST) # Adjusted R square

std_err = np.sqrt(MSE) # Standard Error or Root mean squared error

MSR = (SST-SSE)/(q-1) # Mean Squared Regression

f_static = MSR/MSE # F Statics

MAPE = sum(np.abs(y-normeq_preds))/n # Mean Absolute Percentage Error



print(f'R Squared: {R_squared}\n\nAdj. R-Squared: {Adj_Rsquared}\n\nStd.Error: {std_err}\n\nF Static: {f_static}\n')

print(f'MeanAbsPercErr. : {MAPE}')

# Need to calculate MinMaxAccuracy, AIC and BIC values
from statsmodels.regression.linear_model import OLS
res = OLS(y,X).fit()

coeffs_ols = res.params

print(f'coefficients : {coeffs_ols}')
ols_preds= res.predict()
res.summary()
print(f'Profit Prediction for {x_test*1000} is {res.predict(new_x)*10000}')
np.allclose(normeq_preds, ols_preds)
from sklearn.linear_model import LinearRegression

from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
lr = LinearRegression()

model = lr.fit(X,y)

print(f'R Squared: {model.score(X,y)}')

print(f'intercept: {model.intercept_} | coef: {model.coef_}')
skl_preds = model.predict(X)

model.predict(new_x.reshape(1,-1))
explained_variance = explained_variance_score(y,skl_preds) # explained variance = r2

r_squared = r2_score(y,skl_preds) # r square

mae = mean_absolute_error(y,skl_preds) # Mean absolute Percentag error

mse = mean_squared_error(y,skl_preds) # Mean squared error

print(f'explained Var.: {explained_variance}\n\nR Square: {r_squared}\n\nMean Abs Error:{mae}\n\nMean Squared Error:{mse}')
np.allclose(normeq_preds,skl_preds)
def gradPredict(X,theta):

    return np.dot(X,theta)



def calculateCost(X,y,theta):

    preds = gradPredict(X,theta)

    return ((preds-y)**2).mean()/2



def gradLinearReg(X,y,alpha=0.005,maxIter=15000):

    thetas = []

    costs = []

    i = 0

    theta = np.zeros(2, dtype=np.float32)

    converged=False

    while converged==False:

        preds = gradPredict(X,theta)

        theta0 = theta[0] - alpha*(preds-y).mean()

        theta1 = theta[1] - alpha*((preds-y)*X[:,1]).mean()

        theta = np.array([theta0,theta1])

        J = calculateCost(X,y,theta)

        #print(costs[-1],J)

        #current_threshold = costs[-1]-J

        if i > 0 and (costs[-1]-J) == 0.0001:

            converged = True

        if maxIter == i:

            converged = True

        thetas.append(theta)

        costs.append(J)

        i += 1

    print(f'theta: {thetas[np.argmin(costs)]} | cost: {min(costs)} | iteration: {np.argmin(costs)}  ')

    return thetas[np.argmin(costs)]
thetas = gradLinearReg(X,y)
grad_preds = gradPredict(X,thetas)
np.allclose(grad_preds, normeq_preds)
from sklearn.linear_model import SGDRegressor

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
sgd = SGDRegressor(max_iter=10000, alpha=0.10, epsilon=1e-7)

model = sgd.fit(X,y)

preds = model.predict(X)
sgd
print(model.intercept_, model.coef_)
x = data['population'].values

y = data['profit'].values



def getCoeffs(x,y):

    cov = np.cov(x,y)[0,1]

    var = np.var(x)

    coef = cov/var

    intercept = np.mean(y) - coef * np.mean(x)

    return (coef,intercept)



def predict(x,y):

    coeff,intercept = getCoeffs(x,y)

    preds = (coeff*x)+intercept

    return preds

    

coeff, intercept = getCoeffs(x,y)

print(f'Coefficient: {coeff}')

print(f'Intercept: {intercept}')

npcov_preds = predict(x,y)
np.allclose(npcov_preds, normeq_preds) # little change of decimals in both coeffs
x = data['population']

y = data['profit']



def mean(vals):

    return sum(vals)/float(len(vals))



def variance(vals):

    val_mean = mean(vals)

    return sum([(x-val_mean)**2 for x in vals])



def covariance(x,x_mean,y,y_mean):

    cov = 0.0

    for i in range(len(x)):

        cov += (x[i]-x_mean) * (y[i]-y_mean)

    return cov



def getBetas(x,y):

    x_mean = mean(x)

    y_mean = mean(y)

    cov = covariance(x,x_mean,y,y_mean)

    x_var = variance(x)

    b1 = cov/x_var

    b0 = np.mean(y) - b1 * np.mean(x)

    return b0,b1



def predict(x,y):

    intercept,coeff = getBetas(x,y)

    preds = intercept+(coeff*x)

    return preds



intercept, coeff = getBetas(x,y)

print(f'Intercept: {intercept}\n\nCoefficient: {coeff}')

fcov_preds = predict(x,y)
np.allclose(fcov_preds,normeq_preds)
with plt.style.context('dark_background'):

    plt.figure(figsize=(12,7))

    plt.plot(x,y,'yo',x,fcov_preds,'r-',fillstyle='full')

    plt.xlabel('Population')

    plt.ylabel('Profit')

    plt.title('Linear Regression');