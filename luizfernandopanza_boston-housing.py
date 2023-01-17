# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
warnings.filterwarnings("ignore")
%matplotlib inline
df = pd.read_csv('/kaggle/input/boston-housing-dataset/HousingData.csv')
type (df)
print (df.shape)
df.head()
df = df.rename(columns = {'MEDV':'TARGET'})
df.head()
df.columns
df.dtypes
import pandas_profiling
profile = pandas_profiling.ProfileReport(df); profile
df.isnull().sum()
import missingno as msno
fig, ax = plt.subplots(figsize=(10, 5))
(1 - df.isnull().mean()).abs().plot.bar(ax=ax)
df1 = df.dropna()
print (df1.shape)
df1.isnull().sum()
medium_value = df['TARGET'].mean(); medium_value
squared_errors = pd.Series(medium_value - df['TARGET'])**2
SSE = np.sum(squared_errors)
print ('SSE: %01.f' % SSE)
hist_plot = squared_errors.plot(kind='hist')
x = df1.iloc[:,:-1]
y = df1['TARGET'].values
x.head()
y
import statsmodels.api as sm
import statsmodels.formula.api as smf
xc = sm.tools.tools.add_constant(x)
modelo = sm.OLS(y, xc)
modelo_v1 = modelo.fit()
# this model does not allow missing values
modelo_v1.summary()
x1 = x.drop(columns=['INDUS','AGE'])
# col INDUS and AGE > 0,05
x1.head()
xc1 = sm.tools.tools.add_constant(x1)
modelo1 = sm.OLS(y, xc1)
modelo_v2 = modelo1.fit()
modelo_v2.summary()
x = df1.iloc[:, :-1]
matriz_corr = x.corr()
print (matriz_corr)
observations = len(df1)
variables = df1.columns[:-1]
def visualize_correlation_matrix(data, hurdle = 0.0):
    R = np.corrcoef(data, rowvar = 0)
    R[np.where(np.abs(R) < hurdle)] = 0.0
    heatmap = plt.pcolor(R, cmap = mpl.cm.coolwarm, alpha = 0.8)
    heatmap.axes.set_frame_on(False)
    heatmap.axes.set_yticks(np.arange(R.shape[0]) + 0.5, minor = False)
    heatmap.axes.set_xticks(np.arange(R.shape[1]) + 0.5, minor = False)
    heatmap.axes.set_xticklabels(variables, minor = False)
    plt.xticks(rotation=90)
    heatmap.axes.set_yticklabels(variables, minor = False)
    plt.tick_params(axis = 'both', which = 'both', bottom = 'off', top = 'off', left = 'off', right = 'off') 
    plt.colorbar()
    plt.show()
visualize_correlation_matrix(x, hurdle = 0.5)
corr = np.corrcoef(x, rowvar = 0)
eigenvalues, eigenvectors = np.linalg.eig(corr)
print (eigenvalues)
print (eigenvectors[:,7])
# pos 7 is the smaller value
print (variables[2], variables[8], variables[9])
# the high values are:
observations = len(df1)
variables = df1.columns
from sklearn.datasets import load_boston
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
standardization = StandardScaler()
xst = standardization.fit_transform(x)
original_means = standardization.mean_
original_stds = standardization.scale_
xst = np.column_stack((xst, np.ones(observations)))
y = df1["TARGET"].values
X = df1.iloc[:, :-1]
import random
import numpy as np

def random_w( p ):
    return np.array([np.random.normal() for j in range(p)])

def hypothesis(X,w):
    return np.dot(X,w)

def loss(X,w,y):
    return hypothesis(X,w) - y

def squared_loss(X,w,y):
    return loss(X,w,y)**2

def gradient(X,w,y):
    gradients = list()
    n = float(len( y ))
    for j in range(len(w)):
        gradients.append(np.sum(loss(X,w,y) * X[:,j]) / n)
    return gradients

def update(X,w,y, alpha = 0.01):
    return [t - alpha*g for t, g in zip(w, gradient(X,w,y))]

def optimize(X,y, alpha = 0.01, eta = 10**-12, iterations = 1000):
    w = random_w(X.shape[1])
    path = list()
    for k in range(iterations):
        SSL = np.sum(squared_loss(X,w,y))
        new_w = update(X,w,y, alpha = alpha)
        new_SSL = np.sum(squared_loss(X,new_w,y))
        w = new_w
        if k>=5 and (new_SSL - SSL <= eta and new_SSL - SSL >= -eta):
            path.append(new_SSL)
            return w, path
        if k % (iterations / 20) == 0:
            path.append(new_SSL)
    return w, path                       
alpha = 0.01
w, path = optimize(xst, y, alpha, eta = 10**-12, iterations = 20000)
print ("Coeficientes finais padronizados: " + ', '.join(map(lambda x: "%0.4f" % x, w)))   
# undo the standard scaler to see the real values
unstandardized_betas = w[:-1] / original_stds
unstandardized_bias  = w[-1]-np.sum((original_means / original_stds) * w[:-1])
print ('%8s: %8.4f' % ('bias', unstandardized_bias))
for beta,varname in zip(unstandardized_betas, variables):
    print ('%8s: %8.4f' % (varname, beta))
modelo = linear_model.LinearRegression(normalize = False, fit_intercept = True)
modelo.fit(X,y)
standardization = StandardScaler()
Stand_coef_linear_reg = make_pipeline(standardization, modelo)
Stand_coef_linear_reg.fit(X,y)
for coef, var in sorted(zip(map(abs, Stand_coef_linear_reg.steps[1][1].coef_), df1.columns[:-1]), reverse = True):
    print ("%6.3f %s" % (coef,var))
modelo = linear_model.LinearRegression(normalize = False, fit_intercept = True)
def r2_est(X,y):
    return r2_score(y, modelo.fit(X,y).predict(X))
# this is the accuracy
r2 = r2_est(X,y) * 100
print ('Coeficiente R2: %0.1f' %  r2 + ' %')
# here we can see the value for each variable
r2_impact = list()
for j in range(X.shape[1]):
    selection = [i for i in range(X.shape[1]) if i!=j]
    r2_impact.append(((r2_est(X,y) - r2_est(X.values[:,selection],y)), df1.columns[j]))
    
for imp, varname in sorted(r2_impact, reverse = True):
    print ('%6.3f %s' %  (imp, varname))