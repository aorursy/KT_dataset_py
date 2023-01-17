import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline



# preprocessing

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold

import pandas_profiling as pp



# models

from sklearn.linear_model import LinearRegression, SGDRegressor, RidgeCV

from sklearn.svm import SVR, LinearSVR

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor 

from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor, VotingRegressor 

from sklearn.neural_network import MLPRegressor

from sklearn.tree import DecisionTreeRegressor

import sklearn.model_selection

from sklearn.model_selection import cross_val_predict as cvp

from sklearn import metrics

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import xgboost as xgb

import lightgbm as lgb

from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV, LinearRegression, ElasticNet,  HuberRegressor

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold, StratifiedKFold, RandomizedSearchCV







import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')

matplotlib.rcParams['font.family'] = "Arial"



from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly as py

import plotly.graph_objs as go

from plotly.subplots import make_subplots



init_notebook_mode(connected=True)



import collections

import itertools



import scipy.stats as stats

from scipy.stats import norm

from scipy.special import boxcox1p



import statsmodels

import statsmodels.api as sm

#print(statsmodels.__version__)



from sklearn.preprocessing import scale, StandardScaler, RobustScaler, OneHotEncoder

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold, StratifiedKFold, RandomizedSearchCV

from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV, LinearRegression, ElasticNet,  HuberRegressor

from sklearn.metrics import mean_squared_error, r2_score

from xgboost import XGBRegressor

from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.svm import SVR

from sklearn.kernel_ridge import KernelRidge

from sklearn.utils import resample



from xgboost import XGBRegressor



#Model interpretation modules

import eli5

import lime

import lime.lime_tabular

import shap

shap.initjs()



import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# model tuning

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval



import warnings

warnings.filterwarnings("ignore")
# Distribution graphs (histogram/bar graph) of column data

def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):

    nunique = df.nunique()

    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values

    nRow, nCol = df.shape

    columnNames = list(df)

    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow

    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')

    for i in range(min(nCol, nGraphShown)):

        plt.subplot(nGraphRow, nGraphPerRow, i + 1)

        columnDf = df.iloc[:, i]

        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):

            valueCounts = columnDf.value_counts()

            valueCounts.plot.bar()

        else:

            columnDf.hist()

        plt.ylabel('counts')

        plt.xticks(rotation = 90)

        plt.title(f'{columnNames[i]} (column {i})')

    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)

    plt.show()
# Correlation matrix

def plotCorrelationMatrix(df, graphWidth):

    filename = df.dataframeName

    df = df.dropna('columns') # drop columns with NaN

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    if df.shape[1] < 2:

        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')

        return

    corr = df.corr()

    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')

    corrMat = plt.matshow(corr, fignum = 1)

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)

    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.gca().xaxis.tick_bottom()

    plt.colorbar(corrMat)

    plt.title(f'Correlation Matrix for {filename}', fontsize=15)

    plt.show()
# Scatter and density plots

def plotScatterMatrix(df, plotSize, textSize):

    df = df.select_dtypes(include =[np.number]) # keep only numerical columns

    # Remove rows and columns that would lead to df being singular

    df = df.dropna('columns')

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    columnNames = list(df)

    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots

        columnNames = columnNames[:10]

    df = df[columnNames]

    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')

    corrs = df.corr().values

    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):

        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)

    plt.suptitle('Scatter and Density Plot')

    plt.show()
valid_part = 0.3
train0 = pd.read_csv('/kaggle/input/ammonium-prediction-in-river-water/train.csv')
train0.head(10)
train0.info()
nRowsRead = 1000 # specify 'None' if want to read whole file

# test.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows

df1 = pd.read_csv('/kaggle/input/ammonium-prediction-in-river-water/test.csv', delimiter=',', nrows = nRowsRead)

df1.dataframeName = 'test.csv'

nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')
df1.head()
plotPerColumnDistribution(df1, 10, 5)
plotCorrelationMatrix(df1, 8)
plotScatterMatrix(df1, 20, 10)
nRowsRead = 1000 # specify 'None' if want to read whole file

# train.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows

df2 = pd.read_csv('/kaggle/input/ammonium-prediction-in-river-water/train.csv', delimiter=',', nrows = nRowsRead)

df2.dataframeName = 'train.csv'

nRow, nCol = df2.shape

print(f'There are {nRow} rows and {nCol} columns')
df2.head(5)
plotPerColumnDistribution(df2, 10, 5)
plotCorrelationMatrix(df2, 8)
plotScatterMatrix(df2, 20, 10)
pp.ProfileReport(train0)
train0 = train0.drop(['Id','3','4','5','6','7'], axis = 1)

train0 = train0.dropna()

train0.info()
train0.head(3)
target_name = 'target'
# For boosting model

train0b = train0

train_target0b = train0b[target_name]

train0b = train0b.drop([target_name], axis=1)

# Synthesis valid as test for selection models

trainb, testb, targetb, target_testb = train_test_split(train0b, train_target0b, test_size=valid_part, random_state=0)
train_target0 = train0[target_name]

train0 = train0.drop([target_name], axis=1)
#For models from Sklearn

scaler = StandardScaler()

train0 = pd.DataFrame(scaler.fit_transform(train0), columns = train0.columns)
train0.head(3)
len(train0)
# Synthesis valid as test for selection models

train, test, target, target_test = train_test_split(train0, train_target0, test_size=valid_part, random_state=0)
train.head(3)
test.head(3)
train.info()
test.info()
acc_train_r2 = []

acc_test_r2 = []

acc_train_d = []

acc_test_d = []

acc_train_rmse = []

acc_test_rmse = []
def acc_d(y_meas, y_pred):

    # Relative error between predicted y_pred and measured y_meas values

    return mean_absolute_error(y_meas, y_pred)*len(y_meas)/sum(abs(y_meas))



def acc_rmse(y_meas, y_pred):

    # RMSE between predicted y_pred and measured y_meas values

    return (mean_squared_error(y_meas, y_pred))**0.5
def acc_boosting_model(num,model,train,test,num_iteration=0):

    # Calculation of accuracy of boosting model by different metrics

    

    global acc_train_r2, acc_test_r2, acc_train_d, acc_test_d, acc_train_rmse, acc_test_rmse

    

    if num_iteration > 0:

        ytrain = model.predict(train, num_iteration = num_iteration)  

        ytest = model.predict(test, num_iteration = num_iteration)

    else:

        ytrain = model.predict(train)  

        ytest = model.predict(test)



    print('target = ', targetb[:5].values)

    print('ytrain = ', ytrain[:5])



    acc_train_r2_num = round(r2_score(targetb, ytrain) * 100, 2)

    print('acc(r2_score) for train =', acc_train_r2_num)   

    acc_train_r2.insert(num, acc_train_r2_num)



    acc_train_d_num = round(acc_d(targetb, ytrain) * 100, 2)

    print('acc(relative error) for train =', acc_train_d_num)   

    acc_train_d.insert(num, acc_train_d_num)



    acc_train_rmse_num = round(acc_rmse(targetb, ytrain) * 100, 2)

    print('acc(rmse) for train =', acc_train_rmse_num)   

    acc_train_rmse.insert(num, acc_train_rmse_num)



    print('target_test =', target_testb[:5].values)

    print('ytest =', ytest[:5])

    

    acc_test_r2_num = round(r2_score(target_testb, ytest) * 100, 2)

    print('acc(r2_score) for test =', acc_test_r2_num)

    acc_test_r2.insert(num, acc_test_r2_num)

    

    acc_test_d_num = round(acc_d(target_testb, ytest) * 100, 2)

    print('acc(relative error) for test =', acc_test_d_num)

    acc_test_d.insert(num, acc_test_d_num)

    

    acc_test_rmse_num = round(acc_rmse(target_testb, ytest) * 100, 2)

    print('acc(rmse) for test =', acc_test_rmse_num)

    acc_test_rmse.insert(num, acc_test_rmse_num)
def acc_model(num,model,train,test):

    # Calculation of accuracy of model акщь Sklearn by different metrics   

  

    global acc_train_r2, acc_test_r2, acc_train_d, acc_test_d, acc_train_rmse, acc_test_rmse

    

    ytrain = model.predict(train)  

    ytest = model.predict(test)



    print('target = ', target[:5].values)

    print('ytrain = ', ytrain[:5])



    acc_train_r2_num = round(r2_score(target, ytrain) * 100, 2)

    print('acc(r2_score) for train =', acc_train_r2_num)   

    acc_train_r2.insert(num, acc_train_r2_num)



    acc_train_d_num = round(acc_d(target, ytrain) * 100, 2)

    print('acc(relative error) for train =', acc_train_d_num)   

    acc_train_d.insert(num, acc_train_d_num)



    acc_train_rmse_num = round(acc_rmse(target, ytrain) * 100, 2)

    print('acc(rmse) for train =', acc_train_rmse_num)   

    acc_train_rmse.insert(num, acc_train_rmse_num)



    print('target_test =', target_test[:5].values)

    print('ytest =', ytest[:5])

    

    acc_test_r2_num = round(r2_score(target_test, ytest) * 100, 2)

    print('acc(r2_score) for test =', acc_test_r2_num)

    acc_test_r2.insert(num, acc_test_r2_num)

    

    acc_test_d_num = round(acc_d(target_test, ytest) * 100, 2)

    print('acc(relative error) for test =', acc_test_d_num)

    acc_test_d.insert(num, acc_test_d_num)

    

    acc_test_rmse_num = round(acc_rmse(target_test, ytest) * 100, 2)

    print('acc(rmse) for test =', acc_test_rmse_num)

    acc_test_rmse.insert(num, acc_test_rmse_num)
random_forest = GridSearchCV(estimator=RandomForestRegressor(), param_grid={'n_estimators': [100, 1000]}, cv=5)

random_forest.fit(train, target)

print(random_forest.best_params_)

acc_model(2,random_forest,train,test)
# Extra Trees Regressor



etr = ExtraTreesRegressor()

etr.fit(train, target)

acc_model(12,etr,train,test)
# AdaBoost Regression



Ada_Boost = AdaBoostRegressor()

Ada_Boost.fit(train, target)

acc_model(13,Ada_Boost,train,test)
# Stochastic Gradient Descent



sgd = SGDRegressor()

sgd.fit(train, target)

acc_model(4,sgd,train,test)
models = pd.DataFrame({

    'Model': ['Random Forest','ExtraTreesRegressor', 

              'AdaBoostRegressor', 'SGDRegressor'],

    

    'r2_train': acc_train_r2,

    'r2_test': acc_test_r2,

    'd_train': acc_train_d,

    'd_test': acc_test_d,

    'rmse_train': acc_train_rmse,

    'rmse_test': acc_test_rmse

                     })
pd.options.display.float_format = '{:,.2f}'.format
print('Prediction accuracy for models by R2 criterion - r2_test')

models.sort_values(by=['r2_test', 'r2_train'], ascending=False)
print('Prediction accuracy for models by relative error - d_test')

models.sort_values(by=['d_test', 'd_train'], ascending=True)
print('Prediction accuracy for models by RMSE - rmse_test')

models.sort_values(by=['rmse_test', 'rmse_train'], ascending=True)
# Plot

plt.figure(figsize=[25,6])

xx = models['Model']

plt.tick_params(labelsize=14)

plt.plot(xx, models['r2_train'], label = 'r2_train')

plt.plot(xx, models['r2_test'], label = 'r2_test')

plt.legend()

plt.title('R2-criterion for 3 popular models for train and test datasets')

plt.xlabel('Models')

plt.ylabel('R2-criterion, %')

plt.xticks(xx, rotation='vertical')

plt.savefig('graph.png')

plt.show()
# Plot

plt.figure(figsize=[25,6])

xx = models['Model']

plt.tick_params(labelsize=14)

plt.plot(xx, models['d_train'], label = 'd_train')

plt.plot(xx, models['d_test'], label = 'd_test')

plt.legend()

plt.title('Relative errors for 3 popular models for train and test datasets')

plt.xlabel('Models')

plt.ylabel('Relative error, %')

plt.xticks(xx, rotation='vertical')

plt.savefig('graph.png')

plt.show()
# Plot

plt.figure(figsize=[25,6])

xx = models['Model']

plt.tick_params(labelsize=14)

plt.plot(xx, models['rmse_train'], label = 'rmse_train')

plt.plot(xx, models['rmse_test'], label = 'rmse_test')

plt.legend()

plt.title('RMSE for 3 popular models for train and test datasets')

plt.xlabel('Models')

plt.ylabel('RMSE, %')

plt.xticks(xx, rotation='vertical')

plt.savefig('graph.png')

plt.show()
testn = pd.read_csv('/kaggle/input/ammonium-prediction-in-river-water/test.csv')

testn.info()
testn = testn.drop(['Id','3','4','5','6','7'], axis = 1)

testn.head(3)
#For models from Sklearn

testn = pd.DataFrame(scaler.transform(testn), columns = testn.columns)
random_forest.fit(train0, train_target0)

random_forest.predict(testn)[:3]
etr.fit(train0, train_target0)

etr.predict(testn)[:3]
Ada_Boost.fit(train0, train_target0)

Ada_Boost.predict(testn)[:3]
sgd.fit(train0, train_target0)

sgd.predict(testn)[:3]