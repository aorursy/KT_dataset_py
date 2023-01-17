import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.options.display.float_format = '{:.4f}'.format

from IPython.core.display import display, HTML
display(HTML('<style>.container {width:95% !important}</style>'))

import missingno as msno
import ppscore as pps
abt = pd.read_csv("train.csv")
wdf = abt.copy()
wdf.head()
wdf.info()
msno.matrix(wdf)
df = wdf.columns.difference(['id'])
x = wdf[df].columns.difference(['y'])
x = wdf[x]
x
y = wdf['y']
y


#threshold of 0.6

corr_df = x.corr()

corr_df.where(np.abs(x.corr()) > 0.6, other = 0, inplace = True )
#let us get a correlation heatmap too
plt.figure(figsize=(10,8))


ax = plt.axes()
sns.heatmap(corr_df,
            annot=True,
            linewidths=.5,
            center=0,
            cbar=False,
            cmap="YlGnBu")

ax.set_title("correlation heat map")
plt.show()
#let us also get PPS
pps_df = pps.matrix(x).pivot(columns = 'x', index = 'y', values = 'ppscore')
pps_df.where(np.abs(pps_df) > 0.6, other = 0, inplace = True)

plt.figure(figsize=(10,8))

ax = plt.axes()
sns.heatmap(pps_df,
            annot=True,
            linewidths=.5,
            center=0,
            cbar=False,
            cmap="YlGnBu")
ax.set_title('Predictive Power Score')
sns.pairplot(x)
#let us look at box plot of each
columns_list = x.columns
number_of_columns = len(x.columns)
plots_per_rows = 3
number_of_rows = int(np.ceil(number_of_columns/plots_per_rows))


colindex = 0
fig, axis = plt.subplots(figsize = (20, 10), nrows = number_of_rows, ncols = plots_per_rows)

for r in range(0,number_of_rows):
    for c in range(0,plots_per_rows):
        sns.boxplot(x[columns_list[colindex]], ax = axis[r][c])
        colindex += 1
        if colindex >= len(columns_list):
            break
#every variable is perfectly centered. so we ahve an ideal situation here.
#lets go ahead and craete our linear regression model now.

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
#import all the metrics related to model selection
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score, precision_score, recall_score, mean_squared_error

#import all the capabilities of model_selection
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

#we will now work on the train data

x_train_model = x_train.copy()
x_test_model = x_test.copy()
#we define the function for Polynomial regression

def polyreg(x_train, x_test , y_train, y_test):
    #we define the hyperparameters
    degree = [1, 2, 3 , 4, 5]
    interaction_only = [True , False]
    include_bias = [True, False]
    normalize = [True, False]
    fit_intercept = [True, False]
    order = ['C', 'F']
    
    #instantiate polynomial features :

    polyfeat = PolynomialFeatures()

    linreg = LinearRegression()

    pipe = Pipeline(steps=[('polyfeat', polyfeat), ('linreg', linreg)])

    #we define the param grid for the gridsearchcv:

    param_grid = {
    'polyfeat__degree': degree,
    'polyfeat__interaction_only': interaction_only,
    'polyfeat__include_bias': include_bias,
    'linreg__normalize':normalize,
    'linreg__fit_intercept': fit_intercept
    }


    polyregmodel = GridSearchCV(pipe, param_grid, n_jobs=6, cv = 10, iid = False)

    polyregmodel.fit(x_train, y_train)

    y_predict = polyregmodel.predict(x_test)
    
       

    return ("PolynomialRegression", polyregmodel, polyregmodel.best_params_, r2_score(y_test, y_predict), mean_squared_error(y_test, y_predict) )
#calling the regression function 
    
regresultup = polyreg(x_train_model, x_test_model , y_train, y_test)
print("regression r2: ", regresultup[2])
print("regression mean_squared_error: ", regresultup[3])
print("regression mean_squared_error: ", regresultup[4])


regresultup[4]
#load the test data
testdf = pd.read_csv('test.csv')
testdf.head()
x_testcolumns  = testdf.columns.difference(['id'])
test = testdf[x_testcolumns]
y_test_predict = regresultup[1].predict(test)
y_test_predict
testdf.shape
pd.DataFrame(testdf['id'], y_test_predict)
result_df = pd.DataFrame(data = dict(id = testdf['id'], y=y_test_predict))
result_df.to_csv("submission1.csv", index = False)
