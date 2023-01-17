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
import numpy as np 

from numpy import mean, absolute

import pandas as pd 

import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor 

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error

from sklearn import tree

from scipy.stats import t
df = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict.csv')
df.sample(5)
df.nunique()
df.dtypes
cols = list(df.columns)

target_col = 'Chance of Admit '

X_cols = cols[1:-1] #remove the Serial No and target from the feature columns



X = df[X_cols]

y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.6, random_state=0)
N = len(df) #set the total number of records to a variable N

data = {'Training': [len(X_train), len(X_train)/N],

       'Validation': [len(X_test), len(X_test)/N]}



dfPartitionValidation = pd.DataFrame.from_dict(data, orient='index', columns=['# Records', '% Records'])

dfPartitionValidation
def lowest_mse(X_train, X_test, y_train, y_test, max_depth=30,min_samples_leaf=30,max_leaf_nodes=30):

    """Prune model by testing values of max depth, min samples leaf and max leaf nodes. Return the model

    with the least complexity and lowest mean squared error"""

    mse = []

    maxDepthList = []

    minSamplesLeafList = []

    maxLeafNodes = []

    for i in range(1,max_depth):

        for j in range(2,min_samples_leaf):

            for k in range(2,max_leaf_nodes):

                dtree = DecisionTreeRegressor(random_state=0,max_depth=i, min_samples_leaf=j, max_leaf_nodes=k)

                dtree.fit(X_train, y_train)

                pred = dtree.predict(X_test)

                mse.append(mean_squared_error(y_test, pred))

                maxDepthList.append(i)

                minSamplesLeafList.append(j)

                maxLeafNodes.append(k)

    dfModels = pd.DataFrame({'mse': pd.Series(mse),

                    'max depth': pd.Series(maxDepthList),

                    'min samples leaf': pd.Series(minSamplesLeafList),

                    'max leaf nodes': pd.Series(maxLeafNodes)})

    #sort the dataframe by mse and complexity

    dfModels = dfModels.sort_values(by=['mse','max depth','min samples leaf','max leaf nodes']).reset_index()

    #return the pruned model

    prunedTree = DecisionTreeRegressor(random_state=0,max_depth=dfModels['max depth'][0], min_samples_leaf=dfModels['min samples leaf'][0], max_leaf_nodes=dfModels['max leaf nodes'][0])

    prunedTree.fit(X_train, y_train)

    return prunedTree
prunedTree = lowest_mse(X_train, X_test, y_train, y_test)
plt.figure(figsize=(25,15))

a = tree.plot_tree(prunedTree,

                  feature_names = list(X.columns),

                  filled = True,

                  rounded = True)
trainingTreeScores = pd.DataFrame(y_train.copy())

trainingTreeScores['Prediction'] = prunedTree.predict(X_train)

trainingTreeScores['Residual'] = trainingTreeScores[target_col]-trainingTreeScores['Prediction']



validationTreeScores = pd.DataFrame(y_test.copy())

validationTreeScores['Prediction'] = prunedTree.predict(X_test)

validationTreeScores['Residual'] = validationTreeScores[target_col]-validationTreeScores['Prediction']



data = {'SSE': [sum(trainingTreeScores['Residual']**2),sum(validationTreeScores['Residual']**2)],

        'MSE': [mean_squared_error(trainingTreeScores[target_col], trainingTreeScores['Prediction']), mean_squared_error(validationTreeScores[target_col], validationTreeScores['Prediction'])],

        'RMSE': [mean_squared_error(trainingTreeScores[target_col], trainingTreeScores['Prediction'])**.5, mean_squared_error(validationTreeScores[target_col], validationTreeScores['Prediction'])**.5],

        'MAD': [mean(absolute(trainingTreeScores['Residual'])),mean(absolute(validationTreeScores['Residual']))],

        'R2': [r2_score(trainingTreeScores[target_col], trainingTreeScores['Prediction']), r2_score(validationTreeScores[target_col], validationTreeScores['Prediction'])]}



dfTreeSummary = pd.DataFrame.from_dict(data, orient='index', columns=['Training', 'Validation'])

dfTreeSummary['%Difference'] = (dfTreeSummary['Training']-dfTreeSummary['Validation'])/dfTreeSummary['Training']*100

dfTreeSummary
regr = LinearRegression()

regr.fit(X_train,y_train)

pred = regr.predict(X_test)
#set precision to display in dataframes

pd.set_option('precision',4)



predictor = ['Intercept'] + X_cols

lrCoefficient = pd.DataFrame(zip(predictor,regr.coef_.reshape(7)),columns=['Predictor','Estimate'])

lrCoefficient
trainingScores = pd.DataFrame(y_train.copy())

trainingScores['Prediction'] = regr.predict(X_train)

trainingScores['Residual'] = trainingScores[target_col]-trainingScores['Prediction']



validationScores = pd.DataFrame(y_test.copy())

validationScores['Prediction'] = regr.predict(X_test)

validationScores['Residual'] = validationScores[target_col]-validationScores['Prediction']



data = {'SSE': [sum(trainingScores['Residual']**2),sum(validationScores['Residual']**2)],

        'MSE': [mean_squared_error(trainingScores[target_col], trainingScores['Prediction']), mean_squared_error(validationScores[target_col], validationScores['Prediction'])],

        'RMSE': [mean_squared_error(trainingScores[target_col], trainingScores['Prediction'])**.5, mean_squared_error(validationScores[target_col], validationScores['Prediction'])**.5],

        'MAD': [mean(absolute(trainingScores['Residual'])),mean(absolute(validationScores['Residual']))],

        'R2': [r2_score(trainingScores[target_col], trainingScores['Prediction']), r2_score(validationScores[target_col], validationScores['Prediction'])]}



dfRegrSummary = pd.DataFrame.from_dict(data, orient='index', columns=['Training', 'Validation'])

dfRegrSummary['%Difference'] = (dfRegrSummary['Training']-dfRegrSummary['Validation'])/dfRegrSummary['Training']*100

dfRegrSummary
def steyx(y,x):

    """Determines the standard error of the predicted y value for each actual y or

    the measure of the amount of error in the prediction of y for an individual x"""

    y_mean = mean(y)

    x_mean = mean(x)



    sumYSquare = sum((y-mean(y))**2)

    sumXSquare = sum((x-mean(x))**2)



    sumDiffSquare = sum((y-mean(y))*(x-mean(x)))**2



    dof = len(y)-2



    return ((1/dof)*(sumYSquare-(sumDiffSquare/sumXSquare)))**.5
T = t.ppf(1-(.05/2),(len(validationTreeScores)-1))



data = {'n': [len(validationTreeScores), len(validationScores)],

        'STEYX': [steyx(validationTreeScores[target_col], validationTreeScores['Prediction']), steyx(validationScores[target_col], validationScores['Prediction'])],

        'R2': [r2_score(validationTreeScores[target_col], validationTreeScores['Prediction']), r2_score(validationScores[target_col], validationScores['Prediction'])],

        'alpha': ['5%','5%'],

        'T.INV.2T': [T, T],

        '+/- Interval': [round(T*steyx(validationTreeScores[target_col], validationTreeScores['Residual']),2), round(T*steyx(validationScores[target_col], validationScores['Residual']),2)]

        }



dfValidation = pd.DataFrame.from_dict(data, orient='index', columns=['Tree', 'Regression'])

dfValidation
validationTreeScores['-95% PI'] = validationTreeScores['Prediction']- T*steyx(validationTreeScores[target_col], validationTreeScores['Prediction'])

validationTreeScores['+95% PI'] = validationTreeScores['Prediction']+ T*steyx(validationTreeScores[target_col], validationTreeScores['Prediction'])



validationScores['-95% PI'] = validationScores['Prediction']- T*steyx(validationScores[target_col], validationScores['Prediction'])

validationScores['+95% PI'] = validationScores['Prediction']+ T*steyx(validationScores[target_col], validationScores['Prediction'])
plt.figure(figsize=(25,10))



plt.subplot(121)



a = plt.plot(validationTreeScores['Prediction'],validationTreeScores['Prediction'],label='Prediction: {}'.format(target_col))

a1 = plt.plot(validationTreeScores['Prediction'],validationTreeScores[target_col],'ro', label=target_col)

a2 = plt.plot(validationTreeScores['Prediction'],validationTreeScores['-95% PI'],label='-95% PI')

a3 = plt.plot(validationTreeScores['Prediction'],validationTreeScores['+95% PI'],label='+95% PI')

plt.xlabel('Actual')

plt.ylabel('Estimate')

plt.title('Estimate vs Estimate Plot: Decision Tree')



plt.subplot(122)

a = plt.plot(validationScores['Prediction'],validationScores['Prediction'],label='Prediction: {}'.format(target_col))

a1 = plt.plot(validationScores['Prediction'],validationScores[target_col],'ro', label=target_col)

a2 = plt.plot(validationScores['Prediction'],validationScores['-95% PI'],label='-95% PI')

a3 = plt.plot(validationScores['Prediction'],validationScores['+95% PI'],label='+95% PI')

plt.xlabel('Actual')

plt.ylabel('Estimate')

plt.title('Estimate vs Estimate Plot: Linear Regression')



plt.legend()
plt.figure(figsize=(25,10))



plt.subplot(121)

n, bins, patchs = plt.hist(validationTreeScores['Residual'], 15, facecolor='b',alpha=.9)

plt.xlabel('Bin')

plt.ylabel('#')

plt.title('Histogram of Residuals: Decision Tree')



plt.subplot(122)

n, bins, patchs = plt.hist(validationScores['Residual'], 15, facecolor='b',alpha=.9)

plt.xlabel('Bin')

plt.ylabel('#')

plt.title('Histogram of Residuals: Linear Regression')

plt.show()
data = {'SSE': [sum(validationTreeScores['Residual']**2),sum(validationScores['Residual']**2)],

        'MSE': [mean_squared_error(validationTreeScores[target_col], validationTreeScores['Prediction']), mean_squared_error(validationScores[target_col], validationScores['Prediction'])],

        'RMSE': [mean_squared_error(validationTreeScores[target_col], validationTreeScores['Prediction'])**.5, mean_squared_error(validationScores[target_col], validationScores['Prediction'])**.5],

        'MAD': [mean(absolute(validationTreeScores['Residual'])),mean(absolute(validationScores['Residual']))],

        'R2': [r2_score(validationTreeScores[target_col], validationTreeScores['Prediction']), r2_score(validationScores[target_col], validationScores['Prediction'])]}



dfSummary = pd.DataFrame.from_dict(data, orient='index', columns=['Validation DT', 'Validation LR'])

dfSummary['%Difference'] = round((dfSummary['Validation DT']-dfSummary['Validation LR'])/dfSummary['Validation DT']*100,2)

dfSummary