# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import xlrd 

from operator import itemgetter

import matplotlib.pyplot as plt

import seaborn as sns

import tensorflow as tf

from numpy.random import seed

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import fbeta_score, confusion_matrix, precision_recall_curve, auc, make_scorer

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.tree import DecisionTreeClassifier

from tensorflow.keras.layers import Dense

from tensorflow.keras.models import Sequential

from scipy import stats

from sklearn import preprocessing

import numpy as np

from sklearn import metrics, svm

from sklearn.linear_model import LogisticRegression

from sklearn import preprocessing

from sklearn import utils

from sklearn import datasets, linear_model

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

import math

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error



seed(1)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def Questions1234(df):

    # Dimension of the dataset

        

    print('the shape of the df is \n' ,np.shape(df))

    numberOfNulls = df.isna().sum().sum()

    print('There are {0} Nulls in the data'.format(numberOfNulls))

    df.MEDV.describe()

    fig = plt.figure(figsize=(10, 5))

    ax = sns.distplot(df.MEDV)

    plt.show()

    ax = sns.boxplot(y = df.MEDV)

    plt.show()

    print('we see the MEDV have ~ normal distribution and there are some outliers on the MEDV variable')

    # Correlation mat

    corrMatt = df[df.columns.values].corr().abs()

    mask = np.array(corrMatt)

    mask[np.tril_indices_from(mask)] = False

    fig,ax= plt.subplots()

    fig.set_size_inches(20,10)

    sns.heatmap(corrMatt, mask=mask,vmax=.8, square=True,annot=True)

    print('From correlation matrix, we see TAX and RAD are highly correlated'

          'features. The columns LSTAT, RM has a correlation score above 0.69 '

          'with MEDV which is a good indication of using as predictors, '

          'However RM and LSTAT are also pretty correlated to each other '

          'what might be not a good idea to pick both of them Lets plot these columns against MEDV')

    #Plot top 2 correlated features against MEDV

    min_max_scaler = preprocessing.MinMaxScaler()

    column_sels = ['LSTAT', 'RM']

    x = df.loc[:,column_sels]

    y = df['MEDV']

    x = pd.DataFrame(data=min_max_scaler.fit_transform(x), columns=column_sels)

    fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(10, 10))

    index = 0

    for i, k in enumerate(column_sels):

        sns.regplot(y=y, x=x[k], ax=axs[i])

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)

    # list the top correlated features to the target 'MEDV' (with abs value)

    CorList = df.corr().unstack().sort_values().abs()['MEDV']

    CorList = CorList.sort_values(ascending=False)

    print(CorList)

    
class TestML:

    def __init__(self,df):

        self.df = df

        

    def Eda(self):

        print(df.dtypes)

        fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))

        index = 0

        axs = axs.flatten()

        for k,v in df.items():

            sns.distplot(v, ax=axs[index])

            index += 1

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)

        print('        # The histogram also shows that columns CRIM, ZN has highly skewed distributions.'

              '  #Also MEDV looks to have a normal distribution (the predictions) and other colums '

                 '#seem to have norma or bimodel ditribution of data except CHAS and CAT.MEDV '

              '  #(which are discrete variables)')

        

        fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))

        index = 0

        axs = axs.flatten()

        for k,v in df.items():

            sns.boxplot(y=k, data=df, ax=axs[index])

            index += 1

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)

        for k, v in df.items():

            q1 = v.quantile(0.25)

            q3 = v.quantile(0.75)

            irq = q3 - q1

            v_col = v[(v <= q1 - 1.5 * irq) | (v >= q3 + 1.5 * irq)]

            perc = np.shape(v_col)[0] * 100.0 / np.shape(df)[0]

            print("Column %s outliers = %.2f%%" % (k, perc))

            from sklearn import preprocessing

        

        #Plot top 2 correlated features against MEDV



        from sklearn import preprocessing

        # Let's scale the columns before plotting them against MEDV

        min_max_scaler = preprocessing.MinMaxScaler()

        column_sels = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',

               'PTRATIO', 'LSTAT', 'CAT. MEDV']

        x = df.loc[:,column_sels]

        y = df['MEDV']

        x = pd.DataFrame(data=min_max_scaler.fit_transform(x), columns=column_sels)

        fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))

        index = 0

        axs = axs.flatten()

        for i, k in enumerate(column_sels):

            sns.regplot(y=y, x=x[k], ax=axs[i])

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)

        

    def Model(df):

        #first ill get rid of the outliers 

        df = df[~(df['MEDV'] >= 50.0)]

        print(np.shape(df))

        #remove the skewness of the data trough log transformation

        y = df['MEDV']

        X = df.drop(columns = ['MEDV'])

       

        y =  np.log1p(y)

        for col in X.columns:

            if np.abs(X[col].skew()) > 0.3:

                X[col] = np.log1p(X[col])

        fig = plt.figure(figsize=(10, 5))

        ax = sns.distplot(df.MEDV)

        plt.show()

        print('now we see MEDV after removing the skewness of the data with log transformation') 

        

        print(""" Fitting Linear Regression for more simple Regressor 

                        """)

        from sklearn.model_selection import GridSearchCV, train_test_split

        X = df.loc[:, df.columns.difference(['MEDV'])]

        y = df.loc[:, 'MEDV']

        X_train,X_test, y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        from sklearn.linear_model import LinearRegression

        reg = LinearRegression().fit(X_train, y_train)

        y_pred = reg.predict(X_test)

        print('R2 lin_reg score',metrics.r2_score(y_pred,y_test))

        

        

        print("""" Performing a Regression Using XGBoost to predict MEDV

                            """)

        # A parameter grid for XGBoost

        from sklearn.metrics import r2_score

        from sklearn.model_selection import GridSearchCV, train_test_split

        from xgboost import XGBRegressor 



            

        def algorithm_pipeline(X_train_data, X_test_data, y_train_data, y_test_data, 

                   model, param_grid, cv=10, scoring_fit='neg_mean_squared_error',

                   do_probabilities = False):

            gs = GridSearchCV(

                estimator=model,

                param_grid=param_grid, 

                cv=cv, 

                n_jobs=-1, 

                scoring=scoring_fit,

                verbose=2

            )

            fitted_model = gs.fit(X_train_data, y_train_data)



            if do_probabilities:

              pred = fitted_model.predict_proba(X_test_data)

            else:

              pred = fitted_model.predict(X_test_data)



            return fitted_model, pred

        

        

        X = df.loc[:, df.columns.difference(['MEDV'])]

        y = df.loc[:, 'MEDV']

        X_train,X_test, y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        from sklearn.model_selection import RandomizedSearchCV

        model = XGBRegressor()

        param_grid = {

            'n_estimators': [400, 700, 1000],

            'colsample_bytree': [0.7, 0.8],

            'max_depth': [15,20,25],

            'reg_alpha': [1.1, 1.2, 1.3],

            'reg_lambda': [1.1, 1.2, 1.3],

            'subsample': [0.7, 0.8, 0.9]

        }



        model, pred = algorithm_pipeline(X_train, X_test, y_train, y_test, model, 

                                         param_grid, cv=5)



        # Root Mean Squared Error

        print('R2 score :', metrics.r2_score(pred,y_test))

        

        print("""" We can clearly see that the XGboost model is much better than the lin_reg model by the R^2 Score 

        """)

        

        

    
if __name__ == '__main__':

    df = pd.read_csv("../input/train-boston/DS.csv")

    Questions1234(df)

    TestML.Eda(df)

    TestML.Model(df)

    