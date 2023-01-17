#importing libs

import numpy as np

import pandas as pd

pd.set_option('MAX_COLUMNS', None)



#Data viz

import matplotlib.pyplot as plt

import seaborn as sns



#Modelling and Testing

import statsmodels.api as sm

from scipy.stats import shapiro

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.preprocessing import OneHotEncoder, RobustScaler

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, SGDRegressor



#Evaluating the models

from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error





#Paths

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/car-price-prediction/CarPrice_Assignment.csv')

data.set_index('car_ID', inplace=True)

data.drop('CarName', inplace=True, axis=1)

data.head()
#Describe the data

data.describe()
#My own describe

pd.DataFrame({'missing':data.isna().mean(),

             'unicos':data.nunique(),

             'tipos': data.dtypes})
# Changing the type of variable 'symboling' because it was guessed as int and its in fact a ordinal categorical variable.

data['symboling'] = data['symboling'].astype('object') 
#Spliting the data

target = data[['price']]

categorical_vars = data.select_dtypes('object')

numerical_vars = data.select_dtypes(['int64','float64'])

numerical_vars.drop('price', inplace=True, axis=1)
#Target Analysis

#Skew and Kurt

skew = target.skew()

kurt = target.kurt()

print(f'Skewness: {skew}')

print(f'Kurtosis: {kurt}')



#Distribucion

f,ax = plt.subplots(figsize=(12,6))

sns.distplot(target)



#qqplot

norm = sm.qqplot(target, line='s')



#Normality Shapiro test

stats, p = shapiro(target)

print(f'Statistics p-value: {p}')

# interpret

alpha = 0.05

if p > alpha:

    print('Target looks Gaussian (fail to reject H0)')

else:

    print('Target does not look Gaussian (reject H0)')
#transforming the target



target_norm = np.log(data['price'])



#Skew and Kurt

skew = target_norm.skew()

kurt = target_norm.kurt()

print(f'Skewness: {skew}')

print(f'Kurtosis: {kurt}')



#Distribucion

f,ax = plt.subplots(figsize=(12,6))

sns.distplot(target_norm)



#qqplot

norm = sm.qqplot(target_norm, line='s')



#Normality Shapiro test

stats, p = shapiro(target_norm)

print(f'Statistics p-value: {p}')

# interpret

alpha = 0.05

if p > alpha:

    print('Target looks Gaussian (fail to reject H0)')

else:

    print('Target does not look Gaussian (reject H0)')
# Numerical variables univariate analysis

def univariate_analysis(df):

    """

    Function to perform univariate analysis.

    

    df: DataFrame

    """

    for col in df.columns.to_list():

        plt.figure(figsize=(8,5))

        plt.title(f'{col}\n distribuition',fontsize=16)

        sns.distplot(df[col])

        plt.xlabel(col,fontsize=14)

        plt.show()

        

        #Normality Shapiro test

        stats, p = shapiro(df[col])

        print(f'Statistics p-value: {p}')

        # interpret

        alpha = 0.05

        if p > alpha:

            print('Target looks Gaussian (fail to reject H0)')

        else:

            print('Target does not look Gaussian (reject H0)')



univariate_analysis(numerical_vars)
#Checking for outliers

def outliers_analysis(df):

    """

    Function to check for outliers visually through a boxplot

    

    df: DataFrame

    """

    for col in df.columns.to_list():

        plt.figure(figsize=(8,5))

        plt.title(f'{col}\n',fontsize=16)

        sns.boxplot(x=col, data=df)

        plt.xlabel(col,fontsize=14)

        plt.ylabel('Target',fontsize=14)

        plt.show()



outliers_analysis(numerical_vars)
# Variables vs Target

def variables_vs_target(df,target):

    """

    Function to compare variabels with the target with a scatterplot.

    

    df: DataFrame

    target: Variabel Target

    """

    for col in df.columns.to_list():

        plt.figure(figsize=(8,5))

        plt.title(f'{col} vs. \nTarget',fontsize=16)

        plt.scatter(x=df[col],y=target,color='blue',edgecolor='k')

        plt.xlabel(col,fontsize=14)

        plt.ylabel('Target',fontsize=14)

        plt.show()



variables_vs_target(numerical_vars,target=target)
categorical_boxplots = categorical_vars.copy()

categorical_boxplots['Target'] = target





def categorical_analysis(df):

    """

    Function to analyze the target variable vs categorical ones.

    

    df: DataFrame

    """

    for col in df.columns.to_list():

        plt.figure(figsize=(8,5))

        plt.title(f'{col}\n distribuition',fontsize=16)

        sns.boxplot(x=col, y='Target', data=df)

        plt.xlabel(col,fontsize=14)

        plt.ylabel('Target',fontsize=14)

        plt.show()



categorical_analysis(categorical_boxplots)
#Getting 'symboling' out of the mix, once its an ordinal categorical

labels = categorical_vars.columns.to_list()[1:]

categorical_vars = categorical_vars[labels]



#Spliting the data

X_treino, X_teste, y_treino, y_teste = train_test_split(data, target_norm, test_size=.10, shuffle=True, random_state=3)





#Categorical Pipeline

categorical_pipeline = Pipeline([('ohe', OneHotEncoder())])



#Numerical Pipeline

numerical_pipeline = Pipeline([('scaler', RobustScaler())])



#Pipeline

pipeline = ColumnTransformer([('cat',categorical_pipeline, categorical_vars.columns.to_list()),

                              ('num',numerical_pipeline, numerical_vars.columns.to_list())])
reg_pipe = Pipeline([('ct',pipeline),

                     ('reg',LinearRegression())])



#Fitting

reg_pipe.fit(X_treino,y_treino)



#Predicting

reg_pred = reg_pipe.predict(X_teste)



#Evaluating

print(f'The R2: {r2_score(np.expm1(y_teste),np.expm1(reg_pred))}')

print(f'The RMSE: {np.sqrt(mean_squared_error(np.expm1(y_teste),np.expm1(reg_pred)))}')

print('\n')

print(f'Intercept: {reg_pipe.named_steps.reg.intercept_}')

print(f'Coefs: {reg_pipe.named_steps.reg.coef_}')
lasso_pipe = Pipeline([('ct',pipeline),

                         ('reg',LassoCV(alphas = (0.001, 0.1,1.0,5.0,10.0,50.0,100), cv=5))])



#Fitting

lasso_pipe.fit(X_treino,y_treino)



#Predicting

lasso_pred = lasso_pipe.predict(X_teste)



#Evaluating

print(f'The R2: {r2_score(np.expm1(y_teste),np.expm1(lasso_pred))}')

print(f'The RMSE: {np.sqrt(mean_squared_error(np.expm1(y_teste),np.expm1(lasso_pred)))}')

print('\n')

print(f'Best alpha: {lasso_pipe.named_steps.reg.alpha_}')

print(f'Non-zero coefs:{len(lasso_pipe.named_steps.reg.coef_!=0)} from {X_treino.shape[0]} variables')

print(f'Intercept: {lasso_pipe.named_steps.reg.intercept_}')

print(f'Coefs: {lasso_pipe.named_steps.reg.coef_}')
ridge_pipe = Pipeline([('ct',pipeline),

                         ('reg',RidgeCV(alphas=(0.1,1.0,5.0,10.0,50.0,100), cv=5))])





#Fitting

ridge_pipe.fit(X_treino,y_treino)



#Predicting

ridge_pred = ridge_pipe.predict(X_teste)



#Evaluating

print(f'The R2: {r2_score(np.expm1(y_teste),np.expm1(ridge_pred))}')

print(f'The RMSE: {np.sqrt(mean_squared_error(np.expm1(y_teste),np.expm1(ridge_pred)))}')

print('\n')

print(f'Best alpha: {ridge_pipe.named_steps.reg.alpha_}')

print(f'Intercept: {ridge_pipe.named_steps.reg.intercept_}')

print(f'Coefs: {ridge_pipe.named_steps.reg.coef_}')
#Getting 'symboling' out of the mix, once its an ordinal categorical

labels = categorical_vars.columns.to_list()[1:]

categorical_vars_stats = categorical_vars[labels]



#One-Hot-Encoding

categorical_vars_ohe = pd.get_dummies(categorical_vars_stats)



#Numerical transformation

skewed_vars= numerical_vars.apply(lambda x: x.skew())

skewed_labels = skewed_vars[skewed_vars > 0.75].index

numerical_vars[skewed_labels] = np.log1p(numerical_vars[skewed_labels])



#Putting together

dataset = categorical_vars_ohe.merge(numerical_vars, left_index=True, right_index=True)



#Spliting the data

X_treino, X_teste, y_treino, y_teste = train_test_split(dataset, target_norm, test_size=.10, shuffle=True, random_state=4)
#OLS

model = sm.OLS(y_treino,X_treino)

results = model.fit()

results.summary()
#Independece assumption



variables = X_treino[['wheelbase', 'carlength', 'carwidth', 'carheight',

       'curbweight', 'enginesize', 'boreratio', 'stroke', 'compressionratio',

       'horsepower', 'peakrpm', 'citympg', 'highwaympg']].columns.to_list()



for col in variables:

    plt.figure(figsize=(8,5))

    plt.title('Independece Assumption',fontsize=16)

    plt.scatter(x=X_treino[col],y=results.resid,color='blue',edgecolor='k')

    plt.hlines(y=0,xmin = min(X_treino[col]) , xmax = max(X_treino[col]),color='red',linestyle='--',lw=3)

    plt.xlabel(col,fontsize=14)

    plt.ylabel('Residuals',fontsize=14)

    plt.show()
# Homoscedasticity plot

plt.figure(figsize=(8,5))

plt.title('Fitted vs Residuals',fontsize=16)

plt.scatter(x=results.fittedvalues,y=results.resid,color='blue',edgecolor='k')

plt.hlines(y=0,xmin = 8.5 , xmax = max(results.fittedvalues),color='red',linestyle='--',lw=3)

plt.xlabel('Fitted Values',fontsize=14)

plt.ylabel('Residuals',fontsize=14)

plt.show()
#Distribucion

f,ax = plt.subplots(figsize=(12,6))

sns.distplot(results.resid)



#qqplot

norm = sm.qqplot(results.resid, line='s')
#Normality Shapiro test

stats, p = shapiro(results.resid)

print(f'Statistics p-value: {p}')

# interpret

alpha = 0.05

if p > alpha:

    print('Resids looks Gaussian (fail to reject H0)')

else:

    print('Resids does not look Gaussian (reject H0)')