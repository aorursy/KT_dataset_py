import numpy as np

import pandas as pd



import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set_style('whitegrid')

matplotlib.rcParams['figure.figsize'] = (12,8)



from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression, Ridge, Lasso

from sklearn import metrics
# Importing data set from sklearn

from sklearn.datasets import load_boston

boston = load_boston()
# Exploring boston data set

print(boston.keys())
# About dataset

print(boston.DESCR)
# Creating a DataFrame of Data

df = pd.DataFrame(boston.data, columns=boston.feature_names)

df.head(5)
# Check target of dataset

print(boston.target)
# Add MEDV(target) to df

target = pd.Series(boston.target, name='Target')

df['MEDV'] = target
df.info()
df.describe().T
# Get numeric columns

print('Numeric cols :',df.select_dtypes(include=[np.number]).columns.values)
# Get Non-numeric columns

print('Categorical cols :',df.select_dtypes(exclude=[np.number]).columns.values)
# Number of missing values

print(df.isnull().sum().sort_values(ascending=False))
# Univariate Analysis

fig, axes = plt.subplots(4,3, figsize=(12,15), gridspec_kw={'hspace':0.5,})

fig.suptitle('Distributions of Boston Features')

for ax, cols in zip(axes.flatten(), df.columns[:-1]):

  sns.distplot(df[cols], ax=ax, bins=30, kde_kws={'bw':1.5})

  ax.set(title=cols.upper(), xlabel='')
df.columns[:-1]
# Correlation plot

sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

plt.tight_layout()
# Correlation with Target

df.corr()['MEDV'].sort_values(ascending=False)[1:]
# As rooms increases price increases

# Notice there are some outliers, ex: when MEDV = 50

sns.scatterplot('RM', 'MEDV', data=df)
# As % lower status of the population increases Target decreases

# Notice there are some outliers

sns.scatterplot( 'LSTAT','MEDV', data=df)
#  Distribution of MEDV

df['MEDV'].plot.hist()
X = df.drop('MEDV', axis=1)

y = df['MEDV']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
# Removing outliers using IsolationForest

from sklearn.ensemble import IsolationForest

iso = IsolationForest(contamination=0.2)

yhat = iso.fit_predict(X_train)
# select all rows that are not outliers

mask = yhat != -1

X_train_iso, y_train_iso = X_train.loc[mask, :], y_train.loc[mask]

print('Before: ',X_train.shape, y_train.shape)

print('Updated: ',X_train_iso.shape, y_train_iso.shape)
fig, (ax1,ax2) = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(15,8))

sns.scatterplot(X_train['RM'],y_train, ax=ax1)

sns.scatterplot(X_train_iso['RM'],y_train_iso, ax=ax2)

ax1.set(title='Before')

ax2.set(title='Updated')
fig, (ax1,ax2) = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(15,8))

sns.scatterplot(X_train['LSTAT'],y_train, ax=ax1)

sns.scatterplot(X_train_iso['LSTAT'],y_train_iso, ax=ax2)

ax1.set(title='Before')

ax2.set(title='Updated')
# Function to compare models and parameters

def find_best_model(X,y):

  models = {

      'Linear Regression':{

          'model':LinearRegression(),

          'parameters':{

              'normalize':[True,False]

              }

          },

        'Ridge':{

            'model':Ridge(),  # L2 Reguralization

            'parameters':{

                'normalize':[True,False],

                'alpha':[0.1,0.5,1,10,100]

                }

          },

        'Lasso':{

            'model':Lasso(),  # L1 Regularization

            'parameters':{

                'normalize':[True,False],

                'alpha':[0.1,0.5,1,10,100]

                }

        }

  }



  scores = []

  for model_name, model_params in models.items():

    gs = GridSearchCV(estimator = model_params['model'],

                      param_grid = model_params['parameters'],

                      scoring='r2',

                      cv=10,

                      n_jobs = -1)

    gs.fit(X,y)   # Search for best model and parameters

    scores.append({

        'Model' : model_name,

        'Best_params':gs.best_params_,

        'R^2':gs.best_score_,

         })

  result = pd.DataFrame(scores)  # Result obatined after searching 

  return result.sort_values('R^2', ascending=False) # Higher values the better
# Performance on training set after parameter hypertuning without removing outliers

find_best_model(X_train,y_train)
# Performance on training set after parameter hypertuning by removing outliers

find_best_model(X_train_iso,y_train_iso)
# Ridge and Linear Regression r2 scores are same

# R² value of 0.75 means that the model is accounting for 75% of the variability in the data



rid = Ridge(alpha=0.1, normalize=True) # selecting ridge and best parameters

rid.fit(X_train_iso, y_train_iso)
# Coefficients

coeff_df = pd.DataFrame(rid.coef_, X.columns, columns=['Coefficient'])    

coeff_df
# Predictions

y_pred = rid.predict(X_test)

print(y_pred)
# DataFrame temp with Actual and Predicted values

temp = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred.flatten()})

temp
# MSE and RMSE

print('MSE :',metrics.mean_absolute_error(y_test,y_pred))

print('RMSE :',np.sqrt(metrics.mean_absolute_error(y_test,y_pred)))
# Plotting "Actual v/s Predicted"

sns.regplot(y_test, y_pred, scatter=True)

plt.title("Actual v/s Predicted")

plt.xlabel("Actual")

plt.ylabel("Predicted")