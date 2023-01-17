# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import scipy.stats as stats

from scipy.stats.mstats import winsorize
data = pd.read_csv(r"/kaggle/input/regression-analysis-on-life-data/life_data_WHO.csv")
data.head()
data.tail()
data.columns
data.dtypes
orig_cols = list(data.columns)

new_cols = []

for col in orig_cols:

    new_cols.append(col.strip().replace('  ', ' ').replace(' ', '_').lower())

data.columns = new_cols
data.columns
data.rename(columns={'thinness_1-19_years':'thinness_10-19_years'}, inplace=True)
data.describe().iloc[:,1:]
plt.figure(figsize=(15,10))

for i, col in enumerate(['adult_mortality', 'infant_deaths', 'bmi', 'under-five_deaths', 'gdp', 'population'], start=1):

    plt.subplot(2,3, i)

    data.boxplot(col)
mort_5_percentile = np.percentile(data.adult_mortality.dropna(),5)     

data.adult_mortality = data.apply(lambda x :np.nan if x.adult_mortality < mort_5_percentile else x.adult_mortality,axis=1)    #1 

data.infant_deaths =data.infant_deaths.replace(0,np.nan)    # 2

data.bmi =data.apply(lambda x :np.nan if (x.bmi < 10 or x.bmi > 50) else x.bmi,axis = 1)  #3

data['under-five_deaths'] =data['under-five_deaths'].replace(0,np.nan)   #4
def nulls_breakdown(data=data):

    df_cols =list(data.columns)

    cols_total_count =len(list(data.columns))

    cols_count = 0

    for loc,col in enumerate(df_cols):

        null_count = data[col].isnull().sum()

        total_count =data[col].isnull().count()

        percent_null =round(null_count/total_count*100/2)

        if null_count > 0:

            cols_count+= 1

            print('[iloc = {}] {} has {} null values: {}% null'.format(loc, col, null_count, percent_null))

    cols_percent_null = round(cols_count/cols_total_count*100, 2)

    print('Out of {} total columns, {} contain null values; {}% columns contain null values.'.format(cols_total_count, cols_count, cols_percent_null))

nulls_breakdown()
data.drop(columns='bmi', inplace=True)
imputed_data = []

for year in list(data.year.unique()):

    year_data = data[data.year == year].copy()

    for col in list(year_data.columns)[3:]:

        year_data[col] = year_data[col].fillna(year_data[col].dropna().mean()).copy()

    imputed_data.append(year_data)

data = pd.concat(imputed_data).copy()
nulls_breakdown(data)
#  Outliers Detection

cont_vars =list(data.columns)[3:]

def outlier_visual(data):

    plt.figure(figsize=(15,40))

    i = 0

    for col in cont_vars:

        i+= 1

        plt.subplot(9,4,i)

        plt.boxplot(data[col])

        plt.title('{} boxplot'.format(col))

        i += 1

        plt.subplot(9,4,i)

        plt.hist(data[col])

        plt.title('{} histogram'.format(col))

    plt.show()

outlier_visual(data)
def outlier_count(col,data =data):

    print(15*'-'+ col +15*'-')

    q75,q25 =np.percentile(data[col],[75,25])

    iqr = q75 - q25

    min_val = q25 -(iqr*1.5)

    max_val = q75 +(iqr*1.5)

    outlier_count = len(np.where((data[col] >max_val) | (data[col] < min_val))[0])

    outlier_percent = round(outlier_count/len(data[col])*100,2)

    print('Number of outliers: {}'.format(outlier_count))

    print('Percent of data that is outlier: {}%'.format(outlier_percent))
for col in cont_vars:

    outlier_count(col)
def test_wins(col, lower_limit=0, upper_limit=0, show_plot=True):

    wins_data = winsorize(data[col], limits=(lower_limit, upper_limit))

    wins_dict[col] = wins_data

    if show_plot == True:

        plt.figure(figsize=(15,5))

        plt.subplot(121)

        plt.boxplot(data[col])

        plt.title('original {}'.format(col))

        plt.subplot(122)

        plt.boxplot(wins_data)

        plt.title('wins=({},{}) {}'.format(lower_limit, upper_limit, col))

        plt.show()
wins_dict = {}

test_wins(cont_vars[0], lower_limit=.01, show_plot=True)

test_wins(cont_vars[1], upper_limit=.04, show_plot=False)

test_wins(cont_vars[2], upper_limit=.05, show_plot=False)

test_wins(cont_vars[3], upper_limit=.0025, show_plot=False)

test_wins(cont_vars[4], upper_limit=.135, show_plot=False)

test_wins(cont_vars[5], lower_limit=.1, show_plot=False)

test_wins(cont_vars[6], upper_limit=.19, show_plot=False)

test_wins(cont_vars[7], upper_limit=.05, show_plot=False)

test_wins(cont_vars[8], lower_limit=.1, show_plot=False)

test_wins(cont_vars[9], upper_limit=.02, show_plot=False)

test_wins(cont_vars[10], lower_limit=.105, show_plot=False)

test_wins(cont_vars[11], upper_limit=.185, show_plot=False)

test_wins(cont_vars[12], upper_limit=.105, show_plot=False)

test_wins(cont_vars[13], upper_limit=.07, show_plot=False)

test_wins(cont_vars[14], upper_limit=.035, show_plot=False)

test_wins(cont_vars[15], upper_limit=.035, show_plot=False)

test_wins(cont_vars[16], lower_limit=.05, show_plot=False)

test_wins(cont_vars[17], lower_limit=.025, upper_limit=.005, show_plot=False)
plt.figure(figsize=(15,5))

for i, col in enumerate(cont_vars, 1):

    plt.subplot(2, 9, i)

    plt.boxplot(wins_dict[col])

plt.tight_layout()

plt.show()
wins_df = data.iloc[:, 0:3]

for col in cont_vars:

    wins_df[col] = wins_dict[col]
wins_df.describe()
wins_df.describe(include = 'O')
plt.figure(figsize =(15,20))

for i, col in enumerate(cont_vars,1):

    plt.subplot(5,4,i)

    plt.hist(wins_df[col])

    plt.title(col)
plt.figure(figsize=(15, 25))

wins_df.country.value_counts(ascending=True).plot(kind='barh')

plt.title('Count of Rows by Country')

plt.xlabel('Count of Rows')

plt.ylabel('Country')

plt.tight_layout()

plt.show()
wins_df.year.value_counts().sort_index().plot(kind ='barh')

plt.title('Count of Rows by Year')

plt.xlabel('Count of Rows')

plt.ylabel('Year')

plt.show()
plt.figure(figsize=(10,5))

plt.subplot(121)

wins_df.status.value_counts().plot(kind ='bar')

plt.title('Count of Rows by Country Status')

plt.xlabel('Country Status')

plt.ylabel('Count of Rows')

plt.xticks(rotation=0)





plt.subplot(122)

wins_df.status.value_counts().plot(kind ='pie',autopct = '%.2f')

plt.ylabel('')

plt.title('Country Status Pie Chart')



plt.show()
wins_df[cont_vars].corr()
mask = np.triu(wins_df[cont_vars].corr())

plt.figure(figsize=(15,6))

sns.heatmap(wins_df[cont_vars].corr(), annot=True, fmt='.2g', vmin=-1, vmax=1, center=0, cmap='coolwarm', mask=mask)

plt.ylim(18, 0)

plt.title('Correlation Matrix Heatmap')

plt.show()
life = data[['country','year','status','adult_mortality','infant_deaths','alcohol','percentage_expenditure','hepatitis_b','measles','under-five_deaths', 'polio', 'total_expenditure','diphtheria', 'hiv/aids', 'gdp', 'population', 'thinness_10-19_years','thinness_5-9_years', 'income_composition_of_resources', 'schooling','life_expectancy']] 
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

life['country'] = labelencoder.fit_transform(life['country'])

life['status'] = labelencoder.fit_transform(life['status'])
life.head()
import statsmodels.api as sm

from statsmodels.stats import diagnostic as diag

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
X = life.drop('life_expectancy', axis = 1)

Y = life[['life_expectancy']]

# Split X and y into X_

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=1)
life_exp =pd.concat([y_train,X_train],axis = 1)
# calculate the correlation matrix

corr = life_exp.corr()

display(corr)

# plot the correlation heatmap

sns.heatmap(corr,annot =True)
X_1 = sm.add_constant(X_train)

# create a OLS model

model = sm.OLS(y_train, X_1).fit()

print(model.summary())
X1=life_exp.drop(['life_expectancy'],axis=1)

# the VFI does expect a constant term in the data, 

#so we need to add one using the add_constant method

#X1 = sm.add_constant(econ_df_before)

series_before = pd.Series([variance_inflation_factor(X1.values, i) 

                           for i in range(X1.shape[1])], index=X1.columns)

series_before
# define our intput

X2 = sm.add_constant(X_train)

# create a OLS model

model2 = sm.OLS(y_train, X2).fit()

print(model2.summary())
Data=pd.concat([X_train,y_train],axis=1)
Data['Fitted_value']=model2.fittedvalues

Data['Residual']=model2.resid
p = Data.plot.scatter(x='Fitted_value',y='Residual')

plt.xlabel('Fitted values')

plt.ylabel('Residuals')

p = plt.title('Residuals vs fitted values plot for homoscedasticity check')

plt.show()
import pylab

# check for the normality of the residuals

sm.qqplot(model2.resid, line='s')

pylab.show()
Data['Residual'].plot.hist()
X_test2 = X_test[['country','year','status','adult_mortality','infant_deaths','alcohol','percentage_expenditure','hepatitis_b','measles','under-five_deaths', 'polio', 'total_expenditure','diphtheria', 'hiv/aids', 'gdp', 'population', 'thinness_10-19_years','thinness_5-9_years', 'income_composition_of_resources', 'schooling']]
X_test2 = sm.add_constant(X_test2)
X_test2.head()
y_predict=model2.predict(X_test2)
test=pd.concat([X_test,y_test],axis=1)
test['Predicted']=y_predict
test.head()
import math

# calculate the mean squared error

model_mse = mean_squared_error(test['life_expectancy'], test['Predicted'])

# calculate the mean absolute error

model_mae = mean_absolute_error(test['life_expectancy'], test['Predicted'])

# calulcate the root mean squared error

model_rmse = math.sqrt(model_mse)

# display the output

print("MSE {:.3}".format(model_mse))

print("MAE {:.3}".format(model_mae))

print("RMSE {:.3}".format(model_rmse))
import math

# calculate the mean squared error

model_mse = mean_squared_error(Data['life_expectancy'], Data['Fitted_value'])

# calculate the mean absolute error

model_mae = mean_absolute_error(Data['life_expectancy'], Data['Fitted_value'])

# calulcate the root mean squared error

model_rmse = math.sqrt(model_mse)

# display the output

print("MSE {:.3}".format(model_mse))

print("MAE {:.3}".format(model_mae))

print("RMSE {:.3}".format(model_rmse))