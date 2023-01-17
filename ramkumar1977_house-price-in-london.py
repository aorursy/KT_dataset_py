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
import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import scipy as sc

import pandas_profiling   #need to install using anaconda prompt (pip install pandas_profiling)

%pylab inline
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

import sklearn.metrics as metrics

import statsmodels.formula.api as sm
import numpy as np

import pandas as pd

import scipy.stats as stats

import matplotlib.pyplot as plt

%matplotlib inline

import math
import matplotlib.pyplot as plt

from sklearn import datasets

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression
from collections import defaultdict

import time

import gc

import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.datasets.samples_generator import make_regression

from sklearn.ensemble.forest import RandomForestRegressor

from sklearn.linear_model.ridge import Ridge

from sklearn.linear_model.stochastic_gradient import SGDRegressor

from sklearn.svm.classes import SVR

from sklearn.utils import shuffle
monthly_variables = pd.read_csv("/kaggle/input/housing-in-london/housing_in_london_monthly_variables.csv")

monthly_variables
monthly_variables.head()
monthly_variables.tail()
yearly_variable = pd.read_csv("/kaggle/input/housing-in-london/housing_in_london_yearly_variables.csv")

yearly_variable
yearly_variable.head()
yearly_variable.tail()
yearly_variable.info()
yearly_variable.dtypes
yearly_variable.describe()
yearly_variable.describe(include = 'all')
monthly_variables.info()
monthly_variables.dtypes
monthly_variables.describe()
monthly_variables.describe(include = 'all')
monthly_variables.drop_duplicates(subset=['date']) 
comb_df = yearly_variable.merge(monthly_variables.drop_duplicates(subset=['date']), how='left')

comb_df
numeric_var_names=[key for key in dict(comb_df.dtypes) if dict(comb_df.dtypes)[key] in ['float64', 'int64', 'float32', 'int32']]

cat_var_names=[key for key in dict(comb_df.dtypes) if dict(comb_df.dtypes)[key] in ['object']]

print(numeric_var_names)

print(cat_var_names)
comb_df_num = comb_df[numeric_var_names]
comb_df_cat = comb_df[cat_var_names]
# Use a general function that returns multiple values

def var_summary(x):

    return pd.Series([x.count(), x.isnull().sum(), x.sum(), x.mean(), x.median(),  x.std(), x.var(), x.min(), x.dropna().quantile(0.01), x.dropna().quantile(0.05),x.dropna().quantile(0.10),x.dropna().quantile(0.25),x.dropna().quantile(0.50),x.dropna().quantile(0.75), x.dropna().quantile(0.90),x.dropna().quantile(0.95), x.dropna().quantile(0.99),x.max()], 

                  index=['N', 'NMISS', 'SUM', 'MEAN','MEDIAN', 'STD', 'VAR', 'MIN', 'P1' , 'P5' ,'P10' ,'P25' ,'P50' ,'P75' ,'P90' ,'P95' ,'P99' ,'MAX'])



num_summary = comb_df_num.apply(lambda x: var_summary(x)).T
num_summary
import numpy as np

for col in comb_df_num.columns:

    percentiles = comb_df_num[col].quantile([0.01,0.99]).values

    comb_df_num[col] = np.clip(comb_df_num[col], percentiles[0], percentiles[1])
#Handling missings - Method2

def Missing_imputation(x):

    x = x.fillna(x.median())

    return x



comb_df_num = comb_df_num.apply(lambda x: Missing_imputation(x))
#Handling missings - Method2

def Cat_Missing_imputation(x):

    x = x.fillna(x.mode())

    return x



comb_df_cat = comb_df_cat.apply(lambda x: Cat_Missing_imputation(x))
def create_dummies(df, colname):

    col_dummies = pd.get_dummies(df[colname], prefix=colname, drop_first=True)

    df = pd.concat([df, col_dummies],axis=1)

    df.drop(colname, axis = 1, inplace = True)

    return df



for c_feature in comb_df_cat.columns:

    comb_df_cat[c_feature] = comb_df_cat[c_feature].astype('category')

    comb_df_cat = create_dummies(comb_df_cat , c_feature )
comb_df_cat.head().T
comb_df_num.head().T
comb_df_new = pd.concat([comb_df_num, comb_df_cat], axis=1)

comb_df_new.head()
comb_df_new["ln_average_price"]= np.log(comb_df_new["average_price"])

comb_df_new.head()
np.log(comb_df_new.ln_average_price).hist()
features = comb_df_new[comb_df_new.columns.difference( ['ln_average_price'] )]

target = comb_df_new['ln_average_price']
features.columns
features.shape
target
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import itertools
features.shape
lm = LinearRegression()
# create the RFE model and select 10 attributes

rfe = RFE(lm, n_features_to_select=60)

rfe = rfe.fit(features,target)
rfe.get_support()
from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression

import itertools

lm = LinearRegression()

# create the RFE model and select 10 attributes

rfe = RFE(lm, n_features_to_select=60)

rfe = rfe.fit(features,target)
rfe.get_support()
# summarize the selection of the attributes

feature_map = [(i, v) for i, v in itertools.zip_longest(features.columns, rfe.get_support())]
feature_map
#Alternative of capturing the important variables

RFE_features=features.columns[rfe.get_support()]
RFE_features
features1 = features[RFE_features]
features1.head()
# Feature Selection based on importance

from sklearn.feature_selection import f_regression

F_values, p_values  = f_regression(features1, target)
import itertools

f_reg_results = [(i, v, z) for i, v, z in itertools.zip_longest(features1.columns, F_values,  ['%.3f' % p for p in p_values])]
f_reg_results=pd.DataFrame(f_reg_results, columns=['Variable','F_Value','P_Value'])
f_reg_results.sort_values(by=['F_Value'],ascending = False)
f_reg_results.P_Value = pd.to_numeric(f_reg_results.P_Value)
f_reg_results_new=f_reg_results[f_reg_results.P_Value<=0.2]
f_reg_results_new
f_reg_results_new.info()
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2,f_classif,mutual_info_classif
features = comb_df_new[comb_df_new.columns.difference(['ln_average_price'])]

target = comb_df_new['ln_average_price']

features_new = SelectKBest(f_classif, k=30).fit(features, target )
features_new.get_support()
features_new.scores_
# summarize the selection of the attributes

import itertools

feature_map = [(i, v) for i, v in itertools.zip_longest(features.columns, features_new.get_support())]

feature_map

#Alternative of capturing the important variables

KBest_features=features.columns[features_new.get_support()]

selected_features_from_KBest = features[KBest_features]
KBest_features
list_vars1 = list(f_reg_results_new.Variable)

list_vars1
all_columns = "+".join(list_vars1)

my_formula = "ln_average_price~" + all_columns

print(my_formula)
from statsmodels.formula.api import ols

from statsmodels.stats.outliers_influence import variance_inflation_factor

import statsmodels.api as sm

import statsmodels.formula.api as sm
model = sm.ols('ln_average_price~mean_salary_48922+mean_salary_52203+mean_salary_56450+mean_salary_62819+mean_salary_64272+mean_salary_66628+mean_salary_74004+mean_salary_77754+mean_salary_79489+mean_salary_80655+mean_salary_80769+mean_salary_82808+mean_salary_82973+mean_salary_83403+mean_salary_85886+mean_salary_86987+mean_salary_88342+mean_salary_90028+mean_salary_90842+recycling_pct_34',data = comb_df_new)
model = model.fit()
model.summary()
print(model.summary())
my_formula ='ln_average_price~mean_salary_48922+mean_salary_52203+mean_salary_56450+mean_salary_62819+mean_salary_64272+mean_salary_66628+mean_salary_74004+mean_salary_77754+mean_salary_79489+mean_salary_80655+mean_salary_80769+mean_salary_82808+mean_salary_82973+mean_salary_83403+mean_salary_85886+mean_salary_86987+mean_salary_88342+mean_salary_90028+mean_salary_90842+recycling_pct_34'

my_formula
from statsmodels.stats.outliers_influence import variance_inflation_factor

from patsy import dmatrices
# get y and X dataframes based on this regression

y, X = dmatrices(my_formula,comb_df_new,return_type='dataframe')
# For each X, calculate VIF and save in dataframe

vif = pd.DataFrame()

vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif["features"] = X.columns

vif.round(1)
train, test = train_test_split(comb_df_new, test_size = 0.3, random_state = 123456)
print(len(train))

print(len(test))
import statsmodels.formula.api as smf
my_formula ='ln_average_price~mean_salary_48922+mean_salary_52203+mean_salary_56450+mean_salary_62819+mean_salary_64272+mean_salary_66628+mean_salary_74004+mean_salary_77754+mean_salary_79489+mean_salary_80655+mean_salary_80769+mean_salary_82808+mean_salary_82973+mean_salary_83403+mean_salary_85886+mean_salary_86987+mean_salary_88342+mean_salary_90028+mean_salary_90842'

my_formula
model = smf.ols(my_formula, data=train).fit()

print(model.summary())
model.summary()
train['pred'] = pd.DataFrame(model.predict(train))
train.head()
test['pred'] = pd.DataFrame(model.predict(test))

test.head()
# calculate these metrics by hand!

from sklearn import metrics

import numpy as np

import scipy.stats as stats
#Train Data

MAPE_train = np.mean(np.abs(train.ln_average_price - train.pred)/train.ln_average_price)

print(MAPE_train)



RMSE_train = metrics.mean_squared_error(train.ln_average_price,train.pred)

print(RMSE_train)



Corr_train = stats.stats.pearsonr(train.ln_average_price, train.pred)

print(Corr_train)



#Test Data

MAPE_test = np.mean(np.abs(test.ln_average_price - test.pred)/test.ln_average_price)

print(MAPE_test)



RMSE_test = metrics.mean_squared_error(test.ln_average_price, test.pred)

print(RMSE_test)



Corr_test = stats.stats.pearsonr(test.ln_average_price, test.pred)

print(Corr_test)