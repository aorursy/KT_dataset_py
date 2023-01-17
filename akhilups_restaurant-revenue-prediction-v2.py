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
#import the data

data = pd.read_csv('/kaggle/input/cmpe343/train.csv',index_col='Id')
data
data.dtypes
#profile_report = pandas_profiling.ProfileReport(data)

#profile_report.to_file('profile_report.html')
data.drop(['Open Date','City'], axis=1, inplace= True)
data.info()
#Alternative method to split the numerical & categorical dynamically

numeric_var_names=[key for key in dict(data.dtypes) if dict(data.dtypes)[key] in ['float64', 'int64', 'float32', 'int32']]

cat_var_names=[key for key in dict(data.dtypes) if dict(data.dtypes)[key] in ['object']]

print(numeric_var_names)

print(cat_var_names)
def outlier_miss_treat(x):

    x = x.clip(upper=x.quantile(0.99))

    x = x.clip(lower=x.quantile(0.01))

    x = x.fillna(x.median())

    return x
data_cat = data[cat_var_names]

data_num = data[numeric_var_names]



data_num_new = data_num.apply(outlier_miss_treat)
data_num.head()
def miss_treat_cat(x):

    x = x.fillna(x.mode())

    return x
data_cat_new = data_cat.apply(miss_treat_cat)
cat_dummies = pd.get_dummies(data_cat_new, drop_first=True)
data_new = pd.concat([data_num_new, cat_dummies], axis=1)
data_new.head()
data_new.revenue.hist(bins=20)
np.log(data_new.revenue).hist(bins=20)
data_new['ln_revenue'] =  np.log(data_new.revenue)
data_new.corrwith(data_new.ln_revenue)


from sklearn.feature_selection import RFE, SelectKBest, f_regression, f_oneway
data_new.columns.difference(['revenue', 'ln_revenue'])
features = data_new.columns.difference(['revenue', 'ln_revenue'])
features
X = data_new[features]

Y= data_new['ln_revenue']
F_values, p_values  = f_regression(  X, Y )
f_reg_results = pd.concat([pd.Series(X.columns), pd.Series(F_values), pd.Series(p_values)], axis=1)
f_reg_results.columns = ['Feature', 'F_Value', 'P_Value']
f_reg_results.sort_values('F_Value', ascending=False, inplace=True)
f_reg_results
f_reg_results.to_csv('F_Reg_Results.csv')
from sklearn.linear_model import LinearRegression
RFE_model = RFE(LinearRegression(), n_features_to_select=10)
RFE_model= RFE_model.fit(X, Y)
X.columns
RFE_model.get_support()
RFE_selected_Cols = X.columns[list(RFE_model.get_support())]
RFE_model.ranking_
pd.Series(RFE_selected_Cols)
selectkbest = SelectKBest(f_oneway, k=10)
selectkbest = selectkbest.fit(X,Y)
selectkbest.get_support()
selectKbest_Cols = X.columns[list(selectkbest.get_support())]
pd.Series(selectKbest_Cols)
from statsmodels.stats.outliers_influence import variance_inflation_factor

from patsy import dmatrices
X_New = X[['P10',

'P11',

'P13',

'P16',

'P17',

'P18',

'P25',

'P28',

'P8',

'Type_IL',

'P22',

'P3',

'P33',

'P4',

'P5',

'Type_FC']]
X_New = X_New[X_New.columns.difference(['P10','P13','P8','P4','P3'])]
VIF = [variance_inflation_factor(X_New.values, i) for i in range(X_New.shape[1])]
VIF_results = pd.concat([pd.Series(X_New.columns), pd.Series(VIF)], axis=1)

VIF_results.columns = ['Feature', 'VIF']

VIF_results.sort_values('VIF', ascending=False, inplace=True)

VIF_results
#Final list of columns to be included in the model

X_New.columns
from sklearn.model_selection import train_test_split
data_new2 = pd.concat([X_New, Y], axis=1)
data_new2.head()
train, test = train_test_split(data_new2, test_size = 0.3, random_state =123)
import statsmodels.formula.api as smf
#formula = 'ln_revenue' + '~' + '+'.join(train.columns.difference(['ln_revenue','P16','P11','P25','Type_FC','Type_IL']))

formula = 'ln_revenue' + '~' + '+'.join(train.columns.difference(['ln_revenue']))
model = smf.ols(formula , data = train)

model = model.fit()
print(model.summary())
test_pred = np.exp(model.predict(test))

train_pred = np.exp(model.predict(train))
train_y = np.exp(train.ln_revenue)

test_y = np.exp(test.ln_revenue)
#Metrics for train data

MAPE_train = np.mean(np.abs(train_y - train_pred)/train_y)

print('Train_MAPE=', MAPE_train)

RMSE_train = np.sqrt(np.mean(np.square(train_y - train_pred)))

print('Train_RMSE=',RMSE_train)

RMSPE_train =np.sqrt(np.mean(np.square((train_y - train_pred)/train_y)))

print('Train_RMSPE=',RMSPE_train)

corr_train = np.corrcoef(train_y, train_pred)[1][0]

print('Train_Corr=',corr_train)
#Metrics for test data

MAPE_test = np.mean(np.abs(test_y - test_pred)/test_y)

print('Test_MAPE=', MAPE_test)

RMSE_test = np.sqrt(np.mean(np.square(test_y - test_pred)))

print('Test_RMSE=',RMSE_test)

RMSPE_test =np.sqrt(np.mean(np.square((test_y - test_pred)/test_y)))

print('Test_RMSPE=',RMSPE_test)

corr_test = np.corrcoef(test_y, test_pred)[1][0]

print('Test_Corr=',corr_test)
train_new = pd.concat([train_y, train_pred], axis=1)

train_new.columns = ['actual', 'pred']

test_new = pd.concat([test_y, test_pred], axis=1)

test_new.columns = ['actual', 'pred']
model.resid.hist(bins=100)
import seaborn as sns

sns.distplot(model.resid)
from scipy import stats

import pylab



stats.probplot( model.resid, dist="norm", plot=pylab )

pylab.show()
data_test = pd.read_csv('/kaggle/input/cmpe343/test.csv', index_col='Id')
data_test
#data_test.info()

indices = data_test.index
data_test.drop(['Open Date','City'], axis=1, inplace= True)



#Alternative method to split the numerical & categorical dynamically

numeric_var_names=[key for key in dict(data_test.dtypes) if dict(data_test.dtypes)[key] in ['float64', 'int64', 'float32', 'int32']]

cat_var_names=[key for key in dict(data_test.dtypes) if dict(data_test.dtypes)[key] in ['object']]

print(numeric_var_names)

print(cat_var_names)



def outlier_miss_treat(x):

    x = x.clip(upper=x.quantile(0.99))

    x = x.clip(lower=x.quantile(0.01))

    x = x.fillna(x.median())

    return x



data_test_cat = data_test[cat_var_names]

data_test_num = data_test[numeric_var_names]



data_test_num_new = data_test_num.apply(outlier_miss_treat)



def miss_treat_cat(x):

    x = x.fillna(x.mode())

    return x



data_test_cat_new = data_test_cat.apply(miss_treat_cat)



cat_dummies = pd.get_dummies(data_test_cat_new, drop_first=True)



data_test_new = pd.concat([data_test_num_new, cat_dummies], axis=1)
preds = model.predict(data_test_new).ravel()

output = pd.DataFrame({"Id": indices,"ln_revenue": preds})

output

output['revenue'] = np.exp(output['ln_revenue'])

output.drop(["ln_revenue"],axis=1, inplace= True)
output.to_csv('submissions.csv', index=False)