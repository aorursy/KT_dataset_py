import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler

from sklearn import preprocessing



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



print("setup complete")
data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
X_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
pd.set_option('display.max_columns', None)

#pd.set_option('display.max_rows', None)
data.head(5)
corr = data.corr()

corr = corr.loc[:,['SalePrice']].sort_values(by = ['SalePrice'],ascending = False)

corr = corr[1:]

remove_less_corr_columns = corr.loc[corr['SalePrice'] < 0.30]

remove_less_corr_columns

data.drop(remove_less_corr_columns.index,axis = 1,inplace = True)

X_test.drop(remove_less_corr_columns.index,axis = 1,inplace = True)
plt.figure(figsize = (15,15))

g= sns.barplot(x = corr.index,y = corr['SalePrice'])

g.set_xticklabels(g.get_xticklabels(),rotation = 90)

g.set_ylabel('Correlation with SalePrice')

g.set_xlabel('Column Names')

g.set_ylim(-0.30, 1)
null_values_col = data.isnull().sum()

null_values_col = null_values_col.loc[null_values_col > 0]

dtyes = [data[d].dtype for d in null_values_col.index]

null_values_col = null_values_col.reset_index()

null_values_col['dtypes'] = dtyes

null_values_col.columns = ['Column Name','Number of Missing Values','Data Type']

total_missing = null_values_col['Number of Missing Values'].sum()

null_values_col['% Missing'] = null_values_col['Number of Missing Values'].apply(lambda x: ((x/total_missing)*100)).round(2)

null_values_col.set_index(['Column Name'],inplace = True)
null_values_col
cat_columns = [col for col in data.columns if data[col].dtype=='object']

cat_columns = pd.DataFrame(cat_columns)

cat_columns.rename(columns = {0:'Column Name'},inplace = True)

cat_columns.set_index(['Column Name'],inplace = True)

cat_col_unique_values = [data[col].nunique() for col in cat_columns.index] 

cat_columns['Number of unique values'] = cat_col_unique_values

cat_columns_values = {}

for col in cat_columns.index:

    cat_columns_values[col] = data[col].unique()

cat_columns['Values'] = cat_columns_values.values()
plt.figure(figsize = (15,10))

barplotdata = data.groupby(['Neighborhood'])['SalePrice'].count()

q = sns.barplot(x = barplotdata.index,y = barplotdata)

q.set_xticklabels(q.get_xticklabels(),rotation = 90)
salespricelabels = ['100k','150k','200k','350k','400k','450k','500k','550k','600k','650k','700k','750k','800k']

for i,col in enumerate(cat_columns.index):

    p = sns.boxplot(x = col,y='SalePrice', data = data)

    p.set_yticklabels(salespricelabels)

    p.set_xticklabels(p.get_xticklabels(),rotation = 45)

    p.set_title(col + ' vs SalePrice')

    plt.show()
fig,ax = plt.subplots(nrows = 2,ncols = 2,figsize = (15,8))

sns.stripplot(x = 'HouseStyle',y = 'SalePrice', data = data,jitter = False,ax = ax[0][0])

ax[0][0].set_xticklabels(ax[0][0].get_xticklabels(),rotation = 45)

ax[0][0].set_title('House Style vs SalePrice')

sns.boxplot(x = 'Neighborhood',y = 'SalePrice',data = data, ax = ax[0][1])

ax[0][1].set_xticklabels(ax[0][1].get_xticklabels(),rotation = 90)

ax[0][1].set_title('Neighborhood vs SalePrice')

sns.barplot(x = barplotdata.index,y = barplotdata,ax = ax[1][0],fc = 'cyan',ec = 'k')

ax[1][0].set_xticklabels(ax[1][0].get_xticklabels(),rotation = 90)

ax[1][0].set_title('Number of houses sold by neighborhood')

ax[1][0].set_ylabel("Number of houses sold")

sns.stripplot(x = 'Foundation',y = 'SalePrice',data = data,jitter = False,ax = ax[1][1])

ax[1][1].set_xticklabels(ax[1][1].get_xticklabels(),rotation = 90)

ax[1][1].set_title('Foundation vs SalePrice')

fig.tight_layout()
plt.figure(figsize = (15,15))

sns.distplot(a = data['YearBuilt'],kde = True,

             hist_kws = {"histtype" : "barstacked","fc":"red"},

              kde_kws = {'cut':0}

            )
plt.figure(figsize = (15,15))

sns.distplot(a = data['LotFrontage'],kde = True,

             hist_kws = {"histtype" : "barstacked","fc":"k"},

              kde_kws = {'cut':0}

            )

y_data = data['SalePrice']

data.drop('SalePrice',axis = 1,inplace = True)
X_train, X_valid, y_train, y_valid = train_test_split(data,y_data, train_size=0.8, test_size=0.2,

                                                      random_state=0)
int_columns = [col for col in data.columns if (data[col].dtype=='int64' or data[col].dtype=='float64')]
se = SimpleImputer(strategy = 'mean')

X_train_int_imp = X_train

X_valid_int_imp = X_valid

X_test_int_imp = X_test

X_train_int_imp[int_columns] = se.fit_transform(X_train[int_columns])

X_valid_int_imp[int_columns] = se.transform(X_valid[int_columns])

X_test_int_imp[int_columns] = se.transform(X_test[int_columns])

X_train_int_imp.columns = X_train.columns

X_valid_int_imp.columns = X_train.columns

X_test_int_imp.columns = X_train.columns
se = SimpleImputer(strategy = 'most_frequent')

X_train_cat_imp = X_train_int_imp

X_valid_cat_imp = X_valid_int_imp

X_test_cat_imp = X_test_int_imp

X_train_cat_imp[cat_columns.index] = se.fit_transform(X_train[cat_columns.index])

X_valid_cat_imp[cat_columns.index] = se.transform(X_valid[cat_columns.index])

X_test_cat_imp[cat_columns.index] = se.transform(X_test[cat_columns.index])
X_train_cat_imp.columns = X_train.columns

X_valid_cat_imp.columns = X_train.columns

X_test_cat_imp.columns = X_train.columns
scaler = StandardScaler()

X_train_scale = X_train_cat_imp

X_valid_scale = X_valid_cat_imp

X_test_scale = X_test_cat_imp

X_train_scale[int_columns] = scaler.fit_transform(X_train_cat_imp[int_columns])

X_valid_scale[int_columns] = scaler.transform(X_valid_cat_imp[int_columns])

X_test_scale[int_columns] = scaler.transform(X_test_cat_imp[int_columns])
cat_columns
cat_cardinality_less_10 = cat_columns.loc[cat_columns['Number of unique values'] <= 10]

cat_cardinality_less_10
OHE = OneHotEncoder(handle_unknown = 'ignore',sparse = False)

X_train_OHE = pd.DataFrame(OHE.fit_transform(X_train_scale[cat_cardinality_less_10.index]))

X_valid_OHE = pd.DataFrame(OHE.transform(X_valid_scale[cat_cardinality_less_10.index]))

X_test_OHE = pd.DataFrame(OHE.transform(X_test_scale[cat_cardinality_less_10.index]))



X_train_OHE.index = X_train_scale.index

X_valid_OHE.index = X_valid_scale.index

X_test_OHE.index = X_test_scale.index



X_train_del = X_train_scale.drop(cat_cardinality_less_10.index,axis = 1)

X_valid_del = X_valid_scale.drop(cat_cardinality_less_10.index,axis = 1)

X_test_del = X_test_scale.drop(cat_cardinality_less_10.index,axis = 1)





X_train_total_OHE = pd.concat([X_train_del,X_train_OHE],axis= 1)

X_valid_total_OHE = pd.concat([X_valid_del,X_valid_OHE],axis = 1)

X_test_total_OHE = pd.concat([X_test_del,X_test_OHE],axis = 1)
cardinality_col_more_10 = list(set(cat_columns.index)- set(cat_cardinality_less_10.index))

cardinality_col_more_10
le = LabelEncoder()

final_X_train = X_train_total_OHE

final_X_valid = X_valid_total_OHE

final_X_test = X_test_total_OHE



for col in cardinality_col_more_10:

    final_X_train[col] = le.fit_transform(X_train_total_OHE[col])

    final_X_valid[col] = X_valid_total_OHE[col].map(lambda s: 'other' if s not in le.classes_ else s)

    le.classes_ = np.append(le.classes_, 'other')

    final_X_valid[col] = le.transform(X_valid_total_OHE[col])

    final_X_test[col] =  le.transform(X_test_total_OHE[col])
print(final_X_train.shape)

print(final_X_valid.shape)

print(final_X_test.shape)
model = RandomForestRegressor(n_estimators = 500,random_state=0)
model.fit(final_X_train,y_train)
preds = model.predict(final_X_valid)
from sklearn.metrics import mean_absolute_error

score = mean_absolute_error(y_valid,preds)

score
from xgboost import XGBRegressor

model2 = XGBRegressor(n_estimators = 500)

model2.fit(final_X_train,y_train)

preds2 = model2.predict(final_X_valid)

score2 = mean_absolute_error(y_valid,preds2)
score2
results_on_test = model.predict(final_X_test)
test_id = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
finalresult = pd.DataFrame()

finalresult['Id'] = test_id['Id']

finalresult['SalePrice'] = results_on_test

finalresult.to_csv("FinalPredictions.csv",index = False)
finalresult.head()