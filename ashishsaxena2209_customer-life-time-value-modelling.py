import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from matplotlib import pyplot as plt
src_file = '../input/online-retail-data-set-from-uci-ml-repo/Online Retail.xlsx'
def intconverter(x):

    if not np.isnan(x):

        return np.int32(x)

    else:

        return np.nan

    

def floatconverter(x):

    if not np.isnan(x):

        return np.float32(x)

    else:

        return np.nan
converter_dict = {'Quantity': 'intconverter', 'UnitPrice': 'floatconverter', 'CustomerID': 'intconverter'}
%%time

#original_df = pd.read_excel(inp_file, dtype = {'InvoiceNo': str, 'StockCode': str,'Description': str, 'Quantity': np.int32, 

                                               #'InvoiceDate':np.datetime64, 'UnitPrice': np.float32, 'CustomerID': np.int32, 'Country':str})

original_df = pd.read_excel(src_file, parse_dates=True)
%%time

original_df.head(10)
%%time

# Converting customerID to string

original_df['CustomerID'] = original_df['CustomerID'].astype('str')

original_df.describe(include=['object','int', 'float','datetime'])
# Dropping duplicates if any

duplicate_df_v1 = original_df.drop_duplicates()



duplicate_df_v1.describe(include=['object','int', 'float','datetime'])
%%time

# Getting null index for customer Ids

null_idx = duplicate_df_v1[duplicate_df_v1.CustomerID.isnull()].index.values

null_idx = null_idx.tolist()

unique_invoiceno_with_nullcustID = duplicate_df_v1.iloc[null_idx,duplicate_df_v1.columns.get_loc('InvoiceNo')].unique()



# First check wether we will be able to find any CustomerID for above invoice numbers. 

none_InvNum_found_with_custID = True

for inv_num in unique_invoiceno_with_nullcustID:

    x = original_df[original_df.InvoiceNo == inv_num].count()

    if x.CustomerID == 0:

        pass

    else:

        none_InvNum_found_with_custID = False



print('Any invoice number (for indexes with missing CustomerID) found? ', not none_InvNum_found_with_custID,'\n')
%%time

duplicate_df_v2 = duplicate_df_v1.drop(index=duplicate_df_v1[duplicate_df_v1['CustomerID']=='nan'].index.values)

# Removing indexes with negative Quantity and UnitPrice

duplicate_df_v2 = duplicate_df_v2[(duplicate_df_v2['Quantity'] > 0) & (duplicate_df_v2['UnitPrice'] > 0)]

duplicate_df_v2.describe(include=['object','int', 'float','datetime']).T
fig = plt.figure(figsize=(15, 8))

ax = plt.axes()

#a = duplicate_df_v2.Country.value_counts().plot(kind='bar')

ax.bar(range(0,len(duplicate_df_v2.Country.value_counts())), height=duplicate_df_v2.Country.value_counts())

ax.set_xticks(range(0,len(duplicate_df_v2.Country.value_counts())))

ax.set_xticklabels(duplicate_df_v2.Country.unique().tolist(), rotation='vertical')

ax.tick_params(axis='x', colors='white')

ax.tick_params(axis='y', colors='white')

ax.set_title('Figure 1: Country Counts')

plt.grid(axis='y')
#duplicate_df_v3 = duplicate_df_v2[duplicate_df_v2['Country'] == 'United Kingdom']



# Dropping country column, stock code and Description

duplicate_df_v3 = duplicate_df_v2.drop(columns=['Country','StockCode','Description'])



duplicate_df_v3.describe(include=['object','int', 'float','datetime']).T
duplicate_df_v4 = duplicate_df_v3.sort_values(by=['CustomerID','InvoiceDate'])

# Calulate total purchase

duplicate_df_v4['TotalOrderValue'] = duplicate_df_v3['Quantity'] * duplicate_df_v3['UnitPrice']

duplicate_df_v4.head(10)
duplicate_df_v4_group=duplicate_df_v4.groupby('CustomerID').agg({'InvoiceDate': lambda date: (date.max() - date.min()).days,

                                                                 'InvoiceNo': 'count',

                                                                 'Quantity': 'sum',

                                                                 'TotalOrderValue': 'sum'})

# Renaming the column labels

duplicate_df_v4_group.columns=['purchase_freshness','total_num_transactions','total_num_units','total_money_spent']

duplicate_df_v4_group.head()
# Average Order Value

duplicate_df_v4_group['avg_order_value'] = duplicate_df_v4_group['total_money_spent'] / duplicate_df_v4_group['total_num_transactions']
# Purchase Frequency

purchase_frequency = sum(duplicate_df_v4_group['total_num_transactions']) / duplicate_df_v4_group.shape[0]
# Churn Rate

repeat_rate = duplicate_df_v4_group[duplicate_df_v4_group.total_num_transactions > 1].shape[0] / duplicate_df_v4_group.shape[0]

churn_rate = 1 - repeat_rate
# Profit Margin

# Assuming that this Online Retail company have 7% Profit Margine out of Total Sales.

duplicate_df_v4_group['profit_margin'] = duplicate_df_v4_group['total_money_spent'] * 0.07
# Customer Lifeatime Value

duplicate_df_v4_group['CLTV']=((duplicate_df_v4_group['avg_order_value'] * purchase_frequency) / churn_rate) * duplicate_df_v4_group['profit_margin']
duplicate_df_v4_group.head(10)
dv4_group = duplicate_df_v4.groupby('CustomerID').InvoiceDate.max().reset_index(drop=False)
dv4_group.columns = ['CustomerID', 'Max_InvoiceDate']

dv4_group.head()
duplicate_df_v5 = pd.merge(duplicate_df_v4, dv4_group, on='CustomerID', how='left')

duplicate_df_v5.head(10)
duplicate_df_v5['Difference_Days_from_MaxDate'] = (duplicate_df_v5['Max_InvoiceDate'] - duplicate_df_v5['InvoiceDate']).dt.days

duplicate_df_v5.describe()
duplicate_df_v4['Month_Yr'] = duplicate_df_v4['InvoiceDate'].apply(lambda x: x.strftime('%b-%Y'))

duplicate_df_v4.head(10)
%%time

# Summerizing Data

sale_by_Month_Yr = duplicate_df_v4.pivot_table(index=['CustomerID'],columns=['Month_Yr'],

                                               values='TotalOrderValue',aggfunc='sum',fill_value=0).reset_index()



# Sort columns in ascending order of dates 

from datetime import datetime

dates = sale_by_Month_Yr.columns.to_list()[1:]

dates.sort(key = lambda date: datetime.strptime(date,'%b-%Y'))

dates.insert(0,'CustomerID')

sale_by_Month_Yr = sale_by_Month_Yr[dates]



# Calulating CLV

sale_by_Month_Yr['Latest_6Months_Total_Purchase'] = sale_by_Month_Yr.iloc[:,8:].sum(axis=1)

sale_by_Month_Yr.head(10)
#  Calculating Recency, Frequency and Monetary

PRESENT = datetime(2011,7,1) # Because we are considering transcations upto month Jun-2011 thus taking 1st July as reference

rfm = duplicate_df_v4[duplicate_df_v4['InvoiceDate'] < PRESENT].groupby('CustomerID').agg({'InvoiceDate': lambda date: (PRESENT - date.max()).days,

                                        'InvoiceNo': 'count','TotalOrderValue': 'sum'})



rfm.columns=['recency','frequency','monetary']



sale_by_Month_Yr_withrfm = pd.merge(sale_by_Month_Yr, rfm, on='CustomerID', how='left')



sale_by_Month_Yr_withrfm = sale_by_Month_Yr_withrfm[['CustomerID','Dec-2010','Jan-2011','Feb-2011',	'Mar-2011',	'Apr-2011',	'May-2011','Jun-2011',

                                                     'recency','frequency','Jul-2011','Aug-2011','Sep-2011','Oct-2011','Nov-2011','Dec-2011',

                                                     'Latest_6Months_Total_Purchase']]

sale_by_Month_Yr_withrfm.head(10)
sale_by_Month_Yr_withrfm_clean = sale_by_Month_Yr_withrfm.dropna()

sale_by_Month_Yr_withrfm_clean.describe().T
# Splitting into train and test set

train_set = sale_by_Month_Yr_withrfm_clean.sample(frac=0.70, random_state=22)

test_set = sale_by_Month_Yr_withrfm_clean.drop(train_set.index)
# Creating independent variables aka X for train and test.

independent_vars_train = train_set[train_set.columns.to_list()[1:-7]]

independent_vars_test = test_set[test_set.columns.to_list()[1:-7]]

independent_vars_test.head(10)
# Creating target set for test and train

target_vars_train = train_set['Latest_6Months_Total_Purchase']

target_vars_test = test_set['Latest_6Months_Total_Purchase']
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(independent_vars_train.values, target_vars_train.values)
train_score = lr.score(independent_vars_train.values, target_vars_train.values)

print('Regressor score on train data:', train_score)

test_score = lr.score(independent_vars_test.values, target_vars_test.values)

print('Regressor score on test data:', test_score)
from sklearn.preprocessing import MinMaxScaler
# Scaling data in default range

mscalar = MinMaxScaler()

independent_vars_train_mmscaled = mscalar.fit_transform(independent_vars_train) # Fitting and transformation on train data

independent_vars_test_mmscaled = mscalar.transform(independent_vars_test) # only transformation on test data
# Fitting linear regressor on scaled data

lr_scaled = LinearRegression()

lr_scaled.fit(independent_vars_train_mmscaled, target_vars_train)
train_score_sc = lr_scaled.score(independent_vars_train_mmscaled, target_vars_train)

print('Train score is:',train_score_sc)

test_score_sc = lr_scaled.score(independent_vars_test_mmscaled, target_vars_test)

print('Test score is:',test_score_sc)
import copy

independent_vars_train_mmscaled_stats  =  copy.deepcopy(independent_vars_train_mmscaled)

ones = np.ones((independent_vars_train_mmscaled_stats.shape[0],1))

# OLS regressor expects first column to be a constant having all values 1. Thus, stacking a column to the independent variables

new_in_train = np.hstack((ones, independent_vars_train_mmscaled_stats)) # stacking on train only

from statsmodels.regression.linear_model import OLS

regressor_SLR_OLS = OLS(endog = target_vars_train, exog = new_in_train[:,:-1]).fit()
regressor_SLR_OLS.summary()
#new_in_train = np.hstack((oness, independent_vars_train_mmscaled_stats))

#from statsmodels.regression.linear_model import OLS

regressor_SLR_OLS = OLS(endog = target_vars_train, exog = new_in_train[:,[0,1,3,4,5,6,7,8]]).fit()
regressor_SLR_OLS.summary()
# Again fitting regressor after dropping x7 feature

regressor_SLR_OLS = OLS(endog = target_vars_train, exog = new_in_train[:,[0,1,3,4,5,6,7]]).fit()
regressor_SLR_OLS.summary()
lin_reg2 = LinearRegression()

lin_reg2.fit(independent_vars_train_mmscaled[:,[0,2,3,4,5,6,7]], target_vars_train)
print('Train score is:',lin_reg2.score(independent_vars_train_mmscaled[:,[0,2,3,4,5,6,7]], target_vars_train))

print('Test score is:',lin_reg2.score(independent_vars_test_mmscaled[:,[0,2,3,4,5,6,7]], target_vars_test))
from sklearn.tree import DecisionTreeRegressor



des_reg = DecisionTreeRegressor(random_state=42, max_depth = 6, min_samples_split= 60)

des_reg.fit(independent_vars_train, target_vars_train)
print('Decision Tree model score on training data:',des_reg.score(independent_vars_train, target_vars_train))

print('Decision Tree model score on testing data:', des_reg.score(independent_vars_test, target_vars_test))