# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.preprocessing import Imputer

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Loading datasets into pandas dataframes
application_train = pd.read_csv('../input/application_train.csv')
POS_CASH_balance = pd.read_csv('../input/POS_CASH_balance.csv')
bureau_balance = pd.read_csv('../input/bureau_balance.csv')
previous_application = pd.read_csv('../input/previous_application.csv')
installments_payments = pd.read_csv('../input/installments_payments.csv')
credit_card_balance = pd.read_csv('../input/credit_card_balance.csv')
bureau = pd.read_csv('../input/bureau.csv')
application_test = pd.read_csv('../input/application_test.csv')
#Function for calculating missing data
def missing_data(data):
    total = data.isnull().sum().sort_values(ascending = False)
    proportion = (data.isnull().sum()/data.isnull().count()).sort_values(ascending = False)
    return pd.concat([total, proportion], axis=1, keys=['Total', 'Proportion'])
#Function for plotting missing value bar-plot
def missing_data_barplot(data, df_name,xtick_font=7, rotation=90):
    data = data[data['Proportion']>0]
    fig = plt.figure(figsize=(18,6))
    sns.barplot(x="index", y="Proportion", data=data, palette="Blues_d")
    plt.xticks(rotation =rotation,fontsize =xtick_font)
    plt.title("Proportion of Missing values in %s Dataset" %df_name)
    plt.ylabel("PROPORTION")
    plt.xlabel("COLUMNS")
missing_df_application = missing_data(application_train).reset_index()
missing_data_barplot(missing_df_application, "Application", xtick_font=7)
missing_df_credit = missing_data(credit_card_balance).reset_index()
missing_data_barplot(missing_df_credit, "Credit Card", xtick_font=10, rotation=45)
credit_cols = [ 'AMT_INCOME_TOTAL', 'AMT_CREDIT','AMT_ANNUITY', 'AMT_GOODS_PRICE', 'TARGET']

#for pairplot, first we have to impute missing values. Here, I am using sklearn Imputer to replace Nan with Mean
pairplot_temp_df = application_train[credit_cols].copy()
Imputer = Imputer(missing_values ='NaN', strategy='mean', axis=0)
Imputer = Imputer.fit(pairplot_temp_df)
pairplot_temp_df = pd.DataFrame(Imputer.transform(pairplot_temp_df.values))
pairplot_temp_df.columns = credit_cols

## Alternative way to remove null values
#tempdf = tempdf[(tempdf["AMT_GOODS_PRICE"].notnull()) & (tempdf["AMT_ANNUITY"].notnull())]

# Pairplot of the resulting dataframe
sns.pairplot(data=pairplot_temp_df, hue='TARGET')
import missingno as msno
msno.matrix(application_train.sample(100))
# Zooming-in on the middle section
msno.matrix(application_train.iloc[0:100, 44:91])
'''Credit for the code goes to jpmiller'''
train = application_train.copy()
train['incomplete'] = 1
train.loc[train.isnull().sum(axis=1) < 35, 'incomplete'] = 0

mean_c = np.mean(train.loc[train['incomplete'] == 0, 'TARGET'].values)
mean_i = np.mean(train.loc[train['incomplete'] == 1, 'TARGET'].values)
print('default % for more complete: {:.2}% \ndefault % for less complete: {:.2}%'.format(mean_c*100, mean_i*100))
print('\nBorrowers with incomplete applications are ~30% more likely to default')
# Seperating Numerical and Categorical Data
application_cat = application_train.select_dtypes('object')
application_num = application_train.select_dtypes(exclude=['object'])
credit_cat = credit_card_balance.select_dtypes('object')
credit_num = credit_card_balance.select_dtypes(exclude=['object'])
# Box Plot - Credit Dataset
fig, axes = plt.subplots(11,2, figsize=(20,60))
for i,col in enumerate(credit_num):
        sns.boxplot(y=credit_num[col], ax=axes[(i-1)%11][(i-1)%2])
credit_num.describe()
# Categorical data validation using bar graphs x-values
def plot_categorical(data, col, size=[8 ,4], xlabel_angle=0, title=''):
    '''use this for ploting the count of categorical features'''
    plotdata = data[col].value_counts()
    plt.figure(figsize = size)
    sns.barplot(x = plotdata.index, y=plotdata.values)
    plt.title(title)
    if xlabel_angle!=0: 
        plt.xticks(rotation=xlabel_angle)
    plt.show()

for i,col in enumerate(application_cat):
    plot_categorical(data=application_cat, size=[12 ,2], col=col, xlabel_angle=90, title=col)
#Non-categorical data validation using  graphs x-values - Credit Dataset

# def plot_numerical(data, col, bins=50, index=1):
#     '''use this for ploting the distribution of numercial features'''
#     plt.title("Distribution of %s" % str(col))
#     sns.distplot(, ax=axes[(index-1)%11][(index-1)%2])
#     plt.show()

fig, axes = plt.subplots(11,2, figsize=(20,60))
for i,col in enumerate(credit_num):
    sns.distplot(credit_num[col].dropna(), kde=False, bins=50, ax=axes[(i-1)%11][(i-1)%2])