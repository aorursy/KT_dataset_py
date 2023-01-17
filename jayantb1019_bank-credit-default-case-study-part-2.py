!pip install sidetable
#importing required Libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import sidetable
from tabulate import tabulate

import warnings
warnings.filterwarnings('ignore')
#setting max number of columns to be displayed to 100, to get a better view.
pd.set_option('display.max_columns',100)
pd.set_option('display.max_rows',100)
application_data=pd.read_csv('../input/bank-loans-dataset/application_data.csv')
#Finding the columns that have more than or equal to 50% null values and storing it to columns_to_drop.
columns_to_drop=application_data.columns[100*application_data.isnull().sum()/len(application_data)>=50]
#dropping the columns where the null values are >= 50%.
application_data.drop(labels=columns_to_drop,axis=1,inplace=True)
# Checking for Disguised Missing Values
def cat_value_counts(column_name) : 
    print(tabulate(pd.DataFrame(application_data.stb.freq([column_name])), headers='keys', tablefmt='psql'))
    print(pd.DataFrame(application_data[column_name]).stb.missing(),'\n\n\n')

#Replacing XAN with np.nan in Gender column : 
application_data['CODE_GENDER'] = application_data['CODE_GENDER'].replace('XNA',np.nan)

columns_to_convert = ['DAYS_BIRTH','DAYS_EMPLOYED','DAYS_REGISTRATION','DAYS_ID_PUBLISH']
application_data[columns_to_convert] = application_data[columns_to_convert].abs()
# Adding a new column "AGE_YEARS" using 'DAYS_BIRTH' with age in years
def days_to_years(x) : 
    if x < 0 : 
        x = -1*x 
    return x//365
application_data['AGE_YEARS'] = application_data['DAYS_BIRTH'].apply(days_to_years)
# AMT_INCOME_TOTAL - binning continuous variables
min_income = int(application_data['AMT_INCOME_TOTAL'].min())
max_income = int(application_data['AMT_INCOME_TOTAL'].max())


bins = [0,25000,50000,75000,100000,125000,150000,175000,200000,225000,250000,275000,300000,325000,350000,375000,400000,425000,450000,475000,500000,10000000000]
intervals = ['0-25000', '25000-50000','50000-75000','75000-100000','100000-125000', '125000-150000', '150000-175000','175000-200000',
       '200000-225000','225000-250000','250000-275000','275000-300000','300000-325000','325000-350000','350000-375000',
       '375000-400000','400000-425000','425000-450000','450000-475000','475000-500000','500000 and above']

application_data['AMT_INCOME_CAT']=pd.cut(application_data['AMT_INCOME_TOTAL'],bins,labels=intervals)

#AMT_CREDIT
bins = [0,150000,200000,250000,300000,350000,400000,450000,500000,550000,600000,650000,700000,750000,800000,850000,900000,1000000000]
intervals = ['0-150000', '150000-200000','200000-250000', '250000-300000', '300000-350000', '350000-400000','400000-450000',
        '450000-500000','500000-550000','550000-600000','600000-650000','650000-700000','700000-750000','750000-800000',
        '800000-850000','850000-900000','900000 and above']

application_data['AMT_CREDIT_RANGE']=pd.cut(application_data['AMT_CREDIT'],bins=bins,labels=intervals)
application_data['AMT_CREDIT_RANGE'] = application_data['AMT_CREDIT_RANGE'].astype('category')

columnsForAnalysis = ['SK_ID_CURR', 'TARGET', 'NAME_CONTRACT_TYPE', 'CODE_GENDER',
       'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL',
       'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
       'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
       'NAME_HOUSING_TYPE',
       'DAYS_EMPLOYED','FLAG_MOBIL', 'FLAG_CONT_MOBILE',
       'FLAG_EMAIL', 'OCCUPATION_TYPE', 'CNT_FAM_MEMBERS',
       'REGION_RATING_CLIENT_W_CITY',
                      'REG_CITY_NOT_LIVE_CITY',
       'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY',
       'ORGANIZATIOdN_TYPE', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
        'DAYS_LAST_PHONE_CHANGE' ,'AGE_YEARS', 'AMT_INCOME_CAT',
       'AMT_CREDIT_RANGE']
prev_data = pd.read_csv('../input/bank-loans-dataset/previous_application.csv')
# selecting columns from application Data : 
application_data_columns = '''AMT_ANNUITY
AMT_INCOME_TOTAL
AMT_CREDIT
AMT_GOODS_PRICE
NAME_CONTRACT_TYPE
CODE_GENDER
NAME_INCOME_TYPE
DAYS_EMPLOYED
NAME_EDUCATION_TYPE
SK_ID_CURR
AGE_YEARS
AMT_INCOME_CAT
AMT_CREDIT_RANGE
'''
application_data_columns = application_data_columns.splitlines()
selected_application_data = application_data[application_data_columns]
# selecting columns from Previous application Data : 
prev_application_data_columns = '''AMT_ANNUITY
AMT_APPLICATION
AMT_CREDIT
AMT_GOODS_PRICE
CHANNEL_TYPE
CODE_REJECT_REASON
DAYS_DECISION
NAME_CASH_LOAN_PURPOSE
NAME_CLIENT_TYPE
NAME_CONTRACT_STATUS
NAME_CONTRACT_TYPE
NAME_GOODS_CATEGORY
NAME_PORTFOLIO
NAME_PRODUCT_TYPE
NAME_YIELD_GROUP
PRODUCT_COMBINATION
SK_ID_CURR
SK_ID_PREV
'''
prev_application_data_columns = prev_application_data_columns.splitlines()
selected_prev_data = prev_data[prev_application_data_columns]
# merging data , adding suffix _prev and _curr
merged_data = pd.merge(left=selected_application_data,right=selected_prev_data,on='SK_ID_CURR',how='inner',suffixes=('_curr','_prev'))
merged_data.info()
merged_data.dtypes
# replacing disguised missing values 'XAP', 'XNA'
for column in merged_data.columns : 
    if merged_data[column].dtype == 'object' or merged_data[column].dtype.name == 'category' : 
        merged_data[column].replace({'XAP' : np.nan, 'XNA' : np.nan}, inplace=True)
#Finding Percentage of Missing values in each and every column
def color_red(value):
    '''
      Colors elements in a dateframe
      green if nulls <50% and red if
      >=50%.
    '''
    if value >=50:
        color = 'red'
    elif value <50:
        color = 'green'

    return 'color: %s' % color
pd.set_option('precision', 4)
missing_data = 100*(merged_data.reset_index().isnull().sum())/merged_data.shape[0]

missing_data = pd.DataFrame(data={'Column Name' :missing_data.index, 'Null Percentage' : missing_data.values})
missing_data.style.applymap(color_red,subset=['Null Percentage'])
# dropping columns with high null values 
null_info = pd.DataFrame(100*merged_data.reset_index().isnull().sum()/len(merged_data))
null_info.columns = ['Null Percentage']
high_nulls = null_info[null_info['Null Percentage'] >= 50].index
print(high_nulls)
merged_data = merged_data.drop(columns=high_nulls)
# remaining columns 
merged_data.columns
# taking absolute values of days of processsing 
merged_data['DAYS_DECISION'] = merged_data['DAYS_DECISION'].abs()
#Finding the best values to impute below columns that have <=13% of null values

#Finding columns with <= 13% missing columns

null_info = pd.DataFrame(100*merged_data.isnull().sum()/len(merged_data))
null_info.columns = ['Null Percentage']
null_info[(null_info['Null Percentage'] > 0) & (null_info['Null Percentage'] <=0.13)]
# Boxplot to check for outliers
merged_data['AMT_CREDIT_prev'].plot.box()
plt.title('\n Box Plot of AMT_CREDIT_prev')

# Calculating Quantiles
print('Quantile\tAMT_CREDIT_prev')
merged_data['AMT_CREDIT_prev'].quantile([0.5,0.8,0.85,0.90,0.95,1])
print('Data type of PRODUCT_COMBINATION : ',merged_data['PRODUCT_COMBINATION'].dtype,'\n\n')
print('Category\tNormalized Count\n\n',merged_data['PRODUCT_COMBINATION'].value_counts(normalize=True))
data = merged_data['PRODUCT_COMBINATION'].value_counts(normalize=True)
plt.bar(data.index,data.values)
# data.hist()
plt.xticks(rotation=90)
plt.ylabel('Normalized Value Counts')
plt.title('\nPRODUCT_COMBINATION');
#NAME_CLIENT_TYPE
print('Data type of NAME_CLIENT_TYPE : ',merged_data['NAME_CLIENT_TYPE'].dtype,'\n\n')
print('Category\tNormalized Count\n\n',merged_data['NAME_CLIENT_TYPE'].value_counts(normalize=True))
data = merged_data['NAME_CLIENT_TYPE'].value_counts(normalize=True)
plt.bar(data.index,data.values)
# data.hist()
plt.xticks(rotation=90)
plt.ylabel('Normalized Value Counts')
plt.title('\nNAME_CLIENT_TYPE');
print('Data type of NAME_CONTRACT_TYPE_prev : ',merged_data['NAME_CONTRACT_TYPE_prev'].dtype,'\n\n')
print('Category\tNormalized Count\n\n',merged_data['NAME_CONTRACT_TYPE_prev'].value_counts(normalize=True))
data = merged_data['NAME_CONTRACT_TYPE_prev'].value_counts(normalize=True)
plt.bar(data.index,data.values)
# data.hist()
plt.xticks(rotation=90)
plt.ylabel('Normalized Value Counts')
plt.title('\nNAME_CONTRACT_TYPE_prev');
# unique categories 
merged_data['NAME_CONTRACT_STATUS'].unique()
# splitting into four data frames wrt to NAME_CONTRACT_STATUS

merged_a = merged_data[merged_data['NAME_CONTRACT_STATUS'] == 'Approved']
merged_c = merged_data[merged_data['NAME_CONTRACT_STATUS'] == 'Canceled']
merged_r = merged_data[merged_data['NAME_CONTRACT_STATUS'] == 'Refused']
merged_u = merged_data[merged_data['NAME_CONTRACT_STATUS'] == 'Unused offer']
column_name = 'NAME_CONTRACT_STATUS'
print(tabulate(pd.DataFrame(merged_data.stb.freq([column_name])), headers='keys', tablefmt='psql'))
print(pd.DataFrame(merged_data[column_name]).stb.missing(),'\n')
merged_col_for_analysis = '''
AMT_ANNUITY_prev
AMT_APPLICATION
AMT_CREDIT_prev
AMT_GOODS_PRICE_prev
CHANNEL_TYPE
CODE_REJECT_REASON
DAYS_DECISION
NAME_CASH_LOAN_PURPOSE
NAME_CLIENT_TYPE
NAME_CONTRACT_TYPE_prev
NAME_GOODS_CATEGORY
NAME_PORTFOLIO
NAME_PRODUCT_TYPE
NAME_YIELD_GROUP
PRODUCT_COMBINATION'''
merged_col_for_analysis = merged_col_for_analysis.splitlines()
# taking absolute values of days decision
merged_data['DAYS_DECISION'] = merged_data['DAYS_DECISION'].abs()
merged_data['DAYS_DECISION'].describe()
# Box plots of the above numerical variables 
merged_outlier_check_col = [
    'AMT_ANNUITY_prev',
'AMT_APPLICATION',
'AMT_CREDIT_prev',
'AMT_GOODS_PRICE_prev',
'DAYS_DECISION'
]

fig,ax = plt.subplots(3,2)
fig.set_figheight(15)
fig.set_figwidth(15)
ax[0,0].set_yscale('log')
ax[0,1].set_yscale('log')
ax[1,0].set_yscale('log')
ax[1,1].set_yscale('log')

ax[0,0].set(ylabel ='Annuity in Log Scale')
ax[0,1].set(ylabel ='Application Amount in Log Scale')
ax[1,0].set(ylabel ='Credit Amount in Log Scale')
ax[1,1].set(ylabel ='Goods Price in Log Scale')
ax[2,0].set(ylabel ='Processing Days')

merged_data[merged_outlier_check_col[0]].plot.box(ax=ax[0,0],);
merged_data[merged_outlier_check_col[1]].plot.box(ax=ax[0,1]);
merged_data[merged_outlier_check_col[2]].plot.box(ax=ax[1,0]);
merged_data[merged_outlier_check_col[3]].plot.box(ax=ax[1,1]);

merged_data[merged_outlier_check_col[4]].plot.box(ax=ax[2,0]); 
ax[2,1].axis('off')
print('Box Plots of' + ' '.join(merged_outlier_check_col) +'\n')
# quantiles for outlier checks
pd.options.display.float_format = '{:,.2f}'.format
for col in merged_outlier_check_col : 
    print(col,'\n',merged_data[col].quantile([0.5,0.8,0.85,0.90,0.95,1]),'\n\n')
#AMT_CREDIT_prev
min_credit = int(merged_data['AMT_CREDIT_prev'].min())
max_credit = int(merged_data['AMT_CREDIT_prev'].max())

bins = [0,25000,50000,75000,100000,125000,150000,175000,200000,225000,250000,275000,300000,325000,350000,375000,400000,425000,450000,475000,500000,10000000000]
intervals = ['0-25000', '25000-50000','50000-75000','75000-100000','100000-125000', '125000-150000', '150000-175000','175000-200000',
       '200000-225000','225000-250000','250000-275000','275000-300000','300000-325000','325000-350000','350000-375000',
       '375000-400000','400000-425000','425000-450000','450000-475000','475000-500000','500000 and above']

merged_data['AMT_CREDIT_prev_cat']=pd.cut(merged_data['AMT_CREDIT_prev'],bins,labels=intervals)
print('Credit Range [Prev]\t Count')
print(merged_data['AMT_CREDIT_prev_cat'].value_counts())

credit_cat = merged_data['AMT_CREDIT_prev_cat'].value_counts()
plt.hist(credit_cat)
# (merged_data['AMT_CREDIT_prev_cat'].dropna()).plot.hist()

plt.title('\n Previous Credit Amount vs No of Applications')
plt.xticks(rotation=90);
#AMT_APPLICATION
min_app_amt = int(merged_data['AMT_APPLICATION'].min())
max_app_amt = int(merged_data['AMT_APPLICATION'].max())


bins = [0,25000,50000,75000,100000,125000,150000,175000,200000,225000,250000,275000,300000,325000,350000,375000,400000,425000,450000,475000,500000,10000000000]
intervals = ['0-25000', '25000-50000','50000-75000','75000-100000','100000-125000', '125000-150000', '150000-175000','175000-200000',
       '200000-225000','225000-250000','250000-275000','275000-300000','300000-325000','325000-350000','350000-375000',
       '375000-400000','400000-425000','425000-450000','450000-475000','475000-500000','500000 and above']

merged_data['AMT_APPLICATION_cat']=pd.cut(merged_data['AMT_APPLICATION'],bins,labels=intervals)
print('AMT_APPLICATION [Prev]\t Count')
print(merged_data['AMT_APPLICATION_cat'].value_counts())

credit_cat = merged_data['AMT_APPLICATION_cat'].value_counts()
plt.hist(merged_data['AMT_APPLICATION_cat'].dropna())

plt.title('\n Application Amount vs No of Applications')
plt.xticks(rotation=90);
# function for categorical variable univariate analysis

def merged_cat_univariate_analysis(column_name,figsize=(10,5)) : 
    # print unique values
    print('Approved\n', merged_a[column_name].unique(),'\n')
    print('Canceled\n',merged_c[column_name].unique(),'\n')
    print('Refused\n',merged_r[column_name].unique(),'\n')
    print('Unused offer\n',merged_u[column_name].unique(),'\n')
    
    # column vs target count plot
    plt.figure(figsize=figsize)
    ax = sns.countplot(x=column_name,hue='NAME_CONTRACT_STATUS',data=merged_data)
    title = column_name + ' vs Number of Applications'
    ax.set(title= title)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2,
                height + 10,
                format(height),
                ha="center")
    # Percentages 
    print('Approved\n', merged_a[column_name].unique(),'\n')
    print(tabulate(pd.DataFrame(merged_a.stb.freq([column_name])), headers='keys', tablefmt='psql'),'\n')
    print('Canceled\n',merged_c[column_name].unique(),'\n')
    print(tabulate(pd.DataFrame(merged_c.stb.freq([column_name])), headers='keys', tablefmt='psql'),'\n')
    print('Refused\n',merged_r[column_name].unique(),'\n')
    print(tabulate(pd.DataFrame(merged_r.stb.freq([column_name])), headers='keys', tablefmt='psql'),'\n')
    print('Unused offer\n',merged_u[column_name].unique(),'\n')
    print(tabulate(pd.DataFrame(merged_u.stb.freq([column_name])), headers='keys', tablefmt='psql'),'\n')
# function for numerical variable univariate analysis

def merged_num_univariate_analysis(column_name,scale='linear') : 
    # boxplot for column vs target
    plt.figure(figsize=(8,6))
    ax = sns.boxplot(x='NAME_CONTRACT_STATUS', y = column_name, data = merged_data)
    title = column_name+' vs NAME_CONTRACT_STATUS'
    ax.set(title=title)
    if scale == 'log' :
        plt.yscale('log')
        ax.set(ylabel=column_name + '(Log Scale)')
    # summary statistic
    print('Approved\n', merged_a[column_name].describe(),'\n')
    print('Canceled\n',merged_c[column_name].describe(),'\n')
    print('Refused\n',merged_r[column_name].describe(),'\n')
    print('Unused offer\n',merged_u[column_name].describe(),'\n')
# function to calculate the proportion of applications in a category compared to total applications
def merged_cat_proportions(column_name) : 
    values = merged_data[column_name].unique()
    values=values.dropna()
    values = values.to_numpy()
    values.tolist()
    data_a = merged_a[column_name].value_counts().to_dict()
    data_c = merged_c[column_name].value_counts().to_dict()
    data_r = merged_r[column_name].value_counts().to_dict()
    data_u = merged_u[column_name].value_counts().to_dict()
    data = merged_data[column_name].value_counts().to_dict()

    for i in values : 
        if data_a[i] != np.nan and data_c[i] != np.nan and data_r[i] != np.nan and data_u[i] != np.nan and data[i] != np.nan:
            print('Proportion of '+ str(i) + ' Approved : ', round(data_a[i]*100/data[i],2),'\n')
            print('Proportion of '+ str(i) + ' Cancelled : ', round(data_c[i]*100/data[i],2),'\n')
            print('Proportion of '+ str(i) + ' Refused : ', round(data_r[i]*100/data[i],2),'\n')
            print('Proportion of '+ str(i) + ' Unused Offer : ', round(data_u[i]*100/data[i],2),'\n')
#AMT_ANNUITY_prev
merged_num_univariate_analysis('AMT_ANNUITY_prev',scale='log')
#AMT_APPLICATION
merged_num_univariate_analysis('AMT_APPLICATION',scale='log')
# AMT_CREDIT_prev
merged_num_univariate_analysis('AMT_CREDIT_prev',scale='log')
#AMT_GOODS_PRICE_prev
merged_num_univariate_analysis('AMT_GOODS_PRICE_prev',scale='log')
#CHANNEL_TYPE
merged_cat_univariate_analysis('CHANNEL_TYPE',figsize=(20,6))
plt.xticks(rotation=90)
#DAYS_DECISION
merged_num_univariate_analysis('DAYS_DECISION')
#NAME_CLIENT_TYPE
merged_cat_univariate_analysis('NAME_CLIENT_TYPE')
#NAME_CONTRACT_TYPE_prev
merged_cat_univariate_analysis('NAME_CONTRACT_TYPE_prev')
#NAME_PORTFOLIO
merged_cat_univariate_analysis('NAME_PORTFOLIO')
#NAME_YIELD_GROUP
merged_cat_univariate_analysis('NAME_YIELD_GROUP')
merged_data.columns
# AMT_ANNUITY_prev vs AMT_INCOME_CAT vs NAME_CONTRACT_STATUS
plt.figure(figsize=[20,12])
plt.title('Annuity Amount vs Income Category vs Loan Application Results')
sns.barplot(x='AMT_ANNUITY_prev', y = 'AMT_INCOME_CAT', hue='NAME_CONTRACT_STATUS', data=merged_data)

# AMT_CREDIT_prev - AMT_APPLICATION vs NAME_YIELD_GROUP vs NAME_CONTRACT_STATUS

merged_data['AMT_DIFF'] = merged_data['AMT_CREDIT_prev'] - merged_data['AMT_APPLICATION']
plt.figure(figsize=[8,8])
plt.title('Difference between Approved Loan and Applied Loan vs Interest Rate Category vs Loan Application Results')
sns.barplot(y='NAME_YIELD_GROUP', x = 'AMT_DIFF', hue='NAME_CONTRACT_STATUS', data=merged_data)
# DAYS_DECISION vs NAME_CONTRACT_TYPE vs NAME_CONTRACT_STATUS
plt.figure(figsize=[8,8])
plt.title('Processing Time vs Client Type vs Loan Application Results')
sns.barplot(y='NAME_CONTRACT_TYPE_prev', x = 'DAYS_DECISION', hue='NAME_CONTRACT_STATUS', data=merged_data)
# NAME_PORTFOLIO vs NAME_YIELD_GROUP vs NAME_CONTRACT_STATUS
#merged_data.groupby(['NAME_PORTFOLIO','NAME_YIELD_GROUP'])['NAME_CONTRACT_STATUS'].value_counts(normalize=True).plot.bar()
merged_data.groupby(['NAME_PORTFOLIO','NAME_YIELD_GROUP'])['NAME_CONTRACT_STATUS'].value_counts(normalize=True)\
.unstack()\
   .plot( 
    layout=(2,2),
    figsize=(8,6), kind='barh', stacked=True);

#AMT_CREDIT_prev & DAYS_EMPLOYED vs NAME_CONTRACT_STATUS

plt.figure(figsize=[10,8])
plt.xticks(rotation=45)
sns.barplot(y='DAYS_EMPLOYED', x = 'NAME_EDUCATION_TYPE', hue='NAME_CONTRACT_STATUS', data=merged_data)

plt.yscale('log')
merged_data.columns
#### NAME_EDUCATION_TYPE vs NAME_CLIENT_TYPE vs NAME_CONTRACT_STATUS
merged_data.groupby(['NAME_EDUCATION_TYPE','NAME_CLIENT_TYPE'])['NAME_CONTRACT_STATUS'].value_counts(normalize=True)\
.unstack()\
   .plot( 
    layout=(2,2),
    figsize=(8,6), kind='barh', stacked=True);
#CODE_GENDER vs CHANNEL_TYPE vs NAME_CONTRACT_STATUS

merged_data.groupby(['NAME_INCOME_TYPE','CODE_GENDER'])['NAME_CONTRACT_STATUS'].value_counts(normalize=True)\
.unstack()\
   .plot( 
    layout=(2,2),
    figsize=(8,6), kind='barh', stacked=True);
def correlation(dataframe) : 
    cor0=dataframe.corr()
    type(cor0)
    cor0.where(np.triu(np.ones(cor0.shape),k=1).astype(np.bool))
    cor0=cor0.unstack().reset_index()
    cor0.columns=['VAR1','VAR2','CORR']
    cor0.dropna(subset=['CORR'], inplace=True)
    cor0.CORR=round(cor0['CORR'],2)
    cor0.CORR=cor0.CORR.abs()
    cor0.sort_values(by=['CORR'],ascending=False)
    cor0=cor0[~(cor0['VAR1']==cor0['VAR2'])]
    return pd.DataFrame(cor0.sort_values(by=['CORR'],ascending=False))
# Correlation for Approved
# Absolute values are reported 
pd.set_option('precision', 2)
cor_0 = correlation(merged_a)
cor_0.style.background_gradient(cmap='GnBu').hide_index()
# Correlation for Cancelled
# Absolute values are reported 
pd.set_option('precision', 2)
cor_0 = correlation(merged_c)
cor_0.style.background_gradient(cmap='GnBu').hide_index()
# Correlation for Refused
# Absolute values are reported 
pd.set_option('precision', 2)
cor_0 = correlation(merged_r)
cor_0.style.background_gradient(cmap='GnBu').hide_index()
# Correlation for Unused Loans
# Absolute values are reported 
pd.set_option('precision', 2)
cor_0 = correlation(merged_u)
cor_0.style.background_gradient(cmap='GnBu').hide_index()
