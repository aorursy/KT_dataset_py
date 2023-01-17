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
from IPython.display import HTML
HTML('''<script>
code_show_err=false; 
function code_toggle_err() {
 if (code_show_err){
 $('div.output_stderr').hide();
 } else {
 $('div.output_stderr').show();
 }
 code_show_err = !code_show_err
} 
$( document ).ready(code_toggle_err);
</script>
''')
application_data=pd.read_csv('../input/bank-loans-dataset/application_data.csv')
#checking shape 
application_data.shape
application_data.info()
application_data.describe()
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

#Finding Percentage of Missing values in each and every column
missing_data = np.round(100*application_data.isnull().sum()/len(application_data),2)
missing_data = pd.DataFrame(data={'Column Name' :missing_data.index, 'Null Percentage' : missing_data.values})
missing_data.style.applymap(color_red,subset=['Null Percentage'])

#Finding the columns that have more than or equal to 50% null values and storing it to columns_to_drop.
columns_to_drop=application_data.columns[100*application_data.isnull().sum()/len(application_data)>=50]
columns_to_drop
#dropping the columns where the null values are >= 50%.
application_data.drop(labels=columns_to_drop,axis=1,inplace=True)
#verifying if the columns are dropped.
application_data.shape
#Finding columns with <= 13% missing columns

null_info = pd.DataFrame(100*application_data.isnull().sum()/len(application_data))
null_info.columns = ['Null Percentage']
null_info[(null_info['Null Percentage'] > 0) & (null_info['Null Percentage'] <=13)]

# Boxplot to check for outliers
application_data['AMT_ANNUITY'].plot.box()
plt.title('\n Box Plot of AMT_GOODS_PRICE')

# Calculating Quantiles
print('Quantile\tAMT_ANNUITY')
application_data['AMT_ANNUITY'].quantile([0.5,0.8,0.85,0.90,0.95,1])
print('Quantile\tAMT_GOODS_PRICE')
application_data['AMT_GOODS_PRICE'].plot.box()
plt.title('\n Box Plot of AMT_GOODS_PRICE')
application_data['AMT_GOODS_PRICE'].quantile([0.5,0.8,0.85,0.90,0.95,1])
print('Data type of NAME_TYPE_SUITE : ',application_data['NAME_TYPE_SUITE'].dtype,'\n\n')
print('Category\tNormalized Count\n\n',application_data['NAME_TYPE_SUITE'].value_counts(normalize=True))
data = application_data['NAME_TYPE_SUITE'].value_counts(normalize=True)
plt.bar(data.index,data.values)
# data.hist()
plt.xticks(rotation=90)
plt.ylabel('Normalized Value Counts')
plt.title('\nNAME_TYPE_SUITE');
application_data['CNT_FAM_MEMBERS'] = application_data['CNT_FAM_MEMBERS'].astype('category')

print('Data type of CNT_FAM_MEMBERS : ',application_data['CNT_FAM_MEMBERS'].dtype,'\n\n')

print('Fly Mems | Value Counts\n',application_data['CNT_FAM_MEMBERS'].value_counts(normalize=True))

(application_data['CNT_FAM_MEMBERS'].value_counts()).sort_index().plot(kind='bar')
plt.title('\nNo of Family Members vs Value Counts');
print('Data type of EXT_SOURCE_2 : ',application_data['EXT_SOURCE_2'].dtype,'\n\n')
application_data['EXT_SOURCE_2'].plot.box()
plt.title('\nEXT_SOURCE_2');
print('Quantile\tValue')
application_data['EXT_SOURCE_2'].quantile([0.5,0.8,0.85,0.90,0.95,1])
round(application_data['EXT_SOURCE_2'].mean(),2)
def cat_value_counts(column_name) : 
    print(tabulate(pd.DataFrame(application_data.stb.freq([column_name])), headers='keys', tablefmt='psql'))
    print(pd.DataFrame(application_data[column_name]).stb.missing(),'\n\n\n')

#Replacing XAN with np.nan in Gender column : 

application_data['CODE_GENDER'] = application_data['CODE_GENDER'].replace('XNA',np.nan)
cat_value_counts('CODE_GENDER')
# replacing Unknown in NAME_FAMILY_STATUS with np.nan 
application_data['NAME_FAMILY_STATUS'] = application_data['NAME_FAMILY_STATUS'].replace('Unknown',np.nan)
cat_value_counts('NAME_FAMILY_STATUS')
# replacing XNA values in ORGANIZATION_TYPE 
application_data['ORGANIZATION_TYPE'] = application_data['ORGANIZATION_TYPE'].replace('XNA',np.nan)
cat_value_counts('ORGANIZATION_TYPE')
pd.DataFrame(application_data.dtypes)
application_data[['DAYS_BIRTH','DAYS_EMPLOYED','DAYS_REGISTRATION','DAYS_ID_PUBLISH']].head()
columns_to_convert = ['DAYS_BIRTH','DAYS_EMPLOYED','DAYS_REGISTRATION','DAYS_ID_PUBLISH']
application_data[columns_to_convert] = application_data[columns_to_convert].abs()
application_data[columns_to_convert].head()
# checking columns with binary values
values_per_column = application_data.nunique().sort_values()
col_values_dtype = pd.DataFrame(index=values_per_column.index, data= {'Unique Values' : values_per_column.values, 'Data Type' : application_data.dtypes})
col_values_dtype
# converting to category data type 
convert_to_cat = col_values_dtype[col_values_dtype['Unique Values']<=8].index
application_data[convert_to_cat] = application_data[convert_to_cat].astype('category')
# check if the columns are converted
values_per_column = application_data.nunique().sort_values()
new_categories  = pd.DataFrame(index=values_per_column.index, data= {'Unique Values' : values_per_column.values, 'Data Type' : application_data.dtypes})
new_categories
# Adding a new column "AGE_YEARS" using 'DAYS_BIRTH' with age in years
def days_to_years(x) : 
    if x < 0 : 
        x = -1*x 
    return x//365
application_data['AGE_YEARS'] = application_data['DAYS_BIRTH'].apply(days_to_years)
application_data['AGE_YEARS'].describe()
# Box plots of the above numerical variables 
outlier_check_col = ["AMT_INCOME_TOTAL","AMT_CREDIT","AMT_ANNUITY","AMT_GOODS_PRICE","DAYS_BIRTH"]

fig,ax = plt.subplots(3,2)
fig.set_figheight(15)
fig.set_figwidth(15)
ax[0,0].set_yscale('log')
ax[0,0].set(ylabel ='Income in Log Scale')
application_data[outlier_check_col[0]].plot.box(ax=ax[0,0],);
application_data[outlier_check_col[1]].plot.box(ax=ax[0,1]);
application_data[outlier_check_col[2]].plot.box(ax=ax[1,0]);
application_data[outlier_check_col[3]].plot.box(ax=ax[1,1]);
ax[2,0].set(ylabel ='Age In Days')
application_data[outlier_check_col[4]].plot.box(ax=ax[2,0]); 
ax[2,1].axis('off')
print('Box Plots of "AMT_INCOME_TOTAL","AMT_CREDIT","AMT_ANNUITY","AMT_GOODS_PRICE","DAYS_BIRTH" \n')
    
## Calculating quantiles for the above columns
pd.options.display.float_format = '{:,.2f}'.format
for col in outlier_check_col : 
    print(col,'\n',application_data[col].quantile([0.5,0.8,0.85,0.90,0.95,1]),'\n\n')
    
# AMT_INCOME_TOTAL
min_income = int(application_data['AMT_INCOME_TOTAL'].min())
max_income = int(application_data['AMT_INCOME_TOTAL'].max())


bins = [0,25000,50000,75000,100000,125000,150000,175000,200000,225000,250000,275000,300000,325000,350000,375000,400000,425000,450000,475000,500000,10000000000]
intervals = ['0-25000', '25000-50000','50000-75000','75000-100000','100000-125000', '125000-150000', '150000-175000','175000-200000',
       '200000-225000','225000-250000','250000-275000','275000-300000','300000-325000','325000-350000','350000-375000',
       '375000-400000','400000-425000','425000-450000','450000-475000','475000-500000','500000 and above']

application_data['AMT_INCOME_CAT']=pd.cut(application_data['AMT_INCOME_TOTAL'],bins,labels=intervals)
print('Income Range\t Count')
print(application_data['AMT_INCOME_CAT'].value_counts())

income_cat = application_data['AMT_INCOME_CAT'].value_counts()
plt.hist(application_data['AMT_INCOME_CAT'])

plt.title('\n Income Range vs No of Applications')
plt.xticks(rotation=90);
#AMT_CREDIT
bins = [0,150000,200000,250000,300000,350000,400000,450000,500000,550000,600000,650000,700000,750000,800000,850000,900000,1000000000]
intervals = ['0-150000', '150000-200000','200000-250000', '250000-300000', '300000-350000', '350000-400000','400000-450000',
        '450000-500000','500000-550000','550000-600000','600000-650000','650000-700000','700000-750000','750000-800000',
        '800000-850000','850000-900000','900000 and above']

application_data['AMT_CREDIT_RANGE']=pd.cut(application_data['AMT_CREDIT'],bins=bins,labels=intervals)
application_data['AMT_CREDIT_RANGE'] = application_data['AMT_CREDIT_RANGE'].astype('category')
print('Credit Range\t Count')
credit_range = application_data['AMT_CREDIT_RANGE'].value_counts()
print(credit_range)
plt.hist(application_data['AMT_CREDIT_RANGE'])
plt.xticks(rotation=90)
# Target Variable - 1: Client with Payment difficulties, 0 : All other cases 
cat_value_counts('TARGET')
application_data0=application_data.loc[application_data["TARGET"]==0]
application_data1=application_data.loc[application_data["TARGET"]==1]
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
       'ORGANIZATION_TYPE', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
        'DAYS_LAST_PHONE_CHANGE' ,'AGE_YEARS', 'AMT_INCOME_CAT',
       'AMT_CREDIT_RANGE']
# LOAN DATA
# function for categorical variable univariate analysis
def cat_univariate_analysis(column_name,figsize=(10,5)) : 
    # print unique values
    print('TARGET 0\n', application_data0[column_name].unique(),'\n')
    print('TARGET 1\n',application_data1[column_name].unique(),'\n')
    
    # column vs target count plot
    plt.figure(figsize=figsize)
    ax = sns.countplot(x=column_name,hue='TARGET',data=application_data)
    title = column_name + ' vs Number of Applications'
    ax.set(title= title)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2,
                height + 10,
                format(height),
                ha="center")
    # Percentages 
    print('All Other Cases (TARGET : 0)')
    print(tabulate(pd.DataFrame(application_data0.stb.freq([column_name])), headers='keys', tablefmt='psql'),'\n')
    print('Clients with Payment Difficulties (TARGET : 1)')
    print(tabulate(pd.DataFrame(application_data1.stb.freq([column_name])), headers='keys', tablefmt='psql'),'\n')
# NAME_CONTRACT_TYPE
cat_univariate_analysis('NAME_CONTRACT_TYPE')
# function for numerical variable univariate analysis

def num_univariate_analysis(column_name,scale='linear') : 
    # boxplot for column vs target
    plt.figure(figsize=(8,6))
    ax = sns.boxplot(x='TARGET', y = column_name, data = application_data)
    title = column_name+' vs Target'
    ax.set(title=title)
    if scale == 'log' :
        plt.yscale('log')
        ax.set(ylabel=column_name + '(Log Scale)')
    # summary statistic
    print('All Other Cases (TARGET : 0)')
    print(application_data0[column_name].describe(),'\n')
    print('Clients with Payment Difficulties (TARGET : 1)')
    print(application_data1[column_name].describe())

# AMT_CREDIT 
num_univariate_analysis('AMT_CREDIT')
#AMT_ANNUITY
num_univariate_analysis('AMT_ANNUITY','log')
#AMT_GOODS_PRICE
num_univariate_analysis('AMT_GOODS_PRICE')
#REGION_RATING_CLIENT_W_CITY
cat_univariate_analysis('REGION_RATING_CLIENT_W_CITY')
#EXT_SOURCE_2
num_univariate_analysis('EXT_SOURCE_2')
#EXT_SOURCE_3
num_univariate_analysis('EXT_SOURCE_3')
#CODE_GENDER
cat_univariate_analysis('CODE_GENDER')
print(application_data['CODE_GENDER'].value_counts(),'\n')
print('Proportion of Females (Target 0) :', round(100*188278/202448,2))
print('Proportion of Females (Target 1) :', round(100*14170/202448,2))
print('Proportion of Males (Target 0) :', round(100*94404/105059,2))
print('Proportion of Males (Target 1) :', round(100*10655/105059,2))
def cat_proportions(column_name) : 
    values = application_data[column_name].unique()
    values = values.to_numpy()
    values.tolist()
    data0 = application_data0[column_name].value_counts().to_dict()
    data1 = application_data1[column_name].value_counts().to_dict()
    data = application_data[column_name].value_counts().to_dict()
    

    for i in values : 
        if i in data0 and i in data1 and i in data : 
            print('Proportion of '+ str(i) + ' in Target 0 : ', round(data0[i]*100/data[i],2))
            print('Proportion of '+ str(i) + ' in Target 1 : ', round(data1[i]*100/data[i],2),'\n' )
# FLAG_OWN_CAR
cat_univariate_analysis('FLAG_OWN_CAR')
cat_proportions('FLAG_OWN_CAR')
application_data['FLAG_OWN_CAR'].value_counts(normalize=True)
# FLAG_OWN_REALTY
cat_univariate_analysis('FLAG_OWN_REALTY')
cat_proportions('FLAG_OWN_REALTY')
#NAME_EDUCATION_TYPE

cat_univariate_analysis('NAME_EDUCATION_TYPE')
plt.xticks(rotation=90)
cat_proportions('NAME_EDUCATION_TYPE')
#NAME_FAMILY_STATUS

cat_univariate_analysis('NAME_FAMILY_STATUS')
plt.xticks(rotation=90)
cat_proportions('NAME_FAMILY_STATUS')
#NAME_HOUSING_TYPE
cat_univariate_analysis('NAME_HOUSING_TYPE',figsize=(10,5))
plt.xticks(rotation=90)
cat_proportions('NAME_HOUSING_TYPE')
#AMT_INCOME_CAT
cat_univariate_analysis('AMT_INCOME_CAT',figsize=(20,5))
plt.xticks(rotation=90)
cat_proportions('AMT_INCOME_CAT')
# We will calculate the pecentage of "Clients with Payment difficulties" for every income category.
t0=application_data0['AMT_INCOME_CAT'].value_counts()
t1=application_data1['AMT_INCOME_CAT'].value_counts()
prop = 100*t1/(t1+t0)
print(tabulate(pd.DataFrame(prop), headers=['AMT_INCOME_CAT','PERCENTAGE'], tablefmt='psql'))
#CNT_CHILDREN
#Lets convert this into a categorical variable
application_data['CNT_CHILDREN']=application_data['CNT_CHILDREN'].astype('category')
application_data0['CNT_CHILDREN']=application_data0['CNT_CHILDREN'].astype('category')
application_data1['CNT_CHILDREN']=application_data1['CNT_CHILDREN'].astype('category')
#CNT_CHILDREN
cat_univariate_analysis('CNT_CHILDREN',figsize=(20,5))
column_name='CNT_CHILDREN'
cat_proportions('CNT_CHILDREN')
# values = application_data[column_name].unique()
# values=values.dropna()
# values = values.to_numpy()
# values.tolist()
# data0 = application_data0[column_name].value_counts().to_dict()
# data1 = application_data1[column_name].value_counts().to_dict()
# data = application_data[column_name].value_counts().to_dict()

# for i in values[:7]: 
#     print('Proportion of '+ str(i) + ' in Target 0 : ', round(data0[i]*100/data[i],2))
#     print('Proportion of '+ str(i) + ' in Target 1 : ', round(data1[i]*100/data[i],2),'\n' )
# Calculating the pecentage of "Clients with Payment difficulties" for every income category.
t0=application_data0['CNT_CHILDREN'].value_counts()
t1=application_data1['CNT_CHILDREN'].value_counts()
prop = 100*t1/(t1+t0)

print(tabulate(pd.DataFrame(prop), headers=['No of Children','Percentage'], tablefmt='psql'))
#CNT_FAM_MEMBERS
cat_univariate_analysis('CNT_FAM_MEMBERS',figsize=(20,5))
cat_proportions('CNT_FAM_MEMBERS')
#DAYS_EMPLOYED.
num_univariate_analysis('DAYS_EMPLOYED','log')
#NAME_INCOME_TYPE
cat_univariate_analysis('NAME_INCOME_TYPE',figsize=(15,5))
plt.xticks(rotation=90)
cat_proportions('NAME_INCOME_TYPE')
#DAYS_LAST_PHONE_CHANGE
application_data['DAYS_LAST_PHONE_CHANGE'] = np.abs(application_data['DAYS_LAST_PHONE_CHANGE'])
application_data0['DAYS_LAST_PHONE_CHANGE'] = np.abs(application_data0['DAYS_LAST_PHONE_CHANGE'])
application_data1['DAYS_LAST_PHONE_CHANGE'] = np.abs(application_data1['DAYS_LAST_PHONE_CHANGE'])
num_univariate_analysis('DAYS_LAST_PHONE_CHANGE')
#FLAG_CONT_MOBILE
cat_univariate_analysis('FLAG_CONT_MOBILE',figsize=(15,5))
cat_proportions('FLAG_CONT_MOBILE')
#FLAG_EMAIL.
cat_univariate_analysis('FLAG_EMAIL',figsize=(15,5))
cat_proportions('FLAG_EMAIL')
#FLAG_MOBIL
cat_univariate_analysis('FLAG_MOBIL',figsize=(15,5))
cat_proportions('FLAG_MOBIL')
#LIVE_CITY_NOT_WORK_CITY
cat_univariate_analysis('LIVE_CITY_NOT_WORK_CITY',figsize=(15,5))
cat_proportions('LIVE_CITY_NOT_WORK_CITY')
#REG_CITY_NOT_LIVE_CITY.
cat_univariate_analysis('REG_CITY_NOT_LIVE_CITY',figsize=(15,5))
cat_proportions('REG_CITY_NOT_LIVE_CITY')
#REG_CITY_NOT_WORK_CITY.
cat_univariate_analysis('REG_CITY_NOT_WORK_CITY',figsize=(15,5))
cat_proportions('REG_CITY_NOT_WORK_CITY')
#AMT_ANNUITY, AMT_INCOME_TOTAL vs TARGET
column_names = ['AMT_INCOME_CAT','AMT_ANNUITY']
plt.figure(figsize=(16,8))
sns.barplot(x = column_names[0],y = column_names[1],hue='TARGET',data = application_data)
plt.title(column_names[0] + ' vs '+ column_names[1] + ' vs Target')
plt.xticks(rotation=90);
plt.figure(figsize=(16,8))
# sns.catplot(x = column_names[0],y =column_names[1],hue='TARGET',data = application_data, kind='violin',height=8,aspect=4);
sns.violinplot(x = 'AMT_INCOME_CAT',y='AMT_ANNUITY',hue='TARGET',data = application_data,split=True, inner="quartile")
plt.title(column_names[0] + ' vs '+ column_names[1] + ' vs Target')
plt.xticks(rotation=90);
#AMT_CREDIT, AMT_GOODS_PRICE vs TARGET
column_names = ['AMT_CREDIT', 'AMT_GOODS_PRICE']
fig,ax = plt.subplots(1,2)
fig.set_figheight(8)
fig.set_figwidth(15)

ax[0].set(title = column_names[0] + ' vs '+ column_names[1] + ' vs Target');
sns.scatterplot(x=column_names[0],y=column_names[1],hue='TARGET',data=application_data, alpha=0.8,ax=ax[0])
plt.xticks(rotation=90);

plt.title(column_names[0] + ' vs '+ column_names[1] + ' vs Target for AMT_GOODS_PRICE < 1000000');
sns.scatterplot(x=column_names[0],y=column_names[1],hue='TARGET',data=application_data[application_data['AMT_GOODS_PRICE'] <=1000000], alpha=0.8,ax=ax[1])
plt.xticks(rotation=90);

#NAME_CONTRACT_TYPE vs REGION_RATING_CLIENT_W_CITY vs TARGET

column_names = ['NAME_CONTRACT_TYPE','REGION_RATING_CLIENT_W_CITY']
application_data.groupby(column_names)['TARGET'].value_counts(normalize=True)


sns.catplot(x='REGION_RATING_CLIENT_W_CITY', hue='TARGET', col="NAME_CONTRACT_TYPE", kind="count", data=application_data);
# EXT_SOURCE_2 vs EXT_SOURCE_3 vs TARGET
creditScores = ['EXT_SOURCE_2','EXT_SOURCE_3']
plt.figure(figsize=[8,8])
sns.scatterplot(x=creditScores[0],y = creditScores[1], hue='TARGET',data = application_data, alpha=0.5);
plt.title(creditScores[0] + ' vs '+ creditScores[1]+ ' vs '+ 'TARGET');
#FLAG_OWN_CAR vs FLAG_OWN_REALTY vs TARGET
application_data.groupby(['FLAG_OWN_CAR','FLAG_OWN_REALTY'])['TARGET'].value_counts(normalize=True)
# classification over both categories
pd.DataFrame(application_data.groupby(['CODE_GENDER','NAME_FAMILY_STATUS'])['TARGET'].value_counts(normalize=True))
application_data.groupby(['CODE_GENDER','NAME_FAMILY_STATUS'])['TARGET'].value_counts(normalize=True)\
.unstack()\
   .plot( 
    layout=(2,2),
    figsize=(8,6), kind='barh', stacked=True);


# NAME_HOUSING_TYPE vs NAME_INCOME_TYPE vs TARGET
income_housing = pd.DataFrame(application_data.groupby(['NAME_HOUSING_TYPE','NAME_INCOME_TYPE'])['TARGET'].value_counts(normalize=True))
income_housing.columns = ['Normalized Count']
income_housing 
application_data.groupby(['NAME_HOUSING_TYPE','NAME_INCOME_TYPE'])['TARGET'].value_counts(normalize=True)\
.unstack()\
   .plot( 
    layout=(2,2),
    figsize=(8,6), kind='barh', stacked=True);
income_housing[np.in1d(income_housing.index.get_level_values(2), 1)].sort_values(by='Normalized Count', ascending=True)
#FLAG_EMAIL vs FLAG_MOBIL vs TARGET
pd.DataFrame(application_data.groupby(['FLAG_EMAIL','FLAG_MOBIL'])['TARGET'].value_counts(normalize=True))\
.unstack()\
   .plot( 
    layout=(2,2),
    figsize=(8,6), kind='barh', stacked=True);


pd.DataFrame(application_data.groupby(['FLAG_EMAIL','FLAG_MOBIL'])['TARGET'].value_counts())
# DAYS_LAST_PHONE_CHANGE vs FLAG_CONT_MOBILE vs TARGET

sns.barplot(x = 'FLAG_CONT_MOBILE', y= 'DAYS_LAST_PHONE_CHANGE', hue='TARGET',data = application_data)
plt.title("FLAG_CONT_MOBILE vs DAYS_LAST_PHONE_CHANGE vs TARGET")
sns.violinplot(x = 'FLAG_CONT_MOBILE', y= 'DAYS_LAST_PHONE_CHANGE', hue='TARGET', split=True, data = application_data,inner="quartile", height=5, aspect=1)
plt.title("FLAG_CONT_MOBILE vs DAYS_LAST_PHONE_CHANGE vs TARGET")
application_data['DAYS_EMPLOYED'].plot.hist()
#DAYS_EMPLOYED & NAME_EDUCATION_TYPE vs TARGET
fig,ax = plt.subplots(1,2)
fig.set_figheight(5)
fig.set_figwidth(15)

fig.suptitle(t="DAYS_EMPLOYED & NAME_EDUCATION_TYPE vs TARGET");
ax[0].tick_params(axis='x',rotation=90)
sns.barplot(x = 'NAME_EDUCATION_TYPE', y= 'DAYS_EMPLOYED', hue='TARGET',data = application_data,ax=ax[0])

ax[1].tick_params(axis='x',rotation=90)
# ax[1].set_yscale('log')
sns.violinplot(x = 'NAME_EDUCATION_TYPE', y= 'DAYS_EMPLOYED', hue='TARGET', split=True, data = application_data,inner="quartile", height=5, aspect=1, ax=ax[1])
#sns.swarmplot(x = 'NAME_EDUCATION_TYPE', y= 'DAYS_EMPLOYED', hue='TARGET', data = application_data, ax=ax[1])

#CNT_CHILDREN & CNT_FAM_MEMBERS vs TARGET
subset = application_data[['CNT_CHILDREN','CNT_FAM_MEMBERS','TARGET']] 
subset = subset.dropna().astype('int')
childrenvsFamily = pd.pivot_table(index='CNT_CHILDREN', columns = 'CNT_FAM_MEMBERS',aggfunc=np.mean,data=subset)
childrenvsFamily
# NAME_EDUCATION_TYPE vs AMT_INCOME_TOTAL vs TARGET 

plt.figure(figsize=(10,8))
sns.violinplot(x = 'NAME_EDUCATION_TYPE', y= 'AMT_INCOME_TOTAL', hue='TARGET', split=True, data = application_data,inner="quartile", height=5, aspect=3)
plt.yscale('log')
plt.xticks(rotation=90);

#'NAME_INCOME_TYPE vs NAME_EDUCATION_TYPE'
subset = application_data[['NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','TARGET']] 
subset = subset.dropna()
subset['TARGET'] = subset['TARGET'].astype('int')
# pivot table for percentage of default for education type vs income type
incomeTypevsEdu = pd.pivot_table(index='NAME_EDUCATION_TYPE', columns = 'NAME_INCOME_TYPE',aggfunc=[np.mean],data=subset)
incomeTypevsEdu
# pivot table of the count of defaults for education type vs income type
pd.pivot_table(index='NAME_EDUCATION_TYPE', columns = 'NAME_INCOME_TYPE',aggfunc='count',data=subset)
sns.violinplot(x = 'NAME_INCOME_TYPE', y= 'AMT_INCOME_TOTAL', hue='TARGET', split=True, data = application_data,inner="quartile", height=5, aspect=3)
plt.yscale('log')
plt.xticks(rotation=90);
# function to correlate variables
def correlation(dataframe) : 
    cor0=dataframe[columnsForAnalysis].corr()
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
# Correlation for Target : 0 
# Absolute values are reported 
pd.set_option('precision', 2)
cor_0 = correlation(application_data0)
cor_0.style.background_gradient(cmap='GnBu').hide_index()
# Correlation for Target : 1 
# Absolute values are reported 
pd.set_option('precision', 2)
cor_1 = correlation(application_data1)
cor_1.style.background_gradient(cmap='GnBu').hide_index()