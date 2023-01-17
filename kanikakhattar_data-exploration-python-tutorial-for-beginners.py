import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt 

import warnings

warnings.filterwarnings('ignore')



%matplotlib inline

pd.set_option('display.max_rows', 500)

#df = pd.read_csv("application_data.csv")

df = pd.read_csv("/kaggle/input/application_data.csv")

#Sanity checks on the data

df.head()

df.info()

df.shape

#df.dtypes
# sum it up to check how many rows have all missing values

df.isnull().all(axis=1).sum()
# % of the missing values (column-wise)

col_missing_perc = round(100*(df.isnull().sum()/len(df.index)), 2) 
#getting cols with more than 20% missing and dropping them

col_missing_perc_greater_20 = []

for i in range(0,len(col_missing_perc)):

    if col_missing_perc[i]>20:

        col_missing_perc_greater_20.append(col_missing_perc.index[i])

    

#dropping cols with more than 20% missing

df.drop(col_missing_perc_greater_20, axis = 1,inplace=True)
#remaining columns

df.shape
#subsetting the data

df=df[['SK_ID_CURR',

'TARGET',

'NAME_CONTRACT_TYPE',

'CODE_GENDER',

'CNT_CHILDREN',

'AMT_INCOME_TOTAL',

'AMT_CREDIT',

'AMT_ANNUITY',

'AMT_GOODS_PRICE',

'NAME_INCOME_TYPE',

'NAME_EDUCATION_TYPE',

'NAME_FAMILY_STATUS',

'NAME_HOUSING_TYPE',

'DAYS_BIRTH',

'DAYS_EMPLOYED',

'CNT_FAM_MEMBERS',

'ORGANIZATION_TYPE',

'AMT_REQ_CREDIT_BUREAU_HOUR',

'AMT_REQ_CREDIT_BUREAU_DAY',

'AMT_REQ_CREDIT_BUREAU_WEEK',

'AMT_REQ_CREDIT_BUREAU_MON',

'AMT_REQ_CREDIT_BUREAU_QRT',

'AMT_REQ_CREDIT_BUREAU_YEAR',

'EXT_SOURCE_2'

]]

##final list of columns for analysis

df.columns
#checking missing % in remaining columns

round(100*(df.isnull().sum()/len(df.index)), 2)


#1.Handling missing values -  Categorical



df['ORGANIZATION_TYPE']=np.where(df['ORGANIZATION_TYPE'].isnull(),df['ORGANIZATION_TYPE'].mode(),df['ORGANIZATION_TYPE']) 

df['CODE_GENDER']=np.where(df['CODE_GENDER']=='XNA',df['CODE_GENDER'].mode(),df['CODE_GENDER'])







df.loc[np.isnan(df['AMT_REQ_CREDIT_BUREAU_HOUR']), ['AMT_REQ_CREDIT_BUREAU_HOUR']] = df['AMT_REQ_CREDIT_BUREAU_HOUR'].median()

df.loc[np.isnan(df['AMT_REQ_CREDIT_BUREAU_DAY']),['AMT_REQ_CREDIT_BUREAU_DAY']]=df['AMT_REQ_CREDIT_BUREAU_DAY'].median()

df.loc[np.isnan(df['AMT_REQ_CREDIT_BUREAU_WEEK']),['AMT_REQ_CREDIT_BUREAU_WEEK']]=df['AMT_REQ_CREDIT_BUREAU_WEEK'].median()

df.loc[np.isnan(df['AMT_REQ_CREDIT_BUREAU_MON']),['AMT_REQ_CREDIT_BUREAU_MON']]=df['AMT_REQ_CREDIT_BUREAU_MON'].median()

df.loc[np.isnan(df['AMT_REQ_CREDIT_BUREAU_QRT']),['AMT_REQ_CREDIT_BUREAU_QRT']]=df['AMT_REQ_CREDIT_BUREAU_QRT'].median()

df.loc[np.isnan(df['AMT_REQ_CREDIT_BUREAU_YEAR']),['AMT_REQ_CREDIT_BUREAU_YEAR']]=df['AMT_REQ_CREDIT_BUREAU_YEAR'].median()



#1.Handling missing values -  Numerical

df.loc[np.isnan(df['CNT_FAM_MEMBERS']),['CNT_FAM_MEMBERS']]=df['CNT_FAM_MEMBERS'].median()

df.loc[np.isnan(df['AMT_ANNUITY']),['AMT_ANNUITY']]=round(df['AMT_ANNUITY'].median(),1)
#checking missing % in remaining columns

round(100*(df.isnull().sum()/len(df.index)), 2)
df = df.dropna(axis=0, subset=['EXT_SOURCE_2'])
round(100*(df.isnull().sum()/len(df.index)), 2)
df.shape
# Identifying and treating Outliers on columns - AMT_ANNUITY,AMT_GOODS_PRICE,AMT_CREDIT





df_outliers=df[['AMT_ANNUITY','AMT_GOODS_PRICE','AMT_CREDIT']]

#df_outliers.shape (306574, 3)--before outlier removal

Q1=df_outliers.quantile(0.25)

Q3=df_outliers.quantile(0.75)



IQR=Q3-Q1

print(IQR)



#in case you decide to remove outliers, follow the below command.

df_out_final=df_outliers[~((df_outliers < (Q1-1.5*IQR)) | (df_outliers > ((Q3 + 1.5*IQR)))).any(axis=1)]





# The mean value will be used further to impute missing value in the respective columns

df_out_final['AMT_GOODS_PRICE'].mean() 

len(df_out_final.index)/len(df.index)
#imputing missing value of AMT_GOODS_PRICE with mean of data after removing the outlier

df.loc[np.isnan(df['AMT_GOODS_PRICE']),['AMT_GOODS_PRICE']]=round(df_out_final['AMT_GOODS_PRICE'].mean(),1)
## verification of the fixes

round(100*(df.isnull().sum()/len(df.index)), 2)
#changing datatype of the columns

dt_dict={'AMT_REQ_CREDIT_BUREAU_HOUR':int,

        'CNT_FAM_MEMBERS':int,

        'AMT_REQ_CREDIT_BUREAU_WEEK':int,

        'AMT_REQ_CREDIT_BUREAU_MON':int,

        'AMT_REQ_CREDIT_BUREAU_DAY':int,

        'AMT_REQ_CREDIT_BUREAU_QRT':int,

        'AMT_REQ_CREDIT_BUREAU_YEAR':int}



df=df.astype(dt_dict)





# checking the datatypes

df.info()

   
#removing unnecessary spaces in column names

df.columns=[df.columns[i].strip() for i in range(len(df.columns))]



#renaming columns

df.rename(columns={"EXT_SOURCE_2": "CREDIT_RATINGS"},inplace=True)
#Categorising customers into following

#Youth (<18)

#Young Adult (18 to 35)

#Adult (36 to 55)

#Senior (56 and up)



df['AGE'] = abs(df['DAYS_BIRTH'])

df['AGE'] = round(df['AGE']/365,1)

df['AGE']



df['AGE'].describe()

def age_group(y):

    if y>=56:

        return "Senior"

    elif y>=36 and y<56:

        return "Adult"

    elif y>=18 and y<36:

        return "Young Adult"

    else:

        return "Youth"

    

df['AGE_GROUP'] = df['AGE'].apply(lambda x: age_group(x))



sns.countplot(x='AGE_GROUP',hue='TARGET',data=df)
df['CREDIT_RATINGS'].describe()
sns.boxplot(y=df['CREDIT_RATINGS'])
credit_category_quantile = list(df['CREDIT_RATINGS'].quantile([0.20,0.5,0.80,1]))

credit_category_quantile
def credit_group(x):

    if x>=credit_category_quantile[2]:

        return "C1"

    elif x>=credit_category_quantile[1]:

        return "C2"

    elif x>=credit_category_quantile[0]:

        return "C3"

    else:

        return "C4"

df["CREDIT_CATEGORY"] = df['CREDIT_RATINGS'].apply(lambda x: credit_group(x))



sns.countplot(x='CREDIT_CATEGORY',hue='TARGET',data=df)
df['TARGET'].value_counts(normalize=True)
#checking for unique values per column to see what all columns can be categorised

df.nunique().sort_values()
df0 = df[df['TARGET']==0]

df1 = df[df['TARGET']==1]

# What are average values of numerical features

df.pivot_table(columns = 'TARGET', aggfunc = 'median')
df['NAME_INCOME_TYPE'].unique()
plt.figure(figsize=(20,9))

sns.countplot(x='NAME_INCOME_TYPE',hue='TARGET',data=df)

incomeCategories0 = pd.DataFrame(df0['NAME_INCOME_TYPE'].value_counts().rename("Count_0").reset_index())

incomeCategories0_perct = pd.DataFrame(df0['NAME_INCOME_TYPE'].value_counts(normalize=True).rename("Perct_0").reset_index())

incomeCategories0.rename(columns={"index":"NAME_INCOME_TYPE"})

incomeCategories0_perct.rename(columns={"index":"NAME_INCOME_TYPE"})



#Merging data to get the overall view of the variable "NAME_INCOME_TYPE"

incomeCategories0 = pd.merge(incomeCategories0,incomeCategories0_perct,how="inner").rename(columns={"index":"NAME_INCOME_TYPE"})

incomeCategories0



incomeCategories1 = pd.DataFrame(df1['NAME_INCOME_TYPE'].value_counts().rename("Count_1").reset_index())

incomeCategories1_perct = pd.DataFrame(df1['NAME_INCOME_TYPE'].value_counts(normalize=True).rename("Perct_1").reset_index())

incomeCategories1.rename(columns={"index":"NAME_INCOME_TYPE"})

incomeCategories1_perct.rename(columns={"index":"NAME_INCOME_TYPE"})



#Merging data to get the overall view of the variable "NAME_INCOME_TYPE"

incomeCategories1 = pd.merge(incomeCategories1,incomeCategories1_perct,how="inner").rename(columns={"index":"NAME_INCOME_TYPE"})

incomeCategories1



incomeCategories = pd.merge(incomeCategories0,incomeCategories1,how="inner").rename(columns={"index":"NAME_INCOME_TYPE"})



def income_percentage_contri_0(count_0, count_1):

    return 100*(count_0/(count_0+count_1))



def income_percentage_contri_1(count_0, count_1):

    return 100*(count_1/(count_0+count_1))



incomeCategories['percentage_contri_0'] = incomeCategories[['Count_0','Count_1']].apply(lambda x: income_percentage_contri_0(*x), axis=1)

incomeCategories['percentage_contri_1'] = incomeCategories[['Count_0','Count_1']].apply(lambda x: income_percentage_contri_1(*x), axis=1)

incomeCategories.set_index("NAME_INCOME_TYPE",inplace=True)

incomeCategories
fig = plt.figure(figsize=(20,5))

plt.subplot(1,2,1)

plt.title("Target = 1")

plt.ylabel('Percentage contribution to "defaulters"')

plt.plot(incomeCategories['percentage_contri_1'])

#ax1.set_xticklabels(labels = ax1.get_xticklabels(),rotation=30)

plt.rcParams.update(plt.rcParamsDefault)

incomeCategories = incomeCategories.sort_values(by='percentage_contri_1')

incomeCategories[['percentage_contri_1', 'percentage_contri_0']].plot(kind='bar', stacked=True)
df.CODE_GENDER.unique()
sns.countplot(x='CODE_GENDER',hue='TARGET',data=df)
genderCategories0 = pd.DataFrame(df0['CODE_GENDER'].value_counts().rename("Count_0").reset_index())

genderCategories0_perct = pd.DataFrame(df0['CODE_GENDER'].value_counts(normalize=True).rename("Perct_0").reset_index())

genderCategories0.rename(columns={"index":"CODE_GENDER"})

genderCategories0_perct.rename(columns={"index":"CODE_GENDER"})



#Merging data to get the overall view of the variable "NAME_INCOME_TYPE"

genderCategories0 = pd.merge(genderCategories0,genderCategories0_perct,how="inner").rename(columns={"index":"CODE_GENDER"})

genderCategories0



genderCategories1 = pd.DataFrame(df1['CODE_GENDER'].value_counts().rename("Count_1").reset_index())

genderCategories1_perct = pd.DataFrame(df1['CODE_GENDER'].value_counts(normalize=True).rename("Perct_1").reset_index())

genderCategories1.rename(columns={"index":"CODE_GENDER"})

genderCategories1_perct.rename(columns={"index":"CODE_GENDER"})



#Merging data to get the overall view of the variable "NAME_INCOME_TYPE"

genderCategories1 = pd.merge(genderCategories1,genderCategories1_perct,how="inner").rename(columns={"index":"CODE_GENDER"})

genderCategories1



genderCategories = pd.merge(genderCategories0,genderCategories1,how="inner").rename(columns={"index":"CODE_GENDER"})



def gender_percentage_contri_0(count_0, count_1):

    return 100*(count_0/(count_0+count_1))



def gender_percentage_contri_1(count_0, count_1):

    return 100*(count_1/(count_0+count_1))



genderCategories['percentage_contri_0'] = genderCategories[['Count_0','Count_1']].apply(lambda x: gender_percentage_contri_0(*x), axis=1)

genderCategories['percentage_contri_1'] = genderCategories[['Count_0','Count_1']].apply(lambda x: gender_percentage_contri_1(*x), axis=1)

genderCategories.set_index("CODE_GENDER",inplace=True)

genderCategories
plt.rcParams.update(plt.rcParamsDefault)

genderCategories = genderCategories.sort_values(by='percentage_contri_1')

genderCategories[['percentage_contri_1', 'percentage_contri_0']].plot(kind='bar', stacked=True)
df1['CREDIT_RATINGS'].describe()
df0['CREDIT_RATINGS'].describe()
sns.barplot(x="TARGET",y="CREDIT_RATINGS",data=df)
sns.boxplot(x="TARGET", y="CREDIT_RATINGS", data=df,palette='rainbow')
target =[0,1]

for i in target:

    subset = df[df['TARGET'] == i]

    sns.distplot(subset['CREDIT_RATINGS'],hist=False,kde=True,kde_kws ={'shade':True},label=i)
plt.figure(figsize=(20,9))

plt.subplot(1,2,1)

df0['CREDIT_RATINGS'].hist(bins = 50)

plt.subplot(1,2,2)

df1['CREDIT_RATINGS'].hist(bins = 50)
df['CREDIT_CATEGORY'].value_counts()
creditCategories0 = pd.DataFrame(df0['CREDIT_CATEGORY'].value_counts().rename("Count_0").reset_index())

creditCategories0_perct = pd.DataFrame(df0['CREDIT_CATEGORY'].value_counts(normalize=True).rename("Perct_0").reset_index())

creditCategories0.rename(columns={"index":"CREDIT_CATEGORY"})

creditCategories0_perct.rename(columns={"index":"CREDIT_CATEGORY"})



#Merging data to get the overall view of the variable "NAME_INCOME_TYPE"

creditCategories0 = pd.merge(creditCategories0,creditCategories0_perct,how="inner").rename(columns={"index":"CREDIT_CATEGORY"})

creditCategories0



creditCategories1 = pd.DataFrame(df1['CREDIT_CATEGORY'].value_counts().rename("Count_1").reset_index())

creditCategories1_perct = pd.DataFrame(df1['CREDIT_CATEGORY'].value_counts(normalize=True).rename("Perct_1").reset_index())

creditCategories1.rename(columns={"index":"CREDIT_CATEGORY"})

creditCategories1_perct.rename(columns={"index":"CREDIT_CATEGORY"})



#Merging data to get the overall view of the variable "NAME_INCOME_TYPE"

creditCategories1 = pd.merge(creditCategories1,creditCategories1_perct,how="inner").rename(columns={"index":"CREDIT_CATEGORY"})

creditCategories1



creditCategories = pd.merge(creditCategories0,creditCategories1,how="inner").rename(columns={"index":"CREDIT_CATEGORY"})



def credit_percentage_contri_0(count_0, count_1):

    return 100*(count_0/(count_0+count_1))



def credit_percentage_contri_1(count_0, count_1):

    return 100*(count_1/(count_0+count_1))



creditCategories['percentage_contri_0'] = creditCategories[['Count_0','Count_1']].apply(lambda x: credit_percentage_contri_0(*x), axis=1)

creditCategories['percentage_contri_1'] = creditCategories[['Count_0','Count_1']].apply(lambda x: credit_percentage_contri_1(*x), axis=1)

creditCategories.set_index("CREDIT_CATEGORY",inplace=True)

creditCategories
plt.plot(creditCategories['percentage_contri_1'].sort_values())
creditCategories[['percentage_contri_1', 'percentage_contri_0']].plot(kind='bar', stacked=True)
pt = df.pivot_table(columns='NAME_INCOME_TYPE',index='CREDIT_CATEGORY',values='TARGET',aggfunc='sum',fill_value = 0)

#pt.reset_index()



pt
pt['Row_Total'] = pt['Businessman'] + pt['Commercial associate'] + pt['Maternity leave'] + pt['Pensioner']+pt['State servant'] +pt['Student']+pt['Unemployed']+pt['Working']
Column_Total = []

for c in pt.columns:

    Column_Total.append(pt[c].sum())

Column_Total

pt.loc['Column_Total'] = Column_Total

pt
for i in pt.index:

    pt.loc[i,'Total%'] = 100*(pt.loc[i,'Row_Total']/pt.loc['Column_Total','Row_Total'])



for j in df.NAME_INCOME_TYPE.unique():

    for i in pt.index:

        pt.loc[i,j+'%'] = 100*(pt.loc[i,j]/pt.loc['Column_Total',j])

pt
credit_income_type = pt.iloc[0:-1][['Working%','State servant%','Commercial associate%','Pensioner%','Unemployed%']]

credit_income_type

credit_income_type.T.plot.bar(stacked = 'TRUE')
df1_corr=df[df['TARGET']==1]

df0_corr=df[df['TARGET']==0]



df1_corr=df1_corr[[

'AMT_INCOME_TOTAL',

'AMT_CREDIT',

'AMT_ANNUITY',

'AMT_GOODS_PRICE',

'AGE',

'DAYS_EMPLOYED']]



df0_corr=df0_corr[[

'AMT_INCOME_TOTAL',

'AMT_CREDIT',

'AMT_ANNUITY',

'AMT_GOODS_PRICE',

'AGE',

'DAYS_EMPLOYED']]



df1_corr_matrix=df1_corr.corr()

df0_corr_matrix=df1_corr.corr()

df1_corr_matrix


#narrowing down the data and considering less than the upper quantile AMT_INCOME_TOTAL

df1_corr['AMT_INCOME_TOTAL'] = df1_corr[df1_corr['AMT_INCOME_TOTAL']<df1_corr['AMT_INCOME_TOTAL'].quantile(.85)]['AMT_INCOME_TOTAL']

#df1_corr['AMT_ANNUITY'] = df1_corr[df1_corr['AMT_GOODS_PRICE']<df1_corr['AMT_GOODS_PRICE'].quantile(.85)]['AMT_GOODS_PRICE']



fig, ax = plt.subplots(figsize=(10,10)) 

sns.scatterplot(x='AMT_INCOME_TOTAL', y='AMT_ANNUITY',data=df1_corr)
df1_corr.plot.hexbin(x='AMT_INCOME_TOTAL', y='AMT_ANNUITY', gridsize=30)
#narrowing down the data and considering less than the upper quantile AMT_INCOME_TOTAL

df1_corr['AMT_CREDIT'] = df1_corr[df1_corr['AMT_CREDIT']<df1_corr['AMT_CREDIT'].quantile(.85)]['AMT_CREDIT']

#df1_corr['AMT_ANNUITY'] = df1_corr[df1_corr['AMT_GOODS_PRICE']<df1_corr['AMT_GOODS_PRICE'].quantile(.85)]['AMT_GOODS_PRICE']
df1_corr.plot.hexbin(x='AGE', y='AMT_CREDIT', gridsize=15)
sns.boxplot(x="AGE_GROUP", y="AMT_CREDIT", data=df1,palette='rainbow')
df1_corrdf = df1_corr_matrix.where(np.triu(np.ones(df1_corr_matrix.shape),k=1).astype(np.bool))

df0_corrdf = df0_corr_matrix.where(np.triu(np.ones(df0_corr_matrix.shape),k=1).astype(np.bool))
df1_corrdf = df1_corrdf.unstack().reset_index()

df0_corrdf = df0_corrdf.unstack().reset_index()
df1_corrdf.columns =['var1','var2','correlation']

df0_corrdf.columns=['var1','var2','correlation']
df1_corrdf.dropna(subset=['correlation'],inplace=True)

df0_corrdf.dropna(subset=['correlation'],inplace=True)
df1_corrdf.sort_values(by=['correlation'],ascending=False)

#df0_corrdf.sort_values(by=['correlation'],ascending=False)
sns.heatmap(df1_corr_matrix,annot=True,linewidth=1,annot_kws={"size":10},cbar=False)
#removing outlier for AMT_INCOME_TOTAL

df1_filtered = df1[df1['AMT_INCOME_TOTAL']<df['AMT_INCOME_TOTAL'].quantile(.90)]

df_stats_credit = df1_filtered.groupby('NAME_INCOME_TYPE').mean()[['AMT_CREDIT', 'AMT_INCOME_TOTAL', 'AMT_GOODS_PRICE']]

#df_stats_credit = df1_filtered.groupby('AGE_GROUP').mean()[['CREDIT_RATINGS']]

df_stats_credit.sort_values(by='AMT_CREDIT',ascending=False)
df_stats_credit.plot.line(x_compat=True)
plt.figure(figsize=(10,5))

df_filtered = df[df['AMT_INCOME_TOTAL']<df['AMT_INCOME_TOTAL'].quantile(.90)]



sns.boxplot(x="NAME_CONTRACT_TYPE", y="AMT_CREDIT", data=df_filtered,palette='rainbow',hue='TARGET')

plt.yscale("log")