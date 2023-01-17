import os

print(os.listdir('../input'))
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
#read the csv files

AD = pd.read_csv('../input/bank-loans-dataset/application_data.csv')

PD=pd.read_csv('../input/bank-loans-dataset/previous_application.csv')
#to view large outputs

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)
AD.head()
AD.shape
AD.info()
AD.describe()
PD.head()
PD.shape
PD.info()
PD.describe()
#Checking missing values in Application data

(100*AD.isnull().sum()/len(AD)).round(2)
AD = AD.loc[:, AD.isnull().mean() <= .19]
#Checking the data again

(AD.isnull().sum()*100/len(AD)).round(2)
#Checking the missing values in previous application data.

(100*PD.isnull().sum()/len(PD)).round(2)
PD = PD.loc[:, PD.isnull().mean() <= .20]
(100*PD.isnull().sum()/len(PD)).round(2)
#Missing values in Categorical variables NAME_TYPE_SUITE should be replaced with the MODE value 'Unaccompanied'

Mo=AD.NAME_TYPE_SUITE.mode()

AD['NAME_TYPE_SUITE'].fillna(AD.NAME_TYPE_SUITE.mode(), inplace=True)
#AMT_REQ_CREDIT_BUREAU_HOUR,AMT_REQ_CREDIT_BUREAU_DAY,AMT_REQ_CREDIT_BUREAU_WEEK,AMT_REQ_CREDIT_BUREAU_MON

#have more than 90% values as 0 (0 enquiries 1 hour, 1 week and 1 month before application),

#we can impute the missing values in these column as 0 (0 is the median for these columns)  





#Missing values in AMT_GOODS_PRICE could be imputed by mean value for this var since this is a continuous float var

Avg=AD.AMT_REQ_CREDIT_BUREAU_YEAR.mean()

AD['AMT_GOODS_PRICE'].fillna(Avg, inplace=True)
#Imputing Missing values in the following columns with 0. 

#We are assuming that missisng means that there were no inquiries for the person.

AD[['AMT_REQ_CREDIT_BUREAU_HOUR','AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_MON','AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_YEAR']]= AD[['AMT_REQ_CREDIT_BUREAU_HOUR','AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_WEEK' ,'AMT_REQ_CREDIT_BUREAU_MON','AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_YEAR']].fillna(value=0.0)
AD.select_dtypes('object').columns 
AD.select_dtypes('float').columns 
#Following variabes could not be float (number of days,number of enquiries, 

#number of family members and number of people in social circle). SO converting few of them to integer

AD['DAYS_REGISTRATION'] = AD['DAYS_REGISTRATION'].astype(int,errors='ignore')

AD['CNT_FAM_MEMBERS'] = AD['CNT_FAM_MEMBERS'].astype(int,errors='ignore')

AD['OBS_30_CNT_SOCIAL_CIRCLE'] = AD['OBS_30_CNT_SOCIAL_CIRCLE'].astype(int,errors='ignore')

AD['DEF_30_CNT_SOCIAL_CIRCLE'] = AD['DEF_30_CNT_SOCIAL_CIRCLE'].astype(int,errors='ignore')

AD['DAYS_LAST_PHONE_CHANGE'] = AD['DAYS_LAST_PHONE_CHANGE'].astype(int,errors='ignore')

AD['AMT_REQ_CREDIT_BUREAU_HOUR'] = AD['AMT_REQ_CREDIT_BUREAU_HOUR'].astype(int,errors='ignore')
AD.columns
AD.select_dtypes('int64').columns 
AD.select_dtypes('int64').apply(pd.Series.nunique, axis = 0)
## Flag variables with 0 and 1 values should be converted to Categorical vars

cols=['FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL','REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY', 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21']



#function to convert columns

def convert(AD, cols):

    for y in cols:

        AD.loc[:,y].replace((0, 1), ('N', 'Y'), inplace=True)

    return AD



#calling the function for application_data

convert(AD, cols)

AD.dtypes.value_counts()
AD.describe()

#We can see from the results below that some columns have a significant min or max value. We will probe these.
plt.figure(figsize=(10,2))

sns.boxplot(AD.AMT_INCOME_TOTAL)

plt.show()
sns.boxplot(AD.AMT_ANNUITY)

plt.show()
plt.figure(figsize=(15,2))

sns.boxplot(AD.DAYS_EMPLOYED)

plt.show()
#Excluding values outside 99%ile in each of the 3 variables

AD=AD[AD.AMT_ANNUITY<np.nanpercentile(AD['AMT_ANNUITY'], 99)]

AD=AD[AD.DAYS_EMPLOYED<np.nanpercentile(AD['DAYS_EMPLOYED'], 99)]

AD=AD[AD.AMT_INCOME_TOTAL<np.nanpercentile(AD['AMT_INCOME_TOTAL'], 99)]
#Rechecking the columns

plt.figure(figsize=(10,2))

sns.boxplot(AD.AMT_ANNUITY)

plt.show()
#Rechecking the columns

plt.figure(figsize=(10,2))

sns.boxplot(AD.DAYS_EMPLOYED)

plt.show()
#Rechecking the columns

plt.figure(figsize=(10,2))

sns.boxplot(AD.AMT_INCOME_TOTAL)

plt.show()
AD['AMT_INCOME_TOTAL'].describe()
#Creating binned var

AD.loc[:,'INCOME_RANGE']=pd.qcut(AD.loc[:,'AMT_INCOME_TOTAL'],q=[0,0.20,0.50,0.90,1],

labels=['Low','Medium','High','Very_high'])
#Checking Binned Variable

AD['INCOME_RANGE'].value_counts()
# creating another binnned Var

print(AD['EXT_SOURCE_2'].describe())

AD.loc[:,'Rating2']=pd.qcut(AD.loc[:,'EXT_SOURCE_2'],q=[0,0.20,0.50,0.90,1],

labels=['Low','Medium','High','Very_high'])
#Checking Binned Variable

AD['Rating2'].value_counts()
#DAYS_BIRTH column is age of the peron at the time of loan application.

#This could be converted to age in years by dividing by 365.25(Considering leap years). Also it is with a negative sign, hence needs to be treated.

AD['AGE'] =AD['DAYS_BIRTH']//-365.25

AD.drop(['DAYS_BIRTH'],axis=1,inplace=True)
#Checking the Age variable

AD.AGE.describe()
#Creating binned variable for AGE

AD['AGE_GROUP']= pd.cut(AD.AGE,bins=np.linspace(20 ,70,num=11))
#Checking binned variable

AD.AGE_GROUP.value_counts()
count1 = 0 

count0 = 0

for i in AD['TARGET'].values:

    if i == 1:

        count1 += 1

    else:

        count0 += 1

        

count1 = (count1/len(AD['TARGET']))*100

count0 = (count0/len(AD['TARGET']))*100



x = ['Defaulted Population(TARGET=1)','Non-Defauted Population(TARGET=0)']

y = [count1, count0]



explode = (0.1, 0)  # only "explode" the 1st slice



fig1, ax1 = plt.subplots()

ax1.pie(y, explode=explode, labels=x, autopct='%1.1f%%',

        shadow=True, startangle=110)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.title('Data imbalance',fontsize=25)

plt.show()
AD_t0 =AD[AD.TARGET==0]

AD_t1=AD[AD.TARGET==1]
# function to plot for categorical variables

def plotfunc(var):

    plt.figure(figsize=(15,5))

    plt.subplot(1, 2, 1)

    sns.countplot(var, data=AD_t0)

    plt.title('Distribution of '+ '%s' %var +' for Non-Defaulters', fontsize=14)

    plt.xlabel(var)

    plt.xticks(rotation=90)

    plt.ylabel('Number of cases for non-defaulters')

    plt.subplot(1, 2, 2)

    sns.countplot(var, data=AD_t1)

    plt.title('Distribution of '+ '%s' %var +' for Defaulters', fontsize=14)

    plt.xlabel(var)

    plt.xticks(rotation=90)

    plt.ylabel('Number of cases for defaulters')

    plt.show()
plotfunc('NAME_CONTRACT_TYPE')
plotfunc('NAME_TYPE_SUITE')
plotfunc('NAME_INCOME_TYPE')
plotfunc('NAME_HOUSING_TYPE')
plotfunc('NAME_FAMILY_STATUS')
plotfunc('NAME_EDUCATION_TYPE')
Def=AD_t1.NAME_EDUCATION_TYPE.value_counts(normalize=True)

NonDef=AD_t0.NAME_EDUCATION_TYPE.value_counts(normalize=True)

print(Def, NonDef)
plotfunc('CNT_FAM_MEMBERS')
plotfunc('INCOME_RANGE')
Def=AD_t1.INCOME_RANGE.value_counts(normalize=True)

NonDef=AD_t0.INCOME_RANGE.value_counts(normalize=True)

print(Def, NonDef)
plotfunc('Rating2')
plotfunc('AGE_GROUP')
Def=AD_t1.AGE_GROUP.value_counts(normalize=True)

NonDef=AD_t0.AGE_GROUP.value_counts(normalize=True)

print(Def, NonDef)
#selecting int and float columns for correlation

cols_num=list(AD_t0.select_dtypes('int64').columns)

cols_float=list(AD_t0.select_dtypes('float').columns)



cols=cols_num+cols_float



Nondef_num=AD_t0[cols]

Nondef_corr = Nondef_num.corr()

round(Nondef_corr, 3)
l1=Nondef_corr.unstack()

l1.sort_values(ascending=False).drop_duplicates()
#selecting int and float columns for correlation

cols_num=list(AD_t1.select_dtypes('int64').columns)

cols_float=list(AD_t1.select_dtypes('float').columns)



cols=cols_num+cols_float



def_num=AD_t1[cols]

def_corr = def_num.corr()

round(def_corr, 3)
l1=def_corr.unstack()

l1.sort_values(ascending=False).drop_duplicates()
# defining function for plotting contnous variables

def plotcont(var):

    plt.figure(figsize=(15,5))

    plt.subplot(1, 2, 1)

    #sns.distplot(AD_t0[var].dropna(),kde=True)

    AD_t0[var].plot.hist()

    plt.title('Distribution for Non-Defaulters', fontsize=14)

    plt.xlabel(var)

    #plt.xticks(rotation=90)

    plt.subplot(1, 2, 2)

    #sns.distplot(AD_t1[var].dropna(),kde=True)

    AD_t1[var].plot.hist()

    plt.title('Distribution for Defaulters', fontsize=14)

    plt.xlabel(var)

    #plt.xticks(rotation=90)

    plt.show()
plotcont('AMT_GOODS_PRICE')
plotcont('REGION_POPULATION_RELATIVE')
plt.figure(figsize=(18,6))

plt.subplot(121)

sns.countplot(x='TARGET',hue='CODE_GENDER',data=AD_t0)

plt.subplot(122)

sns.countplot(x='TARGET',hue='CODE_GENDER',data=AD_t1)

plt.show()
mean_income_t_0_m = AD_t0[AD_t0.CODE_GENDER=='M']['AMT_INCOME_TOTAL'].mean()



mean_income_t_0_f = AD_t0[AD_t0.CODE_GENDER=='F']['AMT_INCOME_TOTAL'].mean()



mean_income_t_1_m = AD_t1[AD_t1.CODE_GENDER=='M']['AMT_INCOME_TOTAL'].mean()



mean_income_t_1_f = AD_t1[AD_t1.CODE_GENDER=='F']['AMT_INCOME_TOTAL'].mean()



x_male = ['AMT_INCOME_mean_T_0_Male','AMT_INCOME_mean_T_1_Male']



y_male = [mean_income_t_0_m,mean_income_t_1_m]



x_Female = ['AMT_INCOME_mean_T_0_Female','AMT_INCOME_mean_T_1_Female']



y_Female = [mean_income_t_0_f,mean_income_t_1_f]

plt.figure(figsize=(18,6))

plt.subplot(121)

plt.bar(x_male,y_male)



plt.subplot(122)

plt.bar(x_Female,y_Female)



plt.show()
def plotbivarcontcont(var1,var2):

    plt.figure(figsize=(15,5))

    plt.subplot(1, 2, 1)

    sns.scatterplot(x=var1,y=var2,data=AD_t0)

    plt.title('TARGET=0')

    plt.xlabel(var1)

    plt.xticks(rotation=90)

    plt.subplot(1, 2, 2)

    sns.scatterplot(x=var1,y=var2,data=AD_t1)

    plt.title('TARGET=1')

    plt.xlabel(var1)

    plt.xticks(rotation=90)

    plt.show()
plt.figure(figsize=(18,6))

plt.subplot(121)

sns.scatterplot(x='AMT_CREDIT',y='AMT_INCOME_TOTAL',data=AD_t0)

plt.title('INCOME vs CREDIT for Non-Defaulters')



plt.subplot(122)

sns.scatterplot(x='AMT_CREDIT',y='AMT_INCOME_TOTAL',data=AD_t1)

plt.title('INCOME vs CREDIT for Defaulters')

plt.show()
plt.figure(figsize=(18,6))

plt.subplot(121)

sns.scatterplot(x='AMT_CREDIT',y='AMT_GOODS_PRICE',data=AD_t0)

plt.title('CREDIT vs GOODS PRICE for Non-Defaulters')



plt.subplot(122)

sns.scatterplot(x='AMT_CREDIT',y='AMT_GOODS_PRICE',data=AD_t1)

plt.title('CREDIT vs GOODS PRICE for Defaulters')

plt.show()
PD.shape
PD.info()
PD.dtypes.value_counts()
#Checking missng values

((PD.isnull().sum()*100)/len(PD)).round(2)
#Removing columns having more than 40% missing values

PD = PD.loc[:, PD.isnull().mean() <= .4]
#Rechecking the data

PD.info()
#Since the Previous application data is very large, we are deleting some rows so that it could be easily merged

PD=PD.loc[0:70000]
PD.shape
Combined = pd.merge(AD, PD, how='left', on=['SK_ID_CURR'])
Combined.shape
Combined.columns
sns.countplot(Combined.NAME_CONTRACT_STATUS)

plt.xlabel("Contract Status")

plt.ylabel("Count of Contract Status")

plt.title("Distribution of Contract Status")

plt.show()
approved=Combined[Combined.NAME_CONTRACT_STATUS=='Approved']

refused=Combined[Combined.NAME_CONTRACT_STATUS=='Refused']

canceled=Combined[Combined.NAME_CONTRACT_STATUS=='Canceled']

unused=Combined[Combined.NAME_CONTRACT_STATUS=='Unused Offer']
def plot_func(var):

    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(15,5))

    

    s1=sns.countplot(ax=ax1,x=refused[var], data=refused, order= refused[var].value_counts().index,)

    ax1.set_title("Refused", fontsize=10)

    ax1.set_xlabel('%s' %var)

    ax1.set_ylabel("Count of Loans")

    s1.set_xticklabels(s1.get_xticklabels(),rotation=90)

    

    s2=sns.countplot(ax=ax2,x=approved[var], data=approved, order= approved[var].value_counts().index,)

    s2.set_xticklabels(s2.get_xticklabels(),rotation=90)

    ax2.set_xlabel('%s' %var)

    ax2.set_ylabel("Count of Loans")

    ax2.set_title("Approved", fontsize=10)

    

    

    s3=sns.countplot(ax=ax3,x=canceled[var], data=canceled, order= canceled[var].value_counts().index,)

    ax3.set_title("Canceled", fontsize=10)

    ax3.set_xlabel('%s' %var)

    ax3.set_ylabel("Count of Loans")

    s3.set_xticklabels(s3.get_xticklabels(),rotation=90)

    plt.show()
plot_func('TARGET')
refused.TARGET.value_counts(normalize=True)
approved.TARGET.value_counts(normalize=True)
canceled.TARGET.value_counts(normalize=True)
def plot_func1(var):

    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(15,5))

    

    s1=sns.scatterplot(x='AMT_CREDIT',y='AMT_GOODS_PRICE',data=approved)

    ax1.set_title("Refused", fontsize=10)

    ax1.set_xlabel('%s' %var)

    ax1.set_ylabel("Count of Loans")

    s1.set_xticklabels(s1.get_xticklabels(),rotation=90)

    

    s2=sns.scatterplot(x='AMT_CREDIT',y='AMT_GOODS_PRICE',data=refused)

    s2.set_xticklabels(s2.get_xticklabels(),rotation=90)

    ax2.set_xlabel('%s' %var)

    ax2.set_ylabel("Count of Loans")

    ax2.set_title("Approved", fontsize=10)

    

    

    s3=sns.scatterplot(x='AMT_CREDIT',y='AMT_GOODS_PRICE',data=cancelled)

    ax3.set_title("Canceled", fontsize=10)

    ax3.set_xlabel('%s' %var)

    ax3.set_ylabel("Count of Loans")

    s3.set_xticklabels(s3.get_xticklabels(),rotation=90)

    plt.show()
plt.figure(figsize=(18,6))

plt.subplot(1,2,1)

sns.scatterplot(x='AMT_APPLICATION',y='AMT_INCOME_TOTAL',data=refused)

plt.title('Refused')



plt.subplot(1,2,2)

sns.scatterplot(x='AMT_APPLICATION',y='AMT_INCOME_TOTAL',data=approved)

plt.title('Approved')

plt.show()