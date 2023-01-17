# import packages

import os

import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

import missingno as msno

from collections import Counter

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.metrics import classification_report, confusion_matrix



%matplotlib inline



# list all files

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
def readfile(filevar): 

    try: 

        df = pd.read_csv(filevar)

        return df

    except:

        print('File loading - Failed!')
# LOAD THE TRAINING SET

df_train = readfile('/kaggle/input/home-credit-default-risk/application_train.csv')

df_train.info(verbose=True, null_counts=True)
# LOAD THE TEST SET

df_test = readfile('/kaggle/input/home-credit-default-risk/application_test.csv')

df_test.info(verbose=True, null_counts=True)
# Print first 5 rows of the training file

df_train.head()
# Show target distribution

sns.set_style('darkgrid')

print(df_train.TARGET.value_counts())

df_train.TARGET.plot.hist(color='mediumseagreen').set_xlabel('Target value: 0 or 1');
# COLUMNS TO BE DROPPED

list_col_drop = ['SK_ID_CURR']
tempY=df_train[df_train.FLAG_OWN_CAR=='Y']

tempN=df_train[df_train.FLAG_OWN_CAR=='N']

tempY_targ1=tempY[tempY.TARGET==1]

tempN_targ1=tempN[tempN.TARGET==1]
print('People count who owns a car:',len(tempY),'(',round(len(tempY)/len(df_train.index)*100,2),'%)')

print('People count who DOES NOT own a car:',len(tempN),'(',round(len(tempN)/len(df_train.index)*100,2),'%)')

print('Percentage of people who defaulted (with cars):',round(len(tempY_targ1)/len(tempY)*100,2),'%')

print('Percentage of people who defaulted (no cars):',round(len(tempN_targ1)/len(tempN)*100,2),'%')
tempY=df_train[df_train.FLAG_OWN_REALTY=='Y']

tempN=df_train[df_train.FLAG_OWN_REALTY=='N']

tempY_targ1=tempY[tempY.TARGET==1]

tempN_targ1=tempN[tempN.TARGET==1]
print('People count who owns realty:',len(tempY),'(',round(len(tempY)/len(df_train.index)*100,2),'%)')

print('People count who DOES NOT own realty:',len(tempN),'(',round(len(tempN)/len(df_train.index)*100,2),'%)')

print('Percentage of people who defaulted (with realty):',round(len(tempY_targ1)/len(tempY)*100,2),'%')

print('Percentage of people who defaulted (no realty):',round(len(tempN_targ1)/len(tempN)*100,2),'%')
# CREATE NEW COLUMN : 0 - none, 1 - with car no realty, 2 - no car with realty, 3 - with car with realty

list_col_new_asset = ['FLAG_OWN_CAR','FLAG_OWN_REALTY'] 
sns.pairplot(df_train[['NAME_CONTRACT_TYPE','AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE']],hue='NAME_CONTRACT_TYPE');
corr1=round(df_train.AMT_CREDIT.corr(df_train.AMT_GOODS_PRICE),2)

corr2=round(df_train.AMT_ANNUITY.corr(df_train.AMT_CREDIT),2)

corr3=round(df_train.AMT_ANNUITY.corr(df_train.AMT_GOODS_PRICE),2)
print('Correlation of Credit amount vs Price of goods:',corr1)

print('Correlation of Annuity amount vs Credit amount:',corr2)

print('Correlation of Annuity amount vs Price of goods:',corr3)
cash=df_train[df_train.NAME_CONTRACT_TYPE == 'Cash loans']

rev=df_train[df_train.NAME_CONTRACT_TYPE == 'Revolving loans']

def_cash=cash[cash.TARGET==1]

def_rev=rev[rev.TARGET==1]
print('Percentage of defaulted cash loan:',round(len(def_cash)/len(cash)*100,2,),'%')

print('Percentage of defaulted revolving loan:',round(len(def_rev)/len(rev)*100,2),'%')

sns.catplot(data=df_train,x='NAME_CONTRACT_TYPE',hue='TARGET',kind='count');
# COLUMNS TO BE DROPPED

list_col_drop.extend(['NAME_CONTRACT_TYPE','AMT_GOODS_PRICE'])



# CREATE NEW COLUMN : PERCENT_ANNUITY_INCOME

list_col_new_annuity = ['AMT_ANNUITY','AMT_INCOME_TOTAL']



# CREATE NEW COLUMN : CREDIT_ANNUITY_INCOME

list_col_new_credit = ['AMT_CREDIT','AMT_INCOME_TOTAL']
sns.catplot(data=df_train,x='NAME_FAMILY_STATUS',hue='TARGET',kind='count');

plt.xticks(rotation=90);
df_train[['CNT_CHILDREN','NAME_FAMILY_STATUS','CNT_FAM_MEMBERS']][df_train.NAME_FAMILY_STATUS=='Married'].tail()
# COLUMNS TO BE DROPPED

list_col_drop.extend(['CNT_CHILDREN', 'NAME_FAMILY_STATUS'])



# COMPLETE COLUMN : 

list_col_fill_fam = ['CNT_FAM_MEMBERS']
corr1=round(df_train.OWN_CAR_AGE.corr(df_train.TARGET),2)

corr2=round(df_train.REG_REGION_NOT_WORK_REGION.corr(df_train.TARGET),2)

corr3=round(df_train.REG_CITY_NOT_WORK_CITY.corr(df_train.TARGET),2)

no_car,yes_car = df_train.FLAG_OWN_CAR.value_counts()
print('Correlation of Age of Car vs Target:',corr1)

print('Correlation of Registered Region aint Work Region vs Target:',corr2)

print('Correlation of Registered City aint Work City vs Target:',corr3)
df_train[['OWN_CAR_AGE','REG_REGION_NOT_WORK_REGION','REG_CITY_NOT_WORK_CITY']].describe()
print('How many customers own a car? :',yes_car)

print('How many customers do NOT own a car? :',no_car)

print('How many missing values on OWN_CAR_AGE? :',df_train.OWN_CAR_AGE.isnull().sum())
# CREATE NEW COLUMN : EXPENDITURE_CAR : 0 - no car, +=1 per age band

list_col_new_car = ['OWN_CAR_AGE']



# COLUMNS TO BE DROPPED

list_col_drop.extend(['REG_REGION_NOT_WORK_REGION','REG_CITY_NOT_WORK_CITY'])
temp = ['NAME_HOUSING_TYPE',

        'APARTMENTS_AVG',

        'BASEMENTAREA_AVG',

        'YEARS_BEGINEXPLUATATION_AVG',

        'YEARS_BUILD_AVG',

        'COMMONAREA_AVG',

        'ELEVATORS_AVG',

        'ENTRANCES_AVG',

        'FLOORSMAX_AVG',

        'FLOORSMIN_AVG',

        'LANDAREA_AVG',

        'LIVINGAPARTMENTS_AVG',

        'LIVINGAREA_AVG',

        'NONLIVINGAPARTMENTS_AVG',

        'NONLIVINGAREA_AVG',

        'APARTMENTS_MODE',

        'BASEMENTAREA_MODE',

        'YEARS_BEGINEXPLUATATION_MODE',

        'YEARS_BUILD_MODE',

        'COMMONAREA_MODE',

        'ELEVATORS_MODE',

        'ENTRANCES_MODE',

        'FLOORSMAX_MODE',

        'FLOORSMIN_MODE',

        'LANDAREA_MODE',

        'LIVINGAPARTMENTS_MODE',

        'LIVINGAREA_MODE',

        'NONLIVINGAPARTMENTS_MODE',

        'NONLIVINGAREA_MODE',

        'APARTMENTS_MEDI',

        'BASEMENTAREA_MEDI',

        'YEARS_BEGINEXPLUATATION_MEDI',

        'YEARS_BUILD_MEDI',

        'COMMONAREA_MEDI',

        'ELEVATORS_MEDI',

        'ENTRANCES_MEDI',

        'FLOORSMAX_MEDI',

        'FLOORSMIN_MEDI',

        'LANDAREA_MEDI',

        'LIVINGAPARTMENTS_MEDI',

        'LIVINGAREA_MEDI',

        'NONLIVINGAPARTMENTS_MEDI',

        'NONLIVINGAREA_MEDI',

        'FONDKAPREMONT_MODE',

        'HOUSETYPE_MODE',

        'TOTALAREA_MODE',

        'WALLSMATERIAL_MODE',

        'EMERGENCYSTATE_MODE']
# Nullity by column

msno.bar(df_train[temp],figsize=(20,5));
msno.heatmap(df_train[temp],fontsize=12);
for i in ['APARTMENTS_AVG','LANDAREA_AVG','LIVINGAPARTMENTS_AVG','NONLIVINGAREA_MEDI']:

    temp=df_train[['NAME_HOUSING_TYPE']][df_train[i].isnull()]

    sns.catplot(data=temp,x='NAME_HOUSING_TYPE',kind='count',palette="rocket")

    plt.xticks(rotation=20)

    title = 'Housing Types with null values on ' + i

    plt.title(title)
# CREATE NEW COLUMN : EXPENDITURE_HOUSE 

list_col_new_house = ['NAME_HOUSING_TYPE',

                    'APARTMENTS_AVG',

                    'BASEMENTAREA_AVG',

                    'YEARS_BEGINEXPLUATATION_AVG',

                    'YEARS_BUILD_AVG',

                    'COMMONAREA_AVG',

                    'ELEVATORS_AVG',

                    'ENTRANCES_AVG',

                    'FLOORSMAX_AVG',

                    'FLOORSMIN_AVG',

                    'LANDAREA_AVG',

                    'LIVINGAPARTMENTS_AVG',

                    'LIVINGAREA_AVG',

                    'NONLIVINGAPARTMENTS_AVG',

                    'NONLIVINGAREA_AVG',

                    'APARTMENTS_MODE',

                    'BASEMENTAREA_MODE',

                    'YEARS_BEGINEXPLUATATION_MODE',

                    'YEARS_BUILD_MODE',

                    'COMMONAREA_MODE',

                    'ELEVATORS_MODE',

                    'ENTRANCES_MODE',

                    'FLOORSMAX_MODE',

                    'FLOORSMIN_MODE',

                    'LANDAREA_MODE',

                    'LIVINGAPARTMENTS_MODE',

                    'LIVINGAREA_MODE',

                    'NONLIVINGAPARTMENTS_MODE',

                    'NONLIVINGAREA_MODE',

                    'APARTMENTS_MEDI',

                    'BASEMENTAREA_MEDI',

                    'YEARS_BEGINEXPLUATATION_MEDI',

                    'YEARS_BUILD_MEDI',

                    'COMMONAREA_MEDI',

                    'ELEVATORS_MEDI',

                    'ENTRANCES_MEDI',

                    'FLOORSMAX_MEDI',

                    'FLOORSMIN_MEDI',

                    'LANDAREA_MEDI',

                    'LIVINGAPARTMENTS_MEDI',

                    'LIVINGAREA_MEDI',

                    'NONLIVINGAPARTMENTS_MEDI',

                    'NONLIVINGAREA_MEDI',

                    'FONDKAPREMONT_MODE',

                    'HOUSETYPE_MODE',

                    'TOTALAREA_MODE',

                    'WALLSMATERIAL_MODE',

                    'EMERGENCYSTATE_MODE']
fig,ax = plt.subplots(figsize=(10,7))

sns.countplot(data=df_train,x='REGION_POPULATION_RELATIVE',hue='TARGET',ax=ax);

plt.xticks(rotation=90)

plt.show()
sns.catplot(data=df_train,x='WEEKDAY_APPR_PROCESS_START',hue='TARGET',kind='count');

plt.xticks(rotation=90);
fig,ax = plt.subplots(figsize=(10,3))

sns.countplot(data=df_train,x='HOUR_APPR_PROCESS_START',hue='TARGET',ax=ax);

plt.xticks(rotation=90)

plt.show()
# COLUMNS TO BE DROPPED

list_col_drop.extend(['WEEKDAY_APPR_PROCESS_START','HOUR_APPR_PROCESS_START'])
temp=df_train[['TARGET','EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']]

print(temp.info())

sns.pairplot(temp,hue='TARGET');
# COMPLETE COLUMNS : Base on mean EXT_SOURCE

list_col_fill_ext = ['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']
df_train[['AMT_INCOME_TOTAL','NAME_INCOME_TYPE','DAYS_EMPLOYED','OCCUPATION_TYPE','ORGANIZATION_TYPE']].describe(include='all')
def get_thresh(df,field):

    """ Outliers are usually > 3 standard deviations away from the mean. """

    ave=np.mean(df[field])

    sdev=np.std(df[field])

    threshold=round(ave+(3*sdev),2)

    print('Threshold for',field,':',threshold)

    return threshold
thresh_income = get_thresh(df_train,'AMT_INCOME_TOTAL')

anomaly_emp = int(df_train['DAYS_EMPLOYED'][df_train['DAYS_EMPLOYED']>0].unique())

temp_orig=df_train[['AMT_INCOME_TOTAL','DAYS_EMPLOYED']]

temp_no_outliers=df_train[['AMT_INCOME_TOTAL','DAYS_EMPLOYED']][(df_train.AMT_INCOME_TOTAL<thresh_income)&(df_train['DAYS_EMPLOYED']<=0)]

print('Anomalous data for DAYS_EMPLOYED :',anomaly_emp)
def plotdist(df,f1,f2):

    f,axes = plt.subplots(1,2,figsize=(10,3))

    sns.distplot(df[[f1]],ax=axes[0]).set_title(f1)

    plt.xticks(rotation=75)



    sns.distplot(df[[f2]],ax=axes[1]).set_title(f2)

    plt.xticks(rotation=75)

    plt.tight_layout()
# AMT_INCOME_TOTAL, DAYS_EMPLOYED --> ORIGINAL VALUE WITH OUTLIERS

plotdist(temp_orig,'AMT_INCOME_TOTAL','DAYS_EMPLOYED')
# AMT_INCOME_TOTAL, DAYS_EMPLOYED --> OUTLIERS REMOVED

plotdist(temp_no_outliers,'AMT_INCOME_TOTAL','DAYS_EMPLOYED')
# NUMBER OF INDIVIDUALS HAVING THE DAYS EMPLOYED ANOMALOUS DATA

len(df_train[df_train.DAYS_EMPLOYED==anomaly_emp])
unpaid=df_train[df_train.TARGET==1]

sns.catplot(data=unpaid,x='NAME_INCOME_TYPE',kind='count');

plt.xticks(rotation=90);

plt.title('Income Stream of DEFAULTED LOANS');
fig,ax = plt.subplots(figsize=(15,5))

sns.countplot(data=unpaid,x='ORGANIZATION_TYPE',ax=ax);

plt.xticks(rotation=90)

plt.title('Organization of people with DEFAULTED LOANS')

plt.show()
def plotstrip(df,xval,yval,hueval,yfig):

    fig,ax = plt.subplots(figsize=(15,yfig))

    sns.stripplot(x=xval,y=yval,hue=hueval,data=df,alpha=0.5,jitter=0.8,dodge=True,ax=ax).set_title(yval);

    plt.legend(bbox_to_anchor=(1.05, 1))

    plt.show()
temp=df_train[df_train.OCCUPATION_TYPE.isnull()]

print('Individuals that left OCCUPATION_TYPE field blank:')

print(temp.NAME_INCOME_TYPE.value_counts())
print('Individuals that left ORGANIZATION_TYPE field blank (Top 10 only):')

print(temp.ORGANIZATION_TYPE.value_counts().head(10))
plotstrip(df_train[df_train.AMT_INCOME_TOTAL<thresh_income],'AMT_INCOME_TOTAL','NAME_INCOME_TYPE','TARGET',5)
plotstrip(df_train[df_train.AMT_INCOME_TOTAL<thresh_income],'AMT_INCOME_TOTAL','OCCUPATION_TYPE','TARGET',10)
plotstrip(df_train[df_train.AMT_INCOME_TOTAL<thresh_income],'AMT_INCOME_TOTAL','ORGANIZATION_TYPE','TARGET',20)
# COLUMNS TO BE DROPPED

list_col_drop.extend(['OCCUPATION_TYPE','NAME_INCOME_TYPE'])



# CREATE NEW COLUMN : INCOME_BAND

list_col_new_income = ['AMT_INCOME_TOTAL'] 



# CONVERT COLUMN : 365243 to -29200 

list_col_conv_daysemp = ['DAYS_EMPLOYED']



# CREATE NEW COLUMN : YEARS_EMPLOYED

list_col_new_yrsemp = ['DAYS_EMPLOYED']



# CONVERT COLUMN : 

list_col_conv_org = ['ORGANIZATION_TYPE'] 
df_train[['CODE_GENDER','NAME_TYPE_SUITE','NAME_EDUCATION_TYPE','DAYS_REGISTRATION','DAYS_ID_PUBLISH']].describe(include='all')
g = sns.FacetGrid(df_train,row='CODE_GENDER',col='NAME_EDUCATION_TYPE',hue='TARGET',height=4) # nominal

g.map(plt.scatter,'DAYS_ID_PUBLISH','DAYS_REGISTRATION',alpha=0.5,edgecolor='k',linewidth=0.5) # continuous



fig = g.fig 

fig.set_size_inches(25,10)

fig.subplots_adjust(top=0.85,wspace=0.3)

fig.suptitle('Gender - Educational Attainment - Registration Change - ID Change - Credit Ranking',fontsize=20)



l = g.add_legend(title='Credit Score')
paid = df_train[df_train.TARGET==0]

unpaid = df_train[df_train.TARGET==1]



f,axes = plt.subplots(1,2,figsize=(10,3))

sns.kdeplot(paid['DAYS_ID_PUBLISH'],paid['DAYS_REGISTRATION'],cmap="Blues",shade=True,shade_lowest=False,ax=axes[0]).set_title('Paid');

sns.kdeplot(unpaid['DAYS_ID_PUBLISH'],unpaid['DAYS_REGISTRATION'],cmap="Reds",shade=True,shade_lowest=False,ax=axes[1]).set_title('Unpaid');

sns.set_style('whitegrid')

plt.tight_layout()
df_train.NAME_EDUCATION_TYPE.value_counts()
f,axes = plt.subplots(1,2,figsize=(10,5),sharex=True)

sns.distplot(df_train[['DAYS_BIRTH']][df_train.TARGET==0],hist=False,color="b",kde_kws={"shade":True},ax=axes[0]).set_title('Target == 0 (Paid)');

sns.distplot(df_train[['DAYS_BIRTH']][df_train.TARGET==1],hist=False,color="r",kde_kws={"shade":True},ax=axes[1]).set_title('Target == 1 (Unpaid)');
# COLUMNS TO BE DROPPED

list_col_drop.extend(['NAME_TYPE_SUITE','DAYS_REGISTRATION','DAYS_ID_PUBLISH'])



# CONVERT COLUMN : 

list_col_conv_gender = ['CODE_GENDER']



# CONVERT COLUMN :

list_col_conv_edu = ['NAME_EDUCATION_TYPE']



# CREATE NEW COLUMN : AGE

list_col_new_age = ['DAYS_BIRTH']

temp1=['FLAG_MOBIL',

'FLAG_EMP_PHONE',

'FLAG_WORK_PHONE',

'FLAG_CONT_MOBILE',

'FLAG_PHONE',

'FLAG_EMAIL']



temp2=['FLAG_DOCUMENT_2',

'FLAG_DOCUMENT_3',

'FLAG_DOCUMENT_4',

'FLAG_DOCUMENT_5',

'FLAG_DOCUMENT_6',

'FLAG_DOCUMENT_7',

'FLAG_DOCUMENT_8',

'FLAG_DOCUMENT_9',

'FLAG_DOCUMENT_10',

'FLAG_DOCUMENT_11',

'FLAG_DOCUMENT_12',

'FLAG_DOCUMENT_13',

'FLAG_DOCUMENT_14',

'FLAG_DOCUMENT_15',

'FLAG_DOCUMENT_16',

'FLAG_DOCUMENT_17',

'FLAG_DOCUMENT_18',

'FLAG_DOCUMENT_19',

'FLAG_DOCUMENT_20',

'FLAG_DOCUMENT_21']
df_train[temp1+temp2].describe()
def featsum(cols,newcol):

    """ Sums up items per row across all columns.

        Returns df with new sum column and catplot.

    """

    sample_count=df_train[cols].sum(axis=1)

    sample = df_train.copy()

    sample[newcol]=sample_count

    sns.catplot(data=sample,x=newcol,hue='TARGET',kind='count');
featsum(temp1,'FlagContact')
featsum(temp2,'FlagDocu')
# CREATE NEW COLUMN : FLAG_CONTACTS

list_col_new_flagCont = [ 

    'FLAG_MOBIL',

    'FLAG_EMP_PHONE',

    'FLAG_WORK_PHONE',

    'FLAG_CONT_MOBILE',

    'FLAG_PHONE',

    'FLAG_EMAIL']



# CREATE NEW COLUMN : FLAG_DOCS

list_col_new_flagDoc = [ 

    'FLAG_DOCUMENT_2',

    'FLAG_DOCUMENT_3',

    'FLAG_DOCUMENT_4',

    'FLAG_DOCUMENT_5',

    'FLAG_DOCUMENT_6',

    'FLAG_DOCUMENT_7',

    'FLAG_DOCUMENT_8',

    'FLAG_DOCUMENT_9',

    'FLAG_DOCUMENT_10',

    'FLAG_DOCUMENT_11',

    'FLAG_DOCUMENT_12',

    'FLAG_DOCUMENT_13',

    'FLAG_DOCUMENT_14',

    'FLAG_DOCUMENT_15',

    'FLAG_DOCUMENT_16',

    'FLAG_DOCUMENT_17',

    'FLAG_DOCUMENT_18',

    'FLAG_DOCUMENT_19',

    'FLAG_DOCUMENT_20',

    'FLAG_DOCUMENT_21']
def plotcat(x,r):

    sns.catplot(data=df_train,x=x,hue='TARGET',kind='count');

    plt.xticks(rotation=r);
plotcat('REGION_RATING_CLIENT',0)
plotcat('REGION_RATING_CLIENT_W_CITY',0)
print('Correlation:',round(df_train['REGION_RATING_CLIENT_W_CITY'].corr(df_train['REGION_RATING_CLIENT']),2))
temp=[

    'REG_REGION_NOT_LIVE_REGION',

    'LIVE_REGION_NOT_WORK_REGION',

    'REG_CITY_NOT_LIVE_CITY',

    'LIVE_CITY_NOT_WORK_CITY'

]
featsum(temp,'FlagAddr')
# COLUMNS TO BE DROPPED

list_col_drop.extend(['REGION_RATING_CLIENT_W_CITY'])



# CREATE NEW COLUMN : FLAG_ADDR

list_col_new_flagAddr = ['REG_REGION_NOT_LIVE_REGION','LIVE_REGION_NOT_WORK_REGION','REG_CITY_NOT_LIVE_CITY','LIVE_CITY_NOT_WORK_CITY']
df_train[['OBS_30_CNT_SOCIAL_CIRCLE','DEF_30_CNT_SOCIAL_CIRCLE','OBS_60_CNT_SOCIAL_CIRCLE','DEF_60_CNT_SOCIAL_CIRCLE','DAYS_LAST_PHONE_CHANGE']].describe()
def featsumviolin(df,cols1,cols2,newcol1,newcol2):

    """ Sums up items per row across all columns.

        Returns df with new sum column and violinplot.

    """

    sample_count1=df[cols1].sum(axis=1)

    sample_count2=df[cols2].sum(axis=1)

    sample = df_train.copy()

    sample[newcol1]=sample_count1

    sample[newcol2]=sample_count2

    fig,ax = plt.subplots(figsize=(10,5))

    sns.violinplot(data=sample,hue='TARGET',x=newcol1,y=newcol2,split=True,inner='quart',linewidth=1.3,palette={1:"#FF9999", 0:"white"});
featsumviolin(df_train[(df_train.OBS_30_CNT_SOCIAL_CIRCLE<348)&(df_train.OBS_60_CNT_SOCIAL_CIRCLE<344)],

              ['DEF_30_CNT_SOCIAL_CIRCLE','DEF_60_CNT_SOCIAL_CIRCLE'],

              ['OBS_30_CNT_SOCIAL_CIRCLE','OBS_60_CNT_SOCIAL_CIRCLE'],

              'DEF_NEW','OBS_NEW')
sns.set(palette="muted",color_codes=True)

f,axes = plt.subplots(2,2,figsize=(10,5),sharex=True)



sns.distplot(df_train[['DAYS_LAST_PHONE_CHANGE']][df_train.TARGET==0],kde=False,color="b",ax=axes[0,0]).set_title('Target == 0 (Paid)')

sns.distplot(df_train[['DAYS_LAST_PHONE_CHANGE']][df_train.TARGET==0],hist=False,color="g",kde_kws={"shade":True},ax=axes[0,1]).set_title('Target == 0 (Paid)')



sns.distplot(df_train[['DAYS_LAST_PHONE_CHANGE']][df_train.TARGET==1],kde=False,color="r",ax=axes[1,0]).set_title('Target == 1 (Unpaid)')

sns.distplot(df_train[['DAYS_LAST_PHONE_CHANGE']][df_train.TARGET==1],hist=False,color="m",kde_kws={"shade":True},ax=axes[1,1]).set_title('Target == 1 (Unpaid)')



plt.tight_layout()
# COLUMNS TO BE DROPPED

list_col_drop.extend(['OBS_30_CNT_SOCIAL_CIRCLE','DEF_30_CNT_SOCIAL_CIRCLE','OBS_60_CNT_SOCIAL_CIRCLE','DEF_60_CNT_SOCIAL_CIRCLE','DAYS_LAST_PHONE_CHANGE'])
df_train[['AMT_REQ_CREDIT_BUREAU_HOUR','AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_MON','AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_YEAR']].describe()
for i in ['AMT_REQ_CREDIT_BUREAU_HOUR','AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_WEEK']:

    for j in ['AMT_REQ_CREDIT_BUREAU_MON','AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_YEAR']:

        fig,ax = plt.subplots(figsize=(10,3))

        sns.stripplot(data=df_train[df_train.AMT_REQ_CREDIT_BUREAU_QRT<261],x=i,y=j,hue='TARGET',alpha=0.5,jitter=0.3,dodge=True,ax=ax)

        plt.show()
# COLUMNS TO BE DROPPED

list_col_drop.extend(['AMT_REQ_CREDIT_BUREAU_HOUR','AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_MON','AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_YEAR'])
def getmean(df,ls_cols):

    list_mean = []

    for i in ls_cols:

        mean_val = df[i].mean()

        list_mean.append(mean_val)

    return list_mean



def fill_ave_ext(df,ls_cols):  

    list_mean = getmean(df,ls_cols) # mean of EXT_SOURCE_*

    ctr=0

    for i in ls_cols:

        df[i] = df[i].fillna(list_mean[ctr])

        ctr+=1

    return df

        

# Fill in the training set

fill_ave_ext(df_train,list_col_fill_ext);



# Fill in the testing set

fill_ave_ext(df_test,list_col_fill_ext);
# NO MORE NULL VALUES FOR 'EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3'

print(df_train[list_col_fill_ext].info())

print(df_test[list_col_fill_ext].info())
def fill_0_fam(df,ls_cols):

    df[ls_cols] = df[ls_cols].fillna(0)

    return df

    

    

fill_0_fam(df_train,'CNT_FAM_MEMBERS');

fill_0_fam(df_test,'CNT_FAM_MEMBERS');
# NO MORE NULL VALUES FOR 'CNT_FAM_MEMBERS'

print(df_train[list_col_fill_fam].info())

print(df_test[list_col_fill_fam].info())
# ANOMALY DATA COUNT BEFORE CONVERSION

print('Train set :',len(df_train[df_train.DAYS_EMPLOYED==365243]))

print('Test set  :',len(df_test[df_test.DAYS_EMPLOYED==365243]))
def conv_daysemp(df,ls_cols):

    df[ls_cols[0]].replace(to_replace=365243,value=-29200,inplace=True)

    return df
conv_daysemp(df_train,list_col_conv_daysemp);

conv_daysemp(df_test,list_col_conv_daysemp);
# ANOMALY DATA COUNT AFTER CONVERSION

print('Train set :',len(df_train[df_train.DAYS_EMPLOYED==365243]))

print('Test set  :',len(df_test[df_test.DAYS_EMPLOYED==365243]))
# BEFORE CONVERSION

print(df_train[['CODE_GENDER','NAME_EDUCATION_TYPE','ORGANIZATION_TYPE']].head(3))

print(df_test[['CODE_GENDER','NAME_EDUCATION_TYPE','ORGANIZATION_TYPE']].head(3))
def conv_gender(df,ls_cols):

    df[ls_cols[0]] = df[ls_cols[0]].map({'XNA':0,'M':1,'F':2}).astype(int)

    return df



def conv_education(df,ls_cols):

    temp_dict = {

        'Lower secondary':1,

        'Secondary / secondary special':2,

        'Incomplete higher':3,

        'Higher education':4,

        'Academic degree':5

    }

    df[ls_cols[0]] = df[ls_cols[0]].map(temp_dict).astype(int)

    return df



def conv_org(df,ls_cols):

    ls_ctr=[i for i in np.arange(1,len(ls_cols)+1)]

    temp_dict = dict(zip(ls_cols,ls_ctr))

    df['ORGANIZATION_TYPE'] = df['ORGANIZATION_TYPE'].map(temp_dict).astype(int)

    return df
conv_gender(df_train,list_col_conv_gender);

conv_gender(df_test,list_col_conv_gender);



conv_education(df_train,list_col_conv_edu);

conv_education(df_test,list_col_conv_edu);
orgtype = sorted(df_train.ORGANIZATION_TYPE.unique())

conv_org(df_train,orgtype);

conv_org(df_test,orgtype);
# AFTER CONVERSION

print(df_train[['CODE_GENDER','NAME_EDUCATION_TYPE','ORGANIZATION_TYPE']].head(3))

print(df_test[['CODE_GENDER','NAME_EDUCATION_TYPE','ORGANIZATION_TYPE']].head(3))
# CREATE NEW COLUMN : 0 - none, 1 - with car no realty, 2 - no car with realty, 3 - with car with realty

list_col_new_asset
def create_asset(df):

    df['FLAG_ASSET'] = np.nan

    filter_0 = (df.FLAG_OWN_CAR=='N')&(df.FLAG_OWN_REALTY=='N')

    filter_1 = (df.FLAG_OWN_CAR=='Y')&(df.FLAG_OWN_REALTY=='N')

    filter_2 = (df.FLAG_OWN_CAR=='N')&(df.FLAG_OWN_REALTY=='Y')

    filter_3 = (df.FLAG_OWN_CAR=='Y')&(df.FLAG_OWN_REALTY=='Y')

    

    df['FLAG_ASSET'][filter_0] = 0

    df['FLAG_ASSET'][filter_1] = 1

    df['FLAG_ASSET'][filter_2] = 2

    df['FLAG_ASSET'][filter_3] = 3

    return df



    

create_asset(df_train);

create_asset(df_test);
# SAMPLE OUTPUT

df_test[['FLAG_OWN_CAR','FLAG_OWN_REALTY','FLAG_ASSET']].head()
# SINCE WE NOW HAVE AN ASSET INDICATOR, WE CAN NOW REMOVE list_col_new_asset 

list_col_drop.extend(list_col_new_asset)
# CREATE NEW COLUMN : FLAG_CONTACTS

list_col_new_flagCont
def create_sum_cols(df,ls_cols,newcol):

    """ Sums up items per row across all columns.

        Returns df with new sum column.

    """

    df[newcol] = np.nan

    ls_sum_value = df[ls_cols].sum(axis=1)

    df[newcol] = ls_sum_value

    return df
create_sum_cols(df_train,list_col_new_flagCont,'FLAG_CONTACTS');

create_sum_cols(df_test,list_col_new_flagCont,'FLAG_CONTACTS');
df_test[['FLAG_CONTACTS']].head()
# CREATE NEW COLUMN : FLAG_DOCS

list_col_new_flagDoc
create_sum_cols(df_train,list_col_new_flagDoc,'FLAG_DOCS');

create_sum_cols(df_test,list_col_new_flagDoc,'FLAG_DOCS');
df_test[['FLAG_DOCS']].head()
# CREATE NEW COLUMN : FLAG_ADDR

list_col_new_flagAddr
create_sum_cols(df_train,list_col_new_flagAddr,'FLAG_ADDR');

create_sum_cols(df_test,list_col_new_flagAddr,'FLAG_ADDR');
df_test[['FLAG_ADDR']].head()
# SINCE WE NOW HAVE FLAG COMPLIANCE FIELDS, WE CAN NOW REMOVE EXISTING COLUMNS 

list_col_drop.extend(list_col_new_flagCont)

list_col_drop.extend(list_col_new_flagDoc)

list_col_drop.extend(list_col_new_flagAddr)
# CREATE NEW COLUMN : AGE

list_col_new_age 
def create_day_to_year(df,ls_cols,newcol):

    df[newcol] = round(np.abs(df[ls_cols[0]]/365))

    return df
create_day_to_year(df_train,list_col_new_age,'AGE');

create_day_to_year(df_test,list_col_new_age,'AGE');
f,ax=plt.subplots(figsize=(10,5))

sns.countplot(data=df_train[df_train.TARGET==1],x='AGE',hue='TARGET').set_title('Age of Individuals with default loans')

plt.xticks(rotation=90);
# CREATE NEW COLUMN : YEARS_EMPLOYED

list_col_new_yrsemp
create_day_to_year(df_train,list_col_new_yrsemp,'YEARS_EMPLOYED');

create_day_to_year(df_test,list_col_new_yrsemp,'YEARS_EMPLOYED');
df_test[['YEARS_EMPLOYED']].head()
# SINCE WE NOW HAVE NEW FEATURES, WE CAN NOW REMOVE EXISTING COLUMNS 

list_col_drop.extend(list_col_new_age)

list_col_drop.extend(list_col_new_yrsemp)
# CREATE NEW COLUMN : INCOME_BAND

list_col_new_income
# Create INCOME_BAND to group individuals per income range

def create_income_band(df):

    df.loc[(df.AMT_INCOME_TOTAL < 30000),'INCOME_BAND'] = 1

    df.loc[(df.AMT_INCOME_TOTAL >= 30000)&(df.AMT_INCOME_TOTAL < 65000),'INCOME_BAND'] = 2

    df.loc[(df.AMT_INCOME_TOTAL >= 65000)&(df.AMT_INCOME_TOTAL < 95000),'INCOME_BAND'] = 3

    df.loc[(df.AMT_INCOME_TOTAL >= 95000)&(df.AMT_INCOME_TOTAL < 130000),'INCOME_BAND'] = 4

    df.loc[(df.AMT_INCOME_TOTAL >= 130000)&(df.AMT_INCOME_TOTAL < 160000),'INCOME_BAND'] = 5

    df.loc[(df.AMT_INCOME_TOTAL >= 160000)&(df.AMT_INCOME_TOTAL < 190000),'INCOME_BAND'] = 6

    df.loc[(df.AMT_INCOME_TOTAL >= 190000)&(df.AMT_INCOME_TOTAL < 220000),'INCOME_BAND'] = 7

    df.loc[(df.AMT_INCOME_TOTAL >= 220000)&(df.AMT_INCOME_TOTAL < 275000),'INCOME_BAND'] = 8

    df.loc[(df.AMT_INCOME_TOTAL >= 275000)&(df.AMT_INCOME_TOTAL < 325000),'INCOME_BAND'] = 9

    df.loc[(df.AMT_INCOME_TOTAL >= 325000),'INCOME_BAND'] = 10

    return df
create_income_band(df_train);

create_income_band(df_test);
sns.countplot(data=df_train[df_train.TARGET==1],x='INCOME_BAND',hue='TARGET').set_title('Income of Individuals with default loans');
# CREATE NEW COLUMN : PERCENT_ANNUITY_INCOME

list_col_new_annuity
def create_perc(df,col1,col2,newcol):

    df[newcol] = round((df[col1]/df[col2])*100,2)

    df[newcol] = df[newcol].fillna(0)

    return df
create_perc(df_train,'AMT_ANNUITY','AMT_INCOME_TOTAL','PERCENT_ANNUITY_INCOME');

create_perc(df_test,'AMT_ANNUITY','AMT_INCOME_TOTAL','PERCENT_ANNUITY_INCOME');
df_test[['AMT_INCOME_TOTAL','AMT_ANNUITY','PERCENT_ANNUITY_INCOME']].head()
# CREATE NEW COLUMN : PERCENT_CREDIT_INCOME

list_col_new_credit
create_perc(df_train,'AMT_INCOME_TOTAL','AMT_CREDIT','PERCENT_CREDIT_INCOME');

create_perc(df_test,'AMT_INCOME_TOTAL','AMT_CREDIT','PERCENT_CREDIT_INCOME');
df_test[['AMT_INCOME_TOTAL','AMT_CREDIT','PERCENT_CREDIT_INCOME']].head()
# CREATE NEW COLUMN : EXP_CAR 

list_col_new_car
# Create EXP_CAR to group car age

def create_car_band(df):

    df.loc[(df.OWN_CAR_AGE == 0 ),'EXP_CAR'] = 1

    df.loc[(df.OWN_CAR_AGE >= 1)&(df.OWN_CAR_AGE < 4),'EXP_CAR'] = 2

    df.loc[(df.OWN_CAR_AGE >= 4)&(df.OWN_CAR_AGE < 8),'EXP_CAR'] = 3

    df.loc[(df.OWN_CAR_AGE >= 8)&(df.OWN_CAR_AGE < 11),'EXP_CAR'] = 4

    df.loc[(df.OWN_CAR_AGE >= 11)&(df.OWN_CAR_AGE < 15),'EXP_CAR'] = 5

    df.loc[(df.OWN_CAR_AGE >= 15)&(df.OWN_CAR_AGE < 20),'EXP_CAR'] = 6

    df.loc[(df.OWN_CAR_AGE >= 20 ),'EXP_CAR'] = 7

    df['EXP_CAR'] = df['EXP_CAR'].fillna(0)

    return df
create_car_band(df_train);

create_car_band(df_test);
sns.countplot(data=df_train[df_train.TARGET==1],x='EXP_CAR',hue='TARGET').set_title('Car Age band of Individuals with default loans');
# CREATE NEW COLUMN : EXP_HOUSE 

list_col_new_house;
df_train[list_col_new_house].describe()
list_col_new_house_copy = list_col_new_house.copy()

list_col_new_house_copy.extend(['TARGET'])

plt.subplots(figsize=(10,8))

sns.heatmap(df_train[list_col_new_house_copy].corr(),vmin=-1,vmax=1,cmap='BrBG');
# DROP THE MEDIAN AND MODE APARTMENT MEASUREMENTS SINCE ALL ARE HIGHLY CORRELATED. CREATE ONE FEATURE FOR THE REMAINING AVG MEASUREMENT.

list_col_new_house_avg = list_col_new_house[0:14].copy()
create_sum_cols(df_train,list_col_new_house_avg,'EXP_HOUSE');

create_sum_cols(df_test,list_col_new_house_avg,'EXP_HOUSE');
df_test[['EXP_HOUSE']].head()
# SINCE WE NOW HAVE NEW FEATURES, WE CAN NOW REMOVE EXSITING ONES 

list_col_drop.extend(list_col_new_income)  

list_col_drop.extend(list_col_new_annuity)

list_col_drop.extend(list_col_new_credit) 

list_col_drop.extend(list_col_new_car) 

list_col_drop.extend(list_col_new_house)  
def remove_dups(x):

  return list(dict.fromkeys(x))
print('Original column list count to drop:',len(list_col_drop)) #114
list_col_drop_new = remove_dups(list_col_drop)
print('Column list count to drop (duplicates removed):',len(list_col_drop_new)) #112
# UPDATED DATAFRAME

df_train.drop(list_col_drop_new,axis=1,inplace=True)

df_test.drop(list_col_drop_new,axis=1,inplace=True)



# PRINT UPDATED DATAFRAME

df_train
df_train.describe()
X = df_train.drop(columns='TARGET',axis=1)

y = df_train.TARGET



X_pred = df_test
X
X_pred
import time

import lightgbm as lgb

from lightgbm import LGBMClassifier

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold
folds = StratifiedKFold(n_splits=5)

oof_preds = np.zeros(X.shape[0])

sub_preds = np.zeros(X_pred.shape[0])
start = time.time()

valid_score = 0

for n_fold, (trn_idx, val_idx) in enumerate(folds.split(X, y)):

    trn_x, trn_y = X.iloc[trn_idx], y[trn_idx]

    val_x, val_y = X.iloc[val_idx], y[val_idx]    

    

    train_data = lgb.Dataset(data=trn_x, label=trn_y)

    valid_data = lgb.Dataset(data=val_x, label=val_y)

    

    param = {'application':'binary','num_iterations':4000, 'learning_rate':0.05, 'num_leaves':24, 'feature_fraction':0.8, 'bagging_fraction':0.9,

             'lambda_l1':0.1, 'lambda_l2':0.1, 'min_split_gain':0.01, 'early_stopping_round':100, 'max_depth':7, 'min_child_weight':40, 'metric':'auc'}

    lgb_es_model = lgb.train(param, train_data, valid_sets=[train_data, valid_data], verbose_eval=100) 

    

    oof_preds[val_idx] = lgb_es_model.predict(val_x, num_iteration=lgb_es_model.best_iteration)

    sub_preds += lgb_es_model.predict(X_pred, num_iteration=lgb_es_model.best_iteration) / folds.n_splits

    print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))

    valid_score += roc_auc_score(val_y, oof_preds[val_idx])



print('valid score:', str(round(valid_score/folds.n_splits,4)))



end = time.time()

print('training time:', str(round((end - start)/60)), 'mins')
lgb.plot_importance(lgb_es_model, height=0.5, max_num_features=20, ignore_zero = False, figsize = (12,6), importance_type ='gain')
# HOW TO REMOVE FILE

"""

if os.path.isfile('/kaggle/working/application_pred.csv'):

    os.remove('/kaggle/working/application_pred.csv')

    print("success")

else:    

    print("File doesn't exists!")

""";
# PREDICT!

application_test= pd.read_csv('/kaggle/input/home-credit-default-risk/application_test.csv')

output = pd.DataFrame({'SK_ID_CURR':application_test.SK_ID_CURR, 'TARGET': sub_preds})

output.to_csv('application_pred.csv', index=False)