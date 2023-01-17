# Supress Warnings
import warnings
warnings.filterwarnings('ignore')

# Import the numpy and pandas packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
# Let us read the Data one by one and understand the data given

# LOAD the CSV FILES USING READ_CSV
Application_data = pd.read_csv('C:\\Upgrad\\Assignment\\0826_2019_Credit_EDA_GroupCaseStudy\\application_data.csv')
PreviousApplication_data = pd.read_csv('C:\\Upgrad\\Assignment\\0826_2019_Credit_EDA_GroupCaseStudy\\previous_application.csv')

#Application_data = pd.read_csv('Downloads\\application_data.csv')
#PreviousApplication_data = pd.read_csv('Downloads\\previous_application.csv')
Application_data.head(5) # CHECK THE DATA FOR CURRENT APPLICATION DATA SET

PreviousApplication_data.head(5) # CHECK THE DATA FOR PREVIOUS APPLICATION DATA SET
# OBTAIN THE DIMENSION FOR THE CURRENT DATA SET
Application_data.shape
Application_data.info()
# OBTAIN THE DIMENSION FOR THE PREVIOUS DATA SET
PreviousApplication_data.shape
PreviousApplication_data.info()
# There are two sets of data is there. as we can see the coloumn SK_ID_CURR is same in both the data frames. 
# Let us index it both the data frames using pivot table functions. 

df1 = pd.pivot_table(Application_data,index=["SK_ID_CURR","NAME_CONTRACT_TYPE"])
df1.head(3)
df2 = pd.pivot_table(PreviousApplication_data,index=["SK_ID_CURR","NAME_CONTRACT_TYPE"])
df2.head(2)
# FIND THE NAN VALUE % IN EACH COLUMN
round(Application_data.isnull().mean(axis=0).sort_values(ascending=False)*100,2)
##################################
# CLEAN THE CURRENT DATA SET     #
##################################

# List the cells having less than 13.5% NAN values and store it to variable in descending order.
Df3 = Application_data.loc[:,round(Application_data.isnull().mean().sort_values(ascending=False)*100) < 13.5]

# List the columns and its mean percentage (%) of NAN values
round(Df3.isnull().mean(axis=0).sort_values(ascending=False)*100,2)

# LIST THE COLUMN NAMES IN THE CURRENT APPLICATION
Df3.columns

# MISCELLANEOUS COLUMNS THAT WILL BE DROPPED
curr_to_drop = ['OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE','OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE',
                'DAYS_LAST_PHONE_CHANGE', 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3','FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 
                'FLAG_DOCUMENT_6','FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9','FLAG_DOCUMENT_10', 
                'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12','FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15',
                'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18','FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 
                'FLAG_DOCUMENT_21', 'REGION_POPULATION_RELATIVE', 'FLAG_MOBIL',
                'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE',
                'FLAG_EMAIL', 'WEEKDAY_APPR_PROCESS_START', 'HOUR_APPR_PROCESS_START', 
                'REG_REGION_NOT_LIVE_REGION','REG_REGION_NOT_WORK_REGION', 
                'LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY',
                'LIVE_CITY_NOT_WORK_CITY', 'EXT_SOURCE_2']

# DROP THE COLUMNS THAT NEEDS TO BE EXCLUDED FROM ANALYSIS
Df3 = Df3.drop(curr_to_drop, axis=1)

# DIMENSION of PREVIOUS_APPLICATION DATAFRAME AFTER DROPPING COLUMNS
print(Df3.shape)

# # LIST THE COLUMN NAMES IN THE CURRENT APPLICATION AFTER DROPPING COLUMNS
Df3.columns
# CURRENT DATA SET --  VERIFY THE NAN VALUES PERCENTAGE (%) FOR EACH COLUMN 
round(Df3.isnull().mean(axis=0).sort_values(ascending=False)*100,2)

# CURRENT DATA SET --  CONVERT NUMBER OF DAYS INTO YEARS
round(Df3[['DAYS_BIRTH','DAYS_EMPLOYED','DAYS_REGISTRATION','DAYS_ID_PUBLISH']].abs()/365,2).head(15)

# WE NOTICE THAT 1000.67 IS A INFINITELY HIGH VALUE THAT CAN BE IGNORED
# CURRENT DATA SET --  CONVERT DAYS TO YEARS AND REMOVE THE NEGATIVE VALUE
# STORE THE REVISED VALUES IN THE DATAFRAME DF3 IN DENOMINATION OF "YEARS"

Df3[['DAYS_BIRTH','DAYS_EMPLOYED','DAYS_REGISTRATION','DAYS_ID_PUBLISH']] = round(Df3[['DAYS_BIRTH','DAYS_EMPLOYED','DAYS_REGISTRATION','DAYS_ID_PUBLISH']].abs()/(365),2)

# VERIFY THE NEW VALUES OF COLUMNS
Df3[['DAYS_BIRTH','DAYS_EMPLOYED','DAYS_REGISTRATION','DAYS_ID_PUBLISH']].head()

# CURRENT DATA SET -- FIND THE NAN VALUE PERCENTAGE IN THE ROWS
#(Df3.isnull().sum(axis=1).sort_values(ascending=False)/Df3.shape[0])*100)

(round(Df3.isnull().mean(axis=1).sort_values(ascending=False)*100,2)).head(4)
#########################################
# "PREVIOUS_APPLICANT" PROFILE ANALYSIS #
#########################################

# Check for the NAN values percentages
# Analysis the coloumns in the "PREVIOUS_Applicants" profile 

# List the cells having less than 22% NAN values and store it to variable in descending order.
Df2 = PreviousApplication_data.loc[:,round(PreviousApplication_data.isnull().mean().sort_values(ascending=False)*100) < 24]

# List the columns and its mean percentage (%) of NAN values
round(Df2.isnull().mean(axis=0).sort_values(ascending=False)*100,2)

# MISCELLANEOUS COLUMNS THAT WILL BE DROPPED
prev_to_drop = ['WEEKDAY_APPR_PROCESS_START', 'HOUR_APPR_PROCESS_START','FLAG_LAST_APPL_PER_CONTRACT', 
                'NFLAG_LAST_APPL_IN_DAY', 'NAME_PORTFOLIO', 'NAME_PRODUCT_TYPE', 'CHANNEL_TYPE']

# DROP THE COLUMNS THAT NEEDS TO BE EXCLUDED FROM ANALYSIS
Df2 = Df2.drop(prev_to_drop, axis=1)

# DIMENSION of PREVIOUS_APPLICATION DATAFRAME AFTER DROPPING COLUMNS
print(Df2.shape)

# List the columns after dropping columns
Df2.columns

Df2.head(3)

# FINAL CHECKS FOR NAN VALUES IN THE ROWS 
(round(Df2.isnull().mean(axis=1).sort_values(ascending=False)*100,2)).head(5)
# Now We have cleaned all the data sets of both the data sets. 

# FILTER THE CATEGORICAL VARIABLES BASED ON "OBJECT" DATA TYPE 
application_categorical = Df3.select_dtypes(include = ['object']).columns
application_categorical
# FILTER THE QUANTITATIVE VARIABLES BASED ON "OBJECT" DATA TYPE
application_quant = Df3.select_dtypes(include = ['int64', 'float64']).columns
application_quant
# CONVERT AND STOREG THE CATEGORICAL VARIABLES OF PREVIOUS APPLICATION
previous_categorical = Df2.select_dtypes(include = ['object']).columns
previous_categorical
# CONVERT AND STOREG THE QUANTITATIVE VARIABLES OF PREVIOUS APPLICATION
previous_quant = Df2.select_dtypes(include = ['int64', 'float64']).columns
previous_quant
Df3.describe() # DATA INSPECTION IN CURRENT APPLICATION
Df2.describe() # DATA INSPECTION IN PREVIOUS APPLICATION
Df3.columns # VERIFY THE COLUMNS
#############################################################################################
# PLOT A BOX CHART FOR THE CURRENT APPLICATION'S QUANTITATIVE VARIABLES TO SEE THE OUTLIERS #
#############################################################################################
appl_box = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE']
for i in Df3[appl_box]:
    plt.figure(1,figsize=(15,5))
    sns.boxplot(Df3[i])
    plt.xticks(rotation = 90,fontsize =10)
    plt.show()
# DATA CLEANING
# 1A. REMOVE OUTLIERS FROM QUANTITATIVE VARIABLES of CURRENT APPLICATION
test_box_df3 = ['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE', 'DAYS_EMPLOYED', 'DAYS_BIRTH']
new_copy = Df3[test_box_df3]

for i in new_copy.columns:
    Q1 = new_copy[i].quantile(0.25)
    Q3 = new_copy[i].quantile(0.75)

    IQR = Q3 - Q1
    
    
    lower_fence = Q1 - 1.5*IQR
    upper_fence = Q3 + 1.5*IQR
    
    new_copy[(new_copy.AMT_CREDIT <= upper_fence)].head()
    new_copy[i][new_copy[i] <= lower_fence] = lower_fence
    new_copy[i][new_copy[i] >= upper_fence] = upper_fence
    
    print("OUTLIERS:",i,lower_fence,upper_fence)


############################################################################
# PLOT A BOX CHART FOR THE "PREVIOUS" APPLICATION'S QUANTITATIVE VARIABLES #
############################################################################
prev_box = ['AMT_ANNUITY', 'AMT_APPLICATION', 'AMT_CREDIT', 'AMT_GOODS_PRICE', 'DAYS_DECISION', 'CNT_PAYMENT']
for i in Df2[prev_box]:
    plt.figure(1,figsize=(15,5))
    sns.boxplot(Df2[i])
    plt.xticks(rotation = 90,fontsize =10)
    plt.show()
# DATA CLEANING
# 1B. REMOVAL OF OUTLIERS FROM PREVIOUS DATASET ('DAYS_DECISION','CNT_PAYMENT')
test_box_Df2 = ['DAYS_DECISION','CNT_PAYMENT']
new_copy = Df2[test_box_Df2]
for i in new_copy.columns:
    Q1 = new_copy[i].quantile(0.25)
    Q3 = new_copy[i].quantile(0.75)

    IQR = Q3 - Q1
    
    lower_fence = Q1 - 1.5*IQR
    upper_fence = Q3 + 1.5*IQR

    new_copy[i][new_copy[i] <= lower_fence] = lower_fence
    new_copy[i][new_copy[i] >= upper_fence] = upper_fence
    
    print("oUTLIERS:",i,lower_fence,upper_fence)
    
    plt.figure(1,figsize=(10,5))
    sns.boxplot(new_copy[i])
    plt.xticks(rotation =90,fontsize =10)
    plt.show()

# 2A. COMPARITVE ANALYSIS OF CATEGORICAL VARIABLE (AMT_CREDIT) BEFORE
# AND AFTER CLEANING OUTLIERS IN CURRENT DATA SET

# CREATE A FIGURE
plt.figure(1, figsize = (10,5))

# CREATE A SUB PLOT 
plt.subplot(121)

# CREATE A BOX PLOT
boxplot = Df3.boxplot(['AMT_CREDIT'], notch=True, patch_artist=True)

# DEFINE A TITLE
boxplot.title.set_text('With Outliers')

# DEFINE FACE COLOR TO PINK
boxplot.set(facecolor = 'pink' )

# CREATE A SUB PLOT
plt.subplot(122)

# CREATE A BOX PLOT
boxplot = new_copy.boxplot(['AMT_CREDIT'], notch=True, patch_artist=True)

# DEFINE A TITLE
boxplot.title.set_text('Without Outliers')

# DEFINE FACE COLOR TO PINK
boxplot.set(facecolor = 'pink' )
plt.show()

Df3['CODE_GENDER'].value_counts()

min = Df3['AMT_CREDIT'].describe().min()
max = Df3['AMT_CREDIT'].describe().max()
range = [min, max]
print("Before removing:",range)

min = new_copy['AMT_CREDIT'].describe().min()
max = new_copy['AMT_CREDIT'].describe().max()
range = [min, max]
print("After removing:",range)

Df3['AMT_CREDIT'].describe()
# 2B. COMPARITVE ANALYSIS OF QUANTITATIVE VARIABLE (AMT_INCOME_TOTAL) BEFORE
# AND AFTER CLEANING OUTLIERS

# CREATE A FIGURE
plt.figure(1, figsize = (10,5))

# CREATE A SUB PLOT 
plt.subplot(121)

# CREATE A BOX PLOT
boxplot = Df3.boxplot(['AMT_INCOME_TOTAL'], notch=True, patch_artist=True)

# DEFINE A TITLE
boxplot.title.set_text('With Outliers')

# DEFINE FACE COLOR TO PINK
boxplot.set(facecolor = 'pink' )

# CREATE A SUB PLOT
plt.subplot(122)

# CREATE A BOX PLOT
boxplot = new_copy.boxplot(['AMT_INCOME_TOTAL'], notch=True, patch_artist=True)

# DEFINE A TITLE
boxplot.title.set_text('Without Outliers')

# DEFINE FACE COLOR TO PINK
boxplot.set(facecolor = 'pink' )

plt.show()

# 2.C COMPARITVE ANALYSIS OF QUANTITATIVE VARIABLE (AMT_ANNUITY) BEFORE
# AND AFTER CLEANING OUTLIERS

# CREATE A FIGURE
plt.figure(1, figsize = (10,5))

# CREATE A SUB PLOT 
plt.subplot(121)

# CREATE A BOX PLOT
boxplot = Df3.boxplot(['AMT_ANNUITY'], notch=True, patch_artist=True)

# DEFINE A TITLE
boxplot.title.set_text('With Outliers')

# DEFINE FACE COLOR TO PINK
boxplot.set(facecolor = 'pink' )

# CREATE A SUB PLOT
plt.subplot(122)

# CREATE A BOX PLOT
boxplot = new_copy.boxplot(['AMT_ANNUITY'], notch=True, patch_artist=True)

# DEFINE A TITLE
boxplot.title.set_text('Without Outliers')

# DEFINE FACE COLOR TO PINK
boxplot.set(facecolor = 'pink' )

plt.show()
#2.D COMPARITVE ANALYSIS OF QUANTITATIVE VARIABLE (DAYS_BIRTH) BEFORE
# AND AFTER CLEANING OUTLIERS

# CREATE A FIGURE
plt.figure(figsize = (10,5))

# CREATE A SUB PLOT 
plt.subplot(121)

# CREATE A BOX PLOT
boxplot = Df3.boxplot(['DAYS_BIRTH'], notch=True, patch_artist=True)

# DEFINE A TITLE
boxplot.title.set_text('With Outliers')

# DEFINE FACE COLOR TO PINK
boxplot.set(facecolor = 'pink' )

# CREATE A SUB PLOT
plt.subplot(122)

# CREATE A BOX PLOT
boxplot = new_copy.boxplot(['DAYS_BIRTH'], notch=True, patch_artist=True)

# DEFINE A TITLE
boxplot.title.set_text('Without Outliers')

# DEFINE FACE COLOR TO PINK
boxplot.set(facecolor = 'pink' )
plt.show()
#3A. UNIVAIATE ANALYSIS - SHOW THE DISTRIBUTION OF AGE
Df1_extract = Df3[['DAYS_BIRTH']]
Df1_extract.hist()
plt.show()
Df1_extract.describe()

# INFERENCE : 
# 1) THE AVERAGE AGE of a CLIENT IS AROUND 44 YEARS.
# 2) THE 

# UNIVAIATE ANALYSIS - SHOW THE DAYS_REGISTRATION
Df2_extract = Df3[['DAYS_REGISTRATION']]
# Binning of the continuous variables. 
Df2_extract
Df2_extract.hist()
plt.show()
# SAVE THE DATASET FOR TARGET VALUES EQUAL TO 0 AND 1 IN SEPERATE VARIABLES -- RUN WHEN ERROR IS ENCOUNTERD
Df_0 = Df3[Df3['TARGET'].isin([0])]
Df_1 = Df3[Df3['TARGET'].isin([1])]

plt.figure(figsize=(8,5))
plt.subplot(121); Df_0['TARGET'].value_counts().plot(kind='bar', color = ['C1']); plt.title("Clients who pay on time")
plt.subplot(122); Df_1['TARGET'].value_counts().plot(kind='bar', color = ['C2']); plt.title("Clients with loan payment Difficulty")

# VALIDATE THE CLIENTS WHO HAVE DIFFICULTY IN PAYMENT (0)
Df_0.head(3)
# VALIDATE THE CLIENTS WHO PAY ON TIME (1)
Df_1.head(3)

# Balancing the data check, which will help us to understand the imbalance percentage of the data sets.
# Imbalance %age = (# Rows where target=1/Toatal number of rows)*100

g = Df3['TARGET']
Df3_Imbalance = round(pd.concat([g.value_counts(),g.value_counts(normalize=True).mul(100)],axis=1,keys=('counts','percentage')),2)
Df3_Imbalance

# PLOT THE IMBALANCE COUNT AND PERCENTAGE FOR TARGET VALUES 0 AND 1.
Df3_TARGET = Df3_Imbalance.unstack()

plt.figure(figsize=(8,4))

# PLOT THE VALUES FOR 0 AND 1 BASED ON COUNT
plt.subplot(121); Df3_TARGET['counts'].plot(kind = 'bar'); plt.title("Count of Category 0 and 1")

# PLOT THE VALUES FOR 0 AND 1 BASED ON PERCENTAGE
plt.subplot(122); Df3_TARGET['percentage'].plot(kind = 'bar'); plt.title("Percentage % of Category 0 and 1"); plt.show()
# Show the counts and the percentage of data in the datasets
# finding the maximum in the columns percentage here, to check how much Imbalance is there.
f1=Df3_Imbalance.diff(periods=1,axis=0)
difvalue=Df3_Imbalance[[list(Df3_Imbalance.columns)[-1]]].max()
difvalue
# SHOW THE DISTRIBUTION OF AMT_INCOME_TOTAL
Df3_Income = Df3[['AMT_INCOME_TOTAL']]

# DEFINE THE BINS
bins = [0, 12500, 25000, 30000, 37500, 50000, 62500, 75000, 87500, 100000, 125000, 150000, 175000, 200000,225000,250000,275000,300000,325000,350000,375000,400000,425000, 450000,475000,500000,525000,550000,600000,650000,700000,750000,800000,850000,900000]

# PLOT A HISTOGRAM TO SEE THE DISTRIBUTION OF INCOME
Df3_Income.hist(bins= bins, range=[2.565000e+04,1.170000e+08])

plt.show()
Df3_Income.describe()

# SHOW THE DISTRIBUTION OF AMT_CREDIT
Df3_Amt_Credit = Df3[['AMT_CREDIT']]

# DEFINE THE BINS
bins = [0, 25000, 50000, 62500, 75000, 87500, 100000, 125000, 150000, 175000, 200000,225000,250000,275000,300000,325000,350000,375000,400000,425000, 450000,475000,500000,525000,550000,575000, 600000,625000, 650000,675000, 700000,725000, 750000, 775000, 800000,825000, 850000,875000, 900000]

# PLOT A HISTOGRAM TO SEE THE DISTRIBUTION OF INCOME
Df3_Amt_Credit.hist(bins= bins, range=[2.565000e+04,1.170000e+08])

plt.show()

Df3_Amt_Credit.describe()
# 3.A. UNIVARIATE ANALYSIS - AMT_INCOME_TOTAL
Df_1_AMT_INCOME_TOTAL = Df_1[['AMT_INCOME_TOTAL']]
Df_0_AMT_INCOME_TOTAL = Df_0[['AMT_INCOME_TOTAL']]

min = Df_1_AMT_INCOME_TOTAL.describe().min();max = Df_1_AMT_INCOME_TOTAL.describe().max()
range1=[min['AMT_INCOME_TOTAL'], max['AMT_INCOME_TOTAL']]
bins = [0, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000, 130000, 140000, 150000, 160000, 170000, 180000, 190000, 200000, 210000, 220000, 230000, 240000, 250000, 260000, 270000, 280000, 290000, 300000, 310000, 320000, 330000, 340000, 350000, 360000 ]
Df_1_AMT_INCOME_TOTAL.hist(bins=bins, range=range1, color = ['C1']); plt.title("Customer with Difficulty - AMT_INCOME_TOTAL")
plt.xticks(rotation = 90,fontsize =10)

min = Df_0_AMT_INCOME_TOTAL.describe().min(); max = Df_0_AMT_INCOME_TOTAL.describe().max()
range2=[min['AMT_INCOME_TOTAL'], max['AMT_INCOME_TOTAL']]
Df_0_AMT_INCOME_TOTAL.hist(bins=bins, range=range2, color = ['C2']); plt.title("Customer with No Difficulty - AMT_INCOME_TOTAL")
plt.xticks(rotation = 90,fontsize =10)
plt.show()

# INFERENCE:
# MOST OF THE LOAN DEFAULTS IS FOR CLIENTS WHICH INCOME BETWEEN 112000 TO 202000 

# 3.B. UNIVARIATE ANALYSIS - AMT_CREDIT
Df_1_AMT_CREDIT = Df_1[['AMT_CREDIT']]
Df_0_AMT_CREDIT = Df_0[['AMT_CREDIT']]
# plt.figure(figsize=(8,4))
min = Df_1_AMT_CREDIT.describe().min();max = Df_1_AMT_CREDIT.describe().max()
bins = [0, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000, 130000, 140000, 150000, 160000, 170000, 180000, 190000, 200000, 210000, 220000, 230000, 240000, 250000, 260000, 270000, 280000, 290000, 300000, 310000, 320000, 330000, 340000, 350000, 360000 ]
range1=[min['AMT_CREDIT'], max['AMT_CREDIT']]
Df_1_AMT_CREDIT.hist(bins=bins, range=range1, color = ['C1']); plt.title("Customer with Difficulty - AMT_CREDIT")
plt.xticks(rotation = 90,fontsize =10)

min = Df_0_AMT_CREDIT.describe().min(); max = Df_0_AMT_CREDIT.describe().max()
range2=[min['AMT_CREDIT'], max['AMT_CREDIT']]
Df_0_AMT_CREDIT.hist(bins=bins, range=range2, color = ['C2']); plt.title("Customer with No Difficulty - AMT_CREDIT")
plt.xticks(rotation = 90,fontsize =10)
plt.show()

# INFERENCE:
# MOST OF THE LOAN DEFAULTS IS FOR CLIENTS WHICH INCOME BETWEEN 285000 TO 733000 lOAN AMOUNTS 

# 3.C. UNIVARIATE ANALYSIS - AMT_ANNUITY
Df_1_AMT_ANNUITY = Df_1[['AMT_ANNUITY']]
Df_0_AMT_ANNUITY = Df_0[['AMT_ANNUITY']]
# plt.figure(figsize=(8,4))
min = Df_1_AMT_ANNUITY.describe().min();max = Df_1_AMT_ANNUITY.describe().max()
bins = [0, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000, 90000]
range1=[min['AMT_ANNUITY'], max['AMT_ANNUITY']]
Df_1_AMT_ANNUITY.hist(bins=bins, range=range1, color = ['C1']); plt.title("Customer with Difficulty")
plt.xticks(rotation = 90,fontsize =10)

min = Df_0_AMT_ANNUITY.describe().min(); max = Df_0_AMT_ANNUITY.describe().max()
range2=[min['AMT_ANNUITY'], max['AMT_ANNUITY']]
Df_0_AMT_ANNUITY.hist(bins=bins, range=range2, color = ['C2']); plt.title("Customer with No Difficulty")
plt.xticks(rotation = 90,fontsize =10)
plt.show()


# INFERENCE:
# MOST OF THE LOAN DEFAULTS IS FOR CLIENTS WHOSE ANNUITY IS BETWEEN 24000 TO 26000.


# KEEP IF NEEDED
#DISTRIBUTION OF APPROVED CREDIT WITH 
sns.distplot(Df_1_AMT_CREDIT['AMT_CREDIT'], kde=False).set_title("People with Difficulty")
plt.xticks(rotation='vertical');plt.show()
sns.distplot(Df_0_AMT_CREDIT['AMT_CREDIT'], kde=False).set_title("People without Difficulty")
plt.xticks(rotation='vertical');plt.show()
# SEGMENTED UNIVARIATE UNIVARIATE ANALYSIS 
# DISTRIBUTION OF BASED ON HOUSING TYPE

# PEOPLE WITH DIFFICULTY
plt.figure(figsize=(15,3))
sns.countplot(Df_1['NAME_HOUSING_TYPE']).set_title("People with Difficulty")
plt.show()

# PEOPLE WITHOUT DIFFICULTY
plt.figure(figsize=(15,3))
sns.countplot(Df_0['NAME_HOUSING_TYPE']).set_title("People without Difficulty")
plt.show()
Df_0.columns
# SEGMENTED UNIVARIATE UNIVARIATE ANALYSIS 
# DISTRIBUTION OF BASED ON FAMILY TYPE

# PEOPLE WITH DIFFICULTY
plt.figure(figsize=(15,3))
sns.countplot(Df_1['NAME_FAMILY_STATUS']).set_title("People with Difficulty")
plt.show()

# PEOPLE WITHOUT DIFFICULTY
plt.figure(figsize=(15,3))
sns.countplot(Df_0['NAME_FAMILY_STATUS']).set_title("People without Difficulty")
plt.show()

# SEGMENTED UNIVARIATE UNIVARIATE ANALYSIS 
# DISTRIBUTION OF BASED ON EDUCATION

# PEOPLE WITH DIFFICULTY
plt.figure(figsize=(15,3))
sns.countplot(Df_1['NAME_EDUCATION_TYPE']).set_title("People with Difficulty")
plt.show()

# PEOPLE WITHOUT DIFFICULTY
plt.figure(figsize=(15,3))
sns.countplot(Df_0['NAME_EDUCATION_TYPE']).set_title("People without Difficulty")
plt.show()

# SEGMENTED UNIVARIATE UNIVARIATE ANALYSIS 
# DISTRIBUTION OF BASED ON INCOME

# PEOPLE WITH DIFFICULTY
plt.figure(figsize=(15,3))
sns.countplot(Df_1['NAME_INCOME_TYPE']).set_title("People with Difficulty")
plt.show()

# PEOPLE WITHOUT DIFFICULTY
plt.figure(figsize=(15,3))
sns.countplot(Df_0['NAME_INCOME_TYPE']).set_title("People without Difficulty")
plt.show()
# SEGMENTED UNIVARIATE UNIVARIATE ANALYSIS 
# DISTRIBUTION OF BASED ON ORGANIZATION_TYPE

Df_1['ORGANIZATION_TYPE'] = Df_1['ORGANIZATION_TYPE'][Df_1['ORGANIZATION_TYPE'] != "XNA"]
Df_1['ORGANIZATION_TYPE']

# PEOPLE WITH DIFFICULTY
plt.figure(figsize=(15,10))
sns.countplot(Df_1['ORGANIZATION_TYPE']).set_title("People with Difficulty");plt.xticks(rotation='vertical')
plt.show()

# PEOPLE WITHOUT DIFFICULTY
Df_0_org = Df_0[['ORGANIZATION_TYPE']]

# DROP "XNA" VALUES AS IT IS EQUIVALENT TO NULL VALUE
Df_0_org['ORGANIZATION_TYPE'] = Df_0_org['ORGANIZATION_TYPE'][Df_0_org['ORGANIZATION_TYPE'] != "XNA"]
# DROP NULL VALUES
Df_0_org.dropna(axis=0, inplace=True)
Df_0_org
plt.figure(figsize=(15,6))
sns.countplot(Df_0_org['ORGANIZATION_TYPE']).set_title("People without Difficulty");plt.xticks(rotation='vertical')
plt.show()

# UNIVARIATE ANALYSIS
C_IT = sns.catplot("NAME_EDUCATION_TYPE", data=Df_1, aspect=1.5, kind="count", color="b")
C_IT.set_xticklabels(rotation=30)
plt.show()

C_IT = sns.catplot("NAME_EDUCATION_TYPE", data=Df_0, aspect=1.5, kind="count", color="b")
C_IT.set_xticklabels(rotation=30)
plt.show()
# BIVARIATE ANALYSIS

# JOINT PLOT between AMT_CREDIT AND AMT_INCOME_TOTAL
plt.figure(figsize=(25,25))
sns.jointplot('AMT_CREDIT','AMT_INCOME_TOTAL', Df_0);plt.xticks(rotation='horizontal')
plt.show()
# BIVARIATE ANALYSIS

# JOINT PLOT between AMT_CREDIT AND AMT_INCOME_TOTAL
plt.figure(figsize=(25,25))
sns.jointplot('AMT_CREDIT','AMT_INCOME_TOTAL', Df_0);plt.xticks(rotation='horizontal')
plt.show()
# JOINT PLOT between DAYS_BIRTH AND AMT_CREDIT
plt.figure(figsize=(25,25))
sns.jointplot('DAYS_BIRTH','AMT_CREDIT', Df_1);plt.xticks(rotation='vertical')
plt.show()

# JOINT PLOT between AMT_CREDIT AND AMT_INCOME_TOTAL
plt.figure(figsize=(25,25))
sns.jointplot('DAYS_BIRTH','AMT_CREDIT', Df_0);plt.xticks(rotation='vertical')
plt.show()
# BIVARIATE ANALYSIS
# JOINT PLOT between DAYS_EMPLOYED AND AMT_CREDIT


Df_1_DE = Df_1[['DAYS_EMPLOYED','AMT_CREDIT']]
Df_0_DE = Df_0[['DAYS_EMPLOYED','AMT_CREDIT']]

# PEOPLE WITHOUT DIFFICULTY
# DROP "1000.67" VALUES AS IT IS EQUIVALENT TO NULL VALUE
Df_1_DE['DAYS_EMPLOYED'] = Df_1_DE['DAYS_EMPLOYED'][Df_1_DE['DAYS_EMPLOYED'] != 1000.67]

# DROP NULL VALUES
Df_1_DE.dropna(axis=0, inplace=True)
Df_1_DE

sns.jointplot('DAYS_EMPLOYED','AMT_CREDIT', Df_1_DE);plt.xticks(rotation='vertical')
plt.show()

# PEOPLE WITH DIFFICULTY
# DROP "1000.67" VALUES AS IT IS EQUIVALENT TO NULL VALUE
Df_0_DE['DAYS_EMPLOYED'] = Df_0_DE['DAYS_EMPLOYED'][Df_0_DE['DAYS_EMPLOYED'] != 1000.67]

# DROP NULL VALUES
Df_0_DE.dropna(axis=0, inplace=True)
Df_0_DE

# JOINT PLOT between AMT_CREDIT AND AMT_INCOME_TOTAL

sns.jointplot('DAYS_EMPLOYED','AMT_CREDIT', Df_0_DE);plt.xticks(rotation='vertical')
plt.show()
# JOINT PLOT between CNT_CHILDREN AND AMT_CREDIT

Df_1_CC = Df_1[['CNT_CHILDREN','AMT_CREDIT']]
Df_0_CC = Df_0[['CNT_CHILDREN','AMT_CREDIT']]

# PEOPLE WITHOUT DIFFICULTY
# DROP NULL VALUES
Df_1_CC.dropna(axis=0, inplace=True)

sns.jointplot('CNT_CHILDREN','AMT_CREDIT', Df_1_CC);plt.xticks(rotation='vertical')
plt.title("CLIENTS WITHOUT DIFFICULTY")
plt.show()

# PEOPLE WITH DIFFICULTY
# DROP NULL VALUES
Df_0_CC.dropna(axis=0, inplace=True)

# JOINT PLOT between AMT_CREDIT AND AMT_INCOME_TOTAL

sns.jointplot('CNT_CHILDREN','AMT_CREDIT', Df_0_CC);plt.xticks(rotation='vertical')
plt.title("CLIENTS WITH DIFFICULTY")
plt.show()

# HEAT MAP - TO TEST CORRELATION BETWEEN CRITICAL QUANTIATIVE VALUES
# IN THE CURRENT APPLICATION SET FOR CLIENTS HAVING DIFFICULTY IN PAYMENT
test=Df_1[['AMT_INCOME_TOTAL','AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE','DAYS_BIRTH','CNT_CHILDREN','CNT_FAM_MEMBERS']].corr()
plt.figure(figsize=(15,8))
sns.heatmap(test,annot = True, fmt = ".2f", cmap = "GnBu")
plt.show()

# HEAT MAP - TO TEST CORRELATION BETWEEN CRITICAL QUANTIATIVE VALUES
# IN THE CURRENT APPLICATION SET FOR CLIENTS WHO PAY ON TIME

test=Df_0[['AMT_INCOME_TOTAL','AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE','DAYS_BIRTH','CNT_CHILDREN','CNT_FAM_MEMBERS']].corr()
plt.figure(figsize=(15,8))
sns.heatmap(test,annot = True, fmt = ".2f", cmap = "GnBu")
plt.show()
# BIVARIATE 
plt.figure(figsize=(15,8))
sns.lineplot(x="AMT_INCOME_TOTAL", y="AMT_CREDIT", data=Df_0)
plt.show()
# BIVARIATE ANALYSIS FOR PREVIOUS Application Data
Df_prev = PreviousApplication_data

# LIST DOWN THE COLUMNS
Df_prev.columns

# FIND THE NAN VALUE % IN EACH COLUMN
round(Df_prev.isnull().mean(axis=0).sort_values(ascending=False)*100,2)
Df_prev.columns

# List the cells having less than 55% NAN values and store it to variable in descending order.
Df_prev = PreviousApplication_data.loc[:,round(PreviousApplication_data.isnull().mean().sort_values(ascending=False)*100) < 60]

# List the columns and its mean percentage (%) of NAN values
round(Df_prev.isnull().mean(axis=0).sort_values(ascending=False)*100,2)

# MISCELLANEOUS COLUMNS THAT WILL BE DROPPED
prev_to_drop = ['WEEKDAY_APPR_PROCESS_START', 'HOUR_APPR_PROCESS_START','FLAG_LAST_APPL_PER_CONTRACT', 'NFLAG_LAST_APPL_IN_DAY',
                'CHANNEL_TYPE', 'SELLERPLACE_AREA', 'NAME_SELLER_INDUSTRY','NAME_YIELD_GROUP','PRODUCT_COMBINATION']

# DROP THE COLUMNS THAT NEEDS TO BE EXCLUDED FROM ANALYSIS
Df_prev = Df_prev.drop(prev_to_drop, axis=1)

# LIST THE COLUMN NAMES IN THE CURRENT APPLICATION AFTER DROPPING COLUMNS
Df_prev.columns

# CHANGE NEGATIVE VALUE TO ABSOLUTE VALUE
Df_prev['DAYS_DECISION'] = round(Df_prev['DAYS_DECISION'].abs(),2).head(15)
Df_prev.head(5)

Df_prev.columns
# PLOT A BOX CHART FOR THE CURRENT APPLICATION'S QUANTITATIVE VARIABLES TO SEE THE OUTLIERS
prev_box_2 = ['AMT_APPLICATION','AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE','DAYS_DECISION']
for i in Df_prev[prev_box_2]:
    plt.figure(1,figsize=(15,5))
    sns.boxplot(Df_prev[i])
    plt.xticks(rotation = 90,fontsize =10)
    plt.show()
# DATA CLEANSING
# REMOVE OF OUTLIERS FROM PREVIOUS DATASET 
prev_box_Df2 = ['AMT_APPLICATION','AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE','DAYS_DECISION']
clean_prev_Df2 = Df_prev[prev_box_Df2]
for i in clean_prev_Df2.columns:
    Q1 = clean_prev_Df2[i].quantile(0.25)
    Q3 = clean_prev_Df2[i].quantile(0.75)

    IQR = Q3 - Q1
    
    lower_fence = Q1 - 1.5*IQR
    upper_fence = Q3 + 1.5*IQR

    clean_prev_Df2[i][clean_prev_Df2[i] <= lower_fence] = lower_fence
    clean_prev_Df2[i][clean_prev_Df2[i] >= upper_fence] = upper_fence
    
    print(lower_fence,upper_fence)
    
    plt.figure(1,figsize=(10,5))
    sns.boxplot(clean_prev_Df2[i])
    plt.xticks(rotation =90,fontsize =10)
    plt.show()

# CLEANED DATA COLUMNS FROM PREVIOUS APPLICATION
print(clean_prev_Df2.columns)

# CLEANED DATA FROM PREVIOUS APPLICATION
clean_prev_Df2.head(4)
# UNIVARIATE ANALYSIS - AMT_APPLICATION
clean_prev_Df_AMT_APPLICATION = clean_prev_Df2[['AMT_APPLICATION']]

min = clean_prev_Df_AMT_APPLICATION.describe().min();max = clean_prev_Df_AMT_APPLICATION.describe().max()
range1=[min['AMT_APPLICATION'], max['AMT_APPLICATION']]
clean_prev_Df_AMT_APPLICATION.hist(bins=20, range=range1, color = ['C1']); plt.title("Distribution of AMT_APPLICATION")
plt.xticks(rotation = 90,fontsize =10)
plt.show()

# UNIVARIATE ANALYSIS - AMT_CREDIT
clean_prev_Df_AMT_CREDIT = clean_prev_Df2[['AMT_CREDIT']]
min = clean_prev_Df_AMT_CREDIT.describe().min();max = clean_prev_Df_AMT_CREDIT.describe().max()
range1=[min['AMT_CREDIT'], max['AMT_CREDIT']]
clean_prev_Df_AMT_CREDIT.hist(bins=20, range=range1, color = ['C2']); plt.title("Distribution of AMT_CREDIT")
plt.xticks(rotation = 90,fontsize =10)
plt.show()
# UNIVARIATE ANALYSIS - AMT_ANNUITY
clean_prev_Df_AMT_ANNUITY = clean_prev_Df2[['AMT_ANNUITY']]

min = clean_prev_Df_AMT_ANNUITY.describe().min();max = clean_prev_Df_AMT_ANNUITY.describe().max()
range1=[min['AMT_ANNUITY'], max['AMT_ANNUITY']]
bins1=[0,6250, 12500, 18750, 25000,31250, 37500, 43750, 50000, 56250]
clean_prev_Df_AMT_ANNUITY.hist(bins=bins1, range=range1, color = ['C3']); plt.title("Distribution of AMT_ANNUITY")
plt.xticks(rotation = 90,fontsize =10)
plt.show()

# UNIVARIATE ANALYSIS - AMT_GOODS_PRICE
clean_prev_Df_AMT_GOODS_PRICE = clean_prev_Df2[['AMT_GOODS_PRICE']]
bins=[0,5000,10000,15000,20000,25000,30000,35000,40000,45000,50000,55000,60000,65000,70000,75000,80000,85000,90000,95000,100000,105000,110000,115000,120000,125000, 130000,135000,140000,145000, 150000, 155000, 160000, 170000, 175000, 180000, 185000, 190000, 195000, 200000, 205000, 210000, 215000, 220000, 225000, 230000, 235000, 240000, 245000, 250000, 255000,260000]
min = clean_prev_Df_AMT_GOODS_PRICE.describe().min();max = clean_prev_Df_AMT_GOODS_PRICE.describe().max()
range1=[min['AMT_GOODS_PRICE'], max['AMT_GOODS_PRICE']]
clean_prev_Df_AMT_GOODS_PRICE.hist(bins=bins, range=range1, color = ['C4']); plt.title("Distribution of AMT_GOODS_PRICE")
plt.xticks(rotation = 90,fontsize =10)
plt.show()

# UNIVARIATE ANALYSIS - DAYS_DECISION
clean_prev_Df_DAYS_DECISION = abs(clean_prev_Df2[['DAYS_DECISION']])
min = clean_prev_Df_DAYS_DECISION.describe().min();max = clean_prev_Df_DAYS_DECISION.describe().max()
bins1=[0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220, 230, 240,250,260,270,280,290,300]
range1=[min['DAYS_DECISION'], max['DAYS_DECISION']]
clean_prev_Df_DAYS_DECISION.hist(bins=bins1, range=range1, color = ['C5']); plt.title("Distribution of DAYS_DECISION")
plt.xticks(rotation = 90,fontsize =10)
plt.show()

# BIVARIATE ANALYSIS FOR PREVIOUS APPLICATION DATA BASED ON CONTRACT_STATUS
plt.figure(figsize=(15,5))
sns.barplot(x="NAME_CONTRACT_STATUS", y="AMT_ANNUITY",  data=Df_prev)
plt.show()
# BIVARIATE ANALYSIS FOR PREVIOUS APPLICATION DATA BASED ON CONTRACT_STATUS

plt.figure(figsize=(15,5))
sns.barplot(x="NAME_CONTRACT_STATUS", y="AMT_APPLICATION",  data=Df_prev)
plt.show()

# BIVARIATE ANALYSIS FOR PREVIOUS APPLICATION DATA BASED ON CONTRACT_STATUS

plt.figure(figsize=(15,5))
sns.barplot(x="NAME_CONTRACT_STATUS", y="AMT_CREDIT",  data=Df_prev)
plt.show()
plt.figure(figsize=(15,5))
sns.barplot(x="NAME_CONTRACT_STATUS", y="AMT_GOODS_PRICE",  data=Df_prev)
plt.show()

Tot_contract_status = Df_prev['NAME_CONTRACT_STATUS'].shape[0]
Can_length = len(Df_prev['NAME_CONTRACT_STATUS'][Df_prev['NAME_CONTRACT_STATUS'] == "Canceled"])
Can_length_pct = round((Can_length/Tot_contract_status)*100,2)
Can_length_pct
Can_length,Tot_contract_status,Can_length_pct

# INFERENCE: 19% of loans have been cancelled by the customer.
Tot_contract_status = Df_prev['NAME_CONTRACT_STATUS'].shape[0]
Can_length = len(Df_prev['NAME_CONTRACT_STATUS'][Df_prev['NAME_CONTRACT_STATUS'] == "Canceled"])
Can_length_pct = round((Can_length/Tot_contract_status)*100,2)
Can_length_pct
Can_length,Tot_contract_status,Can_length_pct
# HEAT MAP - TO TEST CORRELATION BETWEEN CRITICAL QUANTIATIVE VALUES
# IN THE PREVIOUS APPLICATION SET FOR CLIENTS WHO PAY ON TIME

test=Df_prev[['AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'AMT_DOWN_PAYMENT','DAYS_DECISION','CNT_PAYMENT']].corr()
plt.figure(figsize=(15,8))
sns.heatmap(test,annot = True, fmt = ".2f", cmap = "GnBu")
plt.show()

# INFERENCE : The loan amount sanctioned has a strong correlation with the AMT_GOODS_PRICE and AMT_ANNUITY
Df_prev.columns
# HEAT MAP - TO TEST CORRELATION BETWEEN CRITICAL QUANTIATIVE VALUES
# IN THE PREVIOUS APPLICATION SET FOR CLIENTS WHO PAY ON TIME

test=Df_prev[['AMT_CREDIT','DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION','DAYS_LAST_DUE', 'DAYS_TERMINATION']].corr()
plt.figure(figsize=(15,8))
sns.heatmap(test,annot = True, fmt = ".2f", cmap = "GnBu")
plt.show()

# INFERENCE : The first drawing, First due, Last Due and Last termination has "NO" bearing to the saction loan amount.




