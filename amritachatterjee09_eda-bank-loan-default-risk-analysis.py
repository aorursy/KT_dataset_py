import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.style as style

import seaborn as sns

import itertools

%matplotlib inline



# setting up plot style 

style.use('seaborn-poster')

style.use('fivethirtyeight')
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)

pd.set_option('display.expand_frame_repr', False)
# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
applicationDF = pd.read_csv(r'/kaggle/input/loan-defaulter/application_data.csv')

previousDF = pd.read_csv('/kaggle/input/loan-defaulter/previous_application.csv')

applicationDF.head()
previousDF.head()
# Database dimension

print("Database dimension - applicationDF     :",applicationDF.shape)

print("Database dimension - previousDF        :",previousDF.shape)



#Database size

print("Database size - applicationDF          :",applicationDF.size)

print("Database size - previousDF             :",previousDF.size)
# Database column types

applicationDF.info(verbose=True)
previousDF.info(verbose=True)
# Checking the numeric variables of the dataframes

applicationDF.describe()
previousDF.describe()
import missingno as mn

mn.matrix(applicationDF)
# % null value in each column

round(applicationDF.isnull().sum() / applicationDF.shape[0] * 100.00,2)
null_applicationDF = pd.DataFrame((applicationDF.isnull().sum())*100/applicationDF.shape[0]).reset_index()

null_applicationDF.columns = ['Column Name', 'Null Values Percentage']

fig = plt.figure(figsize=(18,6))

ax = sns.pointplot(x="Column Name",y="Null Values Percentage",data=null_applicationDF,color='blue')

plt.xticks(rotation =90,fontsize =7)

ax.axhline(40, ls='--',color='red')

plt.title("Percentage of Missing values in application data")

plt.ylabel("Null Values PERCENTAGE")

plt.xlabel("COLUMNS")

plt.show()
# more than or equal to 40% empty rows columns

nullcol_40_application = null_applicationDF[null_applicationDF["Null Values Percentage"]>=40]

nullcol_40_application
# How many columns have more than or euqal to 40% null values ?

len(nullcol_40_application)
mn.matrix(previousDF)
# checking the null value % of each column in previousDF dataframe

round(previousDF.isnull().sum() / previousDF.shape[0] * 100.00,2)
null_previousDF = pd.DataFrame((previousDF.isnull().sum())*100/previousDF.shape[0]).reset_index()

null_previousDF.columns = ['Column Name', 'Null Values Percentage']

fig = plt.figure(figsize=(18,6))

ax = sns.pointplot(x="Column Name",y="Null Values Percentage",data=null_previousDF,color ='blue')

plt.xticks(rotation =90,fontsize =7)

ax.axhline(40, ls='--',color='red')

plt.title("Percentage of Missing values in previousDF data")

plt.ylabel("Null Values PERCENTAGE")

plt.xlabel("COLUMNS")

plt.show()
# more than or equal to 40% empty rows columns

nullcol_40_previous = null_previousDF[null_previousDF["Null Values Percentage"]>=40]

nullcol_40_previous
# How many columns have more than or euqal to 40% null values ?

len(nullcol_40_previous)
# Checking correlation of EXT_SOURCE_X columns vs TARGET column

Source = applicationDF[["EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3","TARGET"]]

source_corr = Source.corr()

ax = sns.heatmap(source_corr,

            xticklabels=source_corr.columns,

            yticklabels=source_corr.columns,

            annot = True,

            cmap ="RdYlGn")
# create a list of columns that needs to be dropped including the columns with >40% null values

Unwanted_application = nullcol_40_application["Column Name"].tolist()+ ['EXT_SOURCE_2','EXT_SOURCE_3'] 

# as EXT_SOURCE_1 column is already included in nullcol_40_application 

len(Unwanted_application)
# Checking the relevance of Flag_Document and whether it has any relation with loan repayment status

col_Doc = [ 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3','FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6','FLAG_DOCUMENT_7', 

           'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9','FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12','FLAG_DOCUMENT_13',

           'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15','FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18',

           'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21']

df_flag = applicationDF[col_Doc+["TARGET"]]



length = len(col_Doc)



df_flag["TARGET"] = df_flag["TARGET"].replace({1:"Defaulter",0:"Repayer"})



fig = plt.figure(figsize=(21,24))



for i,j in itertools.zip_longest(col_Doc,range(length)):

    plt.subplot(5,4,j+1)

    ax = sns.countplot(df_flag[i],hue=df_flag["TARGET"],palette=["r","g"])

    plt.yticks(fontsize=8)

    plt.xlabel("")

    plt.ylabel("")

    plt.title(i)
# Including the flag documents for dropping the Document columns

col_Doc.remove('FLAG_DOCUMENT_3') 

Unwanted_application = Unwanted_application + col_Doc

len(Unwanted_application)
# checking is there is any correlation between mobile phone, work phone etc, email, Family members and Region rating

contact_col = ['FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE',

       'FLAG_PHONE', 'FLAG_EMAIL','TARGET']

Contact_corr = applicationDF[contact_col].corr()

fig = plt.figure(figsize=(8,8))

ax = sns.heatmap(Contact_corr,

            xticklabels=Contact_corr.columns,

            yticklabels=Contact_corr.columns,

            annot = True,

            cmap ="RdYlGn",

            linewidth=1)
# including the 6 FLAG columns to be deleted

contact_col.remove('TARGET') 

Unwanted_application = Unwanted_application + contact_col

len(Unwanted_application)
# Dropping the unnecessary columns from applicationDF

applicationDF.drop(labels=Unwanted_application,axis=1,inplace=True)
# Inspecting the dataframe after removal of unnecessary columns

applicationDF.shape
# inspecting the column types after removal of unnecessary columns

applicationDF.info()
# Getting the 11 columns which has more than 40% unknown

Unwanted_previous = nullcol_40_previous["Column Name"].tolist()

Unwanted_previous
# Listing down columns which are not needed

Unnecessary_previous = ['WEEKDAY_APPR_PROCESS_START','HOUR_APPR_PROCESS_START',

                        'FLAG_LAST_APPL_PER_CONTRACT','NFLAG_LAST_APPL_IN_DAY']
Unwanted_previous = Unwanted_previous + Unnecessary_previous

len(Unwanted_previous)
# Dropping the unnecessary columns from previous

previousDF.drop(labels=Unwanted_previous,axis=1,inplace=True)

# Inspecting the dataframe after removal of unnecessary columns

previousDF.shape
# inspecting the column types after after removal of unnecessary columns

previousDF.info()
# Converting Negative days to positive days



date_col = ['DAYS_BIRTH','DAYS_EMPLOYED','DAYS_REGISTRATION','DAYS_ID_PUBLISH']



for col in date_col:

    applicationDF[col] = abs(applicationDF[col])
# Binning Numerical Columns to create a categorical column



# Creating bins for income amount

applicationDF['AMT_INCOME_TOTAL']=applicationDF['AMT_INCOME_TOTAL']/100000



bins = [0,1,2,3,4,5,6,7,8,9,10,11]

slot = ['0-100K','100K-200K', '200k-300k','300k-400k','400k-500k','500k-600k','600k-700k','700k-800k','800k-900k','900k-1M', '1M Above']



applicationDF['AMT_INCOME_RANGE']=pd.cut(applicationDF['AMT_INCOME_TOTAL'],bins,labels=slot)

applicationDF['AMT_INCOME_RANGE'].value_counts(normalize=True)*100
# Creating bins for Credit amount

applicationDF['AMT_CREDIT']=applicationDF['AMT_CREDIT']/100000



bins = [0,1,2,3,4,5,6,7,8,9,10,100]

slots = ['0-100K','100K-200K', '200k-300k','300k-400k','400k-500k','500k-600k','600k-700k','700k-800k',

       '800k-900k','900k-1M', '1M Above']



applicationDF['AMT_CREDIT_RANGE']=pd.cut(applicationDF['AMT_CREDIT'],bins=bins,labels=slots)
#checking the binning of data and % of data in each category

applicationDF['AMT_CREDIT_RANGE'].value_counts(normalize=True)*100
# Creating bins for Age

applicationDF['AGE'] = applicationDF['DAYS_BIRTH'] // 365

bins = [0,20,30,40,50,100]

slots = ['0-20','20-30','30-40','40-50','50 above']



applicationDF['AGE_GROUP']=pd.cut(applicationDF['AGE'],bins=bins,labels=slots)
#checking the binning of data and % of data in each category

applicationDF['AGE_GROUP'].value_counts(normalize=True)*100
# Creating bins for Employement Time

applicationDF['YEARS_EMPLOYED'] = applicationDF['DAYS_EMPLOYED'] // 365

bins = [0,5,10,20,30,40,50,60,150]

slots = ['0-5','5-10','10-20','20-30','30-40','40-50','50-60','60 above']



applicationDF['EMPLOYMENT_YEAR']=pd.cut(applicationDF['YEARS_EMPLOYED'],bins=bins,labels=slots)
#checking the binning of data and % of data in each category

applicationDF['EMPLOYMENT_YEAR'].value_counts(normalize=True)*100
#Checking the number of unique values each column possess to identify categorical columns

applicationDF.nunique().sort_values()
# inspecting the column types if they are in correct data type using the above result.

applicationDF.info()
#Conversion of Object and Numerical columns to Categorical Columns

categorical_columns = ['NAME_CONTRACT_TYPE','CODE_GENDER','NAME_TYPE_SUITE','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE',

                       'NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','OCCUPATION_TYPE','WEEKDAY_APPR_PROCESS_START',

                       'ORGANIZATION_TYPE','FLAG_OWN_CAR','FLAG_OWN_REALTY','LIVE_CITY_NOT_WORK_CITY',

                       'REG_CITY_NOT_LIVE_CITY','REG_CITY_NOT_WORK_CITY','REG_REGION_NOT_WORK_REGION',

                       'LIVE_REGION_NOT_WORK_REGION','REGION_RATING_CLIENT','WEEKDAY_APPR_PROCESS_START',

                       'REGION_RATING_CLIENT_W_CITY'

                      ]

for col in categorical_columns:

    applicationDF[col] =pd.Categorical(applicationDF[col])
# inspecting the column types if the above conversion is reflected

applicationDF.info()
#Checking the number of unique values each column possess to identify categorical columns

previousDF.nunique().sort_values() 
# inspecting the column types if the above conversion is reflected

previousDF.info()
#Converting negative days to positive days 

previousDF['DAYS_DECISION'] = abs(previousDF['DAYS_DECISION'])
#age group calculation e.g. 388 will be grouped as 300-400

previousDF['DAYS_DECISION_GROUP'] = (previousDF['DAYS_DECISION']-(previousDF['DAYS_DECISION'] % 400)).astype(str)+'-'+ ((previousDF['DAYS_DECISION'] - (previousDF['DAYS_DECISION'] % 400)) + (previousDF['DAYS_DECISION'] % 400) + (400 - (previousDF['DAYS_DECISION'] % 400))).astype(str)

previousDF['DAYS_DECISION_GROUP'].value_counts(normalize=True)*100
#Converting Categorical columns from Object to categorical 

Catgorical_col_p = ['NAME_CASH_LOAN_PURPOSE','NAME_CONTRACT_STATUS','NAME_PAYMENT_TYPE',

                    'CODE_REJECT_REASON','NAME_CLIENT_TYPE','NAME_GOODS_CATEGORY','NAME_PORTFOLIO',

                   'NAME_PRODUCT_TYPE','CHANNEL_TYPE','NAME_SELLER_INDUSTRY','NAME_YIELD_GROUP','PRODUCT_COMBINATION',

                    'NAME_CONTRACT_TYPE','DAYS_DECISION_GROUP']



for col in Catgorical_col_p:

    previousDF[col] =pd.Categorical(previousDF[col])
# inspecting the column types after conversion

previousDF.info()
# checking the null value % of each column in applicationDF dataframe

round(applicationDF.isnull().sum() / applicationDF.shape[0] * 100.00,2)

applicationDF['NAME_TYPE_SUITE'].describe()
applicationDF['NAME_TYPE_SUITE'].fillna((applicationDF['NAME_TYPE_SUITE'].mode()[0]),inplace = True)


applicationDF['OCCUPATION_TYPE'] = applicationDF['OCCUPATION_TYPE'].cat.add_categories('Unknown')

applicationDF['OCCUPATION_TYPE'].fillna('Unknown', inplace =True) 
applicationDF[['AMT_REQ_CREDIT_BUREAU_HOUR','AMT_REQ_CREDIT_BUREAU_DAY',

               'AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_MON',

               'AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_YEAR']].describe()
amount = ['AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_MON',

         'AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_YEAR']



for col in amount:

    applicationDF[col].fillna(applicationDF[col].median(),inplace = True)
# checking the null value % of each column in previousDF dataframe

round(applicationDF.isnull().sum() / previousDF.shape[0] * 100.00,2)
# checking the null value % of each column in previousDF dataframe

round(previousDF.isnull().sum() / previousDF.shape[0] * 100.00,2)
plt.figure(figsize=(6,6))

sns.kdeplot(previousDF['AMT_ANNUITY'])

plt.show()
previousDF['AMT_ANNUITY'].fillna(previousDF['AMT_ANNUITY'].median(),inplace = True)
plt.figure(figsize=(6,6))

sns.kdeplot(previousDF['AMT_GOODS_PRICE'][pd.notnull(previousDF['AMT_GOODS_PRICE'])])

plt.show()
statsDF = pd.DataFrame() # new dataframe with columns imputed with mode, median and mean

statsDF['AMT_GOODS_PRICE_mode'] = previousDF['AMT_GOODS_PRICE'].fillna(previousDF['AMT_GOODS_PRICE'].mode()[0])

statsDF['AMT_GOODS_PRICE_median'] = previousDF['AMT_GOODS_PRICE'].fillna(previousDF['AMT_GOODS_PRICE'].median())

statsDF['AMT_GOODS_PRICE_mean'] = previousDF['AMT_GOODS_PRICE'].fillna(previousDF['AMT_GOODS_PRICE'].mean())



cols = ['AMT_GOODS_PRICE_mode', 'AMT_GOODS_PRICE_median','AMT_GOODS_PRICE_mean']



plt.figure(figsize=(18,10))

plt.suptitle('Distribution of Original data vs imputed data')

plt.subplot(221)

sns.distplot(previousDF['AMT_GOODS_PRICE'][pd.notnull(previousDF['AMT_GOODS_PRICE'])]);

for i in enumerate(cols): 

    plt.subplot(2,2,i[0]+2)

    sns.distplot(statsDF[i[1]])
previousDF['AMT_GOODS_PRICE'].fillna(previousDF['AMT_GOODS_PRICE'].mode()[0], inplace=True)
previousDF.loc[previousDF['CNT_PAYMENT'].isnull(),'NAME_CONTRACT_STATUS'].value_counts()
previousDF['CNT_PAYMENT'].fillna(0,inplace = True)
# checking the null value % of each column in previousDF dataframe

round(previousDF.isnull().sum() / previousDF.shape[0] * 100.00,2)
plt.figure(figsize=(22,10))



app_outlier_col_1 = ['AMT_ANNUITY','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_GOODS_PRICE','DAYS_EMPLOYED']

app_outlier_col_2 = ['CNT_CHILDREN','DAYS_BIRTH']

for i in enumerate(app_outlier_col_1):

    plt.subplot(2,4,i[0]+1)

    sns.boxplot(y=applicationDF[i[1]])

    plt.title(i[1])

    plt.ylabel("")



for i in enumerate(app_outlier_col_2):

    plt.subplot(2,4,i[0]+6)

    sns.boxplot(y=applicationDF[i[1]])

    plt.title(i[1])

    plt.ylabel("")
applicationDF[['AMT_ANNUITY', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_GOODS_PRICE', 'DAYS_BIRTH','CNT_CHILDREN','DAYS_EMPLOYED']].describe()
plt.figure(figsize=(22,8))



prev_outlier_col_1 = ['AMT_ANNUITY','AMT_APPLICATION','AMT_CREDIT','AMT_GOODS_PRICE','SELLERPLACE_AREA']

prev_outlier_col_2 = ['SK_ID_CURR','DAYS_DECISION','CNT_PAYMENT']

for i in enumerate(prev_outlier_col_1):

    plt.subplot(2,4,i[0]+1)

    sns.boxplot(y=previousDF[i[1]])

    plt.title(i[1])

    plt.ylabel("")



for i in enumerate(prev_outlier_col_2):

    plt.subplot(2,4,i[0]+6)

    sns.boxplot(y=previousDF[i[1]])

    plt.title(i[1])

    plt.ylabel("") 
previousDF[['AMT_ANNUITY', 'AMT_APPLICATION', 'AMT_CREDIT', 'AMT_GOODS_PRICE', 'SELLERPLACE_AREA','CNT_PAYMENT','DAYS_DECISION']].describe()
Imbalance = applicationDF["TARGET"].value_counts().reset_index()



plt.figure(figsize=(10,4))

x= ['Repayer','Defaulter']

sns.barplot(x,"TARGET",data = Imbalance,palette= ['g','r'])

plt.xlabel("Loan Repayment Status")

plt.ylabel("Count of Repayers & Defaulters")

plt.title("Imbalance Plotting")

plt.show()
count_0 = Imbalance.iloc[0]["TARGET"]

count_1 = Imbalance.iloc[1]["TARGET"]

count_0_perc = round(count_0/(count_0+count_1)*100,2)

count_1_perc = round(count_1/(count_0+count_1)*100,2)



print('Ratios of imbalance in percentage with respect to Repayer and Defaulter datas are: %.2f and %.2f'%(count_0_perc,count_1_perc))

print('Ratios of imbalance in relative with respect to Repayer and Defaulter datas is %.2f : 1 (approx)'%(count_0/count_1))
# function for plotting repetitive countplots in univariate categorical analysis on applicationDF

# This function will create two subplots: 

# 1. Count plot of categorical column w.r.t TARGET; 

# 2. Percentage of defaulters within column



def univariate_categorical(feature,ylog=False,label_rotation=False,horizontal_layout=True):

    temp = applicationDF[feature].value_counts()

    df1 = pd.DataFrame({feature: temp.index,'Number of contracts': temp.values})



    # Calculate the percentage of target=1 per category value

    cat_perc = applicationDF[[feature, 'TARGET']].groupby([feature],as_index=False).mean()

    cat_perc["TARGET"] = cat_perc["TARGET"]*100

    cat_perc.sort_values(by='TARGET', ascending=False, inplace=True)

    

    if(horizontal_layout):

        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))

    else:

        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(20,24))

        

    # 1. Subplot 1: Count plot of categorical column

    # sns.set_palette("Set2")

    s = sns.countplot(ax=ax1, 

                    x = feature, 

                    data=applicationDF,

                    hue ="TARGET",

                    order=cat_perc[feature],

                    palette=['g','r'])

    

    # Define common styling

    ax1.set_title(feature, fontdict={'fontsize' : 10, 'fontweight' : 3, 'color' : 'Blue'}) 

    ax1.legend(['Repayer','Defaulter'])

    

    # If the plot is not readable, use the log scale.

    if ylog:

        ax1.set_yscale('log')

        ax1.set_ylabel("Count (log)",fontdict={'fontsize' : 10, 'fontweight' : 3, 'color' : 'Blue'})   

    

    

    if(label_rotation):

        s.set_xticklabels(s.get_xticklabels(),rotation=90)

    

    # 2. Subplot 2: Percentage of defaulters within the categorical column

    s = sns.barplot(ax=ax2, 

                    x = feature, 

                    y='TARGET', 

                    order=cat_perc[feature], 

                    data=cat_perc,

                    palette='Set2')

    

    if(label_rotation):

        s.set_xticklabels(s.get_xticklabels(),rotation=90)

    plt.ylabel('Percent of Defaulters [%]', fontsize=10)

    plt.tick_params(axis='both', which='major', labelsize=10)

    ax2.set_title(feature + " Defaulter %", fontdict={'fontsize' : 15, 'fontweight' : 5, 'color' : 'Blue'}) 



    plt.show();
# function for plotting repetitive countplots in bivariate categorical analysis



def bivariate_bar(x,y,df,hue,figsize):

    

    plt.figure(figsize=figsize)

    sns.barplot(x=x,

                  y=y,

                  data=df, 

                  hue=hue, 

                  palette =['g','r'])     

        

    # Defining aesthetics of Labels and Title of the plot using style dictionaries

    plt.xlabel(x,fontdict={'fontsize' : 10, 'fontweight' : 3, 'color' : 'Blue'})    

    plt.ylabel(y,fontdict={'fontsize' : 10, 'fontweight' : 3, 'color' : 'Blue'})    

    plt.title(col, fontdict={'fontsize' : 15, 'fontweight' : 5, 'color' : 'Blue'}) 

    plt.xticks(rotation=90, ha='right')

    plt.legend(labels = ['Repayer','Defaulter'])

    plt.show()
# function for plotting repetitive rel plots in bivaritae numerical analysis on applicationDF



def bivariate_rel(x,y,data, hue, kind, palette, legend,figsize):

    

    plt.figure(figsize=figsize)

    sns.relplot(x=x, 

                y=y, 

                data=applicationDF, 

                hue="TARGET",

                kind=kind,

                palette = ['g','r'],

                legend = False)

    plt.legend(['Repayer','Defaulter'])

    plt.xticks(rotation=90, ha='right')

    plt.show()
#function for plotting repetitive countplots in univariate categorical analysis on the merged df



def univariate_merged(col,df,hue,palette,ylog,figsize):

    plt.figure(figsize=figsize)

    ax=sns.countplot(x=col, 

                  data=df,

                  hue= hue,

                  palette= palette,

                  order=df[col].value_counts().index)

    



    if ylog:

        plt.yscale('log')

        plt.ylabel("Count (log)",fontdict={'fontsize' : 10, 'fontweight' : 3, 'color' : 'Blue'})     

    else:

        plt.ylabel("Count",fontdict={'fontsize' : 10, 'fontweight' : 3, 'color' : 'Blue'})       



    plt.title(col , fontdict={'fontsize' : 15, 'fontweight' : 5, 'color' : 'Blue'}) 

    plt.legend(loc = "upper right")

    plt.xticks(rotation=90, ha='right')

    

    plt.show()
# Function to plot point plots on merged dataframe



def merged_pointplot(x,y):

    plt.figure(figsize=(8,4))

    sns.pointplot(x=x, 

                  y=y, 

                  hue="TARGET", 

                  data=loan_process_df,

                  palette =['g','r'])

   # plt.legend(['Repayer','Defaulter'])
# Checking the contract type based on loan repayment status

univariate_categorical('NAME_CONTRACT_TYPE',True)
# Checking the type of Gender on loan repayment status

univariate_categorical('CODE_GENDER')
# Checking if owning a car is related to loan repayment status

univariate_categorical('FLAG_OWN_CAR')
# Checking if owning a realty is related to loan repayment status

univariate_categorical('FLAG_OWN_REALTY')
# Analyzing Housing Type based on loan repayment status

univariate_categorical("NAME_HOUSING_TYPE",True,True,True)
# Analyzing Family status based on loan repayment status

univariate_categorical("NAME_FAMILY_STATUS",False,True,True)
# Analyzing Education Type based on loan repayment status

univariate_categorical("NAME_EDUCATION_TYPE",True,True,True)
# Analyzing Income Type based on loan repayment status

univariate_categorical("NAME_INCOME_TYPE",True,True,False)
# Analyzing Region rating where applicant lives based on loan repayment status

univariate_categorical("REGION_RATING_CLIENT",False,False,True)
# Analyzing Occupation Type where applicant lives based on loan repayment status

univariate_categorical("OCCUPATION_TYPE",False,True,False)
# Checking Loan repayment status based on Organization type

univariate_categorical("ORGANIZATION_TYPE",True,True,False)
# Analyzing Flag_Doc_3 submission status based on loan repayment status

univariate_categorical("FLAG_DOCUMENT_3",False,False,True)
# Analyzing Age Group based on loan repayment status

univariate_categorical("AGE_GROUP",False,False,True)
# Analyzing Employment_Year based on loan repayment status

univariate_categorical("EMPLOYMENT_YEAR",False,False,True)
# Analyzing Amount_Credit based on loan repayment status

univariate_categorical("AMT_CREDIT_RANGE",False,False,False)
# Analyzing Amount_Income Range based on loan repayment status

univariate_categorical("AMT_INCOME_RANGE",False,False,False)
# Analyzing Number of children based on loan repayment status

univariate_categorical("CNT_CHILDREN",True)
# Analyzing Number of family members based on loan repayment status

univariate_categorical("CNT_FAM_MEMBERS",True, False, False)
applicationDF.groupby('NAME_INCOME_TYPE')['AMT_INCOME_TOTAL'].describe()
# Income type vs Income Amount Range

bivariate_bar("NAME_INCOME_TYPE","AMT_INCOME_TOTAL",applicationDF,"TARGET",(18,10))
applicationDF.columns
# Bifurcating the applicationDF dataframe based on Target value 0 and 1 for correlation and other analysis

cols_for_correlation = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 

                        'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 

                        'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',

                        'NAME_HOUSING_TYPE', 'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 

                        'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'OCCUPATION_TYPE', 'CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT',

                        'REGION_RATING_CLIENT_W_CITY', 'WEEKDAY_APPR_PROCESS_START', 'HOUR_APPR_PROCESS_START',

                        'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION', 

                        'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY', 'ORGANIZATION_TYPE',

                        'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE', 'DAYS_LAST_PHONE_CHANGE', 'FLAG_DOCUMENT_3', 

                        'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_WEEK',

                        'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR']





Repayer_df = applicationDF.loc[applicationDF['TARGET']==0, cols_for_correlation] # Repayers

Defaulter_df = applicationDF.loc[applicationDF['TARGET']==1, cols_for_correlation] # Defaulters
# Getting the top 10 correlation for the Repayers data

corr_repayer = Repayer_df.corr()

corr_repayer = corr_repayer.where(np.triu(np.ones(corr_repayer.shape),k=1).astype(np.bool))

corr_df_repayer = corr_repayer.unstack().reset_index()

corr_df_repayer.columns =['VAR1','VAR2','Correlation']

corr_df_repayer.dropna(subset = ["Correlation"], inplace = True)

corr_df_repayer["Correlation"]=corr_df_repayer["Correlation"].abs() 

corr_df_repayer.sort_values(by='Correlation', ascending=False, inplace=True) 

corr_df_repayer.head(10)
fig = plt.figure(figsize=(12,12))

ax = sns.heatmap(Repayer_df.corr(), cmap="RdYlGn",annot=False,linewidth =1)
# Getting the top 10 correlation for the Defaulter data

corr_Defaulter = Defaulter_df.corr()

corr_Defaulter = corr_Defaulter.where(np.triu(np.ones(corr_Defaulter.shape),k=1).astype(np.bool))

corr_df_Defaulter = corr_Defaulter.unstack().reset_index()

corr_df_Defaulter.columns =['VAR1','VAR2','Correlation']

corr_df_Defaulter.dropna(subset = ["Correlation"], inplace = True)

corr_df_Defaulter["Correlation"]=corr_df_Defaulter["Correlation"].abs()

corr_df_Defaulter.sort_values(by='Correlation', ascending=False, inplace=True)

corr_df_Defaulter.head(10)
fig = plt.figure(figsize=(12,12))

ax = sns.heatmap(Defaulter_df.corr(), cmap="RdYlGn",annot=False,linewidth =1)
# Plotting the numerical columns related to amount as distribution plot to see density

amount = applicationDF[[ 'AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY', 'AMT_GOODS_PRICE']]



fig = plt.figure(figsize=(16,12))



for i in enumerate(amount):

    plt.subplot(2,2,i[0]+1)

    sns.distplot(Defaulter_df[i[1]], hist=False, color='r',label ="Defaulter")

    sns.distplot(Repayer_df[i[1]], hist=False, color='g', label ="Repayer")

    plt.title(i[1], fontdict={'fontsize' : 15, 'fontweight' : 5, 'color' : 'Blue'}) 

    

plt.legend()



plt.show() 
# Checking the relationship between Goods price and credit and comparing with loan repayment staus

bivariate_rel('AMT_GOODS_PRICE','AMT_CREDIT',applicationDF,"TARGET", "line", ['g','r'], False,(15,6))
# Plotting pairplot between amount variable to draw reference against loan repayment status

amount = applicationDF[[ 'AMT_INCOME_TOTAL','AMT_CREDIT',

                         'AMT_ANNUITY', 'AMT_GOODS_PRICE','TARGET']]

amount = amount[(amount["AMT_GOODS_PRICE"].notnull()) & (amount["AMT_ANNUITY"].notnull())]

ax= sns.pairplot(amount,hue="TARGET",palette=["g","r"])

ax.fig.legend(labels=['Repayer','Defaulter'])

plt.show()
#merge both the dataframe on SK_ID_CURR with Inner Joins

loan_process_df = pd.merge(applicationDF, previousDF, how='inner', on='SK_ID_CURR')

loan_process_df.head()
#Checking the details of the merged dataframe

loan_process_df.shape
# Checking the element count of the dataframe

loan_process_df.size
# checking the columns and column types of the dataframe

loan_process_df.info()
# Checking merged dataframe numerical columns statistics

loan_process_df.describe()
# Bifurcating the applicationDF dataframe based on Target value 0 and 1 for correlation and other analysis



L0 = loan_process_df[loan_process_df['TARGET']==0] # Repayers

L1 = loan_process_df[loan_process_df['TARGET']==1] # Defaulters
univariate_merged("NAME_CASH_LOAN_PURPOSE",L0,"NAME_CONTRACT_STATUS",["#548235","#FF0000","#0070C0","#FFFF00"],True,(18,7))



univariate_merged("NAME_CASH_LOAN_PURPOSE",L1,"NAME_CONTRACT_STATUS",["#548235","#FF0000","#0070C0","#FFFF00"],True,(18,7))
# Checking the Contract Status based on loan repayment status and whether there is any business loss or financial loss

univariate_merged("NAME_CONTRACT_STATUS",loan_process_df,"TARGET",['g','r'],False,(12,8))

g = loan_process_df.groupby("NAME_CONTRACT_STATUS")["TARGET"]

df1 = pd.concat([g.value_counts(),round(g.value_counts(normalize=True).mul(100),2)],axis=1, keys=('Counts','Percentage'))

df1['Percentage'] = df1['Percentage'].astype(str) +"%" # adding percentage symbol in the results for understanding

print (df1)
# plotting the relationship between income total and contact status

merged_pointplot("NAME_CONTRACT_STATUS",'AMT_INCOME_TOTAL')
# plotting the relationship between people who defaulted in last 60 days being in client's social circle and contact status

merged_pointplot("NAME_CONTRACT_STATUS",'DEF_60_CNT_SOCIAL_CIRCLE')