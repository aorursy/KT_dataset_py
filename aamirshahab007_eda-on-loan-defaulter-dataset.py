# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Importing Libraries

import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns
pd.set_option('display.max_columns',150)

pd.set_option('display.max_info_columns', 150)

pd.set_option('display.max_rows',150)
# Loading pplication_data  datasets

app_data = pd.read_csv("../input/loan-defaulter/application_data.csv")
# Display top 5 rows of app_data dataframe

app_data.head()
# Printing shape of application_data dataset

print(f'Shape of app_data : {app_data.shape}')
app_data.info()
app_data.describe()
# Removing Unwanted Col.

unwanted_col=['FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE',

       'FLAG_PHONE', 'FLAG_EMAIL','REGION_RATING_CLIENT','REGION_RATING_CLIENT_W_CITY','FLAG_EMAIL','CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT',

       'REGION_RATING_CLIENT_W_CITY','DAYS_LAST_PHONE_CHANGE', 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3','FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6',

       'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9','FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12',

       'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15','FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18',

       'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21','OBS_30_CNT_SOCIAL_CIRCLE','DEF_30_CNT_SOCIAL_CIRCLE',

        'OBS_60_CNT_SOCIAL_CIRCLE','DEF_60_CNT_SOCIAL_CIRCLE','EXT_SOURCE_2','EXT_SOURCE_3']

app_data.drop(unwanted_col, inplace=True, axis=1)
# Column wise Null percentage in app_data

null_percentage_app = round(app_data.isnull().sum()/app_data.shape[0]*100, 2)

print(null_percentage_app)
# Getting list of columns which have more than or equal to 45% missing values

app_colToDrop = list(null_percentage_app[null_percentage_app >= 45].index)
print(f'No. of col. to be drop: {len(app_colToDrop)}')

app_colToDrop
# Dropping Columns having more than 45% of missing values

app_data.drop(app_colToDrop, axis= 1, inplace=True)
# Rechecking column wise Null percentage in app_data

null_percentage_app = round(app_data.isnull().sum()/app_data.shape[0]*100, 2)

print(null_percentage_app)

print(f'New shape of app_data: {app_data.shape}')
# Getting columns with missing values between 0% and 45%

missing_value_col_app = list((null_percentage_app[null_percentage_app > 0].index))

missing_value_col_app
print(app_data.AMT_GOODS_PRICE.describe())

sns.distplot(app_data.AMT_GOODS_PRICE, hist=False)
sns.boxplot(y = app_data.AMT_GOODS_PRICE)
print(app_data.OCCUPATION_TYPE.value_counts())

app_data.OCCUPATION_TYPE.value_counts().plot(kind ='bar', figsize = (10,8))
# print(app_data.EXT_SOURCE_3.describe())

# sns.boxplot(y = app_data.EXT_SOURCE_3)
# sns.distplot(app_data.EXT_SOURCE_3, hist= False)
app_data.info()


Amt_req_credit = list(enumerate(missing_value_col_app[-6:]))

for i in Amt_req_credit:

    print('\n'+i[1])

    print(f'No. of unique values: {app_data[i[1]].nunique()}')

    print(app_data[i[1]].value_counts())

    plt.figure(figsize=(5,10))

    plt.subplot(len(Amt_req_credit), 1, i[0]+1 )

    plt.title(i[1])

    app_data[i[1]].value_counts().plot(kind= 'bar')
app_data
app_data.DAYS_EMPLOYED.value_counts()
app_data.DAYS_EMPLOYED.replace(365243, 0, inplace= True)
app_data.DAYS_EMPLOYED = -1*app_data.DAYS_EMPLOYED
app_data.DAYS_EMPLOYED.value_counts()
# Checking ORGANIZATION_TYPE col

app_data.ORGANIZATION_TYPE.value_counts(normalize = True)*100
# Removing XNA rows from the CODE_GENDER.

app_data.drop(app_data[app_data.ORGANIZATION_TYPE == 'XNA'].index, axis=0, inplace=True)

app_data.ORGANIZATION_TYPE.value_counts()
# Checking ORGANIZATION_TYPE col

app_data.CODE_GENDER.value_counts(normalize = True)*100
# Removing XNA rows from the CODE_GENDER.

app_data.drop(app_data[app_data.CODE_GENDER == 'XNA'].index, axis=0, inplace=True)

app_data.CODE_GENDER.value_counts()
# Defining fun. to fix other col.

Other_colToFix = ['DAYS_BIRTH','DAYS_ID_PUBLISH','DAYS_REGISTRATION']

def fixCol(arr):

    for i in arr:

        app_data[i] = -1 * app_data[i]

        print('\n'+i)

        print(app_data[i].value_counts())
fixCol(Other_colToFix)
# Casting all other columns data type to numeric data type



num_col=['TARGET',

          'CNT_CHILDREN',

          'AMT_INCOME_TOTAL',

          'AMT_CREDIT','AMT_ANNUITY',

          'REGION_POPULATION_RELATIVE','DAYS_BIRTH',

          'DAYS_EMPLOYED',

          'DAYS_REGISTRATION',

          'DAYS_ID_PUBLISH',

          'HOUR_APPR_PROCESS_START',

          'LIVE_REGION_NOT_WORK_REGION',

          'REG_CITY_NOT_LIVE_CITY',

          'REG_CITY_NOT_WORK_CITY',

          'LIVE_CITY_NOT_WORK_CITY']



app_data[num_col]=app_data[num_col].apply(pd.to_numeric)

app_data.head(5)

For_Outliers = list(enumerate(['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE']))
for i in For_Outliers:

    print('\n'+i[1])

    print('-'*30)

    print(app_data[i[1]].describe())
for i in For_Outliers:

    plt.figure(figsize=(10,15))

    plt.subplot(len(For_Outliers), 1, i[0]+1)

    sns.boxplot(app_data[i[1]])
app_data.AMT_CREDIT.quantile([0.5, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99,1])
app_data[app_data.AMT_CREDIT > 1200000].AMT_CREDIT.describe()
app_data.AMT_ANNUITY.quantile([0.5, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99, 1])
app_data[app_data.AMT_ANNUITY > 60000].AMT_ANNUITY.describe()
app_data.AMT_GOODS_PRICE.quantile([0.5, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99, 1])
app_data[app_data.AMT_GOODS_PRICE > 1800000].AMT_GOODS_PRICE.dropna().describe()
# Creating bins for AMT_INCOME_TOTAL

bins = [0,100000,200000,300000,400000,500000,600000,700000,800000, 900000, 1000000, 100000000]

slot = ['0-100000','100000-200000','200000-300000','300000-400000','400000-500000','500000-600000',

        '600000-700000','700000-800000','800000-900000','900000-1000000', '1000000 +']



app_data['AMT_INCOME_RANGE']=pd.cut(app_data['AMT_INCOME_TOTAL'], bins, labels=slot)
app_data['AMT_INCOME_RANGE'].value_counts
# Creating bins for AMT_CREDIT



bins = [0,150000,200000,250000,300000,350000,400000,450000,500000,550000,600000,650000,700000,750000,800000,850000,900000,1000000000]

slots = ['0-150000', '150000-200000','200000-250000', '250000-300000', '300000-350000', '350000-400000','400000-450000',

        '450000-500000','500000-550000','550000-600000','600000-650000','650000-700000','700000-750000','750000-800000',

        '800000-850000','850000-900000','900000 and above']



app_data['AMT_CREDIT_RANGE']=pd.cut(app_data['AMT_CREDIT'], bins=bins, labels=slots)
app_data.head()
# Checking for Imbalance dataset w.r.t. TARGET col

value_Count_Target = app_data.TARGET.value_counts(normalize = True)*100

value_Count_Target.plot(kind= 'bar')

print(value_Count_Target)
# Splitting dataset w.r.t Traget == 0 & Target == 1

Target_0 = app_data[app_data.TARGET == 0]

Target_1 = app_data[app_data.TARGET == 1]
Target_0.head()
Target_1.head()
# Defining a function to plot the countplot for different categories

def UniVarCatPlot(title, hue = None, rotation=None, col_y = None, col_x = None):

    sns.set_style('whitegrid')

    sns.set_context('talk')

    plt.rcParams["axes.labelsize"] = 20

    plt.rcParams['axes.titlesize'] = 30

    plt.rcParams['axes.titlepad'] = 30

    

    if col_x:

        col_name = col_x

        plt.figure(figsize=(30,25))

    else:

        col_name = col_y

        plt.figure(figsize=(15,38))



    #   1st subplot for Target_1    

    plt.subplots_adjust(hspace=0.5)

    plt.subplot(2,1,1)

    

    title1 = title + ' for Target_0 (Client with NO Payment Difficulty)'

    plt.title(title1)



    #   Adjusting scale for horizonatl plot    

    if col_x:

        plt.yscale('log')

        plt.xticks(rotation = rotation)

    else:

        plt.xscale('log')

        plt.yticks(rotation = rotation)

        

    sns.countplot(data = Target_0, x = col_x, y = col_y, order=Target_0[col_name].value_counts().index, hue=hue, palette='dark')

    



    #   2nd subplot for Target_1

    plt.subplot(2,1,2)

    plt.xticks(rotation = rotation)

    title2 = title + ' for Target_1 (Client with Payment Difficulty)'

    plt.title(title2)



    #   Adjusting scale for horizonatl plot

    if col_x:

        plt.yscale('log')

        plt.xticks(rotation = rotation)

    else:

        plt.xscale('log')

        plt.yticks(rotation = rotation)

        

    sns.countplot(data = Target_1, x = col_x, y= col_y, order=Target_1[col_name].value_counts().index, hue=hue, palette='dark')

    plt.legend(loc = 'upper right', fontsize = 'large')

    plt.show()
# Count plot for income range with wrt gender

UniVarCatPlot(col_x = 'AMT_INCOME_RANGE', title= 'Count Plot for Income Range', hue='CODE_GENDER')
# Count Plot for contract type wrt gender

UniVarCatPlot(col_x = 'NAME_CONTRACT_TYPE', title= 'Count Plot for Contract Type', hue='CODE_GENDER')
# Count Plot for type of education wrt gender

UniVarCatPlot(col_x = 'NAME_EDUCATION_TYPE', title= 'Count Plot for Education Type', hue='CODE_GENDER')
# Count Plot for cdifferent housing type

UniVarCatPlot(col_x = 'NAME_HOUSING_TYPE', title= 'Count Plot for Different Housing Type', hue='CODE_GENDER')
# Count Plot for contract type wrt gender

UniVarCatPlot(col_x = 'OCCUPATION_TYPE', title= 'Count Plot for Different Occupation Type', rotation= 45)
UniVarCatPlot(col_y = 'ORGANIZATION_TYPE', title= 'Count Plot for Different Organization Type')
# Finding Correlation between variables for Target_0

corr_0 = Target_0.corr()

# sns.heatmap(corr_0)

corr_0_df = corr_0.where(np.triu(np.ones(corr_0.shape), k=1).astype(np.bool))

corr_0_df = corr_0_df.unstack().reset_index()

corr_0_df.columns = ['Variable_1', 'Variable_2', 'Correlation']

corr_0_df.dropna(subset = ['Correlation'], inplace = True)

corr_0_df['Correlation'] = round(corr_0_df['Correlation'],2)

corr_0_df['Correlation'] = abs(corr_0_df['Correlation'])

corr_0_df.sort_values(by = 'Correlation', ascending = False).head(10)
# Finding Correlation between variables for Target_1

corr_1 = Target_1.corr()

corr_1_df = corr_1.where(np.triu(np.ones(corr_1.shape), k=1).astype(np.bool))

corr_1_df = corr_1_df.unstack().reset_index()

corr_1_df.columns = ['Variable_1', 'Variable_2', 'Correlation']

corr_1_df.dropna(subset = ['Correlation'], inplace = True)

corr_1_df['Correlation'] = round(corr_1_df['Correlation'],2)

corr_1_df['Correlation'] = abs(corr_1_df['Correlation'])

corr_1_df.sort_values(by = 'Correlation', ascending = False).head(10)
# Correlation Matrix for Target_0

plt.figure(figsize=(10, 8))

plt.rcParams['axes.titlesize'] = 20

plt.rcParams['axes.titlepad'] = 30

plt.rcParams['xtick.labelsize']=12

plt.rcParams['ytick.labelsize']=12

plt.title('Correlattion of Target 0')

sns.heatmap(corr_0.iloc[2:-6,2:-6], cmap ='RdYlGn')
# Correlation Matrix for Target_1

plt.figure(figsize=(10, 8))

plt.rcParams['axes.titlesize'] = 20

plt.rcParams['axes.titlepad'] = 30

plt.rcParams['xtick.labelsize']=12

plt.rcParams['ytick.labelsize']=12

plt.title('Correlattion of Target 1')

sns.heatmap(corr_1.iloc[2:-6,2:-6], cmap ='RdYlGn')
# Defining a function to plot the boxplot for different numerical col.

def UniVarNumPlot(title, col):

    sns.set_style('whitegrid')

    sns.set_context('talk')

    plt.figure(figsize=(15,6))

    plt.rcParams["axes.labelsize"] = 12

    plt.rcParams['xtick.labelsize']=12

    plt.rcParams['ytick.labelsize']=12

    plt.rcParams['axes.titlesize'] = 14

    plt.rcParams['axes.titleweight'] = 12

    plt.rcParams['axes.titlepad'] = 30

    



    #   1st subplot for Target_1    

    plt.subplots_adjust(wspace=1.5)

    plt.subplot(1,2,1)

    title1 = title + ' for Target_0'

    plt.title(title1)

    plt.yscale('log')    

    sns.boxplot(data = Target_0, x = col, orient = 'v')

    



    #   2nd subplot for Target_1

    plt.subplot(1,2,2)

    title2 = title + ' for Target_1'

    plt.title(title2)

    plt.yscale('log')

    sns.boxplot(data = Target_1, x = col, orient = 'v')

    plt.show()
# Boxplot plot for Total Income between Target_0 and Target_1

UniVarNumPlot(col ='AMT_INCOME_TOTAL', title = 'Distribution of Client Income' )
# Boxplot plot for Credit Amt between Target_0 and Target_1

UniVarNumPlot(col ='AMT_CREDIT', title = 'Distribution of Credit Amount' )
# Boxplot plot for Credit Amt between Target_0 and Target_1

UniVarNumPlot(col ='AMT_ANNUITY', title = 'Distribution of Annuity Amount' )
# Function for Bi- variate Boxplot analysis

def BiVarPlot(data, col_x, col_y, hue, title, scale=None):

    plt.rcParams["axes.labelsize"] = 12

    plt.rcParams["axes.labelpad"] = 12

    plt.rcParams['xtick.labelsize']=12

    plt.rcParams['ytick.labelsize']=12

    plt.rcParams['axes.titlesize'] = 14

    plt.rcParams['axes.titleweight'] = 12

    plt.rcParams['axes.titlepad'] = 30

    plt.figure(figsize=(16,12))

    plt.xticks(rotation=0)

    if scale:

        plt.yscale(scale)

    sns.boxplot(data = data, x= col_x, y= col_y, hue= hue, orient='v')

    plt.title(title)

    plt.show()
# Box plotting for Credit amount for Target_0

BiVarPlot(data = Target_0,

          col_x='NAME_EDUCATION_TYPE',

          col_y='AMT_CREDIT',

          hue='NAME_FAMILY_STATUS',

          title='Credit amount vs Education Status')
# Box plotting for Credit amount for Target_1

BiVarPlot(data = Target_1,

          col_x='NAME_EDUCATION_TYPE',

          col_y='AMT_CREDIT',

          hue='NAME_FAMILY_STATUS',

          title='Credit amount vs Education Status')
# Box plotting for Income amount in logarithmic scale for Target_0

BiVarPlot(data = Target_0,

          col_x='NAME_EDUCATION_TYPE',

          col_y='AMT_INCOME_TOTAL',

          hue='NAME_FAMILY_STATUS',

          title='Income amount vs Education Status',

          scale='log')
# Box plotting for Income amount in logarithmic scale for Target_1

BiVarPlot(data = Target_1,

          col_x='NAME_EDUCATION_TYPE',

          col_y='AMT_INCOME_TOTAL',

          hue='NAME_FAMILY_STATUS',

          title='Income amount vs Education Status',

          scale='log')
# Loading previous_application and application_data  datasets

prev_application = pd.read_csv('../input/loan-defaulter/previous_application.csv')
# display top 5 rows of prev_application dataframe

prev_application.head()
# Printing shape of both dataset

print(f'Shape of prev_application : {prev_application.shape}')
prev_application.info()
prev_application.describe()
# Column wise Null percentage in prev_application

null_percentage_prev = round(prev_application.isnull().sum()/prev_application.shape[0]*100, 2)

print(null_percentage_prev)
# Getting the list of columns with more than 45% of null values 

prev_colToDrop = list(null_percentage_prev[null_percentage_prev >= 35].index)
print(f'No. of col. to be drop: {len(prev_colToDrop)}')

prev_colToDrop
# Dropping the columns from prev_application dataframe

prev_application.drop(prev_colToDrop, axis= 1, inplace = True)
# Rechecking column wise Null percentage in prev_application

null_percentage_prev = round(prev_application.isnull().sum()/prev_application.shape[0]*100, 2)

print(null_percentage_prev)

print(f'New shape of prev_application: {prev_application.shape}')
# Getting columns with missing values between 0% and 35%

missing_value_col_prev = list(null_percentage_prev[null_percentage_prev > 0].index)

missing_value_col_prev
round(prev_application.NAME_CASH_LOAN_PURPOSE.value_counts()/prev_application.shape[0]*100,2)
# Dropping rows conating 'XNA' and 'XAP'

prev_application=prev_application.drop(prev_application[prev_application['NAME_CASH_LOAN_PURPOSE']=='XNA'].index)

prev_application=prev_application.drop(prev_application[prev_application['NAME_CASH_LOAN_PURPOSE']=='XAP'].index)
prev_application.shape
# Merging prev_application and app_data on SK_ID_CURR

merged_df = pd.merge(left = app_data, right=prev_application, how='inner', on='SK_ID_CURR', suffixes='_o')
merged_df.head()
# Renaming the col

merged_df = merged_df.rename({'NAME_CONTRACT_TYPE_' : 'NAME_CONTRACT_TYPE',

                         'AMT_CREDIT_':'AMT_CREDIT',

                         'AMT_ANNUITY_':'AMT_ANNUITY',

                         'WEEKDAY_APPR_PROCESS_START_' : 'WEEKDAY_APPR_PROCESS_START',

                         'HOUR_APPR_PROCESS_START_':'HOUR_APPR_PROCESS_START',

                         'NAME_CONTRACT_TYPEo':'NAME_CONTRACT_TYPE_PREV',

                         'AMT_CREDITo':'AMT_CREDIT_PREV',

                         'AMT_ANNUITYo':'AMT_ANNUITY_PREV',

                         'WEEKDAY_APPR_PROCESS_STARTo':'WEEKDAY_APPR_PROCESS_START_PREV',

                         'HOUR_APPR_PROCESS_STARTo':'HOUR_APPR_PROCESS_START_PREV'}, axis=1)
merged_df
# Removing Unwanted columns

merged_df.drop(['WEEKDAY_APPR_PROCESS_START',

              'HOUR_APPR_PROCESS_START',

              'REG_REGION_NOT_LIVE_REGION', 

              'REG_REGION_NOT_WORK_REGION',

              'LIVE_REGION_NOT_WORK_REGION',

              'REG_CITY_NOT_LIVE_CITY',

              'REG_CITY_NOT_WORK_CITY', 

              'LIVE_CITY_NOT_WORK_CITY',

              'WEEKDAY_APPR_PROCESS_START_PREV',

              'HOUR_APPR_PROCESS_START_PREV', 

              'FLAG_LAST_APPL_PER_CONTRACT',

              'NFLAG_LAST_APPL_IN_DAY',

              'NAME_GOODS_CATEGORY',

              'SELLERPLACE_AREA',

              'NAME_SELLER_INDUSTRY'],axis=1,inplace=True)
merged_df.info()
# Function for Univariate Analysis on merged_df using count plot

def UniVariatePlot(dataframe, col_y, hue, title):

    sns.set_style('whitegrid')

    sns.set_context('talk')



    plt.figure(figsize=(15,25))

    plt.rcParams["axes.labelsize"] = 20

    plt.rcParams['axes.titlesize'] = 22

    plt.rcParams['axes.titlepad'] = 30

    

    plt.xscale('log')

    plt.title(title)

    sns.countplot(data = merged_df,

                  y= col_y,

                  order=merged_df[col_y].value_counts().index,

                  hue = hue,

                  palette='dark')
# Distribution of contract status in logarithmic scale

UniVariatePlot(dataframe=merged_df,

               col_y = 'NAME_CASH_LOAN_PURPOSE', 

               hue='NAME_CONTRACT_STATUS', 

               title= 'Distribution of contract status with purposes')
# Distribution of Target

UniVariatePlot(dataframe=merged_df, 

               col_y='NAME_CASH_LOAN_PURPOSE', 

               hue='TARGET', 

               title='Distribution occupation type with Target.')
UniVariatePlot(dataframe=merged_df, 

               col_y='OCCUPATION_TYPE', 

               hue='NAME_CONTRACT_STATUS', 

               title='Distribution occupation type with Target.')
UniVariatePlot(dataframe=merged_df, 

               col_y='OCCUPATION_TYPE', 

               hue='TARGET', 

               title='Distribution occupation type with Target.')
# Function for Bivariate Analysis on merged_df using boxplot

def BiVariatePlot(dataframe, col_x, col_y, hue, title):

    sns.set_style('whitegrid')

    sns.set_context('talk')



    plt.figure(figsize=(16,12))

    plt.rcParams["axes.labelsize"] = 20

    plt.rcParams['axes.titlesize'] = 22

    plt.rcParams['axes.titlepad'] = 30

    

    plt.yscale('log')

    plt.title(title)

    plt.xticks(rotation=90)

    sns.boxplot(data = merged_df,

                x = col_x,

                y= col_y,

                hue = hue,

                orient='v')

    plt.show()
# Box plotting for Credit amount in logarithmic scale

BiVariatePlot(dataframe=merged_df,

              col_x='NAME_CASH_LOAN_PURPOSE',

              col_y='AMT_CREDIT_PREV',

              hue='NAME_INCOME_TYPE',

              title='Prev Credit amount vs Loan Purpose')
# Box plotting for Credit amount prev vs Housing type in logarithmic scale

plt.figure(figsize=(16,12))

plt.xticks(rotation=90)

plt.yscale('log')

sns.barplot(data =merged_df, y='AMT_CREDIT_PREV',hue='TARGET',x='NAME_HOUSING_TYPE')

plt.title('Prev Credit amount vs Housing type')

plt.show()