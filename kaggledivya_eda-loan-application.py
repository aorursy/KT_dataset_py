import warnings

warnings.filterwarnings('ignore')
# import application_data.csv



#Load the libraries

import pandas as pd #To work with dataset

import numpy as np #Math library

import seaborn as sns #Graph library that use matplot in background

import matplotlib.pyplot as plt #to plot some parameters in seaborn



pd.set_option('display.max_columns',999) #set column display number

pd.set_option('display.max_rows',200) #set row display number

pd.set_option('float_format', '{:f}'.format) #set float format

 

#Importing the data

df_app = pd.read_csv('../input/loanapplicationdata/application_data.csv')
# Starting with analysis of Application data



# Shape: Displays number of column and rows

print('Shape:',df_app.shape,'\n')



#info: Displays number of rows for each column and its datatype

print(df_app.info())



#Columns display

print(df_app.columns)
#Statistical description of each column in application_data.csv

df_app.describe()
#Display top 5 row

df_app.head()
round(100*(df_app.isnull().sum()/len(df_app.index)),2)
#Assigning NULL percentage value to df_null

df_null = round(100*(df_app.isnull().sum()/len(df_app.index)),2)



# find columns with more than 50% missing values

column = df_null[df_null >= 50].index 



# drop columns with high null percentage

df_app.drop(column,axis = 1,inplace = True)



#check null percentage after dropping

round(100*(df_app.isnull().sum()/len(df_app.index)),2)
# Check the statistical distribution of data.

print(df_app['AMT_REQ_CREDIT_BUREAU_HOUR'].describe())

print(df_app['AMT_REQ_CREDIT_BUREAU_DAY'].describe())

print(df_app['AMT_REQ_CREDIT_BUREAU_WEEK'].describe())

print(df_app['AMT_REQ_CREDIT_BUREAU_MON'].describe())

print(df_app['AMT_REQ_CREDIT_BUREAU_QRT'].describe())

print(df_app['AMT_REQ_CREDIT_BUREAU_YEAR'].describe())
#Check the mode

print(df_app['AMT_REQ_CREDIT_BUREAU_HOUR'].mode())

print(df_app['AMT_REQ_CREDIT_BUREAU_DAY'].mode())

print(df_app['AMT_REQ_CREDIT_BUREAU_WEEK'].mode())

print(df_app['AMT_REQ_CREDIT_BUREAU_MON'].mode())

print(df_app['AMT_REQ_CREDIT_BUREAU_QRT'].mode())

print(df_app['AMT_REQ_CREDIT_BUREAU_YEAR'].mode())
df_app.dtypes
#Checking negative values of Days

print(df_app[df_app['DAYS_BIRTH']<0].DAYS_BIRTH)

print(df_app[df_app['DAYS_EMPLOYED']<0].DAYS_EMPLOYED)

print(df_app[df_app['DAYS_REGISTRATION']<0].DAYS_REGISTRATION)

print(df_app[df_app['DAYS_ID_PUBLISH']<0].DAYS_ID_PUBLISH)
#Converting negative vaules of Days to positive using abs()



df_app['DAYS_EMPLOYED'] = df_app['DAYS_EMPLOYED'].abs()

df_app['DAYS_BIRTH'] = df_app['DAYS_BIRTH'].abs()

df_app['DAYS_REGISTRATION'] = df_app['DAYS_REGISTRATION'].abs()

df_app['DAYS_ID_PUBLISH'] = df_app['DAYS_ID_PUBLISH'].abs()
#Checking if covertion worked

print(df_app[df_app['DAYS_BIRTH']<0].DAYS_EMPLOYED)

print(df_app[df_app['DAYS_EMPLOYED']<0].DAYS_EMPLOYED)

print(df_app[df_app['DAYS_REGISTRATION']<0].DAYS_EMPLOYED)

print(df_app[df_app['DAYS_ID_PUBLISH']<0].DAYS_EMPLOYED)
#select only identified columns from dataset

Col_dtype = ['SK_ID_CURR','TARGET','NAME_CONTRACT_TYPE','CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY','CNT_CHILDREN','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE','NAME_TYPE_SUITE','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','DAYS_BIRTH','DAYS_EMPLOYED','OCCUPATION_TYPE','CNT_FAM_MEMBERS','ORGANIZATION_TYPE']

df_app[Col_dtype].dtypes
#SK_ID_CURR is application ID so it can be converted to String/object

df_app['SK_ID_CURR'] = df_app['SK_ID_CURR'].astype('str')
#TARGET is column on which we are dividing our datasets for analysis,hence we are converting it to the string

df_app['TARGET'] = df_app['TARGET'].astype('str')

df_app['TARGET'].value_counts()
df_app['CNT_FAM_MEMBERS'].isnull().sum()
# imputing null values with mode for CNT_FAM_MEMBERS, as we need this column in further analysis

# Note there are only 2 null value, so it will not make any difference if we impute it with mode value which is 2.0 

print('Mode:',df_app['CNT_FAM_MEMBERS'].mode())

df_app.loc[pd.isnull(df_app['CNT_FAM_MEMBERS']),'CNT_FAM_MEMBERS'] = 2.0



# change it to int datatype as its a number of family members and its should be in integer.

df_app['CNT_FAM_MEMBERS'] = df_app['CNT_FAM_MEMBERS'].astype('int')

df_app['CNT_FAM_MEMBERS'].value_counts()
#Converting the days to year for DAYS_BIRTH would held AGE of a person

df_app['DAYS_BIRTH'] = abs((df_app['DAYS_BIRTH']/365)).astype(int)

#Converting the days to year for DAYS_EMPLOYED would held YEAR_WRK_EXP

df_app['DAYS_EMPLOYED'] = abs((df_app['DAYS_EMPLOYED']/365)).astype(int)



#Renaming the columns

df_app.rename(columns={'DAYS_BIRTH':'AGE','DAYS_EMPLOYED':'YEARS_WRK_EXP'},inplace = True)
#Creating list of varibales

l1 = ['AGE','YEARS_WRK_EXP','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE']

#plotting box plot for analysing the outliers

for i in l1:

    plt.yscale('log')

    sns.boxplot(y=df_app[i])

    plt.title(i)

    plt.show()
#1. AMT_CREDIT



#Categorise the applicants into four groups based on the AMT_CREDIT value 

#Low,Average, Good, Best. Where Best would have the highest AMT_CREDIT and low would have lowest AMT_CREDIT value.



#Function to bin, depending on the values 

def category(x) : 

    if x <= 250000 :

        return 'Low'

    if (x > 250000) & (x <= 500000) :

        return 'Average'

    if (x > 500000) & (x <= 800000) :

        return 'Good'

    if x > 800000 :

        return 'Best'

# Categorise based on categories above

df_app['Bin_AMT_CREDIT'] = df_app['AMT_CREDIT'].apply(category)

df_app.head(5)
#Plotting the BIN_AMT_CREDIT w.r.t Target variable



plt.figure(num=None, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')

plt.subplots_adjust(hspace=0.0, wspace=0.8)

plt.subplot(1,2,1)

plt.gca().set_title("Target_0")

df_app[df_app['TARGET']== '0'].Bin_AMT_CREDIT.value_counts().plot.pie(autopct='%1.1f%%')

plt.ylabel("")

plt.subplot(1,2,2)

plt.gca().set_title("Target_1")

df_app[df_app['TARGET']== '1'].Bin_AMT_CREDIT.value_counts().plot.pie(autopct='%1.1f%%')

plt.ylabel("")

plt.suptitle("Percentage Distribution of Credit Amount")

plt.show()
#2. AGE

#Categorise the applicants into four groups based on the AGE value (20s, 30s, 40s, 50s, Senior_Citizen). 



#Funstion to Bin 

def age(x) :

    if x <= 29 :

        return '20s'

    if x<= 39 :

        return '30s'

    if x<= 49 :

        return '40s'

    if x<= 59 :

        return '50s'

    if x>= 60 :

        return 'Senior_Citizen'

# Categorise based on categories above

df_app['Bin_AGE'] = df_app['AGE'].apply(age)

df_app.head(5)
#Plotting the BIN_AGE w.r.t Target variable



plt.figure(num=None, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')

plt.subplots_adjust(hspace=0.0, wspace=0.8)

plt.subplot(1,2,1)

plt.gca().set_title("Target_0")

df_app[df_app['TARGET']== '0'].Bin_AGE.value_counts().plot.pie(autopct='%1.1f%%')

plt.ylabel("")

plt.subplot(1,2,2)

plt.gca().set_title("Target_1")

df_app[df_app['TARGET']== '1'].Bin_AGE.value_counts().plot.pie(autopct='%1.1f%%')

plt.ylabel("")

plt.suptitle("Percentage Distribution of AGE")

plt.show()
#selecting only relevent columns from dataset for further analysis



df_app = df_app[['SK_ID_CURR','TARGET','NAME_CONTRACT_TYPE','CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY','CNT_CHILDREN','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE','NAME_TYPE_SUITE','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','AGE',

'YEARS_WRK_EXP','OCCUPATION_TYPE','CNT_FAM_MEMBERS','ORGANIZATION_TYPE']]

df_app.head()
df_app.describe()
#Calculating the imbalance percentage of the TARGET column

round((df_app.TARGET[df_app['TARGET']=='1'].count()) / (df_app.TARGET[df_app['TARGET']=='0'].count()) * 100,2)
df_app.TARGET.value_counts()
#let's divide the dataframe into two sets i.e. Target=1 and Target=0



df_target1=df_app[df_app['TARGET']=='1']



df_target0=df_app[df_app['TARGET']=='0']
# Graphical representation for correlation

plt.figure(figsize=(10,6))

sns.heatmap(df_target1.corr(),cmap="YlGnBu")

plt.title("Target_1")

plt.show()
#Get top 10 correlated variables for Target1 dataframe

df_target1.corr().unstack().reset_index()

corr1 = df_target1.corr()

corr1 = corr1.where(np.triu(np.ones(corr1.shape),k=1).astype(np.bool))

corr1 = corr1.unstack().reset_index()

corr1.columns = ['Var1','Var2','Correlation']

corr1.dropna(subset=['Correlation'],inplace=True)

corr1['Correlation'] = abs(corr1['Correlation'])

corr1.sort_values(by = 'Correlation',ascending=False).head(10)
# Graphical representation for correlation

plt.figure(figsize=(10,6))

sns.heatmap(df_target0.corr(),cmap="YlGnBu")

plt.title("Target_0")

plt.show()
#Get top 10 correlated variables for Target0 dataframe

df_target0.corr().unstack().reset_index()

corr0 = df_target0.corr()

corr0 = corr0.where(np.triu(np.ones(corr0.shape),k=1).astype(np.bool))

corr0 = corr0.unstack().reset_index()

corr0.columns = ['Var1','Var2','Correlation']

corr0.dropna(subset=['Correlation'],inplace=True)

corr0['Correlation'] = abs(corr0['Correlation'])

corr0.sort_values(by = 'Correlation',ascending=False).head(10)
l2 = ['NAME_CONTRACT_TYPE','CODE_GENDER','FLAG_OWN_CAR','CNT_CHILDREN','NAME_TYPE_SUITE','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','OCCUPATION_TYPE','CNT_FAM_MEMBERS']

fig = plt.figure(figsize=(15,35))

fig.subplots_adjust(hspace=1.3, wspace=0.4)

for i in enumerate(l2):

    plt.subplot(6,2,i[0]+1)

    sns.countplot(data=df_app,x=i[1],hue='TARGET').set(title=i[1])

    plt.xticks(rotation=90)
def univariate_analysis(df, col):

    sns.set(style='darkgrid')

    plt.figure(figsize=(25, 5))

    

    plt.subplot(1, 3, 1)

    sns.boxplot(data=df, x=col, orient='v').set(title='Box Plot')

    plt.yscale('log')

    

    plt.subplot(1, 3, 2)

    sns.distplot(df[col].dropna()).set(title='Box Plot')

    plt.yscale('log')
#Plotting for clients with defaults

univariate_analysis(df=df_target1,col='AMT_INCOME_TOTAL')
#Plotting for non-defaulter clients

univariate_analysis(df=df_target0,col='AMT_INCOME_TOTAL')
univariate_analysis(df=df_target1,col='AMT_CREDIT')
univariate_analysis(df=df_target0,col='AMT_CREDIT')
univariate_analysis(df=df_target1,col='AMT_ANNUITY')
univariate_analysis(df=df_target0,col='AMT_ANNUITY')
univariate_analysis(df=df_target1,col='AMT_GOODS_PRICE')
univariate_analysis(df=df_target0,col='AMT_GOODS_PRICE')
univariate_analysis(df=df_target1,col='AGE')
univariate_analysis(df=df_target0,col='AGE')
univariate_analysis(df=df_target1,col='YEARS_WRK_EXP')
univariate_analysis(df=df_target0,col='YEARS_WRK_EXP')
#1. Numerical vs Categorical



# AMT_INCOME_TOTAL vs AMT_INCOME_TYPE

# will see how is the Income for various category and its distribution across defaulters and non defaulters group



plt.figure(figsize=(15,6))

sns.barplot(data=df_app,x='NAME_INCOME_TYPE',y='AMT_INCOME_TOTAL',hue='TARGET')

plt.title("Applicants Total Income across Income types")

plt.show()
#2. Categorical vs Categorical



# AGE vs CODE_GENDER



#1. Distribution of Applicant's Age in years across Gender

plt.figure(figsize=(14, 8))

plt.subplots_adjust(hspace=0.5, wspace=0.5)

plt.suptitle("Distribution of Applicant's Age (in years) across Gender")



plt.subplot(2,1,1)



df = df_target0[df_target0.CODE_GENDER == "M"]

sns.distplot(df['AGE'],kde=False, label='Male',bins = 20)

df = df_target0[df_target0.CODE_GENDER == "F"]

sns.distplot(df['AGE'],kde=False, label='Female',bins = 20)

df = df_target0[df_target0.CODE_GENDER == "XNA"]

sns.distplot(df['AGE'],kde=False, label='Other',bins = 20)

plt.xticks(np.arange(0,80,10))

plt.gca().set_title("Target_0")

plt.legend(prop={'size': 12})



plt.subplot(2,1,2)



df = df_target1[df_target1.CODE_GENDER == "M"]

sns.distplot(df['AGE'],kde=False, label='Male',bins = 20)

df = df_target1[df_target1.CODE_GENDER == "F"]

sns.distplot(df['AGE'],kde=False, label='Female',bins = 20)

df = df_target1[df_target1.CODE_GENDER == "XNA"]

sns.distplot(df['AGE'],kde=False, label='Other',bins = 20)

plt.xticks(np.arange(0,80,10))

plt.gca().set_title("Target_1")

plt.legend(prop={'size': 12})



plt.show()
#3. Income vs work experience



#Target_0



sns.scatterplot(y = df_target0['AMT_INCOME_TOTAL'],x = df_target0['YEARS_WRK_EXP'])

plt.yscale('log')
#Target_1

sns.scatterplot(y = df_target1['AMT_INCOME_TOTAL'],x = df_target1['YEARS_WRK_EXP'])

plt.yscale('log')
#load prev data file 



df_prev = pd.read_csv('../input/loanapplicationdata/previous_application.csv')

df_prev.shape
df_prev.describe()
df_prev.head()
# Select only relevent columns from previous application data

df_prev = df_prev[['SK_ID_PREV','SK_ID_CURR','NAME_CONTRACT_TYPE','AMT_ANNUITY','AMT_APPLICATION','AMT_CREDIT','AMT_GOODS_PRICE','FLAG_LAST_APPL_PER_CONTRACT','NAME_CASH_LOAN_PURPOSE','NAME_CONTRACT_STATUS','DAYS_DECISION','NAME_PAYMENT_TYPE','CODE_REJECT_REASON','NAME_CLIENT_TYPE','NAME_GOODS_CATEGORY']]
# change data type 

df_prev['SK_ID_CURR'] = df_prev['SK_ID_CURR'].astype('str')

df_prev['SK_ID_PREV'] = df_prev['SK_ID_PREV'].astype('str')

df_prev.dtypes
#merge the current application and prev application data on column SK_ID_CURR



master_df = pd.merge(df_app, df_prev, on='SK_ID_CURR', how='left')

master_df.shape
master_df.dtypes
#Days decision should be positive, handle negative sign



master_df['DAYS_DECISION'] = abs(master_df['DAYS_DECISION'])
#Divide master data (Current application and their information of previous applications) into Target 0 and 1

#1 = Applicants having payment difficulties in current application

#0 = other cases in current application



master_df_t1 = master_df[master_df['TARGET']== '1']

master_df_t0 = master_df[master_df['TARGET']== '0']



print(master_df_t1.shape)

print('\n')

print(master_df_t0.shape)
#1. NAME_CONTRACT_STATUS 



# Overall % contribution of each previous application's status for rows associated with Target category 1 or 0

# Target 1(curr app have payment diffifulty) or 0 (other cases)



plt.figure(num=None, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')

plt.subplots_adjust(hspace=0.5, wspace=0.5)

plt.subplot(1,2,1)

plt.gca().set_title("Target_0")

master_df_t0['NAME_CONTRACT_STATUS'].value_counts().plot.pie(autopct='%1.1f%%')

plt.ylabel("")

plt.subplot(1,2,2)

plt.gca().set_title("Target_1")

master_df_t1['NAME_CONTRACT_STATUS'].value_counts().plot.pie(autopct='%1.1f%%')

plt.ylabel("")

plt.suptitle("Percentage Distribution of Contract Status of Previous applications")

plt.show()
# 2. CODE_REJECT_REASON



temp1= master_df_t1[master_df_t1['NAME_CONTRACT_STATUS']=="Refused"]

temp0= master_df_t0[master_df_t0['NAME_CONTRACT_STATUS']=="Refused"]

def add_labels_per(ax):

    total = sum([p.get_height() for p in ax.patches])

    for p in ax.patches:        

        height = p.get_height()

        ax.text(p.get_x() + p.get_width()/2., height+3, '{:1.2f}'.format((height/total)*100), fontsize=12, ha='center', va='bottom')

                

plt.figure(num=None, figsize=(20, 6), dpi=80, facecolor='w', edgecolor='k')

plt.subplot(1,2,1)

plt.gca().set_title("Target_0")

sns.countplot(x='CODE_REJECT_REASON',data = temp0)

add_labels_per(plt.gca())

plt.subplot(1,2,2)

plt.gca().set_title("Target_1")

sns.countplot(x='CODE_REJECT_REASON',data = temp1)

plt.suptitle("Distribution of overall Code Reject Reason of previous application",fontsize = 18)

add_labels_per(plt.gca())

plt.show()

#3. DAYS_DECISION



plt.figure(num=None, figsize=(15, 12), dpi=80, facecolor='w', edgecolor='k')

plt.subplots_adjust(hspace=0.3, wspace=0.3)

plt.subplot(2,2,1)

plt.gca().set_title("Target_0")

master_df_t0.DAYS_DECISION.value_counts().plot.hist()



plt.subplot(2,2,2)

plt.gca().set_title("Target_1")

master_df_t1.DAYS_DECISION.value_counts().plot.hist()

plt.suptitle("Distribution of number of days taken for decision of previous application",fontsize = 18)

plt.show()
#AGE group for Previous applications in refused status



plt.figure(num=None, figsize=(15, 12), dpi=80, facecolor='w', edgecolor='k')

plt.subplots_adjust(hspace=0.3, wspace=0.3)

plt.subplot(2,2,1)

plt.gca().set_title("Target_0")

master_df_t0[master_df_t0['NAME_CONTRACT_STATUS']== "Refused"].AGE.plot.hist(bins = 30)



plt.subplot(2,2,2)

plt.gca().set_title("Target_1")

master_df_t1[master_df_t1['NAME_CONTRACT_STATUS']=="Refused"].AGE.plot.hist(bins = 30)

plt.suptitle("AGE group for Previous applications in refused status",fontsize = 18)

plt.show()
# NAME_GOODS_CATEGORY



plt.figure(num=None, figsize=(25, 15), dpi=80, facecolor='w', edgecolor='k')

plt.subplots_adjust(hspace=0.9, wspace=0.5)



plt.subplot(2,1,1)

plt.gca().set_title("Target_0")

sns.countplot(x='NAME_GOODS_CATEGORY',data = master_df_t0)

plt.xticks(rotation = 90,fontsize = 15)

add_labels_per(plt.gca())

plt.subplot(2,1,2)

plt.gca().set_title("Target_1")

sns.countplot(x='NAME_GOODS_CATEGORY',data = master_df_t1)

plt.xticks(rotation = 90,fontsize = 15)

add_labels_per(plt.gca())

plt.show()
#NAME_CLIENT_TYPE



plt.figure(num=None, figsize=(8, 4), dpi=80, facecolor='w', edgecolor='k')

plt.subplots_adjust(hspace=0.5, wspace=0.5)

plt.subplot(1,2,1)

plt.gca().set_title("Target_0")

master_df_t0['NAME_CLIENT_TYPE'].value_counts().plot.pie(autopct='%1.1f%%')

plt.ylabel("")

plt.subplot(1,2,2)

plt.gca().set_title("Target_1")

master_df_t1['NAME_CLIENT_TYPE'].value_counts().plot.pie(autopct='%1.1f%%')

plt.ylabel("")

plt.suptitle("Percentage Distribution of Client Type")

plt.show()
#NAME_PAYMENT_TYPE



plt.figure(num=None, figsize=(20, 6), dpi=80, facecolor='w', edgecolor='k')

plt.subplot(1,2,1)

plt.gca().set_title("Target_0")

sns.countplot(x='NAME_PAYMENT_TYPE',data = master_df_t0)

plt.xticks(rotation = 45,fontsize = 15)

add_labels_per(plt.gca())

plt.subplot(1,2,2)

plt.gca().set_title("Target_1")

sns.countplot(x='NAME_PAYMENT_TYPE',data = master_df_t1)

plt.xticks(rotation = 45,fontsize = 15)

plt.suptitle("NAME_PAYMENT_TYPE for previous applications",fontsize = 18)

add_labels_per(plt.gca())

plt.show()
# NAME_CASH_LOAN_PURPOSE



# % contribution of CASH_LOAN_PURPOSE (XNA excluded as around 80% rows are marked XNA)



plt.figure(num=None, figsize=(25, 15), dpi=80, facecolor='w', edgecolor='k')

plt.subplots_adjust(hspace=1.0, wspace=0.5)



plt.subplot(2,1,1)

plt.gca().set_title("Target_0")

sns.countplot(x='NAME_CASH_LOAN_PURPOSE',data = master_df_t0[(master_df_t0['NAME_CONTRACT_TYPE_y']=="Cash loans") & (master_df_t0['NAME_CASH_LOAN_PURPOSE']!="XNA")])

plt.xticks(rotation = 90,fontsize = 15)

plt.xlabel("")

add_labels_per(plt.gca())

plt.subplot(2,1,2)

plt.gca().set_title("Target_1")

sns.countplot(x='NAME_CASH_LOAN_PURPOSE',data = master_df_t1[(master_df_t1['NAME_CONTRACT_TYPE_y']=="Cash loans") & (master_df_t1['NAME_CASH_LOAN_PURPOSE']!="XNA")])

plt.xticks(rotation = 90,fontsize = 15)

plt.xlabel("NAME_CASH_LOAN_PURPOSE",fontsize = 15)

add_labels_per(plt.gca())

plt.suptitle("% Distribution per Cash Loan Purpose",fontsize = 20)

plt.show()
#Numerical vs Categorical



#Total income for across NAME_CONTRACT_STATUS



plt.figure(figsize = (10,6))

sns.barplot(y = "AMT_CREDIT_y",x  = 'NAME_CLIENT_TYPE' , data = master_df[(master_df['NAME_CONTRACT_TYPE_y']== "Cash loans") | (master_df['NAME_CONTRACT_TYPE_y']== "Revolving loans")],hue = 'TARGET')

#add_labels_per(plt.gca())

plt.title("Credit Amount of previous applications across Client Type",fontsize = 15)

plt.show()