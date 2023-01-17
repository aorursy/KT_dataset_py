# import libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



pd.set_option('display.max_columns',125)

pd.set_option('display.max_rows',200)
# load application_data file

application_data=pd.read_csv("../input/credit-eda/application_data.csv") 

application_data.head()
# check structure of data

print(application_data.shape)
print(application_data.info())
application_data.describe()
#find the percentage of missing values for all the columns

round(100*application_data.isnull().sum()/len(application_data),2)
# remove columns with high missing percentage

# considering 50% as the threshold value

application_data= application_data.loc[:, 100*application_data.isnull().sum()/len(application_data) < 50]

# checking for shape of the data

application_data.shape
# checking for percentage of null values

round(100*application_data.isnull().sum()/len(application_data),2)
# retriving the columns which has any null values

application_data_columns=application_data.columns[application_data.isnull().any()].tolist()

application_data[application_data_columns].isnull().sum()*100/len(application_data)
# AMT_ANNUITY

print(application_data.AMT_ANNUITY.head()) # correct datatype

print(application_data.AMT_ANNUITY.describe())

application_data.boxplot(column=['AMT_ANNUITY'])

plt.show()

# from box plot it seems, it has lot of outliers so considering median measure

application_data.AMT_ANNUITY.median()

# we can impute 24903(median) value in place of missing values
# AMT_GOODS_PRICE

print(application_data.AMT_GOODS_PRICE.head()) # correct datatype

print(application_data.AMT_GOODS_PRICE.describe())

application_data.boxplot(column=['AMT_GOODS_PRICE'])

plt.show()

# from box plot it seems, it has lot of outliers so considering median measure

application_data.AMT_GOODS_PRICE.median()

# we can impute 450000.0 value in place of missing values
# NAME_TYPE_SUITE

print(application_data.NAME_TYPE_SUITE.head()) # correct datatype

print(application_data.NAME_TYPE_SUITE.describe())

# since it is acategorical value, considering mode measure to impute missing values

print(application_data.NAME_TYPE_SUITE.mode())

# considering the value to be imputed is - Unaccompanied
#CNT_FAM_MEMBERS

print(application_data.CNT_FAM_MEMBERS.head()) # correct datatype

print(application_data.CNT_FAM_MEMBERS.describe())

application_data.boxplot(column=['CNT_FAM_MEMBERS'])

plt.show()

# from box plot it seems, it has lot of outliers so considering median measure

application_data.CNT_FAM_MEMBERS.median()

# we can impute "2.0" value in place of missing values
#EXT_SOURCE_2

print(application_data.EXT_SOURCE_2.head()) # correct datatype

print(application_data.EXT_SOURCE_2.describe())

application_data.boxplot(column=['EXT_SOURCE_2'])

plt.show()

# from box plot it seems, mean and median are almost near and no outliers but there is some tilt towards outliers so go with median

application_data.EXT_SOURCE_2.median()

# so, we can impute 0.5659614260608526 value in place of missing values
# checking the datatypes of all the columns and change the data type like negative age and date

print(application_data.info())
application_data.head()
# finding count of unique values in each column

print(application_data.nunique().sort_values())
# converting negative DAYS_BIRTH value to positive value

application_data['DAYS_BIRTH']=application_data['DAYS_BIRTH'].abs()

# converting negative DAYS_EMPLOYED value to positive value

application_data['DAYS_EMPLOYED']=application_data['DAYS_EMPLOYED'].abs()

# converting negative DAYS_REGISTRATION value to positive value

application_data['DAYS_REGISTRATION']=application_data['DAYS_REGISTRATION'].abs()

# converting negative DAYS_ID_PUBLISH value to positive value

application_data['DAYS_ID_PUBLISH']=application_data['DAYS_ID_PUBLISH'].abs()

# converting negative DAYS_LAST_PHONE_CHANGE value to positive value

application_data['DAYS_LAST_PHONE_CHANGE']=application_data['DAYS_LAST_PHONE_CHANGE'].abs()

application_data.head()
# conversion of columns integer to categorical

for col in application_data.columns:

    if application_data[col].nunique() <= 3: # here considering columns with 3 unique values as categorical variables

        application_data[col] = application_data[col].astype(object)



application_data.info() 

application_data.head()
#checked columns with unique count greater than 3 to see if any int/float column is

#worngly read as object, but no such column is found
plt.boxplot(application_data['CNT_CHILDREN'])

plt.show()

# From box plot, we can conclude that there exists values which are above upper whisker(maximum) considered to be as outliers. 

Q1 = application_data['CNT_CHILDREN'].quantile(0.25)

Q3 = application_data['CNT_CHILDREN'].quantile(0.75)

IQR = Q3 - Q1

lowerwhisker=(Q1 - 1.5 * IQR)

upperwhisker=(Q3 + 1.5 * IQR)

# According to Statictics the values above the upper whisker and below the lower whisker are considered as outliers

#and as we can see in plot outliers are present only above the upper wisker so considering them as outliers

print("The values greater than {} are considered to be outliers,since count of children cannot be in decimals we can conclude that count greater than 3 can be an outlier".format(upperwhisker))
plt.boxplot(application_data['AMT_CREDIT'])

plt.title('AMT_CREDIT')

plt.show()

# From box plot, we can conclude that there exists values which are above upper whisker(maximum) considered to be as outliers. 

Q1 = application_data['AMT_CREDIT'].quantile(0.25)

Q3 = application_data['AMT_CREDIT'].quantile(0.75)

IQR = Q3 - Q1

lowerwhisker=(Q1 - 1.5 * IQR)

upperwhisker=(Q3 + 1.5 * IQR)



# the values above the upper whisker and below the lower whisker are considered as outliers

#and as we can see in plot outliers are present only above the upper wisker so considering them as outliers

#print("Lowerwhisker:{}".format(lowerwhisker))

'''according to statistics the the values less than lower whisker value -537975.0 considered as outlier, 

   as credit amount cannot be negative we consider amount greater than  1616625.0 as an outlier.'''

print("The amount credited greater than {} can be considered as an outlier".format(upperwhisker))
application_data['AMT_CREDIT'].describe()

application_data['AMT_CREDIT'].max()
data=application_data['AMT_ANNUITY']

filtered_data = data[~np.isnan(data)]

plt.boxplot(filtered_data)

plt.show()

# From box plot, we can conclude that there exists values which are above upper whisker(maximum) considered to be as outliers. 

Q1 = application_data['AMT_ANNUITY'].quantile(0.25)

Q3 = application_data['AMT_ANNUITY'].quantile(0.75)

IQR = Q3 - Q1

lowerwhisker=(Q1 - 1.5 * IQR)

upperwhisker=(Q3 + 1.5 * IQR)

# the values above the upper whisker and below the lower whisker are considered as outliers

#and as we can see in plot outliers are present only above the upper wisker so considering them as outliers

'''according to statistics the the values less than lower whisker value -10584.0 considered as outlier, 

   as amount cannot be negative we consider count greater than  61704.0 as an outlier.'''

print("Population relative count greater than {} is considered to be an outlier".format(upperwhisker))
plt.boxplot(application_data['REGION_POPULATION_RELATIVE'])

plt.show()

# From box plot, we can conclude that there exists values which are above upper whisker(maximum) considered to be as outliers. 

Q1 = application_data['REGION_POPULATION_RELATIVE'].quantile(0.25)

Q3 = application_data['REGION_POPULATION_RELATIVE'].quantile(0.75)

IQR = Q3 - Q1

lowerwhisker=(Q1 - 1.5 * IQR)

upperwhisker=(Q3 + 1.5 * IQR)

# the values above the upper whisker and below the lower whisker are considered as outliers

#and as we can see in plot outliers are present only above the upper wisker so considering them as outliers

'''according to statistics the the values less than lower whisker value -0.017979500000000002 considered as outlier, 

   as people relative cannot be negative we consider count greater than  0.056648500000000004 as an outlier.'''

print("Population relative count greater than {} is considered to be an outlier".format(upperwhisker))
data=application_data['AMT_GOODS_PRICE']

filtered_data = data[~np.isnan(data)]

plt.boxplot(filtered_data)

plt.show()

# From box plot, we can conclude that there exists values which are above upper whisker(maximum) considered to be as outliers. 

Q1 = application_data['AMT_GOODS_PRICE'].quantile(0.25)

Q3 = application_data['AMT_GOODS_PRICE'].quantile(0.75)

IQR = Q3 - Q1

lowerwhisker=(Q1 - 1.5 * IQR)

upperwhisker=(Q3 + 1.5 * IQR)

# the values above the upper whisker and below the lower whisker are considered as outliers

#and as we can see in plot outliers are present only above the upper wisker so considering them as outliers

'''according to statistics the the values less than lower whisker value -423000.0 considered as outlier, 

   as amount cannot be negative we consider count greater than  1341000.0 as an outlier.'''

print("Population relative count greater than {} is considered to be an outlier".format(upperwhisker))
application_data.head(10)
# Binning of continuous variables.Check if you need to bin any variable in different categories.Do this for atleast 2 variables

# AMT_INCOME_TOTAL

q1=application_data['AMT_INCOME_TOTAL'].quantile(0.25)

q2=application_data['AMT_INCOME_TOTAL'].quantile(0.50)

q3=application_data['AMT_INCOME_TOTAL'].quantile(0.75)

m=application_data['AMT_INCOME_TOTAL'].max()



# Binning AMT_INCOME_TOTAL into AMT_INCOME_TOTAL_bin so we don't loose data and have binned values

application_data['AMT_INCOME_TOTAL_bin'] = pd.cut(application_data['AMT_INCOME_TOTAL'],[q1, q2, q3,m ], labels = ['Low', 'medium', 'High'])

print(application_data.AMT_INCOME_TOTAL_bin.value_counts())
# AMT_CREDIT

q1=application_data['AMT_CREDIT'].quantile(0.25)

q2=application_data['AMT_CREDIT'].quantile(0.50)

q3=application_data['AMT_CREDIT'].quantile(0.75)

m=application_data['AMT_CREDIT'].max()



# Binning AMT_CREDIT into AMT_CREDIT_bin so we don't loose data and have binned values

application_data['AMT_CREDIT_bin'] = pd.cut(application_data['AMT_CREDIT'],[q1, q2, q3,m ], labels = ['Low', 'medium', 'High'])

print(application_data.AMT_CREDIT_bin.value_counts())
application_data.head()
#Checking the imbalance percentage.

print(100*application_data.TARGET.value_counts()/ len(application_data))

(application_data.TARGET.value_counts()/ len(application_data)).plot.bar()

plt.xticks(rotation=0)

plt.show()

# In application_data there exists 91.927118% of "not default" and 8.072882% of "default" customers.
# Divide the data into two sets, i.e., Target-1 and Target-0

application_data_1 = application_data[application_data['TARGET']==1]

application_data_0 = application_data[application_data['TARGET']==0]
#Perfeorming analysis for one column at a time

# perform univariate analysis for categoriacal variables for both 0 and 1

# WEEKDAY_APPR_PROCESS_START (categorical ordered variable)

# for TARGET=0

application_data_0.WEEKDAY_APPR_PROCESS_START.value_counts(normalize=True).plot.bar()

plt.title('for non-default')

plt.show()

# from the graph we can conclude that application starting processes will be less in saturday and sunday.

# for TARGET=1

application_data_1.WEEKDAY_APPR_PROCESS_START.value_counts(normalize=True).plot.bar()

plt.title('for default')

plt.show()

# from the graph we can conclude that application starting processes are generally less in saturday and sunday.

# NAME_EDUCATION_TYPE (categorical ordered variable)

# for Target=0

application_data_0.NAME_EDUCATION_TYPE.value_counts(normalize=True).plot.pie()

plt.tight_layout()

plt.title('for non-default')

plt.show()

# from the plot below, we can conclude that secondary/special educated people are applying loans in high in number.

# for Target=1

application_data_1.NAME_EDUCATION_TYPE.value_counts(normalize=True).plot.pie()

plt.tight_layout()

plt.title('for default')

plt.show()

# from the plot below, we can conclude that secondary/special educated people are applying loans high in number.

#and Academic degree educated people are applying loan in least count.

# for both target= 0 and 1
# NAME_FAMILY_STATUS 

# for TARGET=0

application_data_0.NAME_FAMILY_STATUS.value_counts(normalize=True).plot.barh()

plt.title('for non-default')

plt.show()

# for TARGET=1

application_data_1.NAME_FAMILY_STATUS.value_counts(normalize=True).plot.barh()

plt.title('for default')

plt.show()

# the order of both default and not default customers is same i.e., Married,Single/not married,civil marriage,seperated,widow

# It also shows that there exists few(1 or 2) unknown values in not default client family status.



# We can say more married people tend to take more Loan as compaired to other categories

# and being married is not impacting default and not defaulting
# NAME_INCOME_TYPE

# for TARGET=0

application_data_0.NAME_INCOME_TYPE.value_counts(normalize=True).plot.barh()

plt.title('for non-default')

plt.show()

# for TARGET=1

application_data_1.NAME_INCOME_TYPE.value_counts(normalize=True).plot.barh()

plt.title('for default')

plt.show()

# from the graphs below, we can conclude that

# Pensioner of not default case are high in number compared to Pensioner of default case.

#It seems there exists both loss and profit due to Pension people to the Bank.

# It also shows that majority of defaulters income type is working.

#and at the same time there is good income to bank from working people.
# NAME_HOUSING_TYPE

# for TARGET=0

application_data_0.NAME_HOUSING_TYPE.value_counts(normalize=True).plot.barh()

plt.title('for non-default')

plt.show()

# for TARGET=1

application_data_1.NAME_HOUSING_TYPE.value_counts(normalize=True).plot.barh()

plt.title('for default')

plt.show()

# from graph we can conclude that there exists people who have own house

# lies in both default and non default.
#considering 10 categorical columns

categorical_columns=['NAME_CONTRACT_TYPE','CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY',

                     'NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE',

                    'WEEKDAY_APPR_PROCESS_START','AMT_CREDIT_bin','AMT_INCOME_TOTAL_bin']



plt.figure(figsize=(22,25))

for i in (enumerate(categorical_columns)):

    plt.subplot(len(categorical_columns)//2,2,i[0]+1)

    sns.countplot(x=i[1],hue='TARGET',data=application_data)

    plt.yscale('log')

    #plt.xticks(rotation=90)

plt.show()

#the XNA in Code_gender is not known if it is NA or a category so leaving it as it is.
#considering 10 continous numerical columns

continous_columns=['AMT_ANNUITY','AMT_GOODS_PRICE','CNT_FAM_MEMBERS',

                  'DAYS_LAST_PHONE_CHANGE','DAYS_ID_PUBLISH','DAYS_BIRTH','HOUR_APPR_PROCESS_START',

                  'DAYS_EMPLOYED','AMT_CREDIT','AMT_INCOME_TOTAL']

plt.figure(figsize=(22,25))

for i in (enumerate(continous_columns)):

    plt.subplot(len(continous_columns)//2,2,i[0]+1)

    sns.distplot(application_data_1[i[1]].dropna(),hist=False,label='Target : default')

    sns.distplot(application_data_0[i[1]].dropna(),hist=False,label='Target : no default')

plt.show()    
#application_data_1.corr()

application_data_1.corr().unstack().reset_index().sort_values(by=0,ascending=False)

#there are many repeted values
#finding Top 10 Correlated values for defalut(1)

# finding correlation so that  there are no repeated values

corr=application_data_1.corr()

corrdf=corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))

corrdf=corrdf.unstack().reset_index()

corrdf.columns=['Var1','Var2','Coorelation']

corrdf.dropna(subset=['Coorelation'],inplace=True)

corrdf['Coorelation']=round(corrdf['Coorelation'],2)

corrdf['Coorelation']=abs(corrdf['Coorelation']) #converting -ve values to +ve because they are same

corrdf.sort_values(by='Coorelation',ascending=False).head(10)
#finding Top 10 Correlated values for non-defalut(0)

corr=application_data_0.corr()

corrdf=corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))

corrdf=corrdf.unstack().reset_index()

corrdf.columns=['Var1','Var2','Coorelation']

corrdf.dropna(subset=['Coorelation'],inplace=True)

corrdf['Coorelation']=round(corrdf['Coorelation'],2)

corrdf['Coorelation']=abs(corrdf['Coorelation']) #converting -ve values to +ve because they are same

corrdf.sort_values(by='Coorelation',ascending=False).head(10)
application_data.head()
#Bi-variate categorical plots



table_1= pd.crosstab(index=application_data['TARGET'],columns=application_data['NAME_CONTRACT_TYPE'])

print(table_1)

table_1.plot(kind="bar", figsize=(5,5),stacked=False)

plt.xticks(rotation=0)

plt.show()

# High number of cash loans
table_2= pd.crosstab(index=application_data['TARGET'],columns=application_data['CODE_GENDER'])

print(table_2)

table_2.plot(kind="bar", figsize=(5,5),stacked=False)

plt.xticks(rotation=0)

plt.show()

#Females take more loans
table_3= pd.crosstab(index=application_data['TARGET'],columns=application_data['NAME_TYPE_SUITE'])

print(table_3)

table_3.plot(kind="bar", figsize=(5,5),stacked=False)

plt.xticks(rotation=0)

plt.show()

# Most of the people come alone when taking a loan
table_4= pd.crosstab(index=application_data['TARGET'],columns=application_data['NAME_INCOME_TYPE'])

print(table_4)

table_4.plot(kind="bar", figsize=(5,5),stacked=False)

plt.show()

# working people take more loans
table_5= pd.crosstab(index=application_data['TARGET'],columns=application_data['NAME_HOUSING_TYPE'])

print(table_5)

table_5.plot(kind="bar", figsize=(5,5),stacked=False)

plt.show()

# People having house/appartment tend to take more loans
application_data.head()
#Bi-variate continous plots

continous_columns=['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE',

                  'DAYS_EMPLOYED','DAYS_BIRTH','DAYS_LAST_PHONE_CHANGE','HOUR_APPR_PROCESS_START',

                  'DAYS_ID_PUBLISH','DAYS_REGISTRATION']

plt.figure(figsize=(15,25))

for i in (enumerate(continous_columns)):

    plt.subplot(len(continous_columns)//2,2,i[0]+1)

    sns.boxplot(x='TARGET',y=application_data[i[1]].dropna(),data=application_data)

    plt.yscale('log')

plt.show() 
previous_data=pd.read_csv("../input/credit-eda/previous_application.csv")

previous_data.head()
# data check with respect to size,data type etc,.

print(previous_data.shape)

previous_data.info()

previous_data.describe()
# checking of missing values percentage

round((100*previous_data.isnull().sum()/len(previous_data)),2)
# removing those columns which are having null percentage greater than 50

# AMT_DOWN_PAYMENT,RATE_DOWN_PAYMENT,RATE_INTEREST_PRIMARY,RATE_INTEREST_PRIVILEGED 

previous_data=previous_data.drop(['AMT_DOWN_PAYMENT','RATE_DOWN_PAYMENT','RATE_INTEREST_PRIMARY','RATE_INTEREST_PRIVILEGED'], axis = 1)

previous_data.info()
# converting -ve values to +ve

previous_data['DAYS_DECISION']=previous_data['DAYS_DECISION'].abs()

previous_data['SELLERPLACE_AREA']=previous_data['SELLERPLACE_AREA'].abs()

previous_data['DAYS_FIRST_DUE']=previous_data['DAYS_FIRST_DUE'].abs()

previous_data['DAYS_LAST_DUE_1ST_VERSION']=previous_data['DAYS_LAST_DUE_1ST_VERSION'].abs()

previous_data['DAYS_LAST_DUE']=previous_data['DAYS_LAST_DUE'].abs()

previous_data['DAYS_TERMINATION']=previous_data['DAYS_TERMINATION'].abs()

previous_data['DAYS_FIRST_DRAWING']=previous_data['DAYS_FIRST_DRAWING'].abs()
(previous_data.NAME_CONTRACT_STATUS.value_counts()/len(previous_data)).plot.bar()

plt.show()
# making a left join because we need all the rows in application data 

# by making this left join we get historical application data for each applicant.

# if we made a inner join we would loose the data of a new customer who doesn't have a previous record.

# Current data will get duplicated the exact number of times it is found in previous application data.

# with this in mind we are moving forward.



merged_df=pd.merge(application_data,previous_data,how='left',on='SK_ID_CURR',suffixes=('_Current', '_Previous'))

merged_df.head()
# Univariate Categorical analysis

categorical_columns=['NAME_CONTRACT_TYPE_Current','NAME_CONTRACT_TYPE_Previous',

                     'NAME_TYPE_SUITE_Current','NAME_TYPE_SUITE_Previous',

                     'WEEKDAY_APPR_PROCESS_START_Current','WEEKDAY_APPR_PROCESS_START_Previous',

                    'AMT_INCOME_TOTAL_bin','AMT_CREDIT_bin','NAME_YIELD_GROUP','NAME_CLIENT_TYPE']





plt.figure(figsize=(22,25))

for i in (enumerate(categorical_columns)):

    plt.subplot(len(categorical_columns)//2,2,i[0]+1)

    sns.countplot(x=i[1],hue='NAME_CONTRACT_STATUS',data=merged_df)

    #lt.yscale('log')

    #plt.xticks(rotation=90)

plt.show()

# Univariate Numerical analysis

continous_columns=['AMT_CREDIT_Previous','AMT_CREDIT_Current','AMT_ANNUITY_Current','AMT_ANNUITY_Previous',

                   'AMT_GOODS_PRICE_Current','AMT_GOODS_PRICE_Previous','CNT_FAM_MEMBERS','CNT_CHILDREN',

                  'HOUR_APPR_PROCESS_START_Previous','HOUR_APPR_PROCESS_START_Current']

plt.figure(figsize=(22,25))

for i in (enumerate(continous_columns)):

    plt.subplot(len(continous_columns)//2,2,i[0]+1)

    sns.distplot(merged_df.loc[merged_df.NAME_CONTRACT_STATUS=='Approved',:][i[1]].dropna(),hist=False,label='Approved')

    sns.distplot(merged_df.loc[merged_df.NAME_CONTRACT_STATUS=='Canceled',:][i[1]].dropna(),hist=False,label='Canceled',kde_kws={'bw':0.1})

    sns.distplot(merged_df.loc[merged_df.NAME_CONTRACT_STATUS=='Refused',:][i[1]].dropna(),hist=False,label='Refused',kde_kws={'bw':0.1})

    # we added kde_kws={'bw':0.1} in parameter to overcome bandwidth limitation.

    sns.distplot(merged_df.loc[merged_df.NAME_CONTRACT_STATUS=='Unused offer',:][i[1]].dropna(),hist=False,label='Unused offer')



plt.show() 
table_6= pd.crosstab(index=merged_df['NAME_CONTRACT_STATUS'],columns=merged_df['NAME_CONTRACT_TYPE_Current'])

print(table_6)

table_6.plot(kind="bar", figsize=(5,5),stacked=False)

plt.show()

#Cash loans have the highest count of Approved loans
table_9= pd.crosstab(index=merged_df['NAME_CONTRACT_STATUS'],columns=merged_df['NAME_INCOME_TYPE'])

print(table_9)

table_9.plot(kind="bar", figsize=(5,5),stacked=False)

plt.show()

# Highest number of approvals for working applicant
table_10= pd.crosstab(index=merged_df['NAME_CONTRACT_STATUS'],columns=merged_df['NAME_EDUCATION_TYPE'])

print(table_10)

table_10.plot(kind="bar", figsize=(5,5),stacked=False)

plt.show()

# Highest number of approvals for Secondary/secondary special educated applicant
table_11= pd.crosstab(index=merged_df['NAME_CONTRACT_STATUS'],columns=merged_df['NAME_FAMILY_STATUS'])

print(table_11)

table_11.plot(kind="bar", figsize=(5,5),stacked=False)

plt.show()

# Highest number of approvals for Married applicant
table_12= pd.crosstab(index=merged_df['NAME_CONTRACT_STATUS'],columns=merged_df['NAME_HOUSING_TYPE'])

print(table_12)

table_12.plot(kind="bar", figsize=(5,5),stacked=False)

plt.show()

# Highest number of approvals for House/apartment owner.
table_15= pd.crosstab(index=merged_df['NAME_CONTRACT_STATUS'],columns=merged_df['NAME_CONTRACT_TYPE_Previous'])

print(table_15)

table_15.plot(kind="bar", figsize=(5,5),stacked=False)

plt.show()

# Highest number of approvals for Consumer Loans.
table_17= pd.crosstab(index=merged_df['NAME_CONTRACT_STATUS'],columns=merged_df['NAME_CLIENT_TYPE'])

print(table_17)

table_17.plot(kind="bar", figsize=(5,5),stacked=False)

plt.show()

# repeated applications got approved most number of times
#Bi-variate continous plots

continous_columns=['AMT_ANNUITY_Current','AMT_ANNUITY_Previous',

                   'AMT_GOODS_PRICE_Current','AMT_GOODS_PRICE_Previous','CNT_FAM_MEMBERS','CNT_CHILDREN',

                  'HOUR_APPR_PROCESS_START_Previous','HOUR_APPR_PROCESS_START_Current',

                   'AMT_CREDIT_Current','AMT_CREDIT_Previous']

                   #'AMT_INCOME_TOTAL']

plt.figure(figsize=(15,25))

for i in (enumerate(continous_columns)):

    plt.subplot(len(continous_columns)//2,2,i[0]+1)

    sns.boxplot(x='NAME_CONTRACT_STATUS',y=merged_df[i[1]].dropna(),data=merged_df)

plt.show() 