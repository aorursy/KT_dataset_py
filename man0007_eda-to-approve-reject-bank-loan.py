import pandas as pd



dataset_path = "../input/customers details for bank loan approval/Customers Details for Bank loan Approval/"



df_application_data = pd.read_csv(dataset_path+'application_data.csv')



df_previous_application = pd.read_csv(dataset_path+'previous_application.csv')
df_application_data.head() # glancing the dataset application_data
df_previous_application.head() # glancing the dataset previous_application
# Cheking the shape of the datasets (application_data and previous_application)



print('shape of application_data: ',df_application_data.shape)



print('shape of previous_application: ',df_previous_application.shape)
# Displaying the top5 columns with the highest percentage of missing values in application_data



percent_missing_app = df_application_data.isnull().sum() * 100 / len(df_application_data)



percent_missing_app.sort_values(ascending=False).head() # please remove head() to display all columns
# Displaying the top5 columns with the highest percentage of missing values in previous_application



percent_missing_app = df_previous_application.isnull().sum() * 100 / len(df_previous_application)



percent_missing_app.sort_values(ascending=False).head() # please remove head() to display all columns
# Identifying columns that has 50 percent of the Null values in df_application_data



row_count_50perc_app = df_application_data.shape[0]/2 # Finding the count of 50% of rows



Colms_with_null_50perc_more_app = df_application_data.columns[df_application_data.isnull().sum()>row_count_50perc_app]



print('Total count of columns with 50% null value in df_application_data: ',len(Colms_with_null_50perc_more_app))



# Identifying columns that has 50 percent of the Null values in df_previous_application



row_count_50perc_prev = df_previous_application.shape[0]/2 # Finding the count of 50% of rows



Colms_with_null_50perc_more_prev = df_previous_application.columns[df_previous_application.isnull().sum()>row_count_50perc_prev]



print('Total count of columns with 50% null value in df_previous_application: ',len(Colms_with_null_50perc_more_prev))
# Dropping columns with more than 50% of null values in both dataset



df_application_data = df_application_data.drop(Colms_with_null_50perc_more_app, axis=1)



df_previous_application = df_previous_application.drop(Colms_with_null_50perc_more_prev, axis=1)



# Checking the shape after dropiing the columns



print('shape of application_data after removing 50% of null valued columns: ',df_application_data.shape)



print('shape of previous_application after removing 50% of null valued columns: ',df_previous_application.shape)
# Identifying columns that has 13 percent or less of the Null values in df_application_data



row_count_13perc_app = df_application_data.shape[0] * 0.13 # Finding the count of 13% of rows



Colms_with_null_13perc_or_less_app = df_application_data.columns[df_application_data.isnull().sum()<row_count_13perc_app]



print('Total count of columns with 13% or less null value in df_application_data: ',len(Colms_with_null_13perc_or_less_app))



print('\ncolumns are: ', Colms_with_null_13perc_or_less_app)



# Identifying columns that has 13 percent or less of the Null values in df_previous_application



row_count_13perc_prev = df_previous_application.shape[0]*0.13 # Finding the count of 13% of rows



Colms_with_null_13perc_or_less_prev = df_previous_application.columns[df_previous_application.isnull().sum()<row_count_13perc_prev]



print('\nTotal count of columns with 13% or less null value in df_prev_application: ',len(Colms_with_null_13perc_or_less_prev))



print('\ncolumns are: ', Colms_with_null_13perc_or_less_prev)
# categorical columns in application_data dataset



category_clmns_app_data = ['SK_ID_CURR','TARGET','NAME_CONTRACT_TYPE','CODE_GENDER','FLAG_OWN_CAR',

                                     'FLAG_OWN_REALTY','CNT_CHILDREN','NAME_TYPE_SUITE','NAME_INCOME_TYPE',

                                     'NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE',

                                     'REGION_RATING_CLIENT','FLAG_MOBIL','FLAG_EMP_PHONE','FLAG_WORK_PHONE',

                                     'FLAG_CONT_MOBILE','FLAG_PHONE','FLAG_EMAIL','ORGANIZATION_TYPE','FLAG_MOBIL',

                                     'FLAG_EMP_PHONE','FLAG_WORK_PHONE','FLAG_CONT_MOBILE','FLAG_DOCUMENT_4',

                                     'FLAG_DOCUMENT_5','FLAG_DOCUMENT_6','FLAG_DOCUMENT_7','FLAG_DOCUMENT_8',

                                     'REG_REGION_NOT_WORK_REGION','LIVE_REGION_NOT_WORK_REGION',

                                     'REG_CITY_NOT_LIVE_CITY','REG_CITY_NOT_WORK_CITY','LIVE_CITY_NOT_WORK_CITY',

                                     'WEEKDAY_APPR_PROCESS_START','CNT_FAM_MEMBERS','REG_REGION_NOT_LIVE_REGION','ORGANIZATION_TYPE',

                                     'NAME_TYPE_SUITE','REGION_RATING_CLIENT_W_CITY', 'HOUR_APPR_PROCESS_START','FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3',

                                     'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6',

                                     'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9',

                                     'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12',

                                     'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15',

                                     'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18',

                                     'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21']



# Non-categorical columns in application_data dataset



non_category_clmns_app_data = ['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE',

                                         'REGION_POPULATION_RELATIVE','DAYS_BIRTH','DAYS_EMPLOYED', 

                                         'DAYS_REGISTRATION','DAYS_ID_PUBLISH',

                                         'EXT_SOURCE_2','OBS_30_CNT_SOCIAL_CIRCLE', 

                                         'DEF_30_CNT_SOCIAL_CIRCLE','OBS_60_CNT_SOCIAL_CIRCLE', 

                                         'DEF_60_CNT_SOCIAL_CIRCLE','DAYS_LAST_PHONE_CHANGE',]



# categorical columns in previous_application dataset



category_clmns_prev_data = ['SK_ID_PREV', 'SK_ID_CURR','NAME_CONTRACT_TYPE', 'HOUR_APPR_PROCESS_START','WEEKDAY_APPR_PROCESS_START',

                                     'FLAG_LAST_APPL_PER_CONTRACT','NFLAG_LAST_APPL_IN_DAY','NAME_CASH_LOAN_PURPOSE',

                                      'NAME_CONTRACT_STATUS','NAME_PAYMENT_TYPE', 'CODE_REJECT_REASON', 

                                      'NAME_CLIENT_TYPE','NAME_GOODS_CATEGORY', 'NAME_PORTFOLIO', 'NAME_PRODUCT_TYPE',

                                      'CHANNEL_TYPE', 'NAME_SELLER_INDUSTRY','NAME_YIELD_GROUP', 'PRODUCT_COMBINATION']



# Non-categorical columns in previous_application dataset



non_category_clmns_prev_data = ['AMT_APPLICATION','AMT_CREDIT','DAYS_DECISION',

                                         'SELLERPLACE_AREA']
df_application_data = df_application_data[Colms_with_null_13perc_or_less_app]

df_previous_application = df_previous_application[Colms_with_null_13perc_or_less_prev]



# Checking the shape of the modified dataframes:



print(df_application_data.shape)

print(df_previous_application.shape)
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']



num_df1 = df_application_data.select_dtypes(include=numerics)

num_df2 = df_previous_application.select_dtypes(include=numerics)



# We are not considering the numerical columns that are categorical, in detecting outliers.



categorical_num_columns_df1 = ['SK_ID_CURR','TARGET','FLAG_MOBIL','FLAG_EMP_PHONE','FLAG_WORK_PHONE','FLAG_CONT_MOBILE',

                           'FLAG_PHONE','FLAG_EMAIL','FLAG_DOCUMENT_2','FLAG_DOCUMENT_3','FLAG_DOCUMENT_4','FLAG_DOCUMENT_5',

                           'FLAG_DOCUMENT_6','FLAG_DOCUMENT_7','FLAG_DOCUMENT_8','FLAG_DOCUMENT_9','FLAG_DOCUMENT_10',

                           'FLAG_DOCUMENT_11','FLAG_DOCUMENT_12','FLAG_DOCUMENT_13','FLAG_DOCUMENT_14','FLAG_DOCUMENT_15',

                           'FLAG_DOCUMENT_16','FLAG_DOCUMENT_17','FLAG_DOCUMENT_18','FLAG_DOCUMENT_19','FLAG_DOCUMENT_20',

                           'FLAG_DOCUMENT_21'] 



categorical_num_columns_df2 = ['SK_ID_PREV','SK_ID_CURR','NFLAG_LAST_APPL_IN_DAY']



# Removing the numerical_categorical columns 

num_df1 = num_df1.drop(columns=categorical_num_columns_df1)

num_df2 = num_df2.drop(columns=categorical_num_columns_df2)



clmn_app_non_cate_outlier = num_df1.columns[((num_df1.std()/num_df1.mean())*100).abs()>85].tolist()



print("Columns that has outliers in application_data.csv, \nwhich has a standard deviation of more than 85% from its mean:\n\n ", clmn_app_non_cate_outlier)



clmn_prev_non_cate_outlier = num_df2.columns[((num_df2.std()/num_df2.mean())*100).abs()>85].tolist()



print("\nColumns that has outliers in previous_application.csv, \nwhich has a standard deviation of more than 85% from its mean:\n\n", clmn_prev_non_cate_outlier)
# Finding the list of actual null columns from the dataframes (df_application_data and 

#df_previous_application),that contains all columns which has less than 13% Null values



clmn_app_null_cate = [i for i in df_application_data.columns[df_application_data.isna().any()].tolist() if i in category_clmns_app_data]

                      

print('List of all categorical columns that has actual missing values in among less than 13% null value columns in application_data: ', clmn_app_null_cate)

                      

clmn_app_null_non_cate = [i for i in df_application_data.columns[df_application_data.isna().any()].tolist() if i in non_category_clmns_app_data]                    

                      

print('\n\nList of all non-categorical columns that has actual missing values in among less than 13% null value columns in application_data: ', clmn_app_null_non_cate)



clmn_prev_null_cate = [i for i in df_previous_application.columns[df_previous_application.isna().any()].tolist() if i in category_clmns_prev_data]                      



print('\n\nList of all categorical columns that has actual missing values in previous_application_data: ', clmn_prev_null_cate)                       

                       

clmn_prev_null_non_cate = [i for i in df_previous_application.columns[df_previous_application.isna().any()].tolist() if i in non_category_clmns_prev_data]                      



print('\n\nListist of all non-categorical columns that has actual missing values in previous_application_data: ', clmn_prev_null_non_cate)
# Repalcing null categorical variable with mode in application_data dataframe



for column in clmn_app_null_cate:

    df_application_data[column].fillna(df_application_data[column].mode()[0], inplace=True)
# Repalcing null categorical variable with mode in previous_application dataframe



for column in clmn_prev_null_cate:

    df_previous_application[column].fillna(df_previous_application[column].mode()[0], inplace=True)
# Preparing a list containing null, non-categoricall variables containing outliers in application_data



clmn_app_null_non_cate_outlier = [i for i in clmn_app_null_non_cate if i in clmn_app_non_cate_outlier]



# Preparing a list containing null, non-categoricall variables containing outliers in previous_application



clmn_prev_null_non_cate_outlier = [i for i in clmn_prev_null_non_cate if i in clmn_prev_non_cate_outlier]

# Replacing null,non-categorical variable (having outliers) with median in application_data dataframe



for column in clmn_app_null_non_cate_outlier:

    df_application_data[column].fillna(df_application_data[column].median(), inplace=True)

    

# Replacing null,non-categorical variable (having outliers) with median in previous_application dataframe



for column in clmn_prev_null_non_cate_outlier:

    df_previous_application[column].fillna(df_previous_application[column].median(), inplace=True)
# Preparing a list containing null, non-categoricall variables containing no outliers in application_data



clmn_app_null_non_cate_no_outlier = [i for i in clmn_app_null_non_cate if i not in clmn_app_non_cate_outlier]



# Preparing a list containing null, non-categoricall variables containing no outliers in previous_application



clmn_prev_null_non_cate_no_outlier = [i for i in clmn_prev_null_non_cate if i not in clmn_prev_non_cate_outlier]

# Replacing null,non-categorical variable (having no outliers) with mean in application_data dataframe



for column in clmn_app_null_non_cate_no_outlier:

    df_application_data[column].fillna(df_application_data[column].mean(), inplace=True)

    

# Replacing null,non-categorical variable (having no outliers) with mean in previous_application dataframe



for column in clmn_prev_null_non_cate_no_outlier:

    df_previous_application[column].fillna(df_previous_application[column].mean(), inplace=True)
print('Is there any null values in df_application_data: ',df_application_data.isnull().values.any())

print('Is there any null values in df_previous_application: ',df_previous_application.isnull().values.any())
import matplotlib.pyplot as plt



count_1 = 0; count_0 = 0 # initialisation



for i in df_application_data['TARGET'].values:

    if i == 1:

        count_1 = count_1+1

    else:

        count_0 = count_0+1

        

count_1_perc = (count_1/(count_1 + count_0))*100



count_0_perc = (count_0/(count_1 + count_0))*100



X = ['Defaulter','Non-Defaulter']



Y = [count_1_perc, count_0_perc]



plt.bar(X,Y, width = 0.8)



plt.ylabel('(%) of the defaulter/Non-defaulter data')



plt.show()



print('Ratios of imbalance in percentage with respect to non-defaulter and defaulter datas are: %f and %f'%(count_0_perc,count_1_perc))

print('Ratios of imbalance in real-numbers with respect to non-defaulter and defaulter datas is %f : 1 (approx)'%(count_0/count_1))
df_application_data_T_1 = df_application_data[df_application_data.TARGET == 1]

df_application_data_T_0 = df_application_data[df_application_data.TARGET == 0]
Prev_target=[]

for i in df_previous_application['NAME_CONTRACT_STATUS'].tolist():

    if i == 'Approved':

        Prev_target.append(0)

    elif i == 'Refused':

        Prev_target.append(1)

    else:

        Prev_target.append(None)

        

# Creting a 'Target' variable with  Approved = 0, Refused = 1, all other as Null



df_previous_application['Target'] = Prev_target 



# Removing all rows that are having Null in Target varible.



df_previous_application = df_previous_application.loc[(df_previous_application['Target'] == 1) | (df_previous_application['Target'] == 0)] 



df_previous_application.head()
df_previous_application_T_1 = df_previous_application[df_previous_application.Target == 1]

df_previous_application_T_0 = df_previous_application[df_previous_application.Target == 0]
# Plotting the count and rank varaibles for WEEKDAY_APPR_PROCESS_START in both datasets for Target = 1



plt.figure(figsize=(18,6))



plt.subplot(121)



plt.title('Count of Applications filled by defaulters (T=1) on these days in application_data')



plt.ylabel('Count of defaulters');plt.xlabel('Day on which the loan applciation is filled [FIG: 1]')



df_application_data_T_1['WEEKDAY_APPR_PROCESS_START'].value_counts().plot(kind='bar')



plt.subplot(122)



plt.title('Count of Applications filled by defaulters (T=1) on these days in previous_application')



plt.ylabel('Count of defaulters');plt.xlabel('Day on which the loan applciation is filled [FIG: 2]')



df_previous_application_T_1['WEEKDAY_APPR_PROCESS_START'].value_counts().plot(kind='bar')



plt.show()
# Plotting the count and rank varaibles for WEEKDAY_APPR_PROCESS_START in both datasets for Target = 0



plt.figure(figsize=(18,6))



plt.subplot(121)



plt.title('Count of Applications filled by non-defaulters (T=0) on these days in application_data')



plt.ylabel('Count of non-defaulters');plt.xlabel('Day on which the loan applciation is filled [FIG: 3]')



df_application_data_T_0['WEEKDAY_APPR_PROCESS_START'].value_counts().plot(kind='bar')



plt.subplot(122)



plt.title('Count of Applications filled by non-defaulters (T=0) on these days in application_data')



plt.ylabel('Count of non-defaulters');plt.xlabel('Day on which the loan applciation is filled [FIG: 4]')



df_previous_application_T_0['WEEKDAY_APPR_PROCESS_START'].value_counts().plot(kind='bar')



plt.show()
# Plotting the count and rank varaibles for WEEKDAY_APPR_PROCESS_START in both datasets for Target = 1



plt.figure(figsize=(18,6))



plt.subplot(121)



plt.title('Count of Applications filled by defaulters (T=1) on these hours in application_data')



plt.ylabel('Count of defaulters');plt.xlabel('Hour on which the loan applciation is filled [FIG: 5]')



df_application_data_T_1['HOUR_APPR_PROCESS_START'].value_counts().plot(kind='bar')



plt.subplot(122)



plt.title('Count of Applications filled by defaulters (T=1) on these hours in previous_application')



plt.ylabel('Count of defaulters');plt.xlabel('Hour on which the loan applciation is filled [FIG: 6]')



df_previous_application_T_1['HOUR_APPR_PROCESS_START'].value_counts().plot(kind='bar')



plt.show()
# Plotting the count and rank varaibles for WEEKDAY_APPR_PROCESS_START in both datasets for Target = 0



plt.figure(figsize=(18,6))



plt.subplot(121)



plt.title('Count of Applications filled by non-defaulters (T=0) on these hours in application_data')



plt.ylabel('Count of non-defaulters');plt.xlabel('Hour on which the loan applciation is filled [FIG: 7]')



df_application_data_T_0['HOUR_APPR_PROCESS_START'].value_counts().plot(kind='bar')



plt.subplot(122)



plt.title('Count of Applications filled by non-defaulters (T=0) on these hours in previous_application')



plt.ylabel('Count of non-defaulters');plt.xlabel('Hour on which the loan applciation is filled [FIG: 8]')



df_previous_application_T_0['HOUR_APPR_PROCESS_START'].value_counts().plot(kind='bar')



plt.show()
# Plotting the count and rank varaibles for CNT_CHILDREN in application_data datasets for Target = 1&0



plt.figure(figsize=(18,6))



plt.subplot(121)



plt.title('Count of defaulters (T=1) based on CNT_CHILDREN in application_data')



plt.ylabel('Count of defaulters');plt.xlabel('CNT_CHILDREN [FIG: 9]')



df_application_data_T_1['CNT_CHILDREN'].value_counts().plot(kind='bar')



plt.subplot(122)



plt.title('Count of non-defaulters (T=0) based on CNT_CHILDREN in application_data')



plt.ylabel('Count of non-defaulters');plt.xlabel('CNT_CHILDREN [FIG: 10]')



df_application_data_T_0['CNT_CHILDREN'].value_counts().plot(kind='bar')



plt.show()

# Plotting the count and rank varaibles ('NAME_EDUCATION_TYPE') for in application_data datasets for Target = 1&0



plt.figure(figsize=(18,6))



plt.subplot(121)



plt.title('Count of defaulters (T=1) based on NAME_EDUCATION_TYPE in application_data')



plt.ylabel('Count of defaulters');plt.xlabel('NAME_EDUCATION_TYPE [FIG: 11]')



df_application_data_T_1['NAME_EDUCATION_TYPE'].value_counts().plot(kind='bar')



plt.subplot(122)



plt.title('Count of non-defaulters (T=0) based on CNT_CHILDREN in NAME_EDUCATION_TYPE')



plt.ylabel('Count of non-defaulters');plt.xlabel('NAME_EDUCATION_TYPE [FIG: 12]')



df_application_data_T_0['NAME_EDUCATION_TYPE'].value_counts().plot(kind='bar')



plt.show()
# Plotting the count and rank varaibles ('NAME_CLIENT_TYPE') for in application_data datasets for Target = 1&0



plt.figure(figsize=(18,6))



plt.subplot(121)



plt.title('Count of defaulters (T=1) based on NAME_CLIENT_TYPE in previous_application_data')



plt.ylabel('Count of defaulters');plt.xlabel('NAME_CLIENT_TYPE [FIG: 13]')



df_previous_application_T_1['NAME_CLIENT_TYPE'].value_counts().plot(kind='bar')



plt.subplot(122)



plt.title('Count of non-defaulters (T=0) based on NAME_CLIENT_TYPE in previous_application')



plt.ylabel('Count of non-defaulters');plt.xlabel('NAME_CLIENT_TYPE [FIG: 14]')



df_previous_application_T_0['NAME_CLIENT_TYPE'].value_counts().plot(kind='bar')



plt.show()
# Plotting the count and rank varaibles for NAME_CONTRACT_TYPE in both datasets for Target = 1



plt.figure(figsize=(18,6))



plt.subplot(121)



plt.title('Count of defaulters (T=1) based on the type of loans availed in application_data')



plt.ylabel('Count of defaulters');plt.xlabel('Types of Loan [FIG: 9]')



df_application_data_T_1['NAME_CONTRACT_TYPE'].value_counts().plot(kind='bar', width=0.4)



plt.subplot(122)



plt.title('Count of defaulters (T=1) based on the type of loans availed in previous_application')



plt.ylabel('Count of defaulters');plt.xlabel('Types of Loan [FIG: 10]')



df_previous_application_T_1['NAME_CONTRACT_TYPE'].value_counts().plot(kind='bar',width=0.8)



plt.show()
# Plotting the count and rank varaibles for WEEKDAY_APPR_PROCESS_START in both datasets for Target = 0



plt.figure(figsize=(18,6))



plt.subplot(121)



plt.title('Count of non-defaulters (T=0) based on the type of loans availed in application_data')



plt.ylabel('Count of non-defaulters');plt.xlabel('Types of Loan [FIG: 11]')



df_application_data_T_0['NAME_CONTRACT_TYPE'].value_counts().plot(kind='bar')



plt.subplot(122)



plt.title('Count of non-defaulters (T=0) based on the type of loans availed in previous_application')



plt.ylabel('Count of non-defaulters');plt.xlabel('Types of Loan [FIG: 13]')



df_previous_application_T_0['NAME_CONTRACT_TYPE'].value_counts().plot(kind='bar')



plt.show()
# Plotting the count and rank varaibles for NAME_CONTRACT_TYPE in application_data datasets for Target = 1&0



plt.figure(figsize=(18,6))



plt.subplot(121)



plt.title('Count of defaulters (T=1) based on NAME_TYPE_SUITE in application_data')



plt.ylabel('Count of defaulters');plt.xlabel('NAME_TYPE_SUITE [FIG: 13]')



df_application_data_T_1['NAME_TYPE_SUITE'].value_counts().plot(kind='bar')



plt.subplot(122)



plt.title('Count of non-defaulters (T=0) based on NAME_TYPE_SUITE in application_data')



plt.ylabel('Count of non-defaulters');plt.xlabel('NAME_TYPE_SUITE [FIG: 14]')



df_application_data_T_0['NAME_TYPE_SUITE'].value_counts().plot(kind='bar')



plt.show()

# Plotting the count and rank varaibles for FLAG_OWN_CAR in application_data datasets for Target = 1&0



plt.figure(figsize=(18,6))



plt.subplot(121)



plt.title('Count of defaulters (T=1) based on FLAG_OWN_CAR in application_data')



plt.ylabel('Count of defaulters');plt.xlabel('FLAG_OWN_CAR (Y=owning a car & N = Not owning a car) [FIG: 15]')



df_application_data_T_1['FLAG_OWN_CAR'].value_counts().plot(kind='bar')



plt.subplot(122)



plt.title('Count of non-defaulters (T=0) based on FLAG_OWN_CAR in application_data')



plt.ylabel('Count of non-defaulters');plt.xlabel('FLAG_OWN_CAR (Y=owning a car & N = Not owning a car) [FIG: 16]')



df_application_data_T_0['FLAG_OWN_CAR'].value_counts().plot(kind='bar')



plt.show()
# Plotting the count and rank varaibles for CODE_GENDER in application_data datasets for Target = 1&0



plt.figure(figsize=(18,6))



plt.subplot(121)



plt.title('Count of defaulters (T=1) based on CODE_GENDER in application_data')



plt.ylabel('Count of defaulters');plt.xlabel('CODE_GENDER (F= Female & M = Male) [FIG: 17]')



df_application_data_T_1['CODE_GENDER'].value_counts().plot(kind='bar')



plt.subplot(122)



plt.title('Count of non-defaulters (T=0) based on CODE_GENDER in application_data')



plt.ylabel('Count of non-defaulters');plt.xlabel('CODE_GENDER (F= Female & M = Male) [FIG: 18]')



df_application_data_T_0['CODE_GENDER'].value_counts().plot(kind='bar')



plt.show()
# Plotting the count and rank varaibles for FLAG_OWN_REALTY in application_data datasets for Target = 1&0



plt.figure(figsize=(18,6))



plt.subplot(121)



plt.title('Count of defaulters (T=1) based on FLAG_OWN_REALTY in application_data')



plt.ylabel('Count of defaulters');plt.xlabel('FLAG_OWN_REALTY (Y= owning house/flat & N = not owning house/flat) [FIG: 19]')



df_application_data_T_1['FLAG_OWN_REALTY'].value_counts().plot(kind='bar')



plt.subplot(122)



plt.title('Count of non-defaulters (T=0) based on FLAG_OWN_REALTY in application_data')



plt.ylabel('Count of non-defaulters');plt.xlabel('FLAG_OWN_REALTY (Y= owning house/flat & N = not owning house/flat) [FIG: 20]')



df_application_data_T_0['FLAG_OWN_REALTY'].value_counts().plot(kind='bar')



plt.show()
# Plotting the count and rank varaibles for NAME_INCOME_TYPE in application_data datasets for Target = 1&0



plt.figure(figsize=(18,6))



plt.subplot(121)



plt.title('Count of defaulters (T=1) based on NAME_INCOME_TYPE in application_data')



plt.ylabel('Count of defaulters');plt.xlabel('NAME_INCOME_TYPE [FIG: 21]')



df_application_data_T_1['NAME_INCOME_TYPE'].value_counts().plot(kind='bar')



plt.subplot(122)



plt.title('Count of non-defaulters (T=0) based on NAME_INCOME_TYPE in application_data')



plt.ylabel('Count of non-defaulters');plt.xlabel('NAME_INCOME_TYPE [FIG: 22]')



df_application_data_T_0['NAME_INCOME_TYPE'].value_counts().plot(kind='bar')



plt.show()
# Plotting the count and rank varaibles for NAME_FAMILY_STATUS in application_data datasets for Target = 1&0



plt.figure(figsize=(18,6))



plt.subplot(121)



plt.title('Count of defaulters (T=1) based on NAME_FAMILY_STATUS in application_data')



plt.ylabel('Count of defaulters');plt.xlabel('NAME_FAMILY_STATUS [FIG: 23]')



df_application_data_T_1['NAME_FAMILY_STATUS'].value_counts().plot(kind='bar')



plt.subplot(122)



plt.title('Count of non-defaulters (T=0) based on NAME_FAMILY_STATUS in application_data')



plt.ylabel('Count of non-defaulters');plt.xlabel('NAME_FAMILY_STATUS [FIG: 24]')



df_application_data_T_0['NAME_FAMILY_STATUS'].value_counts().plot(kind='bar')



plt.show()
# Plotting the count and rank varaibles for NAME_HOUSING_TYPE in application_data datasets for Target = 1&0



plt.figure(figsize=(18,6))



plt.subplot(121)



plt.title('Count of defaulters (T=1) based on NAME_HOUSING_TYPE in application_data')



plt.ylabel('Count of defaulters');plt.xlabel('NAME_HOUSING_TYPE [FIG: 25]')



df_application_data_T_1['NAME_HOUSING_TYPE'].value_counts().plot(kind='bar')



plt.subplot(122)



plt.title('Count of non-defaulters (T=0) based on NAME_HOUSING_TYPE in application_data')



plt.ylabel('Count of non-defaulters');plt.xlabel('NAME_HOUSING_TYPE [FIG: 26]')



df_application_data_T_0['NAME_HOUSING_TYPE'].value_counts().plot(kind='bar')



plt.show()
# Plotting the count and rank varaibles for PRODUCT_COMBINATION in previous_application datasets for Target = 1&0



plt.figure(figsize=(18,6))



plt.subplot(121)



plt.title('Count of defaulters (T=1) based on PRODUCT_COMBINATION in previous_application')



plt.ylabel('Count of defaulters');plt.xlabel('PRODUCT_COMBINATION [FIG: 27]')



df_previous_application_T_1['PRODUCT_COMBINATION'].value_counts().plot(kind='bar')



plt.subplot(122)



plt.title('Count of non-defaulters (T=0) based on PRODUCT_COMBINATION in previous_application')



plt.ylabel('Count of non-defaulters');plt.xlabel('PRODUCT_COMBINATION [FIG: 28]')



df_previous_application_T_0['PRODUCT_COMBINATION'].value_counts().plot(kind='bar')



plt.show()
# Plotting the count and rank varaibles for NAME_YIELD_GROUP in previous_application datasets for Target = 1&0



plt.figure(figsize=(18,6))



plt.subplot(121)



plt.title('Count of defaulters (T=1) based on NAME_YIELD_GROUP in previous_application')



plt.ylabel('Count of defaulters');plt.xlabel('NAME_YIELD_GROUP [FIG: 29]')



df_previous_application_T_1['NAME_YIELD_GROUP'].value_counts().plot(kind='bar')



plt.subplot(122)



plt.title('Count of non-defaulters (T=0) based on NAME_YIELD_GROUP in previous_application')



plt.ylabel('Count of non-defaulters');plt.xlabel('NAME_YIELD_GROUP [FIG: 30]')



df_previous_application_T_0['NAME_YIELD_GROUP'].value_counts().plot(kind='bar')



plt.show()
# Plotting the count and rank varaibles for CHANNEL_TYPE in previous_application datasets for Target = 1&0



plt.figure(figsize=(18,6))



plt.subplot(121)



plt.title('Count of defaulters (T=1) based on CHANNEL_TYPE in previous_application')



plt.ylabel('Count of defaulters');plt.xlabel('CHANNEL_TYPE [FIG: 31]')



df_previous_application_T_1['CHANNEL_TYPE'].value_counts().plot(kind='bar')



plt.subplot(122)



plt.title('Count of non-defaulters (T=0) based on CHANNEL_TYPE in previous_application')



plt.ylabel('Count of non-defaulters');plt.xlabel('CHANNEL_TYPE [FIG: 32]')



df_previous_application_T_0['CHANNEL_TYPE'].value_counts().plot(kind='bar')



plt.show()
# Plotting the count and rank varaibles for NAME_PORTFOLIO in previous_application datasets for Target = 1&0



plt.figure(figsize=(18,6))



plt.subplot(121)



plt.title('Count of defaulters (T=1) based on NAME_PORTFOLIO in previous_application')



plt.ylabel('Count of defaulters');plt.xlabel('NAME_PORTFOLIO [FIG: 33]')



df_previous_application_T_1['NAME_PORTFOLIO'].value_counts().plot(kind='bar')



plt.subplot(122)



plt.title('Count of non-defaulters (T=0) based on NAME_PORTFOLIO in previous_application')



plt.ylabel('Count of non-defaulters');plt.xlabel('NAME_PORTFOLIO [FIG: 34]')



df_previous_application_T_0['NAME_PORTFOLIO'].value_counts().plot(kind='bar')



plt.show()
# Plotting the Mean for varible AMT_CREDIT in both datasets for Target = 1&0



num_colm = 'AMT_CREDIT'



mean_prev_T_0 = df_previous_application_T_0[num_colm].mean()



mean_prev_T_1 = df_previous_application_T_1[num_colm].mean()



mean_appl_T_0 = df_application_data_T_0[num_colm].mean()



mean_appl_T_1 = df_application_data_T_1[num_colm].mean()



x = ['AMT_CREDIT_mean_T_0_in_prev_data','AMT_CREDIT_mean_T_1_in_prev_data','AMT_CREDIT_mean_T_0_in_appl_data','AMT_CREDIT_mean_T_1_in_appl_data']



y = [mean_prev_T_0,mean_prev_T_1,mean_appl_T_0,mean_appl_T_1]



plt.figure(figsize=(16,6))



plt.ylabel('AMT_CREDIT_Mean')



plt.title('Mean of "AMT_CREDIT" in both datasets application_data (appl_data) & previous_application (prev_data) for target = 1&0 [FIG:35]')



plt.bar(x,y,width=0.4)



plt.show()
# Plotting the Mean for varible AMT_INCOME_TOTAL in both datasets for Target = 1&0 on the basis of Gender in application_data dataset



num_colm = 'AMT_CREDIT'



mean_appl_T_0_M = df_application_data_T_0[df_application_data_T_0.CODE_GENDER=='M'][num_colm].mean()



mean_appl_T_0_F = df_application_data_T_0[df_application_data_T_0.CODE_GENDER=='F'][num_colm].mean()



mean_appl_T_1_M = df_application_data_T_1[df_application_data_T_1.CODE_GENDER=='M'][num_colm].mean()



mean_appl_T_1_F = df_application_data_T_1[df_application_data_T_1.CODE_GENDER=='F'][num_colm].mean()



x_male = ['AMT_INCOME_TOTAL_mean_T_0_Male','AMT_INCOME_TOTAL_mean_T_1_Male']



y_male = [mean_appl_T_0_M,mean_appl_T_1_M]



x_Female = ['AMT_INCOME_TOTAL_mean_T_0_Female','AMT_INCOME_TOTAL_mean_T_1_Female']



y_Female = [mean_appl_T_0_F,mean_appl_T_1_F]



plt.figure(figsize=(18,6))



plt.subplot(121)



plt.ylabel('AMT_INCOME_TOTAL_Mean')



plt.xlabel('[FIG:36]')



plt.title('Mean of "AMT_CREDIT" in datasets appl_data & prev_data for target = 1&0 For Male')



plt.bar(x_male,y_male)



plt.subplot(122)



plt.ylabel('AMT_INCOME_TOTAL_Mean')



plt.xlabel('[FIG:37]')



plt.title('Mean of "AMT_CREDIT" in datasets appl_data & prev_data for target = 1&0 For Female')



plt.bar(x_Female,y_Female)



plt.show()
# Plotting the Mean for varible AMT_ANNUITY in both datasets for Target = 1&0 on the basis of Gender in application_data dataset



num_colm = 'AMT_ANNUITY'



mean_appl_T_0_M = df_application_data_T_0[df_application_data_T_0.CODE_GENDER=='M'][num_colm].mean()



mean_appl_T_0_F = df_application_data_T_0[df_application_data_T_0.CODE_GENDER=='F'][num_colm].mean()



mean_appl_T_1_M = df_application_data_T_1[df_application_data_T_1.CODE_GENDER=='M'][num_colm].mean()



mean_appl_T_1_F = df_application_data_T_1[df_application_data_T_1.CODE_GENDER=='F'][num_colm].mean()



x_male = ['AMT_ANNUITY_mean_T_0_Male','AMT_ANNUITY_mean_T_1_Male']



y_male = [mean_appl_T_0_M,mean_appl_T_1_M]



x_Female = ['AMT_ANNUITY_mean_T_0_Female','AMT_ANNUITY_mean_T_1_Female']



y_Female = [mean_appl_T_0_F,mean_appl_T_1_F]



plt.figure(figsize=(18,6))



plt.subplot(121)



plt.ylabel('AMT_ANNUITY_Mean')



plt.xlabel('[FIG:38]')



plt.title('Mean of "AMT_ANNUITY" in application_data dataset for target = 1&0 For Male')



plt.bar(x_male,y_male)



plt.subplot(122)



plt.ylabel('AMT_ANNUITY_Mean')



plt.xlabel('[FIG:39]')



plt.title('Mean of "AMT_ANNUITY" in application_data dataset for target = 1&0 For FeMale')



plt.bar(x_Female,y_Female)



plt.show()
# Plotting the Mean for varible AMT_GOODS_PRICE in both datasets for Target = 1&0 on the basis of Gender in application_data dataset



num_colm = 'AMT_GOODS_PRICE'



mean_appl_T_0_M = df_application_data_T_0[df_application_data_T_0.CODE_GENDER=='M'][num_colm].mean()



mean_appl_T_0_F = df_application_data_T_0[df_application_data_T_0.CODE_GENDER=='F'][num_colm].mean()



mean_appl_T_1_M = df_application_data_T_1[df_application_data_T_1.CODE_GENDER=='M'][num_colm].mean()



mean_appl_T_1_F = df_application_data_T_1[df_application_data_T_1.CODE_GENDER=='F'][num_colm].mean()



x_male = ['AMT_GOODS_PRICE_mean_T_0_Male','AMT_GOODS_PRICE_mean_T_1_Male']



y_male = [mean_appl_T_0_M,mean_appl_T_1_M]



x_Female = ['AMT_GOODS_PRICE_mean_T_0_Female','AMT_GOODS_PRICE_mean_T_1_Female']



y_Female = [mean_appl_T_0_F,mean_appl_T_1_F]



plt.figure(figsize=(18,6))



plt.subplot(121)



plt.ylabel('AMT_GOODS_PRICE_Mean')



plt.xlabel('[FIG:40]')



plt.title('Mean of "AMT_GOODS_PRICE" in dataset: appl_data for target = 1&0 For Male')



plt.bar(x_male,y_male)



plt.subplot(122)



plt.ylabel('AMT_GOODS_PRICE_Mean')



plt.xlabel('[FIG:41]')



plt.title('Mean of "AMT_GOODS_PRICE" in dataset: appl_data for target = 1&0 For FeMale')



plt.bar(x_Female,y_Female)



plt.show()
# Plotting the Mean for varible AMT_APPLICATION in previous_application with target = 1&0



num_colm = 'AMT_APPLICATION'



mean_prev_T_0 = df_previous_application_T_0[num_colm].mean()



mean_prev_T_1 = df_previous_application_T_1[num_colm].mean()



x = ['AMT_APPLICATION_mean_T_0','AMT_APPLICATION_mean_T_1']



y = [mean_prev_T_0,mean_prev_T_1]



plt.figure(figsize=(14,6))



plt.ylabel('AMT_APPLICATION_Mean')



plt.title('Mean of "AMT_APPLICATION" in  previous_application for target = 1&0 [FIG:42]')



plt.bar(x,y)



plt.show()
bivaritate_variables = ['AMT_CREDIT','AMT_APPLICATION']



x_t_0 = df_previous_application_T_0[bivaritate_variables[0]].values



x_t_1 = df_previous_application_T_1[bivaritate_variables[0]].values



y_t_0 = df_previous_application_T_0[bivaritate_variables[1]].values



y_t_1 = df_previous_application_T_1[bivaritate_variables[1]].values



############# Function to viualise the Linear regression line  ################



from numpy import *

import matplotlib.pyplot as plt



def plot_best_fit(intercept, slope):

    axes = plt.gca()

    x_vals = array(axes.get_xlim())

    y_vals = intercept + slope * x_vals

    plt.plot(x_vals, y_vals, 'r-')



############# Utilising Linear Regression Algorithm from Sklearn #############



from sklearn.linear_model import LinearRegression

    

def Regression(X,Y):   

    regr = LinearRegression()

    regr.fit(X,Y)

    return regr



######################### Main Code To Plot the Graph ##########################



plt.figure(figsize=(14,16))



plt.subplot(211)



plt.scatter(x_t_0,y_t_0)



regr = Regression(x_t_0.reshape(-1,1) ,y_t_0.reshape(-1,1))



plot_best_fit((regr.intercept_)[0],(regr.coef_)[0][0])



plt.title('Bi-Variate Analysis on AMT_CREDIT & AMT_APPLICATION For T=0 in previous_data(Fig 43)')



plt.xlabel('AMT_CREDIT')



plt.ylabel('AMT_APPLICATION')



plt.subplot(212)



plt.scatter(x_t_1,y_t_1)



regr = Regression(x_t_1.reshape(-1,1) ,y_t_1.reshape(-1,1))



plot_best_fit((regr.intercept_)[0],(regr.coef_)[0][0])



plt.title('Bi-Variate Analysis on AMT_CREDIT & AMT_APPLICATION For T=1 in previous_data(Fig 44)')



plt.xlabel('AMT_CREDIT')



plt.ylabel('AMT_APPLICATION')



plt.show()



print('Actual Co-relation value between ',bivaritate_variables,'for non-default applicants (t-0) \n in previous_application dataset is as shown in below co-relation matrix: \n\n',df_previous_application_T_0[bivaritate_variables].corr())



print('\n\nActual Co-relation value between ',bivaritate_variables,'for default applicants (t-1) \n in previous_application dataset is as shown in below co-relation matrix: \n\n',df_previous_application_T_1[bivaritate_variables].corr())
bivaritate_variables = ['AMT_CREDIT','AMT_INCOME_TOTAL']



x_t_0 = df_application_data_T_0[bivaritate_variables[0]].values



x_t_1 = df_application_data_T_1[bivaritate_variables[0]].values



y_t_0 = df_application_data_T_0[bivaritate_variables[1]].values



y_t_1 = df_application_data_T_1[bivaritate_variables[1]].values



############# Function to viualise the Linear regression line  ################



from numpy import *

import matplotlib.pyplot as plt



def plot_best_fit(intercept, slope):

    axes = plt.gca()

    x_vals = array(axes.get_xlim())

    y_vals = intercept + slope * x_vals

    plt.plot(x_vals, y_vals, 'r-')



############# Utilising Linear Regression Algorithm from Sklearn #############



from sklearn.linear_model import LinearRegression

    

def Regression(X,Y):   

    regr = LinearRegression()

    regr.fit(X,Y)

    return regr



######################### Main Code To Plot the Graph ##########################



plt.figure(figsize=(14,14))



plt.subplot(211)



plt.scatter(x_t_0,y_t_0)



regr = Regression(x_t_0.reshape(-1,1) ,y_t_0.reshape(-1,1))



plot_best_fit((regr.intercept_)[0],(regr.coef_)[0][0])



plt.title('Bi-Variate Analysis on AMT_CREDIT & AMT_INCOME_TOTAL For T=0 in previous_data(Fig 43)')



plt.xlabel('AMT_CREDIT')



plt.ylabel('AMT_INCOME_TOTAL')



plt.subplot(212)



plt.scatter(x_t_1,y_t_1)



regr = Regression(x_t_1.reshape(-1,1) ,y_t_1.reshape(-1,1))



plot_best_fit((regr.intercept_)[0],(regr.coef_)[0][0])



plt.title('Bi-Variate Analysis on AMT_CREDIT & AMT_INCOME_TOTAL For T=1 in previous_data(Fig 44)')



plt.xlabel('AMT_CREDIT')



plt.ylabel('AMT_INCOME_TOTAL')



plt.show()



print('Actual Co-relation value between ',bivaritate_variables,'for non-default applicants (t-0) \n in application_data dataset is as shown in below co-relation matrix: \n\n',df_application_data_T_0[bivaritate_variables].corr())



print('\n\nActual Co-relation value between ',bivaritate_variables,'for default applicants (t-1) \n in application_data dataset is as shown in below co-relation matrix: \n\n',df_application_data_T_1[bivaritate_variables].corr())
bivaritate_variables = ['AMT_CREDIT','AMT_ANNUITY']



x_t_0 = df_application_data_T_0[bivaritate_variables[0]].values



x_t_1 = df_application_data_T_1[bivaritate_variables[0]].values



y_t_0 = df_application_data_T_0[bivaritate_variables[1]].values



y_t_1 = df_application_data_T_1[bivaritate_variables[1]].values



############# Function to viualise the Linear regression line  ################



from numpy import *

import matplotlib.pyplot as plt



def plot_best_fit(intercept, slope):

    axes = plt.gca()

    x_vals = array(axes.get_xlim())

    y_vals = intercept + slope * x_vals

    plt.plot(x_vals, y_vals, 'r-')



############# Utilising Linear Regression Algorithm from Sklearn #############



from sklearn.linear_model import LinearRegression

    

def Regression(X,Y):   

    regr = LinearRegression()

    regr.fit(X,Y)

    return regr



######################### Main Code To Plot the Graph ##########################



plt.figure(figsize=(14,14))



plt.subplot(211)



plt.scatter(x_t_0,y_t_0)



regr = Regression(x_t_0.reshape(-1,1) ,y_t_0.reshape(-1,1))



plot_best_fit((regr.intercept_)[0],(regr.coef_)[0][0])



plt.title('Bi-Variate Analysis on AMT_CREDIT & AMT_ANNUITY For T=0 in previous_data(Fig 45)')



plt.xlabel('AMT_CREDIT')



plt.ylabel('AMT_ANNUITY')



plt.subplot(212)



plt.scatter(x_t_1,y_t_1)



regr = Regression(x_t_1.reshape(-1,1) ,y_t_1.reshape(-1,1))



plot_best_fit((regr.intercept_)[0],(regr.coef_)[0][0])



plt.title('Bi-Variate Analysis on AMT_CREDIT & AMT_ANNUITY For T=1 in previous_data(Fig 46)')



plt.xlabel('AMT_CREDIT')



plt.ylabel('AMT_ANNUITY')



plt.show()



print('Actual Co-relation value between ',bivaritate_variables,'for non-default applicants (t-0) \n in application_data dataset is as shown in below co-relation matrix: \n\n',df_application_data_T_0[bivaritate_variables].corr())



print('\n\nActual Co-relation value between ',bivaritate_variables,'for default applicants (t-1) \n in application_data dataset is as shown in below co-relation matrix: \n\n',df_application_data_T_1[bivaritate_variables].corr())
bivaritate_variables = ['AMT_CREDIT','AMT_GOODS_PRICE']



x_t_0 = df_application_data_T_0[bivaritate_variables[0]].values



x_t_1 = df_application_data_T_1[bivaritate_variables[0]].values



y_t_0 = df_application_data_T_0[bivaritate_variables[1]].values



y_t_1 = df_application_data_T_1[bivaritate_variables[1]].values



############# Function to viualise the Linear regression line  ################



from numpy import *

import matplotlib.pyplot as plt



def plot_best_fit(intercept, slope):

    axes = plt.gca()

    x_vals = array(axes.get_xlim())

    y_vals = intercept + slope * x_vals

    plt.plot(x_vals, y_vals, 'r-')



############# Utilising Linear Regression Algorithm from Sklearn #############



from sklearn.linear_model import LinearRegression

    

def Regression(X,Y):   

    regr = LinearRegression()

    regr.fit(X,Y)

    return regr



######################### Main Code To Plot the Graph ##########################



plt.figure(figsize=(14,14))



plt.subplot(211)



plt.scatter(x_t_0,y_t_0)



regr = Regression(x_t_0.reshape(-1,1) ,y_t_0.reshape(-1,1))



plot_best_fit((regr.intercept_)[0],(regr.coef_)[0][0])



plt.title('Bi-Variate Analysis on AMT_CREDIT & AMT_GOODS_PRICE For T=0 in previous_data(Fig 47)')



plt.xlabel('AMT_CREDIT')



plt.ylabel('AMT_GOODS_PRICE')



plt.subplot(212)



plt.scatter(x_t_1,y_t_1)



regr = Regression(x_t_1.reshape(-1,1) ,y_t_1.reshape(-1,1))



plot_best_fit((regr.intercept_)[0],(regr.coef_)[0][0])



plt.title('Bi-Variate Analysis on AMT_CREDIT & AMT_ANNUITY For T=1 in previous_data(Fig 48)')



plt.xlabel('AMT_CREDIT')



plt.ylabel('AMT_GOODS_PRICE')



plt.show()



print('Actual Co-relation value between ',bivaritate_variables,'for non-default applicants (t-0) \n in application_data dataset is as shown in below co-relation matrix: \n\n',df_application_data_T_0[bivaritate_variables].corr())



print('\n\nActual Co-relation value between ',bivaritate_variables,'for default applicants (t-1) \n in application_data dataset is as shown in below co-relation matrix: \n\n',df_application_data_T_1[bivaritate_variables].corr())