import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.simplefilter('ignore')
df_application_train = pd.read_csv('../input/application_train.csv')
df_application_train.head()
df_application_test = pd.read_csv('../input/application_test.csv')
df_application_test.head()
print(df_application_train.shape, df_application_test.shape)
df_application_train['TARGET'].value_counts()
features_for_loan = [
    'SK_ID_CURR',                 #ID of loan in our sample
    'AMT_ANNUITY',                #Loan annuity
    'AMT_CREDIT',                 #Credit amount of the loan
    'AMT_GOODS_PRICE',            #For consumer loans it is the price of the goods for which the loan is given
    'NAME_CONTRACT_TYPE',         #Identification if loan is cash or revolving
    'HOUR_APPR_PROCESS_START',    #Approximately at what hour did the client apply for the loan
    'WEEKDAY_APPR_PROCESS_START', #On which day of the week did the client apply for the loan
    'NAME_TYPE_SUITE',            #Who was accompanying client when he was applying for the loan
    
    'FLAG_DOCUMENT_2',     #Did client provide document 2
    'FLAG_DOCUMENT_3',     #Did client provide document 3
    'FLAG_DOCUMENT_4',     #Did client provide document 4
    'FLAG_DOCUMENT_5',     #Did client provide document 5
    'FLAG_DOCUMENT_6',     #Did client provide document 6
    'FLAG_DOCUMENT_7',     #Did client provide document 7
    'FLAG_DOCUMENT_8',     #Did client provide document 8
    'FLAG_DOCUMENT_9',     #Did client provide document 9
    'FLAG_DOCUMENT_10',    #Did client provide document 10
    'FLAG_DOCUMENT_11',    #Did client provide document 11
    'FLAG_DOCUMENT_12',    #Did client provide document 12
    'FLAG_DOCUMENT_13',    #Did client provide document 13
    'FLAG_DOCUMENT_14',    #Did client provide document 14
    'FLAG_DOCUMENT_15',    #Did client provide document 15
    'FLAG_DOCUMENT_16',    #Did client provide document 16
    'FLAG_DOCUMENT_17',    #Did client provide document 17
    'FLAG_DOCUMENT_18',    #Did client provide document 18
    'FLAG_DOCUMENT_19',    #Did client provide document 19
    'FLAG_DOCUMENT_20',    #Did client provide document 20
    'FLAG_DOCUMENT_21',    #Did client provide document 21
    'FLAG_EMAIL',          #Did client provide email (1=YES, 0=NO)
    'FLAG_EMP_PHONE',      #Did client provide work phone (1=YES, 0=NO)
    'FLAG_MOBIL',          #Did client provide mobile phone (1=YES, 0=NO)
    'FLAG_PHONE',          #Did client provide home phone (1=YES, 0=NO)
    'FLAG_WORK_PHONE'      #Did client provide home phone (1=YES, 0=NO)
]
len(features_for_loan)
features_for_history = [
                            #time only relative to the application
    'DAYS_BIRTH',                 #Client's age in days at the time of application
    'DAYS_EMPLOYED',              #How many days before the application the person started current employment
    'DAYS_ID_PUBLISH',            #How many days before the application did client change the identity document with which he applied for the loan
    'DAYS_REGISTRATION',          #How many days before the application did client change his registration

    'DAYS_LAST_PHONE_CHANGE',     #How many days before the application did client change phone
    
    'AMT_REQ_CREDIT_BUREAU_DAY',  #Number of enquiries to Credit Bureau about the client one day before application (excluding one hour before application)
    'AMT_REQ_CREDIT_BUREAU_HOUR', #Number of enquiries to Credit Bureau about the client one hour before application
    'AMT_REQ_CREDIT_BUREAU_MON',  #Number of enquiries to Credit Bureau about the client one month before application (excluding one week before application)
    'AMT_REQ_CREDIT_BUREAU_QRT',  #Number of enquiries to Credit Bureau about the client 3 month before application (excluding one month before application)
    'AMT_REQ_CREDIT_BUREAU_WEEK', #Number of enquiries to Credit Bureau about the client one week before application (excluding one day before application)
    'AMT_REQ_CREDIT_BUREAU_YEAR', #Number of enquiries to Credit Bureau about the client one day year (excluding last 3 months before application)
    
    'OBS_30_CNT_SOCIAL_CIRCLE',   #How many observation of client's social surroundings with observable 30 DPD (days past due) default
    'OBS_60_CNT_SOCIAL_CIRCLE',   #How many observation of client's social surroundings with observable 60 DPD (days past due) default
    'DEF_30_CNT_SOCIAL_CIRCLE',   #How many observation of client's social surroundings defaulted on 30 DPD (days past due) 
    'DEF_60_CNT_SOCIAL_CIRCLE'    #How many observation of client's social surroundings defaulted on 60 (days past due) DPD
]
len(features_for_history)
features_for_client = [
    'CODE_GENDER',          #Gender of the client
    
    'NAME_EDUCATION_TYPE',  #Level of highest education the client achieved
    'NAME_FAMILY_STATUS',   #Family status of the client
    'CNT_FAM_MEMBERS',      #How many family members does client have
    'CNT_CHILDREN',         #Number of children the client has
    
    'NAME_INCOME_TYPE',     #Clients income type (businessman, working, maternity leave,…)
    'AMT_INCOME_TOTAL',     #Income of the client
    'OCCUPATION_TYPE',      #What kind of occupation does the client have
    'ORGANIZATION_TYPE',    #Type of organization where client works

    'NAME_HOUSING_TYPE',    #What is the housing situation of the client (renting, living with parents, ...)
    'FLAG_OWN_REALTY',      #Flag if client owns a house or flat
    'LIVE_CITY_NOT_WORK_CITY',     #Flag if client's contact address does not match work address (1=different, 0=same, at city level)
    'LIVE_REGION_NOT_WORK_REGION', #Flag if client's contact address does not match work address (1=different, 0=same, at region level)
    'REG_CITY_NOT_LIVE_CITY',      #Flag if client's permanent address does not match contact address (1=different, 0=same, at city level)
    'REG_CITY_NOT_WORK_CITY',      #Flag if client's permanent address does not match work address (1=different, 0=same, at city level)
    'REG_REGION_NOT_LIVE_REGION',  #Flag if client's permanent address does not match contact address (1=different, 0=same, at region level)
    'REG_REGION_NOT_WORK_REGION',  #Flag if client's permanent address does not match work address (1=different, 0=same, at region level)

    'FLAG_OWN_CAR',         #Flag if the client owns a car
    'OWN_CAR_AGE',          #Age of client's car
    
    'FLAG_CONT_MOBILE'      #Was mobile phone reachable (1=YES, 0=NO)
]
len(features_for_client)
features_for_client_building = [  
    'REGION_POPULATION_RELATIVE',  #Normalized population of region where client lives 
                                   #(higher number means the client lives in more populated region)
    'REGION_RATING_CLIENT',        #Our rating of the region where client lives (1,2,3)
    'REGION_RATING_CLIENT_W_CITY', #Our rating of the region where client lives with taking city into account (1,2,3)

    # Normalized information about building where the client lives
    # average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) 
    # apartment size, common area, living area, age of building, number of elevators, number of entrances, 
    # state of the building, number of floor
    'APARTMENTS_AVG', 
    'APARTMENTS_MEDI', 
    'APARTMENTS_MODE', 
    'BASEMENTAREA_AVG', 
    'BASEMENTAREA_MEDI', 
    'BASEMENTAREA_MODE', 
    'COMMONAREA_AVG', 
    'COMMONAREA_MEDI', 
    'COMMONAREA_MODE', 
    'ELEVATORS_AVG', 
    'ELEVATORS_MEDI', 
    'ELEVATORS_MODE', 
    'EMERGENCYSTATE_MODE', 
    'ENTRANCES_AVG', 
    'ENTRANCES_MEDI', 
    'ENTRANCES_MODE', 
    'FLOORSMAX_AVG',
    'FLOORSMAX_MEDI',
    'FLOORSMAX_MODE',
    'FLOORSMIN_AVG',
    'FLOORSMIN_MEDI',
    'FLOORSMIN_MODE',
    'FONDKAPREMONT_MODE',
    'HOUSETYPE_MODE',
    'LANDAREA_AVG',
    'LANDAREA_MEDI',
    'LANDAREA_MODE',
    'LIVINGAPARTMENTS_AVG',
    'LIVINGAPARTMENTS_MEDI',
    'LIVINGAPARTMENTS_MODE',
    'LIVINGAREA_AVG',
    'LIVINGAREA_MEDI',
    'LIVINGAREA_MODE',
    'NONLIVINGAPARTMENTS_AVG',
    'NONLIVINGAPARTMENTS_MEDI',
    'NONLIVINGAPARTMENTS_MODE',
    'NONLIVINGAREA_AVG',
    'NONLIVINGAREA_MEDI',
    'NONLIVINGAREA_MODE',
    'TOTALAREA_MODE',
    'WALLSMATERIAL_MODE',
    'YEARS_BEGINEXPLUATATION_AVG', 
    'YEARS_BEGINEXPLUATATION_MEDI',
    'YEARS_BEGINEXPLUATATION_MODE',
    'YEARS_BUILD_AVG',
    'YEARS_BUILD_MEDI',
    'YEARS_BUILD_MODE'
]
len(features_for_client_building)
features_unrecognized = [  
    'EXT_SOURCE_1', #Normalized score from external data source
    'EXT_SOURCE_2', #Normalized score from external data source
    'EXT_SOURCE_3'  #Normalized score from external data source
]
len(features_unrecognized)
df_application_train.describe(include = 'object').T
categorical_train = df_application_train.columns[df_application_train.dtypes == 'object']
categorical_test = df_application_test.columns[df_application_test.dtypes == 'object']
print(len(categorical_train), len(categorical_test), len(np.intersect1d(categorical_train, categorical_test)))
for c in categorical_train:
    c_train = set(df_application_train[c].unique())
    c_test = set(df_application_test[c].unique())
    diff = c_train ^ c_test
    if len(diff) > 0:
        print('feature ' + c + ' has different values: ', diff)
df_application_train['CODE_GENDER'].value_counts()
df_application_train['NAME_INCOME_TYPE'].value_counts()
df_application_train['NAME_FAMILY_STATUS'].value_counts()
df_application_train['CODE_GENDER'] = df_application_train['CODE_GENDER'] \
                                                        .map(lambda x: x if x != 'XNA' else np.nan)
df_application_train['NAME_INCOME_TYPE'] = df_application_train['NAME_INCOME_TYPE'] \
                                                        .map(lambda x: x if x != 'Maternity leave' else np.nan)
df_application_train['NAME_FAMILY_STATUS'] = df_application_train['NAME_FAMILY_STATUS'] \
                                                        .map(lambda x: x if x != 'Unknown' else np.nan)
for c in df_application_train.columns[df_application_train.dtypes == 'object']:
    d = df_application_train[c].value_counts()
    if df_application_train[c].nunique() == 2:
        d[0] = 0
        d[1] = 1
            
    df_application_train[c] = df_application_train[c].map(d)
    df_application_test[c] = df_application_test[c].map(d)
df_application_train.info()
df_application_test.info()
features_with_small_variance_train = df_application_train.columns[(df_application_train.std(axis = 0) < .01) \
                                                                                                      .values]
len(features_with_small_variance_train)
df_application_train[features_with_small_variance_train].describe().T
features_with_small_variance_test = df_application_test.columns[(df_application_test.std(axis = 0) < .01).values]
len(features_with_small_variance_test)
df_application_test[features_with_small_variance_test].describe().T
df_application_train.drop(features_with_small_variance_test, axis = 1, inplace = True)
df_application_test.drop(features_with_small_variance_test, axis = 1, inplace = True)
print(df_application_train.shape, df_application_test.shape)
for f in features_with_small_variance_test:
    features_for_loan.remove(f)
len(features_for_loan)
plt.figure(figsize = (25, 10))
sns.heatmap(df_application_train[['TARGET'] + features_for_loan].corr(), 
            cmap = plt.cm.RdYlBu_r, annot = True, vmin = -1., vmax = 1)
plt.title('Correlation map for features_for_loan');
plt.figure(figsize = (25, 10))
sns.heatmap(df_application_train[['TARGET'] + features_for_history].corr(), 
            cmap = plt.cm.RdYlBu_r, annot = True, vmin = -1., vmax = 1)
plt.title('Correlation map for features_for_history');
plt.figure(figsize = (25, 10))
sns.heatmap(df_application_train[['TARGET'] + features_for_client].corr(), 
            cmap = plt.cm.RdYlBu_r, annot = True, vmin = -1., vmax = 1)
plt.title('Correlation map for features_for_client');
plt.figure(figsize = (25, 10))
sns.heatmap(df_application_train[['TARGET'] + features_for_client_building].corr(), 
            cmap = plt.cm.RdYlBu_r, annot = True, vmin = -1., vmax = 1)
plt.title('Correlation map for features_for_client_building');
plt.figure(figsize = (25, 10))
sns.heatmap(df_application_train[['TARGET'] + features_unrecognized].corr(), 
            cmap = plt.cm.RdYlBu_r, annot = True, vmin = -1., vmax = 1)
plt.title('Correlation map for features_unrecognized');
features_with_good_corr = ['DAYS_BIRTH', 'DAYS_LAST_PHONE_CHANGE', 'DAYS_ID_PUBLISH', 
                           'CODE_GENDER', 'REG_CITY_NOT_WORK_CITY', 
                           'REGION_RATING_CLIENT_W_CITY', 'REGION_RATING_CLIENT', 
                           'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
plt.figure(figsize = (25, 6))
sns.heatmap(df_application_train[['TARGET'] + features_with_good_corr].corr(), 
            cmap = plt.cm.RdYlBu_r, annot = True, vmin = -1., vmax = 1)
plt.title('Correlation map for features_with_good_corr');
df_application_train.drop('EXT_SOURCE_1', axis = 1) \
                        .corrwith(df_application_train['EXT_SOURCE_1']) \
                        .map(abs) \
                        .sort_values(ascending = False) \
                        .head(10)
df_application_train.drop('EXT_SOURCE_2', axis = 1) \
                        .corrwith(df_application_train['EXT_SOURCE_2']) \
                        .map(abs) \
                        .sort_values(ascending = False) \
                        .head(10)
df_application_train.drop('EXT_SOURCE_3', axis = 1) \
                        .corrwith(df_application_train['EXT_SOURCE_3']) \
                        .map(abs) \
                        .sort_values(ascending = False) \
                        .head(10)