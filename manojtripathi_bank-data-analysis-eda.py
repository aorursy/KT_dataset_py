import numpy as np
import pandas as pd
import io
import requests
# url = "../input/bank-lone-credit-data/view"
# file1 = requests.get(url).content
# application_data = pd.read_csv(io.StringIO(file1.decode('utf-8')))
application_data=pd.read_csv("../input/application-data/application_data.csv")
previous_application=pd.read_csv("../input/previous/previous_application.csv")
# columns_data=pd.read_csv("../input/columns-descripation/columns_description.csv")
print(application_data.shape)
print(previous_application.shape)
pd.set_option('display.max_columns',400)# to view data with all columns....
application_data.head()
l=[]
l.append(application_data.head(0))
l
application_data.describe() # from this we'll came to know somthing about numerical values like no of recors in each col , mean, std
# ,min, max etc.
application_data.info()# by-default verbose=False if it true it will give info about all the 122 columns in the dataset. 
application_data.select_dtypes(include='object').columns # displaying all columns name whose datatype is string/object.
application_data.select_dtypes(include='int64').columns # displaying all columns name whose datatype is integer.
application_data.select_dtypes(include='float64').columns # displaying all columns name whose datatype is float.
# Checking for missing values in application_data
pd.set_option('display.max_rows', 130)
application_data.isnull().sum()
application_data=application_data.loc[:, application_data.isnull().sum()<= 140000] # here i have droped all col in which missing no are more or equal to 140000.
application_data.isnull().sum() # after removeing some of the col again checking for null values and imputing them too...
# a=application_data.isnull().sum()
# a=pd.DataFrame(a)
# a
# a[0]==True
application_data.select_dtypes(include='object').columns # we can not replace them by mean or any other int vlaue so we'll replace by mode vlaue
# replacing a categorical col null values with mode.
application_data.NAME_TYPE_SUITE.mode()# finding mode=Unaccompanied
application_data['NAME_TYPE_SUITE'].fillna(value='Unaccompanied', inplace=True)
# application_data[['AMT_ANNUITY','AMT_GOODS_PRICE','EXT_SOURCE_2','OBS_30_CNT_SOCIAL_CIRCLE','DEF_30_CNT_SOCIAL_CIRCLE',
#                  'OBS_60_CNT_SOCIAL_CIRCLE','DEF_60_CNT_SOCIAL_CIRCLE']].fillna(value=np.mean(application_data[['AMT_ANNUITY',
#                 'AMT_GOODS_PRICE','EXT_SOURCE_2','OBS_30_CNT_SOCIAL_CIRCLE','DEF_30_CNT_SOCIAL_CIRCLE',
#                 'OBS_60_CNT_SOCIAL_CIRCLE','DEF_60_CNT_SOCIAL_CIRCLE']]),inplace=True)

#replacing all null values where null counts is less the 1021

application_data['AMT_ANNUITY'].fillna(value=np.mean(application_data['AMT_ANNUITY']),inplace=True)
application_data['AMT_GOODS_PRICE'].fillna(value=np.mean(application_data['AMT_GOODS_PRICE']),inplace=True)
application_data['CNT_FAM_MEMBERS'].fillna(value=np.mean(application_data['CNT_FAM_MEMBERS']),inplace=True)
application_data['DAYS_LAST_PHONE_CHANGE'].fillna(value=np.mean(application_data['DAYS_LAST_PHONE_CHANGE']),inplace=True)
application_data['EXT_SOURCE_2'].fillna(value=np.mean(application_data['EXT_SOURCE_2']),inplace=True)
application_data['OBS_30_CNT_SOCIAL_CIRCLE'].fillna(value=np.mean(application_data['OBS_30_CNT_SOCIAL_CIRCLE']),inplace=True)
application_data['DEF_30_CNT_SOCIAL_CIRCLE'].fillna(value=np.mean(application_data['DEF_30_CNT_SOCIAL_CIRCLE']),inplace=True)
application_data['OBS_60_CNT_SOCIAL_CIRCLE'].fillna(value=np.mean(application_data['OBS_60_CNT_SOCIAL_CIRCLE']),inplace=True)
application_data['DEF_60_CNT_SOCIAL_CIRCLE'].fillna(value=np.mean(application_data['DEF_60_CNT_SOCIAL_CIRCLE']),inplace=True)

#replacing all other null values with 0 because they are lot in numbers and replicing them with mean is not a good option.

application_data[['OCCUPATION_TYPE','EXT_SOURCE_3','AMT_REQ_CREDIT_BUREAU_HOUR','AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_MON','AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_YEAR']]=application_data[['OCCUPATION_TYPE','EXT_SOURCE_3','AMT_REQ_CREDIT_BUREAU_HOUR','AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_WEEK' ,'AMT_REQ_CREDIT_BUREAU_MON','AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_YEAR']].fillna(value=0.0)


application_data.isnull().sum()
# previous_application data....................
previous_application.head()
print(previous_application.shape)
previous_application.columns

# Checking for categorical , numerical and float columns.........
previous_application.select_dtypes('object').columns
previous_application.select_dtypes(include='int64').columns
previous_application.select_dtypes(include='float64').columns
previous_application.info(verbose=False)
previous_application.describe() # from this we can say this this data has lots of outliers and most of the col contains squed data
#because some of values are in -ive and 0 and some are large +ive no.

# Checking for null values.........................
previous_application.isnull().sum()
