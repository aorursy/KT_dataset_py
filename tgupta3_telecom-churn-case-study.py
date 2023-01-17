# importing the required packages and libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import graphviz
import pydotplus
from IPython.display import Image
from sklearn.externals.six import StringIO

import sklearn
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression 
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import RFE

from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz

from sklearn.utils import resample
import io

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
# Read the data - please update the path of the csv in the statement below

path = r'../input/telecom_churn_data.csv'
telecom = pd.read_csv(path)

# creating a copy of the dataframe, just in case it is needed further down the assignment
telecom_copy = telecom.copy(deep = True)

# viewing the top few records
telecom.head()
telecom.shape
telecom.info()
telecom.describe()
# Having a look at the list of columns of the dataset before proceeding to clean the data

print(list(telecom.columns))
# Renaming the columns

telecom = telecom.rename(columns={'aug_vbc_3g': 'vbc_3g_8', 'jul_vbc_3g': 'vbc_3g_7',
                                  'jun_vbc_3g': 'vbc_3g_6', 'sep_vbc_3g': 'vbc_3g_9'})
# Checking the list of values in the circle 
telecom['circle_id'].value_counts()
plt.figure(figsize=(2,5))
sns.countplot(telecom['circle_id'])
plt.show()
telecom = telecom.drop('circle_id', axis = 1)
'''

    Since there are a lot of columns in the data, the null values will need to handled iteratively.
    The following function takes the dataframe as input, and returns the null percentage in the data in the descending order
    
'''

def null_pct(df):
    x = round((df.isnull().sum()/len(df.index) * 100),2).sort_values(ascending = False)
    return x
print(null_pct(telecom))
'''
    This method takes, three parameters -:
        1) A dataframe
        2) Lower range
        3) Upper range
        
    This function then returns a list of all the columns which have null values in the given range.

'''

def check_null_columns(df,a,b):
    try:
        x = (df.isna().sum() * 100 / len(df)).sort_values(ascending=False)
        y = x[x.between(a,b)]
        y = str(y).split()
        col = y[::2][:-1]
        if len(col) ==1:
            return []
        return col
    except:
        print("error in the function, no tracebacking :)")
t1 = check_null_columns(telecom, 70,80)
print(t1)
'''

    The following function accepts three parameters
        1. The dataframe in which the columns need to be imputed
        2. The list of columns that need to be imputed
        3. The list of values to be imputed in the respective columns (passed as the second parameter)

'''

def impute_col(df,col,val):
    for (i,j) in zip(col,val):
        df[i] = df[i].fillna(j)
    return df
# looking at the number of 2g recharges done in June

telecom['count_rech_2g_6'].describe()
plt.figure(figsize=(15,5))
sns.countplot(telecom['count_rech_2g_6'])
plt.show()
# Two lists are being created - one for the name of the column and one for the value it needs to be imputed with(for the null data)

col_imp = ['count_rech_2g_6']
val_imp = [0] 
# looking at the values in the date of last recharge for data in June

telecom['date_of_last_rech_data_6'].value_counts()
for i in t1:
    if 'date' in i:
        print(i)
drop_col = ['date_of_last_rech_data_6','date_of_last_rech_data_7','date_of_last_rech_data_8', 'date_of_last_rech_data_9']
# looking at the number of recharges done for a 3g pack in June
telecom['count_rech_3g_6'].describe()
plt.figure(figsize=(15,5))
sns.countplot(telecom['count_rech_3g_6'])
plt.show()
# appending the column in col_imp list and the value that needs to be imputed is appende din the val_imp list
col_imp.append('count_rech_3g_6')
val_imp.append(0)
print(t1)
telecom['av_rech_amt_data_6'].describe()
print(telecom[telecom['av_rech_amt_data_6'].isna()]['vol_2g_mb_6'].value_counts())
print(telecom[telecom['av_rech_amt_data_6'].isna()]['vol_3g_mb_6'].value_counts())
# similarly appending the column and the value to the required lists
col_imp.append('av_rech_amt_data_6')
val_imp.append(0)
telecom['max_rech_data_6'].describe()
# The column max_rech_data_6 needs to be dropped as it is not adding much value to the model.
drop_col.append('max_rech_data_6')
telecom['total_rech_data_6'].describe()
telecom['arpu_3g_6'].describe()
# Imputing the column arpu_3g_6 with the median, as it makes sense to impute the missing values with the median.

col_imp.append('arpu_3g_6')
val_imp.append(telecom['arpu_3g_6'].median())
telecom['arpu_2g_6'].describe()
# Imputing the column arpu_2g_6 with the median, as it makes sense to impute the missing values with the median.

col_imp.append('arpu_2g_6')
val_imp.append(telecom['arpu_2g_6'].median())
telecom['night_pck_user_6'].describe()
plt.figure(figsize=(3,5))
sns.countplot(telecom['night_pck_user_6'])
plt.show()
# Imputing the column night_pck_user_6 with the median, as it makes sense to impute the missing values with the median.

col_imp.append('night_pck_user_6')
val_imp.append(0)
telecom['fb_user_6'].value_counts()
plt.figure(figsize=(3,5))
sns.countplot(telecom['fb_user_6'])
plt.show()
# Getting the value counts for fb_user_6, to understand how is it related to the column vol_2g_mb_6.

print(telecom[telecom['fb_user_6'].isna()]['vol_2g_mb_6'].value_counts())
print(telecom[telecom['fb_user_6'].isna()]['vol_3g_mb_6'].value_counts())
# Imputing the column fb_user_6 with the median, as it makes sense to impute the missing values with the median.

col_imp.append('fb_user_6')
val_imp.append(0)
telecom['count_rech_2g_7'].describe()
plt.figure(figsize=(15,5))
sns.countplot(telecom['count_rech_2g_7'])
plt.show()
# Imputing the column count_rech_2g_7 with 0 as its value.

col_imp.append('count_rech_2g_7')
val_imp.append(0)
telecom['count_rech_3g_7'].describe()
plt.figure(figsize=(15,5))
sns.countplot(telecom['count_rech_3g_7'])
plt.show()
# Here we are appending a column and its value with count_rech_3g_7 and 0 in the required lists. 
# These lists will later be used for imputing.

col_imp.append('count_rech_3g_7')
val_imp.append(0)
telecom['av_rech_amt_data_7'].describe()
# Getting the value counts for av_rech_amt_data_7, to understand how is it related to the column vol_2g_mb_7.

print(telecom[telecom['av_rech_amt_data_7'].isna()]['vol_2g_mb_7'].value_counts())
print(telecom[telecom['av_rech_amt_data_7'].isna()]['vol_3g_mb_7'].value_counts())
# Below lines will help in imputing the column av_rech_amt_data_7 with the vaue 0
col_imp.append('av_rech_amt_data_7')
val_imp.append(0)
telecom['max_rech_data_7'].describe()
# we are dropping this column as well. Since it is not making much sense.
drop_col.append('max_rech_data_7')
telecom['total_rech_data_7'].describe()
#This will be handled later
telecom['arpu_3g_7'].describe()
# Here we have imputed the column arpu_3g_7 with it's median value.

col_imp.append('arpu_3g_7')
val_imp.append(telecom['arpu_3g_7'].median())
telecom['arpu_2g_7'].describe()
# Below lines will help in imputing the column arpu_2g_7 with it's median value.

col_imp.append('arpu_2g_7')
val_imp.append(telecom['arpu_2g_7'].median())
telecom['night_pck_user_7'].describe()
plt.figure(figsize=(3,5))
sns.countplot(telecom['night_pck_user_7'])
plt.show()
# the column night_pck_user_7 will be imputed with it's median vaue.

col_imp.append('night_pck_user_7')
val_imp.append(0)
telecom['fb_user_7'].value_counts()
plt.figure(figsize=(3,5))
sns.countplot(telecom['fb_user_7'])
plt.show()
# Understanding the relation of fb_user_7, when it is null. What happens to the column vol_2g_mb_7. This will help us in 
# indentifying the value with which we can impute the column fb_user_7

print(telecom[telecom['fb_user_7'].isna()]['vol_2g_mb_7'].value_counts())
print(telecom[telecom['fb_user_7'].isna()]['vol_3g_mb_7'].value_counts())
# Imputing the column fb_user_7 with a 0 value.

col_imp.append('fb_user_7')
val_imp.append(0)
telecom['count_rech_2g_8'].describe()
plt.figure(figsize=(15,5))
sns.countplot(telecom['count_rech_2g_8'])
plt.show()
# Since, most of the data for the column count_rech_2g_8 is 0. Hence we are imputing the values with 0.

col_imp.append('count_rech_2g_8')
val_imp.append(0)
telecom['count_rech_3g_8'].describe()
plt.figure(figsize=(15,5))
sns.countplot(telecom['count_rech_3g_8'])
plt.show()
# Since, most of the data for the column count_rech_3g_8 is lying around 0. Hence we are imputing the values with 0.
col_imp.append('count_rech_3g_8')
val_imp.append(0)
telecom['av_rech_amt_data_8'].describe()
# It seems if the column av_rech_amt_data_8 is null, the volume of the data used is 0. 
# Hence we will impute the column av_rech_amt_data_8 with 0.

print(telecom[telecom['av_rech_amt_data_8'].isna()]['vol_2g_mb_8'].value_counts())
print(telecom[telecom['av_rech_amt_data_8'].isna()]['vol_3g_mb_8'].value_counts())
# Iputing the column av_rech_amt_data_8 with a zero value.
col_imp.append('av_rech_amt_data_8')
val_imp.append(0)
telecom['max_rech_data_8'].describe()
# dropping the column  max_rech_data_8 as it does not seem to be making much sense.
drop_col.append('max_rech_data_8')
telecom['total_rech_data_8'].describe()
#This is being handled later
telecom['arpu_3g_8'].describe()
# Below lines will help in imputing the column arpu_3g_8 with it's median value.

col_imp.append('arpu_3g_8')
val_imp.append(telecom['arpu_3g_8'].median())
telecom['arpu_2g_8'].describe()
# Below lines will help in imputing the column arpu_2g_8 with it's median value.

col_imp.append('arpu_2g_8')
val_imp.append(telecom['arpu_2g_8'].median())
telecom['night_pck_user_8'].describe()
plt.figure(figsize=(3,5))
sns.countplot(telecom['night_pck_user_8'])
plt.show()
# Here we have added the column to the col_imp list and the value also to the val_imp list. 
# So, that they an later on be imputed.
col_imp.append('night_pck_user_8')
val_imp.append(telecom['night_pck_user_8'].median())
telecom['fb_user_8'].value_counts()
plt.figure(figsize=(3,5))
sns.countplot(telecom['fb_user_8'])
plt.show()
# Understanding the relation of fb_user_8, when it is null. What happens to the column vol_2g_mb_8. This will help us in 
# indentifying the value with which we can impute the column fb_user_8

print(telecom[telecom['fb_user_8'].isna()]['vol_2g_mb_8'].value_counts())
print(telecom[telecom['fb_user_8'].isna()]['vol_3g_mb_8'].value_counts())
# imputing the column vol_2g_mb_8 with a zero value.
col_imp.append('fb_user_8')
val_imp.append(0)
telecom['count_rech_2g_9'].describe()
plt.figure(figsize=(15,5))
sns.countplot(telecom['count_rech_2g_9'])
plt.show()
# hre We have added the column to the col_imp list and the value also to the val_imp list. SO, that they an later on be imputed.

col_imp.append('count_rech_2g_9')
val_imp.append(0)
telecom['count_rech_3g_9'].describe()
plt.figure(figsize=(15,5))
sns.countplot(telecom['count_rech_3g_9'])
plt.show()
# Here we are adding the column count_rech_3g_9 to a list and the value 0 to a value list. They can be used for imputing.

col_imp.append('count_rech_3g_9')
val_imp.append(0)
telecom['av_rech_amt_data_9'].describe()
# Understanding the relation between av_rech_amt_data_9 and vol_2g_mb_9 and then imputing the column av_rech_amt_data_9.

print(telecom[telecom['av_rech_amt_data_9'].isna()]['vol_2g_mb_9'].value_counts())
print(telecom[telecom['av_rech_amt_data_9'].isna()]['vol_3g_mb_9'].value_counts())
# Imputing the column av_rech_amt_data_9 with a value 0.
col_imp.append('av_rech_amt_data_9')
val_imp.append(0)
telecom['max_rech_data_9'].describe()
# Dropping the column max_rech_data_9 as this does not seem to be making sense.

drop_col.append('max_rech_data_9')
telecom['total_rech_data_9'].describe()
telecom['arpu_3g_9'].describe()
# Imputing the column arpu_3g_9 with it's median.

col_imp.append('arpu_3g_9')
val_imp.append(telecom['arpu_3g_9'].median())
telecom['arpu_2g_9'].describe()
# Imputing the column arpu_2g_9 with it's median.

col_imp.append('arpu_2g_9')
val_imp.append(telecom['arpu_2g_9'].median())
telecom['night_pck_user_9'].describe()
plt.figure(figsize=(3,5))
sns.countplot(telecom['night_pck_user_9'])
plt.show()
# Imputing the column night_pck_user_9 with it's median.

col_imp.append('night_pck_user_9')
val_imp.append(telecom['night_pck_user_9'].median())
telecom['fb_user_9'].value_counts()
plt.figure(figsize=(3,5))
sns.countplot(telecom['fb_user_9'])
plt.show()
# Understand the relation between fb_user_9 and vol_2g_mb_9. 

print(telecom[telecom['fb_user_9'].isna()]['vol_2g_mb_9'].value_counts())
print(telecom[telecom['fb_user_9'].isna()]['vol_3g_mb_9'].value_counts())
# Imputing the column fb_user_9 with the value 0.
col_imp.append('fb_user_9')
val_imp.append(0)
# here we are passing the dataframe, col_imp and val_imp list to a function impute_col. Which will

telecom = impute_col(telecom,col_imp, val_imp)

telecom_copy_2 = telecom.copy(deep=True)
# filling all the total recharge data values in case they have NA with the sum of their respective month's 2g and 3g value.
telecom['total_rech_data_6'].fillna((telecom['count_rech_3g_6'] + telecom['count_rech_2g_6']), inplace = True)
telecom['total_rech_data_7'].fillna((telecom['count_rech_3g_7'] + telecom['count_rech_2g_7']), inplace = True)
telecom['total_rech_data_8'].fillna((telecom['count_rech_3g_8'] + telecom['count_rech_2g_8']), inplace = True)
telecom['total_rech_data_9'].fillna((telecom['count_rech_3g_9'] + telecom['count_rech_2g_9']), inplace = True)
# checking all the columns which have null values between 70 and 80 %
print(sorted(check_null_columns(telecom,70,80)))
# dropping all the date related columns
print(sorted(list(drop_col)))
telecom = telecom.drop(drop_col, axis = 1)
# since we have removed all the date columns, and the other values have been imputed , hence now we have no null values
# for any columns in the range 70 - 80 %
print(check_null_columns(telecom,70,80))
# Similarly there is no column in the range 60 - 70 with Null values
print(check_null_columns(telecom,60,70))
# THere is no column in the range 50 - 60 with Null values also
print(check_null_columns(telecom,50,60))
# checking for the maximum percentage of NA values in the columns
print(null_pct(telecom))
# fetching the null values list of columns which lies in the range 7 - 10.
t1 = check_null_columns(telecom,7,10)
print(sorted(t1))
telecom = telecom.drop(t1, axis = 1)
# Again checking for maximum null valuesin the dataframe.
print(null_pct(telecom))
'''
    Here we still have lot of columns which have null values. Hence we will split the range intervals further
'''
print(list(telecom.columns))
'''
    The null range interval is 4-6. Here we will get a list of all those columns which have null values
'''
t1 = check_null_columns(telecom,4,6)
print(sorted(t1))
telecom[t1].describe()
# impute all the values with the respective median. Almost all the columns have outliers, and imputing the null values with mean
# will distort the interpretations and modeling

col_imp=[]
val_imp=[]
col_imp.extend(['ic_others_8', 'isd_ic_mou_8', 'isd_og_mou_8', 'loc_ic_mou_8', 'loc_ic_t2f_mou_8', 'loc_ic_t2m_mou_8', 
                'loc_ic_t2t_mou_8', 'loc_og_mou_8', 'loc_og_t2c_mou_8', 'loc_og_t2f_mou_8', 'loc_og_t2m_mou_8', 
                'loc_og_t2t_mou_8', 'offnet_mou_8', 'og_others_8', 'onnet_mou_8', 'roam_ic_mou_8', 'roam_og_mou_8', 
                'spl_ic_mou_8', 'spl_og_mou_8', 'std_ic_mou_8', 'std_ic_t2f_mou_8', 'std_ic_t2m_mou_8', 'std_ic_t2o_mou_8', 
                'std_ic_t2t_mou_8', 'std_og_mou_8', 'std_og_t2c_mou_8', 'std_og_t2f_mou_8', 'std_og_t2m_mou_8', 
                'std_og_t2t_mou_8'])

'''
    In the below for loop we are imputing all the remaining columns with their median values. Also, we are not doing any
    kind of imputation if the column ends with _9 i.e. if the column is for Septembet month here.
'''

for i in col_imp:
    if "_9" in i:
        continue
    val_imp.append(telecom[i].median())
'''
    Here, we are passing the lists created in the above cell, containing columns and their median values. With this we can
    get all the columns imputed with their median vlaues.
'''
telecom = impute_col(telecom,col_imp, val_imp)
'''
    Now, we donot see any null values in the range 4-6
'''
t1 = check_null_columns(telecom, 4, 6)
print(t1)
t1 = check_null_columns(telecom, 3, 4)
print(t1)
# impute all the values with the respective median. Almost all the columns have outliers, and imputing the null values with mean
# will distort the interpretations and modeling

col_imp=[]
val_imp=[]
col_imp.extend(['loc_ic_t2f_mou_6', 'std_og_t2c_mou_6', 'roam_og_mou_6', 'std_og_mou_6', 'std_ic_t2m_mou_6', 'loc_og_t2t_mou_6',
                'std_ic_mou_6', 'loc_og_t2m_mou_6', 'loc_og_t2f_mou_6', 'roam_ic_mou_6', 'std_og_t2f_mou_6', 'std_ic_t2f_mou_6',
                'loc_og_t2c_mou_6', 'loc_og_mou_6', 'std_og_t2m_mou_6', 'loc_ic_t2t_mou_6', 'std_ic_t2o_mou_6', 
                'std_og_t2t_mou_6', 'isd_og_mou_6', 'isd_ic_mou_6', 'spl_ic_mou_6', 'offnet_mou_6', 'std_ic_t2t_mou_6', 
                'spl_og_mou_6', 'onnet_mou_6', 'loc_ic_t2m_mou_6', 'og_others_6', 'loc_ic_mou_6', 'ic_others_6', 'spl_ic_mou_7',
                'std_og_t2m_mou_7', 'std_ic_t2o_mou_7', 'std_og_t2f_mou_7', 'isd_ic_mou_7', 'ic_others_7', 'std_ic_mou_7', 
                'spl_og_mou_7', 'std_og_t2c_mou_7', 'isd_og_mou_7', 'std_og_mou_7', 'std_ic_t2f_mou_7', 'std_og_t2t_mou_7', 
                'std_ic_t2m_mou_7', 'loc_ic_t2f_mou_7', 'loc_ic_t2m_mou_7', 'loc_ic_mou_7', 'onnet_mou_7', 'offnet_mou_7', 
                'std_ic_t2t_mou_7', 'loc_og_mou_7', 'roam_og_mou_7', 'loc_og_t2t_mou_7', 'roam_ic_mou_7', 'og_others_7', 
                'loc_ic_t2t_mou_7', 'loc_og_t2c_mou_7', 'loc_og_t2m_mou_7', 'loc_og_t2f_mou_7'])
'''
    In the below for loop we are imputing all the columns with their median values. Also, we are not doing any
    kind of imputation if the column ends with _9 i.e. if the column is for Septembet month here.
'''

for i in col_imp:
    if "_9" in i:
        continue
    val_imp.append(telecom[i].median())
'''
    Here, we are passing the lists created in the above cell, containing columns and their median values. With this we can
    get all the columns imputed with their median vlaues.
'''

telecom = impute_col(telecom,col_imp, val_imp)
'''
    Even in the range 0.1 -3 we find few columns with the null vlaues.
'''
t1 = check_null_columns(telecom, 0.1, 3)
print(t1)
col_imp=[]
val_imp=[]
col_imp=['loc_og_t2o_mou', 'std_og_t2o_mou', 'loc_ic_t2o_mou'] # the other columns in the above list are date columns, 
                                                                # and therfore not being included

'''
    In the below for loop we are imputing all the remaining columns with their median values. Also, we are not doing any
    kind of imputation if the column ends with _9 i.e. if the column is for Septembet month here.
'''

for i in col_imp:
    if "_9" in i:
        continue
    val_imp.append(telecom[i].median())
'''
    Here, we are passing the lists created in the above cell, containing columns and their median values. With this we can
    get all the columns imputed with their median vlaues.
'''

telecom = impute_col(telecom,col_imp, val_imp)
print(telecom.shape)
'''
    Checking if, there are still columns left for imputation.
'''
print(null_pct(telecom))
'''
    Since, we have only date columns left with Null values. So, we will be removing them
'''
telecom = telecom.drop(['date_of_last_rech_9', 'date_of_last_rech_8', 'date_of_last_rech_7', 'last_date_of_month_9', 
                         'date_of_last_rech_6', 'last_date_of_month_8', 'last_date_of_month_7'], axis = 1)
# checking the null percentage between 0.1 and 100
print(check_null_columns(telecom,0.1,100))
# checking if any columns in the data have null values at all
print(null_pct(telecom))
telecom['rech_amt_avg_6_7'] = (telecom['total_rech_amt_6'] + telecom['total_rech_amt_7'])/2
telecom[['total_rech_amt_6', 'total_rech_amt_7', 'rech_amt_avg_6_7']].head(10)
temp = telecom['rech_amt_avg_6_7'].quantile(0.70)
telecom_HVC = telecom[telecom['rech_amt_avg_6_7'] > temp ]
telecom_HVC.shape
telecom_HVC['temp_col'] = (telecom_HVC['total_ic_mou_9'] + telecom_HVC['total_og_mou_9'] + telecom_HVC['vol_2g_mb_9'] + telecom_HVC['vol_3g_mb_9'])
telecom_HVC['churn'] = telecom_HVC['temp_col'].apply(lambda x: 1 if x == 0 else 0 )
telecom_HVC = telecom_HVC.drop('temp_col', axis = 1)
# looking at the churn value in the dataset
telecom_HVC['churn'].head()
t1 = telecom_HVC[['arpu_6','arpu_7', 'arpu_8', 'churn']]
sns.pairplot(t1)
plt.show()
t1 = telecom_HVC[['total_rech_num_6','total_rech_num_7', 'total_rech_num_8', 'churn']]
sns.pairplot(t1)
plt.show()
# identifying the columns that belong to September. These end with _9
drop_9_columns = telecom.filter(regex='_9')
print(telecom.drop(list(drop_9_columns.columns),axis=1,inplace=True))
drop_9_columns.columns
# Creating a new column as the difference of the avg values in "good phase" and values in "action phase"

telecom_HVC['avg_rech_amt_diff'] = (telecom['total_rech_amt_6'] + telecom_HVC['total_rech_amt_7'])/2 - telecom['total_rech_amt_8']
telecom_HVC['avg_rech_num_diff'] = (telecom['total_rech_num_6'] + telecom_HVC['total_rech_num_7'])/2 - telecom['total_rech_num_8']
telecom_HVC['avg_og_mou_diff'] = (telecom['total_og_mou_6'] + telecom_HVC['total_og_mou_7'])/2 - telecom['total_og_mou_8']
telecom_HVC['max_rech_amt_diff'] = (telecom_HVC['max_rech_amt_6']+telecom_HVC['max_rech_amt_7'])/2 - telecom_HVC['max_rech_amt_8']
telecom_HVC['avg_vbc_3g_diff'] = (telecom_HVC['vbc_3g_6']+telecom_HVC['vbc_3g_7'])/2 - telecom_HVC['vbc_3g_8']

# Age of customer on the network is given. 
# This can be used to derive the loyalty status of the customer
telecom_HVC['loyalty_temp'] = round(telecom_HVC['aon']/365)
telecom_HVC['loyalty_temp'].head(10)
telecom_HVC['loyalty_temp'].value_counts()
# creating a categoricalderived feature termed as loyalty.

bins = [0,4,8,13]
labels = ['not loyal','loyal','very loyal']
telecom_HVC['loyalty'] = pd.cut(telecom_HVC['loyalty_temp'], bins=bins, labels=labels)
telecom_HVC['loyalty'].head(10)
telecom_HVC['loyalty'].value_counts()
print(list(telecom_HVC.columns))
print("\nLoaylty split for non-churned customers")
print(telecom_HVC[telecom_HVC['churn']==0]['loyalty'].value_counts())
print("\nLoaylty split for churned customers")
print(telecom_HVC[telecom_HVC['churn']==1]['loyalty'].value_counts())
sns.catplot(x = "loyalty", y ="churn",kind = "violin", data=telecom_HVC)
plt.show()
drop_9_columns = telecom_HVC.filter(regex='_9')
print(telecom_HVC.drop(list(drop_9_columns.columns),axis=1,inplace=True))

drop_9_columns = telecom_HVC.filter(regex='last_date_of_month')
print(telecom_HVC.drop(list(drop_9_columns.columns),axis=1,inplace=True))
telecom_HVC.head()
t_dummies_df = pd.get_dummies(telecom_HVC['loyalty'], drop_first = True)

telecom_HVC = pd.concat([telecom_HVC, t_dummies_df], axis = 1)
telecom_HVC = telecom_HVC.drop(['loyalty', 'loyalty_temp'], axis = 1)
telecom_HVC.shape
telecom_HVC.head()
'''

    Here, we pass a dataframe and all the columns for which we want to check for outliers.
    Post execution of the function we get a dataframe, which has the outliers removed.
    
'''

def remove_outliers(df,col):
    for i in col:
        Q1 = df[i].quantile(0.05)
        Q3 = df[i].quantile(0.995)
        df = df[(df[i] >=Q1) &(df[i] <=Q3)]
    return df
df_no_outlier = remove_outliers(telecom_HVC, list(telecom_HVC.columns))
df_no_outlier.shape
# creating a copy of the dataframe to use it for identifying the number Principal components required
telecom_HVC_copy = telecom_HVC.copy(deep = True)
# creating an index column with the mobile number as its value
y_ind = telecom_HVC_copy.pop('mobile_number')
# storing the target column in the variable "y"
y = telecom_HVC_copy.pop('churn')
# instantiating an object of the class StandardScaler
scaler = StandardScaler()

# fitting and transforming the dataset
telecom_HVC_scaled = scaler.fit_transform(telecom_HVC_copy)

telecom_HVC_scaled.shape
t_HVC_scaled = pd.DataFrame(telecom_HVC_scaled)
t_HVC_scaled.columns = telecom_HVC_copy.columns
t_HVC_scaled.head()
# All the relevant libraries were imported in the beginning
# instantiating an object of the class PCA
pca = PCA(random_state=42)

# fitting on the scaled dataset
pca.fit(t_HVC_scaled)
# Taking a look at the resultant PCA components
pca.components_
# taking a look at the explained variance ratio by the Principal components
pca.explained_variance_ratio_
# plotting the explained variance ratio 
plt.bar(range(1,len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_)
plt.show()
var_cumu = np.cumsum(pca.explained_variance_ratio_)
var_cumu
plt.figure(figsize=(10,8))
plt.vlines(x = 75, ymax=1, ymin=0.1, colors="r", linestyles="--")
plt.vlines(x = 55, ymax=1, ymin=0.1, colors="b", linestyles="--")
plt.hlines(y = 0.90, xmax=120, xmin=0, colors="g", linestyles="--")
plt.hlines(y = 0.96, xmax=120, xmin=0, colors="b", linestyles="--")
plt.plot(var_cumu)
plt.ylabel("Cumulative variance explained")
plt.show()
# creating a PCA object for 55 PCs now
pca_final = IncrementalPCA(n_components = 55)

# this PCA object will now be used to fit and transform the scaled country dataset
# when transforming is done, the inverse of the PCs is multiplied with the scaled dataset and the result is the corresponding values 
# of the scaled dataset with the PCs as the new basis
new_var = pca_final.fit_transform(t_HVC_scaled)
new_var
# To work further with the new dataset, a dataframe needs to be created

columns=[]
n_components = 55
for i in range(1,n_components+1):
    columns.append("PC"+str(i))


df = pd.DataFrame(new_var, columns =columns)
df.head()
cols = list(telecom_HVC_copy.columns)
temp = pd.DataFrame({'PC1':pca_final.components_[0], 'PC2':pca_final.components_[1],
                     'PC3':pca_final.components_[2], 'PC4':pca_final.components_[3],
                     'PC5':pca_final.components_[4], 'Feature':cols})
temp.head()
df_train, df_test = train_test_split(telecom_HVC, test_size=0.2, random_state = 100)
# looking athe counts of the "churn" in the dataset
telecom_HVC['churn'].value_counts()
# calculating the imbalance percentage in the dataset
imbal = round((telecom_HVC['churn'] == 1).sum()/len(telecom_HVC) * 100,2)
imbal
# Separate majority and minority classes
t_majority = df_train[df_train.churn==0]
t_minority = df_train[df_train.churn==1]
 
# Upsample minority class
t_minority_upsampled = resample(t_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=len(t_majority),    # to match majority class
                                 random_state=123) # reproducible results
 
# Combine majority class with upsampled minority class
t_upsampled = pd.concat([t_majority, t_minority_upsampled])
 
# Display new class counts
t_upsampled.churn.value_counts()
df_train = t_upsampled.copy(deep = True)
y_train = df_train.pop('churn')
y_ind = df_train.pop('mobile_number')
X_train = df_train.copy(deep=True)
y_test = df_test.pop('churn')
y_test_ind = df_test.pop('mobile_number')
X_test = df_test.copy(deep = True)
X_train_c = X_train.copy(deep = True)
X_test_c = X_test.copy(deep = True)
y_train_c = y_train.copy(deep = True)
y_test_c = y_test.copy(deep = True)
scaler = StandardScaler() 
  
X_train = scaler.fit_transform(X_train) 
X_test = scaler.transform(X_test) 
pca_final = PCA(n_components = 55) 


X_train_pca = pca_final.fit_transform(X_train)
X_test_pca = pca_final.transform(X_test) 
  
explained_variance = pca_final.explained_variance_ratio_ 
# Fitting Logistic Regression To the training set 

lr_pca = LogisticRegression(random_state = 0) 
lr_pca.fit(X_train_pca, y_train) 
y_train.head()
t = pd.DataFrame(y_train)
t.columns = ['churn']
t['churn'].value_counts()
t1 = pd.DataFrame(y_test)
t1.columns = ['churn']
t1['churn'].value_counts()
y_train_pred_pca = lr_pca.predict(X_train_pca) 
# Predicting the test set result using  
# predict function under LogisticRegression  

lr_y_test_pred_pca = lr_pca.predict(X_test_pca) 
y_train_pred_pca = pd.DataFrame(y_train_pred_pca)
y_train_pred_pca.head()
lr_y_test_pred_pca = pd.DataFrame(lr_y_test_pred_pca)
lr_y_test_pred_pca.columns = ['churn']
lr_y_test_pred_pca.head()
# making confusion matrix between train set of Y and predicted value. 

cm_train_lr_pca = confusion_matrix(y_train, y_train_pred_pca) 
cm_train_lr_pca
# making confusion matrix between the test set of Y and predicted value
  
cm_test_lr_pca = confusion_matrix(y_test, lr_y_test_pred_pca) 
cm_test_lr_pca
'''
    This function takes the confusion metrics as an input and then prints -:
        1) Accuracy
        2) Sensitivity/Recall/TPR
        3) Specificity
        4) FPR and Precision for the same.

'''

def calculate_all_metrics(confusion):
    
    print("Confusion matrix obtained is \n{val}".format(val=confusion))
    
    TP = confusion[1,1] # true positive 
    TN = confusion[0,0] # true negatives
    FP = confusion[0,1] # false positives
    FN = confusion[1,0] # false negatives
    
    print("\nAccuracy score obtained is {val}%".format(val=round(((TP+TN)/(TP+TN+FP+FN))*100,2)))
    print("\nSensitivity/Recall/True Positive Rate for the above confusion matrix obtained is = {val}%".format(val=round((TP/(TP+FN))*100,2)))
    print("\nSpecificity for the above confusion matrix obtained is = {val}%".format(val=round((TN/(TN+FP))*100,2)))
    print("\nFalse Positive Rate for the above confusion matrix obtained is = {val}%".format(val=round((FP/(TN+FP))*100,2)))
    print("\nPrecision for the above confusion matrix obtained is = {val}%".format(val=round((TP/(TP+FP))*100,2)))
# evaluating all the metrics(sensitivity/specificity/precision/recall/TPR/FPR) for cm_train_lr_pca
calculate_all_metrics(cm_train_lr_pca)
# evaluating all the metrics(sensitivity/specificity/precision/recall/TPR/FPR) for cm_test_lr_pca
calculate_all_metrics(cm_test_lr_pca)
X_train_pca_dt = pd.DataFrame(X_train_pca)
X_test_pca_dt = pd.DataFrame(X_test_pca) 


# Fitting the decision tree with default hyperparameter max_depth is 3;
dt_pca_1 = DecisionTreeClassifier(max_depth = 3)
dt_pca_1.fit(X_train_pca, y_train)
# Putting features
features = list(X_train_pca_dt.columns[0:])
# Making predictions
dt_y_test_pred_pca_1 = dt_pca_1.predict(X_test_pca_dt)

# Printing classification report
print(classification_report(y_test, dt_y_test_pred_pca_1))
'''
    
    Here, we take a decision tree and its features as an input. This function returns the image of a decision tree
    which is helpful from the analysis perspective.

'''
def draw_decision_tree(dt,features):
    dot_data = io.StringIO()  
    export_graphviz(dt, out_file=dot_data,
                feature_names=features, filled=True,rounded=True)

    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    return(Image(graph.create_png()))
# evaluating all the metrics(sensitivity/specificity/precision/recall/TPR/FPR) for cm_pca_dt_1

cm_pca_dt_1 = confusion_matrix(y_test,dt_y_test_pred_pca_1)
calculate_all_metrics(cm_pca_dt_1)
'''
    here we have changed a few hyperparameters to understand how
    the confusion matrix and the other evaluation metrics change
'''
dt_pca_2 = DecisionTreeClassifier(max_depth=7,min_samples_split=200)
dt_pca_2.fit(X_train_pca, y_train)
# Making predictions
dt_y_test_pred_pca_2 = dt_pca_2.predict(X_test_pca_dt)
# Printing classification report
print(classification_report(y_test, dt_y_test_pred_pca_2))
# evaluating all the metrics(sensitivity/specificity/precision/recall/TPR/FPR) for cm_pca_dt_2

cm_pca_dt_2 = confusion_matrix(y_test,dt_y_test_pred_pca_2)
calculate_all_metrics(cm_pca_dt_2)
'''
    here we have changed a few hyperparameters to understand how
    the confusion matrix and the other evaluation metrics change
'''
dt_pca_3 = DecisionTreeClassifier(max_depth=5,min_samples_split=150,min_samples_leaf=25,random_state=100,max_leaf_nodes=15,
                                 criterion='entropy')
dt_pca_3.fit(X_train_pca, y_train)
# Making predictions
dt_y_test_pred_pca_3 = dt_pca_3.predict(X_test_pca_dt)
# Printing classification report
print(classification_report(y_test, dt_y_test_pred_pca_3))
# evaluating all the metrics(sensitivity/specificity/precision/recall/TPR/FPR) for cm_pca_dt_3

cm_pca_dt_3 = confusion_matrix(y_test,dt_y_test_pred_pca_3)
calculate_all_metrics(cm_pca_dt_3)
'''
    Plotting the simple decision tree with a depth 4, the first Decision Tree

'''
draw_decision_tree(dt_pca_1,features)
'''
    Plotting a complex decision tree with depth 7, the second Decision Tree

'''
draw_decision_tree(dt_pca_2,features)
'''
    
    Plotting a complex decision tree with depth 5, the third Decision Tree.

'''
draw_decision_tree(dt_pca_3,features)
# Running the random forest with default parameters.
rfc = RandomForestClassifier()
# fit
rfc.fit(X_train_pca,y_train)
# Making predictions for train dataset
predictions_train = rfc.predict(X_train_pca)
print(classification_report(y_train,predictions_train))
# Making predictions for test dataset
rf_y_test_pred_pca_1 = rfc.predict(X_test_pca)
# Let's check the report of our default model
print(classification_report(y_test, rf_y_test_pred_pca_1))
cm_rfc_pca = confusion_matrix(y_test, rf_y_test_pred_pca_1)
calculate_all_metrics(cm_rfc_pca)
# Create the parameter grid based on the results of random search 
param_grid = {
    'max_depth': [4,8,10],
    'min_samples_leaf': range(100, 400, 200),
    'min_samples_split': range(200, 500, 200),
    'n_estimators': [100,200, 300], 
    'max_features': [5, 10]
}
# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1,verbose = 1)
# Fit the grid search to the data
grid_search.fit(X_train, y_train)
# printing the optimal accuracy score and hyperparameters
print('We can get accuracy of',grid_search.best_score_,'using',grid_search.best_params_)
# model with the best hyperparameters
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(bootstrap=True,
                             max_depth=10,
                             max_features=10,
                             min_samples_leaf=100, 
                             min_samples_split=200,
                             n_estimators=200)
# fiting the model
rfc.fit(X_train,y_train)
# predict
rf_y_test_pred_pca_2 = rfc.predict(X_test)
print(classification_report(y_test, rf_y_test_pred_pca_2))
cm_rfc_pca_2 = confusion_matrix(y_test, rf_y_test_pred_pca_2)
calculate_all_metrics(cm_rfc_pca_2)
'''
    This method takes 4 parameters -
    1. The actual value of the target variable
    2. The predicted value of the target variable from the first model
    3. The predicted value of the target variable from the second model
    4 The predicted value of the target variable from the third model
    
    This method then plots thres ROC curves, one for each model, b/w the actual taregt value and the predicted value
'''
def draw_roc_compare( actual, prob1, prob2, prob3):
    fpr1, tpr1, thresholds1 = metrics.roc_curve( actual, prob1, drop_intermediate = False )
    fpr2, tpr2, thresholds2 = metrics.roc_curve( actual, prob1, drop_intermediate = False )
    fpr3, tpr3, thresholds3 = metrics.roc_curve( actual, prob1, drop_intermediate = False )
    auc_score1 = metrics.roc_auc_score( actual, prob1)
    auc_score2 = metrics.roc_auc_score( actual, prob2)
    auc_score3 = metrics.roc_auc_score( actual, prob3)
    plt.figure(figsize=(5, 5))
    plt.plot( fpr1, tpr1, label='Model 1 (area = %0.2f)' % auc_score1, color = "r" )
    plt.plot( fpr2, tpr2, label='Model 2 (area = %0.2f)' % auc_score2, color = "b" )
    plt.plot( fpr3, tpr3, label='Model 3 (area = %0.2f)' % auc_score3, color = "g" )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None
draw_roc_compare(pd.DataFrame(y_test)['churn'] , pd.DataFrame(dt_y_test_pred_pca_1). iloc[:, 0],pd.DataFrame(dt_y_test_pred_pca_2). iloc[:, 0], pd.DataFrame(dt_y_test_pred_pca_3). iloc[:, 0])
draw_roc_compare(pd.DataFrame(y_test)['churn'] , lr_y_test_pred_pca['churn'],pd.DataFrame(dt_y_test_pred_pca_3). iloc[:, 0],pd.DataFrame(rf_y_test_pred_pca_2). iloc[:, 0])
# creating X_train, y_train, X_test and y_test from the respective copies created earlier
X_train = X_train_c.copy(deep = True)
X_test = X_test_c.copy(deep = True)
y_train = y_train_c.copy(deep = True)
y_test = y_test_c.copy(deep = True)
scaler = StandardScaler() 

cols = X_train.columns
X_train[cols] = scaler.fit_transform(X_train[cols]) 
X_test[cols] = scaler.transform(X_test[cols]) 
# Logistic regression model
logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm1.fit().summary()
logreg = LogisticRegression()
rfe = RFE(logreg, 45)
rfe = rfe.fit(X_train, y_train)
rfe.support_
col = X_train.columns[rfe.support_]
X_train.columns[~rfe.support_]
X_train_sm = sm.add_constant(X_train[col])
logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()
# calculate the VIF 

vif = pd.DataFrame()
vif['Features'] = X_train_sm.columns
vif['VIF'] = [variance_inflation_factor(X_train_sm.values,i) for i in range(X_train_sm.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
'''
    This function takes, a dataframe and its columns as input and removes all columns one by one having VIF greater than 5.
    It returns a dataframe at the end, where in the multicollinearity is removed.
    
'''

def calculate_vif(df,col):
    vif = pd.DataFrame()
    X = df[col]
    vif['Features'] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif['VIF'] = round(vif['VIF'], 2)
    vif = vif.sort_values(by = "VIF", ascending = False)
    if vif.iloc[0][1] < 5 :
        return X
    else:
        Y = X.drop(vif.iloc[0][0], axis =1)
        print("column dropped as part of VIF is  {val}".format(val = vif.iloc[0][0]))
        return(calculate_vif(X,Y.columns))
df_num = np.array(X_train[col].select_dtypes(include=[np.number]).columns.values)
df_vif = calculate_vif(X_train[col], df_num)
len(df_vif.columns)
# calculate the VIF again

vif = pd.DataFrame()
vif['Features'] = df_vif.columns
vif['VIF'] = [variance_inflation_factor(df_vif.values,i) for i in range(df_vif.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
X_train_sm_2 = sm.add_constant(df_vif)
logm2 = sm.GLM(y_train,X_train_sm_2, family = sm.families.Binomial())
res = logm2.fit()
res.summary()
df_vif = df_vif.drop("ic_others_8", axis = 1)
X_train_sm_2 = sm.add_constant(df_vif)
logm2 = sm.GLM(y_train,X_train_sm_2, family = sm.families.Binomial())
res = logm2.fit()
res.summary()
df_vif = df_vif.drop("max_rech_amt_diff", axis = 1)
X_train_sm_2 = sm.add_constant(df_vif)
logm2 = sm.GLM(y_train,X_train_sm_2, family = sm.families.Binomial())
res = logm2.fit()
res.summary()
df_vif = df_vif.drop("std_ic_t2t_mou_7", axis = 1)
X_train_sm_2 = sm.add_constant(df_vif)
logm2 = sm.GLM(y_train,X_train_sm_2, family = sm.families.Binomial())
res = logm2.fit()
res.summary()
df_vif = df_vif.drop("loc_ic_t2t_mou_7", axis = 1)
X_train_sm_2 = sm.add_constant(df_vif)
logm2 = sm.GLM(y_train,X_train_sm_2, family = sm.families.Binomial())
res = logm2.fit()
res.summary()
y_train_pred_lr = res.predict(X_train_sm_2)
y_train_pred_lr[:10]
y_train_pred_lr = y_train_pred_lr.values.reshape(-1)
y_train_pred_lr[:10]
y_train_array = np.array(y_train.values)
y_train_array = y_train_array.reshape(-1)
y_train_array[:10]
y_ind = y_ind.reset_index()
y_ind = y_ind.drop("index", axis = 1)
y_ind['mobile_number'].head()
y_train_pred_final_lr = pd.DataFrame({'Churn':y_train_array, 'Churn_Prob':y_train_pred_lr})
y_train_pred_final_lr['mobile_number'] = y_ind['mobile_number']
y_train_pred_final_lr.head()
y_train_pred_final_lr['predicted_Churn'] = y_train_pred_final_lr.Churn_Prob.map(lambda x: 1 if x > 0.5 else 0)

# Let's see the head
y_train_pred_final_lr.head()
# Confusion matrix 
cm_lr_train_1 = metrics.confusion_matrix(y_train_pred_final_lr.Churn, y_train_pred_final_lr.predicted_Churn )
calculate_all_metrics(cm_lr_train_1)
'''
    this method takes two column values as parameters - 1. the actual target variable and 
    2. the predicted value of the target variable from the model built
    
    This method then plots a ROC curve between the two values
'''
def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None
fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final_lr.Churn, y_train_pred_final_lr.Churn_Prob, 
                                         drop_intermediate = False )
#Finding Optimal Cutoff Point
#Optimal cutoff probability is that prob where we get balanced sensitivity and specificity
draw_roc(y_train_pred_final_lr.Churn, y_train_pred_final_lr.Churn_Prob)
# Let's create columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final_lr[i]= y_train_pred_final_lr.Churn_Prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final_lr.head()
# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final_lr.Churn, y_train_pred_final_lr[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
cutoff_df.head()
# Let's plot accuracy sensitivity and specificity for various probabilities.
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.vlines(x = 0.55, ymax=1, ymin=0.1, colors="b", linestyles="--")
plt.vlines(x = 0.52, ymax=1, ymin=0.1, colors="r", linestyles="--")
plt.show()
y_train_pred_final_lr['final_predicted'] = y_train_pred_final_lr.Churn_Prob.map( lambda x: 1 if x > 0.45 else 0)
y_train_pred_final_lr.head()
# Confusion matrix 
cm_lr_train_2 = metrics.confusion_matrix(y_train_pred_final_lr.Churn, y_train_pred_final_lr.final_predicted )
calculate_all_metrics(cm_lr_train_2)
X_train_sm_2.columns
X_train_sm_c = X_train_sm_2.copy(deep = True)
X_train_sm_c = X_train_sm_c.drop('const',axis=1)
col = X_train_sm_c.columns
X_test_sm = sm.add_constant(X_test[col])
y_test_pred_lr = res.predict(X_test_sm)
y_test_pred_lr[:10]
y_test_pred_lr = y_test_pred_lr.values.reshape(-1)
y_test_pred_lr[:10]
y_test_array = np.array(y_test.values)
y_test_array = y_test_array.reshape(-1)
y_test_array[:10]
y_test_pred_final_lr = pd.DataFrame({'Churn':y_test_array, 'Churn_Prob':y_test_pred_lr})
y_test_pred_final_lr['mobile_number'] = y_ind['mobile_number']
y_test_pred_final_lr.head()
y_test_pred_final_lr['final_predicted'] = y_test_pred_final_lr.Churn_Prob.map( lambda x: 1 if x > 0.45 else 0)
y_test_pred_final_lr.head()
cm_lr_test = metrics.confusion_matrix(y_test_pred_final_lr.Churn, y_test_pred_final_lr.final_predicted )
calculate_all_metrics(cm_lr_test)
# The final summary of the model
res.summary()
# Looking at the top 10 coefficients
lr_coeffs = pd.DataFrame(res.params)
lr_coeffs.reset_index(inplace = True)
lr_coeffs.columns = ['Feature', 'coeff value']
top_10 = lr_coeffs.sort_values(['coeff value'], ascending = False).head(10)
top_10
# Looking at the bottom 10 coefficients
bot_10 = lr_coeffs.sort_values(['coeff value'], ascending = False).tail(10)
bot_10
# Combining the coefficients to come up with the features that affect the model the most
lr_coef = pd.concat([top_10, bot_10], axis=0, sort=False)
lr_coef['absolute coeff'] = abs(lr_coef['coeff value'])
lr_coef.sort_values(by = 'absolute coeff', ascending = False).head(10)
# Plotting the coefficients
plt.rcParams['figure.figsize'] = (6.0, 6.0)
sns.barplot(x = 'coeff value', y = 'Feature', data = lr_coef)
plt.title("Telecom Churn Coefficients")
plt.grid()
plt.show()
# Fitting the decision tree with default hyperparameter max_depth = 3 
# The Decision Tree model need not be given the scaled data. 
# Therefore, the paramters passed would be the original training dataset where data imbalance was handled
dt_1 = DecisionTreeClassifier(max_depth = 3)
dt_1.fit(X_train_c, y_train_c)
# Importing required packages for visualization
from IPython.display import Image  
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydotplus, graphviz

# Putting features
features = list(X_train_c.columns[0:])
features
# plotting tree with max_depth=3
dot_data = StringIO()  
export_graphviz(dt_1, out_file=dot_data,
                feature_names=features, filled=True,rounded=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 
#graph.write_pdf("TelecomChurn.pdf")
# to draw it here, the following command can be used
draw_decision_tree(dt_1,features)
# Making predictions
dt_y_test_pred1 = dt_1.predict(X_test_c)

# Printing classification report
print(classification_report(y_test_c, dt_y_test_pred1))
# evaluating all the metrics(sensitivity/specificity/precision/recall/TPR/FPR) for cm_pca_dt_3

cm_dt = confusion_matrix(y_test_c, dt_y_test_pred1)
calculate_all_metrics(cm_dt)
'''
    here we have changed a few hyperparameters to understand how
    the confusion matrix and the other evaluation metrics change
'''

dt_2 = DecisionTreeClassifier(max_depth=7,min_samples_split=200)
dt_2.fit(X_train_c, y_train_c)
# to draw it here, the following command can be used
draw_decision_tree(dt_2,features)
# Making predictions
dt_y_test_pred2 = dt_2.predict(X_test_c)

# Printing classification report
print(classification_report(y_test_c, dt_y_test_pred2))
# evaluating all the metrics(sensitivity/specificity/precision/recall/TPR/FPR) for cm_pca_dt_3

cm_dt = confusion_matrix(y_test_c, dt_y_test_pred2)
calculate_all_metrics(cm_dt)
'''
    here we have changed a few hyperparameters to understand how
    the confusion matrix and the other evaluation metrics change
'''

dt_3 = DecisionTreeClassifier(max_depth=5,min_samples_split=150,min_samples_leaf=25,random_state=100,max_leaf_nodes=15,
                                 criterion='entropy')
dt_3.fit(X_train_c, y_train_c)
# to draw it here, the following command can be used
draw_decision_tree(dt_3,features)
# Making predictions
dt_y_test_pred3 = dt_3.predict(X_test_c)

# Printing classification report
print(classification_report(y_test_c, dt_y_test_pred3))
# evaluating all the metrics(sensitivity/specificity/precision/recall/TPR/FPR) for cm_pca_dt_3

cm_dt = confusion_matrix(y_test_c, dt_y_test_pred3)
calculate_all_metrics(cm_dt)
draw_roc_compare(pd.DataFrame(y_test_c)['churn'] , pd.DataFrame(dt_y_test_pred1). iloc[:, 0],pd.DataFrame(dt_y_test_pred2). iloc[:, 0],pd.DataFrame(dt_y_test_pred3). iloc[:, 0])
# just rechecking the shapes of the dataset
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
# base estimator: a weak learner with max_depth = 9
shallow_tree = DecisionTreeClassifier(max_depth = 9, random_state = 100)
# fit the shallow decision tree 
shallow_tree.fit(X_train, y_train)

# prediction on test dataset
y_pred = shallow_tree.predict(X_test)
cm_boosting_test_1 = confusion_matrix(y_test, y_pred)
calculate_all_metrics(cm_boosting_test_1)
from sklearn.ensemble import AdaBoostClassifier
estimators = list(range(1, 20, 2))

abc_scores = []
for n_est in estimators:
    ABC = AdaBoostClassifier(
    base_estimator=shallow_tree, 
    n_estimators = n_est)
    
    ABC.fit(X_train, y_train)
    y_pred = ABC.predict(X_test)
    cm = confusion_matrix(y_test, y_pred) 
    TP = cm[1,1]
    FN = cm[1,0]
    abc_scores.append(round((TP/(TP+FN))*100,2))
    print("\n\nThe value for all the relevant metrics for number of estimators {} \n".format(n_est))
    calculate_all_metrics(cm)
# plot test scores and n_estimators

plt.plot(estimators, abc_scores)
plt.xlabel('n_estimators')
plt.ylabel('Sensitivity')

plt.show()
# base estimator: a weak learner with max_depth = 11
shallow_tree1 = DecisionTreeClassifier(max_depth = 11, random_state = 100)
# fit the shallow decision tree 
shallow_tree1.fit(X_train_c, y_train_c)

# test error
boosting_y_pred = shallow_tree1.predict(X_test_c)
cm = confusion_matrix(y_test_c, boosting_y_pred)
calculate_all_metrics(cm)
draw_decision_tree(shallow_tree1,features)
draw_roc_compare(pd.DataFrame(y_test)['churn'] , y_test_pred_final_lr['final_predicted'],pd.DataFrame(dt_y_test_pred2). iloc[:, 0],pd.DataFrame(boosting_y_pred). iloc[:, 0])
# The metrics of the best predictor RF model are 
calculate_all_metrics(cm_rfc_pca_2)
# plotting the second decision tree and saving it as a pdf at the location same as the python notebook 
dot_data = StringIO()  
export_graphviz(dt_2, out_file=dot_data,feature_names=features, filled=True,rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 
#graph.write_pdf("TelecomChurn.pdf")
# to draw it here, the following command can be used
draw_decision_tree(dt_2,features)
importances = dt_2.feature_importances_
importances
importance_dict = dict(zip(X_train.columns, dt_2.feature_importances_))
# Getting the top 10 features of the decision tree
listofTuples = sorted(importance_dict.items() , reverse=True, key=lambda x: x[1])
cnt = 0
lst = []

# Iterate over the sorted sequence
for elem in listofTuples :
    lst.append([elem[0], elem[1]])

#converting the list to dataframe
dt_coef = pd.DataFrame(lst)
dt_coef.columns = ["Feature", "Weightage"]
dt_coef_top10 = dt_coef.head(10)
dt_coef.head(10)
# plotting the weightage
plt.rcParams['figure.figsize'] = (6.0, 6.0)
sns.barplot(x = 'Weightage', y = 'Feature', data = dt_coef_top10)
plt.title("Telecom Churn Coefficients")
plt.grid()
plt.show()