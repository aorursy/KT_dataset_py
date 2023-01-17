from google.colab import drive

drive.mount('/gdrive')

%cd /gdrive
# Import our libraries we are going to use for our data analysis.

import tensorflow as tf

import pickle

import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib import pyplot as plt

import nltk

from nltk.corpus import wordnet

from nltk.tokenize import word_tokenize

from nltk.stem import PorterStemmer

from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import preprocessing

from sklearn.utils import class_weight





# For oversampling Library (Dealing with Imbalanced Datasets)

from imblearn.over_sampling import SMOTE

from collections import Counter



# Other Libraries

import time



% matplotlib inline

#NLP

import nltk

nltk.download('words')

nltk.download("stopwords") 

nltk.download('punkt')

nltk.download('wordnet')

# Importing the necessary functions

import nltk, re

from nltk.corpus import stopwords

stop = stopwords.words('english')

from nltk.stem import SnowballStemmer

from nltk.tokenize import word_tokenize

from nltk.corpus import wordnet

from nltk.stem import WordNetLemmatizer

words = set(nltk.corpus.words.words())

import string



#libraries for machine learning algorithms

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

from sklearn.svm import SVC

from sklearn_pandas import DataFrameMapper, CategoricalImputer

from sklearn.svm import LinearSVR

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

#importing necessary Decision Tree libraries

from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier

#importing necessary Random Forest Classifier library

from sklearn.ensemble import RandomForestClassifier

#importing necessary MLP library for Neural Network

from sklearn.neural_network import MLPClassifier

#importing necessary library for Naiye Bayes

from sklearn.naive_bayes import GaussianNB

#importing necessary library for LDA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#importing necessary library for Gradient Boosting

from sklearn.ensemble import GradientBoostingClassifier

#importing necessary library for Support Vector Machines

from sklearn.svm import SVC

#importing necessary libraries for KNN classifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import f1_score

start_df = pd.read_csv("/gdrive/My Drive/Capstone Project - NLP/fullacc.csv",low_memory=False)
df = start_df.copy(deep=True)

df.head()
print('There are {} rows and {} columns in the dataset.'.format(df.shape[0],df.shape[1]))
#printing the name of columns

df.columns
# This will print basic statistics for numerical columns

df.describe()
#Removing all other predictors and their associated predictor columns containing amount related data other than loan amount which we will be using for modelling

df.drop(['funded_amnt','funded_amnt_inv','last_pymnt_amnt','delinq_amnt'],axis=1,inplace=True)
#removing grade, sub grade and interest columns

df.drop(['grade','sub_grade','int_rate'],axis=1,inplace=True)
df_description = pd.read_excel(r'/gdrive/My Drive/Capstone Project - NLP/Harsh/LCDataDictionary.xlsx').dropna()

print(df_description.shape[0])

df_description.style.set_properties(subset=['Description'], **{'width': '1000px'})
drop_list = ['id','member_id','issue_d']

df = df.drop(drop_list,axis=1)
df['emp_title'].value_counts()
#Dropping another column that I deem unnecessary. It contains data in an unorganized way which will not be very helpful for modeling. 

drop_list2 = ['emp_title']

df = df.drop(drop_list2,axis=1)
df.head(1)
#Url is also not much of importance to us. Dropping it. 

drop_list3 = ['url']

df = df.drop(drop_list3,axis=1)
#leaving description field as I will be trying to apply NLP techniques on it for feature engineering. 
#exploring what is there in purpose predictor

df['purpose'].unique()
# recode loan purpose 

df['purpose_n'] = np.nan #Creating new column and filling it with nan values



#filter by debt consolidation, CC and storing them in new columns with a common name "DEBT"

df.loc[(df['purpose'] == 'debt_consolidation')|(df['purpose'] =="credit_card"), 'purpose_n'] = 'debt' 

#filter by home improvement, major purchase, car, house, vacation, renewable energy 

#and storing them in new columns with a common name "major purchases"

df.loc[(df['purpose'] == 'home_improvement')|(df['purpose'] =="major_purchase")|

                 (df['purpose'] == 'car')|(df['purpose'] =="house")|

                 (df['purpose'] == 'vacation')|(df['purpose'] =="renewable_energy"),

                 'purpose_n'] = 'major_purchases' 

#filter by small business, medical, moving, wedding, educational 

#and storing them in new columns with a common name "life events"                 

df.loc[(df['purpose'] == 'small_business')|(df['purpose'] =="medical")|

                 (df['purpose'] == 'moving')|(df['purpose'] =="wedding")|

                 (df['purpose'] == 'educational'),

                 'purpose_n'] = 'life_events'

#the remaining category will remain with the same name 'other' in new columns                 

df.loc[(df['purpose'] == 'other'), 'purpose_n'] = 'other'
df['title'].unique()#.tolist()

#There are too many unique values to get a meaning out of it. Also, it contains more or less similar information as Purpose predictor which is much cleaner. So, we will drop the title field.
#As we have created a new column after recoding the values under purpose, dropping the original purpose predictor. 

#Also, the title predictor contains too many unique values and is more or less similar to purpose predictor which we recoded above

drop_list4 = ['purpose','title']

df = df.drop(drop_list4,axis=1)
print(df['zip_code'].head())

print(df['addr_state'].value_counts().head(5))
#Zip code and State contains similar information. Also, zip code does not even have the entire zip code value and just 3 digits. So, dropping it and keeping state. 

drop_list5 = ['zip_code']

df = df.drop(drop_list5,axis=1)
df.earliest_cr_line.head(3)
import datetime



# calculate time since first credit line

now = datetime.datetime.today() #prints current date

def credit_age (x):

    if x != 'nan': #filter non null

        c1 = datetime.datetime.strptime(x, '%b-%y') #strips the present date in mon-year format 

        #b-Abbreviated month name.	y-Year without century as a zero-padded decimal number.	#Reference: https://www.programiz.com/python-programming/datetime/strftime

        return (now-c1).days/365.25

        #return c1

    else:

        return None



df['earliest_cr_line_n'] = df['earliest_cr_line'].astype(str)

df['earliest_cr_line_n'] = df['earliest_cr_line_n'].apply(credit_age)

df['earliest_cr_line_n'].head()
#Removing the original columns "earliest_6cr_line" because we have transformed it to a new one

drop_list6 = ['earliest_cr_line']

df = df.drop(drop_list6,axis=1)
def null_values(df): #creates a function with below logic

        mis_val = df.isnull().sum() #gives sum of missing values (null values)

        mis_val_percent = 100 * df.isnull().sum() / len(df) #getting percentage of missing values

        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1) #creating a DF containing missing value count and it's percentage

        mis_val_table_ren_columns = mis_val_table.rename( 

        columns = {0 : 'Missing Values', 1 : '% of Total Values'}) #renaming the columns

        mis_val_table_ren_columns = mis_val_table_ren_columns[

            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(

        '% of Total Values', ascending=False).round(1) #getting the percentage of missing values in descending order and rounding it to 1 decimal

        print ("Dataframe has " + str(df.shape[1]) + " columns.\n"      

            "There are " + str(mis_val_table_ren_columns.shape[0]) +

              " columns that have missing values.") #printing number of columns and missing value count

        return mis_val_table_ren_columns
# Missing values statistics

miss_values = null_values(df)

miss_values.head(20)
#First calculate the percentage of missing data for each feature:

missing_features = df.isnull().mean().sort_values(ascending=False)
#Let's visualize the distribution of missing data percentages:

plt.figure(figsize=(6,3), dpi=90)

missing_features.plot.hist(bins=20)

plt.title('Histogram of Missing Features')

plt.xlabel('Fraction of data missing')

plt.ylabel('Feature count')
drop_list = sorted(list(missing_features[missing_features > 0.3].index)) #creating list to store the column names with over 30% missing values

print(drop_list)
#number of features to be dropped?

len(drop_list)
#Dropping these features

df.drop(drop_list, axis=1, inplace=True)
print('Now we are left with {} columns'.format(df.shape[1]))

print('The remaining columns are as follows:')

print(df.columns)
df.head(1)
print(df['pymnt_plan'].unique())

print(df['pymnt_plan'].value_counts())
#As there are mostly 'no' values and very few number of 'yes' values. Dropping it. 

drop_list7 = ['pymnt_plan']

df = df.drop(drop_list7,axis=1)
cor = df.corr() #Checking corelation between features

plt.subplots(figsize=(20,15)) #giving figure size parameters

sns.heatmap(cor, square = True) #plotting heatmap to check corelation
# calcualte mean fico score

df['fico_avg'] = (df['fico_range_high'] + df['fico_range_low'])/2

# calcualte mean last_fico score

df['last_fico_avf'] = (df['last_fico_range_high'] + df['last_fico_range_low'])/2

#Dropping the columns that are now transformed to new columns

drop_list8 = ['fico_range_high','last_fico_range_high','last_fico_range_low','last_fico_range_high']

df = df.drop(drop_list8,axis=1)
#Opps. I missed one column. 

df = df.drop(['fico_range_low'],axis=1)
df.head(1)
df['open_acc'].head()
DROP_LIST = ['funded_amnt', 'funded_amnt_inv', 'pymnt_plan', 'delinq_2yrs', 'inq_last_6mths', 'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'last_pymnt_d', 'last_pymnt_amnt', 'last_credit_pull_d', 'last_fico_range_high', 'last_fico_range_low', 'collections_12_mths_ex_med', 'policy_code', 'acc_now_delinq', 'tot_coll_amt', 'tot_cur_bal', 'total_rev_hi_lim', 'acc_open_past_24mths', 'avg_cur_bal', 'bc_open_to_buy', 'bc_util', 'chargeoff_within_12_mths', 'delinq_amnt', 'mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op', 'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl', 'mths_since_recent_bc', 'mths_since_recent_inq', 'num_accts_ever_120_pd', 'num_actv_bc_tl', 'num_actv_rev_tl', 'num_bc_sats', 'num_bc_tl', 'num_il_tl', 'num_op_rev_tl', 'num_rev_accts', 'num_rev_tl_bal_gt_0', 'num_sats', 'num_tl_120dpd_2m', 'num_tl_30dpd', 'num_tl_90g_dpd_24m', 'num_tl_op_past_12m', 'pct_tl_nvr_dlq', 'percent_bc_gt_75', 'tax_liens', 'tot_hi_cred_lim', 'total_bal_ex_mort', 'total_bc_limit', 'total_il_high_credit_limit', 'hardship_flag', 'disbursement_method', 'debt_settlement_flag']
drop_list9 = ['out_prncp','out_prncp_inv','total_pymnt','total_pymnt_inv','total_rec_prncp','total_rec_int',

        'total_rec_late_fee','recoveries','collection_recovery_fee','last_pymnt_d',

        'last_credit_pull_d']

df = df.drop(drop_list9,axis=1)
drop_list10 = ['inq_last_6mths']

df = df.drop(drop_list10,axis=1)
df.head(1)
print(df['initial_list_status'].head(3))

print(df['initial_list_status'].unique())

print(df['initial_list_status'].count())
df['initial_list_status'].value_counts().plot.bar()

plt.show()
df['policy_code'].value_counts().plot.bar()

plt.show()
print(df.policy_code.value_counts())

print(df.initial_list_status.value_counts())
drop_list11 =['policy_code']

df = df.drop(drop_list11,axis=1)
df['collections_12_mths_ex_med'].value_counts()
df = df.drop(['collections_12_mths_ex_med'],axis=1)
df['application_type'].value_counts()
df.groupby('application_type')['loan_status'].value_counts(normalize=True).loc[:,'Charged Off']
df['acc_now_delinq'].value_counts()
df['acc_now_delinq'].isna().sum()
#Not many missing values and majority of the class contains 0. So, imputing with 0.

df['acc_now_delinq'] = df['acc_now_delinq'].fillna(0)
df.head(1)
#drop_list12 = ['open_acc_6m','open_il_12m','open_il_24m','open_act_il']

#df = df.drop(drop_list12,axis=1)
print(df['tot_coll_amt'].value_counts())

print(df['tot_coll_amt'].describe())

print(df['tot_cur_bal'].value_counts())

print(df['tot_cur_bal'].describe())
#As majority of the class contains 0. So, imputing with 0.

df['tot_coll_amt'] = df['tot_coll_amt'].fillna(0)

df['tot_cur_bal'] = df['tot_cur_bal'].fillna(0)
df.head(1)
def check_stats(col):

  print(df[col].head())

  print(df[col].describe())

  print(df[col].value_counts())
check_stats('total_rev_hi_lim')
#From the dictionary, it is clear that this will not be avaiable at initial state

df = df.drop(['total_rev_hi_lim'],axis=1)
drop_list13 = ['acc_open_past_24mths','avg_cur_bal','bc_open_to_buy','bc_util','mo_sin_old_il_acct','mo_sin_old_rev_tl_op','mo_sin_rcnt_rev_tl_op','mo_sin_rcnt_tl']

df = df.drop(drop_list13,axis=1)
check_stats('chargeoff_within_12_mths')
#As dictionary suggests, this attribute will not be available for investors initially. 

#Also, 'chargeoff_within_12_mths' have almost all values as 0 and will therefore not be very useful for modeling. Removing it.

drop_list14 = ['chargeoff_within_12_mths']

df = df.drop(drop_list14,axis=1)
df.head(1)
new_list_to_check = ['mort_acc',	'mths_since_recent_inq',	'num_accts_ever_120_pd',	'num_actv_bc_tl',	'num_actv_rev_tl',	'num_bc_sats',	'num_bc_tl',	'num_il_tl',	'num_op_rev_tl',	'num_rev_accts',	'num_rev_tl_bal_gt_0',	'num_sats',	'num_tl_120dpd_2m',	'num_tl_30dpd',	'num_tl_90g_dpd_24m',	'num_tl_op_past_12m']

for col in new_list_to_check:

  print(check_stats(col))
df["mort_acc"] = df["mort_acc"].fillna(0) #Imputing with zero as most number of people do not have a mortgage account

#Too many missing values, mostyly biased to one value and not that relevant of a feature. 

drop_list15 = ['mths_since_recent_bc','mths_since_recent_inq','num_accts_ever_120_pd','num_actv_bc_tl','num_actv_rev_tl','num_bc_sats','num_bc_tl','num_il_tl','num_op_rev_tl','num_rev_accts','num_rev_tl_bal_gt_0','num_sats','num_tl_120dpd_2m','num_tl_30dpd','num_tl_90g_dpd_24m','num_tl_op_past_12m']

df = df.drop(drop_list15,axis=1)
df.head(1)
list_next_10_till_debt_settlement = ['pct_tl_nvr_dlq','percent_bc_gt_75',	'pub_rec_bankruptcies',	'tax_liens',	'tot_hi_cred_lim',	'total_bal_ex_mort',	'total_bc_limit',	'total_il_high_credit_limit','hardship_flag',	'debt_settlement_flag']
for col in list_next_10_till_debt_settlement:

  print(check_stats(col))
# majority of the trades never delinquent (can been seen from mean). Therefore, removing pct_tl_nvr_dlq.

# Most of the values are missing and not that relevant of a feature. Therefore, removing them.

drop_list16 = ['pct_tl_nvr_dlq','percent_bc_gt_75','tot_hi_cred_lim','total_bal_ex_mort','total_bc_limit','total_il_high_credit_limit','hardship_flag','debt_settlement_flag']

df.drop(drop_list16, axis=1, inplace=True) 
# The column labeled 'tax_liens' have almost all values as 0 and will therefore not be very useful for modeling. Removing it.

df = df.drop(['tax_liens'],axis=1)
df.head(1)
df.info()
#import seaborn as sns

#from matplotlib import pyplot as plt

cor = df.corr() 

plt.subplots(figsize=(20,15))

sns.heatmap(cor, square = True)
# Missing values statistics

miss_values = null_values(df)

miss_values.head(20)
# Copy Dataframe

complete_df = df.copy()
#What are the value counts for this variable?

complete_df['loan_status'].value_counts()
type(complete_df['loan_status'][0])
#Dependent Variable =  Loan_Status 



#Charged Off = 1

#Default = 1

#Late (31-120 days) = 1

#Does not meet credit policy. Status Charged Off = 1



#Current = 0

#Fully Paid = 0

#In Grace Period = 0

#Late (16-30 days) = 0

#Does not meet credit policy. Status Fully Paid = 0
complete_df['loan_status'] = complete_df['loan_status'].replace("Default", "Charged Off") #renaming the Charged off rows to Charged Off
complete_df['loan_status'] = complete_df['loan_status'].replace("Late (31-120 days)", "Charged Off") #renaming the "Late (31-120 days)" rows to Charged Off
complete_df['loan_status'] = complete_df['loan_status'].replace("Does not meet credit policy. Status Charged Off", "Charged Off") 

#renaming the "Does not meet credit policy. Status Charged Off" rows to Charged Off
complete_df['loan_status'].value_counts(dropna=False)
complete_df['loan_status'] = ['Charged Off' if i=='Charged Off' else 'Fully Paid' for i in complete_df['loan_status']]
complete_df['loan_status'].value_counts(dropna=False)
complete_df['loan_status'].value_counts(normalize=True, dropna=False)
fig, axs = plt.subplots(1,2,figsize=(14,7))

sns.countplot(x='loan_status',data=complete_df,ax=axs[0])

axs[0].set_title("Frequency of each Loan Status")

complete_df.loan_status.value_counts().plot(x=None,y=None, kind='pie', ax=axs[1],autopct='%1.2f%%')

axs[1].set_title("Percentage of each Loan status")

plt.show()
#fig, axs = plt.subplots(1,2,figsize=(14,7))

#sns.countplot(x='TARGET',data=complete_df,ax=axs[0])

#axs[0].set_title("Frequency of each Loan Status")

#complete_df.TARGET.value_counts().plot(x=None,y=None, kind='pie', ax=axs[1],autopct='%1.2f%%')

#axs[1].set_title("Percentage of each Loan status")

#plt.show()
#removing the loan_status predicor as we have recoded the values from it to a new column ("Target") which is out depenednt variable

#complete_df = complete_df.drop('loan_status',axis=1,inplace=True) 

#complete_df = complete_df.drop(['loan_status'],axis=1)
complete_df.head(1)
#Print the remaining predictos for future reference:

print(list(complete_df.columns))
complete_df['loan_amnt'].describe()
complete_df.groupby('loan_status')['loan_amnt'].describe()
complete_df['term'].value_counts(dropna=False)
complete_df['term'] = complete_df['term'].apply(lambda x: np.int8(x.split()[0]))
complete_df['term'].value_counts(normalize=True)
#Compare the charge-off rate by loan period:

complete_df.groupby('term')['loan_status'].value_counts(normalize=True).loc[:,'Charged Off']
complete_df['installment'].describe()
complete_df.groupby('loan_status')['installment'].describe()
complete_df['emp_length'].head(3)
complete_df['emp_length'].fillna(value=0,inplace=True) #filling the missing values with 0 

complete_df['emp_length'].replace(to_replace='[^0-9]+', value='', inplace=True, regex=True ) #checking not numeric value and then replacng it with '' to removing string

complete_df['emp_length'].value_counts().sort_values().plot(kind='barh',figsize=(18,8)) #plotting the bar to see the emp_length

plt.title('Number of loans distributed by Employment Years',fontsize=20) #plotting the title

plt.xlabel('Number of loans',fontsize=15) #plotting number of loans for x axis

plt.ylabel('Years worked',fontsize=15); #plotting years worked for y axis
complete_df['home_ownership'].value_counts(dropna=False)
#any and none are not very relevant for modeling. Therefore, clubbing them with Other. 

complete_df['home_ownership'].replace(['NONE', 'ANY'], 'OTHER', inplace=True)
complete_df['home_ownership'].value_counts(dropna=False)
complete_df.groupby('home_ownership')['loan_status'].value_counts(normalize=True).loc[:,'Charged Off']
complete_df['annual_inc'].describe()
complete_df['log_annual_inc'] = complete_df['annual_inc'].apply(lambda x: np.log10(x+1))

complete_df.drop('annual_inc', axis=1, inplace=True)
complete_df['log_annual_inc'].describe()
complete_df.groupby('loan_status')['log_annual_inc'].describe()
complete_df['verification_status'].value_counts()
complete_df['addr_state'].unique() #seeing the unique values of address state column
# Make a list with each of the regions by state.



west = ['CA', 'OR', 'UT','WA', 'CO', 'NV', 'AK', 'MT', 'HI', 'WY', 'ID'] #all western states

south_west = ['AZ', 'TX', 'NM', 'OK'] #all south western states

south_east = ['GA', 'NC', 'VA', 'FL', 'KY', 'SC', 'LA', 'AL', 'WV', 'DC', 'AR', 'DE', 'MS', 'TN' ] #all south eastern states

mid_west = ['IL', 'MO', 'MN', 'OH', 'WI', 'KS', 'MI', 'SD', 'IA', 'NE', 'IN', 'ND'] #all mid western states

north_east = ['CT', 'NY', 'PA', 'NJ', 'RI','MA', 'MD', 'VT', 'NH', 'ME'] #all north eastern states
complete_df['region'] = np.nan #creating new column region with all nan values

def finding_regions(state): #creating fuction to recode states into region

    if state in west:

        return 'West'

    elif state in south_west:

        return 'SouthWest'

    elif state in south_east:

        return 'SouthEast'

    elif state in mid_west:

        return 'MidWest'

    elif state in north_east:

        return 'NorthEast'

    

complete_df['region'] = complete_df['addr_state'].apply(finding_regions) #apply function to the new column
#Calculate the charge-off rates by region:

complete_df.groupby('region')['loan_status'].value_counts(normalize=True).loc[:,'Charged Off'].sort_values()
#Calculate the charge-off rates by state:

complete_df.groupby('addr_state')['loan_status'].value_counts(normalize=True).loc[:,'Charged Off'].sort_values()
complete_df['dti'].describe()
plt.figure(figsize=(8,3), dpi=90)

sns.distplot(complete_df.loc[complete_df['dti'].notnull() & (complete_df['dti']<60), 'dti'], kde=False)

plt.xlabel('Debt-to-income Ratio')

plt.ylabel('Count')

plt.title('Debt-to-income Ratio')
complete_df.groupby('loan_status')['dti'].describe()
complete_df.groupby('loan_status')['delinq_2yrs'].describe()
complete_df = complete_df.drop(['delinq_2yrs'],axis=1)
complete_df.columns
plt.figure(figsize=(10,3), dpi=90)

sns.countplot(complete_df['open_acc'], order=sorted(complete_df['open_acc'].unique()), color='#5975A4', saturation=1)

_, _ = plt.xticks(np.arange(0, 90, 5), np.arange(0, 90, 5))

plt.title('Number of Open Credit Lines')
#let's see the difference in number of credit lines between fully paid loans and charged-off loans

complete_df.groupby('loan_status')['open_acc'].describe()
complete_df['pub_rec'].value_counts().sort_index()
complete_df.groupby('loan_status')['pub_rec'].describe()
complete_df['revol_bal'].describe()
#doing log transform

complete_df['revol_bal'] = complete_df['revol_bal'].apply(lambda x: np.log10(x+1))
complete_df.groupby('loan_status')['revol_bal'].describe()
complete_df['revol_util'].head()
#We see that term, emp_length, revol_util columns contains numeric values, but is formatted as object. 

complete_df['revol_util'] = complete_df['revol_util'].str.rstrip('%').astype('float') #stripping the % symbol and converting the type to "Float"
complete_df['revol_util'].describe()
complete_df.groupby('loan_status')['revol_util'].describe()
plt.figure(figsize=(12,3), dpi=90)

sns.countplot(complete_df['total_acc'], order=sorted(complete_df['total_acc'].unique()), color='#5975A4', saturation=1)

_, _ = plt.xticks(np.arange(0, 176, 10), np.arange(0, 176, 10))

plt.title('Total Number of Credit Lines')
complete_df.groupby('loan_status')['total_acc'].describe()
#not really sure what W, F means here. 
complete_df.columns
complete_df['application_type'].value_counts()
complete_df.groupby('application_type')['loan_status'].value_counts(normalize=True).loc[:,'Charged Off']
complete_df['acc_now_delinq'].head()
complete_df['acc_now_delinq'].value_counts()
complete_df.groupby('acc_now_delinq')['loan_status'].value_counts(normalize=True).loc[:,'Charged Off']
complete_df = complete_df.drop(['application_type'],axis=1)
complete_df = complete_df.drop(['acc_now_delinq'],axis=1)
complete_df['tot_coll_amt'].head()
complete_df['tot_coll_amt'].value_counts()
complete_df['tot_cur_bal'].value_counts()
complete_df['mort_acc'].describe()
complete_df['mort_acc'].value_counts().head(10)
#comparing statistics by target variable:

complete_df.groupby('loan_status')['mort_acc'].describe()
complete_df['pub_rec_bankruptcies'].value_counts().sort_index()
#comparing statistics by target variable:

complete_df.groupby('loan_status')['pub_rec_bankruptcies'].describe()
complete_df['purpose_n'].unique()
complete_df['earliest_cr_line_n'].head()
complete_df['earliest_cr_line_n'].describe()
#comparing statistics by target variable:

complete_df.groupby('loan_status')['earliest_cr_line_n'].describe()
complete_df.groupby('loan_status')['fico_avg'].describe()
complete_df = complete_df.drop(['last_fico_avf'],axis=1)
complete_df['log_annual_inc'].describe()
complete_df.groupby('loan_status')['log_annual_inc'].describe()
complete_df = complete_df.drop(['region'],axis=1)
complete_df.columns
obj_cols = complete_df.columns[complete_df.dtypes==object]



#Imputer function

imputer = lambda x:x.fillna(x.value_counts().index[0]) 



#Impute dtype=object with most frequent value

complete_df[obj_cols] = complete_df[obj_cols].apply(imputer) 



#Impute the rest of df with median

complete_df = complete_df.fillna(df.median(axis=0)) 
missing_fractions = complete_df.isnull().mean().sort_values(ascending=False) # Fraction of data missing for each variable

print(missing_fractions[missing_fractions > 0]) # Print variables that are missing data
#print(complete_df.isnull().sum())

complete_df.fillna(complete_df.median(), inplace=True)

complete_df.head(1)
complete_df.to_csv("/content/drive/My Drive/Lending_Club/clean_df_23_col.csv",index=False)
#complete_df.to_csv('clean_df_23_col_.csv') #save csv to my drive by the name lending_club_cleaned1

#!cp lending_club_cleaned1.csv "/content/drive/My Drive/Lending_Club/"
#import pickle

#complete_df.to_pickle("clean_final_23_col.pkl")

#df_filtered = pd.read_pickle("/content/drive/My Drive/lending_loan_df.pkl")
#complete_df = pd.read_csv("/content/drive/My Drive/lending_club_cleaned1.csv")
corr = complete_df.corr()['TARGET'].sort_values()

# Display correlations

print('Most Positive Correlations:\n', corr.tail(10))

print('\nMost Negative Correlations:\n', corr.head(10))
complete_df = complete_df.drop(['addr_state'],axis=1)
target_list = [1 if i=='Charged Off' else 0 for i in complete_df['loan_status']]

complete_df['charged_off'] = target_list

complete_df['charged_off'].value_counts()

complete_df.drop('loan_status', axis=1, inplace=True)
#how many variable we have now

print(complete_df.shape)

complete_df.head(1)
null_counts = complete_df.isnull().sum().sort_index()

print("Number of null values in each column:\n{}".format(null_counts))
print("Data types and their frequency\n{}".format(complete_df.dtypes.value_counts()))
object_columns_df = complete_df.select_dtypes(include=['object'])

print(object_columns_df.iloc[0])
cols = ['emp_length','home_ownership','verification_status', 'purpose_n', 'initial_list_status']

for name in cols:

    print(name,':')

    print(object_columns_df[name].value_counts(),'\n')
#converting the type of emp_length column to int64 from string

complete_df['emp_length'] = complete_df['emp_length'].astype('int64') 
object_columns_df = complete_df.select_dtypes(include=['object'])

print(object_columns_df.iloc[0])
#Converting nominal features into numerical features requires encoding them as dummy variables.

nominal_columns = ["home_ownership", "verification_status", "purpose_n", "initial_list_status"]

dummy_df = pd.get_dummies(complete_df[nominal_columns], drop_first=True) #greating dummies for the above nominal columns and removing first dummy variable to 

#drop the first one to avoid linear dependency between the resulted features since some algorithms may struggle with this issue.

complete_df = pd.concat([complete_df, dummy_df], axis=1) #merging the newly created dummy columns with the working dataset

complete_df = complete_df.drop(nominal_columns, axis=1) #dropping the original nominal columns as they are not required anymore



#df = pd.get_dummies(df, columns=["purpose"], drop_first=True)
complete_df.info()
#let's see how many predictors are there now

complete_df.shape
#let's see the dataframe now with dummy variables:

complete_df.head()
#complete_df['region'].unique()
#complete_df['region'].value_counts()
#Converting region into dummy variable too.

#nominal_columns2 = ["region"]

#dummy_df2 = pd.get_dummies(complete_df[nominal_columns2], drop_first=True) #greating dummies for the above nominal columns and removing first dummy variable to 

#avoid linear dependency between the resulted features since some algorithms may struggle with this issue.

#complete_df = pd.concat([complete_df, dummy_df2], axis=1) #merging the newly created dummy columns with the working dataset

#complete_df = complete_df.drop(nominal_columns2, axis=1) #dropping the original nominal columns as they are not required anymore



#df = pd.get_dummies(df, columns=["purpose"], drop_first=True)
complete_df.to_pickle("/content/drive/My Drive/Lending_Club/loan_clean_final.pkl")
DATA_PATH = "/content/drive/My Drive/Capstone Project - NLP"

infile = open(DATA_PATH+'/lending_loan_df_clean_complete.pkl','rb')

complete_df = pickle.load(infile)
#complete_df['desc2'] = complete_df['desc1'].str.len() #creating a column with length of the cleaned description
print(f"{complete_df.dtypes}\n")
var_cor = pd.DataFrame((complete_df.corr()['TARGET'])).reset_index()

var_cor.columns=['index', 'correlation']

var_cor
# Copy Dataframe

new_df = complete_df.copy()

new_df.head(1)
new_df = new_df.drop(['Unnamed: 0'],axis=1)
import matplotlib.pyplot as plt

import seaborn as sb

df2 = new_df

plt.figure()

ax = sb.countplot(x=new_df["TARGET"], y = None, palette = "Reds")

for p in ax.patches:

    x=p.get_bbox().get_points()[:,0]

    y=p.get_bbox().get_points()[1,1]

    ax.annotate('{:.0f}'.format(y), (x.mean(), y), ha='center', va='bottom') 

plt.title('Loan Status (Target Variable)')

plt.show()
from sklearn.utils import shuffle, resample



# Let's seperate our data into two based on the Score (True or False). 

df_zero = df2[df2["TARGET"] == 0]

df_one = df2[df2["TARGET"] == 1]



print("Number of records before upsampling: ")

print("One:", len(df_one), "Zero:", len(df_zero))



# Let's use the resample function for upsampling.

df_one = resample(df_one, replace=True, n_samples=len(df_zero))



# Let's put the separated data frames together. 

df2 = pd.concat([df_zero, df_one], axis=0)



# Let's shuffle the data

df2 = shuffle(df2)



print("Number of records after upsampling: ")

print("One:", len(df2[df2["TARGET"] == 0]), "Zero:", len(df2[df2["TARGET"] == 1]))
import matplotlib.pyplot as plt

import seaborn as sb



plt.figure()

ax = sb.countplot(x=df2["TARGET"], y = None, palette = "Reds")

for p in ax.patches:

    x=p.get_bbox().get_points()[:,0]

    y=p.get_bbox().get_points()[1,1]

    ax.annotate('{:.0f}'.format(y), (x.mean(), y), ha='center', va='bottom') 

plt.title('Loan Status (Target Variable) after upsampling the 1 class')

plt.show()
#df2 = df2.drop(['desc'],axis=1)
df2.to_pickle("/content/drive/My Drive/Lending_Club/loan_ready_for_ml.pkl")
#Reading pickled df

import pickle

#drive.mount('/content/drive')

DATA_PATH = "/content/drive/My Drive/Capstone Project - NLP"

infile = open(DATA_PATH+'/loan_dfready_for_model.pkl','rb')

df2 = picklenull_counts = df2.isnull().sum().sort_index()

print("Number of null values in each column:\n{}".format(null_counts)).load(infile)
null_counts = df2.isnull().sum().sort_index()

print("Number of null values in each column:\n{}".format(null_counts))
df2['desc'].head()
#Impute the rest of df with median

#df2 = df2.fillna(df2.median(axis=0)) 
snow = SnowballStemmer('english') #Initializing snowball stemmer from NLTK library
stop_words = stopwords.words("english")

def process_text(texts): 

    final_text_list=[]

    for sent in texts:

        filtered_sentence=[]

        

        sent = sent.lower() # Lowercase 

        sent = sent.strip() # Remove leading/trailing whitespace

        sent = re.sub('\s+', ' ', sent) # Remove extra space and tabs

        sent = re.compile('<.*?>').sub('', sent) # Remove HTML tags/markups:

        

        for w in word_tokenize(sent):

            # We are applying some custom filtering here.

            # Check if it is not numeric and its length>2 and not in stop words

            if(not w.isnumeric()) and (len(w)>2) and (w not in stop_words):  

                # Stem and add to filtered list

                filtered_sentence.append(snow.stem(w))

        final_string = " ".join(filtered_sentence) #final string of cleaned words

 

        final_text_list.append(final_string)

    

    return final_text_list
print("Pre-processing the training text field")

df2["desc"] = process_text(df2["desc"].tolist()) #applying above function on desc column
df2.columns
X_train, X_val, y_train, y_val = train_test_split(df2.drop(["TARGET"], axis=1), # Input

                                                  df2['TARGET'].tolist(), # Target field

                                                  test_size=0.2, # 20% val, 80% tranining

                                                  shuffle=True) # Shuffle the whole dataset
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

 

# Initialize the binary count vectorizer

tfidf_vectorizer = CountVectorizer(binary=True,

                                   max_features=50    # Limit the vocabulary size

                                  )

# Fit and transform

X_train_text_vectors = tfidf_vectorizer.fit_transform(X_train["desc"].tolist())

# Only transform

X_val_text_vectors = tfidf_vectorizer.transform(X_val["desc"].tolist())
print(tfidf_vectorizer.vocabulary_)
X_train = X_train.drop(['desc'],axis=1) #dropping desc from X_train as we have processed it in X_train_text_vectors

X_val = X_val.drop(['desc'],axis=1) #dropping desc from X_val as well as we have processed it in X_val_text_vectors
# Let' merge our features

X_train_features = np.column_stack((X_train_text_vectors.toarray(), 

                                    X_train)

                                  )

# Let' merge our features

X_val_features = np.column_stack((X_val_text_vectors.toarray(), 

                                    X_val)

                                  )
#importing necessary libraries for logistic regression

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.metrics import make_scorer, accuracy_score, f1_score



lrClassifier = LogisticRegression()

lrClassifier.fit(X_train_features, y_train)

predicted = lrClassifier.predict(X_val_features)

print("LogisticRegression on Validation: Accuracy Score: %f, F1-score: %f" % 

      (accuracy_score(y_val, predicted), f1_score(y_val, predicted)))
# generate evaluation metrics

probs = lrClassifier.predict_proba(X_val_features)

predicted = lrClassifier.predict(X_val_features)

accuracy = accuracy_score(y_val, predicted)

auc = metrics.roc_auc_score(y_val, probs[:, 1])



print(f'AUC estimate: {auc:.3}')

print(f'Mean accuracy score: {accuracy:.3}')

print(metrics.confusion_matrix(y_val, predicted))

print(metrics.classification_report(y_val, predicted))
lrClassifier = LogisticRegression(penalty = 'l2', #calling the object with l2 NORM penalty and alpha value of 0.1

                                  C = 0.1)

lrClassifier.fit(X_train_features, y_train) #fitting the logistic regression on the training part of the data

predicted = lrClassifier.predict(X_val_features) #predicting on validation set



print("LogisticRegression on Validation: Accuracy Score: %f, F1-score: %f" %  

      (accuracy_score(y_val, predicted), f1_score(y_val, predicted)))#printing the accuracy and f1 score 
# generate evaluation metrics

probs = lrClassifier.predict_proba(X_val_features)

predicted = lrClassifier.predict(X_val_features)

accuracy = accuracy_score(y_val, predicted)

auc = metrics.roc_auc_score(y_val, probs[:, 1])



print(f'AUC estimate: {auc:.3}')

print(f'Mean accuracy score: {accuracy:.3}')

print(metrics.confusion_matrix(y_val, predicted))

print(metrics.classification_report(y_val, predicted))
lrClassifier.predict(X_val_features)[0:5] #checking first 5 predicted value via usual binary format
lrClassifier.predict_proba(X_val_features)[0:5] #checking first 5 predicted values via probablity format
%matplotlib inline 

import numpy as np #importing numpy library

import matplotlib.pyplot as plt #importing matplotlib library for plotting the results



# Calculate the accuracy using different values for the classification threshold, 

# and pick the threshold that resulted in the highest accuracy.

highest_accuracy = 0 #initializing highest_accuracy variable

threshold_highest_accuracy = 0 #initializing threshold_highest_accuracy variable



thresholds = np.arange(0,1,0.01) #

scores = []

for t in thresholds:

    # set threshold to 't' instead of 0.5

    y_val_other = (lrClassifier.predict_proba(X_val_features)[:,1] >= t).astype(float)

    score = accuracy_score(y_val, y_val_other)

    scores.append(score)

    if(score > highest_accuracy):

        highest_accuracy = score

        threshold_highest_accuracy = t

print("Highest Accuracy on Validation:", highest_accuracy, \

      ", Threshold for the highest Accuracy:", threshold_highest_accuracy)   



# Let's plot the accuracy versus different choices of thresholds

plt.plot([0.5, 0.5], [np.min(scores), np.max(scores)], linestyle='--')

plt.plot(thresholds, scores, marker='.')

plt.title('Accuracy versus different choices of thresholds')

plt.xlabel('Threshold')

plt.ylabel('Accuracy')

plt.show()
%matplotlib inline 

import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve



# Calculate the precision and recall using different values for the classification threshold

val_predictions_probs = lrClassifier.predict_proba(X_val_features)

precisions, recalls, thresholds = precision_recall_curve(y_val, val_predictions_probs[:, 1])
%matplotlib inline 

import numpy as np

import matplotlib.pyplot as plt



# Calculate the F1 score using different values for the classification threshold, 

# and pick the threshold that resulted in the highest F1 score.

highest_f1 = 0

threshold_highest_f1 = 0



f1_scores = []

for id, threhold in enumerate(thresholds):

    f1_score = 2*precisions[id]*recalls[id]/(precisions[id]+recalls[id])

    f1_scores.append(f1_score)

    if(f1_score > highest_f1):

        highest_f1 = f1_score

        threshold_highest_f1 = threhold

print("Highest F1 score on Validation:", highest_f1, \

      ", Threshold for the highest F1 score:", threshold_highest_f1)



# Let's plot the F1 score versus different choices of thresholds

plt.plot([0.5, 0.5], [np.min(f1_scores), np.max(f1_scores)], linestyle='--')

plt.plot(thresholds, f1_scores, marker='.')

plt.title('F1 Score versus different choices of thresholds')

plt.xlabel('Threshold')

plt.ylabel('F1 Score')

plt.show()
df2.select_dtypes([np.number]).info()
#def min_max(col): #creating a function to apply min max scaling on a column

 # df2[col] = (df2[col] - df2[col].min())/(df2[col].max()-df2[col].min())

  

#col_to_scale = df2.columns #creating a list of columns in our working dataframe

#for col in col_to_scale: #running for loop to apply min_max function on all the columns mentioned in the above list

 # min_max(col)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train_features = sc.fit_transform(X_train_features)

X_val_features = sc.transform(X_val_features)
#X_train, X_val, y_train, y_val = train_test_split(df2.drop("charged_off", axis=1), # Input

#                                                  df2['charged_off'].tolist(), # Target field

##                                                  test_size=0.2, # 20% val, 80% tranining

 #                                                 shuffle=True) # Shuffle the whole dataset
%%time

lr = LogisticRegression()

lr.fit(X_train_features, y_train)
# generate evaluation metrics



probs = lr.predict_proba(X_val_features)

predicted = lr.predict(X_val_features)

accuracy = accuracy_score(y_val, predicted)

auc = metrics.roc_auc_score(y_val, probs[:, 1])



print(f'AUC estimate: {auc:.3}')

print(f'Mean accuracy score: {accuracy:.3}')

print(metrics.confusion_matrix(y_val, predicted))

print(metrics.classification_report(y_val, predicted))
# ROC curve

fpr, tpr, thresholds = roc_curve(y_val, lr.predict_proba(X_val_features)[:,1])

plt.figure(figsize=(6,6))

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.show()
from sklearn.linear_model import SGDClassifier

from sklearn.pipeline import Pipeline

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.model_selection import GridSearchCV



#The machine learning pipeline:

pipeline_sgdlogreg = Pipeline([

    #('imputer', Imputer(copy=False)), # Mean imputation by default

    ('scaler', StandardScaler(copy=False)),

    ('model', SGDClassifier(loss='log', max_iter=1000, tol=1e-3, random_state=1, warm_start=True))

])

#A small grid of hyperparameters to search over:



param_grid_sgdlogreg = {

    'model__alpha': [10**-5, 10**-2, 10**1],

    'model__penalty': ['l1', 'l2']

}

#Create the search grid object:

grid_sgdlogreg = GridSearchCV(estimator=pipeline_sgdlogreg, param_grid=param_grid_sgdlogreg, scoring='roc_auc', n_jobs=-1, pre_dispatch='2*n_jobs', cv=5, verbose=1, return_train_score=False)

#Conduct the grid search and train the final model on the whole dataset:



grid_sgdlogreg.fit(X_train_features, y_train)
#Mean cross-validated AUROC score of the best model:



grid_sgdlogreg.best_score_
#Best hyperparameters:



grid_sgdlogreg.best_params_
# ROC curve

fpr, tpr, thresholds = roc_curve(y_val, grid_sgdlogreg.predict_proba(X_val_features)[:,1])

plt.figure(figsize=(6,6))

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.show()
pipeline_knn = Pipeline([

    ('scaler', StandardScaler(copy=False)),

    ('lda', LinearDiscriminantAnalysis()),

    ('model', KNeighborsClassifier(n_jobs=-1))

])
param_grid_knn = {

    'lda__n_components': [3, 9], # Number of LDA components to keep

    'model__n_neighbors': [5, 25, 125] # The 'k' in k-nearest neighbors

}
grid_knn = GridSearchCV(estimator=pipeline_knn, param_grid=param_grid_knn, scoring='roc_auc', n_jobs=-1, pre_dispatch='2*n_jobs', cv=5, verbose=1, return_train_score=False)
grid_knn.fit(X_train_features, y_train)
#Mean cross-validated AUROC score of the best model:

grid_knn.best_score_
#Best hyperparameters:

grid_knn.best_params_
#K_values = [3, 5, 10, 20, 30]

 

#for K in K_values:

#    knnClassifier = KNeighborsClassifier(n_neighbors=K)

#    knnClassifier.fit(X_train, y_train)

#    val_predictions = knnClassifier.predict(X_val)

#    print("F1 Score for K:", K, "is", f1_score(y_val, val_predictions))
#K_values = [3, 5, 10, 20, 30]

 

#for K in K_values:

#knnClassifier = KNeighborsClassifier(n_neighbors=3)

#knnClassifier.fit(X_train, y_train)



#print("F1 Score for K=3 is", f1_score(y_val, predicted))
probs = grid_knn.predict_proba(X_val_features)

predicted = grid_knn.predict(X_val_features)

#print("knnClassifier on Validation: Accuracy Score: %f, F1-score: %f" % 

      #(accuracy_score(y_val, predicted), f1_score(y_val, predicted)))

accuracy = accuracy_score(y_val, predicted)

auc = metrics.roc_auc_score(y_val, probs[:, 1])



# generate evaluation metrics

print(f'AUC estimate: {auc:.3}')

print(f'Mean accuracy score: {accuracy:.3}')

print(metrics.confusion_matrix(y_val, predicted))

print(metrics.classification_report(y_val, predicted))
# ROC curve



fpr, tpr, thresholds = roc_curve(y_val, grid_knn.predict_proba(X_val_features)[:,1])

plt.figure(figsize=(6,6))

plt.plot(fpr, tpr, label='KNN (area = %0.2f)' % auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.show()
#model_save_name = 'knn_Classifier.pt'

#path = F"/content/gdrive/My Drive/{model_save_name}" 

#torch.save(knnClassifier, path)
%%time

param_grid={'max_depth': [5, 10, 20],

            'min_samples_leaf': [5, 10, 15],

            'min_samples_split': [2, 5, 15, 25] 

           }



dt = DecisionTreeClassifier()



grid_search = GridSearchCV(dt, # Base model

                           param_grid, # Parameters to try

                           cv = 5, # Apply 5-fold cross validation

                           verbose = 1, # Print summary

                           n_jobs = -1 # Use all available processors 

                          )



grid_search.fit(X_train_features, y_train)
grid_search.best_params_ #{'max_depth': 50, 'min_samples_leaf': 5, 'min_samples_split': 5}
# Let's get the input and output data for testing the classifier

predicted = grid_search.predict(X_val_features)
print(metrics.confusion_matrix(y_val, predicted))
print(metrics.classification_report(y_val, predicted))

print("Accuracy:", accuracy_score(y_val, predicted))
grid_search.best_estimator_
#model_save_name = 'Decision_tree.pt'

#path = F"/content/gdrive/My Drive/{model_save_name}" 

#torch.save(dt, path)
#Next we train a random forest model. Note that data standardization is not necessary for a random forest.



pipeline_rfc = Pipeline([

    #('imputer', Imputer(copy=False)),

    ('model', RandomForestClassifier(n_jobs=-1, random_state=1))

])
#The random forest takes very long to train, so we don't test different hyperparameter choices. We'll still use GridSearchCV for the sake of consistency.



param_grid_rfc = {

    'model__n_estimators': [50] # The number of randomized trees to build

}
#The AUROC will always improve (with decreasing gains) as the number of estimators increases, but it's not necessarily worth the extra training time and model complexity.



grid_rfc = GridSearchCV(estimator=pipeline_rfc, param_grid=param_grid_rfc, scoring='roc_auc', n_jobs=-1, pre_dispatch='2*n_jobs', cv=5, verbose=1, return_train_score=False)
grid_rfc.fit(X_train_features, y_train)
#Mean cross-validated AUROC score of the random forest:



grid_rfc.best_score_
%%time

rf = RandomForestClassifier(n_estimators=50, max_depth=3, max_features='log2', oob_score=True,  random_state=0)

rf.fit(X_train_features, y_train)
# generate evaluation metrics



probs = rf.predict_proba(X_val_features)

predicted = rf.predict(X_val_features)

accuracy = accuracy_score(y_val, predicted)

auc = metrics.roc_auc_score(y_val, probs[:, 1])



print(f'AUC estimate: {auc:.3}')

print(f'Out-of-bag score estimate: {rf.oob_score_:.3}')

print(f'Mean accuracy score: {accuracy:.3}')

print(metrics.confusion_matrix(y_val, predicted))

print(metrics.classification_report(y_val, predicted))
# ROC curve



fpr, tpr, thresholds = roc_curve(y_val, rf.predict_proba(X_val_features)[:,1])

plt.figure(figsize=(6,6))

plt.plot(fpr, tpr, label='Random forest (area = %0.2f)' % auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.show()
#model_save_name = 'Random_forest.pt'

#path = F"/content/gdrive/My Drive/{model_save_name}" 

#torch.save(rf, path)
%%time



nn = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)

nn.fit(X_train_features, y_train)
probs = nn.predict_proba(X_val_features)

predicted = nn.predict(X_val_features)

accuracy = accuracy_score(y_val, predicted)

auc = metrics.roc_auc_score(y_val, probs[:, 1])



# generate evaluation metrics

print(f'AUC estimate: {auc:.3}')

print(f'Mean accuracy score: {accuracy:.3}')

print(metrics.confusion_matrix(y_val, predicted))

print(metrics.classification_report(y_val, predicted))
# ROC curve



fpr, tpr, thresholds = roc_curve(y_val, nn.predict_proba(X_val_features)[:,1])

plt.figure(figsize=(6,6))

plt.plot(fpr, tpr, label='Neural network (area = %0.2f)' % auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.show()
#model_save_name = 'Neural_Network.pt'

#path = F"/content/gdrive/My Drive/{model_save_name}" 

#torch.save(nn, path)
%%time

gnb = GaussianNB()

gnb.fit(X_train_features, y_train)
probs = gnb.predict_proba(X_val_features)

predicted = gnb.predict(X_val_features)

accuracy = accuracy_score(y_val, predicted)

auc = metrics.roc_auc_score(y_val, probs[:, 1])



# generate evaluation metrics

print(f'AUC estimate: {auc:.3}')

print(f'Mean accuracy score: {accuracy:.3}')

print(metrics.confusion_matrix(y_val, predicted))

print(metrics.classification_report(y_val, predicted))
# ROC curve



fpr, tpr, thresholds = roc_curve(y_val, gnb.predict_proba(X_val_features)[:,1])

plt.figure(figsize=(6,6))

plt.plot(fpr, tpr, label='Naive Bayes (area = %0.2f)' % auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.show()
#model_save_name = 'NaiyeBayes.pt'

#path = F"/content/gdrive/My Drive/{model_save_name}" 

#torch.save(gnb, path)
%%time

lda = LinearDiscriminantAnalysis()

lda.fit(X_train_features, y_train)
probs = lda.predict_proba(X_val_features)

predicted = lda.predict(X_val_features)

accuracy = accuracy_score(y_val, predicted)

auc = metrics.roc_auc_score(y_val, probs[:, 1])



# generate evaluation metrics

print(f'AUC estimate: {auc:.3}')

print(f'Mean accuracy score: {accuracy:.3}')

print(metrics.confusion_matrix(y_val, predicted))

print(metrics.classification_report(y_val, predicted))
# ROC curve



fpr, tpr, thresholds = roc_curve(y_val, lda.predict_proba(X_val_features)[:,1])

plt.figure(figsize=(6,6))

plt.plot(fpr, tpr, label='Linear Discriminant Analysis (area = %0.2f)' % auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.show()
#model_save_name = 'LDA.pt'

#path = F"/content/gdrive/My Drive/{model_save_name}" 

#torch.save(lda, path)
%%time

gradboost = GradientBoostingClassifier(n_estimators=200,max_depth=3)

gradboost.fit(X_train_features, y_train)
probs = gradboost.predict_proba(X_val_features)

predicted = gradboost.predict(X_val_features)

accuracy = accuracy_score(y_val, predicted)

auc = metrics.roc_auc_score(y_val, probs[:, 1])



# generate evaluation metrics

print(f'AUC estimate: {auc:.3}')

print(f'Mean accuracy score: {accuracy:.3}')

print(metrics.confusion_matrix(y_val, predicted))

print(metrics.classification_report(y_val, predicted))
# ROC curve



fpr, tpr, thresholds = roc_curve(y_val, gradboost.predict_proba(X_val_features)[:,1])

plt.figure(figsize=(6,6))

plt.plot(fpr, tpr, label='Gradient boosting (area = %0.2f)' % auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.show()
gradboost.feature_importances_
from sklearn.ensemble import BaggingClassifier

from sklearn.tree import DecisionTreeClassifier
clf=BaggingClassifier(oob_score=True,n_jobs=-1,n_estimators=20,random_state=400,

                      base_estimator=DecisionTreeClassifier())
clf.fit(X_train_features,y_train)
clf.oob_score_
clf.score(X_val_features,y_val)
#for w in range(10,300,20):

#    clf=BaggingClassifier(oob_score=True,n_jobs=-1,n_estimators=w,random_state=400,

#                          base_estimator=DecisionTreeClassifier())

#    clf.fit(X_train_features,y_train)

##    oob=clf.oob_score_

 #   print('For n_estimators = '+str(w))

#    print('OOB score is '+str(oob))

#    print('************************')
# Feature Importance

clf.estimators_
print(clf.estimators_[0]) #first tree model
print(clf.estimators_[0].feature_importances_)
# We can extract feature importance from each tree then take a mean for all trees

import numpy as np

imp=[]

for i in clf.estimators_:

    imp.append(i.feature_importances_)

imp=np.mean(imp,axis=0)

imp
feature_importance=pd.Series(imp,index=X_train_features.tolist())
feature_importance.sort_values(ascending=False)
feature_importance.sort_values(ascending=False).plot(kind='bar')
#model_save_name = 'gradient_boosting.pt'

#path = F"/content/gdrive/My Drive/{model_save_name}" 

#torch.save(gradboost, path)
#!pip install mxnet
#svm = SVC(kernel='linear', C=1E10)

#svm.fit(X_train, y_train)
#probs = svm.predict_proba(X_val)

#predicted = svm.predict(X_val)

#accuracy = accuracy_score(y_val, predicted)

#auc = metrics.roc_auc_score(y_val, probs[:, 1])



# generate evaluation metrics

#print(f'AUC estimate: {auc:.3}')

#print(f'Mean accuracy score: {accuracy:.3}')

#print(metrics.confusion_matrix(y_val, predicted))

#print(metrics.classification_report(y_val, predicted))
from keras.models import Sequential

from keras.layers import Dense

np.random.seed(777)
X_train_features.shape
# create model

model = Sequential()

model.add(Dense(12, input_dim=93, activation='relu'))

model.add(Dense(93, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

# Compile model

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model

model.fit(X_train_features, y_train, epochs=5, batch_size=10)

# evaluate the model

scores = model.evaluate(X_val_features, y_val)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))



predictions = model.predict(X_val_features)

y_rounded = [round(x[0]) for x in predictions]

scores_test = model.evaluate(X_val_features, y_val)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores_test[1]*100))



accuracy_score(y_val, y_rounded)
# Phase 2: Making the Neural Network (NN)



# Importing the Keras libraries and packages

import keras

from keras.models import Sequential # sequential module reqd to initialize the NN

from keras.layers import Dense # dense module reqd to build the layers of the NN
# Initialising the ANN

loans_predictor = Sequential() 

# creating object of Sequential class
# Adding the input layer and the hidden layer

loans_predictor.add(Dense(input_dim=93, activation="relu", kernel_initializer="uniform", units=5))
# Adding the output layer

loans_predictor.add(Dense(activation = 'sigmoid', kernel_initializer = "uniform", units = 1))
# Compiling the ANN

loans_predictor.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Fitting the ANN to the Training set

loans_predictor.fit(X_train_features, y_train, batch_size = 10, nb_epoch = 10)
# Making the predictions and evaluating the model



# Predicting the Test set results

y_pred = loans_predictor.predict(X_val_features)
y_pred = (y_pred > 0.5)
# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_val, y_pred)



cm