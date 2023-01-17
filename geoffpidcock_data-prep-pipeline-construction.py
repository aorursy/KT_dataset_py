# # Installing libraries if needed
# import sys
# # This workbook uses the Pandas Profiling library - you can install it on your local system using this code
# !{sys.executable} -m pip install pandas_profiling
# # # This workbook may also use the fuzzywuzzy string matching library, which can be installed as follows
# !{sys.executable} -m pip install fuzzywuzzy
# ! {sys.executable} -m pip install python-Levenshtein
%%time
# This workbook uses the following libraries
import sys
import numpy as np
import pandas as pd
pd.set_option('display.max_rows',100)
pd.set_option('display.max_columns', 200) # Good for wide datasets - otherwise it will truncate the data in views like head

# for missing value processing
from sklearn.preprocessing import Imputer

# for numeric processing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from mlxtend.preprocessing import minmax_scaling
# for Box-Cox Transformation
from scipy import stats

# for text processing
import re 
import string
import fuzzywuzzy
from fuzzywuzzy import process

# Setting the seed for reproducibility
np.random.seed(42)
%matplotlib inline

import matplotlib
import matplotlib.pyplot as plt
import pandas_profiling # for exploration of datasets
# importing data
filepath_or_url = "../input/TrainData.csv"
data_raw = pd.read_csv(filepath_or_url,low_memory=False) # import CSV
# raw_data = pd.read_excel(filepath,sheet_name="") # import XLSX
# validating import
display(data_raw.head(),data_raw.shape)
# Evaluate column names, types, nulls, using info.
data_raw.info()
# Evaluate numerical and categorical top line items using describe
data_raw.describe(include=[np.number]).T
# Evaluate object and categorical items using describe - are they high in cardinality?!?
data_raw.describe(include=[np.object,pd.Categorical]).T
# Create a "pandas profile report" to enable efficient deep dives into all features.
report = pandas_profiling.ProfileReport(data_raw)
report
# You can export the profile report if needed
report.to_file(outputfile="raw_data_profile.html")
# You can also see what variables the report recommends excluding based upon a correlation with other variables >0.9
report.get_rejected_variables()
# finally matplot.lib and Boolean Indexing can help deep dive on fields with strange distributions
data_raw.donations_and_bequests.dropna().hist()
data_raw[data_raw.donations_and_bequests<1000000].donations_and_bequests.dropna().hist()
# There is also a strong right skew
# As per Chris Albion, it is best practice to treat a data frame as immutable, and to copy before manipulation (to protect against mistakes)

# Copying the raw data file:
data_prep_1 = data_raw.copy()

# Standardizing column names to snake case - not needed:
# data_prep_1.columns = [c.replace(' ', '_') for c in data_prep_1.columns]
# data_prep_1.columns =  [c.lower() for c in data_prep_1.columns]
# data_prep_1.columns = [re.sub(r'\W+','_',c) for c in data_prep_1.columns]

# Reporting the resulting columns as a list for later reference
data_prep_1.columns.tolist()
# Evaluating the shape of the dataframe, and whether each row is a sample and each column is a variable
# A random sample is thought to be a good way to do this.
data_prep_1.sample(20)
# testing for duplicates - first across all features
data_prep_1[data_prep_1.duplicated()]
# testing for duplicates - then across index candidate
index = "id"
data_prep_1[data_prep_1.duplicated(index)]
# Removing a duplicate from the index
# data_prep_2 = data_prep_1.drop_duplicates(subset=index, keep='first')

# Optional if the previous step is not applicable
data_prep_2 = data_prep_1.copy()
# Reindexing on the new variable
data_prep_3 = data_prep_2.set_index(index) # note this will delete the the original record
# Creating a list of irrelevant columns based on the profile report and other observations
cols_exclude_total = ['abn_hashed','address_line_1','address_line_2','address_type','ais_due_date',\
'ais_due_date_processed','brc','country','conducted_activities',\
'description_of_purposes_change__if_applicable_','fin_report_from',\
'operating_countries','other_activity_description','other_beneficiaries_description',\
'postcode','registration_status','type_of_financial_statement','fin_report_to','accrual_accounting','charity_activities_and_outcomes_helped_achieve_charity_purpose']
cols_exclude_total
# Sorting columns - defining the logic for the sort:
def feature_sort(cols_num,cols_bool,cols_date,cols_cat,cols_other,cols_exclude_total):
    for col in data_prep_3.columns:
        if col not in cols_exclude_total + cols_num + cols_bool + cols_date + cols_cat + cols_other:
            if col in data_prep_3.columns[(data_prep_3.dtypes == np.float64) | (data_prep_3.dtypes == np.float32)]:
                cols_num.append(col)
            elif (data_prep_3[col].nunique() == 2) or ("true" in data_prep_3[col].unique()) or ("false" in data_prep_3[col].unique()) \
            or ("yes" in data_prep_3[col].unique()) or ("no" in data_prep_3[col].unique()):
                cols_bool.append(col)
            elif 'date' in str(col):
                cols_date.append(col)
            elif data_prep_3[col].nunique() < data_prep_3.shape[0]/100: # Arbitrary limit
                cols_cat.append(col)
            else:
                cols_other.append(col)
    return cols_num,cols_bool,cols_date,cols_cat,cols_other
# exceptions can be handled by placing their values in the column names before executing the for loop
cols_num = []
cols_bool = ['purpose_change_in_next_fy']
cols_date = []
cols_cat = []
cols_other = ['staff_volunteers','staff_full_time','town_city']
# Running the sort
cols_num,cols_bool,cols_date,cols_cat,cols_other = feature_sort(cols_num,cols_bool,cols_date,cols_cat,cols_other,cols_exclude_total)
# Evalaute the results of the sort
[print(key+" features:",value,sep='\n') for key,value in {"cols_num":cols_num,"cols_bool":cols_bool,"cols_date":cols_date,\
                                          "cols_cat":cols_cat,"cols_other":cols_other}.items()]
# Removing Irrelevant features
data_prep_4 = data_prep_3.drop(labels=cols_exclude_total,axis = 1)
data_prep_4.columns.tolist()
# Clean up types for "other features" columns, and append to their appropriate category
cols_other
# staff_volunteers - needs strings converted to numeric values, and object converted to numeric
staff_volunteer_cleanupdict = {'1TO10':5,'11TO50':30,'51TO100':75,'101TO500':350,'0TO50':25,'OVER1000':1250,'501TO1000':750,'NONE':0,'None':0,'1-10':5,'11to50':30,'0-30':15,\
                '11-50':30,'0 to 9.': 5}
data_prep_4['staff_volunteers'] = data_prep_4['staff_volunteers'].apply((lambda x: staff_volunteer_cleanupdict.get(x,x)))
data_prep_4['staff_volunteers'] = pd.to_numeric(data_prep_4['staff_volunteers'],errors='coerce')

# staff_full_time - needs to converted to numeric
data_prep_4['staff_full_time'] = pd.to_numeric(data_prep_4['staff_full_time'],errors='coerce')
cols_num+=['staff_volunteers','staff_full_time']
cols_num
# town_city - needs strings to be standard case, cleaned through reg-ex, and then extracted into a new feature "'Capital city'"
data_prep_4['town_city'].value_counts()
punct_reg = re.compile('[%s+]' % re.escape(string.whitespace + string.punctuation))
def text_proc(text):
    proc = str(text)
    proc = punct_reg.sub('_', proc)
    return proc
data_prep_4['town_city'] = data_prep_4['town_city'].str.lower().apply(lambda x: text_proc(x))
data_prep_4['town_city'].value_counts()
capital_dict = {'melbourne':1,'sydney':1,'adelaide':1,'brisbane':1,'hobart':1,'perth':1,'canberra':1,'darwin':1}
data_prep_4['located_in_capital_city'] = data_prep_4['town_city'].apply(lambda x: capital_dict.get(x,0))
data_prep_4['located_in_capital_city'].value_counts()
data_prep_4.drop(columns='town_city',inplace=True)
# Check category features for "nulls" hiding behind other values (a common gotcha!) 
[print(str(c)+' value counts'\
       ,data_prep_4[c].value_counts()\
       ,sep="\n") for c in cols_cat]
# Build a replacement dict of column names, and values, to replace with NaN's
# replace_dict = {"column":{"value":np.nan},"column":{"value":np.nan}}

# Replace "unknown" values with nans
# data_prep_5 = data_prep_4.replace(to_replace=replace_dict)

# Optional if the previous step is not applicable
data_prep_5 = data_prep_4.copy()
# List columns with missing values by %
data_prep_5.isnull().sum()\
    .apply(lambda x: (x/data_prep_5.shape[0])*100)\
    .sort_values(ascending=False)
# Assess impact of dropping all samples with missing values
print("rows before drop: " + str(data_prep_5.shape[0])\
      ,"rows after drop: " + str(data_prep_5.dropna().shape[0])\
      ,sep="\n")
# Are any missing features unimportant? If so, note them down and drop them
# cols_missing =[] # note them in this list
# cols_exclude_total = cols_exclude_total.append(cols_missing)
# data_prep_6=data_prep_5.drop(labels=cols_missing,axis=1)

# Optional if not dropping samples with nulls
data_prep_6=data_prep_5.copy()
# Are the values for the sample never recorded, or do they not exist?
# This can be determined by reading the docs or through EDA
not_recorded = ['donations_and_bequests','main_activity','charity_activities_and_outcomes_helped_achieve_charity_purpose','staff_casual','staff_part_time','staff_volunteers','staff_full_time',\
               'operates_overseas','state','town_city','charity_size','purpose_change_in_next_fy']
dont_exist = ['previous__net_surplus_deficit','previous__donations_and_bequests','previous__government_grants','previous__total_assets','previous__total_gross_income']
# Drop samples with dont_exist now - Note, in this workbook I've decided to keep these columns, as it's important for the model to predict donations
# for charities who are newly registered/do not have historical data. This may make the model less accurate, but more generalizeable.
# print(data_prep_6.shape[0])
# data_prep_7 = data_prep_6.dropna(subset=dont_exist)
# print(data_prep_7.shape[0])

# Optional - don't drop these columns
data_prep_7 = data_prep_6.copy()
# For object columns with nulls, we'll fill the nulls with the most frequent value
object_nulls_cols = data_prep_7.columns[(data_prep_7.isna().any()) & (data_prep_7.dtypes=='O')].tolist()
for c in object_nulls_cols:
    data_prep_7[c] = data_prep_7[c].fillna(data_prep_7[c].mode().iloc[0])
# for numeric columns, we'll fill na values with two different values:
# staff columns as 0 - 
# monetary columns as their median value (protecting against outliers and right skew)
# Note KNN imputation is recommended by Chris Albion and may be worth trying after this
staff_cols_num = [col for col in cols_num if 'staff_' in col]
other_cols_num = [col for col in cols_num if 'staff_' not in col]
display(staff_cols_num,other_cols_num)
for c in staff_cols_num:
    data_prep_7[c] = data_prep_7[c].fillna(0)
for c in other_cols_num:
    data_prep_7[c] = data_prep_7[c].fillna(data_prep_7[c].mode().iloc[0])
# Check for missing values one more time
data_prep_7.isnull().sum()\
    .apply(lambda x: (x/data_prep_7.shape[0])*100)\
    .sort_values(ascending=False)
# Detecting outliers through Interquartile ranges

# Defining a function to return the range and count of outliers
# based on distance to interquartile ranges (<1.5*Q1, >1.5*Q3)

def bounds_number_of_outliers(x): 
    q1, q3 = np.percentile(x, [25, 75]) 
    iqr = q3 - q1 
    lower_bound = q1 - (iqr * 1.5) 
    upper_bound = q3 + (iqr * 1.5) 
    return lower_bound, upper_bound, len(np.where((x > upper_bound) | (x < lower_bound))[0])
[print(str(c)+" outlier lower bound, upper bound, and count:", bounds_number_of_outliers(data_prep_7[c].dropna()),sep="\n") for c in cols_num]
# To manage outliers, we can remove them, mark them, or transform them

# removing them
# data_prep_8[] = data_prep_7[(data_prep_7[''] < lower_bound) |\
# (data_prep_7[''] > upper_bound)]

# marking them
# data_prep_8['_outlier'] = np.where(data_prep_7[''] > upper_bound)

# transforming them - using log transform - note that this will require all negative values to be set to zero
data_prep_8 = data_prep_7.copy()
for c in cols_num:
    data_prep_8[c].clip(lower=0,inplace=True)
    data_prep_8[c] = data_prep_8[c].apply(lambda x: np.log(x+1))

# or something else
# data_prep_8 = data_prep_7.copy()
# Reference (if needed) evaluating distributions of numeric values
data_prep_8['donations_and_bequests'].hist(bins=100)
# Still not very normal - maybe average would have been wiser than mean
# Feature transformation (i.e. rescaling) 
data_prep_9 = data_prep_8.copy()

# The standard approach is minmax scaling. Note that this does not handle null values.
# Removing the target feature from the scaling
cols_num_trans = cols_num.copy()
cols_num_trans.remove('donations_and_bequests')

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data_prep_9[cols_num_trans] = scaler.fit_transform(data_prep_9[cols_num_trans])

# Chris Albion recommends defaulting to standardization unless the model 
# demands otherwise

# def scaler(x):
# # Create scaler scaler = preprocessing.StandardScaler() # Transform the feature standardized = scaler.fit_transform( x) # Show feature standardized
#     scaler = StandardScaler()
#     return scaler.fit_transform(x)

data_prep_9['previous__donations_and_bequests'].hist(bins=100)
cols_bool
# Review bool cols values
[print(c, data_prep_9[c].value_counts(), sep="\n") for c in cols_bool] 
# create a replacement dict
data_prep_10 = data_prep_9.copy()
for col in cols_bool:
    data_prep_10[col] = data_prep_10[col].replace({ 'nan':0, 'n':0, 'N':0, 'y':1, 'Y':1, 'false':0, 'true':1})
    data_prep_10[col] = data_prep_10[col].astype(np.float32)
# Review transformed bool cols values
[print(c, data_prep_10[c].value_counts(), sep="\n") for c in cols_bool] 
# Review date cols
cols_date
%%time
# Date encoding - note that this can be very slow, so it sometimes can be worthwhile specifying the datetime format
data_prep_11 = data_prep_10.copy()

# for col in cols_date:
#     data_prep_11[col] = pd.to_datetime(data_prep_11[col], infer_datetime_format = True)
# Date feature generation - for a tidy dataset, it can make sense to break out a date feature into week, month, and year. 
# data_prep_11['year_'] = data_prep_11[''].dt.year
# data_prep_11['month_'] = data_prep_11[''].dt.month
# data_prep_11['week_'] = data_prep_11[''].dt.week
# Reviewing Categorical Features and Values
[print(col,data_prep_11[col].value_counts(),sep="\n") for col in cols_cat]
# Standardizing all text in categorical columns to protect against data entry errors
punct_reg = re.compile('[%s+]' % re.escape(string.whitespace + string.punctuation))
def text_proc(text):
    proc = str(text)
    proc = proc.lower() #changes case to lower
    proc = proc.strip() #removes leading and trailing spaces/tabs/new lines
    proc = punct_reg.sub('_', proc)
    return proc
%%time
data_prep_12=data_prep_11.copy()
for col in cols_cat:
    data_prep_12[col] = data_prep_12[col].apply(lambda x: text_proc(x))
# Evaluating after transformation
[print(col,data_prep_12[col].value_counts(),sep="\n") for col in cols_cat]
# Encoding categorical features
# Ordinal categories can be handled through replace. There's a cleaning step there as well for state
scale_mapper = {"charity_size":
                    {'small':1,
                    'medium':2,
                    'large':3,},
                "sample_year":
                    {'fy2016':3,
                    'fy2015':2,
                    'fy2014':1},
                "state":
                    {'victoria':'vic'}
               }
data_prep_12.replace(to_replace=scale_mapper,inplace=True)
data_prep_12.sample_year.value_counts()
# Reducing cardinality for many level features
def reduce_cardinality(feats_cols,data):
    feats_distros = dict()
    for c in feats_cols:
        df = data[c]
        df = df.value_counts()
        df.fillna(0, inplace=True)
        df = df.astype(np.int32)
        df.sort_values(ascending = False, inplace = True)
        df = pd.DataFrame(df)
        df.columns = [c + ' count']
        df[c + ' distribution'] = 100*df/df.sum()
        feats_distros.update({c:df})

    for feat in feats_cols:
        feat_distro = feats_distros[feat][feat + ' distribution']
        feat_index = feat_distro[feat_distro < 1].index.tolist()
        lean_feat_index = len(feat_index)
        if lean_feat_index > 0:
            feat_sub = lean_feat_index*[np.nan]
            feat_dict = dict(zip(feat_index, feat_sub))
            data[feat] = data[feat].replace(feat_dict)
        data[feat].fillna('other',inplace=True)
        
    return data
data_prep_13 = data_prep_12.copy()
data_prep_13 = reduce_cardinality(['main_activity'],data_prep_13)
# Evaluating after encoding and further transformation
[print(col,data_prep_13[col].value_counts(),sep="\n") for col in cols_cat]
# Nominal categories can be handled through one hot encoding or dummification
data_prep_14 = pd.get_dummies(data_prep_13, prefix = None, prefix_sep = '-', dummy_na = False, columns = ['state','main_activity'])
data_prep_14.columns.tolist()
# I.e. for Geo features - obtaining postcode for each tree
# Though there is a limit of 1 search per second!
# import sys
# !{sys.executable} -m pip install geopy
# from geopy.geocoders import Nominatim
# geolocator = Nominatim()
# result = geolocator.reverse("-37.794463412577585, 144.93192049089112")
# result.raw['address']['postcode']
# result.raw['address'].keys()
# Profile cleaned data
report2 = pandas_profiling.ProfileReport(data_prep_14)
report2
report2.to_file(outputfile="prepped_data_profile.html")
# Eliminate Rejected variable
data_prep_15 = data_prep_14.copy()
data_prep_15.drop(columns=['males'],inplace=True)
data_prep_15.to_csv('TrainData_Prepped.csv')
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# for text processing
import re 
import string

# for numeric processing
from sklearn.preprocessing import MinMaxScaler

# Setting the seed for reproducibility
np.random.seed(42)
%matplotlib inline
import pandas_profiling

train_raw = pd.read_csv("../input/TrainData.csv",low_memory=False)
test_raw = pd.read_csv("../input/TestData.csv",low_memory=False)
def prep_pipeline(data_prep):
    
    # Set Index
    data_prep = data_prep.set_index("id")
    
    # Specify features to exclude
    cols_exclude_total = ['abn_hashed','address_line_1','address_line_2','address_type','ais_due_date',\
        'ais_due_date_processed','brc','country','conducted_activities',\
        'description_of_purposes_change__if_applicable_','fin_report_from',\
        'operating_countries','other_activity_description','other_beneficiaries_description',\
        'postcode','registration_status','type_of_financial_statement','fin_report_to','accrual_accounting',\
        'charity_activities_and_outcomes_helped_achieve_charity_purpose']
    
    # Sort Columns
    def feature_sort(cols_num,cols_bool,cols_date,cols_cat,cols_other,cols_exclude_total):
        for col in data_prep.columns:
            if col not in cols_exclude_total + cols_num + cols_bool + cols_date + cols_cat + cols_other:
                if col in data_prep.columns[(data_prep.dtypes == np.float64) | (data_prep.dtypes == np.float32)]:
                    cols_num.append(col)
                elif (data_prep[col].nunique() == 2) or ("true" in data_prep[col].unique()) or ("false" in data_prep[col].unique()) \
                or ("yes" in data_prep[col].unique()) or ("no" in data_prep[col].unique()):
                    cols_bool.append(col)
                elif 'date' in str(col):
                    cols_date.append(col)
                elif data_prep[col].nunique() < data_prep.shape[0]/100: # Arbitrary limit
                    cols_cat.append(col)
                else:
                    cols_other.append(col)
        return cols_num,cols_bool,cols_date,cols_cat,cols_other
    
    cols_num = []
    cols_bool = ['purpose_change_in_next_fy']
    cols_date = []
    cols_cat = []
    cols_other = ['staff_volunteers','staff_full_time','town_city']
    
    cols_num,cols_bool,cols_date,cols_cat,cols_other = feature_sort(cols_num,cols_bool,cols_date,cols_cat,cols_other,cols_exclude_total)
    
    # Drop exclusion columns
    data_prep = data_prep.drop(labels=cols_exclude_total,axis = 1)
    
    # Clean up other feature columns
    staff_volunteer_cleanupdict = {'1TO10':5,'11TO50':30,'51TO100':75,'101TO500':350,'0TO50':25,'OVER1000':1250,'501TO1000':750,'NONE':0,'None':0,'1-10':5,'11to50':30,'0-30':15,\
                '11-50':30,'0 to 9.': 5}
    data_prep['staff_volunteers'] = data_prep['staff_volunteers'].apply((lambda x: staff_volunteer_cleanupdict.get(x,x)))
    data_prep['staff_volunteers'] = pd.to_numeric(data_prep['staff_volunteers'],errors='coerce')
    data_prep['staff_full_time'] = pd.to_numeric(data_prep['staff_full_time'],errors='coerce')
    cols_num+=['staff_volunteers','staff_full_time']
    
    punct_reg = re.compile('[%s+]' % re.escape(string.whitespace + string.punctuation))
    def text_proc(text):
        proc = str(text)
        proc = punct_reg.sub('_', proc)
        return proc
    
    data_prep['town_city'] = data_prep['town_city'].str.lower().apply(lambda x: text_proc(x))
    capital_dict = {'melbourne':1,'sydney':1,'adelaide':1,'brisbane':1,'hobart':1,'perth':1,'canberra':1,'darwin':1}
    data_prep['located_in_capital_city'] = data_prep['town_city'].apply(lambda x: capital_dict.get(x,0))
    data_prep.drop(columns='town_city',inplace=True)
    
    # Fill Nulls
    object_nulls_cols = data_prep.columns[(data_prep.isna().any()) & (data_prep.dtypes=='O')].tolist()
    
    # For object cols, replace nulls with most common value
    for c in object_nulls_cols:
        data_prep[c] = data_prep[c].fillna(data_prep[c].mode().iloc[0])
    
    # For staff numeric cols, replace nans with 0
    staff_cols_num = [col for col in cols_num if 'staff_' in col]
    for c in staff_cols_num:
        data_prep[c] = data_prep[c].fillna(0)
    # For other numeric cols, replace nans with median
    other_cols_num = [col for col in cols_num if 'staff_' not in col]
    for c in other_cols_num:
        data_prep[c] = data_prep[c].fillna(data_prep[c].median())
    
    # Numeric cleaning - clip and log transformation
    for c in cols_num:
        data_prep[c].clip(lower=0,inplace=True)
        data_prep[c] = data_prep[c].apply(lambda x: np.log(x+1))
    
    # Numeric cleaning - minmax scaling (on all except target)
    cols_num_trans = cols_num.copy()
    try:
        cols_num_trans.remove('donations_and_bequests')
    except ValueError:
        pass  # do nothing!
    
    scaler = MinMaxScaler()
    data_prep[cols_num_trans] = scaler.fit_transform(data_prep[cols_num_trans])
    
    # Boolean cleaning
    for col in cols_bool:
        data_prep[col] = data_prep[col].replace({ 'nan':0, 'n':0, 'N':0, 'y':1, 'Y':1, 'false':0, 'true':1})
        data_prep[col] = data_prep[col].astype(np.float32)
        
    # Category feature cleaning
    def text_proc_2(text):
        proc = str(text)
        proc = proc.lower() #changes case to lower
        proc = proc.strip() #removes leading and trailing spaces/tabs/new lines
        proc = punct_reg.sub('_', proc)
        return proc
    
    for col in cols_cat:
        data_prep[col] = data_prep[col].apply(lambda x: text_proc_2(x))
        
    # Ordinal encoding    
    scale_mapper = {"charity_size":
                    {'small':1,
                    'medium':2,
                    'large':3,},
                "sample_year":
                    {'fy2016':3,
                    'fy2015':2,
                    'fy2014':1},
                "state":
                    {'victoria':'vic'}
               }
    data_prep.replace(to_replace=scale_mapper,inplace=True)
    
    # Reducing cardinality for nominal features
    def reduce_cardinality(feats_cols,data):
        feats_distros = dict()
        for c in feats_cols:
            df = data[c]
            df = df.value_counts()
            df.fillna(0, inplace=True)
            df = df.astype(np.int32)
            df.sort_values(ascending = False, inplace = True)
            df = pd.DataFrame(df)
            df.columns = [c + ' count']
            df[c + ' distribution'] = 100*df/df.sum()
            feats_distros.update({c:df})

        for feat in feats_cols:
            feat_distro = feats_distros[feat][feat + ' distribution']
            feat_index = feat_distro[feat_distro < 1].index.tolist()
            lean_feat_index = len(feat_index)
            if lean_feat_index > 0:
                feat_sub = lean_feat_index*[np.nan]
                feat_dict = dict(zip(feat_index, feat_sub))
                data[feat] = data[feat].replace(feat_dict)
            data[feat].fillna('other',inplace=True)

        return data
    
    data_prep = reduce_cardinality(['main_activity'],data_prep)
    
    # Getting dummies for Nominal features
    data_prep = pd.get_dummies(data_prep, prefix = None, prefix_sep = '-', dummy_na = False, columns = ['state','main_activity'])
    
    return data_prep
%%time
train_prep = prep_pipeline(train_raw)
%%time
test_prep = prep_pipeline(test_raw)
# the moment of truth
# Determining which columns are in train_prep but not in test_prep
set(train_prep.columns).difference(set(test_prep.columns))
test_prep['main_activity-animal_protection'] = 0
set(train_prep.columns).difference(set(test_prep.columns))
report_3 = pandas_profiling.ProfileReport(test_prep)
report_3
report_3.to_file(outputfile="test_data_profile.html")