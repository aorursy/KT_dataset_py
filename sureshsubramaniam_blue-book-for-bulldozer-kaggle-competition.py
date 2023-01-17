import pandas as pd
import re
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display
import numpy as np
import math
from sklearn import metrics
from pandas.api.types import is_string_dtype, is_numeric_dtype
import matplotlib.pyplot as plt 
from sklearn.ensemble import forest
import scipy
from scipy.cluster import hierarchy as hc
def rmse(x,y): 
    return math.sqrt(((x-y)**2).mean())

def print_score(m):
    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),
                m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)

def split_vals(a,n): 
    return a[:n].copy(), a[n:].copy()

def get_oob(df):
    m = RandomForestRegressor(n_estimators=40, max_features=0.6, n_jobs=-1, oob_score=True)
    x, _ = split_vals(df, n_trn)
    m.fit(x, y_train)
    return m.oob_score_

def add_datepart(df, fldname, drop=True, time=False):
    fld = df[fldname]
    fld_dtype = fld.dtype
    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        fld_dtype = np.datetime64

    if not np.issubdtype(fld_dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    if time: attr = attr + ['Hour', 'Minute', 'Second']
    for n in attr: df[targ_pre + n] = getattr(fld.dt, n.lower())
    df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9
    if drop: df.drop(fldname, axis=1, inplace=True)
        
def train_cats(df):
    for n,c in df.items():
        if is_string_dtype(c): df[n] = c.astype('category').cat.as_ordered()

def fix_missing(df, col, name, na_dict):
    if is_numeric_dtype(col):
        if pd.isnull(col).sum() or (name in na_dict):
            df[name+'_na'] = pd.isnull(col)
            filler = na_dict[name] if name in na_dict else col.median()
            df[name] = col.fillna(filler)
            na_dict[name] = filler
    return na_dict

def proc_df(df, y_fld=None, skip_flds=None, ignore_flds=None, do_scale=False, na_dict=None,
            preproc_fn=None, max_n_cat=None, subset=None, mapper=None):
    if not ignore_flds: ignore_flds=[]
    if not skip_flds: skip_flds=[]
    if subset: df = get_sample(df,subset)
    else: df = df.copy()
    ignored_flds = df.loc[:, ignore_flds]
    df.drop(ignore_flds, axis=1, inplace=True)
    if preproc_fn: preproc_fn(df)
    if y_fld is None: y = None
    else:
        if not is_numeric_dtype(df[y_fld]): df[y_fld] = df[y_fld].cat.codes
        y = df[y_fld].values
        skip_flds += [y_fld]
    df.drop(skip_flds, axis=1, inplace=True)

    if na_dict is None: na_dict = {}
    else: na_dict = na_dict.copy()
    na_dict_initial = na_dict.copy()
    for n,c in df.items(): na_dict = fix_missing(df, c, n, na_dict)
    if len(na_dict_initial.keys()) > 0:
        df.drop([a + '_na' for a in list(set(na_dict.keys()) - set(na_dict_initial.keys()))], axis=1, inplace=True)
    if do_scale: mapper = scale_vars(df, mapper)
    for n,c in df.items(): numericalize(df, c, n, max_n_cat)
    df = pd.get_dummies(df, dummy_na=True)
    df = pd.concat([ignored_flds, df], axis=1)
    res = [df, y, na_dict]
    if do_scale: res = res + [mapper]
    return res

def numericalize(df, col, name, max_n_cat):
    if not is_numeric_dtype(col) and ( max_n_cat is None or col.nunique()>max_n_cat):
        df[name] = col.cat.codes+1
df_raw = pd.read_csv('../input/bulldozer-training-dataset/Train.csv', low_memory=False, parse_dates=['saledate'])
df_raw.head()
#Change SalePrice to log because the evaluation is for RMSLE
df_raw.SalePrice = np.log(df_raw.SalePrice)
#Change dates to date parts
add_datepart(df_raw, 'saledate')
#Add a column for age of bulldozer
df_raw['age'] = df_raw['saleYear'] - df_raw['YearMade'] 
#Change string variables to category type
train_cats(df_raw)
#Specify order for variable UsageBand and change to codes
df_raw.UsageBand.cat.set_categories(['High', 'Medium', 'Low'], ordered=True, inplace=True)
df_raw.UsageBand = df_raw.UsageBand.cat.codes
#Change categories to code and missing values to 0, replace missing numeric values with median, 
#add column to indicate replaced missing values and separate the dependent variable as a separate df
df, y, nas = proc_df(df_raw, 'SalePrice')
df.head()
df.shape
#Split the dataset into training and validation sets. Use 12,000 as the validation set

n_valid = 12000  # same as Kaggle's test set size
n_trn = len(df)-n_valid
raw_train, raw_valid = split_vals(df_raw, n_trn) #for using unprocessed data if needed.
X_train, X_valid = split_vals(df, n_trn)
y_train, y_valid = split_vals(y, n_trn)
X_train.shape, X_valid.shape, y_train.shape, y_valid.shape
#Run base model
m = RandomForestRegressor(n_jobs=-1)
m.fit(X_train, y_train)
print_score(m);
#Use the feature importance to find the most important ones
feature_importance = pd.DataFrame({'Feature' : X_train.columns, 'Importance' : m.feature_importances_})
feature_importance.sort_values('Importance', ascending=False, inplace=True)
feature_importance.head(30)
feature_importance.plot('Feature', 'Importance')
# Run the model for various cut off values for the importance to find the best set of importance features
for i in [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010, 0.011, 0.012]:
    important_features = feature_importance[feature_importance['Importance'] > i]
    df_important = df[important_features['Feature']]
    X_train, X_valid = split_vals(df_important, n_trn)
    y_train, y_valid = split_vals(y, n_trn)

    m = RandomForestRegressor(n_estimators=40, max_features=0.5, n_jobs=-1, oob_score=True)
    m.fit(X_train, y_train)
    print_score(m)

#The best cut off point seems to be 0.0.006 when the RMSE score is 0.22312856564640468.
important_features = feature_importance[feature_importance['Importance'] > 0.006]
df_important = df[important_features['Feature']]
X_train, X_valid = split_vals(df_important, n_trn)
y_train, y_valid = split_vals(y, n_trn)

m = RandomForestRegressor(n_estimators=40, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)

#Detect and remove redundant features
#Draw dendogram of feature clusters
corr = np.round(scipy.stats.spearmanr(df_important).correlation, 4)
corr_condensed = hc.distance.squareform(1-corr)
z = hc.linkage(corr_condensed, method='average')
fig = plt.figure(figsize=(16,10))
dendrogram = hc.dendrogram(z, labels=df_important.columns, orientation='left', leaf_font_size=16)
plt.show()

#These feature pairs are in the same cluster'
cluster_pairs = ['saleDayofyear', 'state', 'Drive_System', 'fiSecondaryDesc', 'MachineID', 'ModelID', 'saleElapsed', 'YearMade', 'Enclosure', 'Coupler_System', 'fiModelDescriptor', 'ProductSize','fiBaseModel', 'fiModelDesc']
#Base OOB score
get_oob(df_important)
#Get the OOB score after dropping each of the variables in the cluster pairs
for c in cluster_pairs:
    print(c, get_oob(df_important.drop(c, axis=1)))

#For each pair select the attribute which impacts the score less (score is higher) and remove it and calculate OOB
to_drop = ['state', 'Drive_System', 'MachineID', 'Coupler_System', 'fiModelDescriptor','fiModelDesc']
get_oob(df_important.drop(to_drop, axis=1))

#OOB score has decreased slightly after removing attributes but model has become simpler.
#Run the random forest on the dataset after dropping the columns
df_keep = df_important.drop(to_drop, axis=1)
X_train, X_valid = split_vals(df_keep, n_trn)
m = RandomForestRegressor(n_estimators=40, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)

df_keep.columns
#Remove time related features to generalize the model more
#Label the validation and training set and calculate the OOB score
df_ext = df_keep.copy()
df_ext['is_valid'] = 1
df_ext.is_valid[:n_trn] = 0
x, y, nas = proc_df(df_ext, 'is_valid')

m = RandomForestClassifier(n_estimators=40, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(x, y);
m.oob_score_

#Very high OOB score
#Find the important features, i.e. the features which help rf predict the validation and training sets
feature_importance_ext = pd.DataFrame({'Feature' : x.columns, 'Importance' : m.feature_importances_})
feature_importance_ext.sort_values('Importance', ascending=False, inplace=True)
feature_importance_ext.head(30)

#Drop the top 1 and see if the RMSe improves
to_drop = ['SalesID']
df_keep = df_important.drop(to_drop, axis=1)
X_train, X_valid = split_vals(df_keep, n_trn)
m = RandomForestRegressor(n_estimators=40, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)

#Run the final model
m = RandomForestRegressor(n_estimators=160, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)

#Import data
df_raw = pd.read_csv('../input/bulldozer-training-dataset/Train.csv', low_memory=False, parse_dates=['saledate'])
df_validation = pd.read_csv('../input/bluebook-for-bulldozers/Valid.csv', low_memory=False, parse_dates=['saledate'])

print('training shape',df_raw.shape)
print('validation shape', df_validation.shape)
print('difference between training and validaiton', set(df_raw.columns) - set(df_validation.columns))
#Separate out the SalePrice as y and change it to log and drop it from the training set
y = np.log(df_raw['SalePrice'])
df_raw = df_raw.drop('SalePrice', axis=1)
#Append the validation set to the training set
df_train_valid = df_raw.append(df_validation)

df_train_valid.shape
#Change dates to date parts
add_datepart(df_train_valid, 'saledate')
#Add a column for age of bulldozer
df_train_valid['age'] = df_train_valid['saleYear'] - df_train_valid['YearMade'] 

#Change string variables to category type
train_cats(df_train_valid)

#Specify order for variable UsageBand and change to codes
df_train_valid.UsageBand.cat.set_categories(['High', 'Medium', 'Low'], ordered=True, inplace=True)
df_train_valid.UsageBand = df_train_valid.UsageBand.cat.codes

#Change other categories into codes and replace NaNs with 0.
cat_cols = list(df_train_valid.select_dtypes(include=['category']).columns)  #Above UsageType is changed to Int 
for col in cat_cols:
    s = df_train_valid[col] 
    df_train_valid[col] = s.cat.codes+1
#Replace the NaNs for the numerical column with mean
df_train_valid['auctioneerID'].fillna(df_train_valid['auctioneerID'].median(), inplace=True)
df_train_valid['MachineHoursCurrentMeter'].fillna(df_train_valid['MachineHoursCurrentMeter'].median(), inplace=True)
#Check if df has NaNs
df_train_valid.isnull().sum()
df_train_valid.head()
df_train_valid.shape
# These were the columns in the final model
df_keep.columns
#Choose only columns which were used in the final model
df_train_valid = df_train_valid[df_keep.columns]

#Separate the training and validation sets
df_valid = df_train_valid.tail(11573)
df_train = df_train_valid.head(401125)

print(df_valid.shape)
print(df_train.shape)
#Train the model on training set and dependent variable using out final model
m = RandomForestRegressor(n_estimators=160, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(df_train, y) 
#Import the validation solution
solution = pd.read_csv('../input/bluebook-for-bulldozers/ValidSolution.csv', low_memory=False)
y_actual = np.log(solution.SalePrice)     
#Calculate the RMSE using the prediction from the validation set and the actual provided by Kaggle in the file 'ValidSolutions.csv'
rmse(m.predict(df_valid), y_actual)