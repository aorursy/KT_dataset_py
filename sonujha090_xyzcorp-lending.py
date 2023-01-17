import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.plotly as py
import plotly.graph_objs as go

% matplotlib inline
plt.style.use('fivethirtyeight')
%time data = pd.read_table('../input/XYZCorp_LendingData.txt',parse_dates=['issue_d'],low_memory=False)
train_df = data[data['issue_d'] < '2015-6-01']
test_df = data[data['issue_d'] >= '2015-6-01']
train = train_df.copy()
test = test_df.copy()
train.dtypes.value_counts()
print(train.shape)
print(test.shape)
train['default_ind'].value_counts().plot.bar()
train.head()
train.describe()
train.describe(exclude=np.number)
train.dtypes.value_counts()
fig, ax = plt.subplots(1, 3, figsize=(16,5))

sns.distplot(train['loan_amnt'], ax=ax[0])
sns.distplot(train['funded_amnt'], ax=ax[1])
sns.distplot(train['funded_amnt_inv'], ax=ax[2])

ax[1].set_title("Amount Funded by the Lender")
ax[0].set_title("Loan Applied by the Borrower")
ax[2].set_title("Total committed by Investors")
train.purpose.value_counts(ascending=False).plot.bar(figsize=(10,5))
plt.xlabel('purpose'); plt.ylabel('Density'); plt.title('Purpose of loan');
plt.figure(figsize=(10,5))
train['issue_year'] = train['issue_d'].dt.year
sns.barplot(x='issue_year',y='loan_amnt',data=train)
# Loan Status 
fig, ax = plt.subplots(1, 2, figsize=(16,5))
train['default_ind'].value_counts().plot.pie(explode=[0,0.25],labels=['good loans','bad loans'],
                                             autopct='%1.2f%%',startangle=70,ax=ax[0])
sns.kdeplot(train.loc[train['default_ind']==0,'issue_year'],label='default_ind = 0')
sns.kdeplot(train.loc[train['default_ind']==1,'issue_year'],label='default_ind = 1')
plt.xlabel('Year'); plt.ylabel('Density'); plt.title('Yearwise Distribution of defaulter')
train.grade.value_counts().plot.bar()
fig,array=plt.subplots(1,2,figsize=(12,5))
train.loc[train['default_ind']==0,'grade'].value_counts().plot.bar(ax=array[0])
train.loc[train['default_ind']==1,'grade'].value_counts().plot.bar(ax=array[1])
array[0].set_title('default_ind=0 vs grade'),array[1].set_title('default_ind=1 vs grade')
train.addr_state.unique()
# Make a list with each of the regions by state.

west = ['WA','CA', 'OR', 'UT','ID','CO', 'NV', 'NM', 'AK', 'MT', 'HI', 'WY']
south_east = ['AZ', 'TX', 'OK','GA', 'NC', 'VA', 'FL', 'KY', 'SC', 'LA', 'AL', 'WV', 'DC', 'AR', 'DE', 'MS', 'TN' ]
mid_west = ['IL', 'MO', 'MN', 'OH', 'WI', 'KS', 'MI', 'SD', 'IA', 'NE', 'IN', 'ND']
north_east = ['CT', 'NY', 'PA', 'NJ', 'RI','MA', 'MD', 'VT', 'NH', 'ME']

train['region'] = np.nan

def fix_regions(addr_state):
        if addr_state in west:
            return 'west'
        elif addr_state in south_east:
            return 'south east'
        elif addr_state in mid_west:
            return 'mid west'
        else:
            return 'north east'
        
train['region'] = train['addr_state'].apply(fix_regions)
date_amt_region = train[['loan_amnt','issue_d','region']]
plt.style.use('dark_background')
cmap = plt.cm.Set3
by_issued_amount = date_amt_region.groupby(['issue_d', 'region']).loan_amnt.sum()
by_issued_amount.unstack().plot(stacked=False, colormap=cmap, grid=False, legend=True, figsize=(15,6))

plt.title('Loans issued by Region', fontsize=16)
train.emp_length.unique()
train.loc[train['emp_length']=='10+ years','emp_len'] = 10
train.loc[train['emp_length']=='<1 year','emp_len'] = .5
train.loc[train['emp_length']=='1 year','emp_len'] = 1
train.loc[train['emp_length']=='3 years','emp_len'] = 3
train.loc[train['emp_length']=='8 years','emp_len'] = 8
train.loc[train['emp_length']=='9 years','emp_len'] = 9
train.loc[train['emp_length']=='4 years','emp_len'] = 4
train.loc[train['emp_length']=='5 years','emp_len'] = 5
train.loc[train['emp_length']=='6 years','emp_len'] = 6
train.loc[train['emp_length']=='2 years','emp_len'] = 2
train.loc[train['emp_length']=='7 years','emp_len'] = 7
train.loc[train['emp_length']=='nan','emp_len'] = 0
# Loan issued by Region ,Credit Score and grade
plt.style.use('seaborn-ticks')


f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,figsize=(16,12))

regional_interest_rate = train.groupby(['issue_year', 'region']).int_rate.mean()
regional_interest_rate.unstack().plot(kind='area',  stacked=True,colormap=cmap, grid=False,
                                      legend=False, figsize=(16,12),ax=ax1)

regional_emp_length = train.groupby(['issue_year', 'region']).emp_len.mean()
regional_emp_length.unstack().plot(kind='area',  stacked=True,colormap=cmap, grid=False,
                                      legend=False, figsize=(16,12),ax=ax2)

regional_dti = train.groupby(['issue_year', 'region']).dti.mean()
regional_dti.unstack().plot(kind='area',  stacked=True,colormap=cmap, grid=False,
                                      legend=False, figsize=(16,12),ax=ax3)

regional_interest_rate = train.groupby(['issue_year', 'region']).annual_inc.mean()
regional_interest_rate.unstack().plot(kind='area',  stacked=True,colormap=cmap, grid=False,
                                      legend=False, figsize=(16,12),ax=ax4)
ax1.set_title('averate interest rate vs region'),ax2.set_title('averate emp_length by region')
ax3.set_title('average dti by region'),ax4.set_title('average annual income by region')

ax4.legend(bbox_to_anchor=(-1.0, -0.5, 1.8, 0.1), loc=10,prop={'size':12},
           ncol=5, mode="expand", borderaxespad=0.)
print(train.int_rate.mean())
print(train.annual_inc.mean())
train['income_category'] = np.nan
train.loc[train['annual_inc'] <= 100000,'income_category'] = 'Low'
train.loc[(train['annual_inc'] > 100000) & (train['annual_inc'] <= 200000),'income_category'] = 'Medium'
train.loc[train['annual_inc'] > 200000,'income_category'] = 'High'
fig, ((ax1, ax2), (ax3, ax4))= plt.subplots(nrows=2, ncols=2, figsize=(14,8))
plt.style.use('bmh')
sns.violinplot(x="income_category", y="loan_amnt", data=train, ax=ax1 )
sns.violinplot(x="income_category", y="default_ind", data=train, ax=ax2)
sns.boxplot(x="income_category", y="emp_len", data=train, ax=ax3)
sns.boxplot(x="income_category", y="int_rate", data=train, ax=ax4)
plt.tight_layout(h_pad=1.5)
defaulter = train.loc[train['default_ind']==1]
plt.figure(figsize=(16,16))
plt.subplot(211)
sns.boxplot(data=defaulter,x = 'home_ownership',y='loan_amnt',hue='default_ind')
plt.subplot(212)
sns.boxplot(data=defaulter,x='issue_year',y='loan_amnt',hue='home_ownership')
def missing_values_table(df):
    total_missing = df.isnull().sum().sort_values(ascending=False)
    percentage_missing = (100*df.isnull().sum()/len(df)).sort_values(ascending=False)
    missing_table = pd.DataFrame({'missing values':total_missing,'% missing':percentage_missing})
    return missing_table
missing_values = missing_values_table(train)
missing_values.head(20)
train.dtypes.value_counts()
def to_datepart(df,fldname,drop=False):
    fld = df[fldname]
    fld_dtype = fld.dtype
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    attr = ['Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 
            'Is_year_end', 'Is_year_start']
    for n in attr: df[targ_pre + n] = getattr(fld.dt, n.lower())
    df[targ_pre + 'Elapsed'] = fld.astype(np.int64)
    if drop: df.drop(fldname, axis=1, inplace=True)
import re
to_datepart(train,'issue_d',drop=True)
to_datepart(test,'issue_d',drop=True)
def treat_missing(df):
    for c in df.columns:
        if df[c].dtype == 'object':
            df.fillna(df[c].mode()[0],inplace=True)
        else:
            df.fillna(df[c].median(),inplace=True)
treat_missing(train)
treat_missing(test)
def train_cat(df):
    for n,c in df.items():
        if df[n].dtype == 'object': df[n] = c.astype('category').cat.as_ordered()
train_cat(train)
train_cat(test)
train.select_dtypes('category').apply(pd.Series.nunique, axis = 0)
to_drop = ['sub_grade','emp_title','desc','title','zip_code',
           'addr_state','earliest_cr_line','last_pymnt_d','last_credit_pull_d']
train.drop(to_drop,axis=1,inplace=True)
test.drop(to_drop,axis=1,inplace=True)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for c in train.columns:
    if train[c].dtype == 'object':
        if len(list(train[c].unique())) <= 2:
            train[c] = le.fit_transform(train[c])
            test[c] = le.transform(test[c])
print(train.shape)
print(test.shape)
train = pd.get_dummies(train)
test = pd.get_dummies(test)
print(train.shape)
print(test.shape)
# train_label = train['default_ind']
# Align the training and testing data, keep only columns present in both dataframes
train, test = train.align(test, join = 'inner', axis = 1)

# Add the target back in
# train['default_ind'] = train_label

print(train.shape)
print(test.shape)
X = train.copy()
y = X.pop('default_ind')
def split_vals(a,n):return a[:n].copy(),a[n:].copy()
n_valid = len(test_df)  # same as test set size
n_trn = len(X)-n_valid
raw_train,raw_valid = split_vals(train_df,n_trn)
X_train, X_valid = split_vals(X, n_trn)
y_train, y_valid = split_vals(y, n_trn)
X_train.shape, y_train.shape, X_valid.shape
from sklearn.ensemble import RandomForestClassifier
m = RandomForestClassifier(n_jobs=-1,n_estimators=100)
%time m.fit(X_train,y_train)
from sklearn.metrics import confusion_matrix,precision_score,recall_score,roc_auc_score
y_pred = m.predict(X_valid)
print(confusion_matrix(y_valid,y_pred))
print(precision_score(y_valid,y_pred))
print(recall_score(y_valid,y_pred))
print(roc_auc_score(y_valid,y_pred))
m = RandomForestClassifier(n_jobs=-1,n_estimators=100,min_samples_leaf=2)
%time m.fit(X_train,y_train)
y_pred = m.predict(X_valid)
print(precision_score(y_valid,y_pred))
y_pred = m.predict(X_valid)
print(confusion_matrix(y_valid,y_pred))
print(precision_score(y_valid,y_pred))
print(recall_score(y_valid,y_pred))
print(roc_auc_score(y_valid,y_pred))
def feat_importance(m, df):
    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}).sort_values('imp', ascending=False)
fi = feat_importance(m, X_train); fi[:10]
fi.plot('cols', 'imp', figsize=(10,6), legend=False)
plt.style.use("fivethirtyeight")
def plot_fi(fi): return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)
plot_fi(fi[:20])
to_keep = fi[fi.imp>0.005].cols; len(to_keep)
X_keep = X[to_keep].copy()
X_train, X_valid = split_vals(X_keep, n_trn)
m = RandomForestClassifier(n_estimators=40, min_samples_leaf=3, max_features=0.5,
                          n_jobs=-1, oob_score=True)
%time m.fit(X_train, y_train)
y_pred = m.predict(X_valid)
print(confusion_matrix(y_valid,y_pred))
print(precision_score(y_valid,y_pred))
print(recall_score(y_valid,y_pred))
print(roc_auc_score(y_valid,y_pred))
fi = feat_importance(m, X_keep)
plt.style.use('fivethirtyeight')
plot_fi(fi)
import scipy 
from scipy.cluster import hierarchy as hc
corr = np.round(scipy.stats.spearmanr(X_keep).correlation, 4)
corr_condensed = hc.distance.squareform(1-corr)
z = hc.linkage(corr_condensed, method='average')
fig = plt.figure(figsize=(16,10))
dendrogram = hc.dendrogram(z, labels=X_keep.columns, orientation='left', leaf_font_size=16)
plt.show()
to_drop = ['member_id','id','out_prncp_inv','loan_amnt','funded_amnt','total_pymnt']
X.drop(to_drop,axis=1,inplace=True)
X_train, X_valid = split_vals(X, n_trn)
y_train, y_valid = split_vals(y, n_trn)
X_train.shape, y_train.shape, X_valid.shape
m = RandomForestClassifier(n_jobs=-1,n_estimators=100,min_samples_leaf=2)
%time m.fit(X_train, y_train)
y_pred = m.predict(X_valid)
print(confusion_matrix(y_valid,y_pred))
print(precision_score(y_valid,y_pred))
print(recall_score(y_valid,y_pred))
print(roc_auc_score(y_valid,y_pred))
test_label = test['default_ind']
# Align the training and testing data, keep only columns present in both dataframes
X, test = train.align(X, join = 'inner', axis = 1)

# Add the target back in
test['default_ind'] = train_label

print(X.shape)
print(test.shape)
y_test = test.pop('default_ind')
y_pred = m.predict(test)
from sklearn.metrics import classification_report
print(confusion_matrix(y_test,y_pred))
print(precision_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(recall_score(y_test,y_pred))
print(roc_auc_score(y_test,y_pred))