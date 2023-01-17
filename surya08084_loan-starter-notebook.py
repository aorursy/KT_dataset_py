import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from imblearn.metrics import sensitivity_specificity_support
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from datetime import timedelta, date
import warnings
warnings.filterwarnings('ignore')
import re
pd.set_option('display.float_format', lambda x: '%.8f' % x)
df = pd.read_csv("../input/loan-data-set/DR_Demo_Lending_Club.csv")
df1 = df.copy()
# There are 10000 data points and 28 features
df1.shape
df1.info()
df1.describe()
df1.dtypes
# 7% values are null
100*(df1.isnull().sum().sum())/(df1.notnull().sum().sum())
# Dropping ID columns as it is just an unique id assigned to the records
df1.drop(columns ='Id',inplace = True)
df1.duplicated().sum()
df1.isnull().sum()
# Dropping the columns which have more than 80% of NA values
for i in df1.columns:
    if (100*df1[i].isnull().sum()/len(df1.index))>62:
        print("Dropped column is : {} and % of null values in this col is : {}".format(i,100*df1[i].isnull().sum()/len(df1)))
        df1.drop(columns=i,inplace = True)
#There are 12.95% loans which are bad
100*df1['is_bad'].value_counts()[1]/(sum(df1['is_bad'].value_counts()))
sns.countplot(x='is_bad',data = df1)
# Dropping emp_title as this is not useful for analysis
df1.drop('emp_title',axis=1,inplace=True)
df1.initial_list_status.value_counts()
#There are l 0.17% m values and this does not contain much information hence dropping this feature
df1.Notes.isnull().sum()
#There are 3230 null values in Notes feature and this feature contains text which is provided by user at the time
# of applying the loan. As category is already captured hence dropping this feature
df1.drop('Notes',axis=1,inplace=True)
df1.drop('initial_list_status',axis=1,inplace=True)
df1.isnull().sum()
df1.isnull().sum()
# There are many feature where there are 4 NA values. Let's check if it all belong to same ID.
# 'purpose', 'delinq_2yrs','earliest_cr_line','inq_last_6mths','open_acc','pub_rec','total_acc'
# only purpose have NA values in different rows and rest above feature have NULL values in same rows hence dropping 
# the NA rows of above feature

df1.drop(df1[df1['delinq_2yrs'].isnull()].index, inplace = True) 
df1.isnull().sum()
# Replacing NA values of revol_util with median

df1['revol_util'] = df1['revol_util'].fillna(df1['revol_util'].median())
# Assuming na values are representing as not available and replacing this with 0
df1['emp_length'] = [c.replace('na','0') for c in df1['emp_length']]
df1.isnull().sum()
df1.collections_12_mths_ex_med.value_counts()
# Dropping collections_12_mths_ex_med as this feature does not have pattern to learn

df1.drop(columns="collections_12_mths_ex_med",inplace=True,axis =1)
df1.isnull().sum()
# Dropping Purpose as purpose category is containing the category of the purpose
df1.drop(columns="purpose",inplace=True,axis =1)
df1.columns
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(20,15))
corr = df1.corr(method = 'spearman')

sns.heatmap(corr, annot = True)

plt.show()
pd.crosstab(df1['purpose_cat'], df1['is_bad'], margins = True)
for i in df1.purpose_cat.unique():
    if re.search('small', i) and len(i)>14:
        print(i)
        df1['purpose_cat'] = df1['purpose_cat'].replace(i,"Other_Small_Business")
df1['purpose_cat'] = df1['purpose_cat'].replace(regex=['other','wedding','major purchase','moving','renewable energy','vacation'], value='Other')
from scipy.stats import pearsonr,chi2_contingency
from itertools import combinations
num_feat = df1.select_dtypes('number').columns.values
comb_num_feat = np.array(list(combinations(num_feat, 2)))
corr_num_feat = np.array([])
for comb in comb_num_feat:
    corr = pearsonr(df1[comb[0]], df1[comb[1]])[0]
    corr_num_feat = np.append(corr_num_feat, corr)
high_corr_num = comb_num_feat[np.abs(corr_num_feat) >= 0.6]
high_corr_num
#Dropping open account as it is highly correlated with total_acc 
df1.drop(columns = 'open_acc',axis=1,inplace=True)
cat_feat = df1.select_dtypes('object').columns.values
comb_cat_feat = np.array(list(combinations(cat_feat, 2)))
corr_cat_feat = np.array([])
for comb in comb_cat_feat:
    table = pd.pivot_table(df1, index=comb[0], columns=comb[1], aggfunc='count').fillna(0)
    corr = np.sqrt(chi2_contingency(table)[0] / (table.values.sum() * (np.min(table.shape) - 1) ) )
    corr_cat_feat = np.append(corr_cat_feat, corr)
high_corr_cat = comb_cat_feat[corr_cat_feat >= 0.5]
high_corr_cat
corr = df1.corr()['is_bad'].sort_values()

# Display correlations
print('Most Positive Correlations:\n', corr.tail(10))
print('\nMost Negative Correlations:\n', corr.head(10))
# Dropping Zip code as it is in encrypted form and we cant extract insight from this feature
df1.drop(columns="zip_code",inplace=True,axis =1)
df1.columns
'''
There are just 2 records of y category for pymnt_plan feature.
As this feature contains less information hence dropping this feature
'''
df1.drop(columns='pymnt_plan',inplace = True)
behav_var = ['delinq_2yrs',
             'earliest_cr_line',
             'inq_last_6mths',
             'pub_rec',
             'revol_bal',
             'revol_util',
             'total_acc',
             'mths_since_last_major_derog',
             'is_bad'
            ]
Demo_var = ['emp_length',
            'home_ownership',
            'annual_inc',
            'verification_status',
            'Notes',
            'addr_state',
            'initial_list_status',
            'debt_to_income',
            'purpose_cat',
            'policy_code',
           ]
df1[behav_var].dtypes
df1['verification_status'] = [i.replace('VERIFIED - income source', 'Verified_Inc_Source') for i in df1['verification_status']]
df1['verification_status'] = [i.replace('not verified', 'Not_Verified') for i in df1['verification_status']]
df1['verification_status'] = [i.replace('VERIFIED - income', 'Verified_Inc') for i in df1['verification_status']]
#Converting variable to category and numeric values data type
df1['is_bad']  =  df1['is_bad'].astype('category')
df1['home_ownership']  =  df1['home_ownership'].astype('category')
df1['verification_status']  =  df1['verification_status'].astype('category')
df1['purpose_cat']  =  df1['purpose_cat'].astype('category')
df1['policy_code']  =  df1['policy_code'].astype('category')
df1['inq_last_6mths']  =  df1['inq_last_6mths'].astype('int')
df1['pub_rec']  =  df1['pub_rec'].astype('int')
df1['pub_rec']  =  df1['pub_rec'].astype('category')
df1['total_acc']  =  df1['total_acc'].astype('int')
df1['emp_length'] = df1['emp_length'].astype('int')
df1.dtypes
plt.figure(figsize=(20, 6))
sns.distplot(df1['annual_inc'],kde=True,color='blue')
pd.crosstab(df1['annual_inc'], df1['is_bad'], margins = True)
def agg_cal(feature):
    '''
    This function will create bivariate plot
    Accept user input and plot the barplot of user input Vs is_bad
    '''
    tot_count = df1.groupby([feature])['is_bad'].count().reset_index(name = 'total_count')
    bad_loan_count = df1[df1['is_bad']==1].groupby([feature])['is_bad'].count().reset_index(name = 'bad_count')
    agg_data = tot_count.merge(bad_loan_count, on=feature)
    agg_data['bad_rate'] = 100*(agg_data['bad_count']/agg_data['total_count']).round(3)
    plt.figure(figsize=(20, 6))
    ax = sns.barplot(x=agg_data[feature], y='bad_rate',order=agg_data.sort_values('bad_rate')[feature],data=agg_data)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=60)
    plt.show()
df1['ann_income_bin']=np.nan
for i in range(df1.shape[0]):
    if df1['annual_inc'].iloc[i]>=0 and df1['annual_inc'].iloc[i]<31000:
        df1['ann_income_bin'].iloc[i]= "Lowest_Income"
    elif df1['annual_inc'].iloc[i]>=31000 and df1['annual_inc'].iloc[i]<42000:
        df1['ann_income_bin'].iloc[i]= "Lower_middle_inc"
    elif df1['annual_inc'].iloc[i]>=42000 and df1['annual_inc'].iloc[i]<126000:
        df1['ann_income_bin'].iloc[i]= "Middle_income"
    elif df1['annual_inc'].iloc[i]>=126000 and df1['annual_inc'].iloc[i]<188000:
        df1['ann_income_bin'].iloc[i]= "Upper_middle_inc"
    elif df1['annual_inc'].iloc[i]>=188000:
        df1['ann_income_bin'].iloc[i]= "Higher_income"
    else:
        df1['ann_income_bin'].iloc[i]='NA'
agg_cal('ann_income_bin')
agg_cal(feature="home_ownership")
agg_cal('verification_status')

agg_cal('purpose_cat')
agg_cal('addr_state')
agg_cal('delinq_2yrs')
agg_cal('inq_last_6mths')
agg_cal('pub_rec')
agg_cal('policy_code')
from datetime import datetime
col = 'earliest_cr_line'
df1[col] = pd.to_datetime(df1[col])
future = df1[col] > pd.to_datetime(date(year=2050,month=1,day=1))
df1.loc[future, col] -= timedelta(days=365.25*100)
df1['earl_cr_line_year'] = pd.DatetimeIndex(df1['earliest_cr_line']).year
df1['earl_cr_line_month'] = pd.DatetimeIndex(df1['earliest_cr_line']).month
df1['earl_cr_line_day'] = pd.DatetimeIndex(df1['earliest_cr_line']).day
df1['length_cr_line'] = datetime.now().year -  df1['earl_cr_line_year']
# dropping earl_cr_line_day as all the records belong to day 1
df1.drop(columns='earl_cr_line_day',inplace=True)
plt.figure(figsize=(20, 6))
sns.countplot(df1.length_cr_line,color='blue')
agg_cal('length_cr_line')
agg_cal('earl_cr_line_month')
def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n

    return x, y

x,y = ecdf(df1.length_cr_line)
# Generate plot
_ = plt.plot(x, y, marker='.', linestyle='none')

# Label the axes
_ = plt.xlabel('length_cr_line')
_ = plt.ylabel('ECDF')

# Display the plot
plt.show()
x,y = ecdf(df1.earl_cr_line_month)
# Generate plot
_ = plt.plot(x, y, marker='.', linestyle='none')

# Label the axes
_ = plt.xlabel('earl_cr_line_month')
_ = plt.ylabel('ECDF')

# Display the plot
plt.show()

# Compute ECDF for annual_income data: x, y
x, y = ecdf(df1.annual_inc)

# Generate plot

_ = plt.plot(x, y, marker='.', linestyle='none')

# Label the axes
_ = plt.xlabel('Annual Income')
_ = plt.ylabel('ECDF')

# Display the plot
plt.show()

# Compute ECDFs
x,y = ecdf(df1.annual_inc)
x_not_bad,y_not_bad = ecdf(df1[df1['is_bad']==0]['annual_inc'])
x_bad,y_bad = ecdf(df1[df1['is_bad']==1]['annual_inc'])

# Plot all ECDFs on the same plot
plt.plot(x,y,marker ='.',linestyle ='none')
plt.plot(x_not_bad,y_not_bad,marker ='.',linestyle ='none')
plt.plot(x_bad,y_bad,marker ='.',linestyle ='none')

# Annotate the plot
plt.legend(('Overall', 'Not bad loan', 'Bad loan'), loc='lower right')
_ = plt.xlabel('Annual Income')
_ = plt.ylabel('ECDF')

# Display the plot
plt.show()
# Compute ECDF for emp_length data: x, y
x, y = ecdf(df1.emp_length)

# Generate plot
_ = plt.plot(x, y, marker='.', linestyle='none')

# Label the axes
_ = plt.xlabel('Emp Length')
_ = plt.ylabel('ECDF')

# Display the plot
plt.show()


agg_cal('emp_length')
# Compute ECDF for emp_length data: x, y
x, y = ecdf(df1.debt_to_income)

# Generate plot
_ = plt.plot(x, y, marker='.', linestyle='none')

# Label the axes
_ = plt.xlabel('Debt to Income')
_ = plt.ylabel('ECDF')

# Display the plot
plt.show()



# Compute ECDF for emp_length data: x, y
x, y = ecdf(df1.revol_util)

# Generate plot
_ = plt.plot(x, y, marker='.', linestyle='none')

# Label the axes
_ = plt.xlabel('Revol Util')
_ = plt.ylabel('ECDF')

# Display the plot
plt.show()




# Compute ECDF for emp_length data: x, y
x, y = ecdf(df1.revol_bal)

# Generate plot
_ = plt.plot(x, y, marker='.', linestyle='none')

# Label the axes
_ = plt.xlabel('Revol Balance')
_ = plt.ylabel('ECDF')

# Display the plot
plt.show()





len(df1[df1['debt_to_income']<=25])/df1.shape[0]
# Compute ECDF for emp_length data: x, y
x, y = ecdf(df1.total_acc)

# Generate plot
_ = plt.plot(x, y, marker='.', linestyle='none')

# Label the axes
_ = plt.xlabel('total account')
_ = plt.ylabel('ECDF')

# Display the plot
plt.show()






#df['emp_length'] = pd.to_numeric(df['emp_length'])
df1['emp_length'] = df1['emp_length'].astype('int')
df1.dtypes
df1.addr_state.unique()
def detect_outlier(data):
    '''
    This function will return number of outlier in entire data frame
    This will store the outlier features and it's value in the form
    of key value pair and once it's called it will return dictionary
    '''
    outlier_dict = dict()
    for i in data.columns:
        if data[i].dtypes=="int64" or data[i].dtypes=="float64":
            Q1 = data[i].quantile(0.25)
            Q3 = data[i].quantile(0.75)
            IQR = Q3-Q1 
            temp_len = len(data[((data[i] < (Q1 - 1.5 * IQR)) | (data[i] > (Q3 + 1.5 * IQR)))])
            if temp_len!=0:
                outlier_dict[i] = len(data[((data[i] < (Q1 - 1.5 * IQR)) | (data[i] > (Q3 + 1.5 * IQR)))])
            else:
                ""
        else:
            ""
    return outlier_dict
detect_outlier(df1)
df1['emp_length'].quantile([0.91,.92,.93,.94,.95,.96,.97,.98,.99,1.0])

#capping the value of emp_length on .99 quantile
df1['emp_length'][df1['emp_length'] >= np.array(df1['emp_length'].quantile([.99]))[0]] = np.array(df1['emp_length'].quantile([.99]))[0]
df1['delinq_2yrs'].quantile([0.89,.92,.93,.94,.95,.96,.97,.98,.99,1.0])


# Compute ECDF for annual_income data: x, y
x, y = ecdf(df1.delinq_2yrs)

# Generate plot

_ = plt.plot(x, y, marker='.', linestyle='none')

# Label the axes
_ = plt.xlabel('delinq_2yrs')
_ = plt.ylabel('ECDF')

# Display the plot
plt.show()


len(df1[df1['delinq_2yrs']<=2])/df1.shape[0]
#capping the value of delinq_2yrs on .97 quantile
df1['delinq_2yrs'][df1['delinq_2yrs'] >= np.array(df1['delinq_2yrs'].quantile([.89]))[0]] = np.array(df1['delinq_2yrs'].quantile([.89]))[0]
detect_outlier(df1)
df1['annual_inc'].quantile([0.89,.92,.93,.94,.95,.96,.97,.98,.99,1.0])

#capping the value of annual_inc on .97 quantile
df1['annual_inc'][df1['annual_inc'] >= np.array(df1['annual_inc'].quantile([.97]))[0]] = np.array(df1['annual_inc'].quantile([.97]))[0]
df1['inq_last_6mths'].quantile([0.89,.92,.93,.94,.95,.96,.97,.98,.99,1.0])

#capping the value of inq_last_6mths on .98 quantile
df1['inq_last_6mths'][df1['inq_last_6mths'] >= np.array(df1['inq_last_6mths'].quantile([.98]))[0]] = np.array(df1['inq_last_6mths'].quantile([.98]))[0]
detect_outlier(df1)
detect_outlier(df1)
df1['revol_bal'].quantile([0.89,.92,.93,.94,.95,.96,.97,.98,.99,1.0])
#capping the value of revol_bal on .97 quantile
df1['revol_bal'][df1['revol_bal'] >= np.array(df1['revol_bal'].quantile([.97]))[0]] = np.array(df1['revol_bal'].quantile([.97]))[0]
detect_outlier(df1)
df1['total_acc'].quantile([0.85,.92,.93,.94,.95,.96,.97,.98,.99,1.0])
sns.boxplot(df1.total_acc)
#capping the value of revol_bal on .98 quantile
df1['total_acc'][df1['total_acc'] >= np.array(df1['total_acc'].quantile([0.98]))[0]] = np.array(df1['total_acc'].quantile([0.98]))[0]
df1['earl_cr_line_year'].quantile([0.08,.92,.93,.94,.95,.96,.97,.98,.99,1.0])
sns.boxplot(df1.earl_cr_line_year)
plt.figure(figsize = (20,6))
sns.distplot(df1.earl_cr_line_year)
#capping the value of earl_cr_line_year 
df1['earl_cr_line_year'][df1['earl_cr_line_year'] <=1985] = 1985
detect_outlier(df1)
sns.boxplot(df1.earl_cr_line_year)
detect_outlier(df1)
df1['length_cr_line'].quantile([0.08,.92,.93,.94,.95,.96,.97,.98,.99,1.0])
sns.boxplot(df1['length_cr_line'])
#capping the value of length_cr_line 
df1['length_cr_line'][df1['length_cr_line'] >=35] = 35
detect_outlier(df1)
detect_outlier(df1)
#from scipy.cluster.vq import kmeans
#from scipy.cluster.vq import vq
#data = df1.mths_since_last_delinq.astype('float64')
#data_raw = data.values
#centroids, avg_distance = kmeans(data_raw, 4)
#groups, cdist = vq(data_raw, centroids)
#y = np.arange(0,9995)
#plt.scatter(data_raw,  y , c=groups)
#plt.xlabel('mths_since_last_delinq')
#plt.ylabel('Indices')
#plt.show()
# Let's convert addr_state feature to real valued feature using probability
# We will replace the state with the probaility of is_bad (for a given state)
country_prob = dict()
for i in df1.addr_state:
    c1 = df1.loc[(df1['addr_state'] == i) & (df1['is_bad'] == 1)]['addr_state'].count()
    c2 = df1.loc[(df1['addr_state'] == i)]['addr_state'].count()
    t = c1/c2
    country_prob[i]=t
df1.replace({"addr_state": country_prob},inplace=True)
# Assiggning 2 levels categorical variable to 0 and 1

#Ignore Target variable

for i in df1.columns[1:]:
    if len(df1[i].value_counts())<=2 and df1[i].dtypes!='int64' and df1[i].dtypes!='float64':
        df1[i] = df1[i].map({df1[i].value_counts().index[0]: 0, df1[i].value_counts().index[1]: 1})
        print(i)
# Dropping ann_income_bin and earliest_cr_line 
df1.drop(columns=['ann_income_bin','earliest_cr_line'],inplace = True)
#Identifying categorical variables which have more than 2 levels
cat_var_g_2level = [i for i in df1.columns if len(df1[i].value_counts())>2 and df1[i].dtypes!='int64' and df1[i].dtypes!='float64' and df1[i].dtypes!='object']
cat_var_g_2level
# Creating a dummy variable for some of the categorical variables and dropping the first one.
dummy1 = pd.get_dummies(df1[cat_var_g_2level], drop_first=True)

# Adding the results to the master dataframe
df1 = pd.concat([df1, dummy1], axis=1)
df1.drop(columns=cat_var_g_2level,inplace=True)
df1.shape
df1.dtypes
scaling_features = ['emp_length','annual_inc','debt_to_income','inq_last_6mths','revol_bal','revol_util','total_acc','mths_since_last_major_derog','earl_cr_line_year','earl_cr_line_month','length_cr_line']
df1[scaling_features]
df1.head()
df2 = df1.copy()
df2['is_bad'] = df2.is_bad.astype('int64')
def iv_woe(data, target, bins=10, show_woe=False):
    
    #Empty Dataframe
    newDF = pd.DataFrame()
    
    #Extract Column Names
    cols = [i for i in data.columns if i!=target]
    
    #Run WOE and IV on all the independent variables
    for ivars in cols:
        if (data[ivars].dtype.kind in 'bifc') and (len(np.unique(data[ivars]))>10):
            binned_x = pd.qcut(data[ivars], bins,  duplicates='drop')
            d0 = pd.DataFrame({'x': binned_x, 'y': data[target]})
        else:
            d0 = pd.DataFrame({'x': data[ivars], 'y': data[target]})

        d = d0.groupby("x", as_index=False).agg({"y": ["count", "sum"]})
        d.columns = ['Cutoff', 'N', 'Events']
        d['% of Events'] = d['Events'] / d['Events'].sum()
        d['Non-Events'] = d['N'] - d['Events']
        d['% of Non-Events'] = d['Non-Events'] / d['Non-Events'].sum()
        d.loc[d['% of Non-Events'] == 0.0,'% of Non-Events'] = 1e-312
        d['WoE'] = np.log(d['% of Events']/d['% of Non-Events'])
        d['IV'] = d['WoE'] * (d['% of Events'] - d['% of Non-Events'])
        print("Information value of " + ivars + " is " + str(round(d['IV'].sum(),6)))
        temp =pd.DataFrame({"Variable" : [ivars], 
                            "IV" : [d['IV'].sum()]}, columns = ["Variable", "IV"])
        newDF=pd.concat([newDF,temp], axis=0)

        #Show WOE Table
        if show_woe == True:
            print(d)
            
    return newDF
iv_table = iv_woe(data = df2, target = 'is_bad')

def indicator(value):
    if value < 0.02 :
        return 'Not useful for prediction'
    elif value >= 0.02 and value < 0.1:
        return 'Weak predictive Power'
    elif value >=0.1 and value < 0.3:
        return 'Medium predictive Power'
    elif value >=0.3 and value < 0.5:
        return 'Strong predictive Power'
    else:
        return 'Suspicious Predictive Power or too good to be true'
    
iv_table['indicator'] = iv_table['IV'].apply(indicator)
iv_table.reset_index(drop=True)
iv_table
from sklearn.model_selection import train_test_split

X = df1.drop(['is_bad'], axis=1)

X.head()
Y = df1['is_bad']

Y.head()
scaler = StandardScaler()

X[scaling_features] = scaler.fit_transform(X[scaling_features])

X.head()
# Splitting the data into train and test

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7, test_size=0.3, random_state=100)
# Tried SMOTE but it did not perform well
#Performing Oversampling on the data using SMOTE function
#from imblearn.over_sampling import SMOTE
#from collections import Counter
#sm = SMOTE(random_state=42)
#X_res, y_res = sm.fit_resample(X_train, Y_train)
# Splitting the data into train and test
#X_train, X_test, Y_train, Y_test = train_test_split(X_res, y_res, train_size=0.7, test_size=0.3, random_state=100)
#Feature Scaling
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
from sklearn.feature_selection import RFE
rfe = RFE(logreg, 20)             # running RFE with 20 variables as output
rfe = rfe.fit(X_train, Y_train)
list(zip(X_train.columns, rfe.support_, rfe.ranking_))
col = X_train.columns[rfe.support_]
X_train.columns[~rfe.support_]
X_train_sm = sm.add_constant(X_train[col]).astype(float)
logm2 = sm.GLM(Y_train.astype(float),X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()
# Dropping highest insignificant feature(pub_rec_3) and train the model with remaining variables
col = col.drop('pub_rec_3', 1)
col
X_train_sm = sm.add_constant(X_train[col]).astype(float)
logm2 = sm.GLM(Y_train.astype(float),X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()
#Checking VIF
# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train[col].columns

vif['VIF'] = [variance_inflation_factor(X_train[col].astype(float).values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
# Dropping the variable purpose_cat_Other_Small_Business and training the model with remianing feature
col = col.drop('purpose_cat_Other_Small_Business', 1)
col
X_train_sm = sm.add_constant(X_train[col]).astype(float)
logm2 = sm.GLM(Y_train.astype(float),X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()
# Check VIF before dropping the insignificant features
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].astype(float).values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
# Dropping the variable purpose_cat_home improvement and training the model with remianing feature
col = col.drop('purpose_cat_home improvement', 1)
X_train_sm = sm.add_constant(X_train[col]).astype(float)
logm2 = sm.GLM(Y_train.astype(float),X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()
# Check VIF before dropping the insignificant features
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].astype(float).values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
# Dropping the variable pub_rec_2 and training the model with remianing feature
col = col.drop('pub_rec_2', 1)
X_train_sm = sm.add_constant(X_train[col]).astype(float)
logm2 = sm.GLM(Y_train.astype(float),X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()
# Check VIF before dropping the insignificant features
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].astype(float).values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
# Dropping the purpose_cat_medical and training the model with remianing feature
col = col.drop('purpose_cat_medical', 1)
X_train_sm = sm.add_constant(X_train[col]).astype(float)
logm2 = sm.GLM(Y_train.astype(float),X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()
# Check VIF before dropping the insignificant features
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].astype(float).values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
# Dropping the home_ownership_OTHER and training the model with remianing feature
col = col.drop('home_ownership_OTHER', 1)
X_train_sm = sm.add_constant(X_train[col]).astype(float)
logm2 = sm.GLM(Y_train.astype(float),X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()
# Check VIF before dropping the insignificant features
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].astype(float).values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
# Dropping the policy_code_PC3 and training the model with remianing feature
col = col.drop('policy_code_PC3', 1)
X_train_sm = sm.add_constant(X_train[col]).astype(float)
logm2 = sm.GLM(Y_train.astype(float),X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()
# Dropping the policy_code_PC3 and training the model with remianing feature
col = col.drop('home_ownership_OWN', 1)
X_train_sm = sm.add_constant(X_train[col]).astype(float)
logm2 = sm.GLM(Y_train.astype(float),X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()
y_train_pred = res.predict(X_train_sm).values.reshape(-1)
y_train_pred_final = pd.DataFrame({'is_bad':Y_train.values, 'is_bad_Prob':y_train_pred})
y_train_pred_final['Id'] = Y_train.index
y_train_pred_final['is_bad_Prob'] = y_train_pred
from sklearn import metrics
y_train_pred_final['is_bad_Prob'] = y_train_pred

# Creating new column 'predicted' with 1 if is_bad_Prob > 0.5 else 0
y_train_pred_final['predicted'] = y_train_pred_final.is_bad_Prob.map(lambda x: 1 if x > 0.5 else 0)
# Let's check the overall accuracy.
print(metrics.accuracy_score(y_train_pred_final.is_bad, y_train_pred_final.predicted))
# Let's take a look at the confusion matrix again 
confusion = metrics.confusion_matrix(y_train_pred_final.is_bad, y_train_pred_final.predicted )
confusion
# Let's check the overall accuracy.
metrics.accuracy_score(y_train_pred_final.is_bad, y_train_pred_final.predicted)
TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives
# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)
# Let us calculate specificity
TN / float(TN+FP)
# Calculate false postive rate
print(FP/ float(TN+FP))
# positive predictive value 
print (TP / float(TP+FP))
# Negative predictive value
print (TN / float(TN+ FN))
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
fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.is_bad, y_train_pred_final.is_bad_Prob, drop_intermediate = False )
draw_roc(y_train_pred_final.is_bad, y_train_pred_final.is_bad_Prob)
# Let's create columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.is_bad_Prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()
# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.is_bad, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)
# Let's plot accuracy sensitivity and specificity for various probabilities.
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()
y_train_pred_final['final_predicted'] = y_train_pred_final.is_bad_Prob.map( lambda x: 1 if x > 0.12 else 0)

y_train_pred_final.head()
# Let's check the overall accuracy.
metrics.accuracy_score(y_train_pred_final.is_bad, y_train_pred_final.final_predicted)
confusion2 = metrics.confusion_matrix(y_train_pred_final.is_bad, y_train_pred_final.final_predicted )
confusion2
TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives
# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)
# Let us calculate specificity
TN / float(TN+FP)
# Calculate false postive rate
print(FP/ float(TN+FP))
# Positive predictive value 
print (TP / float(TP+FP))
# Negative predictive value
print (TN / float(TN+ FN))
confusion = metrics.confusion_matrix(y_train_pred_final.is_bad, y_train_pred_final.predicted )
confusion
from sklearn.metrics import precision_score, recall_score
precision_score(y_train_pred_final.is_bad, y_train_pred_final.predicted)


recall_score(y_train_pred_final.is_bad, y_train_pred_final.predicted)
from sklearn.metrics import precision_recall_curve
p, r, thresholds = precision_recall_curve(y_train_pred_final.is_bad, y_train_pred_final.is_bad_Prob)
plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()
X_test_sm = X_test[col]

X_test_sm = sm.add_constant(X_test_sm)
y_test_pred = res.predict(X_test_sm)
y_test_pred[:10]
# Converting y_pred to a dataframe which is an array
y_pred_1 = pd.DataFrame(y_test_pred)

# Converting y_test to dataframe
y_test_df = pd.DataFrame(Y_test)
# Putting ID to index
y_test_df['Id'] = y_test_df.index

# Removing index for both dataframes to append them side by side 
y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)

# Appending y_test_df and y_pred_1
y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)

# Renaming the column 
y_pred_final= y_pred_final.rename(columns={ 0 : 'Is_bad_Prob'})
y_pred_final.sort_values(by ='Is_bad_Prob',ascending=False)
y_pred_final['final_predicted'] = y_pred_final.Is_bad_Prob.map(lambda x: 1 if x > 0.12 else 0)
# Let's check the overall accuracy.
metrics.accuracy_score(y_pred_final.is_bad, y_pred_final.final_predicted)
confusion2 = metrics.confusion_matrix(y_pred_final.is_bad, y_pred_final.final_predicted )
confusion2
TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives
# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)
# Let us calculate specificity
TN / float(TN+FP)
#from sklearn.preprocessing import StandardScaler
#z_scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
#X_train_z = z_scaler.fit_transform(X_train)
#X_test_z = z_scaler.transform(X_test)
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
pipeline_sgdlogreg = Pipeline([
    ('imputer', SimpleImputer(copy=False)), 
    ('scaler', StandardScaler(copy=False)),
    ('model', SGDClassifier(loss='log',random_state=1))
])
param_grid_sgdlogreg = {
    'model__alpha': [10**-6, 10**-2, 10**1],
    'model__penalty': ['l1', 'l2']
}
grid_sgdlogreg = GridSearchCV(estimator=pipeline_sgdlogreg, param_grid=param_grid_sgdlogreg, scoring='roc_auc', n_jobs=1, pre_dispatch=1, cv=5, verbose=1, return_train_score=False)
grid_sgdlogreg.fit(X_train, Y_train)
grid_sgdlogreg.best_score_
grid_sgdlogreg.best_params_
from sklearn.ensemble import RandomForestClassifier
pipeline_rfc = Pipeline([
    ('imputer', SimpleImputer(copy=False)),
    ('model', RandomForestClassifier(n_jobs=-1, random_state=1))
])
param_grid_rfc = {
    'model__n_estimators': [50,100,200,300] # The number of randomized trees to build
}
grid_rfc = GridSearchCV(estimator=pipeline_rfc, param_grid=param_grid_rfc, scoring='roc_auc', n_jobs=1, pre_dispatch=1, cv=5, verbose=1, return_train_score=False)
grid_rfc.fit(X_train, Y_train)
grid_rfc.best_score_
print('Cross-validated AUCROC scores')
print(grid_sgdlogreg.best_score_, '- Logistic regression')
print(grid_rfc.best_score_, '- Random forest')
param_grid_sgdlogreg = {
    'model__alpha': np.logspace(-4.5, 0.5, 11),
    'model__penalty': ['l1', 'l2']
}

print(param_grid_sgdlogreg)
grid_sgdlogreg = GridSearchCV(estimator=pipeline_sgdlogreg, param_grid=param_grid_sgdlogreg, scoring='roc_auc', n_jobs=1, pre_dispatch=1, cv=5, verbose=1, return_train_score=False)
grid_sgdlogreg.fit(X_train, Y_train)
grid_sgdlogreg.best_score_
grid_sgdlogreg.best_params_
from sklearn.metrics import roc_auc_score
y_score = grid_sgdlogreg.predict_proba(X_test)[:,1]
roc_auc_score(Y_test, y_score)
y_pred = grid_sgdlogreg.predict(X_test)
from imblearn.metrics import sensitivity_specificity_support
sensitivity, specificity, _ = sensitivity_specificity_support(Y_test, y_pred, average='binary')
print("Sensitivity: \t", round(sensitivity, 2), "\n", "Specificity: \t", round(specificity, 2), sep='')
pca = PCA()
pca.fit(X_train)
is_bad_pca = pca.fit_transform(X_train)
# plot feature variance
features = range(pca.n_components_)
cumulative_variance = np.round(np.cumsum(pca.explained_variance_ratio_)*100, decimals=4)
plt.figure(figsize=(175/20,100/20)) # 100 elements on y-axis; 175 elements on x-axis; 20 is normalising factor
plt.plot(cumulative_variance)
# create pipeline
PCA_VARS = 15
steps = [("pca", PCA(n_components=PCA_VARS)),
         ("logistic", SGDClassifier(class_weight='balanced',loss='log',penalty='l2'))
        ]
pipeline = Pipeline(steps)
# fit model
pipeline.fit(X_train, Y_train)

# check score on train data
pipeline.score(X_train, Y_train)
# predict bad loan on test data
y_pred = pipeline.predict(X_test)

# create onfusion matrix
cm = confusion_matrix(Y_test, y_pred)
print(cm)

# check sensitivity and specificity
sensitivity, specificity, _ = sensitivity_specificity_support(Y_test, y_pred, average='binary')
print("Sensitivity: \t", round(sensitivity, 2), "\n", "Specificity: \t", round(specificity, 2), sep='')

# check area under curve
y_pred_prob = pipeline.predict_proba(X_test)[:, 1]
print("AUC:    \t", round(roc_auc_score(Y_test, y_pred_prob),2))
# Getting the predicted values on the train set
y_train_pred = res.predict(X_train_sm)
y_train_pred[:10]
y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]
X_train.shape
y_train_pred_final = pd.DataFrame({'is_bad':Y_train.values, 'is_bad_prob':y_train_pred})
y_train_pred_final['ID'] = Y_train.index
y_train_pred_final.head()
{'model__alpha': 0.0031622776601683794, 'model__penalty': 'l2'}
# Final model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
num_trees = 200
max_depth = 3
clf = RandomForestClassifier(n_estimators=num_trees, max_depth=max_depth, n_jobs = 4, random_state=5)
clf.fit(X_train,Y_train)
pred_prob = clf.predict_proba(X_test)
pred = clf.predict(X_test)
auc = roc_auc_score(y_true = Y_test, y_score = pred_prob[:,1])
print('num_trees =',num_trees,'; depth=',max_depth,'; auc =',auc)
sensitivity, specificity, _ = sensitivity_specificity_support(Y_test, pred, average='binary')
print("Sensitivity: \t", round(sensitivity, 2), "\n", "Specificity: \t", round(specificity, 2), sep='')
print(X_test.columns)
#Final Model
# PCA
pca = PCA()

# logistic regression - the class weight is used to handle class imbalance - it adjusts the cost function
logistic = LogisticRegression(class_weight='balanced')

# create pipeline
steps = [("pca", pca),
         ("logistic", logistic)
        ]

# compile pipeline
pca_logistic = Pipeline(steps)

# hyperparameter space
params = {'pca__n_components': [16,17,18,19,20], 'logistic__C': [0.1,0.2,0.5, 1, 2, 3, 4, 5, 10], 'logistic__penalty': ['l1', 'l2']}

# create 5 folds
folds = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 4)

# create gridsearch object
lr_model = GridSearchCV(estimator=pca_logistic, cv=folds, param_grid=params, scoring='roc_auc')
# fit model
lr_model.fit(X_train, Y_train)
# print best hyperparameters
print("Best AUC: ", lr_model.best_score_)
print("Best hyperparameters: ", lr_model.best_params_)
from sklearn.metrics import roc_curve
# predict bad loan on test data
y_pred = lr_model.predict(X_test)

# create onfusion matrix
cm = confusion_matrix(Y_test, y_pred)
print(cm)

# check sensitivity and specificity
sensitivity, specificity, _ = sensitivity_specificity_support(Y_test, y_pred, average='binary')
print("Sensitivity: \t", round(sensitivity, 2), "\n", "Specificity: \t", round(specificity, 2), sep='')

# check area under curve
y_pred_prob = lr_model.predict_proba(X_test)[:, 1]
print("AUC:    \t", round(roc_auc_score(Y_test, y_pred_prob),2))

lr_accuracy = metrics.accuracy_score(Y_test, y_pred)
print("Accuracy:    \t", round(lr_accuracy,2))

lr_fpr, lr_tpr, thresholds = roc_curve(Y_test, y_pred_prob)

lr_roc_auc = metrics.auc(lr_fpr, lr_tpr)


#RF hyperparameter tuning
# PCA
pca = PCA()

# logistic regression - the class weight is used to handle class imbalance - it adjusts the cost function
RF = RandomForestClassifier(class_weight='balanced')

steps = [("pca", pca),
         ("RF", RF)
        ]

# compile pipeline
pca_RF = Pipeline(steps)

# hyperparameter space
params = {'RF__n_estimators': [50,100], 'RF__max_depth': [20,30],
    'RF__max_features': [5, 7]
         }

# create 5 folds
folds = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 4)

# create gridsearch object
rf_model = GridSearchCV(estimator=pca_RF, cv=folds, param_grid=params, scoring='roc_auc')
rf_model.fit(X_train,Y_train)
# print best hyperparameters
print("Best AUC: ", rf_model.best_score_)
print("Best hyperparameters: ", rf_model.best_params_)
# predict bad loan on test data
rf_y_pred = rf_model.predict(X_test)

# create onfusion matrix
rf_cm = confusion_matrix(Y_test, rf_y_pred)
print(cm)

# check sensitivity and specificity
sensitivity, specificity, _ = sensitivity_specificity_support(Y_test, rf_y_pred, average='binary')
print("Sensitivity: \t", round(sensitivity, 2), "\n", "Specificity: \t", round(specificity, 2), sep='')

# check area under curve
rf_y_pred_prob = rf_model.predict_proba(X_test)[:, 1]
print("AUC:    \t", round(roc_auc_score(Y_test, rf_y_pred_prob),2))

#print(metrics.accuracy_score(Y_test, rf_y_pred))


RF_accuracy = metrics.accuracy_score(Y_test, rf_y_pred)
print("Accuracy:    \t", round(RF_accuracy,2))

rf_fpr, rf_tpr, thresholds = roc_curve(Y_test, rf_y_pred_prob)

rf_roc_auc = metrics.auc(rf_fpr, rf_tpr)

print("RF AUCROC:    \t", round(rf_roc_auc,2))
# Doing parameter tuning,
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV

ada_boost = AdaBoostClassifier(n_estimators = 20, learning_rate = 0.2, random_state = 123)
gridparam ={
        'n_estimators': [100,200,500],
        'learning_rate': [0.2,0.5,1.0],
},
ab = GridSearchCV(ada_boost,cv=3,n_jobs=3, param_grid=gridparam)
ab.fit(X_train, Y_train)
ab_Y_pred = ab.predict(X_test)
ab_predict_proba =ab.predict_proba(X_test)[:,1] 

#Checking the accuracy,
ab_accuracy=ab.score(X_test, Y_test)
print(" Accuracy of AdaBoost  classifier: ",ab_accuracy )

sensitivity, specificity, _ = sensitivity_specificity_support(Y_test, ab_Y_pred, average='binary')
print("Sensitivity: \t", round(sensitivity, 2), "\n", "Specificity: \t", round(specificity, 2), sep='')
##Computing false and true positive rates
from sklearn.metrics import roc_curve 
import sklearn.metrics as metrics
ab_fpr, ab_tpr, thresholds = roc_curve(Y_test, ab_predict_proba) #AdaBoost Classifier
ab_roc_auc = metrics.auc(ab_fpr, ab_tpr)
import matplotlib.pyplot as plt
plt.figure()
##Creating the ROC,
plt.plot(ab_fpr, ab_tpr, color='blue',lw=2, label='ROC curve')
##Finding FPR and TPR,
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
##Splecifying the label and title,
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plot_title = 'ROC curve AUC: {0}'.format(ab_roc_auc)
plt.title(plot_title, size=15)
plt.show()
#compare the ROC curve between different models
plt.figure(figsize=(8,8))
plt.plot(lr_fpr, lr_tpr, label='Logistic regression')
plt.plot(ab_fpr, ab_tpr, label='Adaboost Classifier')
plt.plot(rf_fpr, rf_tpr, label='Randomforest Classifier')
#plt.plot(dt_fpr, dt_tpr, label='Decision Tree')
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='random', alpha=.8)
plt.xlim([0,1])
plt.ylim([0,1])
plt.xticks(np.arange(0,1.1,0.1))
plt.yticks(np.arange(0,1.1,0.1))
plt.grid()
plt.legend()
plt.axes().set_aspect('equal')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
score = [    (lr_accuracy, lr_roc_auc) ,
             (RF_accuracy, rf_roc_auc) ,
             #(dt_accuracy, dt_roc_auc) ,
             (ab_accuracy, ab_roc_auc)     ]
df_score = pd.DataFrame(score, columns = ['Accuracy' , 'AUC'], index=['LogisticRegression','RandomForest','AdaBoost']) 
df_score
y_prob = lr_model.predict_proba(X_test)
y_prob[:,1]
Y_test.head()
probabilities = pd.DataFrame()
probabilities['prob'] = y_prob[:,1]
probabilities['actual'] = np.array(Y_test)
probabilities['pred'] = np.array(y_pred)
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(Y_test,y_pred))
print(confusion_matrix(Y_test,y_pred))
probabilities = pd.DataFrame()
probabilities['pred_prob'] = y_prob[:,1]
probabilities['actual'] = np.array(Y_test)
probabilities['predicted'] = np.array(y_pred)
decile_df = probabilities
decile_df['decile'] = pd.qcut(decile_df['pred_prob'], 10, labels=np.arange(10, 0, -1))
lift_df = decile_df.groupby('decile')['pred_prob'].count().reset_index()
lift_df.rename({'pred_prob':'total'}, axis=1, inplace=True)
lift_df_pred = decile_df[decile_df['actual']==1].groupby('decile')['actual'].count().reset_index()
lift_df_final = lift_df.merge(lift_df_pred,on = 'decile')
lift_df_final = lift_df_final.sort_values(['decile'], ascending=False)
lift_df_final['cumresp'] = lift_df_final['actual'].cumsum()
lift_df_final['gain'] = 100*(lift_df_final['cumresp']/sum(lift_df_final['actual']))
lift_df_final['cumlift'] = lift_df_final['gain']/(lift_df_final['decile'].astype('int')*(100/10))
lift_df_final.plot.line(x='decile', y=['gain'])
df_score
