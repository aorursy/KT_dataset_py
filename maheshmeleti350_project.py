# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
app_train = pd.read_csv('../input/application_train.csv')
print('Training data shape: ',app_train.shape)
app_train.head()
app_test = pd.read_csv('../input/application_test.csv')
print('Testing data shape: ', app_test.shape)
app_test.head()
app_train['TARGET'].value_counts()
app_train['TARGET'].astype(int).plot.hist()
#function to calculate missing values by columns# function
def missing_values_table(df):
    #total missing values
    miss_val = df.isnull().sum()
    
    #percentage of missing values
    miss_val_percent = 100*df.isnull().sum()/len(df)
    
    #Make a table with the results
    miss_val_table = pd.concat([miss_val,miss_val_percent],axis=1)

    #rename the columns
    miss_val_table_ren_columns = miss_val_table.rename(columns={0:'Missing Values',
                                                                 1: '% of Total Values'})
    #sort the table by percent of missing descending
    miss_val_table_ren_columns = miss_val_table_ren_columns[
        miss_val_table_ren_columns.iloc[:,1]!=0].sort_values(
    '% of Total Values', ascending=False).round(1)
    
    #print same summary information
    print("Your selected dataframe has "+str(df.shape[1]) +
         "columns.\n" 
         "There are "+str(miss_val_table_ren_columns.shape[0])+
         "columns that have missing values.")
    return miss_val_table_ren_columns
missing_values = missing_values_table(app_train)
missing_values.head(20)
app_train.dtypes.value_counts()
# number of unique classes in each object column
pd.DataFrame(app_train.select_dtypes('object').apply(pd.Series.nunique,axis=0))
# Create a label encoder object
le = LabelEncoder()
le_count=0

#Iterate through the columns
for col in app_train:
    if app_train[col].dtype == 'object':
        #if 2 or fewer unique categories
        if len(list(app_train[col].unique())) <= 2:
            le.fit(app_train[col])
            app_train[col] = le.transform(app_train[col])
            app_test[col] = le.transform(app_test[col])
            
            le_count +=1
print('%d columns were label encoded. '% le_count)
# one-hot encoding of categorical variables
app_train = pd.get_dummies(app_train)
app_test =pd.get_dummies(app_test)

print('Training Features shape: ',app_train.shape)
print('Test Features shape: ',app_test.shape)
train_labels = app_train['TARGET']

# Align the training and testing data, keep only columns present in both data frames

app_train,app_test = app_train.align(app_test,join='inner',axis=1)

app_train['TARGET'] = train_labels

print('Training Features shape: ',app_train.shape)
print('Testing Features shape: ',app_test.shape)
(app_train['DAYS_BIRTH']/-365).describe()
app_train['DAYS_EMPLOYED'].describe()
app_train['DAYS_EMPLOYED'].plot.hist(title='Days Employmnet Histogram')
plt.xlabel('Days Employment')
anom = app_train[app_train['DAYS_EMPLOYED']==365243]
non_anom = app_train[app_train['DAYS_EMPLOYED']!=365243]
print('The non-anomalies default on %0.2f%% of loans' %(100*non_anom['TARGET'].mean()))

print('The anomalies default on %0.2f%% of loans'%(100*anom['TARGET'].mean()))
print('There are %d anomalous days of employment'%len(anom))
#app_train[app_train['DAYS_EMPLOYED']>0 and app_train['DAYS_EMPLOYED']<365243]
#app_train['DAYS_EMPLOYED']>0 & app_train['DAYS_EMPLOYED']<365243
app_train[(app_train.DAYS_EMPLOYED>0)]['DAYS_EMPLOYED'].plot.hist() #== app_train[(app_train.DAYS_EMPLOYED<365243)]
# Create an anomalous flag column
app_train['DAYS_EMPLOYED_ANOM'] = app_train["DAYS_EMPLOYED"] == 365243

# Replace the anomalous values with nan
app_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)

app_train['DAYS_EMPLOYED'].plot.hist(title = 'Days Employment Histogram');
plt.xlabel('Days Employment');
app_test['DAYS_EMPLOYED_ANOM'] = app_test["DAYS_EMPLOYED"] == 365243
app_test["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)

print('There are %d anomalies in the test data out of %d entries' % (app_test["DAYS_EMPLOYED_ANOM"].sum(), len(app_test)))
correlations = app_train.corr()['TARGET'].sort_values()

print("Most positive correlations:\n",correlations.tail(15))
print("Most negative correlations:\n",correlations.head(15))
app_train['DAYS_BIRTH'] = abs(app_train['DAYS_BIRTH'])
app_train['DAYS_BIRTH'].corr(app_train['TARGET'])
plt.style.use('fivethirtyeight')

plt.hist(app_train['DAYS_BIRTH']/365,edgecolor='k',bins=25)
plt.title('Age of Client');plt.xlabel('AGE(years)');plt.ylabel('Count');
plt.figure(figsize=(10,8))

# KDE plot of loans that were 

sns.kdeplot(app_train.loc[app_train['TARGET']==0,'DAYS_BIRTH']/365, label='target==0')

# KDE plot of loans which were not repaid on time
sns.kdeplot(app_train.loc[app_train['TARGET']==1,'DAYS_BIRTH']/365,
           label='target==1')

plt.xlabel('Age (years)');plt.ylabel('Density');plt.title('Distribution of Ages');
age_data = app_train[['TARGET','DAYS_BIRTH']]
age_data['YEARS_BIRTH'] = age_data['DAYS_BIRTH']/365

#Bin the age data
age_data['YEARS_BINNED'] = pd.cut(age_data['YEARS_BIRTH'],
                                 bins = np.linspace(20,70,num=11))
age_data.head(20)
# group by the bin and calculat averages
age_groups = age_data.groupby('YEARS_BINNED').mean()
age_groups
plt.figure(figsize=(8,8))

# Graph the age bins and the average of the target as a bar plot
plt.bar(age_groups.index.astype(str),100*age_groups['TARGET'])

plt.xticks(rotation=75);plt.xlabel('Age Group (years)'); plt.ylabel('Failure to replay (%)')
plt.title('Failure to Repay by age group')

# most negatively correlated
ext_data = app_train[['TARGET','EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3',
                     'DAYS_BIRTH']]
ext_data_corrs = ext_data.corr()
ext_data_corrs
plt.figure(figsize=(8,6))

#heat map of correlations
sns.heatmap(ext_data_corrs,cmap=plt.cm.RdYlBu_r,vmin=-0.25,annot=True,
           vmax=0.6,)
plt.title('Correlation Heatmap');
plt.figure(figsize=(10,12))

for i,source in enumerate(['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']):
    #create a new plot for each source
    plt.subplot(3,1,i+1)
    
    #plot repaid loans
    sns.kdeplot(app_train.loc[app_train['TARGET']==0,source],label='target==0')
    
    #plot loans not repaid
    sns.kdeplot(app_train.loc[app_train['TARGET']==1,source],label='target==1')
    
    #label the plot
    plt.title('Distribution of %s by Target Value'%source)
    plt.xlabel('%s'%source);plt.ylabel('Density');
plt.tight_layout(h_pad=2.5)
# Copy the data for plotting
plot_data = ext_data.drop(columns = ['DAYS_BIRTH']).copy()

# Add in the age of the client in years
plot_data['YEARS_BIRTH'] = age_data['YEARS_BIRTH']

# Drop na values and limit to first 100000 rows
plot_data = plot_data.dropna().loc[:100000, :]

# Function to calculate correlation coefficient between two columns
def corr_func(x, y, **kwargs):
    r = np.corrcoef(x, y)[0][1]
    ax = plt.gca()
    ax.annotate("r = {:.2f}".format(r),
                xy=(.2, .8), xycoords=ax.transAxes,
                size = 20)
# Create the pairgrid object
grid = sns.PairGrid(data = plot_data, size = 3, diag_sharey=False,
                    hue = 'TARGET', 
                    vars = [x for x in list(plot_data.columns) if x != 'TARGET'])

# Upper is a scatter plot
grid.map_upper(plt.scatter, alpha = 0.2)

# Diagonal is a histogram
grid.map_diag(sns.kdeplot)

# Bottom is density plot
grid.map_lower(sns.kdeplot, cmap = plt.cm.OrRd_r);

plt.suptitle('Ext Source and Age Features Pairs Plot', size = 32, y = 1.05);
"r={:.2f}".format(0.023434)
# Make a new dataframe for polynomila features
poly_features = app_train[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_2',
                          'DAYS_BIRTH','TARGET']]
poly_features_test = app_test[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_2',
                          'DAYS_BIRTH']]

#imputer for handling missing values
from sklearn.preprocessing import Imputer
imputer = Imputer(strategy='median')

poly_target = poly_features['TARGET']
poly_features = poly_features.drop(columns=['TARGET'])

#need to impute missing values
poly_features = imputer.fit_transform(poly_features)
poly_features_test = imputer.fit_transform(poly_features_test)

from sklearn.preprocessing import PolynomialFeatures

poly_transformer = PolynomialFeatures(degree=3)
#Train the polynomial features
#poly_transformer.fit(poly_features)

#Transform the features
poly_features = poly_transformer.fit_transform(poly_features)
poly_features_test = poly_transformer.fit_transform(poly_features_test)
print("Plynomial Features shape: ",poly_features.shape)
poly_transformer.get_feature_names(input_features = ['EXT_SOURCE_1','EXT_SOURCE_2',
                                                    'EXT_SOURCE_3','DAYS_BIRTH'])[:15]
poly_features = pd.DataFrame(poly_features,columns = poly_transformer.get_feature_names(input_features = ['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','DAYS_BIRTH']))
#Add in the target
poly_features['TARGET'] = poly_target

#find the correlations with the target

poly_corrs = poly_features.corr()['TARGET'].sort_values()

#Display most negative and most positive
print(poly_corrs.head(10))
print()
print(poly_corrs.tail(5))

# Put test features into dataframe
poly_features_test = pd.DataFrame(poly_features_test, 
                                  columns = poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2', 
                                                                                'EXT_SOURCE_3', 'DAYS_BIRTH']))

# Merge polynomial features into training dataframe
poly_features['SK_ID_CURR'] = app_train['SK_ID_CURR']
app_train_poly = app_train.merge(poly_features, on = 'SK_ID_CURR', how = 'left')

# Merge polnomial features into testing dataframe
poly_features_test['SK_ID_CURR'] = app_test['SK_ID_CURR']
app_test_poly = app_test.merge(poly_features_test, on = 'SK_ID_CURR', how = 'left')

# Align the dataframes
app_train_poly, app_test_poly = app_train_poly.align(app_test_poly, join = 'inner', axis = 1)

# Print out the new shapes
print('Training data with polynomial features shape: ', app_train_poly.shape)
print('Testing data with polynomial features shape:  ', app_test_poly.shape)
app_train_domain = app_train.copy()
app_test_domain = app_test.copy()

app_train_domain['CREDIT_INCOME_PERCENT'] = app_train_domain['AMT_CREDIT']/app_train_domain['AMT_INCOME_TOTAL']
app_train_domain['ANNUITY_INCOME_PERCENT'] = app_train_domain['AMT_ANNUITY']/app_train_domain['AMT_INCOME_TOTAL']
app_train_domain['CREDIT_TERM'] = app_train_domain['AMT_ANNUITY']/app_train_domain['AMT_CREDIT']
app_train_domain['DAYS_EMPLOYED_PERCENT'] = app_train_domain['DAYS_EMPLOYED']/app_train_domain['DAYS_BIRTH']
app_test_domain['CREDIT_INCOME_PERCENT'] = app_test_domain['AMT_CREDIT'] / app_test_domain['AMT_INCOME_TOTAL']
app_test_domain['ANNUITY_INCOME_PERCENT'] = app_test_domain['AMT_ANNUITY'] / app_test_domain['AMT_INCOME_TOTAL']
app_test_domain['CREDIT_TERM'] = app_test_domain['AMT_ANNUITY'] / app_test_domain['AMT_CREDIT']
app_test_domain['DAYS_EMPLOYED_PERCENT'] = app_test_domain['DAYS_EMPLOYED'] / app_test_domain['DAYS_BIRTH']
plt.figure(figsize=(12,20))
#iterate through the new features
for i,feature in enumerate(['CREDIT_INCOME_PERCENT', 'ANNUITY_INCOME_PERCENT', 'CREDIT_TERM', 'DAYS_EMPLOYED_PERCENT']):
    plt.subplot(4,1,i+1)
    
    #plot repaid loans
    sns.kdeplot(app_train_domain.loc[app_train_domain['TARGET']==0,feature],
               label='target==0')
    #plot loans that were not paid
    sns.kdeplot(app_train_domain.loc[app_train_domain['TARGET']==1,feature],
               label='target==1')
    #label the plots
    plt.title('Distribution of %s by Target Value'%feature)
    plt.xlabel('%s'%feature);plt.ylabel('Density')
plt.tight_layout(h_pad=2.5)
from sklearn.preprocessing import MinMaxScaler, Imputer

# Drop the target from the training data
if 'TARGET' in app_train:
    train = app_train.drop(columns=['TARGET'])
else:
    train = app_train.copy()
    
#feature names
features = list(train.columns)

#copy of the testing data
test = app_test.copy()

imputer = Imputer(strategy='median')

scaler = MinMaxScaler(feature_range=(0,1))

imputer.fit(train)
train = imputer.transform(train)
test = imputer.transform(app_test)

#Repeat with the scaler
scaler.fit(train)
train = scaler.transform(train)
test = scaler.transform(test)

print('Training data shape: ', train.shape)
print('Testing data shape: ', test.shape)

a = np.array([[1,2],[4,5]])
b = np.array([[6,7],[8,9]])

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(C=0.0001)

log_reg.fit(train,train_labels)
log_reg_pred = log_reg.predict_proba(test)[:,1]
submit = app_test[['SK_ID_CURR']]
submit['TARGET'] = log_reg_pred

submit.head()
submit.to_csv('log_reg_baseline.csv', index = False)

