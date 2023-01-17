# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
test = pd.read_csv('../input/application_test.csv')
train = pd.read_csv('../input/application_train.csv')
bureau = pd.read_csv('../input/bureau.csv')
bureau_bal = pd.read_csv('../input/bureau_balance.csv')
cred_bal = pd.read_csv('../input/credit_card_balance.csv')
i_payments = pd.read_csv('../input/installments_payments.csv')
pos_cash_bal = pd.read_csv('../input/POS_CASH_balance.csv')
prev_app = pd.read_csv('../input/previous_application.csv')
sample_sub = pd.read_csv('../input/sample_submission.csv')
train.head()
train.shape
test.shape

test.head()
train['TARGET'].value_counts()
train['TARGET'].astype(int).plot.hist()
def missing_values(df):
    #number of missing values
    missing_val = df.isnull().sum()
    #percentage of missing value
    missing_val_perc = 100*df.isnull().sum()/len(df)
    #making a missing value table 
    mis_table = pd.concat([missing_val,missing_val_perc], axis = 1)
    mis_val_table = mis_table.rename( columns = {0: 'Missing Values', 1: 'Percentage' })
    #sort the table
    mis_val_table = mis_val_table[mis_val_table.iloc[:,1] != 0].sort_values('Percentage',ascending = False).round(1)
    #summary
    print("Your dataframe has "+ str(df.shape[1]) + " columns. Out of which "+str(mis_val_table.shape[0])+" has missing values")
    return mis_val_table
missing_values = missing_values(train)

missing_values.head(10)
train.dtypes.value_counts()
#number of unique classes in each column
train.select_dtypes('object').apply(pd.Series.nunique,axis=0)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le_count = 0


for col in train:
    if train[col].dtype == 'object':
        # If 2 or fewer unique categories
        if len(list(train[col].unique())) <= 2:
            # Train on the training data
            le.fit(train[col])
            # Transform both training and testing data
            train[col] = le.transform(train[col])
            test[col] = le.transform(test[col])
            
            # Keep track of how many columns were label encoded
            le_count += 1
print("%d columns were label encoded" %le_count)
    
#one-hot encoding
train = pd.get_dummies(train)
test = pd.get_dummies(test)

print("Train : "+str(train.shape))
print("Test :" + str(test.shape))
train_labels = train['TARGET']
train,test = train.align(test, join='inner',axis=1)
train['TARGET'] = train_labels

print("Train : "+str(train.shape))
print("Test :" + str(test.shape))
(train['DAYS_BIRTH']/-365).describe()
train['DAYS_EMPLOYED'].describe()
anom = train[ train['DAYS_EMPLOYED'] == 365243]
nom = train[train['DAYS_EMPLOYED'] != 365243]
print(len(anom))
#create anom flag
train['DAYS_EMPLOYED_ANOM'] = train['DAYS_EMPLOYED'] == 365243
train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)
train['DAYS_EMPLOYED'].plot.hist(title='DAYS EMPLOYED')
test['DAYS_EMPLOYED_ANOM'] = test['DAYS_EMPLOYED'] == 365243
test['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)
print(test['DAYS_EMPLOYED_ANOM'].sum())
correlations = train.corr()['TARGET'].sort_values()
print('Most positive correlations',correlations.tail(10))
print('Most negative correlations',correlations.head(10))


train['DAYS_BIRTH'] = abs(train['DAYS_BIRTH'])
train['DAYS_BIRTH'].corr(train['TARGET'])
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('ggplot')

plt.hist(train['DAYS_BIRTH'] / 365, edgecolor = 'k', bins = 25)
plt.title('Age of Client'); plt.xlabel('Age') ; plt.ylabel('Count');
plt.figure(figsize = (10,8))
sns.kdeplot(train.loc[train['TARGET']==0 , 'DAYS_BIRTH']/365 , label = 'Target = 0')
sns.kdeplot(train.loc[train['TARGET']==1 , 'DAYS_BIRTH']/365, label = 'Target=1')
plt.xlabel('Age(years)'); plt.ylabel
age_data = train[['DAYS_BIRTH','TARGET']]
age_data['YEARS'] = age_data['DAYS_BIRTH']/365

age_data['YEARS_BIN'] = pd.cut(age_data['YEARS'], bins = np.linspace(20,70,num = 11))
age_data.head()
age_group = age_data.groupby('YEARS_BIN').mean()
age_group
plt.figure(figsize=(8,8))

plt.bar(age_group.index.astype(str), 100*age_group['TARGET'])
plt.title('FAILURE TO PAY BY AGE GROUP')
plt.xticks(rotation = 70) ; plt.xlabel('AGE GROUP'); plt.ylabel('% failed to repay');
ext_data = train[['TARGET','EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','DAYS_BIRTH']]
ext_data_corrs = ext_data.corr()
ext_data_corrs
plt.figure(figsize = (10,8))
sns.heatmap(ext_data_corrs, annot= True,vmin= -0.25, vmax=0.6,cmap="YlGnBu")

plt.figure(figsize=(10,12))

for i,source in enumerate(['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']):
    plt.subplot(3,1,i+1)
    sns.kdeplot(train.loc[train['TARGET']==0, source], label = 'target==0')
    sns.kdeplot(train.loc[train['TARGET']==1, source], label = 'target ==1')
    plt.title('DISTRIBUTION OF %s BY TARGET VALUE'%source)
    plt.xlabel('%s'%source); plt.ylabel('Density');

plt.tight_layout(h_pad = 3.0)
# Copy the data for plotting
plot_data = ext_data.drop(columns = ['DAYS_BIRTH']).copy()

# Add in the age of the client in years
plot_data['YEARS'] = age_data['YEARS']

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
poly_features = train[['TARGET','EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','DAYS_BIRTH']]
poly_features_test = train[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','DAYS_BIRTH']]


from sklearn.preprocessing import Imputer
imputer = Imputer()
poly_target = poly_features['TARGET']
poly_features = poly_features.drop(columns = ['TARGET'])

poly_features = imputer.fit_transform(poly_features)
poly_features_test = imputer.transform(poly_features_test)

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree =3)

poly.fit(poly_features)

poly_features = poly.transform(poly_features)
poly_features_test = poly.transform(poly_features_test)

print('Polynomial Features shape: ', poly_features.shape)
poly.get_feature_names(input_features = ['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','DAYS_BIRTH'])[:15]
poly_features = pd.DataFrame(poly_features , columns = poly.get_feature_names(input_features = ['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','DAYS_BIRTH']))
poly_features['TARGET'] = poly_target
poly_corrs = poly_features.corr()['TARGET'].sort_values()
print(poly_corrs.head(10))
print(poly_corrs.tail(10))
poly_features_test = pd.DataFrame(poly_features_test, 
                                 columns = poly.get_feature_names(input_features = ['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','DAYS_BIRTH']))
poly_features['SK_ID_CURR'] = train['SK_ID_CURR']
train_poly = train.merge(poly_features,on='SK_ID_CURR', how='left')

poly_features_test['SK_ID_CURR'] = test['SK_ID_CURR']
test_poly = test.merge(poly_features_test, on = 'SK_ID_CURR', how ='left')

train_poly,test_poly = train_poly.align(test_poly, join = 'inner', axis = 1)

print('Training data with features shape', train_poly.shape)
print('Test data with features shape', test_poly.shape)
train_domain = train.copy()
test_domain = test.copy()
train_domain['CREDIT_INCOME_PERCENT'] = train_domain['AMT_CREDIT'] / train_domain['AMT_INCOME_TOTAL']
train_domain['ANNUITY_INCOME_PERCENT'] = train_domain['AMT_ANNUITY'] / train_domain['AMT_INCOME_TOTAL']
train_domain['CREDIT_TERM'] = train_domain['AMT_ANNUITY'] / train_domain['AMT_CREDIT']
train_domain['DAYS_EMPLOYED_PERCENT'] = train_domain['DAYS_EMPLOYED'] / train_domain['DAYS_BIRTH']

test_domain['CREDIT_INCOME_PERCENT'] = test_domain['AMT_CREDIT'] / test_domain['AMT_INCOME_TOTAL']
test_domain['ANNUITY_INCOME_PERCENT'] = test_domain['AMT_ANNUITY'] / test_domain['AMT_INCOME_TOTAL']
test_domain['CREDIT_TERM'] = test_domain['AMT_ANNUITY'] / test_domain['AMT_CREDIT']
test_domain['DAYS_EMPLOYED_PERCENT'] = test_domain['DAYS_EMPLOYED'] / test_domain['DAYS_BIRTH']
plt.figure(figsize=(10,12))

for i,source in enumerate(['CREDIT_INCOME_PERCENT','ANNUITY_INCOME_PERCENT','CREDIT_TERM','DAYS_EMPLOYED_PERCENT']):
    plt.subplot(4,1,i+1)
    sns.kdeplot(train_domain.loc[train['TARGET']==0, source], label = 'target==0')
    sns.kdeplot(train_domain.loc[train['TARGET']==1, source], label = 'target ==1')
    plt.title('DISTRIBUTION OF %s BY TARGET VALUE'%source)
    plt.xlabel('%s'%source); plt.ylabel('Density');

plt.tight_layout(h_pad = 3.0)
