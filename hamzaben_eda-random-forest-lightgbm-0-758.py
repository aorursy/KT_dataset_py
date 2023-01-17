import pandas as pd
import numpy as np
import os
import re
from scipy import stats
from scipy.stats import norm, skew
import matplotlib
import matplotlib.pyplot as plt # for plotting
%matplotlib inline
import seaborn as sns # for making plots with seaborn
color = sns.color_palette()
import warnings
warnings.filterwarnings('ignore') # Suppress warnings 
# List files available
print(os.listdir("../input/"))
# importing the datasets into Pandas dataframes
data_test = pd.read_csv('../input/application_test.csv')
df_test = data_test.copy()
data_train = pd.read_csv('../input/application_train.csv')
df_train = data_train.copy()
(df_train.shape, df_test.shape)
df_train.head()
df_test.head()
df_train.columns
# importing the datasets into Pandas dataframes
bureau_balance = pd.read_csv('../input//bureau_balance.csv')
bureau = pd.read_csv('../input//bureau.csv')
# left joining the dataset on='SK_ID_BUREAU'(left=bureau, right=bureau_balance)
df_bureau_joined = bureau.merge(bureau_balance, 
                                on='SK_ID_BUREAU', 
                                how='left')
df_bureau_joined.head()
target = df_train['TARGET']  #target variable
target.value_counts()
plt.ylabel('Instances')
plt.xlabel('TARGET value')
plt.title('Target Variable Distribution (Training Dataset)')
sns.countplot(x='TARGET', data=df_train);
target.value_counts()[0]/(target.value_counts()[0]+target.value_counts()[1])
# Plot the distribution of ages in years
age_years = df_train['DAYS_BIRTH'] / -365
plt.figure(figsize=(10,6))
plt.style.use('seaborn-colorblind')
plt.grid(True, alpha=0.5)
plt.hist(age_years, edgecolor = 'k', bins = 25)
plt.xlim(17,73)
plt.ylim(0,20000)
(mu, sigma) = norm.fit(age_years)
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.title('Distribution of Age of Client (in Years)')
plt.xlabel('Age (years)')
plt.ylabel('Count in Dataset');
plt.figure(figsize=(10,6))
plt.style.use('seaborn-colorblind')
plt.grid(True, alpha=0.5)
sns.kdeplot(df_train.loc[df_train['TARGET'] == 0, 'DAYS_BIRTH'] / -365, label = 'Repaid Loan')
sns.kdeplot(df_train.loc[df_train['TARGET'] == 1, 'DAYS_BIRTH'] / -365, label = 'Not Repaid Loan')
plt.xlabel('Age (years)')
plt.ylabel('Density')
plt.title('Distribution of Age of Client (in Years)');
# Age information into a separate dataframe
age_data = df_train[['TARGET', 'DAYS_BIRTH']]
age_data['YEARS_BIRTH'] = age_data.loc[:,'DAYS_BIRTH'].copy() / -365
age_data['YEARS_BINNED'] = pd.cut(age_data['YEARS_BIRTH'], bins = np.linspace(20, 70, num = 11))
# Group by the bin and calculate averages
age_groups  = age_data.groupby('YEARS_BINNED').mean()
age_groups.drop(['DAYS_BIRTH'], axis=1, inplace=True)
plt.figure(figsize=(10,6))
plt.style.use('seaborn-colorblind')
plt.grid(True, alpha=0.5)
plt.bar(age_groups.index.astype(str), 100 * age_groups['TARGET'])
plt.xticks(rotation = 75)
plt.xlabel('Age Group (years)')
plt.ylabel('Failure to Repay (%)')
plt.title('Failure to Repay by Client\'s Age Range');
df_train['NAME_CONTRACT_TYPE'].value_counts()
# Count Plot (a.k.a. Bar Plot)
plt.figure(figsize=(10,5))
plt.ylabel('Count')
plt.title('Contract Types by Target Value')
sns.countplot(x='NAME_CONTRACT_TYPE', hue='TARGET', data=df_train);
plt.figure(figsize=(14,5))
plt.style.use('seaborn-colorblind')
plt.grid(True, alpha=0.5)
sns.kdeplot(df_train.loc[df_train['TARGET'] == 0, 'AMT_CREDIT'], 
            label = 'Repaid Loan')
sns.kdeplot(df_train.loc[df_train['TARGET'] == 1, 'AMT_CREDIT'], 
            label = 'Not Repaid Loan')
plt.xlabel('Amount of Credit')
plt.xticks(np.arange(0, 5000000, 500000))
plt.ylabel('Density')
plt.title('Distribution of Amount of Credit by Target Value');
plt.figure(figsize=(14,8))
plt.style.use('seaborn-colorblind')
plt.grid(True, alpha=0.5)
sns.kdeplot(df_train.loc[df_train['TARGET'] == 0, 'DAYS_EMPLOYED'], 
            label = 'Repaid Loan')
sns.kdeplot(df_train.loc[df_train['TARGET'] == 1, 'DAYS_EMPLOYED'], 
            label = 'Not Repaid Loan')
plt.xlabel('Amount of Credit')
plt.ylabel('Density')
plt.title('Distribution of Amount of Credit by Target Value');
# Find correlations with the target and sort
correlations = df_train.corr()['TARGET'].sort_values()
print('Most Positive Correlations: \n', correlations.tail(5))
print('\nMost Negative Correlations: \n', correlations.head(5))
df_train_corr = df_train[['TARGET','DAYS_BIRTH','REGION_RATING_CLIENT_W_CITY','REGION_RATING_CLIENT',
                          'DAYS_LAST_PHONE_CHANGE','FLOORSMAX_AVG','DAYS_EMPLOYED','EXT_SOURCE_1', 
                          'EXT_SOURCE_2', 'EXT_SOURCE_3']].copy()
# Calculate correlations
corr = df_train_corr.corr()
# Heatmap
plt.figure(figsize=(15,8))
sns.heatmap(corr, annot=True, linewidths=.2, cmap="YlGnBu");
# Align the training and testing data, keep only columns present in both dataframes
df_train = df_train.drop('TARGET', axis=1) #drop target variable from training dataset
df_train, df_test = df_train.align(df_test, join = 'inner', axis = 1)
df_train.shape, target.shape, df_test.shape
#assing an extra variable to training and testing dataset before joining them
df_train['training_set'] = True 
df_test['training_set'] = False
df_full = pd.concat([df_train, df_test]) #concatenate both dataframes
df_full = df_full.drop('SK_ID_CURR', axis=1) #drop SK_ID_CURR variable
df_full.shape
print('Size of Full dataset df_full is: {}'.format( df_full.shape))
from sklearn.preprocessing import LabelEncoder
# Create a label encoder object
le = LabelEncoder()
# let's break down the columns by their type (i.e. int64, float64, object)
df_full.dtypes.value_counts()
le_count = 0
for col in df_full.columns[1:]:
    if df_full[col].dtype == 'object':
        if len(list(df_full[col].unique())) <= 2:
            le.fit(df_full[col])
            df_full[col] = le.transform(df_full[col])
            le_count += 1
print('{} columns were label encoded.'.format(le_count))
# convert rest of categorical variable into dummy
df_full = pd.get_dummies(df_full)
print('Size of Full Encoded Dataset', df_full.shape)
df_train['TARGET'] = target
df_doc_corr = df_train[['TARGET','FLAG_DOCUMENT_2','FLAG_DOCUMENT_3','FLAG_DOCUMENT_4',
                        'FLAG_DOCUMENT_5','FLAG_DOCUMENT_6','FLAG_DOCUMENT_7','FLAG_DOCUMENT_8', 
                        'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 
                        'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16',
                        'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20',
                        'FLAG_DOCUMENT_21']].copy()
# Find correlations with the target and sort
correlations = df_doc_corr.corr()['TARGET'].sort_values()
print('Most Positive Correlations: \n', correlations.tail(5))
print('\nMost Negative Correlations: \n', correlations.head(5))
# Calculate correlations
corr = df_doc_corr.corr()
# Heatmap
plt.figure(figsize=(20,10))
sns.heatmap(corr, annot=True, linewidths=.2, cmap="YlGnBu");
df_full = df_full.drop(['FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 
                        'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 
                        'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 
                        'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15'], axis=1)
def missing_val_ratio(df):
    perc_na = (df.isnull().sum()/len(df))*100
    ratio_na = perc_na.sort_values(ascending=False)
    missing_data_table = pd.DataFrame({'% of Total Values' :ratio_na})
    return missing_data_table
df_full_miss = missing_val_ratio(df_full)
df_full_miss.head(20)
plt.figure(figsize=(10,5))
plt.style.use('seaborn-colorblind')
plt.grid(True, alpha=0.5)
x = df_full_miss['% of Total Values']
x.hist(align='left', bins= 15)
plt.xticks(np.arange(0, 75, 5))
plt.xlabel('% of Missing Values')
plt.ylabel('Count (Features)')
plt.title('Distribution of % of Missing Values in Dataset Features');
plt.show();
from sklearn.preprocessing import MinMaxScaler, Imputer
imputer = Imputer(strategy = 'median') # Median imputation of missing values
scaler = MinMaxScaler(feature_range = (0, 1)) # Scale each feature to 0-1
for column in df_full.columns:
    df_full[[column]] = imputer.fit_transform(df_full[[column]])
    df_full[[column]] = scaler.fit_transform(df_full[[column]])
df_full_miss = missing_val_ratio(df_full)
df_full_miss.head(5)
df_train = df_full[df_full['training_set']==True]
df_train = df_train.drop('training_set', axis=1)
df_test = df_full[df_full['training_set']==False]
df_test = df_test.drop('training_set', axis=1)
print('Size of training_set: ', df_train.shape)
print('Size of testing_set: ', df_test.shape)
print('Size of target: ', target.shape)
print('Size of original data_test: ', data_test.shape)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
# Make the model with the specified regularization parameter
log_reg = LogisticRegression(random_state = 42)
# Train on the training data
log_reg.fit(df_train, target)
log_reg_predict = log_reg.predict(df_test)
#roc_auc_score(y_test, log_reg_predict)
from sklearn.ensemble import RandomForestClassifier
# Train on the training data
rf_model = RandomForestClassifier() 
rf_model.fit(df_train, target)
random_forest_pred = rf_model.predict_proba(df_test)[:, 1]
from lightgbm import LGBMClassifier
LGB_clf = LGBMClassifier(n_estimators=100, 
                         boosting_type='gbdt', 
                         objective='binary', 
                         metric='binary_logloss')
LGB_clf.fit(df_train, target)
LGB_clf_pred = LGB_clf.predict_proba(df_test)[:, 1]
# Train on the training data
opt_rf_model = RandomForestClassifier(n_estimators=200, 
                                      min_samples_split=10, 
                                      min_samples_leaf=5, 
                                      n_jobs=-1, 
                                      random_state=42) 
opt_rf_model.fit(df_train, target)
opt_RF_pred = opt_rf_model.predict_proba(df_test)[:, 1]
LGBM_clf = LGBMClassifier(n_estimators = 10000,
                          learning_rate = 0.02,
                          min_data_in_leaf = 30,
                          num_leaves = 31,
                          boosting_type='gbdt', 
                          objective='binary', 
                          metric='binary_logloss')
LGBM_clf.fit(df_train, target)
LGBM_clf_pred = LGBM_clf.predict_proba(df_test)[:, 1]
# Make a submission dataframe
submission = data_test[['SK_ID_CURR']]
submission['TARGET'] = LGBM_clf_pred

# Save the submission file
submission.to_csv("light_gbm_preds.csv", index=False)
d = {'1. Logistic Regression': [0.504], 
     '2. Default Random Forest': [0.610], 
     '4. Default Light GBM': [0.745], 
     '3. Tuned Random Forest': [0.723],
     '5. Tuned Light GBM': [0.758]}
d_i = ['ROC AUC Score']
df_results = pd.DataFrame(data=d, index = d_i)
df_results = df_results.transpose()
df_results
#import scikitplot as skplt
#import matplotlib.pyplot as plt
#skplt.metrics.plot_roc(y_test, LGBM_clf.predict_proba(X_test), figsize=(12,6))
#plt.show();
