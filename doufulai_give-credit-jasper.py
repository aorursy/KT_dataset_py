# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

%matplotlib inline
dict_path  = '/kaggle/input/GiveMeSomeCredit/Data Dictionary.xls'
test_path  = '/kaggle/input/GiveMeSomeCredit/cs-test.csv'
train_path = '/kaggle/input/GiveMeSomeCredit/cs-training.csv'
sample     = '/kaggle/input/GiveMeSomeCredit/sampleEntry.csv'
pd.set_option('display.max_colwidth', -1)
data_info = pd.read_excel(dict_path,header=1)
data_info
train_df = pd.read_csv(train_path)
test_df  = pd.read_csv(test_path)
train_null_sums = train_df.isnull().sum()
test_null_sums = test_df.isnull().sum()
print(f'train_null_sums: {train_null_sums}')
print(f'='*50)
print(f'test_null_sums: {test_null_sums}')
# percentage of missing values
print(f"[MonthlyIncome] missing values percentage: {train_df['MonthlyIncome'].isnull().sum()/len(train_df)*100:.2f}%")
print(f"[NumberOfDependents] missing values percentage: {train_df['NumberOfDependents'].isnull().sum()/len(train_df)*100:.2f}%")
# drop train_df unwanted col
train_df.drop('Unnamed: 0',axis = 1,inplace=True)

# drop test_df unwanted col
test_df.drop('Unnamed: 0',axis = 1,inplace=True)
train_df.describe()
fig,[ax1,ax2,ax3]=plt.subplots(1,3,figsize=(25,6))
sns.distplot(train_df['RevolvingUtilizationOfUnsecuredLines'],ax=ax1)

target_0 = train_df.loc[train_df['SeriousDlqin2yrs'] == 0]
target_1 = train_df.loc[train_df['SeriousDlqin2yrs'] == 1]

sns.distplot(target_0[['RevolvingUtilizationOfUnsecuredLines']], hist=False ,ax=ax2, label='No default', axlabel='RevolvingUtilizationOfUnsecuredLines')
sns.distplot(target_1[['RevolvingUtilizationOfUnsecuredLines']], hist=False ,ax=ax2, label='Default')

sns.boxplot(y='RevolvingUtilizationOfUnsecuredLines',data=train_df,ax=ax3)
fig,[ax1,ax2,ax3]=plt.subplots(1,3,figsize=(25,6))
sns.distplot(train_df.loc[train_df['RevolvingUtilizationOfUnsecuredLines']<1,'RevolvingUtilizationOfUnsecuredLines'],ax=ax1)
sns.distplot(train_df.loc[(train_df['RevolvingUtilizationOfUnsecuredLines']>=1)&(train_df['RevolvingUtilizationOfUnsecuredLines']<10),'RevolvingUtilizationOfUnsecuredLines'],ax=ax2)
sns.distplot(train_df.loc[train_df['RevolvingUtilizationOfUnsecuredLines']>=10,'RevolvingUtilizationOfUnsecuredLines'],ax=ax3)
rev_below_1 = sum(train_df['RevolvingUtilizationOfUnsecuredLines']<1)/len(train_df)*100
rev_1_to_10 = sum((train_df['RevolvingUtilizationOfUnsecuredLines']>=1)&(train_df['RevolvingUtilizationOfUnsecuredLines']<10))/len(train_df)*100
rev_above_10 = sum(train_df['RevolvingUtilizationOfUnsecuredLines']>=10)/len(train_df)*100

print(f'rev_below_1: {rev_below_1:.2}%')
print(f'rev_1_to_10: {rev_1_to_10:.2}%')
print(f'rev_above_10: {rev_above_10:.2}%')
default_count_rev = {}
for i in range(12):
    default_count_rev[i] = sum(train_df.loc[train_df['RevolvingUtilizationOfUnsecuredLines']<i,'SeriousDlqin2yrs'] == 1)/len(train_df)
default_count_rev
x, y = zip(*default_count_rev.items())
plt.plot(x, y)
plt.xlabel('Threshold t')
plt.ylabel('Percentage of defaulters under threshold t')
plt.title('Percentage of defaulters under different thresholds')
plt.show()
print(f"Percentage of removal: {sum(train_df['RevolvingUtilizationOfUnsecuredLines']>9)/len(train_df)*100:.2f}%")
fig,[ax1,ax2,ax3]=plt.subplots(1,3,figsize=(25,6))
sns.distplot(train_df['age'],ax=ax1)

target_0 = train_df.loc[train_df['SeriousDlqin2yrs'] == 0]
target_1 = train_df.loc[train_df['SeriousDlqin2yrs'] == 1]

sns.distplot(target_0[['age']], hist=False ,ax=ax2, label='No default', axlabel='age')
sns.distplot(target_1[['age']], hist=False ,ax=ax2, label='Default')

sns.boxplot(y='age',data=train_df,ax=ax3)
mean_age = train_df['age'].mean()
std_age  = train_df['age'].std()
age_upper_limit = int(mean_age + 3 * std_age)
age_lower_limit = int(mean_age - 3 * std_age)
print(f"Upper limit: {age_upper_limit}")
print(f"Lower limit: {age_lower_limit}")
print(f"Percentage of population that falls above upper limit: {sum(train_df['age']>age_upper_limit)/len(train_df)*100:.2f}%")
print(f"Percentage of population that falls below lower limit: {sum(train_df['age']<age_lower_limit)/len(train_df)*100:.2f}%")
fig,[ax1,ax2,ax3]=plt.subplots(1,3,figsize=(25,6))
sns.distplot(train_df['DebtRatio'],ax=ax1)

target_0 = train_df.loc[train_df['SeriousDlqin2yrs'] == 0]
target_1 = train_df.loc[train_df['SeriousDlqin2yrs'] == 1]

sns.distplot(target_0[['DebtRatio']], hist=False ,ax=ax2, label='No default', axlabel='DebtRatio')
sns.distplot(target_1[['DebtRatio']], hist=False ,ax=ax2, label='Default')

sns.boxplot(y='DebtRatio',data=train_df,ax=ax3)
fig,[[ax1,ax2,ax3],[ax4,ax5,ax6],[ax7,ax8,ax9]]=plt.subplots(3,3,figsize=(25,15))\

target_0 = train_df.loc[train_df['SeriousDlqin2yrs'] == 0]
target_1 = train_df.loc[train_df['SeriousDlqin2yrs'] == 1]

sns.distplot(target_0.loc[target_0['DebtRatio']<1,'DebtRatio'], ax=ax1, label='No default', axlabel='DebtRatio')
sns.distplot(target_1.loc[target_1['DebtRatio']<1,'DebtRatio'], ax=ax1, label='Default')

sns.distplot(target_0.loc[(target_0['DebtRatio']>=1)&(target_0['DebtRatio']<10),'DebtRatio'], ax=ax2, label='No default', axlabel='DebtRatio')
sns.distplot(target_1.loc[(target_1['DebtRatio']>=1)&(target_1['DebtRatio']<10),'DebtRatio'], ax=ax2, label='Default')

sns.distplot(target_0.loc[(target_0['DebtRatio']>=10)&(target_0['DebtRatio']<100),'DebtRatio'], ax=ax3, label='No default', axlabel='DebtRatio')
sns.distplot(target_1.loc[(target_1['DebtRatio']>=10)&(target_1['DebtRatio']<100),'DebtRatio'], ax=ax3, label='Default')

sns.distplot(target_0.loc[(target_0['DebtRatio']>=100)&(target_0['DebtRatio']<1000),'DebtRatio'], ax=ax4, label='No default', axlabel='DebtRatio')
sns.distplot(target_1.loc[(target_1['DebtRatio']>=100)&(target_1['DebtRatio']<1000),'DebtRatio'], ax=ax4, label='Default')

sns.distplot(target_0.loc[(target_0['DebtRatio']>=1000)&(target_0['DebtRatio']<3000),'DebtRatio'], ax=ax5, label='No default', axlabel='DebtRatio')
sns.distplot(target_1.loc[(target_1['DebtRatio']>=1000)&(target_1['DebtRatio']<3000),'DebtRatio'], ax=ax5, label='Default')

sns.distplot(target_0.loc[(target_0['DebtRatio']>=3000)&(target_0['DebtRatio']<6000),'DebtRatio'], ax=ax6, label='No default', axlabel='DebtRatio')
sns.distplot(target_1.loc[(target_1['DebtRatio']>=3000)&(target_1['DebtRatio']<6000),'DebtRatio'], ax=ax6, label='Default')

sns.distplot(target_0.loc[(target_0['DebtRatio']>=6000)&(target_0['DebtRatio']<10000),'DebtRatio'], ax=ax7, label='No default', axlabel='DebtRatio')
sns.distplot(target_1.loc[(target_1['DebtRatio']>=6000)&(target_1['DebtRatio']<10000),'DebtRatio'], ax=ax7, label='Default')

sns.distplot(target_0.loc[(target_0['DebtRatio']>=10000)&(target_0['DebtRatio']<20000),'DebtRatio'], ax=ax8, label='No default', axlabel='DebtRatio')
sns.distplot(target_1.loc[(target_1['DebtRatio']>=10000)&(target_1['DebtRatio']<20000),'DebtRatio'], ax=ax8, label='Default')

sns.distplot(target_0.loc[(target_0['DebtRatio']>=20000),'DebtRatio'], ax=ax9, label='No default', axlabel='DebtRatio')
sns.distplot(target_1.loc[(target_1['DebtRatio']>=20000),'DebtRatio'], ax=ax9, label='Default')
debt_count_rev = {}
for i in range(0,20000,200):
    debt_count_rev[i] = sum(train_df.loc[train_df['DebtRatio']<i,'SeriousDlqin2yrs'] == 1)/len(train_df)
debt_count_rev
x, y = zip(*debt_count_rev.items())
plt.plot(x, y)
plt.xlabel('Threshold t')
plt.ylabel('Percentage of defaulters under threshold t')
plt.title('Percentage of defaulters under different thresholds')
plt.show()
print(f"Percentage of removal: {sum(train_df['DebtRatio']>10000)/len(train_df)*100:.2f}%")
Debt2_after=train_df[train_df['DebtRatio']>=1]
figure=plt.figure(figsize=(12,6))
sns.distplot(Debt2_after['DebtRatio'])
fig,[ax1,ax2]=plt.subplots(1,2,figsize=(20,6))
sns.distplot(train_df['NumberOfOpenCreditLinesAndLoans'],ax=ax1)
sns.boxplot(y=train_df['NumberOfOpenCreditLinesAndLoans'],ax=ax2)
figure=plt.figure(figsize=(12,6))
sns.countplot(train_df['NumberOfOpenCreditLinesAndLoans'])
train_df.loc[train_df['NumberOfOpenCreditLinesAndLoans']>36,'NumberOfOpenCreditLinesAndLoans']=36
count_open = train_df.groupby(['NumberOfOpenCreditLinesAndLoans'])['SeriousDlqin2yrs'].sum()
total_open = train_df.groupby(['NumberOfOpenCreditLinesAndLoans'])['SeriousDlqin2yrs'].count()
ratio_open = count_open/total_open
ratio_open.plot(kind='bar',figsize=(12,6),color='#4682B4', ylabel='Default Rate')
sum(train_df['NumberOfOpenCreditLinesAndLoans']>36)/len(train_df)
fig,[ax1,ax2]=plt.subplots(1,2,figsize=(20,6))
sns.distplot(train_df['NumberRealEstateLoansOrLines'],ax=ax1)
sns.boxplot(y=train_df['NumberRealEstateLoansOrLines'],ax=ax2)
figure=plt.figure(figsize=(12,6))
sns.countplot(train_df['NumberRealEstateLoansOrLines'])
train_df.loc[train_df['NumberRealEstateLoansOrLines']>8,'NumberRealEstateLoansOrLines']=8
count_open = train_df.groupby(['NumberRealEstateLoansOrLines'])['SeriousDlqin2yrs'].sum()
total_open = train_df.groupby(['NumberRealEstateLoansOrLines'])['SeriousDlqin2yrs'].count()
ratio_open = count_open/total_open
ratio_open.plot(kind='bar',figsize=(12,6),color='#4682B4',ylabel='Default Rate')
fig,[ax1,ax2]=plt.subplots(1,2,figsize=(20,6))
sns.countplot(train_df['NumberOfDependents'],ax=ax1)
sns.boxplot(y=train_df['NumberOfDependents'],ax=ax2)
train_df['NumberOfDependents'].isnull().sum()
train_df['MonthlyIncome'].isnull().sum()
train_df.loc[(train_df['NumberOfDependents'].isnull())&(train_df['MonthlyIncome'].isnull()),:].shape[0]
sns.countplot(train_df.loc[(train_df['NumberOfDependents'].notnull())&(train_df['MonthlyIncome'].isnull()),:]['NumberOfDependents'])
fig,[ax1,ax2,ax3]=plt.subplots(1,3,figsize=(20,6))
sns.boxplot(y=train_df['NumberOfTime30-59DaysPastDueNotWorse'],ax=ax1)
sns.boxplot(y=train_df['NumberOfTime60-89DaysPastDueNotWorse'],ax=ax3)
sns.boxplot(y=train_df['NumberOfTimes90DaysLate'],ax=ax2)
past30 = train_df['NumberOfTime30-59DaysPastDueNotWorse']>80
past60 = train_df['NumberOfTime60-89DaysPastDueNotWorse']>80
past90 = train_df['NumberOfTimes90DaysLate']>80

print(f'past30: {sum(past30)}')
print(f'past60: {sum(past60)}')
print(f'past90: {sum(past90)}')

print(f'past30 & past60 & past90: {len(train_df.loc[(past30)&(past60)&(past90),:])}')
fig,[ax1,ax2,ax3]=plt.subplots(1,3,figsize=(25,6))
sns.distplot(train_df['MonthlyIncome'],ax=ax1)

target_0 = train_df.loc[train_df['SeriousDlqin2yrs'] == 0]
target_1 = train_df.loc[train_df['SeriousDlqin2yrs'] == 1]

sns.distplot(target_0[['MonthlyIncome']], hist=False ,ax=ax2, label='No default', axlabel='MonthlyIncome')
sns.distplot(target_1[['MonthlyIncome']], hist=False ,ax=ax2, label='Default')

sns.boxplot(y='MonthlyIncome',data=train_df,ax=ax3)
fig,[ax1,ax2]=plt.subplots(1,2,figsize=(25,10))

target_0 = train_df.loc[train_df['SeriousDlqin2yrs'] == 0]
target_1 = train_df.loc[train_df['SeriousDlqin2yrs'] == 1]

sns.distplot(target_0.loc[(target_0['MonthlyIncome']>=1)&(target_0['MonthlyIncome']<20000),'MonthlyIncome'], ax=ax1, label='No default', axlabel='MonthlyIncome')
sns.distplot(target_1.loc[(target_1['MonthlyIncome']>=1)&(target_1['MonthlyIncome']<20000),'MonthlyIncome'], ax=ax1, label='Default')

sns.distplot(target_0.loc[(target_0['MonthlyIncome']>=20000)&(target_0['MonthlyIncome']<100000),'MonthlyIncome'], ax=ax2, label='No default', axlabel='MonthlyIncome')
sns.distplot(target_1.loc[(target_1['MonthlyIncome']>=20000)&(target_1['MonthlyIncome']<100000),'MonthlyIncome'], ax=ax2, label='Default')
plt.figure(figsize=(12,10))
sns.heatmap(train_df.corr(), annot=True, cmap=plt.cm.CMRmap_r)
plt.show()
sns.pairplot(train_df, hue="SeriousDlqin2yrs", diag_kws={'bw': 0.2})
print(f"Percentage of default: {sum(train_df['SeriousDlqin2yrs'] == 1)/len(train_df)*100:.2f}%")
print(f"Percentage of non-default: {sum(train_df['SeriousDlqin2yrs'] == 0)/len(train_df)*100:.2f}%")
sns.countplot('SeriousDlqin2yrs', data=train_df)
train_df = pd.read_csv(train_path)
test_df  = pd.read_csv(test_path)
def transformation(df,
                   split,
                   age_lower_limit=None, age_upper_limit=None,
                   debt_threshold=None,
                   revol_threshold=20,
                   numlate_threshold=80,
                   numreal_threshold=8,
                   numopen_threshold=36,
                   numdepend_miss=0,
                   monthincome_miss='mean'
                   ):
    
    df.drop('Unnamed: 0',axis = 1,inplace=True)
    
    assert split in ['train','test'], 'split must be train or test'
    
    if split=='train':
        # Filter age
        df = df[df['age']>age_lower_limit]
        df = df[df['age']<age_upper_limit]

        # Filter RevolvingUtilizationOfUnsecuredLines
        df = df[df['RevolvingUtilizationOfUnsecuredLines']<revol_threshold]

        # Filter NumberOfTimes##DaysLate
        df = df[df['NumberOfTime30-59DaysPastDueNotWorse']<=numlate_threshold]
        df = df[df['NumberOfTime60-89DaysPastDueNotWorse']<=numlate_threshold]
        df = df[df['NumberOfTimes90DaysLate']<=numlate_threshold]

        # Filter DebtRatio
        df = df[df['DebtRatio']<=debt_threshold]
    
        # Regroup NumberRealEstateLoansOrLines
        df.loc[df['NumberRealEstateLoansOrLines']>numreal_threshold,'NumberRealEstateLoansOrLines']=numreal_threshold

        # Regroup NumberOfOpenCreditLinesAndLoans
        df.loc[df['NumberOfOpenCreditLinesAndLoans']>numopen_threshold,'NumberOfOpenCreditLinesAndLoans']=numopen_threshold

    # Handling missing values
    # Fill NumberOfDependents missing values with 0
    df.loc[df['NumberOfDependents'].isnull(),'NumberOfDependents'] = numdepend_miss
    
    assert monthincome_miss in ['mean','median'], 'Monthly income must be filled with median or mean'
    
    # Fill MonthlyIncome missing values
    if monthincome_miss == 'mean':
        df.loc[df['MonthlyIncome'].isnull(),'MonthlyIncome'] = df['MonthlyIncome'].mean()
    else:
        df.loc[df['MonthlyIncome'].isnull(),'MonthlyIncome'] = df['MonthlyIncome'].median()
    
    return df
train_df = transformation(train_df, split='train',
                          revol_threshold=9,
                          age_lower_limit=age_lower_limit,
                          age_upper_limit=age_upper_limit,
                          debt_threshold=10000)
test_df  = transformation(test_df, split='test')
train_null_sums = train_df.isnull().sum()
test_null_sums = test_df.isnull().sum()
print(f'train_null_sums: {train_null_sums}')
print(f'='*50)
print(f'test_null_sums: {test_null_sums}')
train_x = train_df.iloc[:,1:]
train_y = train_df['SeriousDlqin2yrs']
test_x = test_df.iloc[:,1:]
import xgboost as xgb
from sklearn import model_selection
params = {'subsample': 0.7,
          'n_estimators':1000,
          'min_child_weight': 9.0,
          'objective': 'binary:logistic',
          'gamma': 0.65,
          'max_depth': 6,
          'max_delta_step': 1.8,
          'colsample_bytree': 0.5,
          'eta': 0.01,
          'tree_method':'gpu_hist'}
xg_cls = xgb.XGBClassifier(**params)
cross_val_score(xg_cls, train_x, train_y, scoring='roc_auc')
# training
xg_cls.fit(train_x, train_y)

# inference
preds_xgb_classifier = xg_cls.predict_proba(test_x)
preds_xgb_classifier = np.clip(preds_xgb_classifier, a_min = 0., a_max = 1.)

# submission
sampleEntry = pd.read_csv(sample)
sampleEntry['Probability'] = preds_xgb_classifier[:,1]
export_csv = sampleEntry.to_csv('export_dataframe.csv',index = None,header=True)
import shap
shap.initjs()

explainer = shap.TreeExplainer(xg_cls)
shap_values = explainer.shap_values(test_x)
shap.summary_plot(shap_values, test_x, plot_type="bar")
shap.summary_plot(shap_values, test_x)
params = {'subsample': 0.7,
          'n_estimators':1000,
          'min_child_weight': 9.0,
          'objective': 'binary:logistic',
          'gamma': 0.65,
          'max_depth': 6,
          'max_delta_step': 1.8,
          'colsample_bytree': 0.5,
          'eta': 0.01,
          'tree_method':'gpu_hist'}
xg_reg = xgb.XGBRegressor(**params)
cross_val_score(xg_reg, train_x, train_y, scoring='roc_auc')
# training
xg_reg.fit(train_x,train_y)

# inference
preds_xgb_regressor = xg_reg.predict(test_x)
preds_xgb_regressor = np.clip(preds_xgb_regressor, a_min = 0., a_max = 1.)
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = MinMaxScaler()
train_x_ = scaler.fit_transform(train_x)
train_x_ = pd.DataFrame(train_x_, index=train_x.index, columns=train_x.columns)

test_x_ = scaler.fit_transform(test_x)
test_x_ = pd.DataFrame(test_x_, index=test_x.index, columns=test_x.columns)
mlpreg = MLPRegressor(random_state=1,
                      max_iter=100,
                      hidden_layer_sizes=(100,)).fit(train_x_, train_y)
cross_val_score(mlpreg, train_x_, train_y, scoring='roc_auc')
preds_mlpregr = mlpreg.predict(test_x_)
preds_mlpregr = np.clip(preds_mlpregr, a_min = 0., a_max = 1.)
from sklearn.neural_network import MLPClassifier
mlpcls = MLPClassifier(random_state=1,
                       max_iter=400,
                       hidden_layer_sizes=(100,),
                       learning_rate='adaptive').fit(train_x_, train_y)
cross_val_score(mlpcls, train_x_, train_y, scoring='roc_auc')
preds_mlpcls = mlpcls.predict_proba(test_x_)
preds_mlpcls = preds_mlpcls[:,1]
preds_mlpcls = np.clip(preds_mlpcls, a_min = 0., a_max = 1.)
import keras
import tensorflow as tf
from keras import Sequential
from keras.layers import LeakyReLU, PReLU, ELU, Dense, Dropout, Input
# model initializer
initializer = tf.keras.initializers.HeUniform()

# build model
deep_classifier = Sequential()
deep_classifier.add(Input(shape=(10,)))
deep_classifier.add(Dense(100,kernel_initializer=initializer,activation='relu',name="layer1"))
deep_classifier.add(Dense(1,kernel_initializer=initializer,activation='sigmoid',name="classifier"))

deep_classifier.compile(loss=tf.keras.losses.BinaryCrossentropy(), 
                        optimizer='Adam')

# start training
model_history = deep_classifier.fit(train_x_.values,
                                    train_y.values,
                                    validation_split=0.1,
                                    batch_size=10,
                                    epochs=10,
                                    callbacks=[tf.keras.callbacks.EarlyStopping(patience=2)])
train_preds_deep_model = deep_classifier.predict(train_x_.values)
roc_auc_score(train_y.values,train_preds_deep_model)
preds_deep_model = deep_classifier.predict(test_x_.values)
preds = (preds_xgb_classifier + preds_xgb_regressor + preds_mlpregr)/3
sampleEntry = pd.read_csv(sample)
sampleEntry['Probability'] = preds
export_csv = sampleEntry.to_csv('export_dataframe.csv',index = None,header=True)