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
import pandas as pd 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline 
import os
PATH="../input"
print(os.listdir(PATH))
application_train = pd.read_csv(PATH+"/application_train.csv")
application_test = pd.read_csv(PATH+"/application_test.csv")
#bureau = pd.read_csv(PATH+"/bureau.csv")
#bureau_balance = pd.read_csv(PATH+"/bureau_balance.csv")
#credit_card_balance = pd.read_csv(PATH+"/credit_card_balance.csv")
#installments_payments = pd.read_csv(PATH+"/installments_payments.csv")
#previous_application = pd.read_csv(PATH+"/previous_application.csv")
#POS_CASH_balance = pd.read_csv(PATH+"/POS_CASH_balance.csv")
print("application_train -  rows:",application_train.shape[0]," columns:", application_train.shape[1])
print("application_test -  rows:",application_test.shape[0]," columns:", application_test.shape[1])
application_train.head()
application_train.columns.values
application_test.head()
application_test.columns.values
application_train['OWN_CAR_AGE'].head(10)
application_train['OWN_CAR_AGE']=application_train['OWN_CAR_AGE'].fillna(0)
plt.figure(figsize = (10, 8))

# KDE plot of loans that were repaid on time
sns.kdeplot(application_train.loc[application_train['TARGET'] == 0, 'OWN_CAR_AGE'] , label = 'target == 0')

# KDE plot of loans which were not repaid on time
sns.kdeplot(application_train.loc[application_train['TARGET'] == 1, 'OWN_CAR_AGE'] , label = 'target == 1')

# Labeling of plot
plt.xlabel('Own Car Age'); plt.ylabel('Density'); plt.title('Distribution');
# Age information into a separate dataframe
age_data = application_train[['TARGET', 'OWN_CAR_AGE']]

# Bin the age data
age_data['YEARS_BINNED'] = pd.cut(age_data['OWN_CAR_AGE'], bins = np.linspace(-1, 71, num =5 ))
age_data.head(10)
# Group by the bin and calculate averages
age_groups  = age_data.groupby('YEARS_BINNED').mean()
#age_groups
age_groups_new0 = age_data['YEARS_BINNED'].to_frame()
age_groups_new0.columns = ['range']
#concatenate age and its bin
age_groups_new = pd.concat([age_data['OWN_CAR_AGE'],age_groups_new0],axis = 1)
age_groups_new
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
 
#draw histogram plot
sns.countplot(x = 'range', data = age_groups_new, palette = 'hls')
plt.show()
plt.figure(figsize = (8, 8))
# Graph the age bins and the average of the target as a bar plot
plt.bar(age_groups.index.astype(str), 100 * age_groups['TARGET'])
# Plot labeling
plt.xticks(rotation = 75); plt.xlabel('Range'); plt.ylabel('Failure to Repay (%)')
plt.title('Failure to Repay by Age Group');
application_train['OWN_CAR_AGE']=age_groups_new['range']
application_train['OWN_CAR_AGE'].head(10)
application_train['OBS_30_CNT_SOCIAL_CIRCLE'] = (application_train['OBS_30_CNT_SOCIAL_CIRCLE']+application_train['DEF_30_CNT_SOCIAL_CIRCLE']+application_train['OBS_60_CNT_SOCIAL_CIRCLE']+application_train['DEF_60_CNT_SOCIAL_CIRCLE'])/4

plt.scatter(application_train['SK_ID_CURR'],application_train['OBS_30_CNT_SOCIAL_CIRCLE'])
plt.figure(figsize = (10, 8))

# KDE plot of loans that were repaid on time
sns.kdeplot(application_train.loc[application_train['TARGET'] == 0, 'OBS_30_CNT_SOCIAL_CIRCLE'] , label = 'target == 0')

# KDE plot of loans which were not repaid on time
sns.kdeplot(application_train.loc[application_train['TARGET'] == 1, 'OBS_30_CNT_SOCIAL_CIRCLE'] , label = 'target == 1')

# Labeling of plot
plt.xlabel('OBS_30_CNT_SOCIAL_CIRCLE'); plt.ylabel('Density'); plt.title('Distribution');
# social information into a separate dataframe
social_data = application_train[['TARGET', 'OBS_30_CNT_SOCIAL_CIRCLE']]

# Bin the social data
social_data['BINNED'] = pd.cut(social_data['OBS_30_CNT_SOCIAL_CIRCLE'], bins = np.linspace(-1, 21, num =6))
social_data.head(17)
# Group by the bin and calculate averages
social_groups_new0 = social_data['BINNED'].to_frame()
social_groups_new0.columns = ['range']
#concatenate age and its bin
social_groups_new = pd.concat([social_data['OBS_30_CNT_SOCIAL_CIRCLE'],social_groups_new0],axis = 1)
social_groups_new
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
 
#draw histogram plot
sns.countplot(x = 'range', data = social_groups_new, palette = 'hls')
plt.show()
social_groups  = social_data.groupby('BINNED').mean()
print(social_groups)
plt.figure(figsize = (8, 8))
# Graph the bins and the average of the target as a bar plot
plt.xticks(range(len(social_groups.index.astype(str))), social_groups.index.astype(str))
plt.bar(range(len(social_groups.index.astype(str))), 100 * social_groups['TARGET'])
# Plot labeling
plt.xticks(rotation = 75); plt.xlabel('Range'); plt.ylabel('Failure to Repay (%)')
plt.title('Failure to Repay by social Group');
application_train['OBS_30_CNT_SOCIAL_CIRCLE']=social_groups_new['range']
application_train['OBS_30_CNT_SOCIAL_CIRCLE']
plt.scatter(application_train['SK_ID_CURR'],application_train['DAYS_LAST_PHONE_CHANGE'])
# date information into ayears
phone_data = application_train[['TARGET', 'DAYS_LAST_PHONE_CHANGE']]
phone_data['DAYS_LAST_PHONE_CHANGE'] = phone_data['DAYS_LAST_PHONE_CHANGE'] / 365

# Bin the years data
phone_data['YEARS_BINNED'] = pd.cut(phone_data['DAYS_LAST_PHONE_CHANGE'], bins = np.linspace(-10, 0, num = 6))
phone_data.head(10)
# Group by the bin and calculate averages
# age_groups  = age_data.groupby('YEARS_BINNED').mean()
phone_groups_new0 = phone_data['YEARS_BINNED'].to_frame()
phone_groups_new0.columns = ['range']
#concatenate age and its bin
phone_groups_new = pd.concat([phone_data['DAYS_LAST_PHONE_CHANGE'],phone_groups_new0],axis = 1)
phone_groups_new
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
 
#draw histogram plot
sns.countplot(x = 'range', data = phone_groups_new, palette = 'hls')
plt.show()
phone_groups  = phone_data.groupby('YEARS_BINNED').mean()
print(phone_groups)
plt.figure(figsize = (8, 8))
# Graph the age bins and the average of the target as a bar plot
plt.xticks(range(len(phone_groups.index.astype(str))), phone_groups.index.astype(str))
plt.bar(range(len(phone_groups.index.astype(str))), 100 * phone_groups['TARGET'])
#plt.bar(phone_groups.index.astype(str), 100 * phone_groups['TARGET'])
# Plot labeling
plt.xticks(rotation = 75); plt.xlabel('Range'); plt.ylabel('Failure to Repay (%)')
plt.title('Failure to Repay by social Group');
application_train['DAYS_LAST_PHONE_CHANGE']=phone_groups_new['range']
application_train['DAYS_LAST_PHONE_CHANGE']
application_train = pd.get_dummies(application_train)
application_test = pd.get_dummies(application_test)

print('Training Features shape: ', application_train.shape)
print('Testing Features shape: ', application_test.shape)