# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
def read_data(path):
    data=pd.read_csv(path)
    print(data.head())
    return data
application_train=read_data("../input/application_train.csv")
POS_CASH_balance=read_data("../input/POS_CASH_balance.csv")
bureau_balance=read_data("../input/bureau_balance.csv")
previous_application=read_data("../input/previous_application.csv")
installments_payments=read_data("../input/installments_payments.csv")
credit_card_balance=read_data("../input/credit_card_balance.csv")
sample_submission=read_data("../input/sample_submission.csv")
application_test=read_data("../input/application_test.csv")
bureau=read_data("../input/bureau.csv")

application_train.head()
application_train.isnull().sum()
application_train.shape
target=application_train.corr().TARGET
target[target>0.0]
application_train.TARGET.value_counts()
sns.countplot(application_train.TARGET)
application_train.head()
credit_card_balance.head()
installments_payments.head()
print(bureau_balance.shape)
bureau_balance.head()
print(previous_application.shape)
previous_application.head()
bureau.head()
previous_application.head()
previous_application.NAME_CONTRACT_TYPE.value_counts()
bureau.head()
# Groupby the client id (SK_ID_CURR), count the number of previous loans, and rename the column
previous_loan_counts = bureau.groupby('SK_ID_CURR', as_index=False)['SK_ID_BUREAU'].count().rename(columns = {'SK_ID_BUREAU': 'previous_loan_counts'})
previous_loan_counts.head(20)
# Join to the training dataframe

application_train = application_train.merge(previous_loan_counts, on = 'SK_ID_CURR', how = 'left')

# Fill the missing values with 0 
application_train['previous_loan_counts'] = application_train['previous_loan_counts'].fillna(0)
application_train.head()
# Checking the correlation
application_train['TARGET'].corr(application_train['previous_loan_counts'])
# Plots the disribution of a variable colored by value of the target
def kde_target(var_name, df):
    
    # Calculate the correlation coefficient between the new variable and the target
    corr = df['TARGET'].corr(df[var_name])
    
    # Calculate medians for repaid vs not repaid
    avg_repaid = df.ix[df['TARGET'] == 0, var_name].median()
    avg_not_repaid = df.ix[df['TARGET'] == 1, var_name].median()
    
    plt.figure(figsize = (12, 6))
    
    # Plot the distribution for target == 0 and target == 1
    sns.kdeplot(df.ix[df['TARGET'] == 0, var_name], label = 'TARGET == 0')
    sns.kdeplot(df.ix[df['TARGET'] == 1, var_name], label = 'TARGET == 1')
    
    # label the plot
    plt.xlabel(var_name); plt.ylabel('Density'); plt.title('%s Distribution' % var_name)
    plt.legend();
    
    # print out the correlation
    print('The correlation between %s and the TARGET is %0.4f' % (var_name, corr))
    # Print out average values
    print('Median value for loan that was not repaid = %0.4f' % avg_not_repaid)
    print('Median value for loan that was repaid =     %0.4f' % avg_repaid)
installments_payments.head()
installments_payments.SK_ID_CURR.nunique()
application_train.shape[0]
application_test.shape[0]
bureau


