# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
bureau_balance = pd.read_csv("../input/bureau_balance.csv",nrows=5000)
application_train = pd.read_csv("../input/application_train.csv",nrows=5000)
application_test = pd.read_csv("../input/application_test.csv",nrows=5000)
bureau = pd.read_csv("../input/bureau.csv",nrows=5000)
credit_card_balance = pd.read_csv("../input/credit_card_balance.csv",nrows=5000)
installments_payments = pd.read_csv("../input/installments_payments.csv",nrows=5000)
previous_application = pd.read_csv("../input/previous_application.csv",nrows=5000)
POS_CASH_balance = pd.read_csv("../input/POS_CASH_balance.csv",nrows=5000)

def missing_data(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data(credit_card_balance)

temp = application_train["TARGET"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
plt.figure(figsize = (6,6))
plt.title('Application loans repayed - train dataset')
sns.set_color_codes("pastel")
sns.barplot(x = 'labels', y="values", data=df)
locs, labels = plt.xticks()
plt.show()

def plot_stats(feature,label_rotation=False,horizontal_layout=True):
    temp = application_train[feature].value_counts()
    df1 = pd.DataFrame({feature: temp.index,'Number of contracts': temp.values})

    # Calculate the percentage of target=1 per category value
    cat_perc = application_train[[feature, 'TARGET']].groupby([feature],as_index=False).mean()
    cat_perc.sort_values(by='TARGET', ascending=False, inplace=True)
    
    if(horizontal_layout):
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))
    else:
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12,14))
    sns.set_color_codes("pastel")
    s = sns.barplot(ax=ax1, x = feature, y="Number of contracts",data=df1)
    if(label_rotation):
        s.set_xticklabels(s.get_xticklabels(),rotation=90)
    
    s = sns.barplot(ax=ax2, x = feature, y='TARGET', order=cat_perc[feature], data=cat_perc)
    if(label_rotation):
        s.set_xticklabels(s.get_xticklabels(),rotation=90)
    plt.ylabel('Percent of target with value 1 [%]', fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=10)

    plt.show()
plot_stats('NAME_CONTRACT_TYPE')
plot_stats('CODE_GENDER')
plot_stats('OCCUPATION_TYPE',True, False)
def count_plots(feature, label_rotation=False):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,9))
    plot1 = sns.countplot(ax = ax1, x=feature, hue = 'TARGET',data =  application_train)
    if(label_rotation):
        plot1.set_xticklabels(plot1.get_xticklabels(),rotation=90)
    plt.tick_params(axis='both', which='major', labelsize=10)
    # since 1 means not repayed the mean will give us the proportion of non repayed loans
    perc_grouped = application_train[[feature, 'TARGET']].groupby(feature).mean().sort_values(by="TARGET", ascending= False)
    plot2 = sns.barplot(ax=ax2, x=perc_grouped['TARGET'], y = perc_grouped.index, orient="h")

    plt.tight_layout()
    plt.show()

#Does owning a car affect their ability to pay?

count_plots("FLAG_OWN_CAR")

