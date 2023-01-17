# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

color = sns.color_palette()
application_train = pd.read_csv('/kaggle/input/home-credit-default-risk/application_train.csv')

application_test= pd.read_csv('/kaggle/input/home-credit-default-risk/application_test.csv')

bureau = pd.read_csv('/kaggle/input/home-credit-default-risk/bureau.csv')

bureau_balance = pd.read_csv('/kaggle/input/home-credit-default-risk/bureau_balance.csv')

POS_CASH_balance = pd.read_csv('/kaggle/input/home-credit-default-risk/POS_CASH_balance.csv')

credit_card_balance = pd.read_csv('/kaggle/input/home-credit-default-risk/credit_card_balance.csv')

previous_application = pd.read_csv('/kaggle/input/home-credit-default-risk/previous_application.csv')

installments_payments = pd.read_csv('/kaggle/input/home-credit-default-risk/installments_payments.csv')
application_train.head()
application_train.columns.values
def find_missing(data):

    # number of missing values

    count_missing = data.isnull().sum().values

    # total records

    total = data.shape[0]

    # percentage of missing

    ratio_missing = count_missing/total

    # return a dataframe to show: feature name, # of missing and % of missing

    return pd.DataFrame(data={'missing_count':count_missing, 'missing_ratio':ratio_missing}, index=data.columns.values)

find_missing(application_train).head(12)
POS_CASH_balance.head()
POS_CASH_balance.columns.values
find_missing(POS_CASH_balance).head(12)
bureau.head()
bureau.columns.values
find_missing(bureau).head(12)
bureau_balance.head()
bureau_balance.columns.values
find_missing(bureau_balance).head(12)
credit_card_balance.head()
credit_card_balance.columns.values
find_missing(credit_card_balance).head(12)
previous_application.head()
previous_application.columns.values
find_missing(previous_application).head(12)
installments_payments.head()
installments_payments.columns.values
find_missing(installments_payments).head(12)
def plot_categorical(data, col, size=[8 ,4], xlabel_angle=0, title=''):

    '''use this for ploting the count of categorical features'''

    plotdata = data[col].value_counts()

    plt.figure(figsize = size)

    sns.barplot(x = plotdata.index, y=plotdata.values)

    plt.title(title)

    if xlabel_angle!=0: 

        plt.xticks(rotation=xlabel_angle)

    plt.show()

plot_categorical(data=application_train, col='TARGET', size=[8 ,4], xlabel_angle=0, title='train set: label')
plot_categorical(data=application_train, col='OCCUPATION_TYPE', size=[12 ,4], xlabel_angle=30, title='Occupation Type')
plot_categorical(data=application_train, col='NAME_INCOME_TYPE', size=[12 ,4], xlabel_angle=0, title='Income Type')
plot_categorical(data=application_train, col='NAME_HOUSING_TYPE', size=[12 ,4], xlabel_angle=0, title='House Type')
def plot_numerical(data, col, size=[8, 4], bins=50):

    '''use this for ploting the distribution of numercial features'''

    plt.figure(figsize=size)

    plt.title("Distribution of %s" % col)

    sns.distplot(data[col].dropna(), kde=True,bins=bins)

    plt.show()

plot_numerical(application_train, 'AMT_CREDIT')
plot_numerical(application_train, 'AMT_ANNUITY')
plot_numerical(application_train, 'DAYS_EMPLOYED')
def plot_categorical_bylabel(data, col, size=[12 ,6], xlabel_angle=0, title=''):

    '''use it to compare the distribution between label 1 and label 0'''

    plt.figure(figsize = size)

    l1 = data.loc[data.TARGET==1, col].value_counts()

    l0 = data.loc[data.TARGET==0, col].value_counts()

    plt.subplot(1,2,1)

    sns.barplot(x = l1.index, y=l1.values)

    plt.title('Default: '+title)

    plt.xticks(rotation=xlabel_angle)

    plt.subplot(1,2,2)

    sns.barplot(x = l0.index, y=l0.values)

    plt.title('Non-default: '+title)

    plt.xticks(rotation=xlabel_angle)

    plt.show()

plot_categorical_bylabel(application_train, 'CODE_GENDER', title='Gender')
plot_categorical_bylabel(application_train, 'NAME_EDUCATION_TYPE', size=[15 ,6], xlabel_angle=15, title='Education Type')
def plot_numerical_bylabel(data, col, size=[12, 6]):

    plt.figure(figsize=size)

    # Calculate the correlation coefficient between the new variable and the target

    corr = data['TARGET'].corr(data[col])

    

    # Calculate medians for repaid vs not repaid

    avg_repaid = data.ix[data['TARGET'] == 0, col].median()

    avg_not_repaid = data.ix[data['TARGET'] == 1, col].median()

    

    plt.figure(figsize = (12, 6))

    

    # Plot the distribution for target == 0 and target == 1

    sns.kdeplot(data.ix[data['TARGET'] == 0, col], label = 'TARGET == 0')

    sns.kdeplot(data.ix[data['TARGET'] == 1, col], label = 'TARGET == 1')

    

    # label the plot

    plt.xlabel(col); plt.ylabel('Density'); plt.title('%s Distribution' % col)

    plt.legend();

    # print out the correlation

    print('The correlation between %s and the TARGET is %0.4f' % (col, corr))

    # Print out average values

plot_numerical_bylabel(application_train, 'EXT_SOURCE_1')
plot_numerical_bylabel(application_train, 'EXT_SOURCE_2')
plot_numerical_bylabel(application_train, 'EXT_SOURCE_3')
corr_mat = application_train.corr()

plt.figure(figsize=[15, 15])

sns.heatmap(corr_mat.values, annot=False)

plt.show()