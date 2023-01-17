import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np
import os

os.getcwd()
loan = pd.read_csv('../input/loan.csv',encoding='ISO-8859–1',low_memory=False)
!pip install pandas_profiling
import pandas_profiling as pp
pp.ProfileReport(loan)
loan.head()
loan.shape
loan.describe()
# Lets see the number of missing values in the each column

loan.isnull().sum()
# lets look at the percentage of missing values in each column

null_col = round(100*(loan.isnull().sum()/len(loan.index)), 2)

null_col[null_col>0]
# Lets look at the columns which has 100% missing values

null_col[null_col==100].index
# Lets look at the total number of columns where each column has 100% missing values

null_col[null_col==100].count()
#lets drop the columns where columns have 100% missing or null values

loan = loan.drop(list(null_col[null_col==100].index),axis=1)
loan.info()
loan.shape
# Lets look at the percentage of missing values for other columns

null_col = round(100*(loan.isnull().sum()/len(loan.index)), 2)

null_col[null_col>0]
loan = loan.drop(list(null_col[null_col>30].index),axis=1)
null_col = round(100*(loan.isnull().sum()/len(loan.index)), 2)

null_col[null_col>0]
loan.shape
# Now lets look at the report of each column using pandas-profiling again

pp.ProfileReport(loan)
# Now lets look at these columns which have null values and 

# decide whether these columns add value to our analysis or not

loan['emp_title'].value_counts()
# Lets remove the null values

loan = loan[~loan['emp_title'].isnull()]
loan['emp_length'].value_counts()
loan['emp_length'].isnull().sum()
# Lets remove the null values

loan = loan[~loan['emp_length'].isnull()]
# lets convert the values and change the data type to int

loan['emp_length'] = loan['emp_length'].replace({'10+ years':10,'2 years':'2','< 1 year':0,'3 years':3,

                                                 '4 years':4,'5 years':5,'6 years':6,'7 years':7,'8 years':8,

                                                 '9 years':9,'1 year':1}).astype('int')
loan['emp_length'].value_counts()
loan['emp_length'].dtype
loan['loan_status'].value_counts()
loan = loan[loan['loan_status'].isin(['Fully Paid','Charged Off'])]
loan['loan_status'].value_counts()
loan.isnull().sum()
# some columns are not useful as their values are not useful to our analysis or 

# these columns are not availabnle during making a decision on whether to approve the loan or reject it,so will remove them

# find out yourself why these columns are not useful by seeing obove analysis and description of each column

loan = loan.drop(['id','member_id','funded_amnt','funded_amnt_inv','url','pymnt_plan','initial_list_status','out_prncp','out_prncp_inv','total_rec_prncp',

                  'total_rec_int','total_rec_late_fee','recoveries','last_pymnt_d','collection_recovery_fee','policy_code',

                  'application_type','acc_now_delinq','delinq_amnt','collections_12_mths_ex_med','chargeoff_within_12_mths',

                  'installment','tax_liens','pub_rec','total_pymnt_inv','zip_code','addr_state'],axis=1)
loan.isnull().sum()
null_per = round(100*(loan.isnull().sum()/len(loan.index)), 2) 

null_per[null_per != 0]
# Lets remove the missing values from these columns

loan = loan[~loan['revol_util'].isnull()]

loan = loan[~loan['title'].isnull()]
loan['loan_amnt'].value_counts()
loan['loan_amnt'].dtype
loan['int_rate'].value_counts()
# As we can see that interest rate column have percentage symbol, will remove that

loan['int_rate'] = loan['int_rate'].str.strip('%')
# lets rename the column int_rate to int_rate(in %) for better understanding

loan.rename(columns={'int_rate':'int_rate(in %)'},inplace = True)
# lets check the data type and here '0' represents object

loan['int_rate(in %)'].dtype
# As we can that data type of int_rate(in %) column is an object, will change the data type to float

loan['int_rate(in %)'] = loan['int_rate(in %)'].astype('float')
loan['int_rate(in %)'].value_counts().head()
loan['revol_util'].value_counts().head()
# As we can see that revol_util column have percentage symbol, will remove that

loan['revol_util'] = loan['revol_util'].str.strip('%')
# lets rename the column int_rate to int_rate(in %) for better understanding

loan.rename(columns={'revol_util':'revol_util(in %)'},inplace = True)
# lets check the data type and here '0' represents object

loan['revol_util(in %)'].dtype
# As we can that data type of int_rate(in %) column is an object, will change the data type to float

loan['revol_util(in %)'] = loan['revol_util(in %)'].astype('float')
loan['revol_util(in %)'].value_counts().head()
loan['pub_rec_bankruptcies'].value_counts()
loan = loan[~loan['pub_rec_bankruptcies'].isnull()]
loan['term'].value_counts()
loan.dtypes
pp.ProfileReport(loan)
# we can see from the overview report that column deling_2yrs has 89% zeros

loan['delinq_2yrs'].value_counts()
loan.info()
# we can remove this column as this column data is not available during

# the process of deciding the loan approval or rejection

loan = loan.drop(['delinq_2yrs'],axis=1)
# As these columns data are not available before hand, so lets drop these columns

loan = loan.drop(['total_pymnt','last_pymnt_amnt','last_credit_pull_d'],axis=1)
loan['issue_d'].dtype
loan['issue_d'].value_counts().head()
# lets convert the data type of date columnn

loan['issue_d'] = pd.to_datetime(loan['issue_d'], format='%b-%y')
loan['issue_d'].dtype
loan['issue_d_month'] = loan['issue_d'].dt.month

loan['issue_d_year']  = loan['issue_d'].dt.year
loan['issue_d_year'].value_counts()
loan['issue_d_month'].value_counts()
loan.shape
loan.info()
loan['grade'].value_counts().head()
loan['sub_grade'].value_counts().head()
loan['home_ownership'].value_counts()
loan['verification_status'].value_counts()
loan['purpose'].value_counts()
loan['title'].value_counts()
# lets remove title column as the column purpose has these categories

loan = loan.drop(['title'],axis=1)
loan['dti'].value_counts().head()
loan['dti'].dtype
loan['open_acc'].value_counts().head()
loan['open_acc'].dtype
loan['revol_util(in %)'].value_counts().head()
loan['revol_bal'].value_counts()
loan['total_acc'].value_counts().head()
# Now lets have a look at the loan_status column

loan['loan_status'].value_counts()
# Lets look at the percentage of each of them

((loan['loan_status'].value_counts()[0])/loan.shape[0])*100
# Lets look at the percentage of each of them

((loan['loan_status'].value_counts()[1])/loan.shape[0])*100
def add_value_labels(ax, d=None):

    plt.margins(0.2, 0.2)

    rects = ax.patches

    i = 0

    locs, labels = plt.xticks() 

    counts = {}

    if not d is None:

        for key, value in d.items():

            counts[str(key)] = value



    # For each bar: Place a label

    for rect in rects:

        # Get X and Y placement of label from rect.

        y_val = rect.get_height()

        x_val = rect.get_x() + rect.get_width() / 2



        # Number of points between bar and label. Change to your liking.

        space = 5

        # Vertical alignment for positive values

        va = 'bottom'



        # If value of bar is negative: Place label below bar

        if y_val < 0:

            # Invert space to place label below

            space *= -1

            # Vertically align label at top

            va = 'top'



        # Use Y value as label and format number with one decimal place

        if d is None:

            label = "{:.1f}%".format(y_val)

        else:

            try:

                label = "{:.1f}%".format(y_val) + "\nof " + str(counts[str(labels[i].get_text())])

            except:

                label = "{:.1f}%".format(y_val)

        

        i = i+1



        # Create annotation

        plt.annotate(

            label,                      # Use `label` as label

            (x_val, y_val),         # Place label at end of the bar

            xytext=(0, space),          # Vertically shift label by `space`

            textcoords="offset points", # Interpret `xytext` as offset in points

            ha='center',                # Horizontally center label

            va=va)                      # Vertically align label differently for

                                        # positive and negative values.
# lets again see the overview of the loan dataset

pp.ProfileReport(loan)
# This method plots a distribution of target column, and its boxplot against loan_status column

def plot_dist(dataframe, col):

    plt.figure(figsize=(15,5))

    plt.subplot(1, 2, 1)

    ax = sns.distplot(dataframe[col])

    plt.subplot(1, 2, 2)

    sns.boxplot(x=dataframe[col], y=dataframe['loan_status'], data=dataframe)

    plt.show()
plot_dist(loan, 'loan_amnt')
loan.groupby('loan_status')['loan_amnt'].describe()
plot_dist(loan, 'int_rate(in %)')
loan.groupby('loan_status')['int_rate(in %)'].describe()
plot_dist(loan, 'emp_length')
loan.groupby('loan_status')['emp_length'].describe()
plot_dist(loan, 'dti')
loan.groupby('loan_status')['dti'].describe()
#Let's see if some of these categorical variables follow the famous power law  by plotting the line plot.

plt.figure(figsize=(15,5))

plt.subplot(1, 3, 1)

loan.groupby('grade')['loan_amnt'].count().sort_values(ascending=False).plot(kind='line',  marker='o', color='g')

plt.legend()

plt.subplot(1, 3, 2)

loan.groupby('purpose')['loan_amnt'].count().sort_values(ascending=False).plot(kind='line', logy = True, marker='o', color='g')

plt.legend()

plt.subplot(1, 3, 3)

loan.groupby('verification_status')['loan_amnt'].count().sort_values(ascending=False).plot(kind='line', marker='o', color='g')

plt.legend()
def BarPlots(df, arr):

    rows = int(len(arr)/2)

    for idx, val in enumerate(arr, start=1):

        plt.subplot(rows, 2, idx)

        ax = df.groupby(val)['loan_amnt'].count().plot.bar(color=sns.color_palette('husl', 16))

    plt.tight_layout()
plt.figure(figsize=(10,10))

BarPlots(loan, ['home_ownership', 'term', 'verification_status', 'purpose'])
plt.figure(figsize=(15,5))

BarPlots(loan,['grade','sub_grade'])
def plotLoanStatus(df, col, loanstatus='Charged Off'):

    """

    

    plotLoanStatus function will plot the bar graphs based on the parameters 

    which show the percentage of loans of that particular column records are charged off

    

    

    df         : dataframe name

    col        : Column name

    loanstatus : 'Charged Off' (default)

    

    """

    grp = df.groupby(['loan_status',col])[col].count()

    cnt = df.groupby(col)[col].count()

    percentages = grp.unstack() * 100 / cnt.T

    ax = percentages.loc[loanstatus].plot.bar(color=sns.color_palette('husl', 16))

    ax.set_ylabel('% of loans ' + loanstatus)

    add_value_labels(ax, grp[loanstatus].to_dict())

    plt.margins(0.2, 0.2)

    plt.tight_layout()

    return ax
def plot_percentages(df, col, sortbyindex=False):

    """

    

    plot_percentages function will plot the bar graphs based on the parameters 

    first bar graph represents the percentage of that particular column records in the dataset and

    second graph represents percentage of charged off loans count for the given column

    

    

    df          : dataframe name

    col         : Column name

    sortbyindex : 'False' (default) 

    

    """

    plt.subplot(1, 2, 1)

    values = (loan[col].value_counts(normalize=True)*100)

    if sortbyindex:

        values = values.sort_index()

    ax = values.plot.bar(color=sns.color_palette('husl', 16))

    ax.set_ylabel('% in dataset', fontsize=16)

    ax.set_xlabel(col, fontsize=12)

    add_value_labels(ax)

    plt.subplot(1, 2, 2)

    values = (loan.loc[loan['loan_status']=='Charged Off'][col].value_counts(normalize=True)*100)

    if sortbyindex:

        values = values.sort_index()

    ax = values.plot.bar(color=sns.color_palette('husl', 16))

    ax.set_ylabel('% in Charged Off loans', fontsize=16)

    ax.set_xlabel(col, fontsize=12)

    add_value_labels(ax)
plt.figure(figsize=(10,5))

plot_percentages(loan, 'term')

plt.tight_layout()
plt.figure(figsize=(15,7))

plot_percentages(loan, 'purpose',True)

plt.tight_layout()
# lets see the percentage of loans which are charged off

plt.figure(figsize=(10, 6))

plotLoanStatus(loan, 'purpose')
plt.figure(figsize=(15,7))

plot_percentages(loan, 'grade',True)

plt.tight_layout()
# lets see the percentage of loans which are charged off for each grade

plt.figure(figsize=(10, 6))

plotLoanStatus(loan, 'grade')
plt.figure(figsize=(15,7))

plot_percentages(loan, 'home_ownership')

plt.tight_layout()
# lets see the percentage of loans which are charged off 

plt.figure(figsize=(10, 6))

plotLoanStatus(loan, 'home_ownership')
plt.figure(figsize=(10, 6))

plotLoanStatus(loan, 'inq_last_6mths')
plt.figure(figsize=(10, 6))

plotLoanStatus(loan, 'pub_rec_bankruptcies')
plt.figure(figsize=(10, 6))

plotLoanStatus(loan, 'issue_d_month')
plt.figure(figsize=(10, 6))

plotLoanStatus(loan, 'issue_d_year')
fig, ax = plt.subplots(figsize=(10,5))

loan.boxplot(column=['int_rate(in %)'],by='grade', ax=ax)
fig, ax = plt.subplots(figsize=(10,5))

loan.boxplot(column=['int_rate(in %)'],by='sub_grade', ax=ax)
fig, ax = plt.subplots(figsize=(10,5))

loan.boxplot(column=['int_rate(in %)'],by='loan_status', ax=ax)
fig, ax = plt.subplots(figsize=(15,5))

loan.boxplot(column=['int_rate(in %)'],by='purpose', ax=ax)

plt.xticks(rotation='vertical')
fig, ax = plt.subplots(figsize=(10,5))

ax.set_yscale("log")

loan.boxplot(column=['annual_inc'],by='grade', ax=ax)
fig, ax = plt.subplots(figsize=(10,5))

ax.set_yscale("log")

loan.boxplot(column=['annual_inc'],by='purpose', ax=ax)

plt.xticks(rotation='vertical')
fig, ax = plt.subplots(figsize=(10,5))

# ax.set_yscale("log")

loan.boxplot(column=['dti'],by='purpose', ax=ax)

plt.xticks(rotation='vertical')
plt.figure(figsize=(8,6))

sns.barplot(x='term', y='loan_amnt', hue='loan_status',data=loan, estimator=np.mean)

plt.show()
plt.figure(figsize=(20,10))

sns.barplot(x='term', y='loan_amnt', hue='purpose',data=loan, estimator=np.mean)

plt.show()
plt.figure(figsize=(20,10))

sns.barplot(x='term', y='loan_amnt', hue='home_ownership',data=loan, estimator=np.mean)

plt.show()
plt.figure(figsize=(20,10))

ax = sns.barplot(x='term', y='loan_amnt', hue='grade',data=loan, estimator=np.mean)

plt.show()
plt.figure(figsize=(8,6))

sns.barplot(x='home_ownership', y='loan_amnt', hue='loan_status',data=loan, estimator=np.mean)

plt.show()
plt.figure(figsize=(8,6))

sns.barplot(x='home_ownership', y='annual_inc', hue='grade',data=loan, estimator=np.mean)

plt.show()
plt.figure(figsize=(15,6))

sns.barplot(x='purpose', y='annual_inc', hue='loan_status',data=loan, estimator=np.mean)

plt.xticks(rotation='vertical')

plt.show()
plt.figure(figsize=(18,6))

sns.barplot(x='purpose', y='loan_amnt', hue='loan_status',data=loan, estimator=np.mean)

plt.xticks(rotation='vertical')

plt.show()
plt.figure(figsize=(8,6))

sns.barplot(x='grade', y='loan_amnt', hue='loan_status',data=loan, estimator=np.mean)

plt.show()
plt.figure(figsize=(8,6))

sns.barplot(x='verification_status', y='loan_amnt', hue='loan_status',data=loan, estimator=np.mean)

plt.show()
plt.figure(figsize=(8,6))

sns.barplot(x='grade', y='annual_inc', hue='verification_status',data=loan, estimator=np.mean)

plt.show()
plt.figure(figsize=(10,6))

sns.barplot(x='purpose', y='int_rate(in %)', hue='loan_status',data=loan, estimator=np.mean)

plt.xticks(rotation='vertical')

plt.show()