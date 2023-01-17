# Import lib. for data analysis

import pandas as pd

import numpy as np



import seaborn as sns

import matplotlib.pyplot as plt



import warnings

warnings.filterwarnings('ignore')





from scipy.stats import ttest_ind

loan = pd.read_csv('../input/loan.csv')
# delete column having 80% of missing values

missing_columns = loan.columns[loan.isnull().sum()/len(loan.index)*100 > 80]

print(missing_columns)
# Dropping all the columns with null values

loan = loan.drop(missing_columns, axis=1)

print(loan.shape)
# All urls are unique. Not required for the analysis. Dropping this column

print(loan['url'].head())

loan = loan.drop('url', axis=1)
# It is string var. Does not seem like it can be used untill key words can be extracted. 

print(loan['desc'].head())

loan = loan.drop('desc', axis=1)
# Removing these columns as they have single value

print(loan['initial_list_status'].unique())

print(loan['collections_12_mths_ex_med'].unique())

print(loan['policy_code'].unique())

print(loan['application_type'].unique())

print(loan['acc_now_delinq'].unique())

print(loan['chargeoff_within_12_mths'].unique())

print(loan['delinq_amnt'].unique())

print(loan['tax_liens'].unique())

print(loan['pymnt_plan'].unique())



loan = loan.drop(['initial_list_status',

                  'collections_12_mths_ex_med',

                  'policy_code',

                  'application_type',

                  'acc_now_delinq',

                  'chargeoff_within_12_mths',

                  'delinq_amnt',

                  'tax_liens',

                  'pymnt_plan'], axis=1)
# Removing member_id column

loan = loan.drop('member_id', axis=1)
loan.shape
# Convert term column to int

print(loan['term'].unique())

loan['term'] = loan['term'].apply(lambda x : 36 if x==' 36 months' else 60)

print(loan['term'].unique())
# convert int_rate to float

print(loan['int_rate'].head(2))

loan['int_rate'] = loan['int_rate'].apply(lambda x : float(x[0:-1]))

print(loan['int_rate'].head(2))
# Convert emp_length to int

loan['emp_length'].unique()

emp_replace_dict = {'10+ years':10, '< 1 year':0, '1 year':1, 

                    '3 years':3, '8 years':8, '9 years':9,'4 years':4,

                    '5 years':5, '6 years':6, '2 years':2, '7 years':7}



loan['emp_length'].replace(emp_replace_dict, inplace=True)

loan['emp_length'].head()
# Convert revol_util into int

loan['revol_util'] = loan['revol_util'].replace(np.nan, '', regex=True)

loan['revol_util'] = loan['revol_util'].apply(lambda x : float(x[0:-1]) if len(x) > 0 else np.nan )
# One hot encoding for loan status

loan_status_dummies = pd.get_dummies(loan['loan_status'])

print(loan_status_dummies.head())



loan = pd.concat([loan, loan_status_dummies], axis=1)
# Analyze correlation between numeric data types

numerics = ['uint8','int16', 'int32', 'int64', 'float16', 'float32', 'float64']

loan_numeric = loan.select_dtypes(include=numerics)

corr = loan_numeric.corr()



# Plot the heat map

fig, ax = plt.subplots(figsize=(22,22))       

sns.heatmap(corr, 

            xticklabels=corr.columns,

            yticklabels=corr.columns,

            fmt=".1f",

            annot=True,

            ax = ax)
# We conclude from above heapmap that these columns have positive correlation.

# 1. loan_amt

# 2. funded_amnt

# 3. funded_amnt_inv

# 4. installment

# 5. total_pymnt

# 6. total_pymnt_inv

# 7. total_rec_prncp

# 8. total_rec_int



corr = loan_numeric.loc[:, ['loan_amnt', 'funded_amnt', 'funded_amnt_inv',

             'installment', 'total_pymnt', 'total_pymnt_inv',

             'total_rec_prncp', 'total_rec_int']].corr()



fig, ax = plt.subplots(figsize=(6,6))       



sns.heatmap(corr, 

            xticklabels=corr.columns,

            yticklabels=corr.columns,

            annot=True,

            ax = ax)
# out_prncp and out_prncp_inv have corr value. They are related.

print(loan_numeric['out_prncp'].corr(loan_numeric['out_prncp_inv']))
# pub_rec and pub_rec_bankruptcies have corr value. They are related.

loan_numeric['pub_rec'].corr(loan_numeric['pub_rec_bankruptcies'])
# recoveries and collection_recovery_fee have corr value. They are related.

loan_numeric['recoveries'].corr(loan_numeric['collection_recovery_fee'])
# delinq_2yrs and mths_since_last_delinq have negative corr value. They are related.

loan_numeric['delinq_2yrs'].corr(loan['mths_since_last_delinq'])
# Remove the 'Current' loan status as it is not relevant in the current context

loan_subset_df = loan.loc[loan['loan_status'] != 'Current']

# Helper function to draw plots for numerical data

class color:

   PURPLE = '\033[95m'

   CYAN = '\033[96m'

   DARKCYAN = '\033[36m'

   BLUE = '\033[94m'

   GREEN = '\033[92m'

   YELLOW = '\033[93m'

   RED = '\033[91m'

   BOLD = '\033[1m'

   UNDERLINE = '\033[4m'

   END = '\033[0m'



def showPlots(tips, colname, scale, lq, rq, title):

    tips_copy = tips.loc[(tips.loc[:, colname] > 0) & 

                         (tips.loc[:,colname] <= tips.loc[:, colname].quantile(rq)) &

                         (tips.loc[:,colname] >= tips.loc[:, colname].quantile(lq))]

        

    if scale == "log":

        tips_copy.loc[:, colname] = np.log(tips_copy.loc[:, colname])

    fig, axs = plt.subplots(ncols=2, figsize=(20, 6))

    ax1 = sns.distplot(tips_copy[colname], ax=axs[0]);

    ax1.set(xlabel=title, ylabel='Fraction', title=title + " distribution")

    ax2=sns.boxplot(x="loan_status", y=colname, data=tips_copy, ax=axs[1])

    ax2.set(xlabel="Loan Status", ylabel=title, title=title + " v/s Loan Status")

    

def showNumericalPlots(**kwargs):  

    showPlots(kwargs['data'],

              kwargs['colname'],

              kwargs['scale'],

              kwargs['left_quantile'],

              kwargs['right_quantile'],

              kwargs['title']

             )

    

def compareLoanStatus(data, colname):

    print(color.BLUE + color.BOLD + 'Charged Off data' + color.END)

    print(data.loc[data['loan_status'] == 'Charged Off'][colname].describe())

    print('')

    print(color.RED + color.BOLD + 'Fully paid data' + color.END)

    print(data.loc[data['loan_status'] == 'Fully Paid'][colname].describe())

    

def performTTest(data, colname):

    print('')

    print(color.GREEN + color.BOLD + 't-score' + color.END)

    print(ttest_ind(data.loc[data['loan_status'] == 'Charged Off'][colname], 

          data.loc[data['loan_status'] == 'Fully Paid'][colname]))
# Helper function to draw plots for categorical data

def showCategoricalPlots(tips, colname, width, xlabel):

    tips_copy = tips.groupby([colname, 'loan_status'])[colname].count().unstack()

    tips_copy['Charged Off %'] = (tips_copy['Charged Off'] / (tips_copy['Charged Off'] + tips_copy['Fully Paid'])) * 100

    tips_copy.sort_values(by="Charged Off %", ascending=False, inplace=True)

    plot = tips_copy.loc[:, ['Charged Off %']].plot.bar(figsize=(width, 4))

    plot.set(xlabel=xlabel, ylabel="Charged Off %", title=xlabel + " v/s Loan Status")

    plot.spines['top'].set_visible(False)

    plot.spines['right'].set_visible(False)

    print(tips_copy.head(10))

    

# Helper function to draw plots for categorical data

def showCategoricalPlotsStacked(tips, colname, width, xlabel):

    tips_copy = tips.groupby([colname, 'loan_status'])[colname].count().unstack().reset_index()

    tips_copy['Fully Paid %'] = (tips_copy['Fully Paid'] / (tips_copy['Charged Off'] + tips_copy['Fully Paid'])) * 100

    tips_copy['Charged Off %'] = (tips_copy['Charged Off'] / (tips_copy['Charged Off'] + tips_copy['Fully Paid'])) * 100

    tips_copy.sort_values(by="Charged Off %", ascending=False, inplace=True)

    ax = tips_copy.loc[:, [colname, 'Fully Paid %', 'Charged Off %']].set_index(colname).plot(kind='bar', stacked=True, figsize=(width, 4))

    ax.set(xlabel=xlabel, ylabel='Loan Status %', title=xlabel + " v/s Loan Status")

    ax.spines['top'].set_visible(False)

    ax.spines['right'].set_visible(False)

    print(tips_copy.loc[:,[colname, 'Charged Off', 'Fully Paid',  'Charged Off %']].set_index(colname).head(10))

# Draw dist plot for int_rate and compare the boxplot for Fully paid and Charged off

showNumericalPlots(data=loan_subset_df,

                   colname='int_rate',

                   left_quantile=0.0,

                   right_quantile=1.0,

                   scale='linear',

                   title='Interest Rate')





# Hypothesis : The int_rate variable has higher median for charged_off loan status

compareLoanStatus(loan_subset_df, 'int_rate')



# median [charged off] = 13.61%

# median [fully paid] = 11.49%



performTTest(loan_subset_df, 'int_rate')



# t-score = 42.48

# Large t-score confirms that the int_rate for defaulters [Fully Paid] and 

# non-defaulters [Charged Off] are different.



# Inference :  In int_rate is higher, then probability of default is higher.

# Draw dist plot for annual_inc and compare the boxplot for Fully paid and Charged off

# Remove outlier and scale to log

showNumericalPlots(data=loan_subset_df,

                   colname='annual_inc',

                   left_quantile=0.1,

                   right_quantile=0.9,

                   scale='log',

                   title='Annual Income')



# Hypothesis : The annual_inc variable has higher median for charged_off loan status

compareLoanStatus(loan_subset_df, 'annual_inc')



# median [charged off] = $53,000

# median [fully paid] = $60,000



# Inference :  If annual_inc is lower, then probability of default is higher.



# Calculate mean of total_rec_late_fee  

# Remove outliers whose fees > $50

loan_subset_df_late_fee = loan_subset_df.loc[(loan_subset_df['total_rec_late_fee'] < 50)]

print(loan_subset_df_late_fee.groupby(['loan_status'])['total_rec_late_fee'].mean())



fig, axs = plt.subplots(ncols=2, figsize=(15, 6))



ax0 = sns.distplot(loan_subset_df_late_fee.loc[loan_subset_df_late_fee['loan_status'] == 'Fully Paid']['total_rec_late_fee'],

             ax=axs[0],

             color="blue",

             hist=False);

ax0.set(xlabel='Received Late Fees', ylabel='%', title='Late free distribution for Fully paid burrowers')



ax1 = sns.distplot(loan_subset_df_late_fee.loc[loan_subset_df_late_fee['loan_status'] == 'Charged Off']['total_rec_late_fee'],

             ax=axs[1], 

             color='red',

             hist=False);



ax1.set(xlabel='Received Late Fees', ylabel='%', title='Late free distribution for Charged off burrowers')





# Calclate number of defaulters with fees > $0

print(loan_subset_df.loc[(loan_subset_df['total_rec_late_fee'] > 0) & 

                   (loan_subset_df_late_fee['loan_status'] == 'Charged Off')].shape)



# Calclate number of defaulters with fees >= $15

print(loan_subset_df.loc[(loan_subset_df['total_rec_late_fee'] >= 15) & 

                   (loan_subset_df_late_fee['loan_status'] == 'Charged Off')].shape)



# Calclate number of non-defaulters with fees > $0

print(loan_subset_df.loc[(loan_subset_df['total_rec_late_fee'] > 0) & 

                   (loan_subset_df_late_fee['loan_status'] == 'Fully Paid')].shape)



# Calclate number of non-defaulters with fees >= $15

print(loan_subset_df.loc[(loan_subset_df['total_rec_late_fee'] >= 15) & 

                   (loan_subset_df_late_fee['loan_status'] == 'Fully Paid')].shape)





# Observation :  The mean total late fee recieved for fully paid burrowers is sigficantly less that those

#                who were charged off. 

#

#                1. In the below graph, atleast 23% of defaulters have paid $15 or more while only 6%

#                   non-defaulters have paid $15.

#                2. Mean for Charged Off is $3.15 while mean for Fully Paid is $0.68.

#                3. There are 13% Defaulters with late fees > $0 in comparison to 3% such non-defaulters.

#                4. There are 9% defaulters with late fess > $15 in comarison to 2.5% such non-defaulters.



# Inference   : Those who have high late fee sum (>= $15) will have high rate of default.

# Scatter plot between loan_amnt and total_pymnt

ax = sns.scatterplot(x="loan_amnt", 

                y="total_pymnt",

                data=loan_subset_df,

                hue='loan_status')



ax.set(xlabel='Loan Amount', ylabel='Total Payment', title='Loan Amount v/s Total Payment')



# Draw dist plot for loan_amnt and compare the boxplot for Fully paid and Charged off

# Remove outlier and scale to log

showNumericalPlots(data=loan_subset_df,

                   colname='loan_amnt',

                   left_quantile=0.1,

                   right_quantile=0.9,

                   scale='log',

                   title='Loan Amount')



# Draw dist plot for total_pymnt and compare the boxplot for Fully paid and Charged off

# Remove outlier and scale to log

showNumericalPlots(data=loan_subset_df,

                   colname='total_pymnt',

                   left_quantile=0.1,

                   right_quantile=0.9,

                   scale='log',

                   title='Total Payment')



# Calculate median

print(loan_subset_df.groupby(['loan_status'])['loan_amnt'].median())

print(loan_subset_df.groupby(['loan_status'])['total_pymnt'].median())





# Observation :  The mean total payment recieved for charged paid burrowers is sigficantly less that those

#                whofully paid. 

#

#                1. In the below scatter plot, despite having same loan amount, the total payment for

#                   the defaulters is lesser than the non-defaulters.

#                2. The median for loan_amnt is almost same for Charged Off ($9,600) and Fully Paid ($10,000) but 

#                   total_pymnt median for Charged Off is only $4842 compared to $10687 for Fully Paid



# Inference   : Those who have less total_pymnt for the same loan_amnt will have high possibility of being

#               Charged Off.

# Scatter plot between loan_amnt and total_rec_prncp

ax = sns.scatterplot(y="total_rec_prncp", 

                     x="loan_amnt",

                     data=loan_subset_df,

                     hue='loan_status')



ax.set(xlabel='Loan Amount', ylabel='Total Principle Recieved', title='Loan Amount v/s Total Principle Recieved')



# Draw dist plot for total_rec_prncp and compare the boxplot for Fully paid and Charged off

# Remove outlier and scale to log

showNumericalPlots(data=loan_subset_df,

                   colname='total_rec_prncp',

                   left_quantile=0.1,

                   right_quantile=0.9,

                   scale='linear',

                   title='Principle Recieved')



# Calculate median

print(loan_subset_df.groupby(['loan_status'])['total_rec_prncp'].median())



# median [charged off] = $2,730

# median [fully paid] = $9,200



# Observation :  The median total principle recieved for charged paid burrowers is sigficantly less that those

#                whofully paid. 

#

#                1. In the below scatter plot, despite having same loan amount, the total principle recieved

#                   from the defaulters is lesser than the non-defaulters.

#                2. The median for loan_amnt is almost same for Charged Off ($9,600) and Fully Paid ($10,000) but 

#                   total_rec_prncp median for Charged Off is only $2730 compared to $9200 for Fully Paid



# Inference :  Those who have less total_rec_prncp for the same loan_amnt will have high possibility of being

#              Charged Off.
# Create a derived column last_pymnt_d_year

loan_subset_df.loc[:, 'last_pymnt_d'] = loan_subset_df['last_pymnt_d'].replace(np.nan, '', regex=True)

loan_subset_df.loc[:, 'last_pymnt_d_year'] = loan_subset_df['last_pymnt_d'].apply(lambda x : (x[-2:]) if len(x) > 0 else np.nan )

showCategoricalPlotsStacked(loan_subset_df, 'last_pymnt_d_year', 10, 'Last payment year')



# Observation : The Charged off % is higher as the last payment date are earlier.

# Inference :   If the last payment date of a loan is older, then there is a high probability

#               that it will be charged off.

showCategoricalPlots(loan_subset_df, 'verification_status', 5, 'Verification Status')



# Observation : The Charged off % is higher if the customer is Verified by LC.

# Inference :   If the customer is Verified, the probability of him defaulting is high.

plot_df = showCategoricalPlotsStacked(loan_subset_df, 'purpose', 6, 'Purpose of the loan')



# Observation : The Charged off % is dependent on the purpose of the loan. There is a significantly high 

#               default rate for small_business

# Inference :   If the burrower is taking the loan for Small Business, then changes of default are high 

#               than any other purpose.

# Removing NE state (outlier) due to it's unusual high % of Charged Off.

loan_subset_df = loan_subset_df.loc[loan_subset_df['addr_state'] != 'NE']

showCategoricalPlotsStacked(loan_subset_df, 'addr_state', 11, 'Address State')



# Observation : The Charged off % is also dependent on the state of the burrower. There is a high 

#               default rate for NV state.

# Inference :   If the burrower is taking the loan belongs to NV the he has 22% change of defaulting.

df_grp = loan.groupby('loan_status')['loan_status'].count()

print(df_grp)



# Charged off (default) % = 14.60



df_grp.plot.bar()
# Analyse grade and sub_grade vs loan_status



df_grp = loan_subset_df.groupby(['grade', 'loan_status'])['loan_status'].count().unstack().reset_index('grade')

df_grp['Charged Off%'] = df_grp['Charged Off'] / (df_grp['Charged Off'] + df_grp['Fully Paid']) * 100

df_grp['Fully Paid%'] = df_grp['Fully Paid'] / (df_grp['Charged Off'] + df_grp['Fully Paid']) * 100

ax1 = df_grp.loc[:, ['grade', 'Fully Paid%', 'Charged Off%']].set_index('grade').plot(kind='bar', stacked=True, figsize=(8, 4))

ax1.set(xlabel='Grade', ylabel='Loan Status %', title='Influence of Grade on Loan Status')

ax1.patch.set_facecolor('white')

ax1.spines['top'].set_visible(False)

ax1.spines['right'].set_visible(False)



df_grp = loan_subset_df.groupby(['sub_grade', 'loan_status'])['loan_status'].count().unstack().reset_index('sub_grade')

df_grp['Charged Off%'] = df_grp['Charged Off'] / (df_grp['Charged Off'] + df_grp['Fully Paid']) * 100

df_grp['Fully Paid%'] = df_grp['Fully Paid'] / (df_grp['Charged Off'] + df_grp['Fully Paid']) * 100

ax2 = df_grp.loc[:, ['sub_grade', 'Fully Paid%', 'Charged Off%']].set_index('sub_grade').plot(kind='bar', stacked=True, figsize=(12, 4))

ax2.set(xlabel='Sub-Grade', ylabel='Loan Status %', title='Influence of Sub-Grade on Loan Status')

ax2.spines['top'].set_visible(False)

ax2.spines['right'].set_visible(False)



# Observation : The Charged off % is also dependent on the grade and sub-grade of the burrower. There is a high 

#               default rate if grade or sub_grade are lower.

#               Grade: Here A is better than B better than C and so on

#               Sub-grade: Here A1 is better than A2 better than A3 and so on.



# Inference :   If the burrower is taking the loan belongs to the lower grade then he has high change of defaulting.

showCategoricalPlots(loan_subset_df, 'term', 6, 'Loan term')



# Observation : The Charged off % is also dependent on the term. There is a high 

#               default rate if term is 60.



# Inference :   If the burrower is taking the loan for 60 months, there is 25% chance of defaulting.

showCategoricalPlots(loan_subset_df, 'emp_length', 8, 'Loan term')



# Observation : The Charged off % is increasing as the emp_length is increading, with a few exceptions



# Inference :   If the burrower is taking the loan whose emp_length is higher, there may be high chance of defaulting.
# Created 4 installment bins with 25% data in each

# Low        => [0, 165]

# Medium     => [166, 275]

# High       => [276, 420]

# Very High  => > 420

def createInstallmentBins(n):

    if n <= 165:

        return 'Low'

    elif n > 165 and n <=275:

        return 'Medium'

    elif n > 275 and n <=420:

        return 'High'

    else:

        return 'Very high'

    

loan_subset_df['installment_bin'] = loan_subset_df['installment'].apply(lambda x: createInstallmentBins(x))

showCategoricalPlots(loan_subset_df, 'installment_bin', 8, 'Installments')



# Inference :   If the burrower is taking the loan whose installment is higher ( > $420), 

#               there may be high chance of defaulting.
# Created 4 loan amount bins with 25% data in each

# Low        => [0, 5500)

# Medium     => [5500, 9600)

# High       => [9600, 15000)

# Very High  => > 15000

def createLoanAmountBins(n):

    if n < 5500:

        return 'Low'

    elif n >=5500 and n < 9600:

        return 'Medium'

    elif n >= 9600 and n < 15000:

        return 'High'

    else:

        return 'Very high'

    

loan_subset_df['loan_amnt_bin'] = loan_subset_df['loan_amnt'].apply(lambda x: createLoanAmountBins(x))

showCategoricalPlots(loan_subset_df, 'loan_amnt_bin', 8, 'Loan amount')



# Inference :   If the burrower is taking the loan whose loan amount is higher ( > $15000), 

#               there may be high chance of defaulting.
# Created 4 annual income bins with 25% data in each

# Low        => [0, 41000)

# Medium     => [41000, 60000)

# High       => [60000, 83500)

# Very High  => > 83500

def createAnnualIncomeBin(n):

    if n < 41000:

        return 'Low'

    elif n >= 41000 and n < 60000:

        return 'Medium'

    elif n >= 60000 and n < 83500:

        return 'High'

    else:

        return 'Very high'

    

loan_subset_df['annual_inc_bin'] = loan_subset_df['annual_inc'].apply(lambda x: createAnnualIncomeBin(x))

showCategoricalPlots(loan_subset_df, 'annual_inc_bin', 8, 'Annual income')



# Inference :   If the burrower is taking the loan whose annual income is lower ( < $41000), 

#               there may be high chance of defaulting.
# Debt to income ratio

# Created 4 dti bins with 25% data in each

# Low        => [0, 8)

# Medium     => [8, 13.2)

# High       => [13.2, 18.5)

# Very High  => > 18.5

def createDTIBin(n):

    if n < 8:

        return 'Low'

    elif n >= 8 and n < 13.2:

        return 'Medium'

    elif n >= 13.2 and n < 18.5:

        return 'High'

    else:

        return 'Very high'

 

loan_subset_df['dti_bin'] = loan_subset_df['dti'].apply(lambda x: createDTIBin(x))

showCategoricalPlots(loan_subset_df, 'dti_bin', 8, 'DTI')



# Inference :   If the burrower is taking the loan whose dti is higher ( >= 18.5), 

#               there may be high chance of defaulting.