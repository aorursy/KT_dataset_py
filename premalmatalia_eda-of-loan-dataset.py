import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

import plotly.express as px

from datetime import date, datetime

from scipy.stats import chi2_contingency

from scipy.stats import chi2

#loan=pd.read_csv('loan.csv',sep=',', skipinitialspace=False,low_memory=False)

loan = pd.read_csv("../input/loan.csv",sep=',', skipinitialspace=False,low_memory=False)
pd.options.display.float_format = "{:.2f}".format # setting the display for avoind exponential based representation
loan.head(n=20)

loan.shape # (39717, 111)--> Before Cleaning

#print(loan.info())
loan.dropna(axis=1,how='all',inplace=True)

print("\n Shape after dropping all the columns having all the NaN")

print(loan.shape) #(39717, 57)



print("\n Check columns for % of null values \n")

print(round(100*(loan.isnull().sum()/len(loan.index)).sort_values(ascending=False).head(10), 2))



loan.drop(columns=['next_pymnt_d','mths_since_last_record','mths_since_last_delinq'],inplace=True)



print("\n Shape after dropping 3 columns with high % of null values")

print(loan.shape)
drop_columns_1 = ['id','member_id','sub_grade','application_type','chargeoff_within_12_mths','collections_12_mths_ex_med',

                  'delinq_amnt','initial_list_status','policy_code','pymnt_plan','tax_liens']



loan.drop(drop_columns_1,axis='columns',inplace=True)

print("Shape after dropping above columns")

loan.shape 
drop_columns_2 = ['collection_recovery_fee','desc','emp_title','last_credit_pull_d',

                'last_pymnt_amnt','last_pymnt_d', 'recoveries','revol_bal','title','total_pymnt','total_pymnt_inv',

                'total_rec_int','total_rec_late_fee','total_rec_prncp','url','zip_code','out_prncp', 'out_prncp_inv', 

                 'acc_now_delinq', 'revol_util', 'delinq_2yrs', 'open_acc', 'total_acc']



loan.drop(drop_columns_2,axis='columns',inplace=True)

print("Shape after dropping above columns")

loan.shape
loan = loan[(loan.loan_status == 'Fully Paid') | (loan.loan_status == 'Charged Off') ]
loan['int_rate'] = loan['int_rate'].str.replace('%','').astype('float')
loan['emp_length'] = loan['emp_length'].str.replace('years','').str.replace('year','').str.strip()

loan['emp_length'] = loan['emp_length'].replace('10+', '10').replace('< 1', '0').astype('float')
loan.term = loan.term.str.strip() # removing whitespaces
loan.to_csv('clean_dataset.csv')
def univariate_continuous(dfc,col,title):

    

    '''Univariate function to plot the graphs for continuous variables based on the parameters provided

        Parameter Details:

        dfc- dataframe name for continuous variable

        col- Column name to be analyzed

        title - Plot title

    '''

    sns.set(context='paper')

    plt.figure(figsize=[9.0,6.0],dpi=100,edgecolor='w',frameon=True)

    plt.subplot(221)

    title_1 = title + ' - Distribution Plot '

    plt.title(title_1,pad=7)

    chart = sns.distplot(dfc[col].dropna(),norm_hist=False)



    plt.subplot(222)

    title_2 = title + ' - Box Plot'

    plt.title(title_2,pad=7)

    chart = sns.boxplot(data=dfc,x=col,orient='v')

    plt.tight_layout()
def univariate_categorical(dfc,col,title,orient ='v',size ='medium',val_cnt_limit =0):

    

    '''Univariate function to plot the graphs for continuous variables based on the parameters provided

        Parameter Details:

        dfc- dataframe name for continuous variable

        col- Column name to be analyzed

        title - Plot title

        size - Plot size

            medium = figsize of [4.0,4.0]

            large  = figsize of [10.0,10.0]

        val_cnt_limit - only value_counts above this limit will be considered

    '''

    dfc=dfc.loc[:,[col]]

    dfc= pd.DataFrame(dfc[col].value_counts().rename('count').reset_index())

    dfc.columns = [col,'count']

    if val_cnt_limit >= 0:

        dfc = dfc.loc[dfc['count'] >= val_cnt_limit]

    #print(loan_grade.head()

    

    if size == 'large':

        plt.figure(figsize=[10.0,10.0],dpi=100,frameon=True)

    else:

        plt.figure(figsize=[4.0,4.0],dpi=100,frameon=True)

        

    if orient == 'v':

        chart = sns.barplot(x=col, y='count', data=dfc,orient=orient)

        plt.ylabel("Count")

        plt.xlabel(col)

        plt.title(title)

    else:

        chart = sns.barplot(x='count', y=col, data=dfc,orient=orient)

        plt.xlabel("Count")

        plt.ylabel(col)

        plt.title(title)

    plt.show()
## Check the counts of unique values for each loan_status category

loan_status_dist = loan.loan_status.value_counts()

loan_status_dist.head()
labels = ['Fully Paid','Charged Off']

loan_status_perc = [loan_status_dist['Fully Paid'], loan_status_dist['Charged Off'] ]



explode = (0, 0.1)  # only "explode" the  slice 'venture'

plt.figure(figsize=[4.0,4.0],dpi=100,edgecolor='w',frameon=True)

plt.pie(loan_status_perc, explode= explode, labels=labels, autopct='%1.1f%%',shadow=True, startangle=0)

plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()
pd.options.display.float_format = '{:.2f}'.format

loan_annual_inc = loan.loc[:,['annual_inc']]

print(loan_annual_inc.describe(percentiles=[0.1,0.25,0.5,0.75,0.90,0.95,0.98,0.99]))
loan_annual_inc['annual_inc'] = loan_annual_inc[loan_annual_inc['annual_inc'] < loan_annual_inc['annual_inc'].quantile(0.99)]

print(loan_annual_inc.annual_inc.describe())
univariate_continuous(loan_annual_inc,'annual_inc','Annual Income')
print(loan.funded_amnt.describe(percentiles=[0.1,0.25,0.5,0.75,0.90,0.95,0.98,0.99]))
univariate_continuous(loan,'funded_amnt','Funded Amount')
print(loan.int_rate.describe(percentiles=[0.1,0.25,0.5,0.75,0.90,0.95,0.98,0.99]))
univariate_continuous(loan,'int_rate','Interest Rate')
print(loan.dti.describe(percentiles=[0.1,0.25,0.5,0.75,0.90,0.95,0.98,0.99]))
univariate_continuous(loan,'dti','dti')
univariate_categorical(loan,'grade','Loan Grade vs. Loan Count','v')
univariate_categorical(loan,'addr_state','State vs. Loan Count',size = 'large', val_cnt_limit=400)
univariate_categorical(loan,'purpose','Purpose vs. Loan Count','h',val_cnt_limit=1000)
univariate_categorical(loan,'inq_last_6mths','inq_last_6mths vs. Loan Count')
loan['monthly_inc'] = loan['annual_inc'] / 12 # Monthly income, type driven metric

# new DTI ratio of installmenet amout to monthly income additional to the existing DTI

loan['newDTI'] = (loan['installment']  / loan['monthly_inc']) + (loan.dti/100)  

univariate_continuous(loan, 'newDTI', 'newDTI')
loan['earliest_cr_line_year'] = loan['earliest_cr_line'].str.split('-').str[1].astype('int')

loan['issue_d_year'] = loan['issue_d'].str.split('-').str[1].astype('int')

# years are represented as 2 digit numbers ,changes due to y2k needs to be corrected

# Current year 2020 is considered as reference for calculating no of year of credit history

loan['earliest_cr_line_age'] = loan['earliest_cr_line_year'].apply(lambda x:20-x if(x<20) else 120-x) 

loan['issue_d_age'] = loan['issue_d_year'].apply(lambda x:20-x if(x<20) else 120-x) 

loan['credit_history'] = loan['earliest_cr_line_age'] - loan['issue_d_age']



univariate_continuous(loan, 'credit_history', 'Crdeit History')
loan["inq_last_6mths_recat"] = loan["inq_last_6mths"].apply(lambda x:3 if(x>3) else x)

loan["inq_last_6mths_recat"].value_counts()
#int_rate_bins = list(np.linspace(loan.int_rate.min(),loan.int_rate.max(), 11)) # creating 10 bins

loan['int_rate_binned'] = pd.cut(loan['int_rate'], 10)
loan['annual_inc_binned'] = pd.qcut(loan['annual_inc'], q=8, precision=0) # segmenting based on percentile

loan['funded_amnt_binned'] = pd.qcut(loan['funded_amnt'], q=8, precision=0) # segmenting based on percentile

loan['dti_binned'] = pd.qcut(loan['dti'], 8)

loan['newDTI_binned'] = pd.qcut(loan['newDTI'], 8) ## new DTI

loan['years_credit_history_binned'] = pd.qcut(loan['credit_history'], 8)

#Segmenting rows with loan_status = 'Charged off' and 'Fully Paid'

loanchargedoff = loan[loan.loan_status == 'Charged Off'] 

loanfullypaid = loan[loan.loan_status == 'Fully Paid']
def segmented_univariate_analysis(col, title, orient='v', size='large', fillzero=False):

    

    # generting a data frame with defualt ratio as a coloumn

    

    if(fillzero == True):

        col_dist = (loanchargedoff[col].value_counts() / loan[col].value_counts()).to_frame(name='default_ratio').rename_axis(col).reset_index().fillna(0)

    else:

        col_dist = (loanchargedoff[col].value_counts() / loan[col].value_counts()).to_frame(name='default_ratio').rename_axis(col).reset_index()

    if((size=='large') & (orient == 'v') ):

        fig = plt.figure(figsize=(15, 6))

    elif((size=='medium') & (orient == 'v') ):

        fig = plt.figure(figsize=(12, 6))

    elif(orient == 'v') :

        fig = plt.figure(figsize=(8, 5))        

    

    if orient == 'v':

        g = sns.barplot(x=col, y='default_ratio', data=col_dist)

        plt.ylabel("Loan default ratio")

        plt.xlabel(col)

        for index, row in col_dist.iterrows():

            g.text( row.name ,row.default_ratio, round(row.default_ratio,2), color='black', ha="center")

    else:

        fig = plt.figure(figsize=(8, 10))   

        g = sns.barplot(x='default_ratio', y=col, data=col_dist)

        plt.xlabel("Loan default ratio")

        plt.ylabel(col)

        for index, row in col_dist.iterrows():

            g.text( row.default_ratio, row.name , round(row.default_ratio,2), color='black', ha="center")        

    g.set_title(title + " Vs Loan defualt ratio")



    plt.show()
segmented_univariate_analysis('int_rate_binned', "Interest")
#loan.term.value_counts(normalize=True)
#loan.groupby(by=['term']).loan_status.value_counts() 
segmented_univariate_analysis('term', "Loan Term", size='small')

segmented_univariate_analysis('grade', "Loan Grade", size='small')
#loan.groupby(by=['pub_rec_bankruptcies']).loan_status.value_counts() 
segmented_univariate_analysis('pub_rec_bankruptcies', "No of public record bankruptcies", size='small')
#loan.groupby(by=['pub_rec']).loan_status.value_counts() 
segmented_univariate_analysis('pub_rec', "No of derogatory public records", size='small', fillzero=True)
segmented_univariate_analysis('dti_binned', "DTI", size='medium')
segmented_univariate_analysis('newDTI_binned', "newDTI", size='medium')
#loanchargedoff.annual_inc.describe()
#loanfullypaid.annual_inc.describe()
segmented_univariate_analysis('annual_inc_binned', "Annual Income", size='medium')
segmented_univariate_analysis('funded_amnt_binned', "Funded Loan amount", size='medium')
loanfullypaid_adr_df = pd.DataFrame({'state':loanfullypaid.addr_state.value_counts().index, 'fullypaid':loanfullypaid.addr_state.value_counts().values})

loanchargedoff_adr_df = pd.DataFrame({'state':loanchargedoff.addr_state.value_counts().index, 'chargedoff':loanchargedoff.addr_state.value_counts().values})



loanstate_df = pd.merge(loanfullypaid_adr_df, loanchargedoff_adr_df,how='outer', on='state')

loanstate_df = loanstate_df[loanstate_df.chargedoff > 10] ## required to avoid outliers, states with very no of loan accounts

loanstate_df['default_ratio'] = loanstate_df.chargedoff/loanstate_df.fullypaid

# loanstate_df
#fig = plt.figure(figsize=(15, 12))

#g = sns.barplot(y="state", x="default_ratio", data=loanstate_df)

#plt.show()
fig = px.choropleth(locations=list(loanstate_df.state), locationmode="USA-states", color=loanstate_df.default_ratio,scope="usa")

fig.show()
#loan.groupby(by=['purpose']).loan_status.value_counts() 
segmented_univariate_analysis('purpose', "Loan purpose", orient='h')
segmented_univariate_analysis('verification_status', "Verification Status", size='small')
segmented_univariate_analysis('home_ownership', "Home ownership", size='small', fillzero=True)
#loan.emp_length.value_counts()
segmented_univariate_analysis('emp_length', "Employment Legth", size='small')
#loan.years_credit_history_binned.value_counts()
segmented_univariate_analysis('years_credit_history_binned', "Credit History(years)", size=',medium')
segmented_univariate_analysis('inq_last_6mths_recat', "No of Inquiries in 6 months", size='small')
def Category_dependancy_test(cat1,cat2):

    

    '''chi2 test is performed to decide whether 2 categorical variables are dependant or not

    '''

    prob = 0.99

    test_df = pd.crosstab(cat1, cat2) # generating contigency table for chi2 test

    #print(test_df)

    stat, p, dof, expected = chi2_contingency(test_df.values) 

    print('chi2 stat : '+ str(stat))

    critical = chi2.ppf(prob, dof)

    print('Critical point : '+ str(critical))

    if abs(stat) >= critical:

        print('Dependent')

    else:

        print('Independent')
loan["pub_rec_bankruptcies"] = loan["pub_rec_bankruptcies"].apply(lambda x:'1/More bankruptcies' if(x>=1) else 'No bankruptcies')

loan["pub_rec_bankruptcies"].value_counts()
Category_dependancy_test(loan.term, loan.pub_rec)
Category_dependancy_test(loan.grade, loan.inq_last_6mths_recat)
Category_dependancy_test(loan.grade, loan.inq_last_6mths_recat)
Category_dependancy_test(loan.pub_rec_bankruptcies, loan.inq_last_6mths_recat)
loan_clean = loan[loan.annual_inc < loan.annual_inc.quantile(0.99)]
loan_cont = loan_clean[['int_rate', 'annual_inc', 'funded_amnt', 'newDTI']].copy()

# g = sns.pairplot(loan_cont)

round(loan_cont.corr(),2)
def bivariate_categorical(df,cat1, cat2 , xlabel, title, figsize):

    

    '''Biivariate function to plot the graphs for categorical variables based on the parameters provided

        Parameter Details:

        cat1, cat2- categorical Column to be analyzed

        title - Plot title

    '''

    ##taking out 'Charged off' loan status count for the columns 'cat1' and 'cat2'

    loan_default = df.loc[loan.loan_status == 'Charged Off'].groupby(by=[cat1,cat2])['loan_status'].count().rename('Default_count').sort_values(ascending=False).reset_index()



    ##taking out Total loan status count for the columns 'cat1' and 'cat2'

    loan_total = df.groupby(by=[cat1,cat2])['loan_status'].count().rename('Total_count').sort_values(ascending=False).reset_index()



    ##merge above two dataframes so percentage for charged off/default can be calculated

    loan_merged = pd.merge(loan_total,loan_default,how='inner',on=[cat1,cat2])

    loan_merged['default_percentage'] = round(loan_merged['Default_count']/loan_merged['Total_count']*100,2)





    plt.figure(figsize=figsize,dpi=120,frameon=True)

    sns.barplot(x=cat1, y='default_percentage', hue=cat2,data=loan_merged)

    plt.ylabel("Loan Status = Default Percentage")

    plt.xlabel(xlabel)

    plt.title(title)

    plt.show()

    
bivariate_categorical(loan,'grade','term', "Loan Grade", "Default loan % by Loan Term and Grade" , [10.0,5.0] )
bivariate_categorical(loan, 'grade','inq_last_6mths_recat', "Loan Grade", 

                      "Default loan % by Loan term and No of Inquiries on customer" , [10.0,5.0] )
bivariate_categorical(loan, 'pub_rec_bankruptcies','inq_last_6mths_recat', "No of of bankruptcies records", 

                      "Default loan % by No of bankruptcies records and No of inquiries on customer", [6.0,3.0])
df_state = loan[loan["addr_state"].isin(["CA", "NY", "FL", "TX", "NJ", "NV", "SD"]) ]

df_state = df_state[df_state["purpose"].isin(["debt_consolidation", "credit_card", "other",

                                              "home_improvement", "major_purchase", "small_business"]) ]

bivariate_categorical(df_state, 'addr_state','purpose', "US-States", 

                      "Default loan % by US-States and Loanpupose", [10.0,5.0])
bivariate_categorical(loan, 'annual_inc_binned','funded_amnt_binned', "Anual Income", 

                      "Default loan % by Annual income and funded loan amount ", [12.0,5.0])

#sns.scatterplot(loan_clean.annual_inc, loan_clean.funded_amnt)

#plt.show()