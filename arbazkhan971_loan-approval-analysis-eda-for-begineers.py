#importing the useful Python library for data handaling 
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns
#loading the from loan.xlsx for df Dataframe using pandas 

df=pd.read_excel('../input/loan-approval-analysis/Loan Data/loan.xlsx')
#Reading the few lines of Data for getting to know about data 
#this head info for knowing the about data 

df.head(3)
len(df.columns) # checking the Total number of  columns in data 
#checking the columns who have nan value 

res = df.columns[df.isnull().all(0)]
df[res[0]].isnull().sum() #checking the number of nan value in the columns 
#checking the whole columns with number of nan values respecting columns
for i in range (0,len(res)):

    print(res[i],df[res[i]].isnull().sum())
#Removing the columns of which have nan in the above Columns we have 39717  nan which almost empty values 
df.drop(columns =['mths_since_last_major_derog', 'annual_inc_joint', 'dti_joint',

       'verification_status_joint', 'tot_coll_amt', 'tot_cur_bal',

       'open_acc_6m', 'open_il_6m', 'open_il_12m', 'open_il_24m',

       'mths_since_rcnt_il', 'total_bal_il', 'il_util', 'open_rv_12m',

       'open_rv_24m', 'max_bal_bc', 'all_util', 'total_rev_hi_lim', 'inq_fi',

       'total_cu_tl', 'inq_last_12m', 'acc_open_past_24mths', 'avg_cur_bal',

       'bc_open_to_buy', 'bc_util', 'mo_sin_old_il_acct',

       'mo_sin_old_rev_tl_op', 'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl',

       'mort_acc', 'mths_since_recent_bc', 'mths_since_recent_bc_dlq',

       'mths_since_recent_inq', 'mths_since_recent_revol_delinq',

       'num_accts_ever_120_pd', 'num_actv_bc_tl', 'num_actv_rev_tl',

       'num_bc_sats', 'num_bc_tl', 'num_il_tl', 'num_op_rev_tl',

       'num_rev_accts', 'num_rev_tl_bal_gt_0', 'num_sats', 'num_tl_120dpd_2m',

       'num_tl_30dpd', 'num_tl_90g_dpd_24m', 'num_tl_op_past_12m',

       'pct_tl_nvr_dlq', 'percent_bc_gt_75', 'tot_hi_cred_lim',

       'total_bal_ex_mort', 'total_bc_limit', 'total_il_high_credit_limit'],axis = 1, inplace = True)
df.head(4) # checking the head info after removing the 
df.columns.unique()  #checking the no. of columns which uniques ()
df.describe() #See Basic view of Data 
month_count=[]

month_type=df['term'].unique()

for i in range(0,len(df['term'].unique())):

    month_count.append((df['term']==month_type[i]).sum())
plt.bar(month_type,month_count)

plt.ylabel('Count')

plt.xlabel("Term")

plt.show()

print("*"*120)

print("30 months:-",month_count[0],"60 months:-",month_count[1])

print("*"*120)
grade_count=[]

grade_type=df['grade'].unique()

for i in range(0,len(df['grade'].unique())):

    grade_count.append((df['grade']==grade_type[i]).sum())
percentage=[]

grade_sum=0

for i in range (0,len(grade_count)):

  grade_sum=grade_count[i]+grade_sum



  





for i in range (0,len(grade_count)):

    percentage.append((grade_count[i]/grade_sum)*100) 

print("*"*120)

for i in range (0,len(percentage)):

    print(grade_type[i]," :- ",percentage[i],"%")

print("*"*120) 

   
plt.bar(grade_type,percentage)

plt.xlabel("Grade Types")

plt.ylabel("Count ")

plt.show()
loan_count=[]

loan_type=df['loan_status'].unique()

for i in range(0,len(df['loan_status'].unique())):

    loan_count.append((df['loan_status']==loan_type[i]).sum())
loan_sum=0

for i in range (0,len(loan_count)):

    loan_sum=loan_count[i]+loan_sum

    

    
percentage=[]

for i in range (0,len(loan_count)):

    percentage.append((loan_count[i]/loan_sum)*100)

    

    

plt.bar(loan_type,percentage)

plt.xlabel("Type of Loan Status ")

plt.ylabel("Count")

plt.show()





pur_count=[]

l=df['purpose'].unique()

for i in range(0,len(df['purpose'].unique())):

    pur_count.append((df['purpose']==l[i]).sum())
percentage=[]

pur_sum=0

for i in range (0,len(pur_count)):

    pur_sum=pur_count[i]+pur_sum



  





for i in range (0,len(pur_count)):

    percentage.append((pur_count[i]/pur_sum)*100) 

print("*"*120)

for i in range (0,len(percentage)):

    print(l[i]," :- ",percentage[i],"%")

print("*"*120) 

   
fig = plt.figure(figsize=(27, 8))

plt.bar(l,percentage)

plt.xlabel("Type of Loan Purpose ")

plt.ylabel("Count")

plt.show()
home_count=[]

home_type=df['home_ownership'].unique()

for i in range(0,len(df['home_ownership'].unique())):

    home_count.append((df['home_ownership']==home_type[i]).sum())
percentage=[]

home_sum=0

for i in range (0,len(home_count)):

    home_sum=home_count[i]+home_sum



  









for i in range (0,len(home_count)):

    percentage.append((home_count[i]/home_sum)*100) 

print("*"*120)

for i in range (0,len(percentage)):

    print(home_type[i]," :- ",percentage[i],"%")

print("*"*120) 

   
plt.bar(home_type,percentage)

plt.xlabel("Type of Home  ")

plt.ylabel("Count")

plt.show()
verification_count=[]

verification_type=df['verification_status'].unique()

for i in range(0,len(df['verification_status'].unique())):

    verification_count.append((df['verification_status']==verification_type[i]).sum())
percentage=[]

verification_sum=0

for i in range (0,len(verification_count)):

    verification_sum=verification_count[i]+verification_sum

    

   



  





for i in range (0,len(verification_count)):

    percentage.append((verification_count[i]/verification_sum)*100) 

print("Percentage wise ")

print("*"*120)



for i in range (0,len(percentage)):

    print(verification_type[i]," :- ",percentage[i],"%")

print("*"*120) 

   
plt.bar(verification_type,percentage)

plt.xlabel("Type of Verification ")

plt.ylabel("Count")

plt.show()
#code copy https://iandzy.com/histograms-cumulative-distribution/

def plot_cdf(data, bin_size,title=None, xlabel=None):



    counts, bin_edges = np.histogram (data, bins=bin_size, normed=True)

    cdf = np.cumsum(counts)

    

    sns.set_style("whitegrid")

    plt.figure(figsize=(18,13))

    plt.plot (bin_edges[1:], cdf/cdf[-1])

    plt.ylabel('CDF')

    

    if title:

        plt.title(title)

    if xlabel:

        plt.xlabel(xlabel)

    plt.show()
plot_cdf(df['loan_amnt'],20,'loan Amount','Amount')#ploting the cdf of amount loan taken by Customer's
plot_cdf(df['installment'],20,'Installment ','Amount') #ploting the Cdf of installment amount 
plot_cdf(df['last_pymnt_amnt'],10,'Last_Payment_Amount','Amount')#Ploting the Cdf of Last Payment Amount
plot_cdf(df['total_pymnt_inv'],10,'total_pymnt_inv','Amount')
# these are package for hiding the waring ariese during the Execution of Cell 

import warnings

warnings.filterwarnings('ignore')
grade_plot=sns.FacetGrid(df,hue='grade',size=10)

grade_plot=grade_plot.map(sns.distplot,'loan_amnt').add_legend()
grade_plot=sns.FacetGrid(df,hue='verification_status',size=10)

grade_plot=grade_plot.map(sns.distplot,'loan_amnt',). add_legend()
grade_plot=sns.FacetGrid(df,hue='term',size=10,)

grade_plot=grade_plot.map(sns.distplot,'loan_amnt',hist=True). add_legend()
grade_plot=sns.FacetGrid(df,hue='loan_status',size=10)

grade_plot=grade_plot.map(sns.distplot,'loan_amnt',). add_legend()
grade_plot=sns.FacetGrid(df,hue='purpose',size=10)

grade_plot=grade_plot.map(sns.distplot,'loan_amnt',). add_legend()
plot=sns.boxplot(x="term", y="loan_amnt", hue="term",data=df, palette="Set2")

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()
plot=sns.violinplot(x="term", y="loan_amnt", hue="term",data=df, palette="Set2")

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()
plot=sns.boxplot(x="verification_status", y="loan_amnt", hue="verification_status",data=df)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()
plot=sns.boxplot(x="home_ownership", y="loan_amnt", hue="home_ownership",data=df)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()
plot=sns.boxplot(x="grade", y="loan_amnt", hue="grade",data=df)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()
plt.figure(figsize=(25,8))

plot=sns.boxplot(x="purpose", y="loan_amnt", hue="purpose",data=df,linewidth=1.5)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()
plt.figure(figsize=(25,8))

plot=sns.boxplot(x="emp_length", y="loan_amnt", hue="emp_length",data=df,linewidth=1.5)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()
plt.figure(figsize=(10,8))

plot=sns.boxplot(x="term", y="installment", hue="term",data=df,linewidth=1.5)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()
df['pub_rec_bankruptcies'].fillna((df['pub_rec_bankruptcies'].mean()), inplace=True)
plot_cdf(df['pub_rec_bankruptcies'],20,'Public Records Bankruptcies  ','Level')
plot_cdf(df['acc_now_delinq'],20,'Public Records Bankruptcies  ','Level')
plot_cdf(df['int_rate'],20,'Interest Rate  ','Rate ')
plt.figure(figsize=(10,8))

plot=sns.boxplot(x="term", y="int_rate",hue="term",data=df,linewidth=1.5)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()


plt.figure(figsize=(10,8))

plot=sns.boxplot(x="grade", y="int_rate",hue="grade",data=df,linewidth=1.5)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()
plt.figure(figsize=(10,8))

plot=sns.boxplot(y="total_rec_prncp", x="grade",hue="grade",data=df,linewidth=1.5)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()
plot_cdf(df['funded_amnt'],5,'Interest Rate  ','Rate ')
plot_cdf(df['out_prncp'],5,'Interest Rate  ','Rate ')
plt.figure(figsize=(20,8))

sns.barplot(x='purpose',y='loan_amnt',data=df,hue='purpose')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()
plt.figure(figsize=(20,8))

sns.barplot(x='purpose',y='loan_amnt',data=df,hue='loan_status')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()
plt.figure(figsize=(20,10))

sns.barplot(x='purpose',y='funded_amnt',data=df,hue='loan_status')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()
plt.figure(figsize=(20,10))

sns.barplot(x='purpose',y='funded_amnt',data=df,hue='loan_status')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()
plt.figure(figsize=(30,10))

sns.barplot(x='purpose',y='funded_amnt',data=df,hue='grade')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()
plt.figure(figsize=(30,10))

sns.barplot(x='purpose',y='funded_amnt',data=df,hue='home_ownership')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()
plt.figure(figsize=(30,10))

sns.barplot(x='purpose',y='funded_amnt',data=df,hue='verification_status')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()
plt.figure(figsize=(30,10))

sns.barplot(x='purpose',y='funded_amnt',data=df)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()