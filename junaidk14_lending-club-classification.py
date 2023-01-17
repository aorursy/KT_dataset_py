import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df=pd.read_csv('../input/lending-club-loan/loan.csv',encoding='latin-1',sep=',',names=['id','member_id','loan_amnt','funded_amnt','funded_amnt_inv','term','int_rate','installment','grade','sub_grade','emp_title',	'emp_length','home_ownership','annual_inc','verification_status','issue_d','loan_status','pymnt_plan','url','desc','purpose','title','zip_code','addr_state','dti','delinq_2yrs','earliest_cr_line','inq_last_6mths',	'mths_since_last_delinq',	'mths_since_last_record','open_acc','pub_rec','revol_bal','revol_util','total_acc','initial_list_status','out_prncp','out_prncp_inv','total_pymnt','total_pymnt_inv','total_rec_prncp','total_rec_int','total_rec_late_fee','recoveries','collection_recovery_fee',	'last_pymnt_d','last_pymnt_amnt','next_pymnt_d','last_credit_pull_d','collections_12_mths_ex_med',	'mths_since_last_major_derog','policy_code','application_type','annual_inc_joint','dti_joint','verification_status_joint','acc_now_delinq','tot_coll_amt','tot_cur_bal','open_acc_6m','open_act_il','open_il_12m','open_il_24m','mths_since_rcnt_il','total_bal_il','il_util','open_rv_12m','open_rv_24m','max_bal_bc','all_util','total_rev_hi_lim','inq_fi','total_cu_tl','inq_last_12m',	'acc_open_past_24mths',	'avg_cur_bal','bc_open_to_buy','bc_util','chargeoff_within_12_mths','delinq_amnt','mo_sin_old_il_acct','mo_sin_old_rev_tl_op',	'mo_sin_rcnt_rev_tl_op','mo_sin_rcnt_tl','mort_acc','mths_since_recent_bc','mths_since_recent_bc_dlq','mths_since_recent_inq','mths_since_recent_revol_delinq','num_accts_ever_120_pd','num_actv_bc_tl','num_actv_rev_tl','num_bc_sats','num_bc_tl','num_il_tl','num_op_rev_tl','num_rev_accts','num_rev_tl_bal_gt_0','num_sats',	'num_tl_120dpd_2m',	'num_tl_30dpd',	'num_tl_90g_dpd_24m','num_tl_op_past_12m','pct_tl_nvr_dlq','percent_bc_gt_75','pub_rec_bankruptcies','tax_liens','tot_hi_cred_lim','total_bal_ex_mort','total_bc_limit','total_il_high_credit_limit','revol_bal_joint','sec_app_earliest_cr_line',	'sec_app_inq_last_6mths',	'sec_app_mort_acc','sec_app_open_acc','sec_app_revol_util','sec_app_open_act_il','sec_app_num_rev_accts','sec_app_chargeoff_within_12_mths','sec_app_collections_12_mths_ex_med','sec_app_mths_since_last_major_derog','hardship_flag','hardship_type','hardship_reason','hardship_status','deferral_term','hardship_amount','hardship_start_date','hardship_end_date','payment_plan_start_date','hardship_length','hardship_dpd',	'hardship_loan_status','orig_projected_additional_accrued_interest','hardship_payoff_balance_amount','hardship_last_payment_amount','debt_settlement_flag','debt_settlement_flag_date','settlement_status','settlement_date','settlement_amount','settlement_percentage','settlement_term']) 
df.shape
pd.set_option('display.max_columns',150)

df.head()
df.tail()
df1=df.drop(['id','member_id'],axis=1)
df1.head()
df1.tail()
df1.shape
#Dropping rows with all NaN values

df1.dropna(how='all',inplace=True,axis=0)
df1.reset_index(drop=True,inplace=True)
df1.shape
df1.head()
df1.drop(index=[0],axis=0,inplace=True)
df1.reset_index(drop=True,inplace=True)
df1.shape
df1.head()
df1.tail()
#Dropping the columns with all NaN values

df1.dropna(how='all',axis=1,inplace=True)
df1.shape
#df1.to_csv('LendingClub.csv')
pd.set_option('display.max_rows',150)

df1.isnull().sum()
colnull=[]

for i in df1.columns:

    if df1[i].isnull().sum()>0:

        colnull.append(i)
#df1 columns list with null values

colnull
df1[df1.purpose=='moving']['title']
#Dropping columns with more than 70 percent null values

df2 = df1[[column for column in df1 if df1[column].count() / len(df1) >= 0.3]]

print("List of dropped columns:", end=" ")

for c in df1.columns:

    if c not in df2.columns:

        print(c, end=", ")
df2.shape
#df2=df1.drop(columns=['debt_settlement_flag_date','settlement_status', 'settlement_date', 'settlement_amount','settlement_percentage', 'settlement_term','next_pymnt_d','mths_since_last_record'],axis=1)
#df2.shape
df2.info()
df2.columns
#Checking for columns with only one unique value
for i in df2.columns:

    print('For {x}'.format(x=i))

    print(df2[i].nunique())

    print()
#Dropping columns with one unique value

df3 = df2[[column for column in df2 if df1[column].nunique()>1]]

print("List of dropped columns:", end=" ")

for c in df2.columns:

    if c not in df3.columns:

        print(c, end=", ")
for i in df3.columns:

    print('For {x}'.format(x=i))

    print(df3[i].unique())

    print()
#Deleting highly imbalanced columns

df3.tax_liens.value_counts()
df3.pub_rec_bankruptcies.value_counts()
df3.pub_rec_bankruptcies.replace({'0':0,'1':1,'2':2},inplace=True)
df3.pub_rec_bankruptcies.value_counts()
df3.delinq_amnt.value_counts()
df3.debt_settlement_flag.value_counts()
df3.acc_now_delinq.value_counts()
# Dropping future events and highly imbalanced columns

df3.drop(columns=['out_prncp','out_prncp_inv','collections_12_mths_ex_med','policy_code',

                  'chargeoff_within_12_mths','tax_liens','delinq_amnt','acc_now_delinq'],axis=1,inplace=True)
df3.shape
intr=[]

for i in df3['int_rate'].values:

    x=i[:-1]

    intr.append(x)
df3['int_rate']=np.array(intr)
ind=df3['revol_util'].dropna().index
rev=[]

for j in df3['revol_util'].dropna():

    y=j[:-1]

    rev.append(y)
rev1=pd.Series(rev,index=ind)
mask=df3['revol_util'].isnull()

rev2=df3[mask]['revol_util']
revol_ut=pd.concat([rev1,rev2],axis=0)
revol_ut.sort_index(inplace=True)
df3['revol_util']=revol_ut.values
df3.info()
# Variable Broadcasting

df4=df3.astype({'loan_amnt':float, 'funded_amnt':float, 'funded_amnt_inv':float,'int_rate':float,

       'installment':float,'annual_inc': float,'dti':float,'mths_since_last_delinq':float,

        'open_acc':float, 'revol_bal':float, 'revol_util':float, 'total_acc':float, 'total_pymnt':float,

                'total_pymnt_inv':float,'total_rec_prncp':float,

        'total_rec_int':float, 'total_rec_late_fee':float,'recoveries':float,

        'collection_recovery_fee':float,'last_pymnt_amnt':float})
df4.info()
df4.drop(columns=['desc','title','zip_code','emp_title'],axis=1,inplace=True)
dir(str)
help(str.split)
ter=[]

for i in df4['term'].values:

    x=i.split()[0]

    ter.append(x)
df4['term']=np.array(ter)
df4.emp_length.unique()
df4.replace({'10+ years':'10','< 1 year':'1','1 year':'1','3 years':'3', '8 years':'8', '9 years':'9',

       '4 years':'4', '5 years':'5', '6 years':'6', '2 years':'2', '7 years':'7'},inplace=True)
df4.head()
df4.shape
# Maximum and Minimum interest rate Grade Wise
df4['int_rate'][df4['grade']=='A'].min()
df4['int_rate'][df4['grade']=='A'].max()
df4['int_rate'][df4['grade']=='B'].min()
df4['int_rate'][df4['grade']=='B'].max()
df4['int_rate'][df4['grade']=='C'].min()
df4['int_rate'][df4['grade']=='C'].max()
df4['int_rate'][df4['grade']=='D'].min()
df4['int_rate'][df4['grade']=='D'].max()
df4['int_rate'][df4['grade']=='E'].min()
df4['int_rate'][df4['grade']=='E'].max()
df4['int_rate'][df4['grade']=='F'].min()
df4['int_rate'][df4['grade']=='F'].max()
df4['int_rate'][df4['grade']=='G'].min()
df4['int_rate'][df4['grade']=='G'].max()
df4.isnull().sum()
df4.loan_status.value_counts()
np.isnan(df4['pub_rec_bankruptcies'].values).sum()
# Box plot to check outliers

df4.boxplot(figsize=(20,12))

plt.xticks(rotation=90)

plt.show()
df4['annual_inc'].min()
Q1=df4['annual_inc'].quantile(0.25)

Q1
Q2=df4['annual_inc'].quantile(0.5)

Q2
Q3=df4['annual_inc'].quantile(0.75)

Q3
IQR=Q3-Q1

IQR
Imin=Q1-1.5*IQR

Imin
Imax=Q3+1.5*IQR

Imax
df4['annual_inc'].max()
df4['annual_inc'].mean()
"""np.set_printoptions(threshold=np.nan)

df4['annual_inc'].unique()

"""
mask=df4['annual_inc'].isnull()

df4[mask]
df4[df4['home_ownership']=='NONE']
df4['annual_inc'][(df4['emp_length']=='1') ].median()
df4['annual_inc'][(df4['emp_length']=='1') ].mean()
df4['home_ownership'].value_counts()
df4['annual_inc'][df4['home_ownership']=='NONE']
#Dropping annual income missing values as many columns contain NaN in that particular rows
df4.dropna(inplace=True,subset=['annual_inc'],axis=0)
df4.reset_index(drop=True,inplace=True)
df4.isnull().sum()
# Variable Creation

(df4['loan_status']=='Does not meet the credit policy. Status:Fully Paid') | (df4['loan_status']=='Does not meet the credit policy. Status:Charged Off')
df4['criteria']=np.where((df4['loan_status']=='Does not meet the credit policy. Status:Fully Paid')|(df4['loan_status']=='Does not meet the credit policy. Status:Charged Off'),'No','Yes')
df4[['criteria','loan_status']]
df4.replace({'Does not meet the credit policy. Status:Fully Paid':'Fully Paid','Does not meet the credit policy. Status:Charged Off':'Charged Off'},inplace=True)
df4['loan_status'].value_counts()
df4.to_csv('P2P1.csv')
df4.shape
df4.isnull().sum()
#mths_since_lst_delin missing value imputation
df4['mths_since_last_delinq'][(df4['loan_status']=='Fully Paid') & (df4['criteria']=='Yes')].median()
a=df4['mths_since_last_delinq'][(df4['loan_status']=='Fully Paid') & (df4['criteria']=='Yes')]

a.fillna(a.median(),inplace=True)
df4['mths_since_last_delinq'][(df4['loan_status']=='Fully Paid') & (df4['criteria']=='No')].median()
b=df4['mths_since_last_delinq'][(df4['loan_status']=='Fully Paid') & (df4['criteria']=='No')]

b.fillna(b.median(),inplace=True)
df4['mths_since_last_delinq'][(df4['loan_status']=='Charged Off')& (df4['criteria']=='Yes') ].median()
c=df4['mths_since_last_delinq'][(df4['loan_status']=='Charged Off') & (df4['criteria']=='Yes')]

c.fillna(c.median(),inplace=True)

df4['mths_since_last_delinq'][(df4['loan_status']=='Charged Off')& (df4['criteria']=='No') ].median()
d=df4['mths_since_last_delinq'][(df4['loan_status']=='Charged Off') & (df4['criteria']=='No')]

d.fillna(d.median(),inplace=True)
e=pd.concat([a,b,c,d])
e.sort_index(inplace=True)
df4['mths_since_last_delinq']=e
df4['mths_since_last_delinq']
df4.isnull().sum()
df4['emp_length']=pd.to_numeric(df4['emp_length'])
#Replacing NONE with OTHER in home_ownership
df4['home_ownership'].replace({'NONE':'OTHER'},inplace=True)
#Emp_length missing value imputation
df4['emp_length']=df4['emp_length'].fillna(df4['emp_length'].median())
df4.isnull().sum()
df4['delinq_2yrs']=df4['delinq_2yrs'].fillna(df4['delinq_2yrs'].median())
df4['inq_last_6mths']=df4['inq_last_6mths'].fillna(df4['inq_last_6mths'].median())
df4['open_acc']=df4['open_acc'].fillna(df4['open_acc'].median())
df4['pub_rec']=df4['pub_rec'].fillna(df4['pub_rec'].median())
df4['revol_util']=df4['revol_util'].fillna(df4['revol_util'].median())
df4['total_acc']=df4['total_acc'].fillna(df4['total_acc'].median())
df4['pub_rec_bankruptcies']=df4['pub_rec_bankruptcies'].fillna(df4['pub_rec_bankruptcies'].median())
df4.columns
df4.to_csv('LC.csv')
sns.distplot(df4.loan_amnt)

plt.show()
sns.distplot(df4.funded_amnt)

plt.show()
sns.distplot(df4.funded_amnt_inv)

plt.show()
sns.distplot(df4.int_rate)

plt.show()
sns.distplot(df4.installment)

plt.show()
sns.distplot(df4[df4['emp_length'].notnull()]['emp_length'])

plt.show()
sns.distplot(df4.annual_inc)

plt.show()
sns.distplot(df4.dti)

plt.show()
sns.distplot(df4.revol_bal)

plt.show()
sns.distplot(df4[df4['revol_util'].notnull()]['revol_util'])

plt.show()
sns.boxplot('home_ownership','funded_amnt', data = df4)

plt.title('Home ownership Vs Funded_amnt')

plt.show()
plt.subplots(figsize=(12,10))

sns.boxplot(x='purpose', y= 'int_rate', data=df4)

plt.xticks(rotation=90)

plt.title(' Funded_amnt Vs Purpose')

plt.show()
m=df4.pivot_table(values = 'funded_amnt', index = 'emp_length', columns = 'loan_status', aggfunc = 'median')

m
m.plot(kind='bar',color=('r','g'))

plt.legend(loc='upper right')

plt.ylim((0,16000))

plt.show()
n=df4.pivot_table(values = 'int_rate', index = 'emp_length', columns = 'loan_status', aggfunc = 'median')

n
n.plot(kind='bar',color=('r','g'))

plt.legend(loc='upper right')

plt.ylim((0,18))

plt.show()
q=df4.pivot_table(values = 'annual_inc', index = 'emp_length', columns = 'loan_status', aggfunc = 'median')

q
q.plot(kind='bar',color=('r','g'))

plt.legend(loc='upper right')

plt.ylim((0,89000))

plt.show()
# Annual_inc of 10 yrs exp customer is highest
p=df4.pivot_table(values = 'funded_amnt', index = 'emp_length', columns = 'term', aggfunc = 'median')

p
p.plot(kind='bar')

plt.legend(loc='upper right')

plt.ylim((0,21000))

plt.show()
# Borrowers opting for 60 months are borrowing more amount. So, opting for high term
sns.countplot(df4['emp_length'].sort_values())
sns.countplot(df4['home_ownership'])
df4.describe()
df4.groupby('home_ownership').apply(lambda x: x[['funded_amnt','total_pymnt','annual_inc']].mean())
#Refer tableau annual_inc vs loan_amnt

sns.lmplot(x='funded_amnt',y='annual_inc',data=df4,hue='home_ownership',fit_reg=False)
df4.head()
sns.countplot(df4.term)
b=df4['emp_length'][df4['term']=='60'].value_counts()

b.plot(kind='bar')

plt.title(' Count of Emp_length in 60 term ')

plt.show()
a=df4['emp_length'][df4['term']=='36'].value_counts()

a.plot(kind='bar')

plt.title(' Count of Emp_length in 36 term ')

plt.show()
df4.columns
sns.FacetGrid(df4,hue='home_ownership',size=4).map(plt.scatter,'funded_amnt','annual_inc').add_legend()

plt.show()
sns.FacetGrid(df4,hue='loan_status',size=4).map(plt.scatter,'annual_inc','funded_amnt').add_legend()

plt.show()
#import plotly.express as px
# fig = px.scatter_3d(df4, x='annual_inc', y='funded_amnt', z='revol_util',

#               color='loan_status')

# fig.show()
# fig = px.scatter_3d(df4, x='annual_inc', y='funded_amnt', z='emp_length',

#               color='loan_status')

# fig.show()
df4.columns
# df10=df5[['loan_amnt', 'funded_amnt', 'funded_amnt_inv','int_rate',

#        'installment','annual_inc','dti','revol_bal', 'revol_util', 'total_acc', 'total_pymnt',

#        'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int',

#        'total_rec_late_fee', 'recoveries', 'collection_recovery_fee','loan_status']]
#sns.pairplot(df10)
df4.shape
import pylab

import scipy.stats as stats
# N(0,1)

std_normal = np.random.normal(loc = 0, scale = 1, size=42531)



# 0 to 100th percentiles of std-normal

for i in range(0,101):

    print(i, np.percentile(std_normal,i))
stats.probplot(df4.funded_amnt, dist="norm", plot=pylab)

pylab.show()
stats.probplot(np.log(df4.funded_amnt), dist="norm", plot=pylab)

pylab.show()

df4.funded_amnt.values.ndim
stats.probplot(df4.loan_amnt, dist="norm", plot=pylab)

pylab.show()
stats.probplot(np.log(df4.loan_amnt), dist="norm", plot=pylab)

pylab.show()
stats.probplot(df4.funded_amnt_inv, dist="norm", plot=pylab)

pylab.show()
stats.probplot(np.sqrt(df4.funded_amnt_inv), dist="norm", plot=pylab)

pylab.show()
stats.probplot(df4.annual_inc, dist="norm", plot=pylab)

pylab.show()
stats.probplot(np.log(df4.annual_inc), dist="norm", plot=pylab)

pylab.show()
stats.probplot(df4.revol_bal, dist="norm", plot=pylab)

pylab.show()
stats.probplot(np.sqrt(df4.revol_bal), dist="norm", plot=pylab) # Due to zero in column sqrt is applied

pylab.show()
stats.probplot(df4.revol_util, dist="norm", plot=pylab)

pylab.show()
stats.probplot(np.sqrt(df4.revol_util), dist="norm", plot=pylab)

pylab.show()
# Boxcox transformation
xt=stats.boxcox(df4.funded_amnt.values,lmbda=0.26044195400343567)

xt
stats.probplot(xt, dist=stats.norm, plot=pylab)

pylab.show()
df4.funded_amnt.skew()
np.log(df4.funded_amnt).skew()
df4.annual_inc.skew()
(np.log(df4.annual_inc)).skew()
df4.revol_bal.skew()
np.sqrt(df4.revol_bal).skew()
df4.revol_util.skew()
np.sqrt(df4.revol_util).skew()
stats.anderson(df4.annual_inc, dist='norm')
stats.anderson(np.log(df4.annual_inc), dist='norm')
stats.anderson(df4.funded_amnt, dist='norm')
stats.anderson(np.log(df4.funded_amnt), dist='norm')
stats.anderson(df4.loan_amnt, dist='norm')
stats.anderson(np.log(df4.loan_amnt), dist='norm')
stats.anderson(df4.funded_amnt_inv, dist='norm')
stats.anderson(np.sqrt(df4.funded_amnt_inv), dist='norm')
stats.anderson(df4.revol_bal, dist='norm')
stats.anderson(np.sqrt(df4.revol_bal), dist='norm')
stats.anderson(df4.revol_util, dist='norm')
stats.anderson(np.sqrt(df4.revol_util), dist='norm')
df4['annual_inc']=np.log(df4.annual_inc)
df4['funded_amnt']=np.log(df4.funded_amnt)
df4['loan_amnt']=np.log(df4.loan_amnt) 
df4['funded_amnt_inv']=np.sqrt(df4.funded_amnt_inv) # due to zero in funded_amnt_inv we use sqrt
df4['revol_bal']=np.sqrt(df4.revol_bal) # Due to zero in revol_bal sqrt is applied
A=df4['int_rate'][df4['grade']=='A'].values

A
B=df4['int_rate'][df4['grade']=='B'].values

B
C=df4['int_rate'][df4['grade']=='C'].values

C
D=df4['int_rate'][df4['grade']=='D'].values

D
E=df4['int_rate'][df4['grade']=='E'].values

E
F=df4['int_rate'][df4['grade']=='F'].values

F
G=df4['int_rate'][df4['grade']=='G'].values

G
intrate=pd.DataFrame()

d1=pd.DataFrame({'grade':'A','int_rate':A})

d2=pd.DataFrame({'grade':'B','int_rate':B})

d3=pd.DataFrame({'grade':'C','int_rate':C})

d4=pd.DataFrame({'grade':'D','int_rate':D})

d5=pd.DataFrame({'grade':'E','int_rate':E})

d6=pd.DataFrame({'grade':'F','int_rate':F})

d7=pd.DataFrame({'grade':'G','int_rate':G})
intrate=intrate.append(d1)

intrate=intrate.append(d2)

intrate=intrate.append(d3)

intrate=intrate.append(d4)

intrate=intrate.append(d5)

intrate=intrate.append(d6)

intrate=intrate.append(d7)
intrate.reset_index(drop=True,inplace=True)
print(intrate.head())
print(intrate.tail())
g=sns.boxplot(x='grade',y='int_rate',data=intrate)

g.set(xlabel='grade',ylabel='int_rate',title='Boxplot-Int_rate VS Grade')

plt.show()
import scipy.stats as stats

import statsmodels.formula.api as stm
stats.f.ppf(q=1-0.05,dfn=6,dfd=425)
model=stm.ols('int_rate~grade',data=intrate).fit()
model.summary()
# Anova b/n funded_amnt and loan_status
from scipy.stats import f_oneway
fp=df4[df4['loan_status']=='Fully Paid']

co=df4[df4['loan_status']=='Charged Off']
f_oneway(fp['funded_amnt'],co['funded_amnt'])
# p<0.05 so reject the null hypo i.e. funded_amnt is dependent on loan_status. So, funded_amnt is present
df4[['delinq_2yrs','int_rate','grade','dti']]
# z test between loan_amnt and funded_amnt
std1=df4['loan_amnt'].std()
std1
std2=df4['funded_amnt'].std()
std2
df4['loan_amnt'].mean()-df4['funded_amnt'].mean()
import statsmodels.stats.weightstats
df4.columns
statsmodels.stats.weightstats.ztest(df4['loan_amnt'].values,df4['funded_amnt'].values,alternative='two-sided')

#p < 0.05 reject null hypo. So, loan_amnt and funded_amnt are dependent
# z test between funded_amnt_inv and funded_amnt
std3=df4['funded_amnt_inv'].std()

std3
df4['funded_amnt'].mean()-df4['funded_amnt_inv'].mean()
statsmodels.stats.weightstats.ztest(df4['funded_amnt_inv'].values,df4['funded_amnt'].values, alternative = 'two-sided',value=681.42)
#p < 0.05 so reject null hypo. So, loan_amnt and funded_amnt are dependent
stats.ttest_ind(df4['funded_amnt_inv'].values,df4['funded_amnt'].values)
sns.boxplot(df4.annual_inc)
a = df4.annual_inc.values

stats.mstats.winsorize(a, limits=[0.025, 0.025],inplace=True)
b = df4.revol_bal

stats.mstats.winsorize(b, limits=[0.025, 0.025],inplace=True)
sns.boxplot(df4.annual_inc)
sns.boxplot(df4.revol_bal)
df5=df4.copy(deep=True)
df5.drop(columns=['loan_amnt','issue_d','funded_amnt_inv','int_rate','installment','sub_grade','addr_state','earliest_cr_line',

                  'last_pymnt_d','total_pymnt','total_pymnt_inv', 'total_rec_prncp', 'open_acc','total_rec_int',

                  'last_credit_pull_d','total_rec_late_fee','recoveries', 'collection_recovery_fee','last_pymnt_amnt'],axis=1,inplace=True)
df5.columns
plt.subplots(figsize=(15,8))

sns.heatmap(df5.corr(),annot=True,linewidth=0.2)
df5
df5.head()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df5['grade']=le.fit_transform(df5.grade)
df5.head()
df5.info()
df5['delinq_2yrs']=pd.to_numeric(df5['delinq_2yrs'])
df5['inq_last_6mths']=pd.to_numeric(df5['inq_last_6mths'])

df5['pub_rec']=pd.to_numeric(df5['pub_rec'])
df6=pd.get_dummies(df5,columns=['term','home_ownership','verification_status','purpose','debt_settlement_flag','criteria'])
df6.columns
# Removing Dummy trap

df6.drop(columns=['criteria_No','debt_settlement_flag_Y','purpose_house','verification_status_Verified',

                  'home_ownership_RENT','term_60'],axis=1,inplace=True)
df6.shape
x=df6.drop(columns=['loan_status'])

y=df6['loan_status']
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
xsc=sc.fit_transform(x)
# ! pip install imbalanced-learn

from imblearn.over_sampling import SMOTE
sm=SMOTE(random_state=2)
x1,y1=sm.fit_sample(xsc,y)
x1.shape
x1=pd.DataFrame(x1)
x1.rename(columns={0:'funded_amnt', 1:'grade', 2:'emp_length', 3:'annual_inc',

       4:'dti', 5:'delinq_2yrs', 6:'inq_last_6mths', 7:'mths_since_last_delinq',

       8:'pub_rec', 9:'revol_bal', 10:'revol_util', 11:'total_acc',

       12:'pub_rec_bankruptcies', 13:'term_36', 14:'home_ownership_MORTGAGE',

       15:'home_ownership_OTHER', 16:'home_ownership_OWN',

       17:'verification_status_Not Verified',

       18:'verification_status_Source Verified', 19:'purpose_car',

       20:'purpose_credit_card', 21:'purpose_debt_consolidation',

       22:'purpose_educational', 23:'purpose_home_improvement',

       24:'purpose_major_purchase', 25:'purpose_medical', 26:'purpose_moving',

       27:'purpose_other', 28:'purpose_renewable_energy', 29:'purpose_small_business',

       30:'purpose_vacation', 31:'purpose_wedding', 32:'debt_settlement_flag_N',

       33:'criteria_Yes'},inplace=True)
y1.shape
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,stratify=y,random_state=2)
xtrain1,xtest1,ytrain1,ytest1=train_test_split(x1,y1,test_size=0.3,stratify=y1,random_state=2)
sc1=StandardScaler()

xtrainsc=sc1.fit_transform(xtrain)

xtestsc=sc1.transform(xtest)
xtrain1
dir(list)
help(list.remove)
v=df6.columns.tolist().remove('loan_status')

v

v
xtrain1.head()
xtrain1.head()
xtrain1.shape
ytrain1.shape
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr1=LogisticRegression()
modellr=lr.fit(xtrainsc,ytrain).predict(xtestsc)
modellr1=lr1.fit(xtrain1,ytrain1).predict(xtest1)
from sklearn import metrics
metric_imbal=pd.Series([metrics.accuracy_score(ytest,modellr),metrics.precision_score(ytest,modellr,average='weighted'),

              metrics.recall_score(ytest,modellr,average='weighted'),metrics.f1_score(ytest,modellr,average='weighted'),

             metrics.cohen_kappa_score(ytest,modellr)],index=['accuracy_score','precision_score','recall_score','f1_score','Cohen_kappa_score'])

print(metric_imbal)

print()

print('Confusion Matrix:')

print(metrics.confusion_matrix(ytest,modellr))
metric_bal=pd.Series([metrics.accuracy_score(ytest1,modellr1),metrics.precision_score(ytest1,modellr1,average='weighted'),

              metrics.recall_score(ytest1,modellr1,average='weighted'),metrics.f1_score(ytest1,modellr1,average='weighted'),metrics.cohen_kappa_score(ytest1,modellr1)],

                     index=['accuracy_score','precision_score','recall_score','f1_score','cohen_kappa_score'])

print(metric_bal)

print()

print(metrics.classification_report(ytest1,modellr1))

print()

print('Confusion Matrix:')

print(metrics.confusion_matrix(ytest1,modellr1))
# K-Fold validation for imbalanced 



le1=LabelEncoder()

z=le1.fit_transform(y)
from sklearn.model_selection import KFold

from sklearn.metrics import roc_curve, auc

kf=KFold(n_splits=5,shuffle=True,random_state=2)

acc=[]

au=[]

for train,test in kf.split(x,z):

    M=LogisticRegression()

    Xtrain,Xtest=x.iloc[train,:],x.iloc[test,:]

    Ytrain,Ytest=z[train],z[test]

    M.fit(Xtrain,Ytrain)

    Y_predict=M.predict(Xtest)

    acc.append(metrics.accuracy_score(Ytest,Y_predict))

    fpr,tpr, _ = roc_curve(Ytest,Y_predict)

    au.append(auc(fpr, tpr))

print("Cross-validated Accuracy Mean Score:%.2f%% " % np.mean(acc))   

print("Cross-validated AUC Mean Score:%.2f%% " % np.mean(au))  

print("Cross-validated AUC Var Score:%.5f%% " % np.var(au,ddof=1)) 
# K-Fold Validation for balanced



le2=LabelEncoder()

z1=le2.fit_transform(y1)
from sklearn.model_selection import KFold

from sklearn.metrics import roc_curve, auc

kf=KFold(n_splits=5,shuffle=True,random_state=2)

acc=[]

au=[]

for train,test in kf.split(x1,z1):

    M=LogisticRegression()

    Xtrain,Xtest=x1.iloc[train,:],x1.iloc[test,:]

    Ytrain,Ytest=z1[train],z1[test]

    M.fit(Xtrain,Ytrain)

    Y_predict=M.predict(Xtest)

    acc.append(metrics.accuracy_score(Ytest,Y_predict))

    fpr,tpr, _ = roc_curve(Ytest,Y_predict)

    au.append(auc(fpr, tpr))

print("Cross-validated Accuracy Mean Score:%.2f%% " % np.mean(acc))   

print("Cross-validated AUC Mean Score:%.2f%% " % np.mean(au))  

print("Cross-validated AUC Var Score:%.5f%% " % np.var(au,ddof=1)) 
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(random_state=2)
modeldt=dt.fit(xtrain1,ytrain1).predict(xtest1)
metric_bal_dt=pd.Series([metrics.accuracy_score(ytest1,modeldt),metrics.precision_score(ytest1,modeldt,average='weighted'),

              metrics.recall_score(ytest1,modeldt,average='weighted'),metrics.f1_score(ytest1,modeldt,average='weighted'),

             metrics.cohen_kappa_score(ytest1,modeldt)],index=['accuracy_score','precision_score','recall_score','f1_score','Cohen_kappa_score'])

print(metric_bal_dt)

print()

print('Confusion Matrix:')

print(metrics.confusion_matrix(ytest1,modeldt))
print(metrics.classification_report(ytest1,modeldt))
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(random_state=2)
modelrf=rf.fit(xtrain1,ytrain1).predict(xtest1)
metric_bal_rf=pd.Series([metrics.accuracy_score(ytest1,modelrf),metrics.precision_score(ytest1,modelrf,average='weighted'),

              metrics.recall_score(ytest1,modelrf,average='weighted'),metrics.f1_score(ytest1,modelrf,average='weighted'),

             metrics.cohen_kappa_score(ytest1,modelrf)],index=['accuracy_score','precision_score','recall_score','f1_score','Cohen_kappa_score'])

print(metric_bal_rf)

print()

print('Confusion Matrix:')

print(metrics.confusion_matrix(ytest1,modelrf))
print(metrics.classification_report(ytest1,modelrf))
Importance = pd.DataFrame({'Importance':rf.feature_importances_*100}, index=xtrain1.columns)

Importance.sort_values('Importance', axis=0, ascending=True).plot(kind='barh', color='r',figsize=(12,18) )

plt.xlabel('Variable Importance')

plt.gca().legend_ = None
from sklearn.model_selection import GridSearchCV

grid = {'n_estimators': np.arange(1,25)}

rf_cv = GridSearchCV(rf, grid, cv=5) # GridSearchCV

rf_cv.fit(xtrain1, ytrain1)



print('The accuracy of the knn classifier is {:.2f} out of 1 on training data'.format(rf_cv.score(xtrain1, ytrain1)))

print('The accuracy of the knn classifier is {:.2f} out of 1 on test data'.format(rf_cv.score(xtest1, ytest1)))

print("Tuned hyperparameter k: {}".format(rf_cv.best_params_)) 

print("Best score: {}".format(rf_cv.best_score_))
from sklearn.model_selection import GridSearchCV

grid = {'n_estimators': np.arange(25,40)}

rf_cv = GridSearchCV(rf, grid, cv=5) # GridSearchCV

rf_cv.fit(xtrain1, ytrain1)



print('The accuracy of the knn classifier is {:.2f} out of 1 on training data'.format(rf_cv.score(xtrain1, ytrain1)))

print('The accuracy of the knn classifier is {:.2f} out of 1 on test data'.format(rf_cv.score(xtest1, ytest1)))

print("Tuned hyperparameter k: {}".format(rf_cv.best_params_)) 

print("Best score: {}".format(rf_cv.best_score_))
from sklearn.model_selection import GridSearchCV

grid = {'n_estimators': np.arange(40,60)}

rf_cv = GridSearchCV(rf, grid, cv=5) # GridSearchCV

rf_cv.fit(xtrain1, ytrain1)



print('The accuracy of the knn classifier is {:.2f} out of 1 on training data'.format(rf_cv.score(xtrain1, ytrain1)))

print('The accuracy of the knn classifier is {:.2f} out of 1 on test data'.format(rf_cv.score(xtest1, ytest1)))

print("Tuned hyperparameter k: {}".format(rf_cv.best_params_)) 

print("Best score: {}".format(rf_cv.best_score_))
rf1=RandomForestClassifier(n_estimators=29,random_state=2)

modelrf1=rf1.fit(xtrain1,ytrain1).predict(xtest1)
metric_bal_rf1=pd.Series([metrics.accuracy_score(ytest1,modelrf1),metrics.precision_score(ytest1,modelrf1,average='weighted'),

              metrics.recall_score(ytest1,modelrf1,average='weighted'),metrics.f1_score(ytest1,modelrf1,average='weighted'),

             metrics.cohen_kappa_score(ytest1,modelrf1)],index=['accuracy_score','precision_score','recall_score','f1_score','Cohen_kappa_score'])

print(metric_bal_rf1)

print()

print('Confusion Matrix:')

print(metrics.confusion_matrix(ytest1,modelrf1))
# K-Fold for all the models togeher with  
from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier
lr2=LogisticRegression()

knn=KNeighborsClassifier()

dt1=DecisionTreeClassifier(random_state=2)

rf1=RandomForestClassifier(random_state=2)

gnb = GaussianNB()


kf=KFold(n_splits=3,shuffle=True,random_state=2)

for model, name in zip([lr1,knn,dt1,rf1,gnb], ['Logistic','KNN','DecisionTree','RandomForest','NaiveBayes']):

    roc_auc=[]

    for train,test in kf.split(x1,z1):

        Xtrain,Xtest=x1.iloc[train,:],x1.iloc[test,:]

        Ytrain,Ytest=z1[train],z1[test]

        model.fit(Xtrain,Ytrain)

        Y_predict=model.predict(Xtest)

        #cm=metrics.confusion_matrix(Ytest,Y_predict)

        fpr,tpr, _ = roc_curve(Ytest,Y_predict)

        roc_auc.append(auc(fpr, tpr))

    print(roc_auc)

    print("AUC scores: %0.02f (+/- %0.5f) [%s]" % (np.mean(roc_auc), np.var(roc_auc,ddof=1), name ))

    print(metrics.confusion_matrix(Ytest,Y_predict))

    print(metrics.cohen_kappa_score(Ytest,Y_predict))

    print()
#t=df4.earliest_cr_line.dropna().index
#df4.earliest_cr_line.dropna().values
# tim=[]

# for i in df4.earliest_cr_line.dropna().values:

#     x=i[-4:]

#     tim.append(x)
# time=pd.Series(tim,index=t)
# time.shape
# mask=df4.earliest_cr_line.isnull()

# timen=df4[mask]['earliest_cr_line']
# timen.shape
# timen
# cr=pd.concat([time,timen],axis=0)
# cr.sort_index(inplace=True)
# df4['earliest_cr_line']=cr.values
#df4['earliest_cr_line']=pd.to_datetime(df4['earliest_cr_line'])
# df4['earliest_cr_line'].head()
#Bagging with logistic

from sklearn.ensemble import BaggingClassifier
bagg=BaggingClassifier(base_estimator=lr)
labels_bagg=bagg.fit(xtrain1,ytrain1).predict(xtest1)
metric_bal_bagg=pd.Series([metrics.accuracy_score(ytest1,labels_bagg),metrics.precision_score(ytest1,labels_bagg,average='weighted'),

              metrics.recall_score(ytest1,labels_bagg,average='weighted'),metrics.f1_score(ytest1,labels_bagg,average='weighted'),

             metrics.cohen_kappa_score(ytest1,labels_bagg)],index=['accuracy_score','precision_score','recall_score','f1_score','Cohen_kappa_score'])

print(metric_bal_bagg)

print()

print('Confusion Matrix:')

print(metrics.confusion_matrix(ytest1,labels_bagg))
#Bagging with Decision Tree

bagg1=BaggingClassifier(base_estimator=dt)

labels_bagg1=bagg1.fit(xtrain1,ytrain1).predict(xtest1)
metric_bal_bagg1=pd.Series([metrics.accuracy_score(ytest1,labels_bagg1),metrics.precision_score(ytest1,labels_bagg1,average='weighted'),

              metrics.recall_score(ytest1,labels_bagg1,average='weighted'),metrics.f1_score(ytest1,labels_bagg1,average='weighted'),

             metrics.cohen_kappa_score(ytest1,labels_bagg1)],index=['accuracy_score','precision_score','recall_score','f1_score','Cohen_kappa_score'])

print(metric_bal_bagg1)

print()

print('Confusion Matrix:')

print(metrics.confusion_matrix(ytest1,labels_bagg1))
#bagging with rf

bagg2=BaggingClassifier(base_estimator=rf)

labels_bagg2=bagg2.fit(xtrain1,ytrain1).predict(xtest1)
metric_bal_bagg2=pd.Series([metrics.accuracy_score(ytest1,labels_bagg2),metrics.precision_score(ytest1,labels_bagg2,average='weighted'),

              metrics.recall_score(ytest1,labels_bagg2,average='weighted'),metrics.f1_score(ytest1,labels_bagg2,average='weighted'),

             metrics.cohen_kappa_score(ytest1,labels_bagg2)],index=['accuracy_score','precision_score','recall_score','f1_score','Cohen_kappa_score'])

print(metric_bal_bagg2)

print()

print('Confusion Matrix:')

print(metrics.confusion_matrix(ytest1,labels_bagg2))
from sklearn.ensemble import GradientBoostingClassifier
gbm=GradientBoostingClassifier(max_depth=4)
labels_gbm=gbm.fit(xtrain1,ytrain1).predict(xtest1)
metric_bal_gbm=pd.Series([metrics.accuracy_score(ytest1,labels_gbm),metrics.precision_score(ytest1,labels_gbm,average='weighted'),

              metrics.recall_score(ytest1,labels_gbm,average='weighted'),metrics.f1_score(ytest1,labels_gbm,average='weighted'),

             metrics.cohen_kappa_score(ytest1,labels_gbm)],index=['accuracy_score','precision_score','recall_score','f1_score','Cohen_kappa_score'])

print(metric_bal_gbm)

print()

print('Confusion Matrix:')

print(metrics.confusion_matrix(ytest1,labels_gbm))
from sklearn.ensemble import VotingClassifier
vc=VotingClassifier(estimators=[('Dt',dt),('Log reg',lr),('RF',rf),('Bagg',bagg),('Boost',gbm)])
labels_vc=vc.fit(xtrain1,ytrain1).predict(xtest1)
metric_bal_vc=pd.Series([metrics.accuracy_score(ytest1,labels_vc),metrics.precision_score(ytest1,labels_vc,average='weighted'),

              metrics.recall_score(ytest1,labels_vc,average='weighted'),metrics.f1_score(ytest1,labels_vc,average='weighted'),

             metrics.cohen_kappa_score(ytest1,labels_vc)],index=['accuracy_score','precision_score','recall_score','f1_score','Cohen_kappa_score'])

print(metric_bal_vc)

print()

print('Confusion Matrix:')

print(metrics.confusion_matrix(ytest1,labels_vc))
from sklearn.ensemble import AdaBoostClassifier
ada=AdaBoostClassifier(base_estimator=rf)
labels_ada=ada.fit(xtrain1,ytrain1).predict(xtest1)
metric_bal_ada=pd.Series([metrics.accuracy_score(ytest1,labels_ada),metrics.precision_score(ytest1,labels_ada,average='weighted'),

              metrics.recall_score(ytest1,labels_ada,average='weighted'),metrics.f1_score(ytest1,labels_ada,average='weighted'),

             metrics.cohen_kappa_score(ytest1,labels_ada)],index=['accuracy_score','precision_score','recall_score','f1_score','Cohen_kappa_score'])

print(metric_bal_ada)

print()

print('Confusion Matrix:')

print(metrics.confusion_matrix(ytest1,labels_ada))
import xgboost as xgb
xg= xgb.XGBClassifier(max_depth=2, learning_rate=0.01) # 0.78947
# Fitting the Model

labels_xgb = xg.fit(xtrain1,ytrain1).predict(xtest1)
print(metrics.precision_score(ytest1,labels_xgb,average='weighted'))

print(metrics.recall_score(ytest1,labels_xgb,average='weighted'))

print(metrics.f1_score(ytest1,labels_xgb,average='weighted'))
metrics.cohen_kappa_score(ytest1,labels_xgb)