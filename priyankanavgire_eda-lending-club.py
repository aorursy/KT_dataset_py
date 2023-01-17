import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns


SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
# import chardet
# rawdata = open('loan.csv', 'rb').read()
# result = chardet.detect(rawdata)
# charenc = result['encoding']
# print(charenc)
df=pd.read_csv('../input/loan.csv', encoding='utf-8',low_memory=False)
df.shape
df.dropna(axis='columns', how='all', inplace = True)
df.shape
df.drop(['next_pymnt_d','mths_since_last_record','mths_since_last_delinq','desc','member_id','url',
         'recoveries','emp_title','earliest_cr_line','open_acc','funded_amnt',
         'initial_list_status','funded_amnt_inv','pymnt_plan','total_pymnt','revol_bal',
         'collections_12_mths_ex_med','acc_now_delinq','chargeoff_within_12_mths','delinq_amnt',
         'tax_liens','title','total_pymnt_inv','out_prncp_inv','zip_code','policy_code','out_prncp','total_pymnt'
         ,'total_rec_prncp','application_type','total_rec_int','total_rec_late_fee','recoveries',
         'collection_recovery_fee', 'last_pymnt_d','last_pymnt_amnt','last_credit_pull_d','id','installment'], axis=1,inplace= True)
df.shape
df.isnull().sum()
df.dropna(how="any", inplace=True)
df.shape
df.isnull().sum()
df.loc[:,'int_rate'] = df['int_rate'].map(lambda x: str(x).split('%')[0])
df['int_rate']=df.int_rate.astype(float)
df.loc[:,'revol_util'] = df['revol_util'].map(lambda x: str(x).split('%')[0])
df['revol_util']=df.revol_util.astype(float)
df['term']=df['term'].map(lambda x: str(x).replace('months','')).astype(int)
df.to_csv("clean_df.csv", index=False )
df=pd.read_csv('clean_df.csv', encoding='utf-8')
df['issue_d']=pd.to_datetime(df["issue_d"], format='%b-%y', yearfirst=False)
df['year']=df['issue_d'].dt.year
print(df.dtypes)
df.shape
print(df.loan_status.value_counts(normalize=True))
df_fp=df.loc[(df.loan_status=='Fully Paid'),]
df_co=df.loc[(df.loan_status=='Charged Off'),]
print(df.loan_amnt.describe())
plt.figure(figsize=(15,5))
# plt.matplotlib.rcParams.update({'font.size': 15})
plt.subplot(131)
plt.title('Loan amount wise distribution')
sns.distplot(df['loan_amnt'])
plt.subplot(132)
plt.title('Year wise loan disbursal- cummulative')
df.groupby('year').loan_amnt.sum().plot()
plt.subplot(133)
plt.title('Loan count - categiory wise')
sns.countplot(x="loan_status", data=df)
print(df.int_rate.describe())
plt.figure(figsize=(15,5))
plt.subplot(131)
plt.title('Interest rate spread- fully paid customer')
sns.distplot(df_fp.int_rate)
plt.subplot(132)
plt.title('Interest rate spread- charge off customer')
sns.distplot(df_co.int_rate)
plt.subplot(133)
plt.title('Interest rate movement-YOY')
df.groupby('year').int_rate.median().plot()
plt.figure(figsize=(15,4))
plt.subplot(121)
plt.title('Salary range for charge off customer')
sns.distplot(df_co.annual_inc, hist=False)
plt.xscale('log')
plt.subplot(122)
plt.title('Salary range for fully paid customer')
sns.distplot(df_fp.annual_inc, hist=False)
plt.xscale('log')
plt.figure(figsize=(20,10))
plt.subplot(121)
plt.title('Statewise fully paid loans count')
sns.countplot(y='addr_state', data=df_fp)
plt.subplot(122)
plt.title('Statewise charge off loans count')
sns.countplot(y='addr_state', data=df_co)
print(round(df.purpose.value_counts(normalize=True)*100),2)
plt.figure(figsize=(10,10))
plt.subplot(211)
plt.title('Purpose wise loan distribtion- fully paid')
sns.countplot(y="purpose", data=df_fp)
plt.subplot(212)
plt.title('Purpose wise loan distribtion- charge off')
sns.countplot(y="purpose", data=df_co)
cm = sns.light_palette("red", as_cmap=True)
pd.crosstab(df['purpose'], df['loan_status']).style.background_gradient(cmap = cm)
print(df.pivot_table( values='dti',index = 'loan_status', aggfunc=np.median))
df.pivot_table( values='dti',index = 'grade', aggfunc=np.median)
plt.title('loan amount spead vs loan status')
sns.boxplot(y='loan_status', x='loan_amnt', data=df)
bins=[0,5600,10000,15000,35000]
groups=['Q1','Q2','Q3','Q4']
df['loan_amount']=pd.cut(df['loan_amnt'],bins, labels=groups)
df.pivot_table( values='term',index = 'loan_status', columns = 'loan_amount', aggfunc=len).div(len(df.index)).mul(100)
round(df.groupby('home_ownership').loan_status.value_counts(normalize=True)*100,2)
plt.figure(figsize=(10,3))
sns.countplot(y='delinq_2yrs', data=df_fp)
sns.countplot(y='delinq_2yrs', data=df_co)
df.pivot_table( values='term',index = 'loan_status', columns = 'delinq_2yrs', aggfunc='count')
print(df.pub_rec_bankruptcies.value_counts(normalize=True))
df.pivot_table( values='term',index = 'loan_status', columns = 'pub_rec_bankruptcies', aggfunc='count')
print(df.pub_rec.value_counts(normalize=True))
df.pivot_table( values='term',index = 'loan_status', columns = 'pub_rec', aggfunc='count')
print(df.pivot_table( values='loan_amnt',index = 'purpose', columns = 'loan_status', aggfunc='count'))
print(df.pivot_table( values='term',index = 'loan_status', columns = 'emp_length', aggfunc=len).div(len(df.index)).mul(100))
sns.barplot(x='annual_inc',y='emp_length',data=df)
plt.xscale('log')
# df.emp_length.value_counts()
plt.figure(figsize=(10,5))
plt.subplot(121)
sns.boxplot(x='loan_status',y='revol_util',data=df)
plt.subplot(122)
sns.boxplot(x='grade',y='revol_util',data=df)
df.pivot_table( values='term',index = 'loan_status', columns = 'home_ownership', aggfunc='count')
plt.figure(figsize=(15,5))
plt.subplot(121)
sns.barplot(x='grade', y='loan_amnt', data=df_fp)
plt.subplot(122)
sns.barplot(x='grade', y='loan_amnt', data=df_co)
df.pivot_table( values='term',index = 'loan_status', columns = 'grade', aggfunc='count')
plt.figure(figsize=(15,5))
sns.boxplot(x='sub_grade',y='revol_util',data=df)
plt.show()
plt.figure(figsize=(20,10))

df_corr = df.copy()
df_corr['grade'] = df_corr['grade'].astype('category')
df_corr['grade'].cat.categories = [6,5,4,3,2,1,0]
df_corr['grade'] = df_corr['grade'].astype('float')

df_corr['verification_status'] = df_corr['verification_status'].astype('category')
df_corr['verification_status'].cat.categories = [0,2,1]
df_corr['verification_status'] = df_corr['verification_status'].astype('float')


ordered_emplength_status = ['< 1 year','1 year','2 years','3 years','4 years','5 years','6 years','7 years','8 years','9 years','10+ years']
df_corr['emp_length'] = df_corr.emp_length.astype("category",ordered=True,categories=ordered_emplength_status).cat.codes

ordered_loan_status = ['Charged Off','Current','Fully Paid']
df_corr['loan_status'] = df_corr.loan_status.astype("category",ordered=True,categories=ordered_loan_status).cat.codes
df_corr['loan_status'].value_counts()

sns.set_context("paper", font_scale=2)
sns.heatmap(df_corr.corr(), vmax=.8, square=True, annot=True, fmt='.1f')
plt.show()