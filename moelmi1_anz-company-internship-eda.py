import matplotlib.pyplot as plt
import seaborn as sns
import pandas as  pd 
plt.rcParams['figure.figsize'] = (20, 10)
%matplotlib inline

df=pd.read_csv("../input/anz-synthesised-transaction-dataset/anz.csv")
df.head()
df.dtypes
df.isnull().sum()
df2=df.drop(['merchant_code','bpay_biller_code'],axis=1)
df2.describe()
upper=df2['amount'].quantile(0.99)
lower=df2['amount'].quantile(0.01)
df2[(df2['amount']<upper)&(df2['amount']>lower)]
lower_limit=df2['balance'].quantile(0.01)
Upper_limit=df2['balance'].quantile(0.99)
df3=df2[(df2['balance']>lower_limit)&(df2['balance']<Upper_limit)]
df3['date']=df3['date'].apply(pd.to_datetime)

#extracting months and day from date column 
df3['Month']=df3['date'].dt.month
#since the months are 8,9,10 ,let's name it august , sept, oct respectively
months={8:'August',9:'Sept',10:'October'}
df3['Month']=df3['Month'].map(months)
df3.head()
#extract the   name of the day from the date column as well using dt.weekday
df3['Weekday']=df3['date'].dt.weekday
days_of_week={0:'MON',1:'Tues',2:'Wed',3:'Thurs',4:'Fri',5:'Sat',6:'Sun'}
df3['Weekday']=df3['Weekday'].map(days_of_week)
#extract the time of transaction for the transaction column 
df3['Time of trasncation']=df3['extraction'].apply(lambda x:x.split('T')[1].split(":")[0])
#now let's check the correlation between the features using heatmap
plt.rcParams['figure.figsize'] = (12, 8)
sns.heatmap(df3.corr() ,annot=True,vmax=0.3,linewidths=0.5)
#checking average transaction of the month of October
oct=df3['Month']=='October'
df3.loc[oct,'amount'].mean()


#maximum transaction of the month of october
oct=df3['Month']=='October'
maximum_transaction= df3.loc[oct,'amount'].max()
maximum_transaction
#minimum transaction of the month of october
oct=df3['Month']=='October'
minimum_transaction =df3.loc[oct,'amount'].min()
minimum_transaction
#let's the customers who made the maximum and minimum transactions on the month of october
oct_trans= df3[df3['Month']=='October']
oct_trans[oct_trans['amount']==oct_trans['amount'].max()]['first_name']

oct_trans= df3[df3['Month']=='October']
oct_trans[oct_trans['amount']==oct_trans['amount'].min()]['first_name']
#checking average transaction of the month of Sept
sept=df3['Month']=='Sept'
df3.loc[sept,'amount'].mean()


#minimum transaction of the month of sept
Sept=df3['Month']=='Sept'
minimum_transaction_Sept =df3.loc[Sept,'amount'].min()
minimum_transaction_Sept
#maximum transaction of the month of Sept
Sept=df3['Month']=='Sept'
maximum_transaction_sept =df3.loc[Sept,'amount'].max()
maximum_transaction_sept
#let's the customers who made the maximum and minimum transactions on the month of Sept
Sept_trans= df3[df3['Month']=='Sept']
Sept_trans[Sept_trans['amount']==Sept_trans['amount'].max()]['first_name']

#let's the customers who made the maximum and minimum transactions on the month of Sept
Sept_trans= df3[df3['Month']=='Sept']
Sept_trans[Sept_trans['amount']==Sept_trans['amount'].min()]['first_name']
#checking average transaction of the month of August
august=df3['Month']=='August'
df3.loc[august,'amount'].mean()

#minimum transaction of the month of August
august=df3['Month']=='August'
minimum_transaction_august =df3.loc[august,'amount'].min()
minimum_transaction_august
#maximum  transaction of the month of August
august=df3['Month']=='August'
max_transaction_august =df3.loc[august,'amount'].max()
max_transaction_august
#let's the customers who made the maximum and minimum transactions on the month of August
August_trans= df3[df3['Month']=='August']
August_trans[August_trans['amount']==August_trans['amount'].max()]['first_name']
#let's the customers who made the maximum and minimum transactions on the month of August
#check for minimum 
August_trans= df3[df3['Month']=='August']
August_trans[August_trans['amount']==August_trans['amount'].min()]['first_name']

df3.head()
#let's check the how the amount of transaction has changed during the three months
sns.catplot(x='Month',y='amount',data=df3,kind='bar' )
plt.title('Monthly transactions last three Months')


#let's check the how the amount of transaction has changed during the three months
sns.catplot(x='Weekday',y='amount',data=df3,kind='bar' )
plt.title('Transactions in days of the week')
plt.xticks(rotation=45)
#let's check what time of the day does most of the transactions occurs
sns.relplot(x='Time of trasncation',y='amount',data=df3,kind='line' )
plt.title('Transactions in days of the week')
plt.xticks(rotation=90)
#let's have a look on the  most time of transactions in the whole week 
sns.relplot(x='Time of trasncation',y='amount',data=df3,kind='line', row='Weekday' )
plt.title('Transactions in days of the week')
plt.xticks(rotation=90)
plt.savefig('output.png',dpi=300)

#let's have a look on the sum of transactions in the last three months 
sum_of_months=df3.groupby('Month')['amount'].agg('sum')


plt.plot(sum_of_months.sort_values(ascending=True),marker='o',linestyle='dashed')
plt.title('Total transactions of last three Months')
plt.savefig('total_three_sum.png')
sum_of_days=df3.groupby('Weekday')['amount'].agg('sum')


plt.plot(sum_of_days.sort_values(ascending=True),marker='o',linestyle='dashed')
plt.title('Total transactions during the days ')
plt.savefig('total_week.png')
mean_of_months=df3.groupby('Month')['amount'].agg('mean').sort_values(ascending=True)
plt.plot(mean_of_months,marker='o',linestyle='dashed',)
plt.title('Mean Transactions of the three Months')
plt.savefig('mean_transactions_in_month.png')
mean_of_the_days=df3.groupby('Weekday')['amount'].agg('mean').sort_values(ascending=True)
plt.plot(mean_of_the_days,marker='o',linestyle='dashed',)
plt.title('Mean Transactions of the Weekdays')
plt.savefig('mean_transactions_in_days.png')
#let's compare the amount of transactions to the gender.
sns.catplot(x='amount',y='gender',data=df3,kind='bar')
plt.savefig('gender.png')
#let's now make the gender the hue 
sns.countplot(x='Month',hue='gender',data=df3)
plt.title('Transaction of the last three months based on gender')
plt.savefig('Transaction of the last three months based on gender.png')
#let's have a look on which gender makes more transactions during the day
ax=sns.countplot(x='Weekday',hue='gender',data=df3,)
total=float(len(df3))
for p in ax.patches:
  height=p.get_height()
  ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:.2%}'.format(height/total),
            ha="center") 
  
plt.title('Number of transactions made on each day  with a gender comparison ')

sns.relplot(x='age',y='balance',data=df3,kind='line')
plt.title("The effect of Age on the Cutomer's Balance")
plt.savefig('age-balance.png')
#let's have a look from where most of the transactions occur
df['merchant_state'].value_counts()
sns.set(style="darkgrid")
sns.countplot(df3['merchant_state'])
plt.title('Number of Transaction done in each state')
plt.savefig('No_trans_per_State')
#let's have a look on where the transactions took place 
sns.set(style="darkgrid")
ax=sns.countplot(df3['txn_description'])
total=float(len(df3))
for p in ax.patches:
  height=p.get_height()
  ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:.2%}'.format(height/total),
            ha="center") 
plt.title('Percentage of Source by reason of the transaction')
plt.savefig('placeoftrans.png')
#let's check the amount of transaction that occured in each state
plt.rcParams['figure.figsize'] = (20, 10)

sns.catplot(x='merchant_state',y='amount',data=df3,kind='bar')
plt.title('Amount of transactions in each state')
#let's take a look on the mean transaction for each gender 
genderism=df3.groupby('gender')
genderism['amount'].mean()
#let's have a look for those who used debit or credit for different gender 
plt.figure(figsize=(8,8))
sns.countplot(x='movement',hue='gender',data=df3)
#let's have a look for our best 5 customer who made the maximum transactions
print(df3['first_name'].value_counts().head(5))
sns.countplot(y='first_name', order=df3['first_name'].value_counts().head(5).index[:5] ,data=df3)
#let's have a look on the least 10 customers 
print(df3['first_name'].value_counts(sort=True).nsmallest(10))
tail_cust=df3['first_name'].value_counts(sort=True).nsmallest(10)
mycolors=['r','b','k','y','m','c','#16A085','salmon' , '#32e0c4']
tail_cust.plot.barh(color=mycolors)
#let's see the min and max amount of trans in the state
merchan_grp=df3.groupby('merchant_state')
agg_merchant_states=merchan_grp['amount'].agg(['mean','max','min'])
agg_merchant_states
#let's see the states with maximum transactions
agg_merchant_states['max'].plot.barh(color=mycolors)
plt.title('Maximum amount of transactions in Each state')
plt.xlabel('Amount')
plt.ylabel('Merchant_state')
month_grp = df3.groupby(['Month'])
avg_amt_tran_month = month_grp['amount'].mean()
avg_amt_tran_month.plot.barh(color=mycolors)
day_grp = df3.groupby(['Weekday'])
avg_amt_tran_day = day_grp['amount'].mean()
avg_amt_tran_day.plot.barh(color=mycolors)