# importing libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import matplotlib.dates
# reading file

file_path="../input/anz-synthesised-transaction-dataset/anz.csv"
df=pd.read_csv(file_path)
df.head()
# shape of the dataframe

df.shape
# info of the dataframe

df.info()
# total null values

df.isnull().sum() #or df.isna().sum() 
# classifying NA as categorical or numerical 

NA=df[['card_present_flag','bpay_biller_code','merchant_id','merchant_code','merchant_suburb','merchant_state','merchant_long_lat']]
NAcat=NA.select_dtypes(include='object')
NAnum=NA.select_dtypes(exclude='object')
print(NAcat.shape[1],'categorical features with missing values')
print(NAnum.shape[1],'numerical features with missing values')
# visulaizing missing values percentage

plt.figure(figsize=(10,5))
allna = (df.isnull().sum() / len(df))*100
allna = allna.drop(allna[allna == 0].index).sort_values()
allna.plot.barh(color=('red', 'black'), edgecolor='black')
plt.title('Missing values percentage per column',bbox={'facecolor':'0.9', 'pad':5})
plt.xlabel('Percentage', fontsize=15)
plt.ylabel('Features with missing values',fontsize=15)
plt.yticks(weight='bold')
plt.show()
# removing columns

df.drop(['bpay_biller_code','merchant_code'],axis=1,inplace=True)
df.info()
# vectorizing categorical values 

# 'status'
df.status = pd.Categorical(df.status)
df['cat_status']=df.status.cat.codes

# 'txn_description'
df.txn_description = pd.Categorical(df.txn_description)
#td_cat=df.txn_description.astype('category').cat.codes
df['cat_txn_description']=df.txn_description.cat.codes

# 'merchant_state'
df.merchant_state = pd.Categorical(df.merchant_state)
df['cat_merchant_state']=df.merchant_state.cat.codes
# correlation between some features

cor_mat = df[['card_present_flag' , 'amount' , 'balance' ,'age','cat_status','cat_txn_description','cat_merchant_state']].corr()
cor_mat
# visualizing the correlation heatmap 

plt.figure(figsize=(8,8))
# Custom cmap pallete
cmap = sns.diverging_palette(220 , 10 , as_cmap=True)

# Building heatmap
sns.heatmap(cor_mat ,vmax=.3 , center=0 , cmap=cmap , square=True , linewidths=.5 , cbar_kws={'shrink': .5})
plt.title("Correlation between features",bbox={'facecolor':'0.9', 'pad':5})
# average of some numerical data

df.mean()
#Colors for the bar of the graph

my_colors = ['r','b','k','y','m','c','#16A085','salmon' , '#32e0c4']
# visualize transaction trend

tt=df.groupby(['date'])
ttc=tt.date.count()

plt.figure(figsize=(20,8))
sns.lineplot(data=ttc)
plt.title("Transaction trend",bbox={'facecolor':'0.9', 'pad':5})
plt.xlabel("Date")
plt.ylabel("Number of transactions")
# average number of transactions per day

ttc.mean()
# converting the date column to pandas Timestap

df['date'] = pd.to_datetime(df['date'])
# extracting day name 

df['day_name'] = df['date'].dt.day_name()
df['day_name'].head()
# extracting month name

df['month_name'] = df['date'].dt.month_name()
df['month_name'].head()
# months generated

df['month_name'].value_counts()
# visualize month wise transaction count

plt.figure(figsize=(15,5))
plt.title("Month wise transaction count",bbox={'facecolor':'0.9', 'pad':5})
sns.countplot(x='month_name' , data=df)
plt.ylabel("Count",fontsize=15)
plt.xlabel("Month",fontsize=15)
# visualize percentage of contribution from each month

pie_color = ['orange' , 'salmon', 'lightblue']
fig,ax = plt.subplots(figsize=(10,10)) # (height,width)

df['month_name'].value_counts(sort=True).plot.pie(labeldistance=0.2 ,
                                         colors=pie_color,
                                        autopct='%.2f', shadow=True, startangle=140,pctdistance=0.8 , radius=1)
plt.title("Percentage of contribution from each months", bbox={'facecolor':'0.8', 'pad':5})
# month wise transaction amount

month_amount=df.groupby(['month_name']).amount.agg([sum])
ma=month_amount.sort_values(by='sum',ascending=False)
ma
# visualize month wise transaction amount

plt.figure(figsize = (10,5))
df.groupby('month_name').amount.sum().plot(kind='bar')
plt.title("Month wise transaction amount",bbox={'facecolor':'0.9', 'pad':5})
plt.ylabel("Amount",fontsize=15)
plt.xlabel("Month",fontsize=15)
# visualize average transaction amount each month

month_grp = df.groupby(['month_name'])
avg_amt_tran_month = month_grp['amount'].mean()

fig,ax = plt.subplots(figsize=(10,5)) # (height,width)
print(avg_amt_tran_month);
avg_amt_tran_month.plot.barh(color=my_colors)
ax.set(xlabel="Average amount",
      ylabel="Month")
plt.title('Average transaction amount each month',bbox={'facecolor':'0.9', 'pad':5})
oct_amt_tran_month = month_grp['amount'].value_counts().loc['October']
oct_amt_tran_month
oct_date = month_grp['date'].value_counts().loc['October']
# amount transacted in October month

filt = (df['month_name'] == 'October')
df.loc[filt , 'amount']
# average amount in october month

df.loc[filt , 'amount'].mean()
# maximum value transacted in October month 

df.loc[filt , 'amount'].max()
# minimum value transacted in October month 

df.loc[filt , 'amount'].min()
# gender wise transaction count

gencg=df.groupby('gender').gender.count()
gencs=gencg.sort_values(ascending=False)
gencs

# visualize gender wise transaction count

plt.figure(figsize = (10,5))
ax = sns.countplot(x = 'gender', data = df, palette = 'pastel')
ax.set_title(label = 'Gender wise transaction count',bbox={'facecolor':'0.9', 'pad':5})
ax.set_ylabel(ylabel = 'Count', fontsize = 16)
ax.set_xlabel(xlabel = 'Gender', fontsize = 16)
plt.legend()
# visualize percentage of contribution from each gender 

plt.figure(figsize=(5,5))
df['gender'].value_counts(normalize=True).plot.pie(autopct='%.2f',labels=['Male',
                                                                         'Female'], labeldistance=0.5 ,
                                                   shadow=True, startangle=140,pctdistance=0.2 , radius=1)
plt.title('Percentage of contribution from gender' , bbox={'facecolor':'0.8', 'pad':5})
# gender wise transaction amount

genag=df.groupby(['gender']).amount.agg([sum])
genag
# average transaction amount by gender 

gender_grp = df.groupby(['gender'])
gen_trans_amt = gender_grp['amount'].mean()
gen_trans_amt
# total transaction amount by gender 

gender_total = df.groupby(['gender'])
gen_total_amt = gender_grp['amount'].sum()
gen_total_amt
# visualize gender wise transaction amount

fig,ax = plt.subplots(figsize=(10,5))
gen_total_amt.plot.barh(color=my_colors)
plt.title("Gender wise transaction amount",bbox={'facecolor':'0.9', 'pad':5})
plt.ylabel("Gender",fontsize=15)
plt.xlabel("Amount",fontsize=15)
# visualize average amount transacted by gender

fig,ax = plt.subplots(figsize=(10,5))
gen_trans_amt.plot.barh(color=my_colors)
plt.title("Average amount of transactions by gender",bbox={'facecolor':'0.9', 'pad':5})
plt.ylabel("Gender",fontsize=15)
plt.xlabel("Amount",fontsize=15)
# visualize month with highest number of transaction based on gender

plt.figure(figsize=(20,6))
sns.countplot(x='month_name' ,hue='gender', data=df)
plt.title('Month with highest number of transaction based on gender',bbox={'facecolor':'0.9', 'pad':5})
plt.xlabel("Month",fontsize=15)
plt.ylabel("Count",fontsize=15)
# visualize day wise transaction count

plt.figure(figsize=(10,5))
sns.countplot(x='day_name' , data=df)
plt.title("Day wise transaction count",bbox={'facecolor':'0.9', 'pad':5})
plt.ylabel("Count",fontsize=15)
plt.xlabel("Day",fontsize=15)
# average amount transacted on particular Day : Monday

day_name_grp = df.groupby(['day_name'])
day_name_grp['amount'].mean().loc['Monday']
# visualize day wise gender transaction count

plt.figure(figsize=(15,7))
ax = sns.countplot(x="day_name", hue="gender", data=df) # for Seaborn version 0.7 and more
total = float(len(df))
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:.2%}'.format(height/total),
            ha="center") 
plt.ylabel("Count",fontsize=15)
plt.xlabel("Day",fontsize=15)
plt.title('Number of transaction made on each day of a week with gender comparison',bbox={'facecolor':'0.9', 'pad':5})
plt.show()
# visualize distribution of age

plt.figure(figsize=(15,5))
sns.distplot(df['age']);
plt.title('Distribution of age',bbox={'facecolor':'0.9', 'pad':5})
plt.xlabel('Age',fontsize=15)
# visualize age with balance

plt.figure(figsize=(15,5))
sns.lineplot(x='age' , y='balance' , data=df)
plt.title('Age and balance',bbox={'facecolor':'0.9', 'pad':5})
plt.xlabel('Age',fontsize=15)
plt.ylabel('Balance',fontsize=15)
# visualize age with amount

plt.figure(figsize=(15,5))
sns.lineplot(x='age' , y='amount' , data=df)
plt.title('Age and amount',bbox={'facecolor':'0.9', 'pad':5})
plt.xlabel('Age',fontsize=15)
plt.ylabel('Amount',fontsize=15)
# age wise transaction count

agecg=df.groupby('age').age.count()
#agecs=agecg.sort_values(ascending=False)
agecg
# visualize age wise transaction count

plt.figure(figsize = (10, 5))
ax = sns.countplot(x = 'age', data = df, palette = 'pastel')
ax.set_title(label = 'Age wise transaction count',bbox={'facecolor':'0.9', 'pad':5})
ax.set_ylabel(ylabel = 'Count', fontsize = 15)
ax.set_xlabel(xlabel = 'Age', fontsize = 15)
plt.show()
# age wise transaction amount

ageag=df.groupby(['age']).amount.agg([sum])
ageag
# visualize age wise transaction amount

plt.figure(figsize = (10,5))
df.groupby('age').amount.sum().plot(kind='bar')
plt.title("Age wise transaction amount",bbox={'facecolor':'0.9', 'pad':5})
plt.ylabel("Total amount",fontsize=15)
plt.xlabel("Age",fontsize=15)
# visualize type of transaction

print(df['txn_description'].value_counts())
sns.set(style="darkgrid")
plt.figure(figsize=(10,5))
ax = sns.countplot(df['txn_description'])
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:.2%}'.format(height/total),
            ha="center") 
plt.title('Percentage of type of transaction',bbox={'facecolor':'0.9', 'pad':5})
plt.ylabel('Number of Transaction',fontsize=15)
plt.xlabel('Transaction Description',fontsize=15)
plt.show()
# visualize state wise transaction count

print(df['merchant_state'].value_counts())
plt.figure(figsize=(10,5))
sns.countplot(df['merchant_state'])
plt.title('Number of transaction done on each state',bbox={'facecolor':'0.9', 'pad':5})
plt.ylabel("Count",fontsize=15)
plt.xlabel("State",fontsize=15)
plt.show()
# group using merchant_state 

mer_state_grp = df.groupby(['merchant_state'])
# visualize number of transaction in merchant state by gender

print(mer_state_grp['gender'].value_counts(normalize=True))
gen_mer_state = mer_state_grp['gender'].value_counts()
fig,ax = plt.subplots(figsize=(15,5))
gen_mer_state.plot.barh()
ax.set(xlabel="Number of transaction",
      ylabel="State and Gender")
plt.title('Number of transaction in a state',bbox={'facecolor':'0.9', 'pad':5})
# maximum,minimum and average amount transacted in each merchant state

agg_amt_state = mer_state_grp['amount'].agg(['min' , 'mean' , 'max'])
agg_amt_state
# visualize minimum amount transacted in each state

fig,ax = plt.subplots(figsize=(15,5)) # (height,width)
print(agg_amt_state['min'])
agg_amt_state['min'].plot.barh(color=my_colors)
ax.set(xlabel="Number of transaction",
      ylabel="State")
plt.title('Minimum Number of transaction in a state',bbox={'facecolor':'0.9', 'pad':5})
# visualize maximum amount transacted in each state

fig,ax = plt.subplots(figsize=(15,5)) # (height,width)
print(agg_amt_state['max'])
agg_amt_state['max'].plot.barh(color=my_colors)
ax.set(xlabel="Amount",
      ylabel="State")
plt.title('Maximum amount transacted in each state',bbox={'facecolor':'0.9', 'pad':5})
# visualize movement type

plt.figure(figsize=(10,5))
print(df['movement'].value_counts())
sns.countplot(df['movement'])
ax.set(xlabel="Movement",
      ylabel="Count")
plt.title('Movement type',bbox={'facecolor':'0.9', 'pad':5})
# visualize transaction movement by gender

plt.figure(figsize=(10,5))
ax = sns.countplot(df['movement'] , hue=df['gender'])
total = float(len(df))
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:.2%}'.format(height/total),
            ha="center") 
plt.title('Transaction movement type by gender',bbox={'facecolor':'0.8', 'pad':5})
#employee dataframe
#df_emp = df[~df['txn_description'].isin(['POS','SALES-POS','PAYMENT','INTER BANK','PHONE BANK'])]
#df_emp.head(10)
# customer dataframe

df_cus = df[~df['txn_description'].isin(['PAY/SALARY'])]
df_cus.head()
# customers with highest transaction count 

top_customers = df_cus['first_name'].value_counts(sort=True).nlargest(20)
top_customers
# visualize top customers

fig,ax = plt.subplots(figsize=(20,6)) # (height,width)
top_customers.plot.barh(color=my_colors)
ax.set(xlabel="Number of transactions",
      ylabel="Name")
plt.title('Top customers',bbox={'facecolor':'0.9', 'pad':5})
michael_tran_each_state = mer_state_grp['first_name'].apply(lambda x: x.str.contains('Michael').sum())
# visualize transaction count by Michael in each state

fig,ax = plt.subplots(figsize=(20,6))
print(michael_tran_each_state);
michael_tran_each_state.plot.barh(color=my_colors)
ax.set(xlabel="Number of transaction",
      ylabel="Merchant State")
plt.title('Transaction count by an individual customer in each state',bbox={'facecolor':'0.9', 'pad':5})
# visualize number of transaction by card

plt.figure(figsize=(10,5))
print(df['card_present_flag'].value_counts())
ax = sns.countplot(x='card_present_flag' , data=df)
total = float(len(df['card_present_flag']))
plt.xlabel("Card or No-card")
plt.ylabel("Count")
plt.title('Transaction count using physical card \n'+'0.0-No 1.0-Yes',bbox={'facecolor':'0.9', 'pad':5} )
plt.show()
# visualize transaction by card 

plt.figure(figsize=(10,7))
df['card_present_flag'].value_counts(normalize=True).plot.pie(autopct='%.2f',labels=['Card',
                                                                         'Non-card'], labeldistance=0.5 ,
                                                   shadow=True, startangle=140,pctdistance=0.2 , radius=1)
plt.title('Percentage of card payment' , bbox={'facecolor':'0.8', 'pad':5})
# visualize transaction status

plt.figure(figsize=(10,7))
df['status'].value_counts(normalize=True).plot.pie(autopct='%.2f',labels=['authorized',
                                                                         'posted'], labeldistance=0.5 ,
                                                   shadow=True, startangle=140,pctdistance=0.2 , radius=1)
plt.title('Percentage of transaction status' , bbox={'facecolor':'0.8', 'pad':5})