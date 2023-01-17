# importing the needed libraries 

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('../input/anz-synthesised-transaction-dataset/anz.csv')

df.head()
# Checking the shape of the dataframe

df.shape
# Getting the info of the dataframe

df.info()
# Checking how many missing values are there

df.isna().sum()

# Converting the date column to pandas Timestap since this is an Time Series data 

df['date'] = pd.to_datetime(df['date'])
# Checking 

type(df['date'][0])
df['date'].head(7)
# By using the date we acquired converting them to days of that particular date

df['day_name'] = df['date'].dt.day_name()

df['day_name'].head()
# Creating Month by using the date gives so can be useful for EDA 

df['month_name'] = df['date'].dt.month_name()

df['month_name'].head()
# Checking out available months generated from the date give

df['month_name'].value_counts()
# Plotting the correlation heatmap 

cor_mat = df[['card_present_flag' , 'amount' , 'balance' ,'date' , 'status', 

             'bpay_biller_code' , 'account' , 'txn_description',

             'gender' , 'age' , 'extraction']].corr()

# Custom cmap pallete

cmap = sns.diverging_palette(220 , 10 , as_cmap=True)



# Building heatmap

sns.heatmap(cor_mat ,vmax=.3 , center=0 , cmap=cmap , square=True , linewidths=.5 , cbar_kws={'shrink': .5})
# Correlation matrix in Tabular form

cor_mat
# Checking amount transacted in October month

filt = (df['month_name'] == 'October')

df.loc[filt , 'amount']
# Average amount in october month

df.loc[filt , 'amount'].mean()
# Maximum Value transacted in October month 

df.loc[filt , 'amount'].max()
# Minimum Value transacted in October month 

df.loc[filt , 'amount'].min()
# Checking amount transacted in September month

filt = (df['month_name'] == 'September')

df.loc[filt , 'amount']
# Average amount in september month

df.loc[filt , 'amount'].mean()
# Maximum amount in september month

df.loc[filt , 'amount'].max()
# Minimum Value transacted in september month 

df.loc[filt , 'amount'].min()
# Checking amount transacted in August month

filt = (df['month_name'] == 'August')

df.loc[filt , 'amount']
# Average amount in august month

df.loc[filt , 'amount'].mean()
# Maximum amount in september month

df.loc[filt , 'amount'].max()
# Minimum amount in september month

df.loc[filt , 'amount'].min()
print(df['gender'].value_counts())

plt.figure(figsize=(8,6))

sns.set(style="darkgrid")

sns.countplot(df['gender'])

plt.show()
# Month where highest number of transaction took place

sns.countplot(x='month_name' , data=df)
# Month where highest number of transaction took place based on gender

plt.figure(figsize=(10,8))

sns.countplot(x='month_name' ,hue='gender', data=df)

plt.title('Month where highest number of\n'+'transaction took place based on gender',bbox={'facecolor':'0.9', 'pad':5})
plt.figure(figsize=(10,7))

sns.countplot(x='day_name' , data=df)
plt.figure(figsize=(10,7))

ax = sns.countplot(x="day_name", hue="gender", data=df) # for Seaborn version 0.7 and more

total = float(len(df))

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:.2%}'.format(height/total),

            ha="center") 



plt.title('Number of transaction made on each day of\n'+'a week with gender comparison',bbox={'facecolor':'0.9', 'pad':5})

plt.show()
plt.figure(figsize=(10,7))

print(df['card_present_flag'].value_counts())

ax = sns.countplot(x='card_present_flag' , data=df)

total = float(len(df['card_present_flag']))

plt.title('Number of customers made transaction\n'+'through a physical card while making purchase\n'+'1.0-Yes 0.0-No',bbox={'facecolor':'0.9', 'pad':5} )

plt.show()
print(df['merchant_state'].value_counts())

plt.figure(figsize=(10,7))

sns.countplot(df['merchant_state'])

plt.title('Number of transaction\n' 'done on each state',bbox={'facecolor':'0.9', 'pad':5})

plt.show()

print(df['txn_description'].value_counts())

sns.set(style="darkgrid")

plt.figure(figsize=(10,7))

ax = sns.countplot(df['txn_description'])

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:.2%}'.format(height/total),

            ha="center") 

plt.title('Percentage of Source by where transaction took place')

plt.ylabel('Number of Transaction')

plt.xlabel('Transaction Description')

plt.show()

# Distribution of Age of the customers.

plt.figure(figsize=(10,7))

sns.distplot(df['age']);

plt.title('Distribution of customers based on age group' , )
# Figuring out which age group has more balance.

plt.figure(figsize=(10,7))

sns.lineplot(x='age' , y='balance' , data=df)
# Figuring out which age group has transacted more

plt.figure(figsize=(10,7))

sns.lineplot(x='age' , y='amount' , data=df)
# Checking the mean for numerical data in dataframe

df.mean()
# making a group with merchant_state dataframe

mer_state_grp = df.groupby(['merchant_state'])
# Number of Male and Female made transaction in the particular merchant state's

print(mer_state_grp['gender'].value_counts(normalize=True))

gen_mer_state = mer_state_grp['gender'].value_counts()

fig,ax = plt.subplots(figsize=(10,10)) # (height,width)

gen_mer_state.plot.barh()

ax.set(xlabel="Number of transaction made",

      ylabel="State and Gender")

plt.title('Number of Male and Female\n'+'made transaction in particular state',bbox={'facecolor':'0.9', 'pad':5})





# Number of debit and credit transaction

plt.figure(figsize=(10,7))

print(df['movement'].value_counts())

sns.countplot(df['movement'])
# Which gender made most debit and credit transaction 

plt.figure(figsize=(10,7))

ax = sns.countplot(df['movement'] , hue=df['gender'])

total = float(len(df))

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:.2%}'.format(height/total),

            ha="center") 

plt.title('Percentage of Male and Female who made\n'+'Debit and Credit Transaction',bbox={'facecolor':'0.8', 'pad':5})
# Percentage of contribution of months

pie_color = ['orange' , 'salmon', 'lightblue']

fig,ax = plt.subplots(figsize=(7,8)) # (height,width)



df['month_name'].value_counts(sort=True).plot.pie(labeldistance=0.2 ,

                                         colors=pie_color,

                                        autopct='%.2f', shadow=True, startangle=140,pctdistance=0.8 , radius=1)

plt.title("Percentage of contribution\n" + "of months", bbox={'facecolor':'0.8', 'pad':5})



# Percentage of contribution of gender 

plt.figure(figsize=(10,7))

df['gender'].value_counts(normalize=True).plot.pie(autopct='%.2f',labels=['Male',

                                                                         'Female'], labeldistance=0.5 ,

                                                   shadow=True, startangle=140,pctdistance=0.2 , radius=1)

plt.title('Percentage of contribution\n'+'of Male and Female' , bbox={'facecolor':'0.8', 'pad':5})

# Top 10 customers 

top_cust = df['first_name'].value_counts(sort=True).nlargest(10)

top_cust
fig,ax = plt.subplots(figsize=(10,10)) # (height,width)

top_cust.plot.barh(color=my_colors)

ax.set(title="Top 10 Customer",

      xlabel="Number of transaction made",

      ylabel="Name")

tail_cust = df['first_name'].value_counts(sort=True).nsmallest(10)

tail_cust



#Colors for the bar of the graph

my_colors = ['r','b','k','y','m','c','#16A085','salmon' , '#32e0c4']
fig,ax = plt.subplots(figsize=(10,10)) # (height,width)

tail_cust.plot.barh(color=my_colors)

ax.set(title="Least 10 Customer",

      xlabel="Number of transaction made",

      ylabel="Name")

gender_grp = df.groupby(['gender'])
# Average transaction amount made by Male and Female 

gen_trans_amt = gender_grp['amount'].mean()

gen_trans_amt
fig,ax = plt.subplots(figsize=(10,8)) # (height,width)

gen_trans_amt.plot.barh(color=my_colors)

ax.set(title="Average amount transacted by Male and Female",

      xlabel="Average amount",

      ylabel="Gender")

agg_amt_state = mer_state_grp['amount'].agg(['min' , 'mean' , 'max'])
agg_amt_state.columns
agg_amt_state
# Minimum ammount transacted in each state

fig,ax = plt.subplots(figsize=(10,8)) # (height,width)

print(agg_amt_state['min'])

agg_amt_state['min'].plot.barh(color=my_colors)

ax.set(title="Minimum amount transacted in each state",

      xlabel="Amount",

      ylabel="Merchant State")
# Maximum amount transacted in each state

fig,ax = plt.subplots(figsize=(10,8)) # (height,width)

print(agg_amt_state['max'])

agg_amt_state['max'].plot.barh(color=my_colors)

ax.set(title="Maximum amount transacted in each state",

      xlabel="Amount",

      ylabel="Merchant State")
trans_desc_grp = df.groupby(['txn_description'])
df['txn_description'].unique()
trans_desc_grp['first_name'].value_counts().loc['SALES-POS'].nlargest(10)
# Printing out Top 5 Customer 

top_cust[:5]
michael_tran_each_state = mer_state_grp['first_name'].apply(lambda x: x.str.contains('Michael').sum())

diana_tran_each_state = mer_state_grp['first_name'].apply(lambda x: x.str.contains('Diana').sum())

jess_tran_each_state = mer_state_grp['first_name'].apply(lambda x: x.str.contains('Jessica').sum())

jose_tran_each_state = mer_state_grp['first_name'].apply(lambda x: x.str.contains('Joseph').sum())

jeff_tran_each_state = mer_state_grp['first_name'].apply(lambda x: x.str.contains('Jeffrey').sum())
fig,ax = plt.subplots(figsize=(10,8))

print(michael_tran_each_state);

michael_tran_each_state.plot.barh(color=my_colors)

ax.set(

    title='Number of transaction made by Michael in each state',

    xlabel='Number of transaction',

    ylabel='Merchant State'

)
fig,ax = plt.subplots(figsize=(10,8))

print(diana_tran_each_state);

diana_tran_each_state.plot.barh(color=my_colors)

ax.set(

    title='Number of transaction made by Diana in each state',

    xlabel='Number of transaction',

    ylabel='Merchant State'

)
fig,ax = plt.subplots(figsize=(10,8))

print(jess_tran_each_state);

jess_tran_each_state.plot.barh(color=my_colors)

ax.set(

    title='Number of transaction made by Jessica in each state',

    xlabel='Number of transaction',

    ylabel='Merchant State'

)
fig,ax = plt.subplots(figsize=(10,8))

print(jose_tran_each_state);

jose_tran_each_state.plot.barh(color=my_colors)

ax.set(

    title='Number of transaction made by Joseph in each state',

    xlabel='Number of transaction',

    ylabel='Merchant State'

)
fig,ax = plt.subplots(figsize=(10,8))

print(jeff_tran_each_state);

jeff_tran_each_state.plot.barh(color=my_colors)

ax.set(

    title='Number of transaction made by Jeffrey in each state',

    xlabel='Number of transaction',

    ylabel='Merchant State'

)
month_grp = df.groupby(['month_name'])
avg_amt_tran_month = month_grp['amount'].mean()

oct_amt_tran_month = month_grp['amount'].value_counts().loc['October']
fig,ax = plt.subplots(figsize=(10,8)) # (height,width)

print(avg_amt_tran_month);

avg_amt_tran_month.plot.barh(color=my_colors)

ax.set(

    title='Average transaction made my customer on average each month',

    xlabel='Average amount',

    ylabel='Month Name '

)
oct_amt_tran_month = month_grp['amount'].value_counts().loc['October']

oct_amt_tran_month
oct_date = month_grp['date'].value_counts().loc['October']
day_name_grp = df.groupby(['day_name'])
day_name_grp['amount'].mean().loc['Monday']
day_name_grp['amount'].mean().loc['Tuesday']
day_name_grp['amount'].mean().loc['Wednesday']
day_name_grp['amount'].mean().loc['Thursday']
day_name_grp['amount'].mean().loc['Friday']
day_name_grp['amount'].mean().loc['Saturday']
day_name_grp['amount'].mean().loc['Sunday']