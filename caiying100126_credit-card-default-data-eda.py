import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
df = pd.read_excel('E:/2020MSBA/BT5151_Foundation_in_Data_Analytics2/assignment2/default-payment.xlsx')
# Already Removed the 1st row(X1,X2....X23) in Excel
# change two column name to make it more clear
df = df.rename(columns={'default payment next month': 'y', 
                        'PAY_0': 'PAY_1'})

df.head()
df.info()
# Categorical variables description
df[['SEX', 'EDUCATION', 'MARRIAGE']].describe()
# Count the number of gender
df.SEX.value_counts().plot(kind = 'bar',figsize=(5,5),rot=0,colormap='Dark2',title="Gender count")
# Count the number of marriage status
df.MARRIAGE.value_counts().plot(kind = 'bar',figsize=(5,5),rot=0,colormap='RdYlGn',title="Marriage Status")
df.EDUCATION.value_counts().plot(kind = 'bar',figsize=(5,5),rot=0,colormap='seismic',title="Education Status")
df.y.value_counts().plot(kind = 'bar',figsize=(5,5),rot=0,colormap='seismic',title="y Label(Defaul Payment Next Month) distribution")
# age distribution
df.AGE.hist(figsize=(6, 4),color='skyblue',align='right', edgecolor='blue',linewidth=1)
df['LIMIT_BAL'].value_counts().head(5)
plt.figure(figsize = (10,6))
plt.title('Amount of the given credit(Density Plot)')
sns.set_color_codes("pastel")
sns.distplot(df['LIMIT_BAL'],kde=True,bins=200, color="green")
plt.show()
# Payment delay description
df[['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']].describe()
def draw_hist(df, variables, n_rows, n_cols, n_bins):
    fig=plt.figure()
    for i, var_name in enumerate(variables):
        ax=fig.add_subplot(n_rows,n_cols,i+1)
        df[var_name].hist(bins=n_bins,ax=ax, figsize=(6, 4),color='skyblue',align='left', edgecolor='blue',linewidth=1)
        ax.set_title(var_name)
    fig.tight_layout()
    plt.show()
delay = df[['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5','PAY_6']]
draw_hist(delay, delay.columns, 2, 3, 10)
# Bill Statement description
df[['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']].describe()
bills = df[['BILL_AMT1','BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']]
draw_hist(bills, bills.columns, 2, 3, 10)
#Previous Payment Description
df[['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']].describe()
pay = df[['PAY_AMT1','PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']]
draw_hist(pay, pay.columns, 2, 3, 10)
var_delay = ['PAY_1','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']

plt.figure(figsize = (6,6))
plt.title('Amount of payment delay \nPearson correlation plot')
corr = df[var_delay].corr()
sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns,cmap="YlGnBu",annot=True,linewidths=.2,vmin=-1, vmax=1,fmt ='.0%')
plt.show()
var_bill = ['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6']

plt.figure(figsize = (6,6))
plt.title('Amount of bill statement (Apr-Sept) \nPearson correlation plot')
corr = df[var_bill].corr()
sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns,annot=True,cmap="Greens",linewidths=.2,vmin=-1, vmax=1,fmt ='.0%')
plt.show()
var_pay = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5','PAY_AMT6']

plt.figure(figsize = (6,6))
plt.title('Amount of previous payment  (Apr-Sept) \nPearson correlation plot')
corr = df[var_pay].corr()
sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns,annot=True,cmap='BuPu',linewidths=.2,vmin=-1, vmax=1,fmt ='.0%')
plt.show()
cols = list(df)
sns.set(font_scale=1)

fig, ax = plt.subplots(figsize=(15,15))
sns.heatmap(df.corr(), ax=ax, cmap="BuPu",cbar=True, annot=True, fmt ='.0%', square=True, yticklabels=cols, xticklabels=cols)
plt.show()
