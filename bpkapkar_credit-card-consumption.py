import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv("../input/credit-card-consumption-prediction/train and test data.csv")

abbr=pd.read_excel("../input/credit-card-consumption-prediction/Data_Dictionary.xlsx")
df.shape
df.head()
abbr.head(44)
df.drop("id", axis=1, inplace=True)
df.info()
plt.figure(figsize = (16,5),dpi = 100)

sns.heatmap(df.isnull(),yticklabels=False);
df.fillna(0, inplace=True)
plt.figure(dpi = 100)

sns.heatmap(df.isnull(),yticklabels=False);
df.info()
df.describe()
import numpy as np



corr = df.corr()

mask = np.triu(np.ones_like(corr,dtype = bool))



plt.figure(figsize = (12,12),dpi = 80)

plt.title('Correlation Analysis')

sns.heatmap(df.corr(),mask=mask,annot=False,cmap='viridis')

plt.xticks(rotation=90)

plt.yticks(rotation = 0)

plt.show()
sns.countplot(x="gender", data=df,hue="account_type");

df.groupby(["gender" , "account_type"]).count()["age"]
sns.barplot(x="gender" , y="cc_cons",hue="account_type", data=df)
plt.figure(figsize = (16,8))

sns.boxplot(x="gender", y="cc_cons", data=df,palette='rainbow')
sns.catplot(x="gender" , y="cc_cons",hue="account_type", data=df,jitter = 0.3)
plt.figure(figsize = (20,8))

sns.countplot(x="region_code", data=df)

print( "Total nos of region_code are ", df["region_code"].nunique())
plt.figure(figsize = (8,8))

df["region_code"].value_counts().nlargest(20)

sns.barplot(x=df["region_code"].value_counts().nlargest(20).index,y=df["region_code"].value_counts().nlargest(20))
sns.distplot(df["age"],bins=30);
sns.boxplot(df["age"])
#Outlier with more than 50 age

df[df["age"]>75]
# Credit card limit vs Credit consumption 

df.plot(kind='scatter', x='cc_cons', y='card_lim',) 
df.plot(kind='scatter', x='age', y='cc_cons')
df['creditcard_exp']= df['cc_cons_apr'] + df['cc_cons_may'] + df['cc_cons_jun'] 

df['debitcard_exp']= df['dc_cons_apr'] + df['dc_cons_may'] + df['dc_cons_jun']
print("Total credit card cosumption for month of april is", df["cc_cons_apr"].mean())     

print("Total credit card cosumption for month of May is",df['cc_cons_may'].mean())

print("Total credit card cosumption for month of june is",df["cc_cons_jun"].mean())

sns.barplot(x=["april", "may", "june"],y=[df["cc_cons_jun"].mean(),df['cc_cons_may'].mean(),df["cc_cons_jun"].mean()])
print("Total Debit card cosumption for month of april is",df["dc_cons_jun"].mean())

print("Total Debit card cosumption for month of May is",df['dc_cons_may'].mean())

print("Total Debit card cosumption for month of june is",df["dc_cons_jun"].mean())

sns.barplot(x=["april", "may", "june"],y=[df["dc_cons_jun"].mean(),df['dc_cons_may'].mean(),df["dc_cons_jun"].mean()])

print("Total Debit card count for month of april is",df["cc_count_apr"].mean())

print("Total Debit card count for month of May is",df['cc_count_may'].mean())

print("Total Debit card count for month of june is",df["cc_count_jun"].mean())

sns.barplot(x=["april", "may", "june"],y=[df["cc_count_apr"].mean(),df['cc_count_may'].mean(),df["cc_count_jun"].mean()])
sns.scatterplot(df['emi_active'],df['cc_cons'])
#Total Investment 

df["Total_investement"]= df["investment_1"] + df['investment_2'] + df['investment_3']+ df['investment_4']
#Total Debit Amount

df["Total_debit_amount"]=df['debit_amount_apr']+df['debit_amount_may']+df['debit_amount_jun']
#Total Credit Amount

df["Total_credit_amount"]=df['credit_amount_apr']+df['credit_amount_may']+df['credit_amount_jun']
#Total Max Credit amount

df["Total_max_credit_acmout"]=df['max_credit_amount_apr']+df['max_credit_amount_may']+df['max_credit_amount_jun']
#Total Active Loan

df["Totat_active_loan"]= df['personal_loan_active'] + df['vehicle_loan_active']
#Toatal Closed Loan

df["Total_closed_loan"]= df['personal_loan_closed'] + df['vehicle_loan_closed']
sns.relplot(x="Totat_active_loan", y="Total_investement", hue='account_type', size='cc_cons',

            sizes=(40, 400), alpha=0.5, palette="muted",

            height=6, data=df)
sns.pairplot(df[['creditcard_exp','card_lim','debitcard_exp','cc_cons']])
newdf = df.drop(columns=['cc_cons_apr','cc_cons_apr','dc_cons_apr', 'cc_cons_may', 'dc_cons_may', 'cc_cons_jun',

       'dc_cons_jun', 'cc_count_apr', 'cc_count_may', 'cc_count_jun','dc_count_apr', 'dc_count_may', 'dc_count_jun',

        'debit_amount_apr', 'credit_amount_apr','debit_count_apr', 'credit_count_apr', 'max_credit_amount_apr',

       'debit_amount_may', 'credit_amount_may', 'credit_count_may','debit_count_may', 'max_credit_amount_may', 'debit_amount_jun',

       'credit_amount_jun', 'credit_count_jun', 'debit_count_jun','max_credit_amount_jun','personal_loan_active', 'vehicle_loan_active',

       'personal_loan_closed','vehicle_loan_closed', 'investment_1', 'investment_2', 'investment_3','investment_4', ])
newdf.head()