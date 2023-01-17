#i) Process data
#ii)  Important Step==> Engineer some features
#iii)  Important step=>  Use graphs to develop relationships between various 
#      variables and the target.
#iv)  Maybe after step (iii) you find that some more features can be created
#v)   Develop prediction model using one of the Decision tree algorithms
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
#Checking the availble files in the working Directory
os.listdir()
#Creating a dataset "data" for reading the .csv file
data = pd.read_csv("../input/UCI_Credit_Card.csv")

data.shape  # We have 30,000 records with 25 columns 
#Reading the column names 

data.columns
#or 
data.dtypes
#or 
data.info()
#Converting all the columns to small case for ease of coding 

data.columns = map(str.lower, data.columns)

data.dtypes
# View the data
data.head(10)

# Max & Min of Limit_bal
# Limits given to Sex category 1 & 2
# Limits given to educations category 1, 2, 3
# Limits given to marriage 
# Add a column for age class young/middle/old 
data.head()
#There are 25 variables:

#ID: ID of each client
#LIMIT_BAL: Amount of given credit in NT dollars (includes individual and family/supplementary credit
#SEX: Gender (1=male, 2=female)
#EDUCATION: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
#MARRIAGE: Marital status (1=married, 2=single, 3=others)
#AGE: Age in years
#PAY_0: Repayment status in September, 2005 (-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, ... 8=payment delay for eight months, 9=payment delay for nine months and above)
#PAY_2: Repayment status in August, 2005 (scale same as above)
#PAY_3: Repayment status in July, 2005 (scale same as above)
#PAY_4: Repayment status in June, 2005 (scale same as above)
#PAY_5: Repayment status in May, 2005 (scale same as above)
#PAY_6: Repayment status in April, 2005 (scale same as above)
#BILL_AMT1: Amount of bill statement in September, 2005 (NT dollar)
#BILL_AMT2: Amount of bill statement in August, 2005 (NT dollar)
#BILL_AMT3: Amount of bill statement in July, 2005 (NT dollar)
#BILL_AMT4: Amount of bill statement in June, 2005 (NT dollar)
#BILL_AMT5: Amount of bill statement in May, 2005 (NT dollar)
#BILL_AMT6: Amount of bill statement in April, 2005 (NT dollar)
#PAY_AMT1: Amount of previous payment in September, 2005 (NT dollar)
#PAY_AMT2: Amount of previous payment in August, 2005 (NT dollar)
#PAY_AMT3: Amount of previous payment in July, 2005 (NT dollar)
#PAY_AMT4: Amount of previous payment in June, 2005 (NT dollar)
#PAY_AMT5: Amount of previous payment in May, 2005 (NT dollar)
#PAY_AMT6: Amount of previous payment in April, 2005 (NT dollar)
#default.payment.next.month: Default payment (1=yes, 0=no)
#Feature 1: Categorise on the basis of limit balance : bronze , silver , gold 
data.head()
data.limit_bal.min()

data.limit_bal.max()
data.limit_bal.mean()
#data['cust_cat'] = pd.cut(data.limit_bal,[10000, 200000, 500000, 100000], labels = ["bronze","silver","gold"])
#Feature 2: Categorise on the basis of age : young, middle, old 

data['age_group'] = pd.cut(data.age, [21,40,60,81], labels=["young","middle","senior"])
data.head()

#Lets random check if categories are in place 

data[data['age_group'] == 'senior'].head(10)
#Feature 3: Categorise Risk on the basis of previous payment defaults.
#lets add a column pay_total. This have total number of payments done or missed. 

data['pay_total'] = (data.pay_0 + data.pay_2 + data.pay_3 + data.pay_4 + data.pay_5 + data.pay_6)

data.pay_total.head(10)
#Observations 

#1. Id's with score above -2 shows that the customers have always paid on time.
#2. Id's with positive values shows the number of payment default month by customer. 
#3. Customers have duly paid over a year are good customers.
#Categorise Risk on the basis of previous payment defaults.
data['risk_cat'] = pd.cut(data.pay_total, [-20,-10,0,10], labels=["low","medium","high"])

#Feature 4: Categories on the basis of usage : high, medium, low. 
data[['pay_total','risk_cat']].head(20)
import seaborn as sns

sns.jointplot("age", "pay_total", data, kind = 'resid')
#data.age_group.value_count()
#We can clearly see that payment default are being done more by age group of 21 to 55 yrs 
#beyond the age of 55yrs patment default falls drstically
#It is the highest withing the age group of 21 to 40 yrs
# Going beyond 30-36 months 
data.pay_total.max()
data.pay_total.min()
sns.jointplot('education', 'pay_total', data, kind = 'regid')
#How many males have defaulted the payment recently
data[data['sex'] == 1].groupby(data['pay_total']).count().plot(figsize = (10,10))
plt.legend(bbox_to_anchor=(1, 1), loc=2)
data.dtypes
#Male & Female default ratio on the basis of pay_total & age
sns.lmplot(x='age', y='pay_total', data=data, hue ='sex')


#EDUCATION: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
#Risk category on the basis of their education
plt.figure(figsize=(15,8))
sns.boxplot(x= 'education', y = 'pay_total',data=data, hue = 'risk_cat')
plt.legend(bbox_to_anchor=(1, 1), loc=2)
#Relationship between age group & Risk category they belong to
#AGE: Age in years
#age_group :  Young, Middle & Senior
plt.figure(figsize=(15,8))
sns.set_style('whitegrid')
sns.violinplot('age_group', 'pay_total', data = data, hue = 'risk_cat')

#Check for Nill Values
#data.isnull().sum() #There are no columns with null 
data.head()
#MARRIAGE: Marital status (1=married, 2=single, 3=others)
#age_group :  Young, Middle & Senior
#risk_cat : High, Medium & Low
#EDUCATION: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)

n_df = data
n_df.head()

n_df = n_df[['sex','marriage', 'education', 'age_group', 'pay_total', 'risk_cat']].copy()
n_df.head()
#Correlation between education, marriage, sex & risk category 
corr = n_df.corr()
plt.figure(figsize=(15,8))
sns.heatmap(corr, annot=True, annot_kws={"size": 15})

#Comparison of Defaulters male/female 
plt.figure(figsize = (15,10))
sns.distplot(n_df.sex)
#Division of people on the basis of their risk category
plt.figure(figsize = (15,10))
sns.countplot(x='risk_cat', data=n_df)
#More than 20k people are in Medium Risk Category
#Around 6000 people are in High Risk Category 
#Only 2500 people are in Low Risk Category 
#Total number of people 28544
n_df.risk_cat.count()
#Factor plots for categorical classes age_group & risk 
#Time taken = Takes more than an hour !!!!

#g = sns.factorplot(x='age_group', y = 'pay_total', data = data,hue = 'sex', col = 'sex', kind = 'swarm')
#g.set_xticklabels(rotation=-45)
#Density plot for understanding risk category as per sex 
#Risk of giving credit to females is higher.
plt.figure(figsize = (15,10))
sns.kdeplot(n_df.sex, n_df.pay_total)
#Joint Distribution Plot distribution of limit balance with education
#EDUCATION: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
sns.jointplot(x='education',y='limit_bal', data = data)
#We can clearly See above that high limits have been given to education category 1, 2 & 3 
sns.pairplot(n_df, kind = 'reg')
plt.show()
sns.pairplot(n_df, kind ='scatter')
