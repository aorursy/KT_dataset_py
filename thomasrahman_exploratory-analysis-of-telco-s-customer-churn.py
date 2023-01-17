import numpy as np

import pandas as pd

import math

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder
df = pd.read_csv("../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.shape
df.size
df.columns
df.head(3)
df.sample(5)
df.info()
df.describe(include='all') # Description of all fields without differentiating their dtypes.
df.describe() #Without the 'include' condition, this function will return only the numeric fields.
SeniorCitizens = df.SeniorCitizen[df.SeniorCitizen == 1].count()

print(f'The amount of Senior Citizens customers is {SeniorCitizens}')

SeniorCitizens_Ratio = round(SeniorCitizens / df.SeniorCitizen.count(),2)

print(f'The ratio of Senior Citizens is {SeniorCitizens_Ratio}' )
SeniorCitizens = df.SeniorCitizen[df.SeniorCitizen == 1].count()

NoSeniorCitizens = df.SeniorCitizen[df.SeniorCitizen == 0].count()



x_variables = ['Senior Citiziens', 'No Senior Citizens']

y_variables = [SeniorCitizens, NoSeniorCitizens]

barChar = plt.bar(x_variables,y_variables, color=['Lightblue', 'Lightyellow'])



#Set descriptions:

plt.ylabel('Amount of Customers')

plt.title('Senior Citizens Customers')

#Display values:

plt.text(-0.06, 200, SeniorCitizens, fontsize=10, color= 'White')

plt.text(0.93, 200, NoSeniorCitizens, fontsize=10, color= 'Black')
df_gender = df[['customerID','gender']]

df_gender = df_gender.rename(columns={'customerID':'Amount of Customers'})

gender_count = df_gender.groupby('gender').count()

gender_count['Percentage']=round((gender_count['Amount of Customers']/ df_gender['gender'].count()*100),2)

gender_count
sns.set_palette(['pink', 'lightblue'])

sns.set_context("talk", font_scale=0.8)

plt.figure(figsize=(7,7))

gender_chart = sns.catplot(x="gender",

              hue ="gender",

                 data = df,

             kind="count",

           height=4,

           aspect=1.5).set(title = "Customer Gender")

GenderPartner = df[['customerID','gender','Partner']]

GenderPartner = GenderPartner.rename(columns={'customerID':'Amount of Customers'})

GenderPartner_count = GenderPartner.groupby([GenderPartner.gender, GenderPartner.Partner])[['Amount of Customers']].count()

GenderPartner_count['Percentage']=round((GenderPartner_count['Amount of Customers']/ GenderPartner['gender'].count()*100),2)

GenderPartner_count
ChurnCustomers = df.Churn[df.Churn == "Yes"].count()

print(f'The amount of customers that left the company is {ChurnCustomers}')

ChurnCustomers_Ratio = round(ChurnCustomers / df.Churn.count(),2)

print(f'The churn ratio is {ChurnCustomers_Ratio}' )
x_variables = ['Churn', 'No Churn']

y_variables = [df.Churn[df.Churn == "Yes"].count(), df.Churn[df.Churn == "No"].count()]

barChar = plt.bar(x_variables,y_variables, color=['violet','orange'])



#Set descriptions:

plt.ylabel('Amount of Customers')

plt.title('Churn Customers')

#Display values:

plt.text(-0.06, 200,df.Churn[df.Churn == "Yes"].count(), fontsize=10, color= 'White')

plt.text(0.93, 200, df.Churn[df.Churn == "No"].count(), fontsize=10, color= 'White')
TotalMonthlyCharges = df[['Churn','MonthlyCharges']]

Total = TotalMonthlyCharges.groupby('Churn').sum()

Total['Percentage']=round((Total['MonthlyCharges']/ TotalMonthlyCharges['MonthlyCharges'].sum()*100),2)

Total
x_variables2 = ['Churn', 'No Churn']

y_variables2 = [TotalMonthlyCharges[TotalMonthlyCharges.Churn== 'Yes']['MonthlyCharges'].sum(), TotalMonthlyCharges[TotalMonthlyCharges.Churn== 'No']['MonthlyCharges'].sum()]

barChar2 = plt.bar(x_variables2,y_variables2, color=['lightgreen','lightblue'])



#Set descriptions:

plt.ylabel('Monthly Charges')

plt.title('Monthly Charges Churn')

#Display values:

plt.text(-0.15, 10000,TotalMonthlyCharges[TotalMonthlyCharges.Churn== 'Yes']['MonthlyCharges'].sum(), fontsize=10, color= 'Grey')

plt.text(0.85, 10000, TotalMonthlyCharges[TotalMonthlyCharges.Churn== 'No']['MonthlyCharges'].sum(), fontsize=10, color= 'Grey')
GenderPartner_Churn = df[['customerID','gender','Partner', 'Churn']]

GenderPartner_Churn = GenderPartner_Churn.rename(columns={'customerID':'Amount of Customers'})

GenderPartner_Churn_count = GenderPartner_Churn.groupby([GenderPartner_Churn.gender, GenderPartner_Churn.Partner, GenderPartner_Churn.Churn])[['Amount of Customers']].count()

GenderPartner_Churn_count['Percentage']=round((GenderPartner_Churn_count['Amount of Customers']/ GenderPartner['Amount of Customers'].count()*100),2)

GenderPartner_Churn_count
Partner_Churn1 = df[df.Partner == 'Yes'][['customerID','Partner', 'Churn']]

Partner_Churn1 = Partner_Churn1.rename(columns={'customerID':'Amount of Customers'})

Partner_Churn_count1 = Partner_Churn1.groupby([Partner_Churn1.Partner, Partner_Churn1.Churn])[['Amount of Customers']].count()

Partner_Churn_count1['Percentage']=round((Partner_Churn_count1['Amount of Customers']/ Partner_Churn1['Amount of Customers'].count()*100),2)

Partner_Churn_count1
Partner_Churn2 = df[df.Partner == 'No'][['customerID','Partner', 'Churn']]

Partner_Churn2 = Partner_Churn2.rename(columns={'customerID':'Amount of Customers'})

Partner_Churn_count2 = Partner_Churn2.groupby([Partner_Churn2.Partner, Partner_Churn2.Churn])[['Amount of Customers']].count()

Partner_Churn_count2['Percentage']=round((Partner_Churn_count2['Amount of Customers']/ Partner_Churn2['Amount of Customers'].count()*100),2)

Partner_Churn_count2
no_numerical = (df.dtypes == 'object')

no_numerical_list = list(no_numerical[no_numerical].index)



encdata = df.copy()

enc = LabelEncoder()

columns = df.columns



for col in no_numerical_list:

    encdata[col] = enc.fit_transform(encdata[col])



encdata = pd.DataFrame(encdata, columns=columns)
plt.figure(figsize=(9,9))

sns.heatmap(encdata.corr(), vmin=-1, vmax=1,cmap=sns.diverging_palette(20, 220, n=200))
encadata2 = encdata.corr()

encadata2 = encadata2[['Churn']]

plt.figure(figsize=(3,9))

sns.heatmap(encadata2, annot = True, vmin=-1, vmax=1,cmap=sns.diverging_palette(20, 220, n=200))