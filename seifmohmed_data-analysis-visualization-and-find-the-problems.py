# import the important libraries

import pandas as pd     # for dataframe

import numpy as np      # for arraies

import matplotlib.pyplot as plt  # for visualization 

%matplotlib inline

import seaborn as sns           # for visualization 
telco = pd.read_csv("../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")
telco.head()
telco.info()
telco['TotalCharges'] = telco['TotalCharges'].replace(" ",np.nan)
telco.isna().sum() # number of missing values for each columns
telco.dropna(axis=0,inplace=True)
#Converting type of TotalCharges to float

telco["TotalCharges"] = telco["TotalCharges"].astype(float)

type(telco["TotalCharges"])
telco.describe() # to know some statistical information about the numerical data
telco.nunique()
telco.PaymentMethod.unique()
telco.MultipleLines.unique()
telco.head()
sns.countplot(telco.Churn)

plt.title("Number of people who left and who's still in company")

plt.show()
telco.Churn.value_counts()
sns.countplot(telco.gender,hue=telco.Churn)

plt.title('the gender of customers who left nd who not')

plt.show()
female_left = telco.query("gender == 'Female' and Churn == 'Yes'")
female_left.shape
male_left = telco.query("gender == 'Male' and Churn == 'Yes'")
male_left.shape
print('the percentage of female who left is {}%'.format((939/1869)*100))
print('the percentage of male who left is {}%'.format((930/1869)*100))
sns.countplot(telco.SeniorCitizen)

plt.title("Number of people who are senior citizen")

plt.show()
telco.SeniorCitizen.value_counts()
sns.countplot(telco.SeniorCitizen,hue=telco.Churn)

plt.title('senior citizen who left and who not')

plt.show()
senior_left =telco.query("SeniorCitizen == 1 and Churn == 'Yes'")
senior_left.shape
print('the percentage of senior citizen who left is {}%'.format((476/1142)*100))
sns.countplot(telco.Partner)

plt.title("Number of customer who have partener")

plt.show()
telco.Partner.value_counts()
sns.countplot(telco.Partner,hue=telco.Churn)

plt.show()
partner_left = telco.query("Partner == 'Yes' and Churn == 'Yes'")
partner_left.shape
telco.query("Partner == 'No' and Churn == 'Yes'").shape
print("there are {}% of customers who have partner left the company - and there are {}% of customers who din't have partner left"

     .format((669/3402)*100,(1200/3641)*100))
telco.Dependents.value_counts()
sns.countplot(telco.Dependents)

plt.title('Number of customers who are dependents VS independents')

plt.show()
telco.query("Dependents == 'Yes' and Churn == 'Yes'").shape
telco.query("Dependents == 'No' and Churn == 'Yes'").shape
print("there are {}% of customers who are Dependents left the company - and there are {}% of customers who are independents left"

     .format((326/2110)*100,(1543/4933)*100))
telco.head()
plt.figure(figsize=(20,5))

sns.countplot(telco.tenure,order=(telco.tenure.value_counts().index))

plt.title('Number of months the customer has stayed with the company')

plt.show()
print('there are {}% of customers left the company after the first month'.format((telco.query("tenure ==1").shape[0]/telco.shape[0])*100))
print('in general {}% of customers left the company after first 5 months'

      .format((telco.query("tenure ==[1,2,3,4,5]").shape[0]/telco.shape[0])*100))
num_col = telco[['tenure','MonthlyCharges','TotalCharges']]
num_col.head()
telco.head()
telco.shape
df=telco.drop('customerID',axis=1)

df.head()
def pie(x):

    sorted_counts = df[x].value_counts()

    plt.pie(sorted_counts, labels = sorted_counts.index, startangle = 90,

            counterclock = False,autopct='%.1f%%',shadow=True);

    plt.axis('square')

    plt.show()
plt.figure(figsize=(5,5))

plt.title('The percentage of each gender in the company')

pie('gender')

plt.show()
plt.figure(figsize=(5,5))

plt.title('The percentage of variation in ages in the company')

pie('SeniorCitizen')

plt.show()
plt.figure(figsize=(5,5))

plt.title('Percentage of having a partner VS those who do not have')

pie('Partner')

plt.show()
plt.figure(figsize=(5,5))

plt.title('The percentage of Dependent clients in the company VS Independent clients')

pie('Dependents')

plt.show()
num_col.head()
plt.figure(figsize=(5,5))

plt.hist(num_col.tenure)

plt.title('tenure destribution')

plt.xlabel('tenure ')

plt.ylabel('count')

plt.show()
plt.figure(figsize=(5,5))

plt.hist(np.log10(num_col.tenure)) # here i count the logarithm 10 for the tenure feature

plt.title('tenure destribution after count logarithm ')

plt.xlabel('tenure')

plt.ylabel('Clients number')

plt.show()
plt.figure(figsize=(20,6))

plt.subplot(1,2,1)

plt.hist(num_col.TotalCharges)

plt.ylim(0,3000)

plt.title('TotalCharges destribution')

plt.xlabel('TotalCharges')

plt.ylabel('Clients number')



plt.subplot(1,2,2)

plt.hist(num_col.MonthlyCharges)

plt.ylim(0,3000)

plt.title('MonthlyCharges destribution')

plt.xlabel('MonthlyCharges')

plt.show()
plt.figure(figsize=(12,6))





for i in range(len(list(num_col.columns))):

    plt.subplot(1,3,i+1)

    sns.boxplot(num_col.iloc[:,i])
service = telco.iloc[:,6:-4]
service.head()
plt.figure(figsize=(25,20))

for i,feature in enumerate(list(service.columns)):

    plt.subplot(3,4,i+1)

    sns.countplot(feature,hue = telco.Churn,data=telco)

    plt.ylim(0,5000)

    plt.title('num of clints who used {} left vs stay'.format(feature))

plt.show()
plt.figure(figsize=(25,20))

for i,feature in enumerate(list(service.columns)):

    plt.subplot(3,4,i+1)

    sns.countplot(feature,hue = telco.SeniorCitizen,data=telco)

    plt.ylim(0,6000)

    plt.title('senior citizen VS youth in using {}'.format(feature))

plt.show()
plt.figure(figsize=(25,20))

for i,feature in enumerate(list(service.columns)):

    plt.subplot(3,4,i+1)

    sns.countplot(feature,hue = telco.Dependents,data=telco)

    plt.ylim(0,5000)

    plt.title('Dependents VS Independent in using {}'.format(feature))

plt.show()
plt.figure(figsize=(25,20))

for i,feature in enumerate(list(service.columns)):

    plt.subplot(3,4,i+1)

    sns.countplot(feature,hue = telco.Partner,data=telco)

    plt.ylim(0,5000)

    plt.title('clints who have partner VS not in using {}'.format(feature))

plt.show()
sns.pairplot(num_col)

plt.show()
sns.heatmap(num_col.corr(),annot=True,linewidths=2)

plt.title('correlation between the numerical variables in the dataset')

plt.show()