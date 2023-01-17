# importing Modules
import numpy as np # linear algebra
import pandas as pd # data processing

import seaborn as sns # for creating plots
sns.set_style('whitegrid') # sns style
import matplotlib.pyplot as plt # for creating plots
plt.style.use('seaborn-white') # plt style

!pip install chart-studio
import chart_studio.plotly as py
from plotly import __version__

import cufflinks as cf
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
# loading dataset
df = pd.read_csv("../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.info()
df.head(5)
df.tail(5)
# renaming 'tenure' and 'gender'
df = df.rename(columns={'tenure': 'Tenure', 'gender': 'Gender'})

# converting 'TotalCharges' to numerical data type
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce', downcast='float')

# converting 'SeniorCitizen' to object data type
df['SeniorCitizen'] = df['SeniorCitizen'].astype(np.object)

# check
df.info()
# Pie chart of churn
churn_rate = df['Churn'].value_counts() / len(df['Churn'])
labels = 'Non Churn', 'Churn'

fig, ax = plt.subplots()
ax.pie(churn_rate, labels=labels, autopct='%.f%%')  
ax.set_title('Churn vs Non Churn', fontsize=16)
# numeric features grouped by churn
df.hist(bins= 15, figsize=(10,10), label="Churn")

Churn = df[df.Churn == 'Yes']
No_Churn = df[df.Churn == 'No']
Churn.hist(bins= 15, figsize=(10,10), label="Churn", color='red', alpha=0.5)
No_Churn.hist(bins= 15, figsize=(10,10), label="Churn", color='blue', alpha=0.5)
plt.show() 
def sephist(col):
    Churn = df[df['Churn'] == 'Yes'][col]
    No_Churn = df[df['Churn'] == 'No'][col]
    return Churn, No_Churn

for num, alpha in enumerate('Tenure'):
    #plt.subplot(2, 2, num)
    plt.hist(sephist(alpha)[0], bins=25, alpha=0.5, label='Churn', color='b')
    plt.hist(sephist(alpha)[1], bins=25, alpha=0.5, label='No_Churn', color='r')
    plt.legend(loc='upper right')
    plt.title(alpha)
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

plt.figure(figsize=(15,8))
telco_df_dummies.corr()['Churn'].sort_values(ascending = False).plot(kind='bar')
# converting all categorical features into dummy variables
telco_df_dummies = pd.get_dummies(telco_df, prefix=None)
telco_df_dummies.head()

# removing 'customerID' from the data set
telco_df = telco_df.iloc[:,1:]

# converting the target variable Churn to binary numeric data type
telco_df['Churn'] = telco_df['Churn'].map({'Yes': 1, 'No': 0})

# check 
telco_df.head()
# summarize missing values
df2.isnull().sum()
# fill missing value w/ mean
df2['TotalCharges'].fillna(value=df2['TotalCharges'].mean(), inplace=True)
# Checking missing values
df2['TotalCharges'].isnull().sum()
# personal note:
churn_rate = telco_df['Churn'].value_counts() / len(telco_df['Churn'])
churn_rate
# personal code:
# distplot: Tenure grouped by Churn
plt.figure(figsize=(10, 6))

df1 = telco_df[telco_df['Churn'] == 0]
ax = sns.distplot(df1['Tenure']/12, hist=False, kde=True, kde_kws={"shade": True}, label='Non Churn')

df0 = telco_df[telco_df['Churn'] == 1]
ax = sns.distplot(df0['Tenure']/12, hist=False, kde=True, kde_kws={"shade": True}, label='Churn')

# plot formatting
ax.legend()
ax.set_ylabel('Density')
ax.set_xlabel('Tenure(Year)')
ax.set_title('Distribution of Tenure by Churn')

# distplot: Monthly Charges grouped by Churn
plt.figure(figsize=(10, 6))

df1 = telco_df[telco_df['Churn'] == 0]
ax = sns.distplot(df1['MonthlyCharges'], hist = False, kde = True, kde_kws={"shade": True}, label='Non Churn')

df0 = telco_df[telco_df['Churn'] == 1]
ax = sns.distplot(df0['MonthlyCharges'], hist = False, kde = True, kde_kws={"shade": True}, label='Churn')

# plot formatting
ax.legend()
ax.set_ylabel('Density')
ax.set_xlabel('Monthly Charges')
ax.set_title('Distribution of Monthly Charges by Churn')

# distplot: Total Charges grouped by Churn
plt.figure(figsize=(10, 6))

df1 = telco_df[telco_df['Churn'] == 0]
ax = sns.distplot(df1['TotalCharges'], hist = False, kde = True, kde_kws={"shade": True}, label='Non Churn')

df0 = telco_df[telco_df['Churn'] == 1]
ax = sns.distplot(df0['TotalCharges'], hist = False, kde = True, kde_kws={"shade": True}, label='Churn')

# Plot formatting
ax.legend()
ax.set_ylabel('Density')
ax.set_xlabel('Total Charges')
ax.set_title('Distribution of Total Charges by Churn')