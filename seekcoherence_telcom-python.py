# Imported libraries used for analysis

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import ttest_ind
# loads csv file as a pandas dataframe

Telco_data = pd.read_csv('../input/telco-customer-dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Describes dataset columns

Telco_data.info()
# Displays the first 10 rows of the Telco_data dataframe.

Telco_data.head(10)
# Creates distribution plots of numerical features in the Telco_data dataframe.

distributions = Telco_data.select_dtypes([np.int, np.float])

for i, col in enumerate(distributions.columns):

    plt.figure(i)

    sns.distplot(distributions[col])



# A boxplot is made of each numerical feature in the Teclo_data dataframe to identify median and outlier values

Telco_data.boxplot(figsize=(15,10))
# Changes 'TotalCharges' series from object to float format

Telco_data['TotalCharges'] = pd.to_numeric(Telco_data['TotalCharges'], errors='coerce')

Telco_data['TotalCharges'].dtypes
# Searches dataframe for any missing or null values. The count returned is 11

Telco_data.isnull().sum().sum()
# Drops rows in Telco_data dataframe with null values

Telco_data = Telco_data.dropna(how='any',axis=0) 
# Searches dataframe for any missing or null values. The count returned is now 0

Telco_data.isnull().sum().sum()
# Removes the "CustomerID" column from dataframe



del Telco_data['customerID']

# Scatter plot of monthly charges vs tenure(months) 

sns.scatterplot(Telco_data['MonthlyCharges'],Telco_data['tenure'])

plt.title('Monthly Charges VS Tenure')

plt.xlabel('Monthly Charges')

plt.ylabel('Tenure')

#Creates a new dataframe from non-senior Telco_data

#nonsenior_df = Telco_data[Telco_data.SeniorCitizen == 0]

#sns.boxplot(nonsenior_df['Contract'],nonsenior_df['MonthlyCharges'])
# Boxplot showing whether or not the customer has any dependents in the household

sns.boxplot(Telco_data['Dependents'],Telco_data['MonthlyCharges'])

plt.title('Customer Dependents in Household')

# Boxplot of non-senior citizens(0)/senior citizens(1) vs monthly charges

sns.boxplot(Telco_data['SeniorCitizen'],Telco_data['MonthlyCharges'])

plt.title('Non-Senior Citizen/Senior Citizen Charges by Month')

plt.xlabel('Indicates 0 for non-senior citizens & 1 for senior citizens')

plt.ylabel('Monthly charges')
# Barplot of payment methods vs monthly charges

chart = sns.barplot(Telco_data['PaymentMethod'],Telco_data['MonthlyCharges'])

plt.title('Payment Method vs Monthly Charges')

plt.xlabel('Payment Method')

plt.ylabel('Monthly Charges')

chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
# Barplot of multiple lines vs monthly charges

sns.barplot(Telco_data['MultipleLines'],Telco_data['MonthlyCharges'])

plt.title('Multiple Lines VS Monthly Charges')

plt.xlabel('Multiple Lines')

plt.ylabel('Monthly Charges')
# Barplot indicating whether or not the customer has tech support vs monthly charges

sns.barplot(Telco_data['TechSupport'],Telco_data['MonthlyCharges'])

plt.title('Tech Support VS Monthly Charges')

plt.xlabel('Tech Support')

plt.ylabel('Monthly Charges')
# Barplot that shows whether or not senior citizens have tech support

sns.barplot(Telco_data['TechSupport'],Telco_data['SeniorCitizen'])

plt.title('Tech Support VS Senior Citizen Status')

plt.xlabel('Tech Support')

plt.ylabel('Point Percentage of Seniors')
# Barplot that indicates the point percentage of senior citizens who have tv streaming services

sns.barplot(Telco_data['StreamingTV'],Telco_data['SeniorCitizen'])

plt.title('Senior TV Streaming Services')

plt.xlabel('Streaming TV Status')

plt.ylabel('Point Percentage of Seniors')
# Barplot that displays the point percentage of seniors who have internet service through Telco

sns.barplot(Telco_data['InternetService'],Telco_data['SeniorCitizen'])

plt.title('Senior Internet Services')

plt.xlabel('Internet Service')

plt.ylabel('Point Percentage of Seniors')
# Original Telco_data split into 2 data frames each having non-senior or senior field values

nonsenior_df = Telco_data[Telco_data.SeniorCitizen == 0]

senior_df = Telco_data[Telco_data.SeniorCitizen == 1]

sns.boxplot(nonsenior_df['SeniorCitizen'],nonsenior_df['MonthlyCharges'])
# Boxplot of non-seniors who have churned based on total spend

sns.boxplot(nonsenior_df['Churn'],nonsenior_df['TotalCharges'])
# Boxplot of seniors who have churned based on total spend

sns.boxplot(senior_df['Churn'],senior_df['TotalCharges'])
# The code below preforms a 2 sample t-test against senior monthly charges and non-senior monthly charges

# Null Hypothesis: There is no difference in total charges between the senior and non-senior groups

# Alternative Hypothesis: There is a difference in total charges between the senior and non-senior groups

ttest_ind(senior_df['TotalCharges'], nonsenior_df['TotalCharges'])

# The code below preforms a 2 sample t-test against senior tenure and non-senior tenure

# Null Hypothesis: There is no difference in tenure between the senior and non-senior groups

# Alternative Hypothesis: There is a difference in tenure between the senior and non-senior groups

ttest_ind(senior_df['tenure'], nonsenior_df['tenure'])