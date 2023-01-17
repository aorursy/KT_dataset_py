# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline
# Load the data
data_dict = pd.read_excel('../input/Data_Dictionary.xlsx')
historical_transactions = pd.read_csv('../input/historical_transactions.csv')
merchants = pd.read_csv('../input/merchants.csv')
new_transactions = pd.read_csv('../input/new_merchant_transactions.csv')
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
print(data_dict)

# The Excel documents actually has several sheets, which we load in turn and append
excel_doc = pd.ExcelFile('../input/Data_Dictionary.xlsx')
data_dict_train  = pd.read_excel(excel_doc, 'train', skiprows=2)
data_dict_historical = pd.read_excel(excel_doc, 'history', skiprows=2)
data_dict_new_merchants = pd.read_excel(excel_doc, 'new_merchant_period', skiprows=2)
data_dict_merchant = pd.read_excel(excel_doc, 'merchant', skiprows=2)

data_dict = data_dict_train.append(data_dict_historical) \
                            .append(data_dict_new_merchants).append(data_dict_merchant)

print(data_dict)
historical_transactions.head()
merchants.head()
new_transactions.head()
train_data.head()
# Check for null values
data_dict.isnull().sum()
# Check for duplicates
data_dict.duplicated().sum()

# Remove duplicates
data_dict.drop_duplicates(inplace=True)
# Check for null values
historical_transactions.isnull().sum()
# Let's examine what categories 2 and 3 represent, from the data dictionary
data_dict.loc[(data_dict.Columns == 'category_3') | (data_dict.Columns == 'category_2'),]
# Let's examine a few rows with missing merchant ID's
historical_transactions.loc[historical_transactions.merchant_id.isnull(),].head()
# Let's drop rows with missing merchant ID's, but keep those with missing categories 2 and 3
historical_transactions.dropna(subset=['merchant_id'], axis=0, inplace=True)
# Check for duplicates
historical_transactions.duplicated().sum()
# Visualize possible values for categories 1, 2 and 3 to check for invalid data
print(historical_transactions.category_1.unique(),
      historical_transactions.category_2.unique(),
      historical_transactions.category_3.unique())
# Definition of purchase amount
print(data_dict.loc[data_dict.Columns == 'purchase_amount',])

# Boxplot
plt.boxplot(historical_transactions.purchase_amount)
plt.show();
# Remove 1st outlier at 600,000 (probably was not normalized), and visualize again
historical_transactions = historical_transactions.loc[historical_transactions.purchase_amount < 500000,]
plt.boxplot(historical_transactions.purchase_amount)
plt.show();
# Remove more outliers, greater than 2
historical_transactions = historical_transactions.loc[historical_transactions.purchase_amount < 2,]
plt.boxplot(historical_transactions.purchase_amount)
plt.show();
# Boxplot for month lag
plt.boxplot(historical_transactions.month_lag)
plt.show();
# Boxplot for installments
plt.boxplot(historical_transactions.installments)
plt.show();

# Remove the outlier at 1000 installments
historical_transactions = historical_transactions.loc[historical_transactions.installments < 900,]
# Examine null values
merchants.isnull().sum()
# Define the lagged sales
print(data_dict.loc[data_dict.Columns == 'avg_sales_lag3','Description'])

# Drop rows with missing values of lagged sales
merchants.dropna(subset=['avg_sales_lag3', 'avg_sales_lag6', 'avg_sales_lag12'], axis=0, inplace=True)
# Create boxplots for numerical columns
merchants.boxplot()
plt.xticks(rotation='vertical')
plt.show();
# Check for missing values
new_transactions.isnull().sum()
# Similar to the historical transactions, we drop missing merchant ID's
new_transactions.dropna(subset = ['merchant_id'], axis=0, inplace=True)
# Check for duplicates
new_transactions.duplicated().sum()
# Check for outliers
new_transactions.boxplot()
plt.xticks(rotation='vertical')
plt.show();
# Similarly to historical transactions, we remove outliers for the purchase amount and installments
new_transactions = new_transactions.loc[(new_transactions.purchase_amount < 2) & (new_transactions.installments < 900),]
# Check for null values
train_data.isnull().sum()
# Check for duplicates
train_data.isnull().sum()
# Visualize the range of values for the target
plt.boxplot(train_data.target)
plt.show();
historical_transactions.to_csv('historical_transactions_clean.csv')
