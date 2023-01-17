# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Code to Check the Path of the File

import os

for dirname in os.walk('/kaggle/input'):

    print(dirname)
# Import Important Libraries & Packages

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sb

%matplotlib inline



# Suppress Warnings from Final Output

import warnings

warnings.simplefilter("ignore")
# Load Data into DataFrame

loan_df = pd.read_csv('/kaggle/input/prosper-loan/prosperLoanData.csv')
loan_df.head(5)
loan_df = loan_df[['ListingNumber', 'ListingCreationDate', 'BorrowerRate', 'ListingCategory (numeric)', 'IncomeRange', 'EmploymentStatus', 'EmploymentStatusDuration', 'IsBorrowerHomeowner', 'TotalCreditLinespast7years', 'LoanOriginalAmount', 'MonthlyLoanPayment']]
loan_df.head()
loan_df.info()
loan_df.describe()
loan_df[loan_df.duplicated()]
# Change Datatypes: ListingCreationDate to datetime

loan_df['ListingCreationDate'] = pd.to_datetime(loan_df['ListingCreationDate'])
# Remove Duplicate Rows

loan_df.drop_duplicates(inplace=True)



# Reset Index of the Dataset

loan_df.reset_index(drop= True, inplace=True)
# Dictionary of Listing Categories

listing_cat = {0: 'Not Available', 1: 'Debt Consolidation', 2: 'Home Improvement', 3: 'Business', 4: 'Personal Loan', 5: 'Student Use', 6: 'Auto', 7: 'Other', 8: 'Baby&Adoption', 9: 'Boat', 10: 'Cosmetic Procedure', 11: 'Engagement Ring', 12: 'Green Loans', 13: 'Household Expenses', 14: 'Large Purchases', 15: 'Medical/Dental', 16: 'Motorcycle', 17: 'RV', 18: 'Taxes', 19: 'Vacation', 20: 'Wedding Loans'}



# DataFrame for Listing Categories 

df_cat = pd.DataFrame(list(listing_cat.items()))



# Apply Correct Naming for Columns

df_cat.rename(columns={0: "CategoryNum", 1: "LoanCategory"}, inplace= True)



# Merge Listing Category Names & Drop the Listing Category Code

loan_df = loan_df.merge(df_cat, how= 'left', left_on= 'ListingCategory (numeric)', right_on='CategoryNum')

loan_df.drop(labels=['ListingCategory (numeric)', 'CategoryNum'], axis = 1, inplace= True)



# Turn into Category Datatype

loan_df['LoanCategory'] = loan_df['LoanCategory'].astype('category')
# Reset All Category Types



# Convert Datatype of ListingCreationDate to datetime 

loan_df['ListingCreationDate'] = pd.to_datetime(loan_df['ListingCreationDate'])



# Loan category to category

loan_df['LoanCategory'] = loan_df['LoanCategory'].astype('category')



# Converting the EmploymentStatus into an Ordered Category Type

order_empl_status = ['Employed', 'Full-time', 'Self-employed', 'Part-time', 'Retired', 'Not employed', 'Not available', 'Other']

ordered_empl_status = pd.api.types.CategoricalDtype(order_empl_status, ordered= True)

loan_df['EmploymentStatus'] = loan_df['EmploymentStatus'].astype(ordered_empl_status)



# Converting the IncomeRange into an Ordered Category Type

order_income_range = ['$100,000+', '$75,000-99,999', '$50,000-74,999', '$25,000-49,999', '$1-24,999', '$0', 'Not employed', 'Not displayed']

ordered_income_range = pd.api.types.CategoricalDtype(order_income_range, ordered= True)

loan_df['IncomeRange'] = loan_df['IncomeRange'].astype(ordered_income_range)
# Start with a Standard-Scaled Plot

binsize = 0.01

bins = np.arange(0, loan_df['BorrowerRate'].max()+binsize, binsize)

plt.figure(figsize=[8, 5])

plt.hist(data = loan_df, x = 'BorrowerRate', bins = bins)

plt.xlabel('Interest Rate')

plt.title('Distribution of Interest Rates')

plt.text(0.21, 6000, 'mean = 0.19', bbox=dict(facecolor='orange', alpha=0.5))

plt.axvline(loan_df['BorrowerRate'].mean(), color='orange', linestyle='-.', linewidth=1)

plt.show()
# Plot Clustered Bar Charts to Evaluate Relationship Between Categorical Variables 

plt.figure(figsize = [14, 5])

plt.subplot(1,2,1)

ax = sb.countplot(data = loan_df, x = 'EmploymentStatus', hue = 'IsBorrowerHomeowner')

ax.legend(loc = 1, ncol = 3, framealpha = 1, title = 'IsBorrowerHomeowner?')

ax.set_title('Number of Loans by Employment Status of the Borrower')

ax.set_xlabel('Employment Status')

ax.set_ylabel('Number of Loans')



plt.xticks(rotation= 90)



plt.subplot(1,2,2)

ax = sb.countplot(data = loan_df, x = 'IncomeRange', hue = 'IsBorrowerHomeowner')

ax.legend(loc = 1, ncol = 3, framealpha = 1, title = 'Is Borrower Homeowner?')

ax.set_title('Number of Loans by Income Range of the Borrower')

ax.set_xlabel('Income Range')

ax.set_ylabel('Number of Loans')



plt.xticks(rotation= 90);
# Create Column Employed vs. Not Employed

loan_df['Employed'] = loan_df['EmploymentStatus'].apply(lambda x: 'False' if x == "Not employed" else 'True')



# Boxplot 2 Categorical Variables (Homeowner & Employed) vs. Interest Rate

g = sb.FacetGrid(data = loan_df, col = 'Is Borrower Homeowner', height = 4)

g.map(sb.boxplot, 'Employed', 'BorrowerRate')

plt.subplots_adjust(top=0.8)

g.fig.suptitle('Interest Rate by Employment Status & Homeownership')

g.axes[0,0].set_ylabel('Interest Rate');
# Plot boxplots to illustrate relationship between borrower's rate and categorical variables

plt.figure(figsize=(16, 6))

base_color = sb.color_palette()[0]

sb.boxplot(data = loan_df, x = 'IncomeRange', y = 'BorrowerRate', color = 'base_color')

plt.axhline(loan_df['BorrowerRate'].mean(), color='orange', linestyle='-.', linewidth=1)

plt.text(0, loan_df['BorrowerRate'].mean()+0.2, 'mean = 0.19', bbox=dict(facecolor='orange', alpha=0.5))

plt.xticks(rotation = 90)

plt.ylabel('Interest Rate')

plt.xlabel('Income Range')

plt.title('Interest Rate by Income Range');
# Scatter Plots of Loan Amounts vs. Interest Rate (Employed vs. Non-Employed) and also Separate by Loan Category

g = sb.FacetGrid(data = loan_df, col='LoanCategory' , col_wrap=3, hue = 'Employed', height = 5)

g.map(plt.scatter, 'LoanOriginalAmount', 'BorrowerRate')

g.add_legend()

g.set_titles('{col_name}')

g.axes[0].set_ylabel('Interest Rate')

g.axes[3].set_ylabel('Interest Rate')

g.axes[6].set_ylabel('Interest Rate')

g.axes[9].set_ylabel('Interest Rate')

g.axes[12].set_ylabel('Interest Rate')

g.axes[15].set_ylabel('Interest Rate')

g.axes[18].set_ylabel('Interest Rate')

g.axes[18].set_xlabel('Loan Amount')

g.axes[19].set_xlabel('Loan Amount')

g.axes[20].set_xlabel('Loan Amount')

plt.subplots_adjust(top=0.95)

g.fig.suptitle('Interest Rate by Loan Amount & Split by Loan Category (Employed vs. Non-Employed Borrowers)', fontsize=16);