import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# First read in the required libraries that you will be using
import numpy as np # Library for numerical arrays
import pandas as pd # library for data manupilation
import matplotlib.pyplot as plt # Library for data visualization
import seaborn as sns # library for data visualization
import warnings

# Import the Kiva_loan data set using pd.read_csv() format
kiva=pd.read_csv('/kaggle/input/data-science-for-good-kiva-crowdfunding/kiva_loans.csv')
kiva.head()
# We call the first few rows to understand the dataset using head() method

kiva.head()# Now we have the first five rows.
# We run info() to understand the  data types within the columns

kiva.info()
# We use unique() to view the values in the country column.
kiva['country'].unique() 
#Now we can get the Kiva loan data for the Rwanda region 

kiva_rwanda = kiva[kiva['country']=='Rwanda']
kiva_rwanda.head(2) # The first five rows will be displayed to have an understanding of the data set
# We can extract the required variables from (kiva) for our analysis
# We can run exploratory analysis on funded amount, 
# loan amount, activity, sector, term in months, borrower genders and repayment intervals
# we name our new data set rwanda

rwanda= kiva_rwanda[['region','funded_amount', 'loan_amount', 'activity', 'sector',
                    'term_in_months', 'borrower_genders','lender_count','repayment_interval']]
rwanda.head().reset_index()
rwanda.info()
# There are three data types, 6735 rows and 9 columns
#Check if there are any missing values on the columns
rwanda.isna().any()

# The borrower_genders and the region column contain some missing values
rwanda.isna().all() # No single column has all the data missing( This is great aspect to our analysis)
rwanda.isna().sum() # Checking the total number of missing values

# there are 14 missing values in the gender column and 6138 missing values on the region column
# Create a copy of the data frame and work with the new copy
new_rwanda=rwanda.copy()
new_rwanda.isna().sum()
# Dropping the missing values on the borrower_genders Column
new_rwanda.dropna(subset=['borrower_genders'], inplace = True)
new_rwanda.isna().sum()
# Let us examine the borrower_genders column using unique()
new_rwanda['borrower_genders'].unique()
# gettng the total unique variables in the column
new_rwanda['borrower_genders'].nunique()
# We need to clean the gender column from 2407  to 2 variables
# we create a function to fix the gender column

def fix_gender(gender):
    gender =str(gender)
    if gender.startswith('f'):
        gender = 'female'
    else:
        gender ='male'
    return gender
#Then we apply the function on the borrower_genders column to fix the problem

new_rwanda['borrower_genders']= new_rwanda['borrower_genders'].apply(fix_gender)
# lets check if the problem is sorted

new_rwanda['borrower_genders'].unique()
# Boom!!!!! Now we have two variables as needed for our anlysis

new_rwanda['borrower_genders'].nunique()
# Let us explore the numerical values by using describe() function
new_rwanda.describe()

# The maximum loan given was 50,000; the longest loan term was 41 months, the lender_count maximum was 1302
# We can check the data distribution by using hist() functions

new_rwanda['funded_amount'].hist(bins=50) # No extreme values so the data is okay

new_rwanda['loan_amount'].hist(bins=50) # No extreme values so the data is okay
new_rwanda.info()
new_rwanda.head()
funded_amount = new_rwanda['funded_amount']
loan_amount =new_rwanda['loan_amount']
activity=new_rwanda['activity']
sector=new_rwanda['sector']
term=new_rwanda['term_in_months']
gender =new_rwanda['borrower_genders']
count=new_rwanda['lender_count']
repayment =new_rwanda['repayment_interval']
region= new_rwanda['region']
region.isna().sum() # There are many missing values, so no analysis will be carried out on regions in Rwanda
# Lets check for duplicates

new_rwanda.duplicated().sum() # There are 753 duplicates
# Drop the duplicates
new_rwanda_dups = rwanda.copy()
new_rwanda_dups.head().isna()
new_rwanda_dups.drop_duplicates(inplace=True)
new_rwanda_dups
loan_amount.sum() # The total Kiva loan in Rwanda region 
# We start grouping the values with sector

rwanda_region =new_rwanda.groupby(['region','sector','activity','borrower_genders', 'repayment_interval']).sum().sort_values(by='loan_amount', ascending=False).reset_index()
rwanda_region.head()
rwanda_region_sector =new_rwanda.groupby(['region','sector','borrower_genders', 'repayment_interval']).sum().sort_values(by='loan_amount', ascending =False).reset_index()
rwanda_region_sector.head()
# we can compare loan amount given across different sectors
plt.figure(figsize=(15,10))
plt.title('Loan amount in every sector')
plt.xlabel('sector')
plt.ylabel('loan_amount')

plt.xticks(rotation =75)

sns.barplot(x='sector', y='loan_amount', data =rwanda_region_sector, ci =None)

plt.show()

# we can compare loan amount given across different sectors per gender
plt.figure(figsize=(15,10))
plt.title('Loan amount in every sector by gender')
plt.xlabel('sector')
plt.ylabel('loan_amount')

plt.xticks(rotation =75)

sns.barplot(x='sector', y='loan_amount', data =rwanda_region_sector, ci =None, hue ='borrower_genders')

plt.show()
# we can compare loan amount given across different sectors by repayment interval
plt.figure(figsize=(15,10))
plt.title('Loan amount in every sector by repayment interval')
plt.xlabel('sector')
plt.ylabel('loan_amount')

plt.xticks(rotation =75)

sns.barplot(x='sector', y='loan_amount', data =rwanda_region_sector, ci =None, hue ='repayment_interval')

plt.show()
# Loan amount vs term in months
plt.figure(figsize=(15,10))
plt.title('Loan amount vs term in months')
plt.xticks(rotation=75)

sns.barplot(x='term_in_months', y = 'loan_amount', data =new_rwanda, ci=None)
plt.show()
# Loan amount vs term in months by repayment interval
plt.figure(figsize=(15,10))
plt.title('Loan amount vs term in months by repayment interval')
plt.xticks(rotation=75)

sns.barplot(x='term_in_months', y = 'loan_amount', data =new_rwanda, ci=None, hue= 'repayment_interval')
plt.show()
# We compare the loan amount repayment interval
plt.figure(figsize=(15,10))
plt.title('loan amount vs Repayment interval')
plt.xticks(rotation=75)

sns.barplot(x='repayment_interval', y='loan_amount', data = rwanda_region, ci=None )
plt.show()
# We compare the loan amount repayment interval between the genders in Rwanda
plt.figure(figsize=(15,10))
plt.title('loan amount vs Repayment interval by gender ')
plt.xticks(rotation=75)

sns.barplot(x='repayment_interval', y='loan_amount', data = rwanda_region, ci=None, hue ='borrower_genders')
plt.show()
new_rwanda.info()
new_rwanda['term_in_months'].nunique()
new_rwanda['term_in_months'].unique()
new_rwanda['activity'].nunique()
new_rwanda['activity'].unique()
new_rwanda['region'].nunique()
new_rwanda['region'].unique()
new_rwanda['sector'].nunique()
new_rwanda['sector'].unique()
new_rwanda['borrower_genders'].nunique()
new_rwanda['borrower_genders'].unique()
new_rwanda['repayment_interval'].nunique()
new_rwanda['repayment_interval'].unique()
# Scatter plot to show relationship between loan amount and the term_in_months

plt.figure(figsize=(15,10))
plt.title('Loan amount vs Term in months')

sns.scatterplot(x='term_in_months', y='loan_amount', data =new_rwanda)
plt.show()
# Scatter plot to show relationship between loan amount and the term_in_months per sector

plt.figure(figsize=(15,10))
plt.title('Loan amount vs Term in months by sector per gender')

sns.scatterplot(x='term_in_months', y='loan_amount', data =new_rwanda, hue= 'sector', size='loan_amount',
                style='borrower_genders',  sizes=(100,500))
plt.show()
# Scatter plot to show relationship between loan amount and lender count

plt.figure(figsize=(15,10))
plt.title('Loan amount vs lender count')

sns.scatterplot(x='lender_count', y='loan_amount', data =rwanda_region_sector)
plt.show()
# Scatter plot to show relationship between loan amount and lender count per sector

plt.figure(figsize=(15,10))
plt.title('Loan amount vs lender count')

sns.scatterplot(x='lender_count', y='loan_amount', data =rwanda_region_sector, hue= 'sector', size='loan_amount',
                style='borrower_genders',  sizes=(100,500))
plt.show()
# We plot the relational plot to show relationship between loan amount, 
# term in months between genders among different sectors

plt.figure(figsize=(15,10))
sns.relplot(y='loan_amount', x='term_in_months', data =rwanda_region_sector, row='borrower_genders', col='repayment_interval',
           hue='sector',size='loan_amount', sizes=(100,500),aspect=1.5)
plt.show()
# We plot the relational plot to show relationship between loan amount and lender count between genders 
# among different sectors

plt.figure(figsize=(15,10))
sns.relplot(y='loan_amount', x='lender_count', data =rwanda_region_sector, row='borrower_genders', col='repayment_interval',
           hue='sector', size='loan_amount', sizes=(100,500), aspect=1.5)
plt.show()
# first is to generate correlation values for the rwanda_region_sector variable

rwanda_region_sector_corr = rwanda_region_sector.corr()
rwanda_region_sector_corr
# Then we apply the heatmap() function

sns.heatmap(rwanda_region_sector_corr, cmap= 'coolwarm', annot=True, linewidth=0.5)
plt.show()
sns.pairplot(rwanda_region_sector)
