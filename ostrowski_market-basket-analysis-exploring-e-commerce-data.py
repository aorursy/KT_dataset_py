# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# Loading libraries for python
import numpy as np # Linear algebra
import pandas as pd # Data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # Data visualization
import seaborn as sns # Advanced data visualization
import re # Regular expressions for advanced string selection
from mlxtend.frequent_patterns import apriori # Data pattern exploration
from mlxtend.frequent_patterns import association_rules # Association rules conversion
from mlxtend.preprocessing import OnehotTransactions # Transforming dataframe for apriori
import missingno as msno # Advanced missing values handling
%matplotlib inline
# Reading input, converting InvoiceDate to TimeStamp, and setting index: df
# Note that we're working only with 30000 rows of data for a methodology concept proof 
df = pd.read_csv('../input/data.csv', nrows=30000)
df.InvoiceDate = pd.to_datetime(df.InvoiceDate)
df.set_index(['InvoiceDate'] , inplace=True)

# Dropping StockCode to reduce data dimension
# Checking df.sample() for quick evaluation of entries
df.drop('StockCode', axis=1, inplace=True)
df.sample(5, random_state=42)
# Checking missing and data types
# Experimenting missingno library for getting deeper understanding of missing values 
# Checking functions on msno with dir(msno)
print(df.info())
msno.bar(df);
msno.heatmap(df);

# InvoiceNo should be int64, there must be something wrong on this variable
# When trying to use df.InvoiceNo.astype('int64') we receive an error 
# stating that it's not possible to convert str into int, meaning wrong entries in the data.
# Zoming into missing values
# On df.head() only CustomerID is missing
# We notice the same problem in Description when exploring find_nans a bit
find_nans = lambda df: df[df.isnull().any(axis=1)]
# For a data loss management (dlm) we will track data dropped every .drop() step
dlm = 0
og_len = len(df.InvoiceNo)

# It does not matter not having CustomerID in this analysis
# however a NaN Description shows us a failed transaction
# We will drop NaN CustomerID when analysing customer behavior 
df.dropna(inplace=True, subset=['Description'])

# data_loss report
new_len = len(df.InvoiceNo)
dlm += (og_len - new_len)
print('Data loss report: %.2f%% of data dropped, total of %d rows' % (((og_len - new_len)/og_len), (og_len - new_len)))
print('Data loss totals: %.2f%% of total data loss, total of %d rows\n' % ((dlm/og_len), (dlm)))
mod_len = len(df.InvoiceNo)
df.info()
# Note that for dropping the rows we need the .index not a boolean list
# to_drop is a list of indices that will be used on df.drop()
to_drop = df[df.InvoiceNo.str.match('^[a-zA-Z]')].index

# Droping wrong entries starting with letters
# Our assumption is that those are devolutions or system corrections
df.drop(to_drop, axis=0, inplace=True)

# Changing data types for reducing dimension and make easier plots
df.InvoiceNo = df.InvoiceNo.astype('int64')
df.Country = df.Country.astype('category')
new_len = len(df.InvoiceNo)

# data_loss report
new_len = len(df.InvoiceNo)
dlm += (mod_len - new_len)
print('Data loss report: %.2f%% of data dropped, total of %d rows' % (((mod_len - new_len)/mod_len), (mod_len - new_len)))
print('Data loss totals: %.2f%% of total data loss, total of %d rows' % ((dlm/og_len), (dlm)))
mod_len = len(df.InvoiceNo)
# Checking categorical data from df.Country
# unique, counts = np.unique(df.Country, return_counts=True)
# print(dict(zip(unique, counts)))
country_set = df[['Country', 'InvoiceNo']]
country_set = country_set.pivot_table(columns='Country', aggfunc='count')
country_set.sort_values('InvoiceNo', axis=1, ascending=False).T
# Plotting InvoiceNo distribution per Country
plt.figure(figsize=(14,6))
plt.title('Distribuition of purchases in the website according to Countries');
sns.countplot(y='Country', data=df);
# Plotting InvoiceNo without United Kingdom
df_nUK = country_set.T.drop('United Kingdom')
plt.figure(figsize=(14,6))
plt.title('Distribuition of purchases in the website according to Countries');
# Note that since we transformed the index in type category the .remove_unused_categories is used
# otherwise it woul include a columns for United Kingdom with 0 values at the very end of the plot
sns.barplot(y=df_nUK.index.remove_unused_categories(), x='InvoiceNo', data=df_nUK, orient='h');
# Creating subsets of df for each unique country
def df_per_country(df):
    df_dict = {}
    unique_countries, counts = np.unique(df.Country, return_counts=True)
    for country in unique_countries:
        df_dict["df_{}".format(re.sub('[\s+]', '', country))] = df[df.Country == country].copy()
        # This line is giving me the warning, I will check in further research
        # After watching Data School video about the SettingWithCopyWarning I figured out the problem
        # When doing df[df.Country == country] adding the .copy() points pandas that this is an actual copy of the original df
        df_dict["df_{}".format(re.sub('[\s+]', '', country))].drop('Country', axis=1, inplace=True)
    return df_dict

# Trick to convert dictionary key/values into variables
# This way we don't need to access dfs by df_dict['df_Australia'] for example
df_dict = df_per_country(df)
locals().update(df_dict)
# Series plot function summarizing df_Countries
def series_plot(df, by1, by2, by3, period='D'):
    df_ts = df.reset_index().pivot_table(index='InvoiceDate', 
                                values=['InvoiceNo', 'Quantity', 'UnitPrice'], 
                                aggfunc=('count', 'sum'))
    df_ts = df_ts.loc[:, [('InvoiceNo', 'count'), ('Quantity', 'sum'), ('UnitPrice', 'sum')]]
    df_ts.columns = df_ts.columns.droplevel(1)
    plt.figure(figsize=(14, 6))
    
    plt.subplot(2, 2, 1)
    plt.plot(df_ts.resample(period).sum().bfill()[[by1]], color='navy')
    plt.title('{}'.format(by1));
    plt.xticks(rotation=60);
    plt.subplot(2, 2, 2)
    plt.title('{}'.format(by2));
    plt.plot(df_ts.resample(period).sum().bfill()[[by2]], label='Total Sale', color='orange');
    plt.xticks(rotation=60)
    plt.tight_layout()
    
    plt.figure(figsize=(14, 8))
    plt.title('{}'.format(by3));
    plt.plot(df_ts.resample(period).sum().bfill()[[by3]], label='Total Invoices', color='green');
    plt.tight_layout()
series_plot(df_UnitedKingdom, 'Quantity', 'UnitPrice', 'InvoiceNo')
# Starting preparation of df for receiving product association
# Cleaning Description field for proper aggregation 
df_UnitedKingdom.loc[:, 'Description'] = df_UnitedKingdom.Description.str.strip().copy()
# Once again, this line was generating me the SettingWithCopyWarning, solved by adding the .copy()

# Dummy conding and creation of the baskets_sets, indexed by InvoiceNo with 1 corresponding to every item presented on the basket
# Note that the quantity bought is not considered, only if the item was present or not in the basket
basket = pd.get_dummies(df_UnitedKingdom.reset_index().loc[:, ('InvoiceNo', 'Description')])
basket_sets = pd.pivot_table(basket, index='InvoiceNo', aggfunc='sum')
# Apriori aplication: frequent_itemsets
# Note that min_support parameter was set to a very low value, this is the Spurious limitation, more on conclusion section
frequent_itemsets = apriori(basket_sets, min_support=0.03, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

# Advanced and strategical data frequent set selection
frequent_itemsets[ (frequent_itemsets['length'] > 1) &
                   (frequent_itemsets['support'] >= 0.02) ].head()
# Generating the association_rules: rules
# Selecting the important parameters for analysis
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules[['antecedants', 'consequents', 'support', 'confidence', 'lift']].sort_values('support', ascending=False).head()
# Visualizing the rules distribution color mapped by Lift
plt.figure(figsize=(14, 8))
plt.scatter(rules['support'], rules['confidence'], c=rules['lift'], alpha=0.9, cmap='YlOrRd');
plt.title('Rules distribution color mapped by lift');
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.colorbar();