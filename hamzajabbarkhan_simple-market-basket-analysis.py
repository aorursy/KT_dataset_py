import pandas as pd
data = pd.read_excel("../input/online-retail-store-data-from-uci-ml-repo/Online Retail.xlsx")
data.head()
data.shape
data.info()
data['Country'].value_counts()
data.describe()
data.head()
data.isnull().sum()
data.dropna(subset = ['Description'], inplace = True)
data.shape
germany_data = data.query("Country == 'Germany'")
germany_data.shape
germany_data.isnull().sum()
germany_data['Description'].head()
germany_data['Description'] = germany_data['Description'].str.strip()
germany_data['InvoiceNo'].str.contains('C').sum()
germany_data['InvoiceNo'] = germany_data['InvoiceNo'].astype('str')
germany_data = germany_data[~germany_data['InvoiceNo'].str.contains('C')]
germany_data.shape
basket = germany_data.groupby(['InvoiceNo','Description'])['Quantity'].sum().unstack()
basket.head()
basket.notnull().sum()
basket = basket.fillna(0)
basket.head()
def convert_values(value):

    if value >= 1:

        return 1

    else:

        return 0 
basket = basket.applymap(convert_values)
basket = basket.drop('POSTAGE', axis = 1)
from mlxtend.frequent_patterns import apriori 

from mlxtend.frequent_patterns import association_rules
basket_items = apriori(basket, min_support = 0.05, use_colnames = True)
rules = association_rules(basket_items, metric = 'lift')
rules