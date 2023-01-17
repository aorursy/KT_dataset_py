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
import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker
customer = pd.read_csv('../input/clients/customers.csv')

transaction = pd.read_csv('../input/customer-transactions/transactions.csv')
customer = customer.drop( labels = ['Unnamed: 0'], axis = 1)

transaction = transaction.drop( labels = ['Unnamed: 0'], axis = 1)
customer[['birth_date']] = pd.to_datetime(customer['birth_date'])
customer.birth_date
customer['age'] = np.nan
current_year = pd.Timestamp.now().year

for index, row in customer.iterrows():

    customer.age[index] = current_year - customer.birth_date[index].year

customer.state.value_counts()
customer.state = customer.state.replace(to_replace='NSW', value = 'New South Wales')

customer.state = customer.state.replace(to_replace='QLD', value = 'Queensland')

customer.state = customer.state.replace(to_replace='VIC', value = 'Victoria')
newCustomer = customer.loc[ customer.customer_id > 4003].reset_index()

customer = customer.loc[ customer.customer_id < 4003]
customer.isnull().sum()
print('percent of missing values {} %'.format(88/ customer.shape[0] * 100))

print('shape before treatment', customer.shape)
customer = customer.dropna( axis = 0, subset=['gender', 'birth_date', 'tenure'])

print('shape after treatment', customer.shape)
customer.isnull().sum()
print('missing values in job_title {} %'.format( 495 / customer.shape[0]* 100))

print('missing values in job_category {} %'.format( 655 / customer.shape[0]* 100))
customer.job_title = customer.job_title.replace( np.nan, 'No job')

customer.job_industry_category = customer.job_industry_category.replace( np.nan, 'No industry category')
fig, axs = plt.subplots(3, 3, figsize = (20, 15))



sns.countplot(x = 'state', data = customer, ax = axs[0][0], palette="Blues", order = customer['state'].value_counts().index)

axs[0][0].set_title('Customer by state')

axs[0][0].set_xlabel('State')

axs[0][0].set_ylabel('Number of customers')





sns.countplot(x = 'wealth_segment', data = customer, ax = axs[0][1], palette="Greens", order = customer['wealth_segment'].value_counts().index)

axs[0][1].set_title('Customer by affluence')

axs[0][1].set_xlabel('Affluence')

axs[0][1].set_ylabel('Number of customers')



sns.countplot(x = 'gender', data = customer, ax = axs[0][2], palette="Reds", order = customer['gender'].value_counts().index)

axs[0][2].set_title('Customer by gender')

axs[0][2].set_xlabel('Gender')

axs[0][2].set_ylabel('Number of customers')



sns.countplot(x = 'owns_car', data = customer, ax = axs[1][0], palette="Oranges", order = customer['owns_car'].value_counts().index)

axs[1][0].set_title('Customer who owns car')

axs[1][0].set_xlabel('Car owner')

axs[1][0].set_ylabel('Number of customers')



sns.distplot( customer.property_valuation, ax = axs[1][1])

axs[1][1].set_title('Customer by property valuation')

axs[1][1].set_xlabel('Property valuation')

axs[1][1].set_ylabel('Number of customers')



sns.distplot( customer.past_3_years_bike_related_purchases, ax = axs[1][2])

axs[1][2].set_title('Customer by related purchases')

axs[1][2].set_xlabel('Related purchases')

axs[1][2].set_ylabel('Number of customers')



sns.countplot(y = 'job_industry_category', data = customer, ax = axs[2][0], palette="Purples", order = customer['job_industry_category'].value_counts().index)

axs[2][0].set_title('Customer by job industry')

axs[2][0].set_xlabel('Job industry')

axs[2][0].set_ylabel('Number of customers')



sns.distplot( customer.age, ax = axs[2][1])

axs[2][1].set_title('Customer by age')

axs[2][1].set_xlabel('Age')

axs[2][1].set_ylabel('Number of customers')



sns.distplot( customer.tenure, ax = axs[2][2])

axs[2][2].set_title('Customer by tenure')

axs[2][2].set_xlabel('Tenure')

axs[2][2].set_ylabel('Number of customers')



fig.suptitle('Customer profile')
sns.countplot( x = 'gender', hue = 'wealth_segment', data = customer,  palette = 'Blues')
sns.countplot( x = 'gender', hue = 'owns_car', data = customer,  palette = 'Blues')
fig, ax = plt.subplots( figsize = (20, 5))

sns.countplot( x = 'job_industry_category', hue = 'gender', data = customer,ax = ax, palette = 'Blues')
females = customer[ customer.gender == 'F']

males = customer[ customer.gender == 'M']



sns.kdeplot( data = females.age, legend = 'females')

sns.kdeplot( data = males.age, legend = 'males')
sales = transaction.groupby('transaction_date', as_index = False).list_price.sum()

fig, ax = plt.subplots( figsize = (20, 5))

sns.lineplot( x = 'transaction_date', y = 'list_price', data = sales, ax = ax)

ax.xaxis.set_major_locator(ticker.MaxNLocator(12))
sns.countplot( x = 'online_order', data = transaction)
fig, ax = plt.subplots( figsize = (10, 5))

sns.countplot( x = 'brand', data = transaction, hue = 'product_class', ax = ax, palette = 'Blues', order = transaction['brand'].value_counts().index)
transaction['profit'] = transaction['list_price'] - transaction ['standard_cost']
profit = transaction.groupby('transaction_date', as_index  = False).profit.sum()

fig, ax = plt.subplots( figsize = (20, 5))

sns.lineplot( x = 'transaction_date', y = 'profit', data = profit, ax = ax)

ax.xaxis.set_major_locator(ticker.MaxNLocator(12))
profit_by_brand_and_product_class = transaction.groupby(['brand', 'product_class'], as_index = False).profit.sum()

fig, ax = plt.subplots( figsize = (10, 5))

sns.barplot( x = 'brand', y = 'profit', hue = 'product_class', data = profit_by_brand_and_product_class, ax = ax)
profit_by_product_line_and_product_size = transaction.groupby(['product_size', 'product_line'], as_index = False).profit.sum()

profit_by_product_line_and_product_size

sns.barplot( x = 'product_line', y = 'profit', hue = 'product_size', data = profit_by_product_line_and_product_size)
dataset = transaction.merge( customer, on = 'customer_id', suffixes=['_customer', '_transaction'])

dataset
print('customer shape : ', customer.shape[0])

print('transaction shape : ', transaction.shape[0])
profit_by_gender_and_affluence = dataset.groupby(['gender', 'wealth_segment'], as_index = False).profit.sum()

sns.barplot( x = 'wealth_segment', y = 'profit', hue = 'gender', data = profit_by_gender_and_affluence)
profit_by_brand_productline_and_affluence = dataset.groupby(['state', 'wealth_segment'], as_index = False).profit.sum()

sns.barplot( x = 'state', y = 'profit', hue = 'wealth_segment', data = profit_by_brand_productline_and_affluence)
corr = dataset.corr()

sns.heatmap(corr)
dataset