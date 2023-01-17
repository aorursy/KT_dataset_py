import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import re



%matplotlib inline
def get_column_null_count(data_frame):

    """

    Function prints all column names and count of null values in it

    :param data_frame: a pandas DataFrame

    :return: null

    """

    for i in data_frame.columns:

        print("{} : {}".format(i, len(data_frame[pd.isnull(data_frame[i])])))





def update_column_names(data_frame):

    """

    Function updates all the column names to lowercase 

    and spaces with underscore

    :param data_frame: a pandas DataFrame

    :return: null

    """

    new_col = []

    for name in data_frame.columns:

        new_col.append(name.lower().replace(' ', '_'))

    data_frame.columns = new_col





def get_data_info(data_frame):

    """

    Function to print description of a Data Frame

    It prints Shape, Index, Size, Column Names and column wise null count 

    :param data_frame: a pandas DataFrame

    :return: null

    """

    print("Shape : {}".format(data_frame.shape))

    print("Index : {}".format(data_frame.index))

    print("Size : {}".format(data_frame.size))

    print("Column Names : {}".format(data_frame.columns))

    print("Null count in each columns ")

    get_column_null_count(data_frame)





def get_data(data_frame, head=3, tail=3, mid=6):

    """

    Function to get a particular number of rows from a DataFrame

    :param data_frame: a pandas DataFrame

    :param head: count of rows from top

    :param tail: count of rows from bottom

    :param mid: count of rows from middle

    :return: a pandas DataFrame

    """

    df_head = data_frame.head(head)

    df_tail = data_frame.tail(tail)

    df_mid = data_frame.iloc[np.random.randint(3000, 7000, size=mid)]

    df = df_head.append(df_mid).append(df_tail)

    return df





def display(series):

    """

    Function to print a series in a table/frame format

    :param series: a series

    :return: series in a table format

    """

    return series.to_frame()
# Reading from dataset

ecom = pd.read_csv("../input/ecommerce-purchases/Ecommerce_Purchases.csv")
# Calling our function to get information of our Dataframe 

get_data_info(ecom)
# Calling our function to get rows of our Dataframe

get_data(ecom)
# updating column names to a common format

update_column_names(ecom)

ecom.columns
get_data(ecom)
# printing unique languages in the data

ecom['language'].unique()
# Preapring a dictionary to map the languages with its ISO code

lang_map = {

    'de': 'German',

    'el': 'Greek',

    'en': 'English',

    'es': 'Spanish',

    'fr': 'French',

    'it': 'Italian',

    'pt': 'Portuguese',

    'ru': 'Russian',

    'zh': 'Chinese'

}

# Replacing the language data.

ecom.language.replace(lang_map, inplace=True)

get_data_info(ecom)

get_data(ecom)
# Spliting the browser info column data to get browser and OS names

data = ecom['browser_info'].str.split('(')

browser = []

os_name = []

for d in data:

    browser.append(d[0])

    if re.search('Mac', d[1]):

        os_name.append("Mac")

    elif re.search('Windows', d[1]):

        os_name.append("Windows")

    elif re.search('Linux', d[1]):

        os_name.append("Linux")

    else:

        os_name.append("")

# Creating new columns

ecom['browser_with_version'] = browser

ecom['os'] = os_name

update_column_names(ecom)

get_data_info(ecom)
get_data(ecom, mid=0)
# Getting only browser name

data = ecom['browser_with_version'].str.split('/')

li = []

for d in data:

    li.append(d[0])

ecom['browser_name'] = li

update_column_names(ecom)

get_data(ecom, head=2, mid=10, tail=1)
# Extracting cleaned data to a new csv file

ecom.to_csv("Ecommerce_Purchases_cleaned.csv", index = False)
ecom['language'].unique()
lang_count = ecom.language.value_counts()

display(lang_count)
lang_count.plot(kind='bar',title='Bar Plot for the number of languages',rot=45)

plt.savefig("lang_bar.png")

plt.show()
lang_price_sum = ecom.groupby(['language'])['purchase_price'].sum()
plt.figure(figsize=(20,10))

lang_price_sum.plot(kind='barh',title='Bar plot for the total purchase made by languages',color='red')

for i, v in enumerate(lang_price_sum):

    plt.text(v, i , str(round(v, 2)), color='red')

plt.savefig("lang_vs_purchase.png")

plt.show()
lang_broswer = ecom.groupby('language')['browser_name'].value_counts()
fig, ax = plt.subplots(figsize=(10,7))

lang_broswer.unstack().plot(kind='bar', title='Plot of Browsers used in diffrent Languages',ax=ax)

plt.savefig("browser_vs_lang.png")

plt.show()
lang_period = ecom.groupby('language')['am_or_pm'].value_counts()

lang_period.unstack()
os = set(ecom['os'])

os
display(ecom.os.value_counts())
plt.pie(ecom.os.value_counts(), autopct = '%.1f%%', radius = 1.2, labels = ['Windows', 'Mac','Linux'])

plt.title("Pie Chart of OS Used", pad=20)

plt.savefig("os_pie.png")

plt.show()
browser = ecom['browser_name'].unique()

browser
os_broswer = ecom.groupby('os')['browser_name'].value_counts()
fig, ax = plt.subplots(figsize=(10,7))

os_broswer.unstack().plot(kind='barh', title='Browser count used in diffrent OS',ax=ax)

plt.savefig("os_vs_browser.png")

plt.show()
os_broswer_price = ecom.groupby(['os', 'browser_name'])['purchase_price'].sum()

display(os_broswer_price)
display(ecom['job'])
job_sort = ecom.sort_values(by='purchase_price')

job_sort[['job', 'email', 'purchase_price']].head(10)
job_sort[['job', 'email', 'purchase_price']].tail(10)
jobs = ecom['job'].unique()

len(jobs)
job_purchase = ecom.groupby('job')['purchase_price'].sum()

display(job_purchase)
job_purchase_sort = job_purchase.sort_values(ascending=False)
top_10_purchase_job = job_purchase_sort.head(10)

display(top_10_purchase_job)
bottom_10_purchase_job = job_purchase_sort.tail(10)

display(bottom_10_purchase_job)
sns.distplot(ecom.purchase_price, kde=True, rug=True,bins=20)

plt.title("Distribution plot of Purchase Price")

plt.savefig("price_distribution.png")

plt.show()
job_am_pm = pd.pivot_table(ecom, index='job', columns='am_or_pm', values='purchase_price')

job_am_pm
job_am_pm.sort_values(by='AM', ascending=False).head(10)
job_am_pm.sort_values(by='PM', ascending=False).head(10)
period_purchase = ecom.groupby('am_or_pm')['purchase_price'].sum()
period_purchase.plot(kind='pie', autopct = '%.2f%%', title="Total amount Purchase in the Period(AM/PM)")

plt.savefig("am_pm_price.png")

plt.show()
set(ecom['cc_provider'])
plt.pie(ecom['cc_provider'].value_counts(), labels=set(ecom['cc_provider']), autopct="%1.2f%%")

plt.title("Distibution of Credit Cards Used")

plt.savefig("card_distribution.png")

plt.show()
ecom[ecom.duplicated("credit_card")]
cc_purchase = ecom.groupby('cc_provider').sum()
plt.figure(figsize=(20,10))

plt.bar(x=cc_purchase.index, height = cc_purchase.purchase_price)

plt.title("Total Purchase amout amoung Credit Card Providers")

plt.savefig("card_price.png")

plt.show()
email_purchase = ecom.groupby('email').sum()
len(email_purchase)
top_customer = email_purchase.sort_values('purchase_price', ascending=False)
display(top_customer.purchase_price.head(10))
customer_dup = ecom[ecom.duplicated('email', keep=False)]
customer_dup.shape
display(customer_dup['email'])
email_split = ecom['email'].str.split('@')

domain_list = []

for email in email_split:

    domain_list.append(email[1].split('.')[0])

    

ecom['domain'] = domain_list

domain = ecom['domain'].value_counts()
len(domain)
display(domain.head(10))
len(domain[domain == 1])
top3 = domain[:3]

others = domain[4:].sum()

email_dist = top3.append(pd.Series([others], index=['others']))

display(email_dist)
plt.pie(email_dist, labels=email_dist.index, autopct="%1.2f%%")

plt.title("Email Domians")

plt.savefig("email_distribution.png")

plt.show()