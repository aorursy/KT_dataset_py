import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
file_name = '/kaggle/input/kmpg-virtual-internship/KPMG_VI_New_raw_data_update_final.xlsx'

print(pd.ExcelFile(file_name).sheet_names)
Transactions = pd.read_excel(file_name, header=1, sheet_name='Transactions')
Transactions.head()
Transactions.shape
Transactions.info()
Transactions.online_order = Transactions.online_order.astype(str)

Transactions.transaction_id = Transactions.transaction_id.astype(str)

Transactions.product_id = Transactions.product_id.astype(str)

Transactions.customer_id = Transactions.customer_id.astype(str)
Transactions.describe()
Transactions.describe(include=[np.object])
Transactions['transaction_date'].describe(datetime_is_numeric=True)
Transactions.duplicated().sum()
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow , cols_index):

    nRow, nCol = df.shape

    subnum = 0

    columnNames = list(df)

    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow

    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')

    for i in cols_index:

        subnum += 1

        plt.subplot(nGraphRow, nGraphPerRow, subnum )

        columnDf = df.iloc[:, i]

        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):

            valueCounts = columnDf.value_counts()

            valueCounts.plot.bar()

        else:

            columnDf.hist()

        plt.ylabel('counts')

        plt.title(f'{columnNames[i]} (column {i})')

    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)

    plt.show();
plotPerColumnDistribution(Transactions, 9, 3 , [4,5,6,7,8,9,10,11,12])
CustomerDemographic = pd.read_excel(file_name, header=1, sheet_name='CustomerDemographic')

CustomerAddress = pd.read_excel(file_name, header=1, sheet_name='CustomerAddress')
pd.set_option('display.max_columns', None)

CustomerDemographic.head()
CustomerDemographic.shape
pd.set_option('display.max_columns', None)

CustomerAddress.head()
CustomerAddress.shape
Customers = CustomerDemographic.join(CustomerAddress.set_index('customer_id'), on='customer_id')
Customers.head()
Customers.drop(['default'], axis = 1, inplace=True)

Customers.info()
Customers.customer_id = Customers.customer_id.astype(str)

Customers.customer_id.dtypes
Customers.duplicated().sum()
Customers['postcode'] = pd.to_numeric(Customers.postcode, errors='coerce')

Customers.postcode
Customers.postcode = Customers.postcode.apply(lambda x: "{:.0f}".

                                          format(x) if not pd.isnull(x) else x)

Customers.postcode
Customers.gender.value_counts()
Customers.gender.replace('M', 'Male' , inplace=True)

Customers.gender.replace('F', 'Female' , inplace=True)

Customers.gender.replace('Femal', 'Female' , inplace=True)

Customers.gender.value_counts()
Customers.DOB.describe(datetime_is_numeric=True)
Customers.loc[Customers.query('DOB == DOB.min()').index , 'DOB'] = Customers.query('DOB == DOB.min()')['DOB']+ pd.DateOffset(years=100)

Customers.query('DOB == DOB.min()')['DOB']
Customers.deceased_indicator.value_counts()
Customers.deceased_indicator.replace('N', False , inplace=True)

Customers.deceased_indicator.replace('Y', True , inplace=True)

Customers.deceased_indicator.value_counts()
Customers.owns_car.value_counts()
Customers.owns_car.replace('No', False , inplace=True)

Customers.owns_car.replace('Yes', True , inplace=True)

Customers.owns_car.value_counts()
Customers.country.value_counts()
Customers.drop(['country'], axis=1, inplace=True)

Customers.info()
Customers.state.value_counts()
Customers.state.replace('New South Wales', 'NSW' , inplace=True)

Customers.state.replace('Victoria', 'VIC' , inplace=True)

Customers.state.value_counts()
Customers.sample(5)
Customers.shape
plotPerColumnDistribution(Customers, 9, 3 , [3,4,7,8,9,10,11,14,15])