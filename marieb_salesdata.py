# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas_profiling 


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
sales = pd.read_csv('../input/supermarket-sales/supermarket_sales - Sheet1.csv')
sales.shape
print(sales.dtypes)
sales.info()
sales.head()
sales.tail()
sales.columns = ['Invoice_ID', 'Branch', 'City', 'Customer_Types', 'Gender', 'Product_Line', 'Unit_Price', 'Quantity', 'Tax', 'Total', 'Date', 'Time', 'Payment', 'cogs', 'Gross_Margin_Percentage', 'Gross_Income','Rating']

sales.describe()
sales = sales.drop(['Gross_Income'], axis=1)
members = sales.groupby("Customer_Types")["Total"].count()
print(members)
totalSales = sales.groupby("Customer_Types")["Total"].sum()
print(totalSales)
salesPercent = (totalSales/members)
print(salesPercent)
memberSales = sales[sales.Customer_Types == 'Member']
salesType = memberSales.groupby('Product_Line')['Customer_Types'].count()
print(salesType)
nonMemberSales = sales[sales.Customer_Types == 'Normal']
salesTypeNonMember = nonMemberSales.groupby('Product_Line')['Customer_Types'].count()
print(salesTypeNonMember)

salesMemberBranch = memberSales.groupby('Branch')['Customer_Types'].count()
print(salesMemberBranch)

salesNonMemberBranch = nonMemberSales.groupby('Branch')['Customer_Types'].count()
print(salesNonMemberBranch)
sales.groupby("Gender").sum().sort_values("Total", ascending=False)
sales.groupby("Gender")["Total"].sum()
#print(salesGender)
sales.nlargest(10, 'Total')

