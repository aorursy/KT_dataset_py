# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sn

import matplotlib.pyplot as plt

from sklearn import tree

from sklearn import linear_model



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv ('../input/anz-synthesised-transaction-dataset/anz.csv')

df.head(5)
# check numerical data statistics

df.describe()
# check data types

df.dtypes
# check missing value ratio

df.isnull().sum() / len(df)
# check number of customers. Assume each customer has one unique customer_id.

print("number of customer: ", len(df.customer_id.unique()))
print("first day of data: ", df.date.iloc[0])

print("last day of data: ", df.date.iloc[-1])

print("duration: ", 92)

print("recorded days: ", len(df.date.unique()))
# list all unique customer ids

customer_list = df.customer_id.unique()

# filter out useless information

df_cus_info = pd.DataFrame(columns = ["customer_id", "annual salary", "age", "avg_transaction_amount", "transaction_number", "max_transaction_amount", "avg_balance", "gender", "state"])



for index, id in enumerate(customer_list):

    # extract payment information of this customer 

    df_cus = df[(df.customer_id == id) & (df.txn_description == 'PAY/SALARY')]

    # calculate annual salary

    pay_period = pd.to_datetime(df_cus.date).diff().mean().total_seconds() / 60 / 60 / 24

    pay_amount = df_cus.amount.mean()

    daily_pay = pay_amount / pay_period

    yearly_pay = 365 * daily_pay

    # get age

    age = df_cus.age.mean()

    # get average balance

    balance = df_cus.balance.mean()

    # get gender

    gender = df_cus["gender"].mode()[0]

    # store all the payment related info in the dataframe

    df_cus_info.loc[index, ["customer_id", "annual salary", "age", "avg_balance", "gender"]] = [id, yearly_pay, age, balance, gender]
# iterate through each customer

for index, id in enumerate(customer_list):

    # extract all info of this customer

    df_cus = df[df.customer_id == id]

    # assume mode of transaction merchant state is the state of this customer                     

    state = df_cus["merchant_state"].mode()[0]

    # calculate average transaction amount of this customer

    avg_transaction_amount = df_cus["amount"].mean()

    # calculate the number of transaction during a certain time of period

    transaction_number = df_cus["transaction_id"].count() 

    # calculate the max transaction amount during a certain time of period 

    max_transaction_amount = df_cus["amount"].max() 

    # put all calculted results above in the data frame

    df_cus_info.loc[index, ["state", "avg_transaction_amount", "transaction_number", "max_transaction_amount"]] = [state, avg_transaction_amount, transaction_number, max_transaction_amount]
# transform the data type

df_cus_info["annual salary"] = df_cus_info["annual salary"].astype(float)

df_cus_info["age"] = df_cus_info["age"].astype(float)

df_cus_info["avg_transaction_amount"] = df_cus_info["avg_transaction_amount"].astype(float)

df_cus_info["transaction_number"] = df_cus_info["transaction_number"].astype(float)

df_cus_info["max_transaction_amount"] = df_cus_info["max_transaction_amount"].astype(float)

df_cus_info["avg_balance"] = df_cus_info["avg_balance"].astype(float)

df_cus_info.dtypes
# calculate correlation matrix

corrMatrix = df_cus_info.loc[:, ["annual salary", "age", "avg_balance", "avg_transaction_amount", "transaction_number", "max_transaction_amount"]].astype('float64').corr(method='pearson', min_periods=1)

corrMatrix

sn.heatmap(corrMatrix, annot=True)

plt.show()
X = df_cus_info.drop(labels=["customer_id", "annual salary"], axis=1)

X_OHE = pd.get_dummies(X, columns=["state", "gender"])

Y = df_cus_info["annual salary"].apply(lambda x: 1 if x >  60000 else 0)

clf = tree.DecisionTreeClassifier(max_depth=2)

clf = clf.fit(X_OHE, Y)

tree.plot_tree(clf) 



print()
# df_cus_info

X_rgr = df_cus_info[["age", "avg_transaction_amount", "transaction_number", "max_transaction_amount", "avg_balance"]]

Y_rgr = df_cus_info["annual salary"]
# build linear regression model

rgr = linear_model.LinearRegression()

# fit model

rgr.fit(X_rgr, Y_rgr)

# coefficient of determination R^2

print(rgr.score(X_rgr, Y_rgr))
# plot regression coefficients

names = ["age", "avg_transaction_amount", "transaction_number", "max_transaction_amount", "avg_balance"]

values = rgr.coef_

plt.figure(figsize=(16,8))

plt.bar(names, values)
# build linear regression model

rgr_elastic = linear_model.ElasticNet(random_state=0, l1_ratio=0.5)

# fit model

rgr_elastic.fit(X_rgr, Y_rgr)

# coefficient of determination R^2

print(rgr_elastic.score(X_rgr, Y_rgr))
# plot regression coefficients

names = ["age", "avg_transaction_amount", "transaction_number", "max_transaction_amount", "avg_balance"]

values = rgr_elastic.coef_

plt.figure(figsize=(16,8))

plt.bar(names, values)
# plot

plt.figure(figsize=(8, 5))

plt.hist(df_cus_info.transaction_number)

plt.xlabel('Transaction Quantity')

plt.ylabel("Frequency")

plt.title('Histogram of Transaction Quantity')

plt.show()
plt.figure(figsize=(8, 5))

plt.hist(df_cus_info.avg_transaction_amount)

plt.xlabel('Transaction Amount')

plt.ylabel("Frequency")

plt.title('Histogram of Transaction Amount')

plt.show()
plt.figure(figsize=(8, 5))

plt.hist(df_cus_info["annual salary"])

plt.xlabel('Annual Salary')

plt.ylabel("Frequency")

plt.title('Histogram of Annual Salary')

plt.show()