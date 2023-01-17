# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        creditcard_dataset_path = os.path.join(dirname, filename)

        creditcard_data = pd.read_csv(creditcard_dataset_path)

        

# Any results you write to the current directory are saved as output.

print("Number of transactions: ", len(creditcard_data))

creditcard_data.tail()
plt.figure(figsize=(18,6))

sorted_creditcard_data = creditcard_data.sort_values(by='Amount', ascending=True)

# sns.lineplot(data=sorted_creditcard_data['Amount'], label = "Amount")

# plot = sns.lineplot(data = 500 * sorted_creditcard_data['Class'], label = "IsFraud")



# plot.set(xlabel ='Transactions', ylabel ='Amount')

# plt.show()

 

# plot of 2 variables

p1=sns.kdeplot(sorted_creditcard_data['Amount'], shade=True, color="r")

# p1=sns.kdeplot(2000*sorted_creditcard_data['Class'], shade=True, color="b")

fraudulent_amount = 0

number_of_fraudulent_transactions = 0;

for record in creditcard_data.itertuples():

    if record.Class == True:

        fraudulent_amount += record.Amount

        number_of_fraudulent_transactions += 1

print('The total amount of fraudulent transactions: ', fraudulent_amount)    

print('The total number of fraudulent transactions: ', number_of_fraudulent_transactions)  

fraudulent_mean = fraudulent_amount / number_of_fraudulent_transactions

        

nonfraudulent_amount = 0

number_of_nonfraudulent_transactions = 0

for record in creditcard_data.itertuples():

    if record.Class == False:

        nonfraudulent_amount += record.Amount

        number_of_nonfraudulent_transactions += 1

print('The total amount of non-fraudulent transactions: ', nonfraudulent_amount)    

print('The total number of fraudulent transactions: ', number_of_nonfraudulent_transactions)  

nonfraudulent_mean = nonfraudulent_amount / number_of_nonfraudulent_transactions

        

d = {'Class':[0, 1], 'Mean':[nonfraudulent_mean, fraudulent_mean]}



fig, ax = plt.subplots(figsize=(18,8))



ax.pie(d["Mean"],

       explode=[0, 0.1],

       labels=['The average amount of a non-fraudulent transaction', 'The average amount of a fraudulent transaction'],

       shadow=True, startangle=90)



ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()
correlated_data = creditcard_data.corr()



plt.figure(figsize=(18,12))

sns.heatmap(correlated_data, annot=False)
nonfraudulent_data = creditcard_data[creditcard_data["Class"] == 0]

correlated_data = nonfraudulent_data.corr()



plt.figure(figsize=(18,12))

sns.heatmap(correlated_data, annot=False)
fraudulent_data = creditcard_data[creditcard_data["Class"] == 1]

correlated_data = fraudulent_data.corr()



plt.figure(figsize=(18,12))

sns.heatmap(correlated_data, annot=False)
plt.figure(figsize=(18,12))

sns.scatterplot(x=creditcard_data["Time"], y=creditcard_data["Amount"], hue=creditcard_data["Class"], size=creditcard_data["Class"], sizes=(40, 8), marker="+")

plt.figure(figsize=(18,12))

plot = sns.distplot(a=creditcard_data["Time"], kde=True, color='purple')

plot.set(xlabel ='Time', ylabel ='Frequency')

plt.show()
fraudulent_transactions = creditcard_data[creditcard_data['Class'] == 1]

nonfraudulent_transactions = creditcard_data[creditcard_data['Class'] == 0]



plt.figure(figsize=(18,12))



sns.distplot(a=nonfraudulent_transactions["Time"], kde=True)

plot = sns.distplot(a=fraudulent_transactions["Time"], kde=True)



plot.set(xlabel ='Time', ylabel ='Frequency')

plot.legend(['Not Fraud', 'Fraud'])

plt.show()
plt.figure(figsize=(18,12))



nonfraudulent_transactions = creditcard_data[creditcard_data['Class'] == 0]

plot = sns.kdeplot(data=nonfraudulent_transactions["Amount"], label="Not Fraud", shade=True)

plot.set(xlabel ='Amount', ylabel ='Density')

plt.show()
plt.figure(figsize=(18,12))



fraudulent_transactions = creditcard_data[creditcard_data['Class'] == 1]

plot = sns.kdeplot(data=fraudulent_transactions["Amount"], label="Fraud", shade=True, color='orange')

plot.set(xlabel ='Amount', ylabel ='Density')

plt.show()
# plt.figure(figsize=(18,8))

# cmap = sns.cubehelix_palette(light=1, as_cmap=True) 

# sns.jointplot(x=nonfraudulent_transactions["Time"], y=nonfraudulent_transactions["Amount"], cmap=cmap, kind="kde")
plt.figure(figsize=(18,8))



fraudulent_transactions = creditcard_data[creditcard_data['Class'] == 1]



cmap = sns.cubehelix_palette(light=1, as_cmap=True) 

sns.jointplot(x=fraudulent_transactions["Time"], y=fraudulent_transactions["Amount"], cmap=cmap, kind="kde")
fraudulent_transactions = creditcard_data[creditcard_data['Class'] == 1]

# v1_matched_df = []

# for record in fraudulent_transactions.itertuples():

#     v1 = record.V1

#     result = fraudulent_transactions.loc[fraudulent_transactions["V1"] == v1]

#     if len(result.() != 1:

#         v1_matched_df.append(record)

#         v1_matched_df.append(fraudulent_transactions.loc[fraudulent_transactions["V1"] == v1])

# v1_matched_df

plt.figure(figsize=(18,8))



v1_grouped_fraudulent = fraudulent_transactions[["Time", "Amount"]].groupby("Time", as_index = False).agg("mean")

v1_grouped_fraudulent

# sns.distplot(a=v1_grouped_fraudulent["V1"])



# v2_grouped_fraudulent = fraudulent_transactions.groupby("V2", as_index = False)["Amount"].mean()



# sns.distplot(a=v2_grouped_fraudulent["V2"])



# v3_grouped_fraudulent = fraudulent_transactions.groupby("V3", as_index = False)["Amount"].mean()



# sns.distplot(a=v3_grouped_fraudulent["V3"])
plt.figure(figsize=(18,8))



nonfraudulent_data = creditcard_data[creditcard_data["Class"] == 0]

sns.distplot(a=nonfraudulent_data["V7"], kde="True")



fraudulent_data = creditcard_data[creditcard_data["Class"] == 1]

sns.distplot(a=fraudulent_data["V7"], kde="True")
plt.figure(figsize=(18,8))



nonfraudulent_data = creditcard_data[creditcard_data["Class"] == 0]

sns.distplot(a=nonfraudulent_data["V2"], kde="True")



fraudulent_data = creditcard_data[creditcard_data["Class"] == 1]

sns.distplot(a=fraudulent_data["V2"], kde="True")
plt.figure(figsize=(18,8))



nonfraudulent_data = creditcard_data[creditcard_data["Class"] == 0]

sns.distplot(a=nonfraudulent_data["V3"], kde="True")



fraudulent_data = creditcard_data[creditcard_data["Class"] == 1]

sns.distplot(a=fraudulent_data["V3"], kde="True")
plt.figure(figsize=(18,8))



nonfraudulent_data = creditcard_data[creditcard_data["Class"] == 0]

sns.distplot(a=nonfraudulent_data["V4"], kde="True")



fraudulent_data = creditcard_data[creditcard_data["Class"] == 1]

sns.distplot(a=fraudulent_data["V4"], kde="True")
plt.figure(figsize=(18,8))



nonfraudulent_data = creditcard_data[creditcard_data["Class"] == 0]

sns.distplot(a=nonfraudulent_data["V9"], kde="True")



fraudulent_data = creditcard_data[creditcard_data["Class"] == 1]

sns.distplot(a=fraudulent_data["V9"], kde="True")
plt.figure(figsize=(18,8))



nonfraudulent_data = creditcard_data[creditcard_data["Class"] == 0]

sns.distplot(a=nonfraudulent_data["V10"], kde="True")



fraudulent_data = creditcard_data[creditcard_data["Class"] == 1]

sns.distplot(a=fraudulent_data["V10"], kde="True")
plt.figure(figsize=(18,8))



nonfraudulent_data = creditcard_data[creditcard_data["Class"] == 0]

sns.distplot(a=nonfraudulent_data["V11"], kde="True")



fraudulent_data = creditcard_data[creditcard_data["Class"] == 1]

sns.distplot(a=fraudulent_data["V11"], kde="True")
plt.figure(figsize=(18,8))



nonfraudulent_data = creditcard_data[creditcard_data["Class"] == 0]

sns.distplot(a=nonfraudulent_data["V12"], kde="True")



fraudulent_data = creditcard_data[creditcard_data["Class"] == 1]

sns.distplot(a=fraudulent_data["V12"], kde="True")
plt.figure(figsize=(18,8))



nonfraudulent_data = creditcard_data[creditcard_data["Class"] == 0]

sns.distplot(a=nonfraudulent_data["V16"], kde="True")

sns.distplot(a=nonfraudulent_data["V17"], kde="True")

sns.distplot(a=nonfraudulent_data["V18"], kde="True")
plt.figure(figsize=(18,8))



fraudulent_data = creditcard_data[creditcard_data["Class"] == 1]

sns.distplot(a=fraudulent_data["V16"], kde="True")

sns.distplot(a=fraudulent_data["V17"], kde="True")

sns.distplot(a=fraudulent_data["V18"], kde="True")