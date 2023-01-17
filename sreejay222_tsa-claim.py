# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Let's look at the data first

df = pd.read_csv('../input/tsa_claims.csv')

df.head()
#Data Cleaning 

#Data Cleaning 

#1) Replace $ with space 

#2) Replaced - with nan

#3) Removed Null record from Claim Amount 

#4) Incuded new columns for Status as Up Status 

#   because for certain columns Claim Amount and Close Amount 

#   is there but the Status is Denied. Hence Updated the those Status as Approved



df["Claim Amount"] = df["Claim Amount"].str.replace("$","")

df["Close Amount"] = df["Close Amount"].str.replace("$","")

df["Claim Amount"] = df["Claim Amount"].str.replace(";","")

df["Close Amount"] = df["Close Amount"].str.replace(";","")

df = df.replace('-',np.nan)

df = df[pd.notnull(df['Claim Amount'])]

df1=[]

for each in df.itertuples():

    if each[11] == 'Denied' and (each[10] == each[12]):

        new = 'Approved'

        df1.append(new)

    else:

        new = each[11]

        df1.append(new)

df['Up_Status'] = df1
#Updating the Status

df['Up_Status'] = df['Up_Status'].replace('Deny', 'Denied')

df['Up_Status'] = df['Up_Status'].replace('Approve in Full', 'Approved')

df['Up_Status'] = df['Up_Status'].replace('Settle', 'Settled')

df['Up_Status'] = df['Up_Status'].replace(('Canceled','In review'), 'Canceled')

df['Up_Status'] = df['Up_Status'].replace(('Insufficient; one of the following items required: sum certain; statement of fact; signature; location of incident; and date.','Closed as a contractor claim','In litigation','Pending response from claimant','Claim has been assigned for further investigation'), 'other')



# Certain records where claim amount is there and status is there so updating the close amount

df2 = []

for each in df.itertuples():

    if pd.isnull(each[12]) and (each[11] == 'Deny' or each[11] == 'Approve in Full'):

        df2.append(each[10])

    elif pd.isnull(each[12]) and each[11] == 'Settle':

                            df2.append(each[12])

    else:

        df2.append(each[10])

df['New Claim Amount'] = df2
vc = df['Up_Status'].value_counts().head(20).plot(kind='bar')

plt.show()
vc = df['Disposition'].value_counts().head(10).plot(kind='bar',color = 'green')

plt.show()