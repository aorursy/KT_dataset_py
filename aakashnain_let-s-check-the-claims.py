# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import warnings

warnings.filterwarnings('ignore')



import os

import glob

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Let's look at the data first

data = pd.read_csv('../input/tsa_claims.csv')

print("Number of samples in the data  : ", data.shape[0])

print("Columns in the dataset  : ", list(data.columns))

print(" ")

data.head()
# Let's have an overview of the dataset

data.info()
# Check for NaN or null values in the dataset

data.isnull().sum()
# We will start with Airport code column first

codes = data['Airport Code'].value_counts()

print("Total number of unique airport codes : ", len(codes))

print("Maximun number of times an airport has been reported : ", codes.values.max())

print("Airport code where maximum number of incidents happened : ", codes.index[codes.values == codes.values.max()].tolist()[0])

print("Least number of incidents that has happened on any airport : ", codes.values.min())

print("Airport code where least number of incidents has happened: ", codes.index[codes.values == codes.values.min()].tolist()[0])

print("Average number of incidents that happened over the period of time : ", int(codes.values.mean()))
# Get the names of the airports with minimum and maximum number of incidents happened

print("Airport name with maximum number of incidents : ", list(data['Airport Name'][data['Airport Code'] =='LAX'])[0])

print("Airport name with minimum number of incidents : ", list(data['Airport Name'][data['Airport Code'] =='ADK'])[0])
# Let's move to the claim type column

unique_claim_type = data['Claim Type'].value_counts()

print("Total number of different claims : ", len(unique_claim_type))
# Let's visualize these claims along with the numbers they have been reported

claim_index = unique_claim_type.index

claim_values = unique_claim_type.values



plt.figure(figsize=(20,10))

sns.barplot(y=claim_index, x=claim_values, orient='horizontal')

plt.xlabel('Claim type', fontsize=14)

plt.ylabel('Count', fontsize=14)

plt.xticks(fontsize=16)

plt.yticks(fontsize=16)

plt.show()
# Let's move to the claim site

unique_claim_sites = data['Claim Site'].value_counts()

print("Total number of unique claim sites : ", len(unique_claim_sites))
# Let's visualize the actual number of instances for each class of claim site

x = unique_claim_sites.index

y = unique_claim_sites.values



f = plt.figure(figsize=(20,10))

sns.barplot(x, y)

plt.xlabel('Claim site')

plt.ylabel('Count')

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.show()
# Let's move to the date columns for now. There are too many null values. Converting the column to datetime without

# removing them may produce undesirable results. Will be happy to see some good

# solution to it

data['Date Received'] = pd.to_datetime(data['Date Received'])

data['Received day'] = data['Date Received'].dt.weekday

data['Received month'] = data['Date Received'].dt.month
# Check the number of claims on different days

claims_count = data['Received day'].value_counts()



plt.figure(figsize=(20,10))

sns.barplot(claims_count.index, claims_count.values)

plt.xlabel('Day of the week')

plt.ylabel('Claims count')

plt.xticks(range(7), ['Sun', 'Mon', 'Tues', 'Wed', 'Thu', 'Fri', 'Sat'])

plt.yticks(fontsize=14)

plt.show()
claims_count = data['Received month'].value_counts()

months = ['Jan', 'Feb', 'March', 'April', 'May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']



plt.figure(figsize=(20,10))

sns.barplot(claims_count.index, claims_count.values)

plt.xlabel('Month')

plt.ylabel('Claims count')

plt.xticks(range(12), months, fontsize=14)

plt.yticks(fontsize=12)

plt.show()
def split_amount(x):

    try:

        if x is not None:

            a = x.split('$')[1]

            if ';' in a:

                b,c = a.split(';')

                return eval(b + c)

            return eval(a)

    except:

        return 0
data['Claim Amount'] = data['Claim Amount'].apply(split_amount)

print("Maximum amount claimed : ", data['Claim Amount'].max())

print("Average amount claimed : ", data['Claim Amount'].mean())
data['Close Amount'] = data['Close Amount'].apply(split_amount)

print("Maximum closed amount  : ", data['Close Amount'].max())

print("Average closed amount  : ", data['Close Amount'].mean())
# Let's move to the disposition column finally

dispos = data['Disposition'].value_counts()



plt.figure(figsize=(10,5))

sns.barplot(dispos.index, dispos.values)

plt.xlabel('Disposition')

plt.ylabel('Count')

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.show()