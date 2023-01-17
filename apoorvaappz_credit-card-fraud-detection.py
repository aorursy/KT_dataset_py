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
df = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')

df.head()
df.shape
df.columns
df.describe()
df[["Time","Amount","Class"]].describe()
import matplotlib.pyplot as plt

import seaborn as sns



print("Distribuition of Normal(0) and Frauds(1): ")

print(df["Class"].value_counts())



plt.figure(figsize=(7,5))

sns.countplot(df['Class'])

plt.title("Class Count", fontsize=18)

plt.xlabel("Is fraud?", fontsize=15)

plt.ylabel("Count", fontsize=15)

plt.show()
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))

s = sns.boxplot(ax = ax1, x="Class", y="Amount", hue="Class",data=df, palette="PRGn",showfliers=True)

s = sns.boxplot(ax = ax2, x="Class", y="Amount", hue="Class",data=df, palette="PRGn",showfliers=False)

plt.show();

tmp = df[['Amount','Class']].copy()

class_0 = tmp.loc[tmp['Class'] == 0]['Amount']

class_1 = tmp.loc[tmp['Class'] == 1]['Amount']

class_0.describe()
class_1.describe()
import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot



fraud = df.loc[df['Class'] == 1]



trace = go.Scatter(

    x = fraud['Time'],y = fraud['Amount'],

    name="Amount",

     marker=dict(

                color='rgb(238,23,11)',

                line=dict(

                    color='blue',

                    width=1),

                opacity=0.5,

            ),

    text= fraud['Amount'],

    mode = "markers"

)

data = [trace]

layout = dict(title = 'Amount of fraudulent transactions',

          xaxis = dict(title = 'Time [s]', showticklabels=True), 

          yaxis = dict(title = 'Amount'),

          hovermode='closest'

         )

fig = dict(data=data, layout=layout)

iplot(fig, filename='fraud-amount')
Fraud = df[df['Class']==1]



Normal = df[df['Class']==0]



f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

f.suptitle('Amount per transaction by class')

bins = 50

ax1.hist(Fraud.Amount, bins = bins , color ='red')

ax1.set_title('Fraud')

ax2.hist(Normal.Amount, bins = bins, color = 'green')

ax2.set_title('Normal')

plt.xlabel('Amount ($)')

plt.ylabel('Number of Transactions')

plt.xlim((0, 20000))

plt.yscale('log')

plt.show();
fig, ax = plt.subplots(1, 2, figsize=(18,4))



amount_val = df['Amount'].values

time_val = df['Time'].values



sns.distplot(amount_val, ax=ax[0], color='r')

ax[0].set_title('Distribution of Transaction Amount', fontsize=14)

ax[0].set_xlim([min(amount_val), max(amount_val)])



sns.distplot(time_val, ax=ax[1], color='b')

ax[1].set_title('Distribution of Transaction Time', fontsize=14)

ax[1].set_xlim([min(time_val), max(time_val)])







plt.show()
timedelta = pd.to_timedelta(df['Time'], unit='s')

df['Time_hour'] = (timedelta.dt.components.hours).astype(int)



#Exploring the distribuition by Class types throught hours and minutes

plt.figure(figsize=(12,5))

sns.distplot(df[df['Class'] == 0]["Time_hour"], 

             color='g')

sns.distplot(df[df['Class'] == 1]["Time_hour"], 

             color='r')

plt.title('Fraud x Normal Transactions by Hours', fontsize=17)

plt.xlim([-1,25])

plt.show()
df.drop(['Time_hour'], axis=1, inplace=True)



df.hist(figsize=(20,20))

plt.show()
correlation_matrix = df.corr()

fig = plt.figure(figsize=(12,9))

sns.heatmap(correlation_matrix,vmax=0.8,square = True)

plt.show()
# Scaling

from sklearn.preprocessing import StandardScaler, RobustScaler



# RobustScaler is less prone to outliers.



std_scaler = StandardScaler()

rob_scaler = RobustScaler()



df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))

df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1))



df.drop(['Time','Amount'], axis=1, inplace=True)

scaled_amount = df['scaled_amount']

scaled_time = df['scaled_time']



df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)

df.insert(0, 'scaled_amount', scaled_amount)

df.insert(1, 'scaled_time', scaled_time)



# Amount and Time are Scaled!

df.head()
# df.isnull().sum()

df.isnull().values.any()