import numpy as np # linear algebra
import pandas as pd
import seaborn as sns 
from matplotlib import pyplot as plt
%matplotlib inline                     

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
file_name = "../input/hotel-booking-demand/hotel_bookings.csv"
df = pd.read_csv(file_name)
df.sample(10)
df.info()
df.describe()
df.shape
df.isnull().sum()
print("The Total number of null values = ",df.isnull().sum().sum())
df.dtypes
df.columns
df.duplicated().sum()
df.drop_duplicates()
df.nunique()
cor= df.corr()
cor
df["children"].replace(np.nan, 0, inplace=True)
df['children'].unique()
df['agent'].mean()
df['country'].fillna(method='ffill',inplace=True)
df['country'].unique()
df.value_counts()
df['customer_type'].value_counts()
avg_agent = df['agent'].astype('float').mean(axis=0)
print("The Average Agent Column is given that ", avg_agent)
df['agent'].replace(np.nan, avg_agent, inplace=True)
print("Hotel :",df['hotel'].unique())
print("Customers : ",df['customer_type'].unique())
fig,axes = plt.subplots(1,1,figsize=(9,6))
sns.heatmap(cor, annot= True)
plt.show()
sns.countplot(x=df['is_canceled'])
sns.countplot(x=df['arrival_date_month'])
sns.countplot(x=df['lead_time'],order=(df['lead_time'].value_counts().head(20)).index)
plt.xticks(rotation=90)
plt.figure(figsize = (10,4))
sns.lineplot('agent', 'adr', data = df, color = 'r', label= 'agent')
plt.legend()

sns.pairplot(df)
canceled_data = df['is_canceled']
sns.countplot(canceled_data, palette='husl')

plt.show()
cols = ['Red', 'Blue']
df['is_canceled'].value_counts().plot.pie(autopct='%1.1f%%',shadow=True, colors=cols)
plt.figure(figsize = (18,8))
sns.set_style("whitegrid")
ax = sns.violinplot(x = 'arrival_date_month', y = 'lead_time' ,data=df)
ax.set_xlabel('Month', fontsize = 20)
ax.set_ylabel('Lead Time', fontsize = 20)
ax.set_title('Most Number of Lead Time', fontsize = 30)
plt.show()