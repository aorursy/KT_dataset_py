import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()
import seaborn as sns
%matplotlib inline
df = pd.read_csv('../input/BlackFriday.csv')
df.head()
users = df[['User_ID', 'Gender', 'Age', 'Occupation', 'City_Category', 'Stay_In_Current_City_Years', 'Marital_Status']].copy()
users = users.drop_duplicates().reset_index(drop=True)
users['Purchase'] = pd.DataFrame(df.groupby(by= 'User_ID')['Purchase'].sum()).reset_index(drop=True)
users.head()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (11, 6))

ax1.pie(users['Gender'].value_counts(), autopct='%1.1f%%')
ax1.set_title('Number males and females')

ax2.pie(users.groupby('Gender')['Purchase'].sum().sort_values(ascending = False), autopct='%1.1f%%')
ax2.set_title('Purchase amount by males and females')

fig.legend(('Male', 'Female'))

plt.tight_layout()
sns.set_style('whitegrid')
plt.subplots(figsize=(12,5))
sns.countplot(users['Age']).set_title('Number of customers from different age groups')
sns.despine()
fig, ax = plt.subplots(figsize = (12, 5))

N = len(users['Age'].unique())
ind = np.arange(N)
width = 0.35

male = users[users['Gender'] == 'M']['Age'].value_counts()
female = users[users['Gender'] == 'F']['Age'].value_counts()

p1 = plt.bar(ind, male, width)
p2 = plt.bar(ind, female, width, bottom = male)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.set_title('Number of customers in different age groups by gender')
plt.ylabel('Count')
plt.xticks(ind, users[users['Gender'] == 'F']['Age'].value_counts().index)
plt.yticks(np.arange(0, 2400, 300))
plt.legend((p1[0], p2[0]), ('Male', 'Female'))
plt.show()
age_purchase = users.groupby('Age').sum()['Purchase']

trace = [go.Pie(
    values = age_purchase,
    labels = age_purchase.index)]

layout = go.Layout(title = 'Purchase amount in different age groups')

fig = go.Figure(data = trace, layout=layout)
iplot(fig)
occ_sum = users.groupby('Occupation').sum()['Purchase']
occ_count = users['Occupation'].value_counts()

fig, (ax1, ax2) = plt.subplots(1,2, figsize = (16, 9))

ax1.barh(occ_count.index, occ_count)
ax1.invert_yaxis()
ax1.set_yticks(np.arange(0, 21, 1))
ax1.set_title('Number of customers in different occupations')

ax2.barh(occ_sum.index, occ_sum)
ax2.invert_xaxis()
ax2.invert_yaxis()
ax2.yaxis.tick_right()
ax2.set_title('Purchased amount by customers in different occupations')

plt.yticks(np.arange(0, 21, 1))
plt.show()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (11, 6))

explode = (0, 0.1, 0.1)

ax1.pie(users['City_Category'].value_counts(), autopct='%1.1f%%', explode=explode, shadow=True)
ax1.set_title('Number people from different cities')

ax2.pie(users.groupby('City_Category')['Purchase'].sum().sort_values(ascending = False), autopct='%1.1f%%', explode=explode, shadow=True)
ax2.set_title('Purchase amount by city categories')

fig.legend(('C', 'B', 'A'))

plt.tight_layout()
plt.subplots(figsize = (12, 5))

sns.countplot(users['Stay_In_Current_City_Years']).set_title('Purchases by stay in current city')
sns.despine()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (11, 5))

ax1.pie(users['Marital_Status'].value_counts(), autopct='%1.1f%%')
ax1.set_title('Number of married and unmarried customers')

ax2.pie(users.groupby('Marital_Status')['Purchase'].sum(), autopct='%1.1f%%')
ax2.set_title('Amount purchased by marital status')

fig.legend(('Unmarried', 'Married'))

plt.show()
prod = df[['Product_ID', 'Product_Category_1', 'Product_Category_2', 'Product_Category_3', 'Purchase']].copy()
prod.head()
fig, ax = plt.subplots(figsize = (12,5))
ax.set_title('Product category 1')

prod.groupby('Product_Category_1')['Purchase'].count().sort_values().plot('barh')
plt.show()
fig, ax = plt.subplots(figsize = (12,5))
ax.set_title('Product category 2')

prod.groupby('Product_Category_2')['Purchase'].count().sort_values().plot('barh')
plt.show()
fig, ax = plt.subplots(figsize = (12,5))
ax.set_title('Product category 3')

prod.groupby('Product_Category_3')['Purchase'].count().sort_values().plot('barh')
plt.show()
fig, ax = plt.subplots(figsize=(12, 5))
ax.set_title('Top 10 items sold')

prod.groupby('Product_ID')['Purchase'].count().nlargest(10).sort_values().plot('barh')
plt.show()