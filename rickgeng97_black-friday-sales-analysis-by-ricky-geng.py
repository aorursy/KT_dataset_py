import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
bf_df = pd.read_csv("../input/BlackFriday.csv")
bf_df.head()
bf_df.info()
bf_df.isnull().sum()
bf_df.columns
bf_df['Gender'].unique()
plt.subplot(1,2,1)
sns.countplot(bf_df['Gender']) #attendance

m_purchase = bf_df.groupby(['Gender'])['Purchase'].sum()
plt.subplot(1,2,2)
sns.barplot(m_purchase.index, m_purchase.values) #dollar value
sns.countplot(bf_df['Marital_Status'])
sns.countplot(bf_df['Gender'], hue = bf_df['Marital_Status'])
sns.countplot(bf_df['Stay_In_Current_City_Years'])
pie = bf_df.groupby(['Product_Category_1']).count()['User_ID']
pie_sorted = pie.sort_values(ascending = False)
plt.figure(figsize = (10,8))
plt.pie(x = pie_sorted.values, labels = pie_sorted.index, autopct='%1.1f%%')
plt.axis('equal')
plt.show()
pie2 = bf_df.groupby(['Product_Category_2']).count()['User_ID']
pie2_sorted = pie2.sort_values(ascending = False)
plt.figure(figsize = (10,8))
plt.pie(x = pie2_sorted.values, labels = pie2_sorted.index, autopct='%1.1f%%')
plt.axis('equal')
plt.show()
pie3 = bf_df.groupby(['Product_Category_3']).count()['User_ID']
pie3_sorted = pie3.sort_values(ascending = False)

plt.figure(figsize = (10,8))
plt.pie(x = pie3_sorted.values, labels = pie3_sorted.index, autopct='%1.1f%%')
plt.axis('equal')
plt.show()
#combined Pie Chart

plt.figure(figsize = (20,8))

plt.subplot(1,3,1)
pie = bf_df.groupby(['Product_Category_1']).count()['User_ID']
pie_sorted = pie.sort_values(ascending = False)
plt.pie(x = pie_sorted.values, labels = pie_sorted.index, autopct='%1.1f%%')
plt.axis('equal')
plt.title('Product_Category_1')

plt.subplot(1,3,2)
pie2 = bf_df.groupby(['Product_Category_2']).count()['User_ID']
pie2_sorted = pie2.sort_values(ascending = False)
plt.pie(x = pie2_sorted.values, labels = pie2_sorted.index, autopct='%1.1f%%')
plt.axis('equal')
plt.title('Product_Category_2')

plt.subplot(1,3,3)
pie3 = bf_df.groupby(['Product_Category_3']).count()['User_ID']
pie3_sorted = pie3.sort_values(ascending = False)
plt.pie(x = pie3_sorted.values, labels = pie3_sorted.index, autopct='%1.1f%%')
plt.axis('equal')
plt.title('Product_Category_3')

plt.show()
plt.figure(figsize = (20,8))
sns.countplot(bf_df['Product_Category_1'], hue = bf_df['Gender'])
bf_df.head()
bf_df.columns
train_raw = bf_df[['Gender', 'Age', 'Occupation', 'City_Category', 'Stay_In_Current_City_Years', 'Marital_Status']]
test = bf_df['Product_Category_1']
train_raw['Stay_In_Current_City_Years'] = train_raw['Stay_In_Current_City_Years'].apply(lambda x: '4' if x == '4+' else x)
train_raw['Stay_In_Current_City_Years'] = train_raw['Stay_In_Current_City_Years'].apply(lambda x: int(x))
train_raw['City_Category'] = train_raw['City_Category'].apply(lambda x: 0 if x == 'A' else (1 if x == 'B' else 2))
train_raw['Age'] = train_raw['Age'].apply(lambda x: '00-17' if x == '0-17' else x)
train_raw['Age'] = train_raw['Age'].apply(lambda x: '56-60' if x == '55+' else x)
train_raw['Age'] = train_raw['Age'].apply(lambda x: 0.5 * (int(x[0])*10 + int(x[1]) + int(x[-2])*10 + int(x[-1])))
train_raw['Gender'] = train_raw['Gender'].apply({'M':1, 'F':0}.get)
train_raw.info()
train = train_raw.copy()
x_train, y_train, x_test, y_test = train_test_split(train, test, test_size=0.1, random_state=42)
tree = DecisionTreeClassifier()
tree.fit(x_train, x_test)
y_pred = tree.predict(y_train)
accuracy_score(y_test, y_pred)