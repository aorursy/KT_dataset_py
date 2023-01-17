import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))



import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

%matplotlib inline
data = pd.read_csv('../input/BlackFriday.csv')

df = data.copy()
df.head()
df.info()
df.isnull().sum()
df['Product_Category_2'].fillna(0, inplace=True)

df['Product_Category_3'].fillna(0, inplace=True)
sns.set(style='ticks', palette='Paired')
df['Gender_bin'] = 0
def gender_bin(col):

    if col == 'M':

        return 1

    if col == 'F':

        return 0
df.loc[:,'Gender_bin'] = df['Gender'].apply(gender_bin)
plt.figure(figsize=(6, 6))

# data

male = df['Gender_bin'].sum()

female = df['Gender_bin'].count() - male

names = ['Males', 'Females']

size = [male, female]



# Create a circle for the center of the plot

my_circle = plt.Circle( (0,0), 0.7, color='white')



# Give color names

plt.pie(size, labels = names, autopct='%1.1f%%')

p = plt.gcf()

p.gca().add_artist(my_circle)

plt.show()
plt.figure(figsize=(15, 25))

plt.subplot(521)

sns.countplot(x='Gender', data=df);

plt.subplot(522)

sns.countplot(x='Age', data=df, order=['0-17', '18-25', '26-35', '36-45', '46-50', '51-55', '55+']);

plt.subplot(523)

sns.countplot(x='City_Category', data=df, order=['A', 'B', 'C']);

plt.subplot(524)

sns.countplot(x='Stay_In_Current_City_Years', data=df, order=['0', '1', '2', '3', '4+']);

plt.subplot(525)

sns.countplot(x='Marital_Status', data=df);
plt.figure(figsize=(15, 10))

plt.subplot(221)

sns.distplot(df['Purchase']);

plt.subplot(222)

sns.kdeplot(df['Purchase'], shade=True);
plt.figure(figsize=(14, 8))

plt.subplot(221)

sns.barplot(x='Gender', y='Purchase', data=df, ci=None);

plt.subplot(222)

sns.boxplot(y='Gender', x='Purchase', data=df);
plt.figure(figsize=(14, 8))

plt.subplot(221)

sns.barplot(x='Age', y='Purchase', data=df, order=['0-17', '18-25', '26-35', '36-45', '46-50', '51-55', '55+'], ci=None)

plt.subplot(222)

sns.boxplot(y='Age', x='Purchase', data=df);
plt.figure(figsize=(14, 8))

plt.subplot(221)

sns.barplot(x='City_Category', y='Purchase', data=df, order=['A', 'B', 'C'], ci=None);

plt.subplot(222)

sns.boxplot(y='City_Category', x='Purchase', data=df, order=['A', 'B', 'C']);

plt.figure(figsize=(14, 8))

plt.subplot(221)

sns.barplot(x='Stay_In_Current_City_Years', y='Purchase', data=df, ci=None, order=['0', '1', '2', '3', '4+']);

plt.subplot(222)

sns.boxplot(y='Stay_In_Current_City_Years', x='Purchase', data=df, order=['0', '1', '2', '3', '4+']);
plt.figure(figsize=(15, 10))

plt.subplot(221)

sns.distplot(df['Product_Category_1']);

plt.subplot(222)

sns.kdeplot(df['Product_Category_1'], shade=True);
plt.figure(figsize=(15, 10))

plt.subplot(221)

sns.distplot(df['Product_Category_2']);

plt.subplot(222)

sns.kdeplot(df['Product_Category_2'], shade=True);
plt.figure(figsize=(15, 10))

plt.subplot(221)

sns.distplot(df['Product_Category_3']);

plt.subplot(222)

sns.kdeplot(df['Product_Category_3'], shade=True);