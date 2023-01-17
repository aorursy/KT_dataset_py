# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("udemy_courses.csv")
df.head()
df.info()
df.shape
sns.set()
plt.figure(figsize=(8,5))
sns.countplot(y= df['subject'],data=df)
plt.title("Subjects in Udemy")
plt.show()
plt.figure(figsize=(8,5))
sns.countplot(y=df['level'], data=df)
plt.title("Level wise course Distribution")
plt.show()
df.loc[df.num_subscribers.idxmax()]
df.drop(['course_id','url'], axis=1, inplace=True)
df['is_paid'].value_counts()
df['is_paid'] = df['is_paid'].replace('https://www.udemy.com/learnguitartoworship/',
                                      'True')
df.replace(to_replace = 'TRUE', value = 'True', inplace = True)
df.replace(to_replace = 'FALSE', value = 'False', inplace = True)
df['is_paid'].value_counts()
x = df['is_paid']
plt.figure(figsize=(8,5))
sns.countplot(x)
plt.show()
df.price.value_counts()

#Step 1: Converting Free
df.price = df.price.replace('Free', 0)

#Step 2: Delete the 1 row where price is = TRUE
that_one_element = df[df.price == 'True'].index
df.drop(that_one_element, inplace = True, axis = 0)

#Step 3: Convert column to integer
df.price = pd.to_numeric(df['price'])
plt.figure()
plt.subplots(figsize=(14,2))
sns.boxplot(x=df['price'], data=df, color='gray')
plt.title("Price of courses")
plt.show()

plt.figure()
plt.subplots(figsize=(14,2))
sns.boxplot(x=df['price'], y=df['subject'])
plt.title("Price Variation for each subject")
plt.show()
x = df['price']
y = df['num_subscribers']
sns.set()
plt.figure(figsize=(8,5))
sns.scatterplot(x,y)
plt.show()
x = df['price']
y = df['num_reviews']
sns.set()
plt.figure(figsize=(8,5))
sns.scatterplot(x,y)
plt.title("Does Price affect number of reviews?")
plt.show()
plt.figure(figsize=(8,5))
sns.barplot(x=df['is_paid'], y=df['num_subscribers'])
plt.title("Which has more number of subscribers?")
plt.show()
plt.figure(figsize=(10,5))
y=df['num_subscribers']
x=df['level']
sns.barplot(x,y)
plt.title("Which level has most subscribers?")
plt.show()
df.groupby(['is_paid']).mean()