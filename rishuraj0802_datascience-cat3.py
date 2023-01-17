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
df = pd.read_csv("/kaggle/input/okcupid-profiles/okcupid_profiles.csv")

print(df.head())
ok_pr = df.iloc[:,0:21]

total_count = ok_pr.isna().count()

count_na = ok_pr.isna().sum().sort_values(ascending = False)

pct_na =(ok_pr.isna().sum()/total_count).sort_values(ascending = False)

pd.options.display.float_format = '{:.1%}'.format

missing_data = pd.concat([count_na, pct_na], axis=1, keys=['Count', 'Percent'])

# Only show columns with more than 5% missing data

missing_data[missing_data['Percent']>=0.05].head(20)
import matplotlib.pyplot as plt

%matplotlib inline

plt.figure(figsize=(12, 5))

ok_pr.isna().sum().plot(kind="bar")

plt.xticks(rotation=50)
df1=df["sex"].value_counts()

df1

pd.value_counts(df['sex']).plot.bar()
df2=df["age"].value_counts()

df2
pd.value_counts(df['age']).plot.bar()
df["status"].value_counts()
pd.value_counts(df["status"]).plot.bar()
by_gender = df.groupby('sex')

by_gender.size().plot(kind='bar')
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
plt.figure(figsize=(15, 7))

sns.catplot(x='age', data = ok_pr, kind='count', palette='pastel', hue='sex', 

            height=5, # make the plot 5 units high

            aspect=2)

plt.xticks(rotation = 25)

plt.title('Age distribution')

plt.show()
def show_percetage(plot, feature):

    total = len(feature)

    for p in ax.patches:

        percentage = '{:.1f}%'.format(100 * p.get_height()/total)

        x = p.get_x()+p.get_width()/2-0.1

        y = p.get_y()+p.get_height()

        ax.annotate(percentage, (x, y), size=10)
plt.figure(figsize=(9, 5))

ax = sns.countplot(x='orientation', data=ok_pr, 

                   hue='sex', 

                   palette='PRGn',

                   order=ok_pr['orientation'].value_counts().iloc[:10].index) 

total = float(len(ok_pr))

show_percetage(ax, ok_pr)

plt.show()
plt.figure(figsize=(9, 5))

ax = sns.countplot(x='body_type', data=ok_pr, 

                   hue='sex', 

                   palette='rocket',

                   order=ok_pr['body_type'].value_counts().iloc[:10].index)

ax.set_title("Body type by gender")

ax.set(xlabel='Count', ylabel='Body Type') 

show_percetage(ax, ok_pr)

plt.show()
plt.figure(figsize=(9, 5))

pr = sns.countplot(x='diet', data=ok_pr, 

                   hue='sex', palette='Paired',

                   order = ok_pr['diet'].value_counts().iloc[:10].index) 

plt.xticks(rotation = 30)

pr.set_title("Diet by gender")

show_percetage(pr, ok_pr)
plt.figure(figsize=(9, 5))

sns.countplot(y='job', data=ok_pr, 

              hue='sex', palette='Reds',

              order = ok_pr['job'].value_counts().iloc[:10].index)

plt.show()
plt.figure(figsize=(9, 5))

sns.countplot(y = 'pets', data=ok_pr, 

              hue='sex', palette='PuRd',

              order=ok_pr['pets'].value_counts().iloc[:].index)

plt.show()