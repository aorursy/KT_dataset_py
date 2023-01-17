import numpy as np 

import pandas as pd 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('darkgrid')

sns.set_palette('Set2')
df = pd.read_csv('/kaggle/input/passenger-list-for-the-estonia-ferry-disaster/estonia-passenger-list.csv')
df.shape
df.isnull().sum().to_frame(name='nulls').T
plt.figure(figsize=(20,4))



plt.subplot(1,4,1)

df['Country'].value_counts().plot.bar(title='country')

plt.subplot(1,4,2)

df['Sex'].value_counts().plot.bar(title='sex')

plt.subplot(1,4,3)

df['Category'].value_counts().plot.bar(title='category')

plt.subplot(1,4,4)

df['Survived'].value_counts().plot.bar(title='survived')
sub_df = pd.crosstab(df['Country'], df['Survived'])

sub_df['s_rate'] = sub_df[1] / (sub_df[0] + sub_df[1])

cm = sns.light_palette("green", as_cmap=True)

sub_df.style.background_gradient(cmap=cm)
sub_df = pd.crosstab(df['Sex'], df['Survived'])

sub_df['s_rate'] = sub_df[1] / (sub_df[0] + sub_df[1])

cm = sns.light_palette("red", as_cmap=True)

sub_df.style.background_gradient(cmap=cm)
sub_df = pd.crosstab(df['Category'], df['Survived'])

sub_df['s_rate'] = sub_df[1] / (sub_df[0] + sub_df[1])

cm = sns.light_palette("blue", as_cmap=True)

sub_df.style.background_gradient(cmap=cm)
sns.kdeplot(df[df['Survived']==0]['Age'], label='0')

sns.kdeplot(df[df['Survived']==1]['Age'], label='1')
df['age'] = pd.cut(df['Age'], [0,9,19,29,39,49,59,69,79,89,99], 

                  labels = ['0s','10s','20s','30s','40s','50s','60s','70s','80s','90s'])

sns.countplot(df['age'], hue=df['Survived'])