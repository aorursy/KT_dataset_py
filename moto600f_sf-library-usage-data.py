import numpy as np

import pandas as pd

from pandas import Series,DataFrame

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



library = pd.read_csv('../input/Library_Usage.csv')

library.head()
library['Age Range'] = library['Age Range'].fillna('Not Defined')
age_ranges_order=['0 to 9 years','10 to 19 years','20 to 24 years','25 to 34 years',

            '35 to 44 years','45 to 54 years','55 to 59 years','60 to 64 years',

            '65 to 74 years','75 years and over','Not Defined']
plt.title('Age Range Destribution',fontsize=15)

sns.countplot(y='Age Range',

              data=library,

              order=age_ranges_order)
plt.figure(figsize=(5,10))

plt.title('Home Library Registration', fontsize=15)

plt.xticks(rotation=45)

sns.countplot(y='Home Library Definition', data=library)
library_oview = library.groupby('Age Range', as_index=False).sum()

library_oview['Total Activity'] = library_oview['Total Checkouts'] + library_oview['Total Renewals']



ax = sns.set_color_codes('pastel')

ax = sns.barplot(y='Age Range',x='Total Activity',label='Renewals',data=library_oview,color="g")

ax = sns.set_color_codes('muted')

ax = sns.barplot(y='Age Range',x='Total Checkouts',label='Checkouts',data=library_oview,color="g")

ax = plt.gca()

ax = ax.get_xaxis().get_major_formatter().set_scientific(False)

ax = plt.legend(ncol=2,loc='lower right', frameon=True)

ax = plt.title('Action vs Age Range')

ax = plt.xticks(rotation=45)
sns.stripplot(y='Age Range',x='Total Checkouts',data=library, jitter=True,order=age_ranges_order)

plt.title('Age Range vs Total Checkout', fontsize = 15)

plt.xticks(rotation=45)
sns.stripplot(y='Age Range',x='Total Renewals',data=library, jitter=True,order=age_ranges_order)

plt.title('Age Range vs Total Renewals', fontsize=15)

plt.xticks(rotation=45)
library['Year Patron Registered'].unique()
plt.title('Age Range vs Year Patron Registerd')

sns.boxplot(y='Age Range',

              x='Year Patron Registered',

              data=library,order=age_ranges_order)
plt.figure(figsize=(5,10))

plt.title('Registered Year distribution per Library')

sns.boxplot(y='Home Library Definition',

              x='Year Patron Registered',

              data=library)