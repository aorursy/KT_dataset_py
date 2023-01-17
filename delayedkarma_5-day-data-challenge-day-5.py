# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy.stats

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

df = pd.read_csv('../input/ufo_sighting_data.csv')

df.head()

df.info()
df['country'].value_counts()
len(df['UFO_shape'].value_counts().unique())
df = df[df['country'].notnull()]

df = df[df['UFO_shape'].notnull()]

df.shape
df[df['UFO_shape'].isnull()] # No null values
df[df['country'].isnull()]
contingencytab = pd.crosstab(df['country'],df['UFO_shape'])

contingencytab
# Let's do a chisquared test on the contingency table and print out the p-value



print(scipy.stats.chi2_contingency(contingencytab)[1])
# This plotting could have been done much more elegantly 

# but I had spent too much time on this already -- need loops







fig,(ax1,ax2,ax3,ax4,ax5) = plt.subplots(5,1,figsize=(12,24))

sns.countplot(x='country',hue='UFO_shape',data=df[df['country']=='us'],ax=ax1,palette='bright',alpha=.75)

sns.countplot(x='country',hue='UFO_shape',data=df[df['country']=='ca'],ax=ax2,palette='bright',alpha=.75)

sns.countplot(x='country',hue='UFO_shape',data=df[df['country']=='gb'],ax=ax3,palette='bright',alpha=.75)

sns.countplot(x='country',hue='UFO_shape',data=df[df['country']=='au'],ax=ax4,palette='bright',alpha=.75)

sns.countplot(x='country',hue='UFO_shape',data=df[df['country']=='de'],ax=ax5,palette='bright',alpha=.75)



ax1.legend_.remove()

ax2.legend_.remove()

ax3.legend_.remove()

ax4.legend_.remove()

ax5.legend_.remove()



ax1.set_xlabel('USA', fontsize=15)

ax2.set_xlabel('Canada', fontsize=15)

ax3.set_xlabel('Great Britan', fontsize=15)

ax4.set_xlabel('Australia', fontsize=15)

ax5.set_xlabel('Germany', fontsize=15)



# What inferences can we draw from this? W

# plt.subplots_adjust(hspace = 5.0)



plt.legend(bbox_to_anchor=(0,1), bbox_transform=fig.transFigure)

fig.suptitle('Types of UFO shapes sighted by country',fontsize=20);