# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

df = pd.read_csv(r'../input/cereal.csv')
df.describe()
df.describe(include=['O'])
df.head()
df.info()
import seaborn as sns

sns.distplot(df['sodium'],kde=True).set_title('Distribution of Sodium')
sns.distplot(df['sugars'],kde=True).set_title('Distribution of Sugar')
import pylab 

import scipy.stats as stats



stats.probplot(df['sugars'], dist="norm", plot=pylab)

pylab.show()
hotOnly = df.loc[df['type']=='H']

sodiumHot = hotOnly.loc[:,['sodium']]

sodiumHot.describe()
coldOnly = df.loc[df['type']=='C']

sodiumCold = coldOnly.loc[:,['sodium']]

sodiumCold.describe()
print('Mean for potassium in hot cereal: ' + str(sodiumHot.mean()))

print('Mean for potassium in cold cereal: ' + str(sodiumCold.mean()))
from scipy.stats import ttest_ind

ttest_ind(sodiumCold, sodiumHot, equal_var=False)
sns.distplot(df['potass'],kde=True).set_title('Distribution of Potassium2')
potassHot = df['potass'][df['type'] == 'H']

potassCold = df['potass'][df['type'] == 'C']

print('Mean for potassium in hot cereal: ' + str(potassHot.mean()))

print('Mean for potassium in cold cereal: ' + str(potassCold.mean()))
ttest_ind(potassHot, potassCold, equal_var=False)
import matplotlib.pyplot as plt

plt.hist(potassHot, alpha=0.5, label='hot')

plt.hist(potassCold, alpha=0.5, label='cold')

plt.legend(loc='upper right')
plt.hist(sodiumHot, alpha=0.5, label='hot')

plt.hist(sodiumCold, alpha=0.5, label='cold')

plt.legend(loc='upper right')