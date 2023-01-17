# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import math
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)


from wordcloud import WordCloud, STOPWORDS

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import matplotlib 
import matplotlib.pyplot as plt
import sklearn
%matplotlib inline
import matplotlib.pyplot as plt 
plt.rcParams["figure.figsize"] = [16, 12]
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
filenames = check_output(["ls", "../input"]).decode("utf8").strip().split('\n')
filenames
df = pd.read_csv("../input/"+filenames[1]) 
print(df.dtypes)
df.head()
dc = pd.read_csv("../input/"+filenames[0]) 
print(dc.dtypes)
dc.head()
df.describe()
dc.describe()
dc['outcome_age_(years)'].hist()
plt.title('Outcome Age (years)')
from scipy.stats import ttest_ind
dc.nunique()
dc['Spay/Neuter'].unique()
ttest_ind([1,2,3],[4,5,5], equal_var= False)


intactCats = dc["outcome_age_(days)"][dc['Spay/Neuter'] == 'No']
# get the sodium for cold ceareals
neutralCats = dc["outcome_age_(days)"][dc['Spay/Neuter'] == "Yes"]


femaleCats = dc["outcome_age_(days)"][dc.sex == 'Female']
# get the sodium for cold ceareals
maleCats = dc["outcome_age_(days)"][dc.sex == "Male"]

# compare them
print(ttest_ind(femaleCats, maleCats, equal_var=False))

print(ttest_ind(femaleCats, femaleCats, equal_var=False))


print(ttest_ind(femaleCats,- femaleCats, equal_var=False))

ttest_ind(intactCats, neutralCats, equal_var=False)

print(femaleCats.mean())
print(maleCats.mean())
print(intactCats.mean())
print(neutralCats.mean())
import pylab 
from scipy.stats import probplot # for a qqplot
probplot(dc["outcome_age_(days)"], dist="norm", plot=pylab)
dc['outcome_age_(days)'].hist(by = dc.sex)

plt.hist()
plt.title('Outcome Age (days) by Gender')
# plot the cold cereals
plt.hist(maleCats,  label='Male', alpha = 0.4)
# and the hot cereals
plt.hist(femaleCats, label='Female', alpha = 0.4)
# and add a legend
plt.legend(loc='upper right')
# add a title
plt.title('Outcome Age (days) by Gender')
# plot the cold cereals
plt.hist(intactCats,  label='Intact', alpha = 0.4)
# and the hot cereals
plt.hist(neutralCats, label='Neutral', alpha = 0.4)
# and add a legend
plt.legend(loc='upper right')
# add a title
plt.title('Outcome Age (days) by Surgery')
dc.columns
dc.outcome_type.value_counts()
sns.barplot( x = 'Spay/Neuter', y ='outcome_age_(days)', hue = 'sex', data = dc)
sns.barplot( x = 'outcome_type', y ='outcome_age_(days)', hue = 'sex', data = dc)
plt.title('Outcome age versus outcome type and gender')
sns.countplot(x = 'outcome_type', data = dc)
sns.countplot(y="outcome_type", hue="sex", data=dc, palette="Greens_d")
sns.countplot(x="outcome_type", hue="Spay/Neuter", data=dc)
import scipy.stats
 
scipy.stats.chisquare(dc["outcome_type"].value_counts())
 
scipy.stats.chisquare(dc["outcome_hour"].value_counts())

scipy.stats.chi2_contingency(pd.crosstab(dc['outcome_type'], dc['sex']))
scipy.stats.chi2_contingency(pd.crosstab(dc['outcome_type'], dc['Spay/Neuter']))
scipy.stats.chi2_contingency(pd.crosstab(dc['sex'], dc['Spay/Neuter']))
