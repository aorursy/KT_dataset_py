# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import sqlite3 

import statsmodels.formula.api as sm



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/celebrity_deaths.csv')

df.head()
death_by_year = df.groupby('death_year')['name'].count()



ax = death_by_year.plot(kind = 'bar')



ax.set_xlabel('Year')

ax.set_ylabel('Death Count')

cxn = sqlite3.connect(':memory:')

df.to_sql('celebs',cxn)
query = '''

Select 

    death_year as Year,

    case

        when cause_of_death LIKE "%natural%" then 'Natural'

        when cause_of_death LIKE "%cancer%" then 'Cancer'

        when cause_of_death LIKE "%murder%" then 'Murder'

        when cause_of_death LIKE "%alzheimer's%" then 'Alzheimers'

        else 'Other' end as Cause

From

celebs

'''





deaths = pd.read_sql(query,cxn)
years = deaths.groupby(['Year','Cause']).size()
years.reset_index().pivot('Year','Cause',0).fillna(0).drop('Other',axis= 1).plot()
cancer = years.loc[:,'Cancer'].reset_index()

cancer.columns = 'Year Cancer'.split()



cancer['Year']= np.arange(11)

model = sm.ols(formula="Cancer ~ Year", data=cancer).fit()

model.summary()

ax = years.reset_index().pivot('Year','Cause',0).fillna(0).drop('Other',axis= 1).apply(lambda x: x/x.sum(), axis = 1).plot(kind = 'barh', stacked = True)



ax.set_xlim(0,1.3)

ax.set_xlabel('Proportion of Deaths')