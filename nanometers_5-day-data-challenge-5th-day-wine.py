import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy.stats

import seaborn as sns # visuals

import matplotlib



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output

winedata = pd.read_csv("../input/winemag-data_first150k.csv")

winedata.head()
# Looking at data types

winedata.info()
# Check separation of provinces, states and areas of countries

winedata.country = winedata.country.astype('category')

winedata.country.cat.categories



winedata.variety = winedata.variety.astype('category')

winedata.variety.cat.categories

# sns.countplot(y = winedata['province'], bins = 25)

# scipy.stats.chisquare(winedata['province'].value_counts())
# Visualize the wines by 'points' % as total

sns.set_style("darkgrid")

m1 = sns.distplot(winedata.points, bins = 10)
# Visual of Price vs Points by Country

CAwine = winedata[winedata.province == 'California']

CAwine.info()

sns.countplot(y = CAwine.variety)



# vis1 = sns.lmplot(data = winedatalim, x='points', y='price', fit_reg = False, hue = 'country', size = 7, aspect = 1)
contingencyTable = pd.crosstab(CAwine["region_1"], CAwine["variety"])



scipy.stats.chi2_contingency(contingencyTable)