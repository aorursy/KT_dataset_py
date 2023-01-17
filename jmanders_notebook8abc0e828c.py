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
gtd = pd.read_csv('../input/gtd/globalterrorismdb_0617dist.csv', encoding='ISO-8859-1',

                  usecols=[0, 1, 2, 3, 8, 10, 11, 12, 25, 26, 27, 28, 29 , 34, 35, 81, 82])

type(gtd)

gtd.shape
gtd.head()
gtd.columns
gtd.attacktype1_txt.unique()
gtd.attacktype1_txt.value_counts()
import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns

sns.set_style('whitegrid')
ax = sns.countplot(x="attacktype1", data=gtd)
attacktype_month = sns.factorplot(x="imonth", hue="attacktype1_txt", 

                                  kind="count", data=gtd, size=10, palette="muted")
succes_month = sns.factorplot(x="imonth", hue="success", 

                                  kind="count", data=gtd, size=10, palette="muted")
gtd.region_txt.unique()
#only europe

gtd_europe = gtd[(gtd.region_txt == 'Western Europe')]

#test if only europe is remaining

gtd_europe.region_txt.unique()
ax = sns.countplot(x="attacktype1", data=gtd_europe)
attacktype_month = sns.factorplot(x="imonth", hue="attacktype1_txt", 

                                  kind="count", data=gtd_europe, size=10, palette="muted")
succes_month = sns.factorplot(x="imonth", hue="success", 

                                  kind="count", data=gtd_europe, size=10, palette="muted")
popdens = pd.read_csv('../input/world-population/API_EN.POP.DNST_DS2_en_csv_v2.csv', skiprows=[0,1,2], index_col='Country Name')

popdens.drop

popdens.shape
popdens.head()
popdens.columns

popdensT = popdens.drop(popdens.columns[[0,1,2,3,59,60]], axis=1)

popdensT = popdensT.T
popdensT.head()
popdensT.Germany.plot(legend=True)
gtd_eventyear = gtd.set_index('iyear')

gtd_eventyear = gtd_eventyear.drop(gtd_eventyear.columns[[0,4,5,6]],axis=1)
gtd_eventyear.head()