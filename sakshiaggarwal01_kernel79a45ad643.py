# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

nC0v_20200121_20200126_SUMMARY = pd.read_csv("../input/2019-coronavirus-dataset-01212020-01262020/2019_nC0v_20200121_20200126 - SUMMARY.csv")

nC0v_20200121_20200126_cleaned = pd.read_csv("../input/2019-coronavirus-dataset-01212020-01262020/2019_nC0v_20200121_20200126_cleaned.csv")

nCoV_20200121_20200127 = pd.read_csv("../input/2019-coronavirus-dataset-01212020-01262020/2019_nCoV_20200121_20200127.csv")

nCoV_20200121_20200128 = pd.read_csv("../input/2019-coronavirus-dataset-01212020-01262020/2019_nCoV_20200121_20200128.csv")

nCoV_20200121_20200130 = pd.read_csv("../input/2019-coronavirus-dataset-01212020-01262020/2019_nCoV_20200121_20200130.csv")

nCoV_20200121_20200131 = pd.read_csv("../input/2019-coronavirus-dataset-01212020-01262020/2019_nCoV_20200121_20200131.csv")

nCoV_20200121_20200201 = pd.read_csv("../input/2019-coronavirus-dataset-01212020-01262020/2019_nCoV_20200121_20200201.csv")

nCoV_20200121_20200205 = pd.read_csv("../input/2019-coronavirus-dataset-01212020-01262020/2019_nCoV_20200121_20200205.csv")

nCoV_20200121_20200206 = pd.read_csv("../input/2019-coronavirus-dataset-01212020-01262020/2019_nCoV_20200121_20200206.csv")
nC0v_20200121_20200126_SUMMARY
#Deaths Suspected in various cities of China

nC0v_20200121_20200126_SUMMARY['Deaths'].unique()
nC0v_20200121_20200126_SUMMARY['Recovered'].unique()
nC0v_20200121_20200126_SUMMARY['Country'].unique()
nC0v_20200121_20200126_SUMMARY[['Country','Deaths']][nC0v_20200121_20200126_SUMMARY.Deaths>20]
nC0v_20200121_20200126_SUMMARY.apply(lambda x: x.dtype)
#For all the numerical columns statistics is given.

nC0v_20200121_20200126_SUMMARY.describe()
#Country-wise total no. of cases observed in various states.

grouped = nC0v_20200121_20200126_SUMMARY.groupby("Country")

grouped.size().sort_values(ascending=False)
nC0v_20200121_20200126_SUMMARY['Date last updated'].min()

#output: 21st Jan 2020
nC0v_20200121_20200126_SUMMARY['Date last updated'].max()

#output: 26th Jan 2020
from matplotlib import pyplot as plt

%matplotlib inline

import seaborn as sns
#plt.hist(nC0v_20200121_20200126_SUMMARY['Date last updated'],bins=['1/21/2020', '1/22/2020 12:00', '1/23/20 12:00 PM',

#       '1/24/2020 12:00 AM', '1/24/2020 12:00 PM', '1/24/2020 4:00 PM',

#       '1/25/2020 12:00 AM', '1/25/2020 12:00 PM', '1/25/2020 10:00 PM',

#       '1/26/2020 11:00 AM'])

#plt.show()
nC0v_20200121_20200126_SUMMARY['Date last updated'].unique()
country_list=['Mainland China','Japan','Thailand','South Korea','United States','China']

#country_list=nC0v_20200121_20200126_SUMMARY['Country'].unique()

deaths_list=[1,17,24,39,40,52]

#print(nC0v_20200121_20200126_SUMMARY['Country'].unique().shape)

#print(nC0v_20200121_20200126_SUMMARY['Deaths'].unique().shape)

plt.bar(country_list,deaths_list)
print(nC0v_20200121_20200126_SUMMARY['Country'].unique())

print(nC0v_20200121_20200126_SUMMARY['Deaths'].unique())