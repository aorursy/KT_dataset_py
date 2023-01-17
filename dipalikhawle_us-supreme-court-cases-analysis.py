# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import folium

import squarify



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import os

print(os.listdir("../input"))
us_supreme_court_data = "../input/supreme-court/database.csv"

df = pd.read_csv(us_supreme_court_data)
df.head()
df.tail()
df.shape
df.describe()
df['majority_votes'].max()
df['case_name'][df['chief_justice'] == 'Warren']
df[45:55]
df.columns
df.law_type.head()
df[['case_origin','case_source']]
df.corr()
df['law_supplement'].max()
df[df.issue_area == df.issue_area.max()]
df.court[df.issue_area == df.issue_area.max()]
c = df.groupby('chief_justice')
c = df.groupby('chief_justice')

for chief_justice,chief_justice_df in c:

    print(chief_justice)

    print(chief_justice_df)
import matplotlib.pyplot as plt

import seaborn as sns

plt.rcParams['figure.figsize'] = (20, 9)

plt.style.use('dark_background')



sns.countplot(df['chief_justice'], palette = 'gnuplot')



plt.title('cheif justice', fontweight = 30, fontsize = 20)

plt.xticks(rotation = 90)

plt.show() 
import seaborn as sns

import matplotlib.pyplot as plt

import folium

import squarify



plt.style.use('seaborn')





df['majority_votes'].value_counts().head(15).plot.pie(figsize = (15, 8))



plt.title('majority_votes',fontsize = 20)



plt.xticks(rotation = 90)
plt.plot(df['majority_votes'])
plt.plot(df['decision_type'])