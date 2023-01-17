# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib 

import matplotlib.pyplot as plt

color = sns.color_palette()

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
country_total=pd.read_csv('../input/GlobalLandTemperaturesByCountry.csv')
country_total.head(2)
country_total.head(20)

type(country_total)
USA=country_total.ix[country_total['Country']=='United States']
countries=[]

for i in country_total['Country']:

    countries.append(i)

country_list=set(countries)
country_list
USA.head(5)
sns.factorplot(data=USA[['dt','AverageTemperature']],x="dt", y="AverageTemperature")
plt.figure(figsize=(100,10))

sns.factorplot(data=USA[['dt','AverageTemperature']],x="dt", y="AverageTemperature",color=color[3])

plt.ylabel('avg temp', fontsize=12)

plt.xlabel('date', fontsize=12)

plt.xticks(rotation='vertical')

plt.show()