# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt 



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
covid=pd.read_csv('/kaggle/input/covid-data100-days/covid19_Confirmed_dataset.csv')

covid
covid_1=covid.drop(['Lat','Long'],axis=1)
covid_1
covid_2=covid_1.groupby("Country/Region").sum()
covid_2
countries=covid_2.index

countries
covid_2.loc['US'].plot()

covid_2.loc['Spain'].plot()

plt.legend(loc=2)
covid_2.loc['China'].diff().plot()

covid_2.loc['US'].diff().plot()

plt.legend()
covid_2.loc['China'].diff().max()
covid_2.loc['US'].diff().max()
max_infection_rate=[]

for country in countries:

    max_infection_rate.append(covid_2.loc[country].diff().max())
covid_2['Max Infection Rate']=max_infection_rate

covid_2.head()
Covid=pd.DataFrame(covid_2['Max Infection Rate'])
Covid.head(20)
happiness_report=pd.read_csv('/kaggle/input/world-happiness/worldwide_happiness_report.csv')
happiness_report.head()
useless=['Regional indicator','Standard error of ladder score','upperwhisker','lowerwhisker','Generosity','Perceptions of corruption']
happiness_report=happiness_report.drop(useless,axis=1)
happiness_report.head()
happiness_report=happiness_report.set_index('Country name')

happiness_report
data= Covid.join(happiness_report,how='inner')

data.head()
y1=data['Max Infection Rate']

x1=data['Logged GDP per capita']
sns.regplot(x1,np.log(y1))

plt.title("Plot of Max Infection Rate Vs. GDP")
x2=data['Healthy life expectancy']

y2=data['Max Infection Rate']
sns.regplot(np.log(y2),x2)
x5=data['Ladder score']

y5=data['Max Infection Rate']
sns.regplot(np.log(y5),x5)