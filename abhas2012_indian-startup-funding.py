# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/startup_funding.csv')
df.head()
df.count()
df=df.drop(['SNo'],axis=1)
df.head()
df['CityLocation'].nunique()
df['InvestmentType']=df['InvestmentType'].str.lower()
df['InvestmentType']=df['InvestmentType'].str.replace("\\s","")
df['InvestmentType']
import matplotlib.pyplot as plt
countbyinvestmenttypes=df.groupby('InvestmentType').count()
countbyinvestmenttypes=countbyinvestmenttypes['StartupName']
countbyinvestmenttypes.plot.bar()
plt.show()
city=df.groupby('CityLocation').count()
city
city=city['StartupName']
type(city)
city=city.sort_values(ascending=False)[:10]
city

plt.figure(figsize=[10,10])
city.plot.bar()
plt.ylabel('No of Startups')
plt.title("Top Startup Locations",size=20)
plt.show()
amount=df.sort_values(['AmountInUSD'],ascending=False)

amount=amount[:10]
amount
df['IndustryVertical']=df['IndustryVertical'].str.replace("\\s","")
df['IndustryVertical']=df['IndustryVertical'].str.upper()
df['AmountInUSD']=df['AmountInUSD'].str.replace(",","")
amount=df.sort_values(by='AmountInUSD',ascending=False)
amount.head()
