# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df_countries = pd.read_csv('../input/Country.csv')

df_indicators = pd.read_csv('../input/Indicators.csv')

df_series = pd.read_csv('../input/Series.csv')
df_indicators[df_indicators.CountryName == 'Indonesia'].drop_duplicates('IndicatorCode')['IndicatorName'].iloc[1]
df_indo = df_indicators[df_indicators.CountryName == 'Indonesia']
ind_code = 'SP.ADO.TFRT'

data = df_indo[df_indo.IndicatorCode == ind_code]

plt.grid()

plt.ylabel('Value')

plt.xlabel('Year')



# 1st Plot

line1, = plt.plot(data['Year'], data['Value'], label=data['IndicatorCode'])

# 2nd Plot



plt.legend([line1])
len(df_indicators[df_indicators.CountryName == 'Indonesia'])
df_countries = pd.read_csv('../input/Country.csv')

df_indicators = pd.read_csv('../input/Indicators.csv')

df_series = pd.read_csv('../input/Series.csv')
df_indicators[df_indicators.CountryName == 'Indonesia'].drop_duplicates('IndicatorCode')
df_countries[df_countries.CountryCode == 'IDN']