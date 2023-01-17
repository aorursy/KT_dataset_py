# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))
# read data
country = pd.read_csv('../input/Country.csv')
country_notes = pd.read_csv('../input/CountryNotes.csv')
indicators = pd.read_csv('../input/Indicators.csv')
series = pd.read_csv('../input/Series.csv')
series_notes = pd.read_csv('../input/SeriesNotes.csv')


birthrate=indicators[indicators['CountryName'].isin({'Poland'}) & indicators['IndicatorName'].isin({'Birth rate, crude (per 1,000 people)'})]
birthrateall=indicators[indicators['IndicatorName'].isin({'Birth rate, crude (per 1,000 people)'})]
birthrateallgrouped=birthrateall.groupby('Year')
birthrateall=birthrateallgrouped.mean()

deathrate=indicators[indicators['CountryName'].isin({'Poland'}) & indicators['IndicatorName'].isin({'Death rate, crude (per 1,000 people)'})]
deathrateall=indicators[indicators['IndicatorName'].isin({'Death rate, crude (per 1,000 people)'})]
deathrateallgrouped=deathrateall.groupby('Year')
deathrateall=deathrateallgrouped.mean()
plt.figure()
pol, =plt.plot(deathrate['Year'], deathrate['Value'], label='Poland')
wrld, =plt.plot(deathrateall, label='World')
plt.legend([pol, wrld], ['Poland', 'World'])
plt.title('Death rate in Poland & World')
plt.figure()
pol, =plt.plot(birthrate['Year'],birthrate['Value'])
wrld, =plt.plot(birthrateall)
plt.legend([pol, wrld], ['Poland', 'World'])
plt.title('Birth Rate in Poland and World')