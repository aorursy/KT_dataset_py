# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

#Read Datasets
country = pd.read_csv('../input/Country.csv')
country_notes = pd.read_csv('../input/CountryNotes.csv')
indicators = pd.read_csv('../input/Indicators.csv')
food_import = indicators[(indicators.CountryName=='India')&(indicators.IndicatorCode=='TM.VAL.FOOD.ZS.UN')]
food_export = indicators[(indicators.CountryName=='India')&(indicators.IndicatorCode=='TX.VAL.FOOD.ZS.UN')]
plt.plot(food_import.Year, food_import.Value, 'o-',label='Imports')
plt.plot(food_export.Year, food_export.Value, 'o-',label='Exports')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel('Years',  fontsize=14)
plt.ylabel('% of Merchandise',  fontsize=14)
plt.title('Trends in Food Import/Export in India', fontsize=14)
arable_land = indicators[(indicators.CountryCode == 'IND') & (indicators.IndicatorCode == 'AG.LND.ARBL.ZS')]
arable_land_per_person = indicators[(indicators.CountryName == 'India') & (indicators.IndicatorCode == 'AG.LND.ARBL.HA.PC')]
Years = arable_land.Year
Land = arable_land.Value
#sns.violinplot(x='Years', y='Land', data=arable_land, size=7)
plt.plot(Years, Land)
plt.xlabel("Years", fontsize=14)
plt.ylabel("Arable Area as % of Land Area", fontsize=14)
plt.title("Arable Land in India as % of Land Area", fontsize=16)
