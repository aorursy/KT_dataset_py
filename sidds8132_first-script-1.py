# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



#Load in files
crashdata = pd.read_csv("../input/3-Airplane_Crashes_Since_1908.txt",index_col='Date',parse_dates=[0])

crashdata[:5]

temps = pd.read_csv('../input/3-Airplane_Crashes_Since_1908.txt')


#Set the time to an easier format for graphing
temps['Date'] = pd.to_datetime(temps['Date'])
temps['Year'] = temps['Date'].map(lambda x: x.year)
temps['Survivors'] = temps['Aboard'] - temps['Fatalities']

#min and max years
yearmin = temps['Year'].min()
yearmax = temps['Year'].max()
years = range(yearmin, yearmax)

planecrash = []
peopleonboard = []
survivors = []
dead = []
yearseries = pd.Series(years)


for year in years:
    currentyeardata = temps[temps['Year'] == year]
    planecrash.append(len(currentyeardata.index))
    peopleonboard.append(sum(currentyeardata['Aboard']))
    dead.append(sum(currentyeardata['Fatalities']))
    survivors.append(sum(currentyeardata['Survivors']))
len(yearseries), len(planecrash), len(peopleonboard), len(dead), len(survivors), len(years)

sns.set_color_codes(palette='bright')
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
sns.barplot(x=yearseries, y=planecrash, palette = sns.color_palette(["#ff021a"]), ax=ax1)
plt.setp(ax1.patches, linewidth=0)
text = ax1.set(ylabel="Crashes", xlabel="Year", title="Planes Crashed Per Year")

sns.barplot(x=yearseries, y=peopleonboard, palette = sns.color_palette(["#ffa535"]), ax=ax2)
plt.setp(ax2.patches, linewidth=0)

text = ax2.set(ylabel="Aboard", xlabel="Year", title="People Aboard Per Year")

sns.barplot(x=yearseries, y=dead, palette = sns.color_palette(["#ff061c"]), ax=ax3)
plt.setp(ax3.patches, linewidth=0)
text = ax3.set(ylabel="Fatalitites", xlabel="Year", title="Dead Per Year")

sns.barplot(x=yearseries, y=survivors, palette = sns.color_palette(["#04ff4d"]), ax=ax4)
plt.setp(ax4.patches, linewidth=0)

text = ax4.set(ylabel="Survived", xlabel="Year", title="Survived Per Year")

