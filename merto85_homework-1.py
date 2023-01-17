# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns  # visualization tool
import matplotlib.pyplot as plt   # basic plot

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
terror_data = pd.read_csv('../input/globalterrorismdb_0617dist.csv', encoding = 'ISO-8859-1')
terror_data.rename(columns = {'iyear':'Year', 'imonth':'Month', 'iday':'Day','country_txt':'Country', 'region_txt':'Region', 'attacktype1_txt':'AttackType','target1':'Target','nkill':'Killed','nwound':'Wounded','ndays':'DaysOfKidnapping','summary':'Summary','gname':'TerrorGroup','targtype1':'TargetType','natlty1_txt':'NationalityOfTarget','weaptype1_txt':'WeaponOfAttack','motive':'Motivation'}, inplace = True)
terror_data=terror_data[['Day','Month','Year','Country','Region','AttackType','Target','Killed','Wounded','DaysOfKidnapping','Summary','TerrorGroup','TargetType','NationalityOfTarget','WeaponOfAttack','Motivation']]
terror_data.index
terror_data.head(10)
terror_data['Fatalities'] = terror_data['Wounded'] + terror_data['Killed']
print('Country with Highest Terror Attacks Occurance:',terror_data['Country'].value_counts().index[0])
print('Year of Highest Terror Attack Occurance',terror_data['Year'].value_counts().index[0])
print('Most Frequent Type Of Teror Attack Occurance', terror_data['AttackType'].value_counts().index[0])
terror_data.plot(x='Year',y ='Fatalities',color ='Blue',figsize=(10,10))
terror_data_Poland = terror_data[terror_data['Country'] == 'Poland']
terror_data_Germany = terror_data[terror_data['Country'] == 'Germany']
terror_data_Poland.info()
terror_data_Poland['Fatalities'] = terror_data_Poland['Wounded'] + terror_data_Poland['Killed']
terror_data_Germany['Fatalities'] = terror_data_Germany['Wounded'] + terror_data_Germany['Killed']
terror_data_Germany.plot(kind ='line', x ='Year', y ='Fatalities', color = 'r',alpha = 0.8,grid = True,figsize=(20,10),linestyle = ':',linewidth=2)
plt.legend(loc='upper right')     # legend = puts label into plot
plt.axis('auto')
plt.xlabel('Year')
plt.ylabel('Fatalities')
plt.title('Terror Attacks in Germany by Years')

terror_data_Poland.plot(kind ='line', x ='Year', y ='Fatalities', color = 'g',alpha = 0.8,grid = True,figsize=(20,10),linestyle = '-.',linewidth=2)
plt.legend(loc='upper right')     # legend = puts label into plot
plt.axis('auto')
plt.xlabel('Year')
plt.ylabel('Fatalities')
plt.title('Terror Attacks in Poland by Years')

plt.show()
