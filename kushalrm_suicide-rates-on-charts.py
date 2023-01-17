import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sb

%matplotlib inline

plt.rcParams['figure.figsize'] = (10.0,10.0)

import warnings

warnings.filterwarnings("ignore")





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
suicide_data = pd.read_csv('/kaggle/input/suicide-rates-overview-1985-to-2016/master.csv')

suicide_data.head()
# Describing the dataset:

suicide_data.describe()
suicide_data.columns
suicide_data = suicide_data.rename(columns={'country':'Country','year':'Year','sex':'Gender','age':'Age','suicides_no':'SuicidesNo',

                          'population':'Population','suicides/100k pop':'Suicides100kPop','country-year':'CountryYear',

                          'HDI for year':'HDIForYear',' gdp_for_year ($) ':'GdpForYear',

                          'gdp_per_capita ($)':'GdpPerCapital','generation':'Generation'})

suicide_data.head()
suicide_data.columns
suicide_data.isnull().sum()
suicide_data = suicide_data.drop(['HDIForYear', 'CountryYear'],axis=1)
suicide_data.head()
data = suicide_data.groupby('Country').agg({'SuicidesNo':'sum'}).sort_values(by='SuicidesNo', ascending = False)

data = data.head(15)

data
sb.barplot(data['SuicidesNo'],data.index, palette='Reds_r')
data1 = suicide_data.groupby('Gender').agg({'SuicidesNo':'sum'}).sort_values(by='SuicidesNo', ascending = False)

data1 = data1.head(15)
sb.barplot(data1.index, data1['SuicidesNo'], palette='Blues_r')
#unique values

pd.unique(suicide_data['Age'])
data2 = suicide_data.groupby('Age').agg({'SuicidesNo':'sum'}).sort_values(by='SuicidesNo', ascending = False)

data2
# Pie chart

labels = data2.index

sizes = data2['SuicidesNo']

#colors

colors = ['#ff6666','#ff9999','#66b3ff','#99ff99','#ffcc99','#ffcc60']

 

fig1, ax1 = plt.subplots()

ax1.pie(sizes, colors = colors, labels=labels, autopct='%1.1f%%', startangle=90)

#draw circle

centre_circle = plt.Circle((0,0),0.70,fc='white')

fig = plt.gcf()

fig.gca().add_artist(centre_circle)

# Equal aspect ratio ensures that pie is drawn as a circle

ax1.axis('equal')  

plt.tight_layout()

plt.show()
suicide_data['Year'].unique()
data3 = suicide_data.groupby('Year').agg({'SuicidesNo':'sum'}).sort_values(by='SuicidesNo', ascending = False)
plt.figure(figsize=(15,7))

sb.barplot(data3.index, data3['SuicidesNo'])

plt.title('Total number of Suicides from year 1985-2016 ')

plt.show()
data4 = suicide_data.groupby('Generation').agg({'SuicidesNo':'sum'}).sort_values(by='SuicidesNo', ascending = False)

data4
data5 = suicide_data.groupby('Generation').agg({'Suicides100kPop':'sum'}).sort_values(by='Suicides100kPop', ascending = True)

data5.head()
plt.figure(figsize=(10,9))

sb.barplot(data4.index,data4['SuicidesNo'], palette = 'Blues_d')# color = '#ff9449')

#sb.barplot(data5.index,data5['Suicides100kPop'])

plt.plot()

plt.show()
data6 = suicide_data.groupby('Generation').agg({'Population':'sum'}).sort_values(by='Population', ascending = True)

data6 

data6['suicides'] = data5.values

data6['suicide_percentage'] = (data6['suicides']/ data6['Population'])*100

data6.sort_values(by='suicide_percentage', ascending = False)
# Pie chart

labels = data6.index

sizes = data6['Population']

#colors

colors = ['#ff6666','#ff9999','#66b3ff','#99ff99','#ffcc99','#ffcc60']

 

fig1, ax1 = plt.subplots()

ax1.pie(sizes, colors = colors, labels=labels, autopct='%1.1f%%', startangle=90)

#draw circle

centre_circle = plt.Circle((0,0),0.70,fc='white')

fig = plt.gcf()

fig.gca().add_artist(centre_circle)

# Equal aspect ratio ensures that pie is drawn as a circle

ax1.axis('equal')  

plt.tight_layout()

plt.show()
sb.countplot(x='Generation',hue ="Gender",

                 data=suicide_data)

plt.xticks(rotation=45)

plt.title('Generations vs Gender count')

plt.show()
sb.pairplot(suicide_data, hue= "Generation", diag_kind = "kde", kind = "scatter", palette = "husl")

plt.show()
sb.pairplot(suicide_data, hue= "Gender", diag_kind = "kde", kind = "scatter", palette = "autumn")

plt.show()
print('Total population of 101 countries:', suicide_data['Population'].sum())

print('Total number of suicide deaths:', suicide_data['SuicidesNo'].sum())