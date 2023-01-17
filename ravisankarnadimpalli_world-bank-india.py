# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

Country = pd.read_csv("../input/world-development-indicators/Country.csv")

CountryNotes = pd.read_csv("../input/world-development-indicators/CountryNotes.csv")

Footnotes = pd.read_csv("../input/world-development-indicators/Footnotes.csv")

Indicators = pd.read_csv("../input/world-development-indicators/Indicators.csv")

Series = pd.read_csv("../input/world-development-indicators/Series.csv")

SeriesNotes = pd.read_csv("../input/world-development-indicators/SeriesNotes.csv")

GDPRaise = pd.read_csv("../input/india-gdp-growth-world-bank-1961-to-2017/India GDP from 1961 to 2017.csv")

wbidata = pd.read_csv("../input/world-bank-data-of-indian-economy-since-1991/World_Bank_Data_India.csv")

ile = pd.read_csv("../input/world-bank-data-of-indian-economy-since-1991/World_Bank_Data_India.csv")
import pandas as pd

country_population = pd.read_csv("../input/world-bank-data-1960-to-2016/country_population.csv")

fertility_rate = pd.read_csv("../input/world-bank-data-1960-to-2016/fertility_rate.csv")

life_expectancy = pd.read_csv("../input/world-bank-data-1960-to-2016/life_expectancy.csv")
import pandas as pd

World_Bank_Data_India = pd.read_csv("../input/world-bank-data-of-indian-economy-since-1991/World_Bank_Data_India.csv")

World_Bank_Data_India_Definitions = pd.read_csv("../input/world-bank-data-of-indian-economy-since-1991/World_Bank_Data_India_Definitions.csv")
import pandas as pd

pd.plotting.register_matplotlib_converters()

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

import datetime

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")

# Set the width and height of the figure

plt.figure(figsize=(16,6))



# Line chart showing how FIFA rankings evolved over time 

#sns.lineplot(data=GDPRaise)

sns.lineplot(data=GDPRaise, x='1961',y='3.722742533')

plt.title("Indian GDP from 1960 to 2017")

plt.figure(figsize=(20,6))

sns.barplot(data=GDPRaise, x='1961',y='3.722742533')

plt.title("Indian GDP from 1960 to 2017")
plt.figure(figsize=(20,6))

plt.title("Indian GDP from 1960 to 2017")

wbidatas= wbidata.pivot("EMP_TOTL","Years","GDP_AGR")

ax = sns.heatmap(data=wbidatas, annot=True)

#wbidata.head()
#life_expectancy.head()

#sns.scatterplot(x=ile['bmi'], y=ile['charges'])

#from pandas import DataFrame

#df=DataFrame(life_expectancy)



China=life_expectancy.loc[life_expectancy["Country Name"] == "China"]

columnsNamesArr=China.columns.values

Chinaa=China.drop(columns=["Country Name", "Country Code","Indicator Name", "Indicator Code"])

Chinaaa=Chinaa.transpose()

Chinaaa.columns = ['LE']

Chinaaa.head()

new=columnsNamesArr[4:63]

plt.figure(figsize=(20,6))

plt.title("Graph of Chinese Life Expectency over the years!")

sns.scatterplot(new, y=Chinaaa['LE'])





#plt.title("Graph of Chinese Life Expectency over the years!")

#sns.scatterplot(new, y=Chinaaa['LE'])

#sns.scatterplot(Chinaaa)

#Test = pd.DataFrame(columnsNamesArr,index=['Years'])

#print(columnsNamesArr)

#Test = pd.DataFrame(columnsNamesArr,index=['Product A', 'Product B'])

#new=columnsNamesArr.remove['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code']

#abc = np.delete(listOfColumnNames, 'Country Name')

#print(abc)

#Years = listOfColumnNames.drop(rows=["Country Name", "Country Code", "Indicator Name", "Indicator Code"])

#Years.head()

#Chinaa=df.drop(columns=["Country Name", "Country Code", "Indicator Name", "Indicator Code"])

#Chinaaa=DataFrame.transpose(Chinaa)

#colnames(Chinaaa) == c("COL1","VAL1","VAL2")

#sns.scatterplot(x=Chinaaa['years'], y=Chinaaa['0'])

#Chinaaa.head()
#WDDIData = pd.read_csv("../input/world-development-indicators/Indicators.csv")

wbidata.head()

plt.figure(figsize=(20,6))

plt.title("India: GDP Aggregate vs Inflation - Regression Plot")

sns.regplot(x=wbidata['GDP_AGR'], y=wbidata['INFL'])
wbidata.head()

plt.figure(figsize=(20,6))

plt.title("India: GDP Aggregate vs Inflation - Scatter Plot")

sns.scatterplot(x=wbidata['GDP_AGR'], y=wbidata['INFL'])
plt.figure(figsize=(20,6))

plt.title("Chinese Life Expectency")

sns.distplot(a=Chinaaa['LE'], kde=False)
# KDE plot 

plt.figure(figsize=(20,6))

plt.title("Chinese Life Expectency")

sns.kdeplot(data=Chinaaa['LE'], shade=True)
#plt.figure(figsize=(20,20))



sns.jointplot(x=wbidata['GDP_AGR'], y=wbidata['INFL'], kind="kde")

plt.title("India: GDP Aggregate vs Inflation - 2 Dimensional KDE Plot")
# Histograms for each species

China=life_expectancy.loc[life_expectancy["Country Name"] == "China"]

columnsNamesArr=China.columns.values

Chinaa=China.drop(columns=["Country Name", "Country Code","Indicator Name", "Indicator Code"])

Chinaaa=Chinaa.transpose()

Chinaaa.columns = ['LE']

Chinaaa.head()

new=columnsNamesArr[4:63]

#plt.figure(figsize=(20,6))

#plt.title("Graph of Chinese Life Expectency over the years!")

#sns.scatterplot(new, y=Chinaaa['LE'])





BTN=life_expectancy.loc[life_expectancy["Country Name"] == "Bhutan"]

columnsNamesArrbtn=BTN.columns.values

BTN_new=BTN.drop(columns=["Country Name", "Country Code","Indicator Name", "Indicator Code"])

BTN_a=BTN_new.transpose()

BTN_a.columns = ['LE']

BTN_a.head()

btnnew=columnsNamesArr[4:63]

#plt.figure(figsize=(20,6))



AFG=life_expectancy.loc[life_expectancy["Country Name"] == "Afghanistan"]

columnsNamesArrafg=AFG.columns.values

AFG_new=AFG.drop(columns=["Country Name", "Country Code","Indicator Name", "Indicator Code"])

AFG_a=AFG_new.transpose()

AFG_a.columns = ['LE']

AFG_a.head()

afgnew=columnsNamesArr[4:63]

#plt.figure(figsize=(20,6))



plt.figure(figsize=(20,6))

sns.distplot(a=AFG_a['LE'], label="Afghanistan", kde=False)

sns.distplot(a=BTN_a['LE'], label="Bhutan", kde=False)

sns.distplot(a=Chinaaa['LE'], label="China", kde=False)



# Add title

plt.title("Histogram of Life Expectency in Afghanistan, Bhutan & China")



# Force legend to appear

plt.legend()
# Histograms for each species

China=life_expectancy.loc[life_expectancy["Country Name"] == "China"]

columnsNamesArr=China.columns.values

Chinaa=China.drop(columns=["Country Name", "Country Code","Indicator Name", "Indicator Code"])

Chinaaa=Chinaa.transpose()

Chinaaa.columns = ['LE']

Chinaaa.head()

new=columnsNamesArr[4:63]

#plt.figure(figsize=(20,6))

#plt.title("Graph of Chinese Life Expectency over the years!")

#sns.scatterplot(new, y=Chinaaa['LE'])





BTN=life_expectancy.loc[life_expectancy["Country Name"] == "Bhutan"]

columnsNamesArrbtn=BTN.columns.values

BTN_new=BTN.drop(columns=["Country Name", "Country Code","Indicator Name", "Indicator Code"])

BTN_a=BTN_new.transpose()

BTN_a.columns = ['LE']

BTN_a.head()

btnnew=columnsNamesArr[4:63]

#plt.figure(figsize=(20,6))



AFG=life_expectancy.loc[life_expectancy["Country Name"] == "Afghanistan"]

columnsNamesArrafg=AFG.columns.values

AFG_new=AFG.drop(columns=["Country Name", "Country Code","Indicator Name", "Indicator Code"])

AFG_a=AFG_new.transpose()

AFG_a.columns = ['LE']

AFG_a.head()

afgnew=columnsNamesArr[4:63]

#plt.figure(figsize=(20,6))



plt.figure(figsize=(20,6))

#sns.distplot(a=AFG_a['LE'], label="Afghanistan", kde=False)

#sns.distplot(a=BTN_a['LE'], label="Bhutan", kde=False)

#sns.distplot(a=Chinaaa['LE'], label="China", kde=False)



# KDE plots for each species

sns.kdeplot(data=AFG_a['LE'], label="Afghanistan", shade=True)

sns.kdeplot(data=BTN_a['LE'], label="Bhutan", shade=True)

sns.kdeplot(data=Chinaaa['LE'], label="China", shade=True)



# Add title

plt.title("KDE Plot of Life Expectency in Afghanistan, Bhutan & China")



# Force legend to appear

plt.legend()





# KDE plots for each species

#sns.kdeplot(data=iris_set_data['Petal Length (cm)'], label="Iris-setosa", shade=True)

#sns.kdeplot(data=iris_ver_data['Petal Length (cm)'], label="Iris-versicolor", shade=True)

#sns.kdeplot(data=iris_vir_data['Petal Length (cm)'], label="Iris-virginica", shade=True)