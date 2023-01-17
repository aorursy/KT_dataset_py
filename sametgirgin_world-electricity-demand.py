import pandas as pd

import numpy as np 

import matplotlib as plt

import seaborn as sns
dataset= pd.read_csv("../input/all_energy_statistics.csv")

dataset.head(4)
dataset.tail(4)

dataset.head()

dataset.describe()

dataset.shape
#Drop the column that majorly include null value

dataset.drop("quantity_footnotes", axis=1, inplace=True)
US = dataset[dataset.country_or_area.isin(["United States"])].sort_values('year')

Turkey = dataset[dataset.country_or_area.isin(["Turkey"])].sort_values('year')
dataset.head()
#How many rows are there for each state? 

country_top=dataset["country_or_area"].value_counts()
#To see which states have most rows of data

country_top[:30].plot(kind="barh", rot=0,figsize=(12,5) )
category_top=dataset["category"].value_counts()

category_top[:100].plot(kind="barh", rot=0, figsize=(10,30))
#To pick up the electricity suppy/demand per years for each state

Total_Electricity = dataset[dataset.category.isin(["total_electricity"])].sort_values(['year',"country_or_area"])

Total_Electricity.head()
#Which electricity commodity_transaction is there in dataset

commodity_top=Total_Electricity["commodity_transaction"].value_counts()

commodity_top[:10].plot(kind="barh", rot=0, figsize=(6,4))
#The situation of electricity losses pear per year for each state

Electricity_Demand = Total_Electricity[Total_Electricity.commodity_transaction == "Electricity - Gross demand"].sort_values(["year","country_or_area"])

Electricity_Demand.head()
Electricity_Production= Total_Electricity[Total_Electricity.commodity_transaction == "Electricity - Gross production"].sort_values(["year","country_or_area"])

Electricity_Production.head(2)
#Combine the demand and production dataframes: 

Demand_and_Prod=pd.concat([Electricity_Demand,Electricity_Production]).sort_values(["country_or_area"])

Demand_and_Prod.head(2)
Tur_Electricity=Demand_and_Prod[Demand_and_Prod.country_or_area=="Turkey"].sort_values("year")

Tur_Electricity.head(2)
#Turkey's electricity demand vs production between 1990-2014

Turkey_Elect= Tur_Electricity.pivot_table("quantity","year", columns="commodity_transaction")

Turkey_Elect.head(2)
#We can see the demand and the production vs years in Turkey: 

import matplotlib.pyplot as plt 

Turkey_Elect.plot(kind='line', figsize=(10,4))

#Turkey_Elect.plot(kind='bar',figsize=(10,4))
Electricity_Demand = Total_Electricity[Total_Electricity.commodity_transaction == "Electricity - Gross demand"].sort_values(["year","country_or_area"])
US_Demand = Electricity_Demand[Electricity_Demand.country_or_area == "United States"].sort_values("year")

China_Demand = Electricity_Demand[Electricity_Demand.country_or_area == "China"].sort_values("year")

Germany_Demand = Electricity_Demand[Electricity_Demand.country_or_area == "Germany"].sort_values("year")

Japan_Demand=Electricity_Demand[Electricity_Demand.country_or_area == "Japan"].sort_values("year")

India_Demand=Electricity_Demand[Electricity_Demand.country_or_area == "India"].sort_values("year")

UK_Demand= Electricity_Demand[Electricity_Demand.country_or_area == "United Kingdom"].sort_values("year")

Russia_Demand= Electricity_Demand[Electricity_Demand.country_or_area == "Russian Federation"].sort_values("year")

Brazil_Demand= Electricity_Demand[Electricity_Demand.country_or_area == "Brazil"].sort_values("year")

France_Demand= Electricity_Demand[Electricity_Demand.country_or_area == "France"].sort_values("year")

Mexico_Demand=Electricity_Demand[Electricity_Demand.country_or_area == "Mexico"].sort_values("year")

Italy_Demand= Electricity_Demand[Electricity_Demand.country_or_area == "Italy"].sort_values("year")

Turkey_Demand= Electricity_Demand[Electricity_Demand.country_or_area == "Turkey"].sort_values("year")

Korea_Demand= Electricity_Demand[Electricity_Demand.country_or_area == "Korea, Republic of"].sort_values("year") 

Saudi_Demand=Electricity_Demand[Electricity_Demand.country_or_area == "Saudi Arabia"].sort_values("year") 

Canada_Demand= Electricity_Demand[Electricity_Demand.country_or_area == "Canada"].sort_values("year") 

Australia_Demand= Electricity_Demand[Electricity_Demand.country_or_area == "Australia"].sort_values("year") 

Argentina_Demand= Electricity_Demand[Electricity_Demand.country_or_area == "Argentina"].sort_values("year") 

South_Africa_Demand=Electricity_Demand[Electricity_Demand.country_or_area == "South Africa"].sort_values("year") 

Indonesia_Demand=Electricity_Demand[Electricity_Demand.country_or_area == "Indonesia"].sort_values("year") 
#Put all the states quantity vs years on the axis:



y1=US_Demand.quantity

x1=US_Demand.year

y2=China_Demand.quantity

x2=China_Demand.year

y3=Germany_Demand.quantity

x3=Germany_Demand.year

y4=Japan_Demand.quantity

x4=Japan_Demand.year

y5=UK_Demand.quantity

x5=UK_Demand.year

y6=Russia_Demand.quantity

x6=Russia_Demand.year

y7=Brazil_Demand.quantity

x7=Brazil_Demand.year

y8=France_Demand.quantity

x8=France_Demand.year

y9=Mexico_Demand.quantity

x9=Mexico_Demand.year

y10=Italy_Demand.quantity

x10=Italy_Demand.year

y11=Turkey_Demand.quantity

x11=Turkey_Demand.year

y12=Korea_Demand.quantity

x12=Korea_Demand.year

y13=Saudi_Demand.quantity

x13=Saudi_Demand.year

y14=Canada_Demand.quantity

x14=Canada_Demand.year

y15=Australia_Demand.quantity

x15=Australia_Demand.year

y16=Argentina_Demand.quantity

x16=Argentina_Demand.year

y17=Indonesia_Demand.quantity

x17= Indonesia_Demand.year

y18=India_Demand.quantity

x18=India_Demand.year

y19=South_Africa_Demand.quantity

x19=South_Africa_Demand.year
plt.figure(figsize=(20,10))

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.plot(x1,y1,label="US")

plt.plot(x2,y2,'r',label="China")

plt.plot(x3,y3,'y',label="Germany")

plt.plot(x4,y4,'k',label="Japan")

plt.plot(x5,y5,'g',label="UK")

plt.plot(x6,y6,'c',label="Russia")

plt.plot(x7,y7,'m',label="Brazil")

plt.plot(x8,y8,'orange',label="France")

plt.plot(x9,y9,label="Mexico")

plt.plot(x10,y10,label="Italy")

plt.plot(x11,y11,label="Turkey")

plt.plot(x12,y12,label="Korea")

plt.plot(x13,y13,label="Saudi")

plt.plot(x14,y14,label="Canada")

plt.plot(x15,y15,label="Australia")

plt.plot(x16,y16,label="Argentina")

plt.plot(x17,y17,label="Indonesia")

plt.plot(x18,y18,label="India")

plt.plot(x19,y19,label="South Africa")



plt.legend(fontsize=16)

plt.ylabel("Millions of Kilowatts-Hour",fontsize=20)

plt.xlabel('Year',fontsize=20)

plt.title('G-20 Electiricty Demand per year',fontsize=24)

plt.xlim(1989.8, 2014.2)

plt.show()
plt.figure(figsize=(20,5))

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.plot(x1,y1,label="US")

"""plt.plot(x2,y2,'r',label="China")"""

plt.plot(x3,y3,'y',label="Germany")

plt.plot(x4,y4,'k',label="Japan")

plt.plot(x5,y5,'g',label="UK")

"""plt.plot(x6,y6,'c',label="Russia")

plt.plot(x7,y7,'m',label="Brazil")"""

plt.plot(x8,y8,'orange',label="France")

"""plt.plot(x9,y9,label="Mexico")"""

plt.plot(x10,y10,label="Italy")

"""plt.plot(x11,y11,label="Turkey")

plt.plot(x12,y12,label="Korea")

plt.plot(x13,y13,label="Saudi")"""

plt.plot(x14,y14,label="Canada")

"""plt.plot(x15,y15,label="Australia")

plt.plot(x16,y16,label="Argentina")

plt.plot(x17,y17,label="Indonesia")

plt.plot(x18,y18,label="India")

plt.plot(x19,y19,label="South Africa")"""



plt.legend(fontsize=16)

plt.ylabel("Millions of Kilowatts-Hour",fontsize=20)

plt.xlabel('Year',fontsize=20)

plt.title('G-7 Electiricty Demand per year',fontsize=24)

plt.xlim(1989.8, 2014.2)

plt.show()
plt.figure(figsize=(20,5))

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.plot(x1,y1,label="US")

plt.plot(x2,y2,'r',label="China")



plt.legend(fontsize=16)

plt.ylabel("Millions of Kilowatts-Hour",fontsize=20)

plt.xlabel('Year',fontsize=20)

plt.title('US vs China Electiricty Demand per year',fontsize=24)

plt.xlim(1989.8, 2014.2)

plt.show()
plt.figure(figsize=(20,5))

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

"""plt.plot(x1,y1,label="US")"""

"""plt.plot(x2,y2,'r',label="China")"""

plt.plot(x3,y3,'y',label="Germany")

"""plt.plot(x4,y4,'k',label="Japan")"""

plt.plot(x5,y5,'g',label="UK")

"""plt.plot(x6,y6,'c',label="Russia")

plt.plot(x7,y7,'m',label="Brazil")"""

plt.plot(x8,y8,'orange',label="France")

"""plt.plot(x9,y9,label="Mexico")"""

plt.plot(x10,y10,label="Italy")

"""plt.plot(x11,y11,label="Turkey")

plt.plot(x12,y12,label="Korea")

plt.plot(x13,y13,label="Saudi")"""

plt.plot(x14,y14,label="Canada")

"""plt.plot(x15,y15,label="Australia")

plt.plot(x16,y16,label="Argentina")

plt.plot(x17,y17,label="Indonesia")

plt.plot(x18,y18,label="India")

plt.plot(x19,y19,label="South Africa")"""



plt.legend(fontsize=16)

plt.ylabel("Millions of Kilowatts-Hour",fontsize=20)

plt.xlabel('Year',fontsize=20)

plt.title('Germany-UK-France-Italy-Canada Electiricty Demand per year',fontsize=24)

plt.xlim(1989.8, 2014.2)

plt.show()
plt.figure(figsize=(20,10))

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.plot(x7,y7,'m',label="Brazil")

plt.plot(x9,y9,label="Mexico")

plt.plot(x11,y11,label="Turkey")

plt.plot(x12,y12,label="Korea")

plt.plot(x13,y13,label="Saudi")

plt.plot(x15,y15,label="Australia")

plt.plot(x16,y16,label="Argentina")

plt.plot(x17,y17,label="Indonesia")

plt.plot(x19,y19,label="South Africa")



plt.legend(fontsize=16)

plt.ylabel("Millions of Kilowatts-Hour",fontsize=20)

plt.xlabel('Year',fontsize=20)

plt.title('G-20 Developing States Electiricty Demand per year',fontsize=24)

plt.xlim(1989.8, 2014.2)

plt.show()
plt.figure(figsize=(20,10))

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.plot(x6,y6,'c',label="Russia")

plt.plot(x18,y18,label="India")



plt.legend(fontsize=16)

plt.ylabel("Millions of Kilowatts-Hour",fontsize=20)

plt.xlabel('Year',fontsize=20)

plt.title('Russia vs India Electiricty Demand per year',fontsize=24)

plt.xlim(1989.8, 2014.2)

plt.show()
Electricity_Demand = Total_Electricity[Total_Electricity.commodity_transaction == "Electricity - Gross demand"].sort_values(["year","country_or_area"])

Electricity_Demand.head()
World= Electricity_Demand['quantity'].groupby(Electricity_Demand['year']).sum()

World_Electricity_By_Year= pd.DataFrame({'year':World.index, 'quantity':World.values})

World_Electricity_By_Year.head()
bar_width =.7

plt.figure(figsize=(15,5))

plt.xticks(fontsize=12)

plt.yticks(fontsize=14)

x1=World_Electricity_By_Year.year

plt.bar(x1, World_Electricity_By_Year["quantity"],bar_width, color='c',capstyle= 'projecting', label="World Electricity Demand", alpha=.5)



plt.legend(fontsize=16)

plt.xlabel("Years", fontsize=20)

plt.ylabel("Millions of Kilowatts-Hour", fontsize=20)

plt.title("World Total Electricity Demand Between 1990-2014", fontsize=24)

#plt.xticks(x + bar_width / 6, US_Wind["year"])

#plt.xlim(-.5,24.5)

plt.show()

"""METHODS in Python: 



pandas.DataFrame.pct_change:



Percentage change between the current and a prior element.



Computes the percentage change from the immediately previous row by default. 

This is useful in comparing the percentage of change in a time series of elements."""
#World_Electricity_By_Year['Percentage Growth'] = (World_Electricity_By_Year['Open'] / df['Close'].shift(1) - 1).fillna(0)

World_Electricity_By_Year['Percentage_Growth']=100*World_Electricity_By_Year.sort_values(['year'])['quantity'].pct_change()
World_Electricity_By_Year.head()
x1=World_Electricity_By_Year.year

y1=World_Electricity_By_Year.Percentage_Growth
plt.figure(figsize=(20,5))

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.plot(x1,y1,'c',label="Growth Percentage")



plt.legend(fontsize=16)

plt.ylabel("%",fontsize=20)

plt.xlabel('Year',fontsize=20)

plt.title('World Total Electricty Growth Percentage Between 1990-2014',fontsize=24)

plt.xlim(1989.8, 2014.2)

plt.show()
d = {'year': [1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014],

         'gdp_growth': [2.919, 1.431, 1.785,1.536, 3.014, 3.039,3.391,3.704,2.542,3.259, 4.392, 1.924, 2.202,2.904,4.372,3.823,

                       4.276,4.182,1.802,-1.741,4.358, 3.152, 2.521, 2.653, 2.842]}

world_econ = pd.DataFrame(data=d)

world_econ.head()
electricity_vs_gdp= pd.merge(World_Electricity_By_Year,world_econ)

electricity_vs_gdp.head()
x1=electricity_vs_gdp.year

y1=electricity_vs_gdp.gdp_growth

x2=electricity_vs_gdp.year

y2=electricity_vs_gdp.Percentage_Growth



plt.figure(figsize=(20,5))

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.plot(x2,y2,'c',label="Electricity Demand Growth Percentage")

plt.plot(x1,y1,'r',label="GDP Growth Percentage")



plt.legend(fontsize=16)

plt.ylabel("%",fontsize=20)

plt.xlabel('Year',fontsize=20)

plt.title('World Total Electricty & GDP Growth Between 1990-2014',fontsize=24)

plt.xlim(1989.8, 2014.2)

plt.show()