# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas import DataFrame
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth',-1)
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
from plotly.graph_objs import *



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

sns.set_style('whitegrid')
%matplotlib inline
plt.rcParams["patch.force_edgecolor"]= True
df = pd.read_csv("../input/all_energy_statistics.csv")
df.describe()
#List of countries to work on.
US = df[df.country_or_area.isin(["United States"])].sort_values('year')
BR= df[df.country_or_area.isin(['Brazil'])].sort_values('year')
CAN = df[df.country_or_area.isin(["Canada"])].sort_values('year')
CHI = df[df.country_or_area.isin(["China"])].sort_values('year')
IND = df[df.country_or_area.isin(['India'])].sort_values('year')
JAP = df[df.country_or_area.isin(['Japan'])].sort_values('year')
UK =df[df.country_or_area.isin(['United Kingdom'])].sort_values('year')
#List of countries to make the European Union.
SP = df[df.country_or_area.isin(["Spain"])].sort_values('year')
ITA = df[df.country_or_area.isin(['Italy'])].sort_values('year')
GER = df[df.country_or_area.isin(["Germany"])].sort_values('year')
FRA = df[df.country_or_area.isin(["France"])].sort_values('year')
NETH = df[df.country_or_area.isin(['Netherlands'])].sort_values('year')
#List of countries to work on.
US_Wind = US[US.commodity_transaction == "Electricity - total wind production"].sort_values("year")
BR_Wind = BR[BR.commodity_transaction == "Electricity - total wind production"].sort_values("year")
CAN_Wind = CAN[CAN.commodity_transaction == "Electricity - total wind production"].sort_values("year")
CHI_Wind = CHI[CHI.commodity_transaction == "Electricity - total wind production"].sort_values("year")
IND_Wind = IND[IND.commodity_transaction == "Electricity - total wind production"].sort_values("year")
JAP_Wind = JAP[JAP.commodity_transaction == "Electricity - total wind production"].sort_values("year")
UK_Wind = UK[UK.commodity_transaction == "Electricity - total wind production"].sort_values("year")

#List of countries to make the European Union.
SP_Wind = SP[SP.commodity_transaction == "Electricity - total wind production"].sort_values("year")
ITA_Wind = ITA[ITA.commodity_transaction == "Electricity - total wind production"].sort_values("year")
FRA_Wind = FRA[FRA.commodity_transaction == "Electricity - total wind production"].sort_values("year")
GER_Wind = GER[GER.commodity_transaction == "Electricity - total wind production"].sort_values("year")
NETH_Wind = NETH[NETH.commodity_transaction == "Electricity - total wind production"].sort_values("year")

EU_Wind = pd.merge(SP_Wind,ITA_Wind,on='year',how='outer')
EU_Wind.rename(columns={'country_or_area_x':'Spain','commodity_transaction_x':
                   'commodity1','unit_x':'unit1','quantity_x':'quantity1',
                   'country_or_area_y':'Italy','quantity_y':'quantity2'},inplace=True)

EU_Wind.drop(['commodity_transaction_y','unit_y','category_y'], axis=1,inplace=True)

#Adding France.
EU_Wind = EU_Wind.merge(FRA_Wind,on='year',how='outer')
EU_Wind.rename(columns={'country_or_area':'France','quantity':'quantity3',},inplace=True)
EU_Wind.rename(columns={'country_or_area':'France','quantity':'quantity3',},inplace=True)
EU_Wind.drop(['commodity_transaction','unit','category'], axis=1,inplace=True)

#Adding Germany.
EU_Wind = EU_Wind.merge(GER_Wind,on='year',how='outer')
EU_Wind.rename(columns={'country_or_area':'Germany','quantity':'quantity4',},inplace=True)
EU_Wind.drop(['commodity_transaction','unit','category'], axis=1,inplace=True)

#Adding Netherlands.
EU_Wind = EU_Wind.merge(NETH_Wind,on='year',how='outer')
EU_Wind.rename(columns={'country_or_area':'Netherlands','quantity':'quantity5',},inplace=True)
EU_Wind.drop(['commodity_transaction','unit','category'], axis=1,inplace=True) 

  #Here I would fill all the Nan values.
values = {'France':'France','quantity3':0,'Germany':'Germany',
           'quantity4':0,'Netherlands':'Netherlands','quantity5':0}

#Here I would add all the columns to create a total quantity for the countries that would represent the EU.
EU_Wind.fillna(value=values,inplace=True)

#If you are interested you can drop the countries and their values, but I would let them in the data set, maybe they would be 
#useful in another time.

#Here I would add all the quantities from all countries to create one European quantity
EU_Wind['quantity'] = EU_Wind['quantity1'] + EU_Wind['quantity2'] + EU_Wind['quantity3'] + EU_Wind['quantity4'] + EU_Wind['quantity5']
EU_Wind.head()
#List of countries to work on.
US_Solar = US[US.commodity_transaction == "Electricity - total solar production"].sort_values("year")
BR_Solar = BR[BR.commodity_transaction == "Electricity - total solar production"].sort_values("year")
CAN_Solar = CAN[CAN.commodity_transaction == "Electricity - total solar production"].sort_values("year")
CHI_Solar = CHI[CHI.commodity_transaction == "Electricity - total solar production"].sort_values("year")
IND_Solar = IND[IND.commodity_transaction == "Electricity - total solar production"].sort_values("year")
JAP_Solar = JAP[JAP.commodity_transaction == "Electricity - total solar production"].sort_values("year")
UK_Solar = UK[UK.commodity_transaction == "Electricity - total solar production"].sort_values("year")

#List of countries to make the European Union.
SP_Solar = SP[SP.commodity_transaction == "Electricity - total solar production"].sort_values("year")
ITA_Solar = ITA[ITA.commodity_transaction == "Electricity - total wind production"].sort_values("year")
FRA_Solar = FRA[FRA.commodity_transaction == "Electricity - total solar production"].sort_values("year")
GER_Solar = GER[GER.commodity_transaction == "Electricity - total solar production"].sort_values("year")
NETH_Solar = NETH[NETH.commodity_transaction == "Electricity - total solar production"].sort_values("year")

 #Here I would create the European Union based on Solar Production, the merge would be done on year, 
#also the name of the columns would be modify until the last version is complete.
EU_Solar = pd.merge(SP_Solar,ITA_Solar,on='year',how='outer')
EU_Solar.rename(columns={'country_or_area_x':'Spain','commodity_transaction_x':
                   'commodity1','unit_x':'unit1','quantity_x':'quantity1',
                   'country_or_area_y':'Italy','quantity_y':'quantity2'},inplace=True)

#Adding France.
EU_Solar = EU_Solar.merge(FRA_Solar,on='year',how='outer')
EU_Solar.rename(columns={'country_or_area':'France','quantity':'quantity3',},inplace=True)
EU_Solar.drop(['commodity_transaction','unit','category'], axis=1,inplace=True)

#Adding Germany.
EU_Solar = EU_Solar.merge(GER_Solar,on='year',how='outer')
EU_Solar.rename(columns={'country_or_area':'Germany','quantity':'quantity4',},inplace=True)
EU_Solar.drop(['commodity_transaction','unit','category'], axis=1,inplace=True)

#Adding Netherlands.
EU_Solar = EU_Solar.merge(NETH_Solar,on='year',how='outer')
EU_Solar.rename(columns={'country_or_area':'Netherlands','quantity':'quantity5',},inplace=True)
EU_Solar.drop(['commodity_transaction','unit','category'], axis=1,inplace=True)

 #Here I would fill all the Nan values.
values = {'France':'France','quantity3':0,'Germany':'Germany',
           'quantity4':0,'Netherlands':'Netherlands','quantity5':0}

EU_Solar.fillna(value=values,inplace=True)

#Here I would add all the columns to create a total quantity for the countries that would represent the EU.
EU_Solar['quantity'] = EU_Solar['quantity1'] + EU_Solar['quantity2'] + EU_Solar['quantity3'] + EU_Solar['quantity4'] + EU_Solar['quantity5']
EU_Solar.head()
EU_Solar.describe()
EU_Solar
#List of countries to work on with Nuclear production.
US_Nuclear = US[US.commodity_transaction == "Electricity - total nuclear production"].sort_values("year")
BR_Nuclear = BR[BR.commodity_transaction == "Electricity - total nuclear production"].sort_values("year")
CAN_Nuclear = CAN[CAN.commodity_transaction == "Electricity - total nuclear production"].sort_values("year")
CHI_Nuclear = CHI[CHI.commodity_transaction == "Electricity - total nuclear production"].sort_values("year")
IND_Nuclear = IND[IND.commodity_transaction == "Electricity - total nuclear production"].sort_values("year")
JAP_Nuclear = JAP[JAP.commodity_transaction == "Electricity - total nuclear production"].sort_values("year")
UK_Nuclear = UK[UK.commodity_transaction == "Electricity - total nuclear production"].sort_values("year")

#List of countries to make the European Union. Italy would be out because it do not have nuclear power"
SP_Nuclear = SP[SP.commodity_transaction == "Electricity - total nuclear production"].sort_values("year")
FRA_Nuclear = FRA[FRA.commodity_transaction == "Electricity - total nuclear production"].sort_values("year")
GER_Nuclear = GER[GER.commodity_transaction == "Electricity - total nuclear production"].sort_values("year")
NETH_Nuclear = NETH[NETH.commodity_transaction == "Electricity - total nuclear production"].sort_values("year")


           #Here I would create the European Union based on Nuclear Production, the merge would be done on year,
#also the name of the columns would be modify until the last version is complete.
EU_Nuclear = pd.merge(SP_Nuclear,FRA_Nuclear,on='year',how='outer')
EU_Nuclear.rename(columns={'country_or_area_x':'Spain','commodity_transaction_x':
                   'commodity1','unit_x':'unit1','quantity_x':'quantity1',
                   'country_or_area_y':'France','quantity_y':'quantity2'},inplace=True)

#Adding Germany
EU_Nuclear = EU_Nuclear.merge(GER_Nuclear,on='year',how='outer')
EU_Nuclear.rename(columns={'country_or_area':'Germany','quantity':'quantity3',},inplace=True)

#Adding Netherlands
EU_Nuclear = EU_Nuclear.merge(NETH_Nuclear,on='year',how='outer')
EU_Nuclear.rename(columns={'country_or_area':'Netherlands','quantity':'quantity4',},inplace=True)

 #Here I would fill all the Nan values.
values = {'Germany':'Germany','quantity3':0}
EU_Nuclear.fillna(value=values,inplace=True)

#Here I would add all the columns to create a total quantity for the countries that would represent the EU.
EU_Nuclear['quantity'] = EU_Nuclear['quantity1'] + EU_Nuclear['quantity2'] + EU_Nuclear['quantity3'] + EU_Nuclear['quantity4'] 
EU_Nuclear.head()
#This is for the Solar Production
y1b = US_Solar.quantity
x1b = US_Solar.year
y2b = CAN_Solar.quantity
x2b = CAN_Solar.year
y3b = CHI_Solar.quantity
x3b = CHI_Solar.year
x4b = UK_Solar.year
y4b = UK_Solar.quantity
x5b = EU_Solar.year
y5b = EU_Solar.quantity
x6b = BR_Solar.year
y6b = BR_Solar.quantity
x7b = IND_Solar.year
y7b = IND_Solar.quantity
x8b = JAP_Solar.year
y8b = JAP_Solar.quantity

plt.figure(figsize=(15,10))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.plot(x1b,y1b,label="US")
plt.plot(x2b,y2b,'r--',label="Canada")
plt.plot(x3b,y3b,'y--',label="China")
plt.plot(x4b,y4b,'k',label="UK")
plt.plot(x5b,y5b,'g',label="European Union")
plt.plot(x6b,y6b,'c',label="Brazil")
plt.plot(x7b,y7b,'m',label="India")
plt.plot(x8b,y8b,'orange',label="Japan")




plt.legend(fontsize=16)
plt.ylabel("Millions of Kilowatts-Hour",fontsize=20)
plt.xlabel('Year',fontsize=20)
plt.title('Total Solar production of all countries',fontsize=24)
plt.xlim(1989.8, 2014.2)
plt.show()
#This is for the Solar Production
y1c = SP_Solar.quantity
x1c = SP_Solar.year
y2c = ITA_Solar.quantity
x2c = ITA_Solar.year
y3c = FRA_Solar.quantity
x3c = FRA_Solar.year
x4c = UK_Solar.year
y4c = UK_Solar.quantity
x5c = GER_Solar.year
y5c = GER_Solar.quantity
x6c = NETH_Solar.year
y6c = NETH_Solar.quantity

plt.figure(figsize=(15,10))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.plot(x1c,y1c,label="Spain")
plt.plot(x2c,y2c,'r--',label="Italy")
plt.plot(x3c,y3c,'y--',label="France")
plt.plot(x4c,y4c,'k',label="UK")
plt.plot(x5c,y5c,'g',label="Germany")
plt.plot(x6c,y6c,'c',label="Netherlands")
plt.legend(fontsize=16)
plt.ylabel("Millions of Kilowatts-Hour",fontsize=20)
plt.xlabel('Year',fontsize=20)
plt.title('Total Solar production of EU countries',fontsize=24)
plt.xlim(1989.8, 2014.2)
plt.show()
br= .5
plt.figure(figsize=(12,8))
plt.barh(US_Solar["year"],US_Solar["quantity"],height = br , label="US Solar production")
plt.barh(EU_Solar["year"] , EU_Solar["quantity"],align ='edge', height=br-.2, label="EU Solar production" )
plt.yticks(EU_Solar["year"])
range = np.arange(2000,2015)
plt.legend(fontsize=16)
plt.ylabel("Years", fontsize=14)
plt.xlabel("Quantity", fontsize=14)
plt.title("Total Solar production in the US and EU", fontsize=16)
plt.xlim(0, 73500)
plt.show()
br= .5
plt.figure(figsize=(12,8))
plt.barh(SP_Solar["year"],SP_Solar["quantity"],height = br , label="Spain Solar production")
plt.barh(GER_Solar["year"] , GER_Solar["quantity"],align ='edge', height=br-.2, label="Germany Solar production" )
plt.yticks(SP_Solar["year"])

plt.legend(fontsize=16)
plt.ylabel("Years", fontsize=14)
plt.xlabel("Quantity", fontsize=14)
plt.title("Comparison between Spain and Germany", fontsize=16)
range = np.arange(2000,2015)
plt.show()
br= .5
plt.figure(figsize=(16,12))
plt.barh(SP_Nuclear["year"] , SP_Nuclear["quantity"],align ='edge', height=br-.3, label="Nuclear production" )
plt.barh(SP_Wind["year"] , SP_Wind["quantity"],align ='edge', height=br-.1, label="Wind production" )
plt.barh(SP_Solar["year"],SP_Solar["quantity"],height = br , label="Solar production")
range = np.arange(2000,2015)
plt.legend(mode = "expand",fontsize=12)
plt.ylabel("Years", fontsize=14)
plt.xlabel("Quantity", fontsize=14)
plt.title("Total Wind, Solar and Nuclear production in Spain", fontsize=14)
plt.xlim()
plt.legend(loc="lower right") 
plt.show()