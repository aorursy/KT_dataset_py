import pandas as pd 
import numpy as np
import matplotlib as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline

dataset= pd.read_csv("../input/all_energy_statistics.csv")
#Drop the column that majorly include null value
dataset.drop("quantity_footnotes", axis=1, inplace=True)
dataset.head(3)
#dataset.commodity_transaction.isnull().sum() (commodity_transaction ve category kolonlarında eksik yok)
#How many categories in the dataset
print(dataset['category'].value_counts().count())
#How many commodity_transaction in the dataset
print(dataset['commodity_transaction'].value_counts().count())
#G-7 COUNTRIES: 
US = dataset[dataset.country_or_area.isin(["United States"])].sort_values('year')
CAN = dataset[dataset.country_or_area.isin(["Canada"])].sort_values('year')
JAP = dataset[dataset.country_or_area.isin(['Japan'])].sort_values('year')
UK =dataset[dataset.country_or_area.isin(['United Kingdom'])].sort_values('year')
GER = dataset[dataset.country_or_area.isin(["Germany"])].sort_values('year')
ITA = dataset[dataset.country_or_area.isin(['Italy'])].sort_values('year')
FRA = dataset[dataset.country_or_area.isin(["France"])].sort_values('year')
#List of various EU countries 
SP = dataset[dataset.country_or_area.isin(["Spain"])].sort_values('year')
NETH = dataset[dataset.country_or_area.isin(['Netherlands'])].sort_values('year')
NOR = dataset[dataset.country_or_area.isin(["Norway"])].sort_values('year')
POR = dataset[dataset.country_or_area.isin(["Portugal"])].sort_values('year')
#DEVELOPING STATES: 
TUR = dataset[dataset.country_or_area.isin(["Turkey"])].sort_values('year')
SKOR = dataset[dataset.country_or_area.isin(["Korea, Republic of"])].sort_values('year')
SAUD = dataset[dataset.country_or_area.isin(["Saudi Arabia"])].sort_values('year')
AUST= dataset[dataset.country_or_area.isin(["Australia"])].sort_values('year') 
ARG= dataset[dataset.country_or_area.isin(["Argentina"])].sort_values('year') 
INDO= dataset[dataset.country_or_area.isin(["Indonesia"])].sort_values('year') 
MEX= dataset[dataset.country_or_area.isin(["Mexico"])].sort_values('year') 
#BRICS STATES: 
BR= dataset[dataset.country_or_area.isin(['Brazil'])].sort_values('year')
RUS = dataset[dataset.country_or_area.isin(['Russian Federation'])].sort_values('year')
IND = dataset[dataset.country_or_area.isin(['India'])].sort_values('year')
CHI = dataset[dataset.country_or_area.isin(["China"])].sort_values('year')
SAfr = dataset[dataset.country_or_area.isin(["South Africa"])].sort_values('year')
gas_oil= dataset[dataset.category=="gas_oil_diesel_oil"].sort_values("year")
gas_oil["commodity_transaction"].value_counts().head(10)
conventional_crude=dataset[dataset.category=="conventional_crude_oil"].sort_values("year")
conventional_crude["commodity_transaction"].value_counts().head(10)
lpg=dataset[dataset.category=="liquified_petroleum_gas"].sort_values("year")
lpg["commodity_transaction"].value_counts().head(10)
biodiesel=dataset[dataset.category=="biodiesel"].sort_values("year")
biodiesel["commodity_transaction"].value_counts().head(6)
                                                        #Crude Oil: 
#G-7 States: 
US_CrudeSupply=US[US.commodity_transaction=="Conventional crude oil - total energy supply"].sort_values("year")
US_CrudeProd=US[US.commodity_transaction=="Conventional crude oil - production"].sort_values("year")
US_Refinery=US[US.commodity_transaction=="Crude petroleum - refinery capacity"].sort_values("year")
US_CrudeExports=US[US.commodity_transaction=="Conventional crude oil - exports"].sort_values("year")
US_CrudeImports=US[US.commodity_transaction=="Conventional crude oil - imports"].sort_values("year")

CAN_CrudeSupply = CAN[CAN.commodity_transaction=="Conventional crude oil - total energy supply"].sort_values("year")
CAN_CrudeProd=CAN[CAN.commodity_transaction=="Conventional crude oil - production"].sort_values("year")
CAN_Refinery=CAN[CAN.commodity_transaction=="Crude petroleum - refinery capacity"].sort_values("year")
CAN_CrudeExports=CAN[CAN.commodity_transaction=="Conventional crude oil - exports"].sort_values("year")
CAN_CrudeImports=CAN[CAN.commodity_transaction=="Conventional crude oil - imports"].sort_values("year")

JAP_CrudeSupply = JAP[JAP.commodity_transaction=="Conventional crude oil - total energy supply"].sort_values("year")
JAP_CrudeProd=JAP[JAP.commodity_transaction=="Conventional crude oil - production"].sort_values("year")
JAP_Refinery=JAP[JAP.commodity_transaction=="Crude petroleum - refinery capacity"].sort_values("year")
JAP_CrudeExports=JAP[JAP.commodity_transaction=="Conventional crude oil - exports"].sort_values("year")
JAP_CrudeImports=JAP[JAP.commodity_transaction=="Conventional crude oil - imports"].sort_values("year")

UK_CrudeSupply = UK[UK.commodity_transaction=="Conventional crude oil - total energy supply"].sort_values("year")
UK_CrudeProd=UK[UK.commodity_transaction=="Conventional crude oil - production"].sort_values("year")
UK_Refinery=UK[UK.commodity_transaction=="Crude petroleum - refinery capacity"].sort_values("year")
UK_CrudeExports=UK[UK.commodity_transaction=="Conventional crude oil - exports"].sort_values("year")
UK_CrudeImports=UK[UK.commodity_transaction=="Conventional crude oil - imports"].sort_values("year")

GER_CrudeSupply = GER[GER.commodity_transaction=="Conventional crude oil - total energy supply"].sort_values("year")
GER_CrudeProd=GER[GER.commodity_transaction=="Conventional crude oil - production"].sort_values("year")
GER_Refinery=GER[GER.commodity_transaction=="Crude petroleum - refinery capacity"].sort_values("year")
GER_CrudeExports=GER[GER.commodity_transaction=="Conventional crude oil - exports"].sort_values("year")
GER_CrudeImports=GER[GER.commodity_transaction=="Conventional crude oil - imports"].sort_values("year")

ITA_CrudeSupply = ITA[ITA.commodity_transaction=="Conventional crude oil - total energy supply"].sort_values("year")
ITA_CrudeProd=ITA[ITA.commodity_transaction=="Conventional crude oil - production"].sort_values("year")
ITA_Refinery=ITA[ITA.commodity_transaction=="Crude petroleum - refinery capacity"].sort_values("year")
ITA_CrudeExports=ITA[ITA.commodity_transaction=="Conventional crude oil - exports"].sort_values("year")
ITA_CrudeImports=ITA[ITA.commodity_transaction=="Conventional crude oil - imports"].sort_values("year")

FRA_CrudeSupply =FRA[FRA.commodity_transaction=="Conventional crude oil - total energy supply"].sort_values("year")
FRA_CrudeProd=FRA[FRA.commodity_transaction=="Conventional crude oil - production"].sort_values("year")
FRA_Refinery=FRA[FRA.commodity_transaction=="Crude petroleum - refinery capacity"].sort_values("year")
FRA_CrudeExports=FRA[FRA.commodity_transaction=="Conventional crude oil - exports"].sort_values("year")
FRA_CrudeImports=FRA[FRA.commodity_transaction=="Conventional crude oil - imports"].sort_values("year")

#List of various EU countries 
SP_CrudeSupply=SP[SP.commodity_transaction=="Conventional crude oil - total energy supply"].sort_values("year")
SP_CrudeProd=SP[SP.commodity_transaction=="Conventional crude oil - production"].sort_values("year")
SP_Refinery=SP[SP.commodity_transaction=="Crude petroleum - refinery capacity"].sort_values("year")
SP_CrudeExports=SP[SP.commodity_transaction=="Conventional crude oil - exports"].sort_values("year")
SP_CrudeImports=SP[SP.commodity_transaction=="Conventional crude oil - imports"].sort_values("year")

NETH_CrudeSupply=NETH[NETH.commodity_transaction=="Conventional crude oil - total energy supply"].sort_values("year")
NETH_CrudeProd=NETH[NETH.commodity_transaction=="Conventional crude oil - production"].sort_values("year")
NETH_Refinery=NETH[NETH.commodity_transaction=="Crude petroleum - refinery capacity"].sort_values("year")
NETH_CrudeExports=NETH[NETH.commodity_transaction=="Conventional crude oil - exports"].sort_values("year")
NETH_CrudeImports=NETH[NETH.commodity_transaction=="Conventional crude oil - imports"].sort_values("year")

NOR_CrudeSupply=NOR[NOR.commodity_transaction=="Conventional crude oil - total energy supply"].sort_values("year")
NOR_CrudeProd=NOR[NOR.commodity_transaction=="Conventional crude oil - production"].sort_values("year")
NOR_Refinery=NOR[NOR.commodity_transaction=="Crude petroleum - refinery capacity"].sort_values("year")
NOR_CrudeExports=NOR[NOR.commodity_transaction=="Conventional crude oil - exports"].sort_values("year")
NOR_CrudeImports=NOR[NOR.commodity_transaction=="Conventional crude oil - imports"].sort_values("year")

POR_CrudeSupply=POR[POR.commodity_transaction=="Conventional crude oil - total energy supply"].sort_values("year")
POR_CrudeProd=POR[POR.commodity_transaction=="Conventional crude oil - production"].sort_values("year")
POR_Refinery=POR[POR.commodity_transaction=="Crude petroleum - refinery capacity"].sort_values("year")
POR_CrudeExports=POR[POR.commodity_transaction=="Conventional crude oil - exports"].sort_values("year")
POR_CrudeImports=POR[POR.commodity_transaction=="Conventional crude oil - imports"].sort_values("year")

#DEVELOPING STATES: 
TUR_CrudeSupply=TUR[TUR.commodity_transaction=="Conventional crude oil - total energy supply"].sort_values("year")
TUR_CrudeProd=TUR[TUR.commodity_transaction=="Conventional crude oil - production"].sort_values("year")
TUR_Refinery=TUR[TUR.commodity_transaction=="Crude petroleum - refinery capacity"].sort_values("year")
TUR_CrudeExports=TUR[TUR.commodity_transaction=="Conventional crude oil - exports"].sort_values("year")
TUR_CrudeImports=TUR[TUR.commodity_transaction=="Conventional crude oil - imports"].sort_values("year")

SKOR_CrudeSupply=SKOR[SKOR.commodity_transaction=="Conventional crude oil - total energy supply"].sort_values("year")
SKOR_CrudeProd=SKOR[SKOR.commodity_transaction=="Conventional crude oil - production"].sort_values("year")
SKOR_Refinery=SKOR[SKOR.commodity_transaction=="Crude petroleum - refinery capacity"].sort_values("year")
SKOR_CrudeExports=SKOR[SKOR.commodity_transaction=="Conventional crude oil - exports"].sort_values("year")
SKOR_CrudeImports=SKOR[SKOR.commodity_transaction=="Conventional crude oil - imports"].sort_values("year")

SAUD_CrudeSupply=SAUD[SAUD.commodity_transaction=="Conventional crude oil - total energy supply"].sort_values("year")
SAUD_CrudeProd=SAUD[SAUD.commodity_transaction=="Conventional crude oil - production"].sort_values("year")
SAUD_Refinery=SAUD[SAUD.commodity_transaction=="Crude petroleum - refinery capacity"].sort_values("year")
SAUD_CrudeExports=SAUD[SAUD.commodity_transaction=="Conventional crude oil - exports"].sort_values("year")
SAUD_CrudeImports=SAUD[SAUD.commodity_transaction=="Conventional crude oil - imports"].sort_values("year")

AUST_CrudeSupply=AUST[AUST.commodity_transaction=="Conventional crude oil - total energy supply"].sort_values("year")
AUST_CrudeProd=AUST[AUST.commodity_transaction=="Conventional crude oil - production"].sort_values("year")
AUST_Refinery=AUST[AUST.commodity_transaction=="Crude petroleum - refinery capacity"].sort_values("year")
AUST_CrudeExports=AUST[AUST.commodity_transaction=="Conventional crude oil - exports"].sort_values("year")
AUST_CrudeImports=AUST[AUST.commodity_transaction=="Conventional crude oil - imports"].sort_values("year")

ARG_CrudeSupply=ARG[ARG.commodity_transaction=="Conventional crude oil - total energy supply"].sort_values("year")
ARG_CrudeProd=ARG[ARG.commodity_transaction=="Conventional crude oil - production"].sort_values("year")
ARG_Refinery=ARG[ARG.commodity_transaction=="Crude petroleum - refinery capacity"].sort_values("year")
ARG_CrudeExports=ARG[ARG.commodity_transaction=="Conventional crude oil - exports"].sort_values("year")
ARG_CrudeImports=ARG[ARG.commodity_transaction=="Conventional crude oil - imports"].sort_values("year")

INDO_CrudeSupply=INDO[INDO.commodity_transaction=="Conventional crude oil - total energy supply"].sort_values("year")
INDO_CrudeProd=INDO[INDO.commodity_transaction=="Conventional crude oil - production"].sort_values("year")
INDO_Refinery=INDO[INDO.commodity_transaction=="Crude petroleum - refinery capacity"].sort_values("year")
INDO_CrudeExports=INDO[INDO.commodity_transaction=="Conventional crude oil - exports"].sort_values("year")
INDO_CrudeImports=INDO[INDO.commodity_transaction=="Conventional crude oil - imports"].sort_values("year")

MEX_CrudeSupply=MEX[MEX.commodity_transaction=="Conventional crude oil - total energy supply"].sort_values("year")
MEX_CrudeProd=MEX[MEX.commodity_transaction=="Conventional crude oil - production"].sort_values("year")
MEX_Refinery=MEX[MEX.commodity_transaction=="Crude petroleum - refinery capacity"].sort_values("year")
MEX_CrudeExports=MEX[MEX.commodity_transaction=="Conventional crude oil - exports"].sort_values("year")
MEX_CrudeImports=MEX[MEX.commodity_transaction=="Conventional crude oil - imports"].sort_values("year")

#BRICS STATES: 
BR_CrudeSupply=BR[BR.commodity_transaction=="Conventional crude oil - total energy supply"].sort_values("year")
BR_CrudeProd=BR[BR.commodity_transaction=="Conventional crude oil - production"].sort_values("year")
BR_Refinery=BR[BR.commodity_transaction=="Crude petroleum - refinery capacity"].sort_values("year")
BR_CrudeExports=BR[BR.commodity_transaction=="Conventional crude oil - exports"].sort_values("year")
BR_CrudeImports=BR[BR.commodity_transaction=="Conventional crude oil - imports"].sort_values("year")

RUS_CrudeSupply=RUS[RUS.commodity_transaction=="Conventional crude oil - total energy supply"].sort_values("year")
RUS_CrudeProd=RUS[RUS.commodity_transaction=="Conventional crude oil - production"].sort_values("year")
RUS_Refinery=RUS[RUS.commodity_transaction=="Crude petroleum - refinery capacity"].sort_values("year")
RUS_CrudeExports=RUS[RUS.commodity_transaction=="Conventional crude oil - exports"].sort_values("year")
RUS_CrudeImports=RUS[RUS.commodity_transaction=="Conventional crude oil - imports"].sort_values("year")

IND_CrudeSupply=IND[IND.commodity_transaction=="Conventional crude oil - total energy supply"].sort_values("year")
IND_CrudeProd=IND[IND.commodity_transaction=="Conventional crude oil - production"].sort_values("year")
IND_Refinery=IND[IND.commodity_transaction=="Crude petroleum - refinery capacity"].sort_values("year")
IND_CrudeExports=IND[IND.commodity_transaction=="Conventional crude oil - exports"].sort_values("year")
IND_CrudeImports=IND[IND.commodity_transaction=="Conventional crude oil - imports"].sort_values("year")

CHI_CrudeSupply=CHI[CHI.commodity_transaction=="Conventional crude oil - total energy supply"].sort_values("year")
CHI_CrudeProd=CHI[CHI.commodity_transaction=="Conventional crude oil - production"].sort_values("year")
CHI_Refinery=CHI[CHI.commodity_transaction=="Crude petroleum - refinery capacity"].sort_values("year")
CHI_CrudeExports=CHI[CHI.commodity_transaction=="Conventional crude oil - exports"].sort_values("year")
CHI_CrudeImports=CHI[CHI.commodity_transaction=="Conventional crude oil - imports"].sort_values("year")

SAfr_CrudeSupply=SAfr[SAfr.commodity_transaction=="Conventional crude oil - total energy supply"].sort_values("year")
SAfr_CrudeProd=SAfr[SAfr.commodity_transaction=="Conventional crude oil - production"].sort_values("year")
SAfr_Refinery=SAfr[SAfr.commodity_transaction=="Crude petroleum - refinery capacity"].sort_values("year")
SAfr_CrudeExports=SAfr[SAfr.commodity_transaction=="Conventional crude oil - exports"].sort_values("year")
SAfr_CrudeImports=SAfr[SAfr.commodity_transaction=="Conventional crude oil - imports"].sort_values("year")

#To merge the dataset of some EU members to represent the EU oil data...
EU_CrudeSupply=pd.merge(SP_CrudeSupply,ITA_CrudeSupply, on="year", how="outer")
EU_CrudeSupply.rename(columns={"country_or_area_x":"Spain","commodity_transaction_x":"commodity1", "unit_x":"unit1", 
                               "quantity_x":"quantity1", "country_or_area_y":"Italy", "quantity_y":"quantity2"},inplace=True)
EU_CrudeSupply.drop(['commodity_transaction_y','unit_y','category_y'], axis=1,inplace=True)
#Merge with Germany: 
EU_CrudeSupply=EU_CrudeSupply.merge(GER_CrudeSupply, on="year",how="outer")
EU_CrudeSupply.rename(columns={"country_or_area":"Germany", "quantity":"quantity3"},inplace=True)
EU_CrudeSupply.drop(['commodity_transaction','unit','category'],axis=1,inplace=True)
#Merge with Netherlands: 
EU_CrudeSupply=EU_CrudeSupply.merge(NETH_CrudeSupply, on="year",how="outer")
EU_CrudeSupply.rename(columns={"country_or_area":"Netherlands", "quantity":"quantity4"},inplace=True)
EU_CrudeSupply.drop(['commodity_transaction','unit','category'],axis=1,inplace=True)
#Merge with France: 
EU_CrudeSupply=EU_CrudeSupply.merge(FRA_CrudeSupply, on="year",how="outer")
EU_CrudeSupply.rename(columns={"country_or_area":"France", "quantity":"quantity5"},inplace=True)
EU_CrudeSupply.drop(['commodity_transaction','unit','category'],axis=1,inplace=True)
#Merge with Portugal: 
EU_CrudeSupply=EU_CrudeSupply.merge(POR_CrudeSupply, on="year",how="outer")
EU_CrudeSupply.rename(columns={"country_or_area":"Portugal", "quantity":"quantity6"},inplace=True)
EU_CrudeSupply.drop(['commodity_transaction','unit','category'],axis=1,inplace=True)
EU_CrudeSupply.head(2)
#Fill all NaN values: 
values = {'France':'France','quantity3':0,'Germany':'Germany', 'quantity4':0,'Netherlands':'Netherlands','quantity5':0}

EU_CrudeSupply.fillna(value=values,inplace=True)
EU_CrudeSupply.head(2)
EU_CrudeSupply["totalquality"]=EU_CrudeSupply["quantity1"]+EU_CrudeSupply["quantity2"]+EU_CrudeSupply["quantity3"]+ EU_CrudeSupply["quantity4"]+EU_CrudeSupply["quantity5"]+EU_CrudeSupply["quantity6"]
EU_CrudeSupply.head(2)
EU_CrudeImports=pd.merge(SP_CrudeImports,ITA_CrudeImports, on="year", how="outer")
EU_CrudeImports.rename(columns={"country_or_area_x":"Spain","commodity_transaction_x":"commodity1", "unit_x":"unit1", 
                               "quantity_x":"quantity1", "country_or_area_y":"Italy", "quantity_y":"quantity2"},inplace=True)
EU_CrudeImports.drop(['commodity_transaction_y','unit_y','category_y'], axis=1,inplace=True)
#Merge with Germany: 
EU_CrudeImports=EU_CrudeImports.merge(GER_CrudeImports, on="year",how="outer")
EU_CrudeImports.rename(columns={"country_or_area":"Germany", "quantity":"quantity3"},inplace=True)
EU_CrudeImports.drop(['commodity_transaction','unit','category'],axis=1,inplace=True)
#Merge with Netherlands: 
EU_CrudeImports=EU_CrudeImports.merge(NETH_CrudeImports, on="year",how="outer")
EU_CrudeImports.rename(columns={"country_or_area":"Netherlands", "quantity":"quantity4"},inplace=True)
EU_CrudeImports.drop(['commodity_transaction','unit','category'],axis=1,inplace=True)
#Merge with France: 
EU_CrudeImports=EU_CrudeImports.merge(FRA_CrudeImports, on="year",how="outer")
EU_CrudeImports.rename(columns={"country_or_area":"France", "quantity":"quantity5"},inplace=True)
EU_CrudeImports.drop(['commodity_transaction','unit','category'],axis=1,inplace=True)
#Merge with Portugal: 
EU_CrudeImports=EU_CrudeImports.merge(POR_CrudeImports, on="year",how="outer")
EU_CrudeImports.rename(columns={"country_or_area":"Portugal", "quantity":"quantity6"},inplace=True)
EU_CrudeImports.drop(['commodity_transaction','unit','category'],axis=1,inplace=True)
#Fill all NaN values: 
values = {'France':'France','quantity3':0,'Germany':'Germany', 'quantity4':0,'Netherlands':'Netherlands','quantity5':0}

EU_CrudeImports.fillna(value=values,inplace=True)
EU_CrudeImports.head(2)
EU_CrudeImports["totalquality"]=EU_CrudeImports["quantity1"]+EU_CrudeImports["quantity2"]+EU_CrudeImports["quantity3"]+ EU_CrudeImports["quantity4"]+EU_CrudeImports["quantity5"]+EU_CrudeImports["quantity6"]
EU_CrudeImports.head(2)
                                           #Gas Oil & Diesel Oil Final Consumption
US_gas_oil=US[US.commodity_transaction=="Gas Oil/ Diesel Oil - Final consumption"].sort_values("year")
CAN_gas_oil=CAN[CAN.commodity_transaction=="Gas Oil/ Diesel Oil - Final consumption"].sort_values("year")
JAP_gas_oil=JAP[JAP.commodity_transaction=="Gas Oil/ Diesel Oil - Final consumption"].sort_values("year")
UK_gas_oil=UK[UK.commodity_transaction=="Gas Oil/ Diesel Oil - Final consumption"].sort_values("year")
GER_gas_oil=GER[GER.commodity_transaction=="Gas Oil/ Diesel Oil - Final consumption"].sort_values("year")
ITA_gas_oil=ITA[ITA.commodity_transaction=="Gas Oil/ Diesel Oil - Final consumption"].sort_values("year")
FRA_gas_oil=FRA[FRA.commodity_transaction=="Gas Oil/ Diesel Oil - Final consumption"].sort_values("year")
#List of various EU countries 
SP_gas_oil=SP[SP.commodity_transaction=="Gas Oil/ Diesel Oil - Final consumption"].sort_values("year")
NETH_gas_oil=NETH[NETH.commodity_transaction=="Gas Oil/ Diesel Oil - Final consumption"].sort_values("year")
NOR_gas_oil=NOR[NOR.commodity_transaction=="Gas Oil/ Diesel Oil - Final consumption"].sort_values("year")
POR_gas_oil=POR[POR.commodity_transaction=="Gas Oil/ Diesel Oil - Final consumption"].sort_values("year")
#DEVELOPING STATES: 
TUR_gas_oil=TUR[TUR.commodity_transaction=="Gas Oil/ Diesel Oil - Final consumption"].sort_values("year")
SKOR_gas_oil=SKOR[SKOR.commodity_transaction=="Gas Oil/ Diesel Oil - Final consumption"].sort_values("year")
SAUD_gas_oil=SAUD[SAUD.commodity_transaction=="Gas Oil/ Diesel Oil - Final consumption"].sort_values("year")
AUST_gas_oil=AUST[AUST.commodity_transaction=="Gas Oil/ Diesel Oil - Final consumption"].sort_values("year")
ARG_gas_oil=ARG[ARG.commodity_transaction=="Gas Oil/ Diesel Oil - Final consumption"].sort_values("year")
INDO_gas_oil=INDO[INDO.commodity_transaction=="Gas Oil/ Diesel Oil - Final consumption"].sort_values("year")
MEX_gas_oil=MEX[MEX.commodity_transaction=="Gas Oil/ Diesel Oil - Final consumption"].sort_values("year")
#BRICS STATES: 
BR_gas_oil=BR[BR.commodity_transaction=="Gas Oil/ Diesel Oil - Final consumption"].sort_values("year")
RUS_gas_oil=RUS[RUS.commodity_transaction=="Gas Oil/ Diesel Oil - Final consumption"].sort_values("year")
IND_gas_oil=IND[IND.commodity_transaction=="Gas Oil/ Diesel Oil - Final consumption"].sort_values("year")
CHI_gas_oil=CHI[CHI.commodity_transaction=="Gas Oil/ Diesel Oil - Final consumption"].sort_values("year")
SAfr_gas_oil=SAfr[SAfr.commodity_transaction=="Gas Oil/ Diesel Oil - Final consumption"].sort_values("year")
EU_gas_oil=pd.merge(SP_gas_oil, ITA_gas_oil, on="year", how="outer")
EU_gas_oil.rename(columns={"country_or_area_x":"Spain","commodity_transaction_x":"commodity1", "unit_x":"unit1", 
                               "quantity_x":"quantity1", "country_or_area_y":"Italy", "quantity_y":"quantity2"},inplace=True)
EU_gas_oil.drop(['commodity_transaction_y','unit_y','category_y'], axis=1,inplace=True)
#Merge with Germany: 
EU_gas_oil=EU_gas_oil.merge(GER_gas_oil, on="year",how="outer")
EU_gas_oil.rename(columns={"country_or_area":"Germany", "quantity":"quantity3"},inplace=True)
EU_gas_oil.drop(['commodity_transaction','unit','category'],axis=1,inplace=True)
#Merge with Netherlands: 
EU_gas_oil=EU_gas_oil.merge(NETH_gas_oil, on="year",how="outer")
EU_gas_oil.rename(columns={"country_or_area":"Netherlands", "quantity":"quantity4"},inplace=True)
EU_gas_oil.drop(['commodity_transaction','unit','category'],axis=1,inplace=True)
#Merge with France: 
EU_gas_oil=EU_gas_oil.merge(FRA_gas_oil, on="year",how="outer")
EU_gas_oil.rename(columns={"country_or_area":"France", "quantity":"quantity5"},inplace=True)
EU_gas_oil.drop(['commodity_transaction','unit','category'],axis=1,inplace=True)
#Merge with Portugal: 
EU_gas_oil=EU_gas_oil.merge(POR_gas_oil, on="year",how="outer")
EU_gas_oil.rename(columns={"country_or_area":"Portugal", "quantity":"quantity6"},inplace=True)
EU_gas_oil.drop(['commodity_transaction','unit','category'],axis=1,inplace=True)
#Fill all NaN values: 
values = {'France':'France','quantity3':0,'Germany':'Germany', 'quantity4':0,'Netherlands':'Netherlands','quantity5':0}

EU_gas_oil.fillna(value=values,inplace=True)
EU_gas_oil.head(2)
EU_gas_oil["totalquality"]=EU_gas_oil["quantity1"]+EU_gas_oil["quantity2"]+EU_gas_oil["quantity3"]+ EU_gas_oil["quantity4"]+EU_gas_oil["quantity5"]+EU_gas_oil["quantity6"]
EU_gas_oil.head(2)
#This is for the Conventional Crude Production: 
y1 = US_CrudeProd.quantity
x1 = US_CrudeProd.year
y2 = CAN_CrudeProd.quantity
x2 = CAN_CrudeProd.year
y3 = CHI_CrudeProd.quantity
x3 = CHI_CrudeProd.year
x4 = UK_CrudeProd.year
y4 = UK_CrudeProd.quantity
x5 = BR_CrudeProd.year
y5 = BR_CrudeProd.quantity
x6 = NOR_CrudeProd.year
y6 = NOR_CrudeProd.quantity
x7 = MEX_CrudeProd.year
y7 = MEX_CrudeProd.quantity
x8 = SAUD_CrudeProd.year
y8 = SAUD_CrudeProd.quantity
x9 = RUS_CrudeProd.year
y9 = RUS_CrudeProd.quantity
import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.plot(x1,y1,label="US")
plt.plot(x2,y2,'r',label="Canada")
plt.plot(x3,y3,'y',label="China")
plt.plot(x4,y4,'k',label="UK")
plt.plot(x5,y5,'g',label="Brasil")
plt.plot(x6,y6,'c',label="Norway")
plt.plot(x7,y7,'m',label="Mexico")
plt.plot(x8,y8,'orange',label="Saudi")
plt.plot(x9,y9,'purple',label="Russia")

plt.legend(fontsize=16)
plt.ylabel("Metric tons, thousand",fontsize=20)
plt.xlabel('Year',fontsize=20)
plt.title('Conventional crude oil - production ',fontsize=24)
plt.xlim(1989.8, 2014.2)
plt.show()

y1 = US_CrudeImports.quantity
x1 = US_CrudeImports.year
y2 = CHI_CrudeImports.quantity
x2 = CHI_CrudeImports.year
x3 = UK_CrudeImports.year
y3 = UK_CrudeImports.quantity
x4 = EU_CrudeImports.year
#y4 = EU_CrudeImports.quantity
x5 = TUR_CrudeImports.year
y5 = TUR_CrudeImports.quantity
x6 =SKOR_CrudeImports.year
y6 =SKOR_CrudeImports.quantity
x7 =JAP_CrudeImports.year
y7=JAP_CrudeImports.quantity
x8= IND_CrudeImports.year
y8= IND_CrudeImports.quantity
import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.plot(x1,y1,label="US")
plt.plot(x2,y2,'r',label="China")
plt.plot(x3,y3,'y',label="UK")
plt.plot(x4,y4,'k',label="EU")
plt.plot(x5,y5,'g',label="Turkey")
plt.plot(x6,y6,'c',label="South Korea")
plt.plot(x7,y7,'m',label="Japon")
plt.plot(x8,y8,'orange',label="India")

plt.legend(fontsize=16)
plt.ylabel("Metric tons, thousand",fontsize=15)
plt.xlabel('Year',fontsize=20)
plt.title('Conventional crude oil - imports ',fontsize=24)
plt.xlim(1989.8, 2014.2)
plt.show()
y1 = US_Refinery.quantity
x1 = US_Refinery.year
y2 = CHI_Refinery.quantity
x2 = CHI_Refinery.year
x3 = UK_Refinery.year
y3 = UK_Refinery.quantity
x4 = TUR_Refinery.year
y4 = TUR_Refinery.quantity
x5 =SKOR_Refinery.year
y5 =SKOR_Refinery.quantity
x6 =JAP_Refinery.year
y6=JAP_Refinery.quantity
x7= IND_Refinery.year
y7= IND_Refinery.quantity
x8= GER_Refinery.year
y8=GER_Refinery.quantity
import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.plot(x1,y1,label="US")
plt.plot(x2,y2,'r',label="China")
plt.plot(x3,y3,'y',label="UK")
plt.plot(x4,y4,'k',label="Turkey")
plt.plot(x5,y5,'g',label="South Korea")
plt.plot(x6,y6,'c',label="Japon")
plt.plot(x7,y7,'m',label="India")
plt.plot(x8,y8,'orange',label="Germany")

plt.legend(fontsize=16)
plt.ylabel("Metric tons, thousand",fontsize=15)
plt.xlabel('Year',fontsize=20)
plt.title('Crude petroleum - refinery capacity',fontsize=24)
plt.xlim(1989.8, 2014.2)
plt.show()
width=0.25

fig, ax=plt.subplots(figsize=(15,10))
kare1=ax.bar(IND_CrudeImports["year"], IND_CrudeImports["quantity"], width, color="green")
kare2=ax.bar(CHI_CrudeImports["year"]+ width -.5, CHI_CrudeImports["quantity"], width, color="red")
kare3=ax.bar(INDO_CrudeImports["year"] + width, INDO_CrudeImports["quantity"], width, color="purple")
#kare4=ax.bar(SKOR_CrudeImports["year"], SKOR_CrudeImports["quantity"], width, color="black")
#kare5=ax.bar(JAP_CrudeImports["year"], JAP_CrudeImports["quantity"], width, color="blue")

ax.set_ylabel("Metric tons, thousands", fontsize=(15))
ax.set_title("Annual Crude Imports of East Asian States")
ax.set_xticks(US_CrudeImports["year"])
ax.set_xticklabels(US_CrudeImports["year"], rotation=45)

ax.legend((kare1[0],kare2[0],kare3[0]),("India Crude Imports","China Crude Imports","Indonesia Crude Imports"),fontsize=14)

#ax.legend((kare1[0],kare2[0],kare3[0],kare4[0],kare5[0]), 
#          ("India Crude Imports","China Crude Imports","Indonesia Crude Imports", "South Korea Crude Imports", "Japon Crude Imports"))
plt.show()
y1 = US_Refinery.quantity
x1 = US_Refinery.year
y2 = CHI_Refinery.quantity
x2 = CHI_Refinery.year
x3 = UK_Refinery.year
y3 = UK_Refinery.quantity
x4 = TUR_Refinery.year
y4 = TUR_Refinery.quantity
x5 =SKOR_Refinery.year
y5 =SKOR_Refinery.quantity
x6 =JAP_Refinery.year
y6=JAP_Refinery.quantity
x7= IND_Refinery.year
y7= IND_Refinery.quantity
x8= GER_Refinery.year
y8=GER_Refinery.quantity
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
sns.regplot(x1,y1,order=3)
sns.regplot(x2,y2, order=3)
plt.legend(labels=["US","China"])
plt.title("Regression plot for the Refinery Capacity between US, China")
plt.show()
plt.figure(figsize=(10,5))
sns.regplot(x6,y6,order=3)
sns.regplot(x8,y8, order=3)
plt.legend(labels=["Japon","Germany"])
plt.title("Regression plot for the Refinery Capacity between Japon, Germany")
plt.show()
US_naturalgas= US[US.category=="natural_gas_including_lng"].sort_values("year") 
CHI_naturalgas= CHI[CHI.category=="natural_gas_including_lng"].sort_values("year") 
JAP_naturalgas= JAP[JAP.category=="natural_gas_including_lng"].sort_values("year")
GER_naturalgas= GER[GER.category=="natural_gas_including_lng"].sort_values("year")
IND_naturalgas=IND[IND.category=="natural_gas_including_lng"].sort_values("year")
RUS_naturalgas= RUS[RUS.category=="natural_gas_including_lng"].sort_values("year")
CAN_naturalgas=CAN[CAN.category=="natural_gas_including_lng"].sort_values("year")
UK_naturalgas= UK[UK.category=="natural_gas_including_lng"].sort_values("year")
FRA_naturalgas=UK[UK.category=="natural_gas_including_lng"].sort_values("year")
ITA_naturalgas= ITA[ITA.category=="natural_gas_including_lng"].sort_values("year")
SKOR_naturalgas=SKOR[SKOR.category=="natural_gas_including_lng"].sort_values("year")
TUR_naturalgas=TUR[TUR.category=="natural_gas_including_lng"].sort_values("year")


US_gasexports= US_naturalgas[US_naturalgas.commodity_transaction=="Natural gas (including LNG) - exports"].sort_values("year")

CHI_gasexports= CHI_naturalgas[CHI_naturalgas.commodity_transaction=="Natural gas (including LNG) - exports"].sort_values("year")

#Gas Consumption by Countries: 
US_gasconsumption=US_naturalgas[US_naturalgas.commodity_transaction=="Natural gas (including LNG) - final consumption"].sort_values("year") 
CHI_gasconsumption=CHI_naturalgas[CHI_naturalgas.commodity_transaction=="Natural gas (including LNG) - final consumption"].sort_values("year") 
JAP_gasconsumption=JAP_naturalgas[JAP_naturalgas.commodity_transaction=="Natural gas (including LNG) - final consumption"].sort_values("year") 
GER_gasconsumption=GER_naturalgas[GER_naturalgas.commodity_transaction=="Natural gas (including LNG) - final consumption"].sort_values("year") 
IND_gasconsumption=IND_naturalgas[IND_naturalgas.commodity_transaction=="Natural gas (including LNG) - final consumption"].sort_values("year") 
import matplotlib.pyplot as plt
plt.figure(figsize=(20,8))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.plot(US_gasconsumption.year,US_gasconsumption.quantity,label="US")
plt.plot(CHI_gasconsumption.year,CHI_gasconsumption.quantity,label="China")
plt.plot(JAP_gasconsumption.year,JAP_gasconsumption.quantity,label="Japon")
plt.plot(GER_gasconsumption.year,GER_gasconsumption.quantity,label="Germany")
plt.plot(IND_gasconsumption.year,IND_gasconsumption.quantity,label="India")

plt.legend(fontsize=16)
plt.ylabel("Metric tons, thousand",fontsize=15)
plt.xlabel('Year',fontsize=20)
plt.title('Natural gas (including LNG) - final consumption',fontsize=24)
plt.xlim(1989.8, 2014.2)
plt.show()
#Gas Production by Countries: 
US_gasproduction=US_naturalgas[US_naturalgas.commodity_transaction=="Natural gas (including LNG) - production"].sort_values("year")
RUS_gasproduction=RUS_naturalgas[RUS_naturalgas.commodity_transaction=="Natural gas (including LNG) - production"].sort_values("year")
plt.figure(figsize=(20,5))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.plot(US_gasproduction.year,US_gasproduction.quantity,label="US")
plt.plot(RUS_gasproduction.year,RUS_gasproduction.quantity,label="Russia")

plt.legend(fontsize=16)
plt.ylabel("Metric tons, thousand",fontsize=15)
plt.xlabel('Year',fontsize=20)
plt.title('Natural gas (including LNG) - production',fontsize=24)
plt.xlim(1989.8, 2014.2)
plt.show()
#Natural Gas Imports by Countries
US_gasimports= US_naturalgas[US_naturalgas.commodity_transaction=="Natural gas (including LNG) - imports"].sort_values("year")
CHI_gasimports= CHI_naturalgas[CHI_naturalgas.commodity_transaction=="Natural gas (including LNG) - imports"].sort_values("year")
UK_gasimports= UK_naturalgas[UK_naturalgas.commodity_transaction=="Natural gas (including LNG) - imports"].sort_values("year")
FRA_gasimports= FRA_naturalgas[FRA_naturalgas.commodity_transaction=="Natural gas (including LNG) - imports"].sort_values("year")
GER_gasimports= GER_naturalgas[GER_naturalgas.commodity_transaction=="Natural gas (including LNG) - imports"].sort_values("year")
JAP_gasimports= JAP_naturalgas[JAP_naturalgas.commodity_transaction=="Natural gas (including LNG) - imports"].sort_values("year")
SKOR_gasimports= SKOR_naturalgas[SKOR_naturalgas.commodity_transaction=="Natural gas (including LNG) - imports"].sort_values("year")
TUR_gasimports= TUR_naturalgas[TUR_naturalgas.commodity_transaction=="Natural gas (including LNG) - imports"].sort_values("year")
IND_gasimports= IND_naturalgas[IND_naturalgas.commodity_transaction=="Natural gas (including LNG) - imports"].sort_values("year")
plt.figure(figsize=(20,5))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.plot(US_gasimports.year,US_gasimports.quantity,label="US")
plt.plot(CHI_gasimports.year,CHI_gasimports.quantity,label="China")
plt.plot(JAP_gasimports.year,JAP_gasimports.quantity,label="Japon")
plt.plot(GER_gasimports.year,GER_gasimports.quantity,label="Germany")


plt.legend(fontsize=16)
plt.ylabel("Metric tons, thousand",fontsize=15)
plt.xlabel('Year',fontsize=20)
plt.title('Natural gas (including LNG) - imports of Big Four Economy',fontsize=24)
plt.xlim(1989.8, 2014.2)
plt.show()
plt.figure(figsize=(20,5))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.plot(TUR_gasimports.year,TUR_gasimports.quantity,label="Turkey")
plt.plot(SKOR_gasimports.year,SKOR_gasimports.quantity,label="South Korea")
plt.plot(UK_gasimports.year,UK_gasimports.quantity,label="UK")

plt.legend(fontsize=16)
plt.ylabel("Metric tons, thousand",fontsize=15)
plt.xlabel('Year',fontsize=20)
plt.title('Natural gas (including LNG) - imports of Turkey, South Korea, UK',fontsize=24)
plt.xlim(1989.8, 2014.2)
plt.show()