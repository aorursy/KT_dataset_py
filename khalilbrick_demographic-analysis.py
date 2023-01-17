import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
!pip install xlrd
Population_total = pd.read_csv("../input/demographic-data/API_SP.POP.TOTL_DS2_en_csv_v2_1217749.csv",header = 2)
Population_total = Population_total.drop(["Unnamed: 64","Indicator Code","Country Code","Indicator Name"],axis = 1)
Population_total = Population_total.set_index("Country Name")
Population_total.columns = pd.to_numeric(Population_total.columns)
Population_total
plt.figure(figsize = (15,8))


plt.bar(Population_total.loc["World"].index[2::],Population_total.loc["World"].values[2::])
plt.plot(Population_total.loc["World"].index[2::],Population_total.loc["World"].values[2::],color = "black",marker = "o")

plt.xticks([i for i in range(1960,2021,5)])
plt.yticks(size = 15)
plt.xticks(size = 15,rotation = 15)
plt.ylabel("Population in Billions",size = 15)
plt.title("World Population from 1960 to 2020",fontdict = {"fontsize":25,"fontweight":50})
plt.savefig("World Population from 1960 to 2020",dpi = 300)
plt.show()
plt.figure(figsize = (15,8))

plt.plot(Population_total.loc["China"].index[2::],Population_total.loc["China"].values[2::],marker = "o",label = "China")

plt.plot(Population_total.loc["United States"].index[2::],Population_total.loc["United States"].values[2::],marker = "o",label = "United States")

plt.plot(Population_total.loc["United Kingdom"].index[2::],Population_total.loc["United Kingdom"].values[2::],marker = "o",label = "United Kingdom")

plt.plot(Population_total.loc["Russian Federation"].index[2::],Population_total.loc["Russian Federation"].values[2::],marker = "o",label = "Russia")

plt.plot(Population_total.loc["India"].index[2::],Population_total.loc["India"].values[2::],marker = "o",label = "India")


plt.xticks([i for i in range(1960,2021,5)])
plt.yticks(size = 15)
plt.xticks(size = 15,rotation = 15)
plt.ylabel("Population in Billions",size = 15)
plt.legend(fontsize = 13)
plt.title("Population of Some Countries from 1960 to 2020",fontdict = {"fontsize":25,"fontweight":50})
plt.savefig("Population of Some Countries from 1960 to 2020",dpi = 300)
plt.show()
Life_Expectancy = pd.read_csv("../input/demographic-data/Life Expectancy Data.csv")
Life_Expectancy = Life_Expectancy.set_index("Country")
Life_Expectancy = Life_Expectancy.rename(columns = {"Life expectancy ":"Life expectancy"})

Life_Expectancy_top_10_countries = (Life_Expectancy.loc[Life_Expectancy.Year == 2015]).sort_values(by = "Life expectancy")[-1:-11:-1]
Life_Expectancy_top_10_countries
plt.figure(figsize = (15,8))
color = 'tab:blue'
plt.bar(Life_Expectancy_top_10_countries["Life expectancy"].index, Life_Expectancy_top_10_countries["Life expectancy"].values, color=color)
plt.plot(Life_Expectancy_top_10_countries["Life expectancy"].index, Life_Expectancy_top_10_countries["Life expectancy"].values, color = "black",marker= "o")
plt.xticks(size = 15,rotation = 15)
plt.yticks(size = 15)
plt.title("Top 10 Highest Life Expectancy in The World in 2015",fontdict = {"fontsize":25})
plt.savefig("Top 10 Highest Life Expectancy in The World in 2015",dpi = 300)
plt.show()
plt.figure(figsize = (15,8))
#United States of America
plt.plot(Life_Expectancy.loc["United States of America"]["Year"],Life_Expectancy.loc["United States of America"]["Life expectancy"],marker = "o",label = "United States of America")
#Spain
plt.plot(Life_Expectancy.loc["Spain"]["Year"],Life_Expectancy.loc["Spain"]["Life expectancy"],marker = "o",label = "Spain")
#China
plt.plot(Life_Expectancy.loc["China"]["Year"],Life_Expectancy.loc["China"]["Life expectancy"],marker = "o",label = "China")
#Japan
plt.plot(Life_Expectancy.loc["Japan"]["Year"],Life_Expectancy.loc["Japan"]["Life expectancy"],marker = "o",label = "Japan")
#Italy
plt.plot(Life_Expectancy.loc["Italy"]["Year"],Life_Expectancy.loc["Italy"]["Life expectancy"],marker = "o",label = "Italy")
#Tunisia
plt.plot(Life_Expectancy.loc["Tunisia"]["Year"],Life_Expectancy.loc["Tunisia"]["Life expectancy"],marker = "o",label = "Tunisia")
#Australia
plt.plot(Life_Expectancy.loc["Australia"]["Year"],Life_Expectancy.loc["Australia"]["Life expectancy"],marker = "*",label = "Australia")

plt.xticks([i for i in range(2000,2017,3)],size = 15)
plt.yticks(size = 15)
plt.xlabel("Year",size = 15)
plt.ylabel("Life expectancy in Years",size = 15)
plt.legend(ncol=1,loc = 1)
plt.title("Life expectancy of Some Countries from 2000 To 2015",fontdict = {"fontsize":25})
plt.savefig("Life expectancy of Some Countries from 2000 To 2015",dpi = 300)
plt.show()
countries = ["USA","China","Japan","Germany","India","UK","France","Italy","Brazil","Canada"]
GDP = pd.read_csv("../input/demographic-data/API_NY.GDP.MKTP.CD_DS2_en_csv_v2_559588.csv")
GDP = GDP.set_index("Country Name")
GDP = GDP.rename(index = {"United States":"USA","United Kingdom":"UK"})
GDP = GDP.drop(["Indicator Name","Indicator Code","Unnamed: 64","Country Code"],axis = 1)
GDP.columns =pd.to_numeric(GDP.columns)

GDP_top_10 = GDP.loc[countries]

GDP_top_10

a = np.linspace(1.712510e+12,2.049410e+13,10)
GDP_top_10[2018].plot(kind = "line",figsize = (15,8),marker = "o",color = "black")
GDP_top_10[2018].plot(kind = "bar",figsize = (15,8))
plt.xticks(rotation = 20,size = 15)
plt.yticks(a,size = 15)
plt.xlabel(None)
plt.ylabel("GDP (1 = 10 billions USD )",size = 15)
plt.title("Top 10 GDP in The World in 2018",fontdict = {"fontsize":25})

plt.savefig("Top 10 GDP in The World in 2018",dpi = 300)
plt.show()
plt.figure(figsize = (15,8))
#France
plt.plot(GDP.loc["France"].index,GDP.loc["France"].values,label = "France")

#USA
plt.plot(GDP.loc["USA"].index,GDP.loc["USA"].values,label = "USA",marker = "o")

#Canada
plt.plot(GDP.loc["Canada"].index,GDP.loc["Canada"].values,label = "Canada")

#Germany
plt.plot(GDP.loc["Germany"].index,GDP.loc["Germany"].values,label = "Germany")

#China
plt.plot(GDP.loc["China"].index,GDP.loc["China"].values,label = "China")

#Japan
plt.plot(GDP.loc["Japan"].index,GDP.loc["Japan"].values,label = "Japan")

#Canada
plt.plot(GDP.loc["Canada"].index,GDP.loc["Canada"].values,label = "Canada")

#India
plt.plot(GDP.loc["India"].index,GDP.loc["India"].values,label = "India",color = "black")


plt.yticks(size = 15)
plt.ylabel("GDP (1 = 10 billions USD )",size = 15)
plt.xticks([i for i in range(1960,2021,4)],size = 15,rotation = 20)
plt.legend(ncol = 2,fontsize = 15)
plt.title("GDP of Some Countries from 1960 to 2020",fontdict = {"fontsize":25})
plt.savefig("GDP of Some Countries from 1960 to 2020",dpi = 300)
plt.show()
plt.figure(figsize = (15,8))
#World
plt.plot(GDP.loc["World"].index,GDP.loc["World"].values,label = "World",marker = "+")

#USA
plt.plot(GDP.loc["USA"].index,GDP.loc["USA"].values,label = "USA",marker = "o")


plt.yticks(size = 15)
plt.ylabel("GDP (1 = 10 billions USD )",size = 15)
plt.xticks([i for i in range(1960,2021,4)],size = 15,rotation = 20)
plt.legend(ncol = 2,fontsize = 15)
plt.title("GDP of USA and the World from 1960 to 2020",fontdict = {"fontsize":25})
plt.savefig("GDP of USA and the World from 1960 to 2020",dpi = 300)
plt.show()
alcohol = pd.read_csv("../input/demographic-data/API_SH.ALC.PCAP.FE.LI_DS2_fr_csv_v2_1245707.csv",header = 2)
alcohol = alcohol.drop(["Unnamed: 64","Country Code","Indicator Name","Indicator Code"],axis = 1)
alcohol = alcohol.set_index("Country Name")
alcohol_top_10 = alcohol["2016"].dropna().sort_values()[-1:-11:-1]
alcohol_top_10 = alcohol_top_10.rename(index = {"République tchèque":"Czech","Allemagne":"Germany","Fédération de Russie":"Russia"})
alcohol_top_10
plt.figure(figsize = (15,8))
alcohol_top_10.plot(kind = "bar")
plt.plot(alcohol_top_10.index,alcohol_top_10.values,marker = "o",color = "black")
plt.xticks(size = 15,rotation = 20)
plt.yticks(size = 15)
plt.ylabel("Alcohol Consumption",size = 15)
plt.xlabel(None)

plt.title("Top 10 Countries with the Most Alcohol Use",fontdict = {"fontsize":25})
plt.savefig("Top 10 Countries with the Most Alcohol Use",dpi = 300)
plt.show()
Access_to_electricity = pd.read_excel("../input/demographic-data/API_EG.ELC.ACCS.ZS_DS2_fr_excel_v2_1226266.xls",sheet_name = "Data",header = 3 )
Access_to_electricity = Access_to_electricity.drop(["Country Code","Indicator Name","Indicator Code","2019"],axis = 1)
Access_to_electricity = Access_to_electricity.set_index('Country Name')
Access_to_electricity.columns = pd.to_numeric(Access_to_electricity.columns)
Access_to_electricity 
plt.figure(figsize = (15,8))
#Gabon
plt.plot(Access_to_electricity.loc["Gabon"].index,Access_to_electricity.loc["Gabon"].values,label = "Gabon",marker = "o")

#Maroc
plt.plot(Access_to_electricity.loc["Maroc"].index,Access_to_electricity.loc["Maroc"].values,label = "Maroc",marker = "o")

#Nigéria
plt.plot(Access_to_electricity.loc["Nigéria"].index,Access_to_electricity.loc["Nigéria"].values,label = "Nigéria",marker = "o")

#North America
plt.plot(Access_to_electricity.loc["Amérique du Nord"].index,Access_to_electricity.loc["Amérique du Nord"].values,label = "Amérique du Nord",marker = "o")

#Tunisia
plt.plot(Access_to_electricity.loc["Tunisie"].index,Access_to_electricity.loc["Tunisie"].values,label = "Tunisie",marker = "o")

#Zambie
plt.plot(Access_to_electricity.loc["Zambie"].index,Access_to_electricity.loc["Zambie"].values,label = "Zambie",marker = "o")

plt.xticks([i for i in range(1990,2019,4)],size = 15,rotation = 20)
plt.ylabel("Access to electricity (% of population)",size = 15)
plt.yticks(size = 15)
plt.title("Access to electricity (% of population) of Some Countries Of the World",fontdict = {"fontsize":20})
plt.legend(fontsize = 11,ncol = 2)
plt.savefig("Access to electricity % of population of Some Countries Of the World",dpi = 300)
plt.show()
Agriculture = pd.read_excel("../input/demographic-data/API_NV.AGR.TOTL.ZS_DS2_fr_excel_v2_1225080.xls",sheet_name = "Data",header = 3)
Agriculture = Agriculture.set_index("Country Name")
Agriculture = Agriculture.drop(["Country Code","Indicator Name","Indicator Code"],axis = 1)
Agriculture.columns = pd.to_numeric(Agriculture.columns)
Agriculture[2019] = Agriculture[2019].fillna(0)
Agriculture = Agriculture.drop(["IDA seulement","Pays pauvres très endettés (PPTE)","IDA mélange","IDA totale"],axis = 0)
Agriculture_top_28 = Agriculture.sort_values(by = 2019)[-1:-29:-1]
Agriculture_top_28 = Agriculture_top_28.rename(index = {"Congo, République démocratique du":"Congo","République centrafricaine":"Centrafricaine"})
Agriculture_top_28
Agriculture_top_28[2019].plot(kind = "bar",figsize = (15,8))
plt.plot(Agriculture_top_28[2019].index,Agriculture_top_28[2019].values,marker = "o",color = "green")
plt.xlabel(None)
plt.xticks(rotation = 60,size = 10)
plt.yticks(size = 12)
plt.ylabel("Agriculture value added (% of GDP)",size = 15)
plt.title("Agriculture value added (% of GDP) Top 28 Countries",fontdict = {"fontsize":25})
plt.savefig("Agriculture value added % of GDP Top 28 Countries",dpi = 300)

plt.show()
plt.figure(figsize = (15,8))
plt.bar(Agriculture.loc["Monde"].index,Agriculture.loc["Monde"].values)
plt.plot(Agriculture.loc["Monde"],marker = "o",color = "black")
plt.yticks(size = 15)
plt.ylabel("Agriculture value added (% of GDP)",size = 15)
plt.title("Agriculture value added (% of GDP) of The World",fontdict = {"fontsize":25})
plt.savefig("Agriculture value added % of GDP of The World",dpi = 300)
plt.show()
Primary_school_completion_rate = pd.read_excel("../input/demographic-data/API_SE.PRM.CMPT.ZS_DS2_fr_excel_v2_1219975.xls",sheet_name = "Data",header = 3)
Primary_school_completion_rate = Primary_school_completion_rate.set_index("Country Name")
Primary_school_completion_rate = Primary_school_completion_rate.drop(["Country Code","Indicator Name","Indicator Code"],axis = 1)
Primary_school_completion_rate.columns = pd.to_numeric(Primary_school_completion_rate.columns)
Primary_school_completion_rate
plt.figure(figsize = (15,8))
plt.plot(Primary_school_completion_rate.loc["Monde"],marker = ".")
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.ylabel("Primary school completion rate",size = 20)
plt.title("Primary school completion rate, total (% of relevant age group) in the World",fontdict = {"fontsize":22})
plt.savefig("Primary school completion rate, total % of relevant age group World",dpi = 300)
plt.show()
Government_expenditure_on_education = pd.read_excel("../input/demographic-data/API_SE.XPD.TOTL.GD.ZS_DS2_en_excel_v2_1217796.xls",header = 3,sheet_name = "Data")
Government_expenditure_on_education = Government_expenditure_on_education.set_index("Country Name")
Government_expenditure_on_education = Government_expenditure_on_education.drop(["Country Code","Indicator Name","Indicator Code"],axis = 1)
Government_expenditure_on_education.columns = pd.to_numeric(Government_expenditure_on_education.columns)
Government_expenditure_on_education
plt.figure(figsize = (15,8))
plt.plot(Government_expenditure_on_education.loc["World"],marker = "o")
plt.xticks([i for i in range(1998,2018,2)],size = 15,rotation = 20)
plt.yticks(size = 15)
plt.ylabel("Government expenditure on education (% of GDP)",size = 15)
plt.title("Government expenditure on education, total (% of GDP) in The World",size = 25)
plt.savefig("Government expenditure on education, total % of GDP World",dpi = 300)

plt.show()
plt.figure(figsize = (15,8))
#South Africa
plt.plot(Government_expenditure_on_education.loc["South Africa",2000:2018].index,Government_expenditure_on_education.loc["South Africa",2000:2018].values,label = "South Africa",marker = "o")

#Colombia
plt.plot(Government_expenditure_on_education.loc["Colombia",2000:2018].index,Government_expenditure_on_education.loc["Colombia",2000:2018].values,label = "Colombia",marker = "o")

#Europe & Central Asia (IDA & IBRD countries)
plt.plot(Government_expenditure_on_education.loc["Europe & Central Asia (IDA & IBRD countries)",2000:2018].index,Government_expenditure_on_education.loc["Europe & Central Asia (IDA & IBRD countries)",2000:2018].values,label = "Europe & Central Asia",marker = "o")

#Hong Kong SAR, China
plt.plot(Government_expenditure_on_education.loc["Hong Kong SAR, China",2000:2018].index,Government_expenditure_on_education.loc["Hong Kong SAR, China",2000:2018].values,label = "Hong Kong",marker = "o")

#Pakistan
plt.plot(Government_expenditure_on_education.loc["Pakistan",2003:2018].index,Government_expenditure_on_education.loc["Pakistan",2003:2018].values,label = "Pakistan",marker = "o")

#Ghana
plt.plot(Government_expenditure_on_education.loc["Ghana",2000:2018].index,Government_expenditure_on_education.loc["Ghana",2000:2018].values,label = "Ghana",marker = "o")

plt.xticks([i for i in range(2000,2020,2)],size = 15,rotation = 20)
plt.yticks(size = 15)
plt.ylabel("Government expenditure on education (% of GDP)",size = 15)
plt.title("Government expenditure on education, total (% of GDP) in Some Countries",size = 25)
plt.legend(ncol = 2)
plt.savefig("Government expenditure on education, total (% of GDP) in Some Countries",dpi = 300)
plt.show()
Current_account_balance = pd.read_excel("../input/demographic-data/API_BN.CAB.XOKA.CD_DS2_en_excel_v2_1224045.xls",sheet_name = "Data",header = 3)
Current_account_balance = Current_account_balance.set_index("Country Name")
Current_account_balance = Current_account_balance.drop(["Country Code","Indicator Name","Indicator Code"],axis = 1)
Current_account_balance.columns = pd.to_numeric(Current_account_balance.columns)
Current_account_balance
plt.figure(figsize = (15,8))
#China
plt.plot(Current_account_balance.loc["China"],marker = "o",label = "China")

#United States
plt.plot(Current_account_balance.loc["United States",1980:],marker = "o",label = "United States")

#Canada
plt.plot(Current_account_balance.loc["Canada",1980:],marker = "o",label = "Canada")

#Germany
plt.plot(Current_account_balance.loc["Germany",1980:],marker = "o",label = "Germany")

#Italy
plt.plot(Current_account_balance.loc["Italy",1980:],marker = "+",label = "Italy")

plt.xticks([i for i in range(1980,2022,4)],size = 15,rotation = 20)
plt.yticks(size = 15)
plt.ylabel("Current account balance (BoP, current US)",size = 15)
plt.title("Current account balance (BoP, current US) in Some Countries",size = 25)
plt.legend()
plt.savefig("Current account balance (BoP, current US) in Some Countries",dpi = 300)
plt.show()
plt.figure(figsize = (15,8))
Current_account_balance = Current_account_balance.rename(index = {"Russian Federation":"Russia"})
plt.barh(Current_account_balance[2015].dropna().sort_values()[-1:-11:-1].index,Current_account_balance[2015].dropna().sort_values()[-1:-11:-1].values)
plt.xticks(rotation = 45,size = 15)
plt.yticks(size = 15)
plt.xlabel("Current account balance (BoP, current US)",size = 20)
plt.title("Current account balance (BoP, current US) Top 10 Countries",size = 25)
plt.savefig("Current account balance (BoP, current US) Top 10 Countries",dpi = 300)
plt.show()