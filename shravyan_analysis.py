# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
cases =pd.read_csv("../input/hiv-aids-dataset/no_of_cases_adults_15_to_49_by_country_clean.csv")

deaths = pd.read_csv("../input/hiv-aids-dataset/no_of_deaths_by_country_clean.csv")

living = pd.read_csv("../input/hiv-aids-dataset/no_of_people_living_with_hiv_by_country_clean.csv")

coverage = pd.read_csv("../input/hiv-aids-dataset/art_coverage_by_country_clean.csv")

pediatric = pd.read_csv("../input/hiv-aids-dataset/art_pediatric_coverage_by_country_clean.csv")

prevention = pd.read_csv("../input/hiv-aids-dataset/prevention_of_mother_to_child_transmission_by_country_clean.csv")
cases.head()
cases.info()
deaths.head()
living.head()
coverage.head()
coverage.info()
pediatric.head()
pediatric.info()
prevention.head()
prevention.info()
sns.set(color_codes=True)
# fig= plt.figure(figsize=(10,10))

# sns.barplot(cases["WHO Region"],cases["Count_median"],hue=cases["Year"]).set_title("HIV cases by WHO Region and Year")

# fig.show()

casesbyregion = cases.groupby(["WHO Region","Year"]).mean()["Count_median"]

fig=plt.figure(figsize=(10,5))

sns.heatmap(casesbyregion.unstack(level=0),annot=True).set_title("HIV Cases by region and year");

fig.show()
deathbyregion =deaths.groupby(["WHO Region","Year"]).sum()["Count_median"]

fig=plt.figure(figsize=(10,5))

sns.heatmap(deathbyregion.unstack(level=0),annot=True).set_title("HIV deaths by region and year");

fig.show()
livingbyregion =living.groupby(["WHO Region","Year"]).sum()["Count_median"]

fig=plt.figure(figsize=(10,5))

sns.heatmap(livingbyregion.unstack(level=0),annot=True).set_title("People living with HIV by region and year");

fig.show()
coverage.columns
coverage = coverage.drop([

       'Estimated number of people living with HIV','Estimated ART coverage among people living with HIV (%)',

       'Estimated number of people living with HIV_min',

       'Estimated number of people living with HIV_median',

       'Estimated number of people living with HIV_max',

       'Estimated ART coverage among people living with HIV (%)_min',

       'Estimated ART coverage among people living with HIV (%)_max'],axis=1)
coverage["ART"]=coverage["Reported number of people receiving ART"]
coverage["ART"] = pd.to_numeric(coverage["ART"], errors="coerce")
coverage["ART"] = coverage.ART.astype(float)
coverage.info()
coveragebyregion =coverage.groupby("WHO Region").sum()["ART"]

#coveragebyregion.plot(kind='bar',subplots=True, figsize=(30,30))
coveragebyregion
fig=plt.figure(figsize=(10,5))

sns.barplot(coveragebyregion.index,coveragebyregion).set_title("People receiving ART by region and year");

fig.show()
coveragebyregion1 =coverage.groupby("WHO Region").mean()["Estimated ART coverage among people living with HIV (%)_median"]

fig=plt.figure(figsize=(10,5))

sns.barplot(coveragebyregion1.index,coveragebyregion1).set_title("Estimated ART Coverage(%) by region and year");

fig.show()
pediatric.columns
pediatric = pediatric.drop([

       'Estimated number of children needing ART based on WHO methods',

       'Estimated ART coverage among children (%)',

       'Estimated number of children needing ART based on WHO methods_min',

       'Estimated number of children needing ART based on WHO methods_max',

       'Estimated ART coverage among children (%)_min',

       'Estimated ART coverage among children (%)_max'],axis=1)
pediatric["childrenART"] = pediatric["Reported number of children receiving ART"]
pediatric["childrenART"] = pd.to_numeric(pediatric["childrenART"],errors="coerce")
pediatric["childrenART"] = pediatric["childrenART"].astype(float)
pediatric.info()
pediatricbyregion =pediatric.groupby("WHO Region").sum()

fig=plt.figure(figsize=(10,5))

sns.barplot(pediatricbyregion.index,pediatricbyregion["childrenART"]).set_title("Children receiving ART by region and year");

fig.show()
pediatric1byregion =pediatric.groupby("WHO Region").mean()["Estimated ART coverage among children (%)_median"]

fig=plt.figure(figsize=(10,5))

sns.barplot(pediatric1byregion.index,pediatric1byregion).set_title("ART coverage among children(%) by region and year");

fig.show()
prevention.info()
prevention.columns
prevention= prevention.drop([ 'Needing antiretrovirals',

       'Percentage Recieved', 

       'Needing antiretrovirals_min', 'Needing antiretrovirals_max',

       'Percentage Recieved_min',

       'Percentage Recieved_max'],axis=1)
prevention["recART"]= prevention["Received Antiretrovirals"]
prevention["recART"]= pd.to_numeric(prevention["recART"],errors='coerce')
prevention["recART"] = prevention["recART"].astype(float)
preventionbyregion = prevention.groupby("WHO Region").sum()

fig=plt.figure(figsize=(10,5))

sns.barplot(preventionbyregion.index,preventionbyregion["recART"]).set_title("People received Antiretrovirals by region and year");

fig.show()
prevention1byregion = prevention.groupby("WHO Region").mean()["Percentage Recieved_median"]

fig=plt.figure(figsize=(10,5))

sns.barplot(prevention1byregion.index,preventionbyregion["Percentage Recieved_median"]).set_title("People(%) received Antiretrovirals by region and year");

fig.show()
greater10 =cases[cases["Count_median"]>=10]

fig= plt.figure(figsize=(15,10))



sns.lineplot(data=greater10,x="Year",y="Count_median",hue="Country").set_title("Trend of top 10 countries with highest cases")

fig.show()



cases.nsmallest(25,"Count_median")
deaths[deaths["Count_median"]==deaths["Count_median"].max()]
highestdeaths =deaths[deaths["Count_median"]>=50000]

fig=plt.figure(figsize=(15,10))



sns.lineplot(data=highestdeaths,x="Year",y="Count_median",hue="Country").set_title("Trend in countries with high HIV deaths");

fig.show()
highestdeaths
deaths.nsmallest(25,"Count_median")
living[living["Count_median"]==living["Count_median"].max()]
highestliving =living[living["Count_median"]>=1000000]

fig=plt.figure(figsize=(15,10))

sns.lineplot(data=highestliving,x="Year",y="Count_median",hue="Country").set_title("Trend in countries with high population living with HIV");

fig.show()
highestliving
living.nsmallest(25,"Count_median")
coverage.nlargest(25, ["ART"])
coverage.nlargest(25, ["Estimated ART coverage among people living with HIV (%)_median"])
coverage.nsmallest(25, ["Estimated ART coverage among people living with HIV (%)_median"])
coverage.nsmallest(25, ["ART"])
pediatric.columns
pediatric.nlargest(25,"Estimated number of children needing ART based on WHO methods_median")
pediatric.nsmallest(25,"Estimated number of children needing ART based on WHO methods_median")
pediatric.nlargest(25,"childrenART")
pediatric.nsmallest(25,"childrenART")
pediatric.nlargest(25,"Estimated ART coverage among children (%)_median")
pediatric.nsmallest(25,"Estimated ART coverage among children (%)_median")
prevention.nlargest(25, "Needing antiretrovirals_median")
prevention.nsmallest(25, "Needing antiretrovirals_median")
prevention.nlargest(25,"recART")
prevention.nsmallest(25,"recART")
prevention.nlargest(25,"Percentage Recieved_median")
prevention.nsmallest(25,"Percentage Recieved_median")
Europe = cases[cases["WHO Region"]=="Europe"]

Europe[Europe["Count_median"]==Europe["Count_median"].max()]
Europed = deaths[deaths["WHO Region"]=="Europe"]

Europed[Europed["Count_median"]==Europed["Count_median"].max()]
Europel = living[living["WHO Region"]=="Europe"]

Europel[Europel["Count_median"]==Europel["Count_median"].max()]
Europetop = Europe[Europe["Count_median"]>=0.4]

fig = plt.figure(figsize=(30,15))

sns.set(rc={"lines.linewidth":6})



ax=sns.lineplot(data=Europetop,x="Year",y="Count_median",hue="Country").set_title("Trend of countries with high HIV cases in Europe",fontsize="25");



plt.legend(fontsize="x-large");



Europedtop = Europed[Europed["Count_median"]>=700]

fig = plt.figure(figsize=(30,15))



ax=sns.lineplot(data=Europedtop,x="Year",y="Count_median",hue="Country").set_title("HIV deaths: Trend in countries whith high deaths in Europe",fontsize="25");

plt.legend(fontsize="x-large");
Europetop
Europeltop = Europel[Europel["Count_median"]>=50000]

fig = plt.figure(figsize=(30,15))



ax=sns.lineplot(data=Europeltop,x="Year",y="Count_median",hue="Country").set_title("Living: Trend in countries with high no. of people living with HIV",fontsize="25");

plt.legend(fontsize="x-large");
EuropeART= coverage[coverage["WHO Region"]=="Europe"]

EuropeART.nlargest(10, ["Estimated ART coverage among people living with HIV (%)_median"])
EuropeART.nlargest(10, ["ART"])
EuropeART[EuropeART["Country"]=="Ukraine"]
Europeped = pediatric[pediatric["WHO Region"]=="Europe"]

Europeped.nlargest(10,"Estimated number of children needing ART based on WHO methods_median")
Europeped.nlargest(10,"childrenART")
Europeped.nlargest(10,"Estimated ART coverage among children (%)_median")
Europeped.nsmallest(10,"Estimated ART coverage among children (%)_median")
Europeped.info()
Europeprev = prevention[prevention["WHO Region"]=="Europe"]

Europeprev.info()
Europeprev.nlargest(15,"Needing antiretrovirals_median")
Europeprev.nlargest(10,"Percentage Recieved_median")
Americas = cases[cases["WHO Region"]=="Americas"]

Americas[Americas["Count_median"]==Americas["Count_median"].max()]
Americasd = deaths[deaths["WHO Region"]=="Americas"]

Americasd[Americasd["Count_median"]==Americasd["Count_median"].max()]
Americasl = living[living["WHO Region"]=="Americas"]

Americasl[Americasl["Count_median"]==Americasl["Count_median"].max()]

living[living["Country"]=="United States of America"]

Americastop = Americas[Americas["Count_median"]>=1]

fig = plt.figure(figsize=(30,15))



ax1=sns.lineplot(data=Americastop,x="Year",y="Count_median",hue="Country").set_title("Trend in countries with high HIV cases in Americas",fontsize="25");

plt.legend(fontsize="x-large");
Americasdtop = Americasd[Americasd["Count_median"]>=1000]

fig = plt.figure(figsize=(30,15))



ax1=sns.lineplot(data=Americasdtop,x="Year",y="Count_median",hue="Country").set_title("Deaths: Trend in countries with high deaths in Americas",fontsize="25");

plt.legend(fontsize="x-large");
Americasltop = Americasl[Americasl["Count_median"]>=100000]

fig = plt.figure(figsize=(30,15))



ax1=sns.lineplot(data=Americasltop,x="Year",y="Count_median",hue="Country").set_title("Living: Trend in countries with high no. of people living with HIV in Americas",fontsize="25");

plt.legend(fontsize="x-large");
Americasltop
Americastop
Americasdtop
AmericasART= coverage[coverage["WHO Region"]=="Americas"]

AmericasART.nlargest(10, "Estimated ART coverage among people living with HIV (%)_median")
AmericasART.nlargest(10, "ART")
AmericasART[AmericasART["Country"]=="Barbados"]
AmericasART[AmericasART["Country"]=="Guatemala"]
Americasped = pediatric[pediatric["WHO Region"]=="Americas"]

Americasped.nlargest(10,"Estimated number of children needing ART based on WHO methods_median")
Americasped.info()
Americasped.nlargest(10,"childrenART")
Americasped.nlargest(10,"Estimated ART coverage among children (%)_median")
Americasprev= prevention[prevention["WHO Region"]=="Americas"]

Americasprev.info()
Americasprev.nlargest(10,"Needing antiretrovirals_median")
Americasprev.nsmallest(10,"Needing antiretrovirals_median")
Americasprev.nlargest(10,"recART")
Americasprev.nsmallest(10,"recART")
Americasprev.nlargest(10,"Percentage Recieved_median")
Americasprev.nsmallest(10,"Percentage Recieved_median")
EM = cases[cases["WHO Region"]=="Eastern Mediterranean"]

EM[EM["Count_median"]==EM["Count_median"].max()]
EMd = deaths[deaths["WHO Region"]=="Eastern Mediterranean"]

EMd[EMd["Count_median"]==EMd["Count_median"].max()]
EMl = living[living["WHO Region"]=="Eastern Mediterranean"]

EMl[EMl["Count_median"]==EMl["Count_median"].max()]
EMtop = EM[EM["Count_median"]>=0.2]

fig = plt.figure(figsize=(30,15))



ax1=sns.lineplot(data=EMtop,x="Year",y="Count_median",hue="Country").set_title("Trend in countries with high HIV cases in Eastern Mediterranean",fontsize="25");

plt.legend(fontsize="x-large");
EMdtop = EMd[EMd["Count_median"]>=500]

fig = plt.figure(figsize=(30,15))



ax1=sns.lineplot(data=EMdtop,x="Year",y="Count_median",hue="Country").set_title("Deaths: Trend in countries with high deaths in Eastern Mediterranean",fontsize="25");

plt.legend(fontsize="x-large");
EMltop = EMl[EMl["Count_median"]>=50000]

fig = plt.figure(figsize=(30,15))



ax1=sns.lineplot(data=EMltop,x="Year",y="Count_median",hue="Country").set_title("Living: Trend in countries with high no. of people living with HIV in Eastern Mediterranean",fontsize="25");

plt.legend(fontsize="x-large");
EMART = coverage[coverage["WHO Region"]=="Eastern Mediterranean"]

EMART.nlargest(10, "Estimated ART coverage among people living with HIV (%)_median")
EMART.nlargest(10,"ART")
EMtop
EMped = pediatric[pediatric["WHO Region"]=="Eastern Mediterranean"]

EMped.info()
EMped.nlargest(10,"Estimated number of children needing ART based on WHO methods_median")
EMped.nlargest(10,"childrenART")
EMped.nlargest(10,"Estimated ART coverage among children (%)_median")
EMprev=prevention[prevention["WHO Region"]=="Eastern Mediterranean"]

EMprev.info()
EMprev.nlargest(17,"recART")
SA = cases[cases["WHO Region"]=="South-East Asia"]

SA[SA["Count_median"]==SA["Count_median"].max()]
SAd = deaths[deaths["WHO Region"]=="South-East Asia"]

SAd[SAd["Count_median"]==SAd["Count_median"].max()]
SAl = living[living["WHO Region"]=="South-East Asia"]

SAl[SAl["Count_median"]==SAl["Count_median"].max()]
SAtop = SA[SA["Count_median"]>=0.2]

fig = plt.figure(figsize=(30,15))



ax1=sns.lineplot(data=SAtop,x="Year",y="Count_median",hue="Country").set_title("Trend in countries with high HIV cases in South East Asia",fontsize="25");

plt.legend(fontsize="x-large");
SAdtop = SAd[SAd["Count_median"]>=0.2]

fig = plt.figure(figsize=(30,15))



ax1=sns.lineplot(data=SAdtop,x="Year",y="Count_median",hue="Country").set_title("Deaths: Trend in countries with high deaths in South East Asia",fontsize="25");

plt.legend(fontsize="x-large");
SAltop = SAl[SAl["Count_median"]>=100000]

fig = plt.figure(figsize=(30,15))



ax1=sns.lineplot(data=SAltop,x="Year",y="Count_median",hue="Country").set_title("Living: Trend in countries with high no .of people living with HIV in South East Asia",fontsize="25");

plt.legend(fontsize="x-large");
SAART = coverage[coverage["WHO Region"]=="South-East Asia"]

SAART.nlargest(10,"Estimated ART coverage among people living with HIV (%)_median")
SAART.nlargest(10,"ART")
SMped = pediatric[pediatric["WHO Region"]=="South-East Asia"]

SMped.info()
SMped
SAprev=prevention[prevention["WHO Region"]=="South-East Asia"]

SAprev.info()
SAprev
WP = cases[cases["WHO Region"]=="Western Pacific"]

WP[WP["Count_median"]==WP["Count_median"].max()]
WPd = deaths[deaths["WHO Region"]=="Western Pacific"]

WPd[WPd["Count_median"]==WPd["Count_median"].max()]
WPl = living[living["WHO Region"]=="Western Pacific"]

WPl[WPl["Count_median"]==WPl["Count_median"].max()]
WPtop = WP[WP["Count_median"]>=0.2]

fig = plt.figure(figsize=(30,15))



ax1=sns.lineplot(data=WPtop,x="Year",y="Count_median",hue="Country").set_title("Trend in countries with high HIV cases in Western Pacific",fontsize="25");

plt.legend(fontsize="x-large");
WPdtop = WPd[WPd["Count_median"]>=500]

fig = plt.figure(figsize=(30,15))



ax1=sns.lineplot(data=WPdtop,x="Year",y="Count_median",hue="Country").set_title("Deaths: Trend in countries with high deaths in Western Pacific",fontsize="25");

plt.legend(fontsize="x-large");
WPltop = WPl[WPl["Count_median"]>=50000]

fig = plt.figure(figsize=(30,15))



ax1=sns.lineplot(data=WPltop,x="Year",y="Count_median",hue="Country").set_title("Living: Trend in countries with high no. of people living with HIV in Western Pacific",fontsize="25");

plt.legend(fontsize="x-large");
WPtop
WPART = coverage[coverage["WHO Region"]=="Western Pacific"]

WPART.nlargest(10,"Estimated ART coverage among people living with HIV (%)_median")
WPART.nlargest(10,"ART")
WPped= pediatric[pediatric["WHO Region"]=="Western Pacific"]

WPped.info()
WPped
WPprev=prevention[prevention["WHO Region"]=="Western Pacific"]

WPprev.info()
WPprev
Africa = cases[cases["WHO Region"]=="Africa"]

Africa[Africa["Count_median"]==Africa["Count_median"].max()]
Africad = deaths[deaths["WHO Region"]=="Africa"]

Africad[Africad["Count_median"]==Africad["Count_median"].max()]
Africatop = Africa[Africa["Count_median"]>=10]

fig = plt.figure(figsize=(30,15))



ax1=sns.lineplot(data=Africatop,x="Year",y="Count_median",hue="Country").set_title("Trend in countries with high HIV cases in Africa",fontsize="25");

plt.legend(fontsize="x-large");
Africadtop = Africad[Africad["Count_median"]>=12000]

fig = plt.figure(figsize=(30,15))



ax1=sns.lineplot(data=Africadtop,x="Year",y="Count_median",hue="Country").set_title("Deaths: Trend in countries with high deaths in Africa",fontsize="25");

plt.legend(fontsize="x-large");
AfricaART=coverage[coverage["WHO Region"]=="Africa"]

AfricaART.nlargest(10,"Estimated ART coverage among people living with HIV (%)_median")
AfricaART.nlargest(10, "ART")
Africaped= pediatric[pediatric["WHO Region"]=="Africa"]

Africaped.info()
Africaped.nlargest(10,"Estimated number of children needing ART based on WHO methods_median")
Africaped.nlargest(10,"childrenART")
Africaped.nlargest(10,"Estimated ART coverage among children (%)_median")
Africaprev=prevention[prevention["WHO Region"]=="Africa"]

Africaprev.info()
Africaprev.nlargest(10,"Needing antiretrovirals_median")
Africaprev.nsmallest(10,"Needing antiretrovirals_median")
Africaprev.nlargest(10,"recART")
Africaprev.nsmallest(10,"recART")
Africaprev.nlargest(10,"Percentage Recieved_median")
Africaprev.nsmallest(10,"Percentage Recieved_median")
cases["Year"].unique()
c2000 = cases[cases["Year"]==2000][["Country","Count_median"]].set_index("Country")

c2005 = cases[cases["Year"]==2005][["Country","Count_median","WHO Region"]].set_index("Country")

c2010 = cases[cases["Year"]==2010][["Country","Count_median"]].set_index("Country")

c2018 = cases[cases["Year"]==2018][["Country","Count_median"]].set_index("Country")



d2000 = deaths[deaths["Year"]==2000][["Country","Count_median","WHO Region"]].set_index("Country")



d2010 = deaths[deaths["Year"]==2010][["Country","Count_median"]].set_index("Country")

d2018 = deaths[deaths["Year"]==2018][["Country","Count_median"]].set_index("Country")
l2000 = living[living["Year"]==2000][["Country","Count_median"]].set_index("Country")

l2005 = living[living["Year"]==2005][["Country","Count_median","WHO Region"]].set_index("Country")

l2010 = living[living["Year"]==2010][["Country","Count_median"]].set_index("Country")

l2018 = living[living["Year"]==2018][["Country","Count_median"]].set_index("Country")
Trend = pd.DataFrame()

Trend["Country"]=cases["Country"].unique()

Trend["Increasing"]= Trend["Country"].apply(lambda x: c2018.loc[x,"Count_median"]-c2000.loc[x,"Count_median"])

Trend["Inc2010"] = Trend["Country"].apply(lambda x: c2018.loc[x,"Count_median"]-c2010.loc[x,"Count_median"])
Trendd = pd.DataFrame()

Trendd["Country"]=cases["Country"].unique()

Trendd["Increasing"]= Trendd["Country"].apply(lambda x: d2018.loc[x,"Count_median"]-d2000.loc[x,"Count_median"])

Trendd["Inc2010"] = Trendd["Country"].apply(lambda x: d2018.loc[x,"Count_median"]-d2010.loc[x,"Count_median"])
Trendl = pd.DataFrame()

Trendl["Country"]=cases["Country"].unique()

Trendl["Increasing"]= Trendl["Country"].apply(lambda x: l2018.loc[x,"Count_median"]-l2000.loc[x,"Count_median"])

Trendl["Inc2010"] = Trend["Country"].apply(lambda x: l2018.loc[x,"Count_median"]-l2010.loc[x,"Count_median"])
Trend
Trend["WHO Region"]=  Trend["Country"].apply(lambda x: c2005.loc[x,"WHO Region"])
Trendd["WHO Region"]=  Trendd["Country"].apply(lambda x: d2000.loc[x,"WHO Region"])
Trendl["WHO Region"]=  Trendl["Country"].apply(lambda x: l2005.loc[x,"WHO Region"])
Trend[Trend["Increasing"]> 0]
Trendd[Trendd["Increasing"]> 0]
Trendl[Trendl["Increasing"]> 0]
Trendl[Trendl["Increasing"]>= 100000]
Trend[Trend["Inc2010"]>0]
Trendd[Trendd["Inc2010"]>0]
Trendl[Trendl["Inc2010"]>0]
Trend[Trend["Increasing"]< 0]
Trendd[Trendd["Increasing"]< 0]
Trendl[Trendl["Increasing"]< 0]
Trend[Trend["Inc2010"]<0]
Trendd[Trendd["Inc2010"]<0]
Trendl[Trendl["Inc2010"]<0]