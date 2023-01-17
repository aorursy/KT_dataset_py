import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.plotly as py
import warnings
warnings.filterwarnings("ignore")
%matplotlib inline
data_2015 = pd.read_csv("../input/2015.csv")
data_2016 = pd.read_csv("../input/2016.csv")
data_2017 = pd.read_csv("../input/2017.csv")
plt.figure(figsize=(20,10))

plt.plot( data_2015['Happiness Rank'],data_2015['Region'])

plt.plot( data_2016['Happiness Rank'],data_2016['Region'])

plt.ylabel("Region")
plt.xlabel("Happiness Rank")
sns.pairplot(data_2015.drop(['Standard Error','Dystopia Residual'],axis=1))
plt.figure(figsize=(20,10))

plt.subplot(411)
plt.plot(data_2015['Happiness Score'], data_2015['Economy (GDP per Capita)'])
plt.ylabel("Economy")

plt.subplot(412)
plt.plot(data_2015['Happiness Score'], data_2015['Family'])
plt.ylabel("Family")
plt.xlabel("Happiness_score_ratio")

plt.subplot(413)
plt.plot(data_2015['Happiness Score'], data_2015['Health (Life Expectancy)'])
plt.ylabel("Health")

plt.subplot(414)
plt.plot(data_2015['Happiness Score'], data_2015['Freedom'])

plt.xlabel("Happiness_score_ratio")
plt.ylabel("Freedom")
x=data_2015['Economy (GDP per Capita)']
y=data_2015['Happiness Score']
sns.set_context({"figure.figsize": (10, 5)})
sns.kdeplot(x,data_2015['Happiness Score'],shade=True, cut=3)
sns.lmplot(x="Trust (Government Corruption)", y="Economy (GDP per Capita)", data=data_2015)
data_2015.plot.scatter(x='Family', y='Happiness Score',color='r')
data_2015.plot.scatter(x='Family', y='Health (Life Expectancy)',color='r')
sns.lmplot(x="Generosity", y="Happiness Score", data=data_2015)
region_list15=data_2015['Region'].unique()
happiness_score_ratio15 = []
Economy = []
Family = []
Health = []
Freedom = []
Trust = []
Generosity = []

for i in region_list15 :
    x=data_2015[data_2015['Region']==i]
    happiness_score_ratio15.append(x["Happiness Score"].sum()/len(x))
    Economy.append(x["Economy (GDP per Capita)"].sum()/len(x))
    Family.append(x["Family"].sum()/len(x))
    Health.append(x["Health (Life Expectancy)"].sum()/len(x))
    Freedom.append(x["Freedom"].sum()/len(x))
    Trust.append(x["Trust (Government Corruption)"].sum()/len(x))
    Generosity.append(x["Generosity"].sum()/len(x))
    
    
data = pd.DataFrame(
    {
        "region_list":region_list15, "happiness_score_ratio":happiness_score_ratio15,"Economy":Economy,
        "Family":Family,"Health":Health,"Freedom":Freedom,"Trust":Trust,"Generosity":Generosity
    })

new_index = (data["happiness_score_ratio"].sort_values(ascending=False)).index.values

colors = np.random.rand(10)
area = (data['happiness_score_ratio'])**4   
plt.figure(figsize=(10,8))
plt.scatter(data['happiness_score_ratio'],data['region_list'],  s=area, c=colors, alpha=0.5)
plt.title("Happiness v/s Region")
plt.xlabel("Happiness_score_ratio")
plt.ylabel("Region_list")
plt.show()
colors = np.random.rand(10)
area = (data['happiness_score_ratio'])**4   
plt.figure(figsize=(10,8))
plt.scatter(data['happiness_score_ratio'],data['Economy'],  s=area, c=colors, alpha=0.5)
plt.title("Happiness v/s Economy")
plt.xlabel("Happiness_score_ratio")
plt.ylabel("Economy")
plt.show()
colors = np.random.rand(10)
area = (data['happiness_score_ratio'])**4   
plt.figure(figsize=(10,8))
plt.scatter(data['happiness_score_ratio'],data['Family'],  s=area, c=colors, alpha=0.5)
plt.title("Happiness v/s Family")
plt.xlabel("Happiness_score_ratio")
plt.ylabel("Family")
plt.show()
colors = np.random.rand(10)
area = (data['happiness_score_ratio'])**4   
plt.figure(figsize=(10,8))
plt.scatter(data['happiness_score_ratio'],data['Health'],  s=area, c=colors, alpha=0.5)
plt.title("Happiness v/s Health")
plt.xlabel("Happiness_score_ratio")
plt.ylabel("Health")
plt.show()
colors = np.random.rand(10)
area = (data['happiness_score_ratio'])**4   
plt.figure(figsize=(10,8))
plt.scatter(data['happiness_score_ratio'],data['Freedom'],  s=area, c=colors, alpha=0.5)
plt.title("Happiness v/s Freedom")
plt.xlabel("Happiness_score_ratio")
plt.ylabel("Freedom")
plt.show()
colors = np.random.rand(10)
area = (data['happiness_score_ratio'])**4   
plt.figure(figsize=(10,8))
plt.scatter(data['happiness_score_ratio'],data['Trust'],  s=area, c=colors, alpha=0.5)
plt.title("Happiness v/s Trust")
plt.xlabel("Happiness_score_ratio")
plt.ylabel("Trust")
plt.show()
colors = np.random.rand(10)
area = (data['happiness_score_ratio'])**4   
plt.figure(figsize=(10,8))
plt.scatter(data['happiness_score_ratio'],data['Generosity'],  s=area, c=colors, alpha=0.5)
plt.title("Happiness v/s Generosity")
plt.xlabel("Happiness_score_ratio")
plt.ylabel("Generosity")
plt.show()
label=[ 'happiness_score_ratio', 'Economy', 'Family', 'Health', 'Freedom', 'Trust', 'Generosity']
plt.figure(figsize=(25,8))
plt.stackplot(data['region_list'],data['happiness_score_ratio'],data['Economy'],data['Family'],data['Health'],data['Freedom'],
              data['Trust'],data['Generosity'],labels=label)
plt.legend(loc='upper right')
plt.xlabel("region_list")
plt.ylabel("features")




