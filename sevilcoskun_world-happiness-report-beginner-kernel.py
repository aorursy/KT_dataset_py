# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt#plot drawing
import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/2017.csv")
print("Data Info")
data.info()
#Change . sepertor with _ 
data.columns = data.columns.str.replace(".", "_")



data.columns = data.columns.str.replace("Economy__GDP_per_Capita_", "Economy")
data.columns = data.columns.str.replace("Health__Life_Expectancy_","Health")
data.columns = data.columns.str.replace("Trust__Government_Corruption_","Government_trust")
data.columns
print("2017 data set")
data.corr() 
#correlation map view
f, ax = plt.subplots(figsize = (8,8))
sns.heatmap(data.corr(), annot = True, linewidths=.5, fmt = ".2f", ax=ax)
plt.show()
data.describe()
plt.plot(data.Happiness_Score, data.Economy, color = "red", label = "Economy",alpha = 0.8)
plt.plot(data.Happiness_Score, data.Family, color = "yellow", label = "Family",alpha = 0.8)
plt.plot(data.Happiness_Score, data.Freedom, color = "blue", label = "Freedom",alpha = 0.8)
plt.plot(data.Happiness_Score, data.Health, color = "green", label = "Health",alpha = 0.8)

plt.legend()    
plt.xlabel('Happiness Score')           
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()

#data.plot(kind="scatter", x = "Economy", y ="Family", color = "yellow", label = "Economy-Family")
#data.plot(kind="scatter", x = "Economy", y ="Health", color = "green", label = "Economy-Health")
#data.plot(kind="scatter", x = "Economy", y ="Freedom", color = "blue", label = "Economy- Freedom")
plt.scatter(data.Economy, data.Family, color = "yellow", label = "Family")
plt.scatter(data.Economy, data.Health, color = "green", label = "Health")
plt.scatter(data.Economy, data.Freedom, color = "blue", label = "Freedom")
plt.xlabel('Economy') 
plt.legend()
plt.title('Economy Relations')            
plt.show()
plt.scatter(data.Happiness_Score, data.Health, color = "green", label = "Health")
plt.scatter(data.Happiness_Score, data.Freedom, color = "blue", label = "Freedom")
plt.xlabel('Happiness Score') 
plt.legend()
plt.title('Happiness Score Relations')            
plt.show()
happy = data[data["Happiness_Score"] > 5.3] 
unhappy = data[data["Happiness_Score"] < 5.3] 

happy.Economy.plot(kind = 'hist',bins = 50,figsize = (8,8), color = "blue", alpha = 0.7, label = "Happy")
unhappy.Economy.plot(kind = 'hist',bins = 50,figsize = (8,8), color = "red", alpha = 0.7, label = "Unhappy")

plt.xlabel('Economy')             
plt.title('Happiness and Economy relations') 
plt.legend()
plt.show()
happy = data[data["Happiness_Score"] > 5.3] 
unhappy = data[data["Happiness_Score"] < 5.3]

happy.Freedom.plot(kind = 'hist',bins = 50,figsize = (8,8), color = "blue", alpha = 0.7, label = "Happy")
unhappy.Freedom.plot(kind = 'hist',bins = 50,figsize = (8,8), color = "red", alpha = 0.7, label = "Unhappy")

plt.xlabel('Freedom')             
plt.title('Happiness and Freedom relations') 
plt.legend()
plt.show()
data = pd.read_csv('../input/2016.csv')
data.columns
#I need to make some changes about column names to work more better
#Change . sepertor with _ 
data.columns = data.columns.str.replace(" ", "")
data.columns = data.columns.str.replace("(", "")
data.columns = data.columns.str.replace(")", "")

data.columns = data.columns.str.replace("EconomyGDPperCapita", "Economy")
data.columns = data.columns.str.replace("HealthLifeExpectancy","Health")
data.columns = data.columns.str.replace("TrustGovernmentCorruption","Government_trust")
data.columns
series = data['HappinessScore']
print(type(series))
#Creating DataFrames to work more quicky
#df is a column of happiness score --> new data frame creation
df = data[['HappinessScore']]
print(type(df))

happy = data[data["HappinessScore"] > 5.3]
unhappy = data[data["HappinessScore"] < 5.3]
print("First 5 happy Countries scores")
happy.head()
happy.columns
print("TOP 10 HAPPY COUNTRIES")
west = happy[happy["Region"] == "Western Europe"]
cnt = 0
for index,row in west.iterrows():
    if(cnt < 10):
        print(happy.Country[index])
        cnt += 1

print("FIRST 10 UNHAPPY COUNTRIES")

westu = unhappy[unhappy["Region"] == "Western Europe"]
cnt = 0
for index,row in westu.iterrows():
    if(cnt < 10):
        print(unhappy.Country[index])
        cnt += 1



data.info()
#Find the happiest countries above from threashold value
df_a = pd.DataFrame(columns=["Country","HappinessScore", "Economy"])
df_b = pd.DataFrame(columns=["Country","HappinessScore", "Economy"])
#Find the happiest countries above from threashold value
def above_threashold(thr,avg = 5.3):
    if(thr >= avg):
        for index, value in data.iterrows():
            if((data.HappinessScore[index] > thr) & (data.HappinessScore[index] > avg)):
                print("(",data.Country[index],"-", data.HappinessScore[index], ") above ", thr)
            elif((data.HappinessScore[index] < thr) & (data.HappinessScore[index] > avg)):
                print("(",data.Country[index],"-", data.HappinessScore[index], ") below ", thr)
            else:
                continue
    else:
        print("your threashold is lower than average happiness score")        

print(above_threashold(7))




#df_a.HappinessScore.plot(kind = 'line', figsize = (8,8), color = "blue", alpha = 0.7, label = "Above Countries")
#df_b.HappinessScore.plot(kind = 'line',figsize = (8,8), color = "red", alpha = 0.7, label = "Below Countries")
#plt.plot(df_a.Country, df_a.HappinessScore, color = "red", label = "Above Counries",alpha = 0.7)
#plt.plot(df_b.Country, df_b.HappinessScore, color = "blue", label = "Below Counries",alpha = 0.7)
west_countries = list(west.Country)
happy_score = list(west.HappinessScore)
c_list = []
s_list = []
for each in range(len(west_countries)):
    c_name = west_countries[each]
    h_score = int(happy_score[each])
    i_name = iter(c_name)
    c_list.append(next(i_name))
    s_list.append(h_score)
print("clist: ", c_list, "slist: ", s_list)
#ZIP 2 list to combine
z = zip(c_list, s_list)
print(z)
z_list = list(z)
print(z_list)

data.Economy.describe()
# Conditionals on iterable
threshold = sum(data.Economy) / len(data.Economy)
data["Economical_level"] = ["high" if i > threshold else "low" for i in data.Economy]

hec = data[data["Economical_level"] == "high"]
lec = data[data["Economical_level"] == "low"]

plt.plot(hec.Generosity, hec.HappinessScore, color = "red", label = "Economically High",alpha = 0.8 )
plt.plot(lec.Generosity, lec.HappinessScore, color = "blue", label = "Economically Low",alpha = 0.8 )

plt.xlabel('Generosity')             
plt.title('How Generosity is effect on Happiness in economically high and low countries') 
plt.legend()
plt.show()



