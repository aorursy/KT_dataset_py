import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib



data = pd.read_csv("../input/2015.csv")
data.info()
data.columns
data.rename(columns={'Economy (GDP per Capita)': 'Economy'}, inplace=True)
data.rename(columns={'Happiness Score': 'Happiness'}, inplace=True)
data.rename(columns={'Health (Life Expectancy)': 'Health'}, inplace=True)
data.rename(columns={'Trust (Government Corruption)': 'Trust'}, inplace=True)
data.columns
data.describe()
data.head(10)
data.tail(10)
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
data.plot(kind='scatter', x='Economy', y='Happiness',alpha = 0.5,color = 'red')
plt.xlabel('Economy')              
plt.ylabel('Happiness')
plt.title('Economy / Happiness Score  Plot')
plt.show()
data1 = data.loc[:,["Health","Trust","Economy"]]
data1.plot()
plt.show()
AVG_ECONOMY = data.Economy.mean()
data["COMPRISE"]=["LOW" if AVG_ECONOMY> each else "HIGH" for each in data.Economy]
a=data["COMPRISE"]=="HIGH"
data[a]
plt.figure(figsize=(20,7))
h = plt.hist(pd.to_numeric(data.Happiness).dropna(), facecolor='g', alpha=0.75, bins=100)
plt.title("Distribution of Happiness")
plt.xlabel("Happiness")
plt.ylabel("Count")
plt.show()
AVG_ECONOMY = data.Economy.mean()
AVG_Happiness=data.Happiness.mean()
AVG_HEALTH=data.Health.mean()
AVG_FREEDOM=data.Freedom.mean()
AVG_TRUST=data.Trust.mean()
filter1=data["Economy"]>AVG_ECONOMY
filter2=data["Trust"]>AVG_TRUST
filter3=data["Health"]>AVG_HEALTH
filter4=data["Happiness"]>AVG_Happiness
data[filter1&filter2&filter3&filter4]
print(data.mean())
data2=data.copy()
df = pd.DataFrame(data2)
df.drop(['Happiness Rank', 'Standard Error','COMPRISE'], axis=1,inplace = True)
gr=df.groupby('Region')
gr
ort=gr.mean()
ort



df2=pd.DataFrame(data2)
df2=data2.loc[:,['Country','Happiness' ,'Economy','Family','Health','Freedom','Trust','Generosity','Dystopia Residual']]
x=df2['Country']=='Turkey'
df2[x]
