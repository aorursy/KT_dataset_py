import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
print('Modules are imported.')
df=pd.read_csv("../input/covid19/covid19_Confirmed_dataset.csv")
df.head()
df.shape
df.drop(["Lat","Long"],axis=1,inplace=True)
df.head()
aggregating=df.groupby("Country/Region").sum()
aggregating.head()
aggregating.shape
aggregating.loc["China"].plot()
aggregating.loc["Italy"].plot()
aggregating.loc["Spain"].plot()
plt.legend()
aggregating.loc['China'].plot()
aggregating.loc['China'][:3].plot()
aggregating.loc['China'].diff().plot()
aggregating.loc['China'].diff().max()
aggregating.loc['Italy'].diff().max()
aggregating.loc['Spain'].diff().max()
countries=list(aggregating.index)
max_infection_rates=[]
for c in countries:
    max_infection_rates.append(aggregating.loc[c].diff().max())
aggregating["max_infection_rates"]=max_infection_rates
aggregating.head()
data=pd.DataFrame(aggregating["max_infection_rates"])
data.head()
happiness=pd.read_csv("../input/covid19/worldwide_happiness_report.csv")
happiness.head()
cols=["Overall rank","Score","Generosity","Perceptions of corruption"]
happiness.drop(cols,axis=1,inplace=True)
happiness.head()
happiness.set_index("Country or region",inplace=True)
happiness.head()
data.head()
happiness.head()
final=data.join(happiness,how="inner")
final.head()
final.corr()
final.head()
x=final["GDP per capita"]
y=final["max_infection_rates"]
sns.scatterplot(x,np.log(y))
sns.regplot(x,np.log(y))
x=final["Social support"]
y=final["max_infection_rates"]
sns.scatterplot(x,np.log(y))
sns.regplot(x,np.log(y))
x=final["Healthy life expectancy"]
y=final["max_infection_rates"]
sns.scatterplot(x,np.log(y))
sns.regplot(x,np.log(y))
x=final["Freedom to make life choices"]
y=final["max_infection_rates"]
sns.scatterplot(x,np.log(y))
sns.regplot(x,np.log(y))