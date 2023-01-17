import pandas as pd
#Answer 1
df1=pd.read_csv(r'/kaggle/input/world-happiness/2019.csv')
df2=pd.read_csv(r'/kaggle/input/world-happiness/2018.csv')
df3=pd.read_csv(r'/kaggle/input/world-happiness/2017.csv')
df4=pd.read_csv(r'/kaggle/input/world-happiness/2016.csv')
df5=pd.read_csv(r'/kaggle/input/world-happiness/2015.csv')
df1
#Answer 1

df2
#Answer 1
df3
#Answer 1
df4
#Answer 1
df5
#Answer 2
df1.info()
#Answer 3
df1.mean()
#Answer 3
df1.median()
#Answer 3
df=df1[['Score','GDP per capita','Social support','Healthy life expectancy','Freedom to make life choices','Generosity']]
df.mode()
#Answer 4
data=list(df1._get_numeric_data().columns)
data
#Answer 4
ordin = list(set(df1.columns) - set(df1._get_numeric_data().columns))
ordin
#Answer 5
#Bar graph to compare GDP per capita of all countries in 2019
plot1=df1
plot_main=plot1[['Country or region','GDP per capita']]
plot_main.plot.bar('Country or region',figsize=(20,10))
#Answer 5
#Histogram of scores in the year 2018
import matplotlib.pyplot as plt

plt.hist(df2.Score)
plt.title("Histogram")
plt.xlabel('Score')
plt.show()
#Answer 5
#Comparing the all countries in the year 2019
plot1=df1
plot_main=plot1[['Country or region','GDP per capita','Social support','Healthy life expectancy','Freedom to make life choices','Generosity']]
plot_main.plot('Country or region',figsize=(20,10))

#Answer 6 =>  If the value is >=0 then columns are correlated 
#Answer 7 =>  If the value is <0 then columns are correlated 
corr=df1.corr()
corr
#Answer 8
#Comparing the ranks of Countries in the year 2019 and 2018
import numpy as np
compare=np.where(df1["Overall rank"] == df2["Overall rank"], True, False)
df1["Same as last year"] = compare
df1