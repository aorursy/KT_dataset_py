import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib.cm as cm
df_2017 = pd.read_csv("../input/2017.csv")
df_2016 = pd.read_csv("../input/2016.csv")
df_2015 = pd.read_csv("../input/2015.csv")
df_2017.head()

df_2017['year'] = 2017
df_2016['year'] = 2016
df_2015['year'] = 2015
df_2017_top10 = df_2017.loc[df_2017['Happiness.Rank']<=10,:]
df_2016_top10 = df_2016.loc[df_2016['Happiness Rank']<=10,:]
df_2015_top10 = df_2015.loc[df_2015['Happiness Rank']<=10,:]
(df_2017.shape,
 df_2016.shape,
 df_2015.shape
)
df_2017_top10 = df_2017_top10.iloc[:,[0,1,2,12]]
df_2016_top10 = df_2016_top10.iloc[:,[0,2,3,13]]
df_2015_top10 = df_2015_top10.iloc[:,[0,2,3,12]]
(list(df_2017_top10.columns),
 list(df_2016_top10.columns),
 list(df_2015_top10.columns),
)
df_2016_top10.columns = ['Country', 'Happiness.Rank', 'Happiness.Score','year']
df_2015_top10.columns = ['Country', 'Happiness.Rank', 'Happiness.Score','year']
df_concat = pd.concat([df_2017_top10,df_2016_top10,df_2015_top10], sort = True)
fig, ax = plt.subplots(figsize=(15,7), subplot_kw={'ylim': (1,10)})
df_grouped = df_concat.groupby(['Country','year'])['Happiness.Score'].mean()
ax.set_ylabel('Happiness Score')
df_grouped.unstack().plot.bar(ax= ax)

#Top 10 countries
df_scatter = df_2017[:10]
#preparing columns
country =df_scatter['Country']
happy_Score = df_scatter['Happiness.Score']
bubble_Size = df_scatter['Happiness.Score']*250
life_Exp = df_scatter['Health..Life.Expectancy.']
#rounding 3 digit after decimal
gpd = df_scatter['Economy..GDP.per.Capita.'].round(3)
# x,y
x = np.arange(10)
ys = [i+x+(i*x)**2 for i in range(10)]
# Choose some random colors
colors = cm.rainbow(np.linspace(0, 1, len(ys)))
#figure size
plt.figure(figsize=(10,8))
#scatter plot
plt.scatter(happy_Score,life_Exp, s = bubble_Size, color=colors)
#text (country and gpd)
for i in range(country.shape[0]):
    plt.annotate((country[i],gpd[i]),xy=(happy_Score[i],life_Exp[i]))
plt.xlabel('Happiness Score')
plt.ylabel('LifeExp')

# Move title up with the "y" option
plt.title('LifeExp vs Happiness Score and GDP per Capita.')
plt.show()