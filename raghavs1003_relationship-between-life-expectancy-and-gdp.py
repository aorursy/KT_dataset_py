import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
lifeExp = pd.read_csv("../input/life-expectancy-and-gdp/dataset1_life_expectancy.csv")

worldPop = pd.read_csv("../input/life-expectancy-and-gdp/dataset3_World_Population.csv")

gdp = pd.read_csv("../input/life-expectancy-and-gdp/dataset2_GDP_per_capita.csv")
lifeExp.head()
lifeExp2017 = lifeExp[lifeExp.Year == 2017]
lifeExp2017_colm = lifeExp2017[['Entity', 'Code', 'Year', 'Life expectancy (years)']]
lifeExp2017_colm.head(10)
worldPop2017 = worldPop[['Country Name', 'Country Code', '2017']]
worldPop2017.head()
gdp2017 = gdp[['Country Name', 'Country Code', '2017']]
gdp2017.head()
merged_1 = pd.merge(lifeExp2017_colm, worldPop2017 , left_on=['Entity'], right_on=['Country Name'], how='inner')
merged_1.head()
merged_1_afterRename = merged_1.rename({'2017':'Population'}, axis='columns')
gdp2017_afterRename = gdp2017.rename({'Country Name':'Country', 'Country Code':'Country Code in GDP', '2017':'GDP'}, axis=1)
merged_1_afterRename[merged_1_afterRename["Entity"] == "World"]
merged_1_afterRename.drop([117], axis=0)
merged_2 = pd.merge(merged_1_afterRename, gdp2017_afterRename , left_on=['Entity'], right_on=['Country'], how='inner')
merged_2.count()
merged_2.sort_values('Life expectancy (years)').head()
merged_2.sort_values('Life expectancy (years)', ascending=False).head(10)
forGraph1_1 = merged_2.sort_values('Life expectancy (years)', ascending=False).head(10)
lifeExp1987 = lifeExp[lifeExp.Year == 1987]
lifeExp1987_colm = lifeExp1987[['Entity', 'Code', 'Year', 'Life expectancy (years)']]
lifeExp1987_before_final = lifeExp1987_colm[(lifeExp1987_colm.Entity == 'Monaco') | (lifeExp1987_colm.Entity == 'San Marino') | (lifeExp1987_colm.Entity == 'Hong Kong')| (lifeExp1987_colm.Entity == 'Japan')| (lifeExp1987_colm.Entity == 'Macao') | (lifeExp1987_colm.Entity == 'Cayman Islands')| (lifeExp1987_colm.Entity == 'Switzerland') | (lifeExp1987_colm.Entity == 'Andorra')| (lifeExp1987_colm.Entity == 'Spain')| (lifeExp1987_colm.Entity == 'Singapore')]
lifeExp1987_final=lifeExp1987_before_final.rename({'Entity':'Entity 1987'}, axis='columns')
lifeExp1987_final.head(2)
dataForGraph_1 = pd.merge(forGraph1_1, lifeExp1987_final , left_on=['Entity'], right_on=['Entity 1987'], how='inner')
dataForGraph_1.head(11)
import matplotlib.ticker as ticker

import numpy as np

import matplotlib.pyplot as plt



# data to plot

#n_groups = 10

#means_frank = (90, 55, 40, 65)

#means_guido = (85, 62, 54, 20)



tick_spacing = 3



# create plot

fig, ax = plt.subplots(figsize=(10,7))

ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

index = np.arange(10)

bar_width = .45

opacity = 0.8

ax.set_axisbelow(True)

ax.grid(color='whitesmoke')



#rects1 = plt.bar(index, dataForGraph_1['Life expectancy (years)_x'], bar_width, alpha=opacity,color='yellow',label='2017')

rects1 = plt.bar(index, dataForGraph_1['Life expectancy (years)_x'], bar_width, alpha=opacity,color='c',label='2017')



#rects2 = plt.bar(index + bar_width, dataForGraph_1['Life expectancy (years)_y'], bar_width, alpha=opacity, color='k', label='1987')

rects2 = plt.bar(index + bar_width, dataForGraph_1['Life expectancy (years)_y'], bar_width, alpha=opacity, color='lightblue', label='1987')



plt.ylim(0, 95)

plt.xlabel('Countries with highest life expectancy',fontsize=13.5)

plt.ylabel('Life expectancy (in years)',fontsize=13.5)

#plt.figtext(0.47, 0.99, "2017", fontsize='medium', color='k', ha ='right')

#plt.figtext(0.47, 0.96, "1987", fontsize='medium', color='c', ha ='left')

plt.title('Life expectancy comparision between 2017 and 1987',fontsize=15)

plt.xticks(index + bar_width/2,('Monaco', 'San Marino', 'Hong Kong', 'Japan', 'Macao', 'Cayman Islands', 'Switzerland', 'Andorra', 'Spain', 'Singapore'), rotation='90')

plt.legend()



#plt.tight_layout()

plt.show()
#merged_2_updated = merged_2['GDP'].replace('', np.nan, inplace=True)

#merged_2_updated.dropna(subset=['GDP'], inplace=True)

fig, ax = plt.subplots(figsize=(10, 7))

ax.grid(color='whitesmoke')

colors = np.random.rand(188) 

plt.xlim(0,120)

#plt.ylim(-1, 1)

plt.xlabel('GDP per capita (in per 1000 US $)',fontsize=13.5)

plt.ylabel('Life Expectancy (in years)',fontsize=13.5)

plt.title('Life expectancy Vs GDP in 2017',fontsize=15)

# use the scatter function

plt.scatter(x=merged_2['GDP']/1000,y=merged_2['Life expectancy (years)'],s=merged_2['Population']/1000000, marker="o",c=colors)

plt.show()
fig, ax = plt.subplots()

ax.grid(color='grey')

colors = np.random.rand(10) 

plt.xlabel('GDP per capita')

plt.ylabel('Life Expectancy (in years)')

plt.title('Life expectancy with GDP')

# use the scatter function

plt.scatter(x=forGraph1_1['GDP']/1000,y=forGraph1_1['Life expectancy (years)'],s=forGraph1_1['Population']/100000, marker="8", c=colors)

plt.show()
forGraph3 = merged_2.sort_values('Life expectancy (years)', ascending=True).head(10)
forGraph3
lifeExp1987_before_final = lifeExp1987_colm[(lifeExp1987_colm.Entity == 'Central African Republic') | (lifeExp1987_colm.Entity == 'Lesotho') | (lifeExp1987_colm.Entity == 'Chad')| (lifeExp1987_colm.Entity == 'Sierra Leone')| (lifeExp1987_colm.Entity == 'Nigeria') | (lifeExp1987_colm.Entity == 'Somalia')| (lifeExp1987_colm.Entity == "Cote d'Ivoire") | (lifeExp1987_colm.Entity == 'South Sudan')| (lifeExp1987_colm.Entity == 'Guinea-Bissau')| (lifeExp1987_colm.Entity == 'Equatorial Guinea')]
lifeExp1987_final=lifeExp1987_before_final.rename({'Entity':'Entity 1987'}, axis='columns')
#lifeExp1987_final

dataForGraph_3 = pd.merge(forGraph3, lifeExp1987_final , left_on=['Entity'], right_on=['Entity 1987'], how='inner')
dataForGraph_3
import matplotlib.ticker as ticker

import numpy as np

import matplotlib.pyplot as plt

#import matplotlib.pyplot as plt2



# data to plot

#n_groups = 10

#means_frank = (90, 55, 40, 65)

#means_guido = (85, 62, 54, 20)

tick_spacing = 3

fig = plt.figure(figsize=(14,7))

ax2 = fig.add_subplot(1,2,1)

ax2.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

index = np.arange(10)

bar_width = .45

opacity = 0.8

ax2.set_axisbelow(True)

ax2.grid(color='whitesmoke')



#rects1 = plt.bar(index, dataForGraph_1['Life expectancy (years)_x'], bar_width, alpha=opacity,color='yellow',label='2017')

rects1 = plt.bar(index, dataForGraph_1['Life expectancy (years)_x'], bar_width, alpha=opacity,color='c',label='2017')



#rects2 = plt.bar(index + bar_width, dataForGraph_1['Life expectancy (years)_y'], bar_width, alpha=opacity, color='k', label='1987')

rects2 = plt.bar(index + bar_width, dataForGraph_1['Life expectancy (years)_y'], bar_width, alpha=opacity, color='lightblue', label='1987')



plt.ylim(0, 95)

plt.xlabel('Countries with highest life expectancy',fontsize=13.5)

plt.ylabel('Life expectancy (in years)',fontsize=13.5)

#plt.figtext(0.47, 0.99, "2017", fontsize='medium', color='k', ha ='right')

#plt.figtext(0.47, 0.96, "1987", fontsize='medium', color='c', ha ='left')

plt.title('Comparing life expectancy between 2017 and 1987',fontsize=15)

plt.xticks(index + bar_width/2,('Monaco', 'San Marino', 'Hong Kong', 'Japan', 'Macao', 'Cayman Islands', 'Switzerland', 'Andorra', 'Spain', 'Singapore'), rotation='90')

plt.legend()



#plt.tight_layout()

#plt.show()



#----------------------------------------------------------------------------------------------





# create plot

#fig = plt.figure()

#fig = plt.figure(figsize=(10,7))

ax = fig.add_subplot(1,2,2)

#ax2 = fig.add_subplot(1,2,2)

#fig, ax = plt.subplots(figsize=(10,7),sharex='True')

#fig, ax = plt.subplots(121,50)

index = np.arange(10)

bar_width = .45

opacity = 0.8

ax.set_axisbelow(True)

ax.grid(color='whitesmoke')

tick_spacing = 3

ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

#rects1 = plt.bar(index, dataForGraph_1['Life expectancy (years)_x'], bar_width, alpha=opacity,color='yellow',label='2017')

rects3 = plt.bar(index, dataForGraph_3['Life expectancy (years)_x'], bar_width, alpha=opacity,color='darkorange',label='2017')







#rects2 = plt.bar(index + bar_width, dataForGraph_1['Life expectancy (years)_y'], bar_width, alpha=opacity, color='k', label='1987')

rects4 = plt.bar(index + bar_width, dataForGraph_3['Life expectancy (years)_y'], bar_width, alpha=opacity, color='moccasin', label='1987')



plt.ylim(0,95)

plt.xlabel('Countries with lowest life expectancy',fontsize=13.5)

plt.ylabel('Life expectancy (in years)',fontsize=13.5)

#plt.figtext(0.47, 0.99, "2017", fontsize='medium', color='k', ha ='right')

#plt.figtext(0.47, 0.96, "1987", fontsize='medium', color='c', ha ='left')

plt.title('Comparing life expectancy between 2017 and 1987',fontsize=15)

plt.xticks(index + bar_width/2,('Central African Republic', 'Chad', "Cote d'Ivoire", 'Equatorial Guinea', 'Guinea-Bissau', 'Lesotho', 'Nigeria', 'Sierra Leone', 'Somalia', 'South Sudan'), rotation='90')

plt.legend()

plt.show()