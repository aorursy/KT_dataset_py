import numpy as np 

import pandas as pd
df = pd.read_excel('../input/Canada.xlsx', sheet_name='Canada by Citizenship', skiprows=range(20), skipfooter=2)

df.head()
df.info()
# in pandas axis=0 represents rows (default) and axis=1 represents columns.

df.drop(['AREA','REG','DEV','Type','Coverage'], axis=1, inplace=True)

df.rename(columns={'OdName':'Country', 'AreaName':'Continent', 'RegName':'Region'}, inplace=True)

df.head()
df['Total'] = df.sum(axis=1)

df.isnull().sum().any()
df.describe()
df.set_index('Country', inplace=True)



# optional: to remove the name of the index

df.index.name = None
df.columns = list(map(str, df.columns))



# useful for plotting later on

years = list(map(str, range(1980, 2014)))
import matplotlib.pyplot as plt



# we are using the inline backend

%matplotlib inline 
print(plt.style.available)

plt.style.use(['ggplot']) # for ggplot-like style
import seaborn as sns
haiti = df.loc['Haiti', years] # passing in years 1980 - 2013 to exclude the 'total' column

haiti.head()
haiti.plot()
haiti.index = haiti.index.map(int) # let's change the index values of Haiti to type integer for plotting

haiti = haiti.astype(int)



fig = plt.figure(figsize=(20, 8))

ax = fig.add_subplot(121)

haiti.plot(kind='line',ax=ax)



ax.set_title('Immigration from Haiti Matplotlib')

ax.set_ylabel('Number of immigrants')

ax.set_xlabel('Years')



# ax = fig.add_subplot(122)

# sns.lineplot(x=haiti.index, y=haiti.values, ax=ax)



# ax.set_title('Immigration from Haiti Seaborn')

# ax.set_ylabel('Number of immigrants')

# ax.set_xlabel('Years')



plt.tight_layout()

plt.show() # need this line to show the updates made to the figure
fig = plt.figure(figsize=(20, 8))

ax = fig.add_subplot(121)

haiti.plot(kind='line',ax=ax)



ax.set_title('Immigration from Haiti Matplotlib')

ax.set_ylabel('Number of immigrants')

ax.set_xlabel('Years')

# annotate the 2010 Earthquake. 

# syntax: text(x, y, label)

ax.text(2005, 6000, '2010 Earthquake') # see note below



ax = fig.add_subplot(122)

sns.lineplot(x=haiti.index, y=haiti.values, ax=ax)



ax.set_title('Immigration from Haiti Seaborn')

ax.set_ylabel('Number of immigrants')

ax.set_xlabel('Years')



ax.text(2005, 6000, '2010 Earthquake') 



plt.tight_layout()

plt.show() # need this line to show the updates made to the figure
data = df.loc[['China', 'India'], years]

data.head()
data = data.T

data.head()
data.index = data.index.map(int) # let's change the index values of data to type integer for plotting

data = data.astype(int)



fig = plt.figure(figsize=(20, 8))

ax = fig.add_subplot(121)

data.plot(kind='line', ax=ax)



ax.set_title('Immigrants from China and India Matplotlib')

ax.set_ylabel('Number of immigrants')

ax.set_xlabel('Years')





ax = fig.add_subplot(122)

sns.lineplot(data=data, ax=ax)



ax.set_title('Immigrants from China and India Seaborn')

ax.set_ylabel('Number of immigrants')

ax.set_xlabel('Years')





plt.tight_layout()

plt.show() # need this line to show the updates made to the figure
# Step 1: Get the dataset. We will sort on this column to get our top 5 countries 

# using pandas sort_values() method

df.sort_values(by='Total', ascending=False, axis=0, inplace=True)



# get the top 5 entries

df_top5 = df.head()



# transpose the dataframe

df_top5 = df_top5[years].T 



df_top5.index = df_top5.index.map(int) 



# Step 2: Plot the dataframe.

fig = plt.figure(figsize=(20,7))

ax = fig.add_subplot(121)

df_top5.plot(kind='line', ax=ax) 



ax.set_title('Immigration Trend of Top 5 Countries Matplotlib')

ax.set_ylabel('Number of Immigrants')

ax.set_xlabel('Years')



ax = fig.add_subplot(122)

sns.lineplot(data=df_top5, ax=ax) 



ax.set_title('Immigration Trend of Top 5 Countries Seaborn')

ax.set_ylabel('Number of Immigrants')

ax.set_xlabel('Years')



plt.tight_layout()

plt.show()
# np.histogram returns 2 values

count, bin_edges = np.histogram(df['2013'])



print(count) # frequency count

print(bin_edges) # bin ranges, default = 10 bins
fig = plt.figure(figsize=(20,8))

ax = fig.add_subplot(121)

df['2013'].plot(kind='hist', ax=ax)



ax.set_title('Histogram of Immigration from 195 Countries in 2013 Matplotlib') 

ax.set_ylabel('Number of Countries') 

ax.set_xlabel('Number of Immigrants') 



ax = fig.add_subplot(122)

sns.distplot(df['2013'], kde=False, ax=ax) 



ax.set_title('Histogram of Immigration from 195 Countries in 2013 Seaborn') 

ax.set_ylabel('Number of Countries') 

ax.set_xlabel('Number of Immigrants')



plt.tight_layout()

plt.show()
# 'bin_edges' is a list of bin intervals

count, bin_edges = np.histogram(df['2013'])



fig = plt.figure(figsize=(20,8))

ax = fig.add_subplot(121)

df['2013'].plot(kind='hist', ax=ax)



ax.set_xticks(bin_edges)

ax.set_title('Histogram of Immigration from 195 Countries in 2013 Matplotlib') 

ax.set_ylabel('Number of Countries') 

ax.set_xlabel('Number of Immigrants') 



ax = fig.add_subplot(122)

sns.distplot(df['2013'], kde=False, bins=bin_edges, ax=ax) 



ax.set_xticks(bin_edges)

ax.set_title('Histogram of Immigration from 195 Countries in 2013 Seaborn') 

ax.set_ylabel('Number of Countries') 

ax.set_xlabel('Number of Immigrants')



plt.tight_layout()

plt.show()
df.loc[['Denmark', 'Norway', 'Sweden'], years].T.columns.tolist()
# transpose dataframe

df_t = df.loc[['Denmark', 'Norway', 'Sweden'], years].T



# generate histogram

fig = plt.figure(figsize=(20,8))

ax = fig.add_subplot(121)

df_t.plot(kind='hist', ax=ax)



ax.set_title('Immigration from Denmark, Norway, and Sweden from 1980 - 2013 Matplotlib')

ax.set_ylabel('Number of Years')

ax.set_xlabel('Number of Immigrants') 



ax = fig.add_subplot(122)

sns.distplot(df_t, kde=False, ax=ax, color=['r', 'g', 'b'], label=df_t.columns.tolist()) 



ax.legend()

ax.set_title('Histogram of Immigration from 195 Countries in 2013 Seaborn') 

ax.set_ylabel('Number of Countries') 

ax.set_xlabel('Number of Immigrants')



plt.tight_layout()

plt.show()
# let's get the x-tick values

count, bin_edges = np.histogram(df_t, 15)



# un-stacked histogram

fig = plt.figure(figsize=(20,8))

ax = fig.add_subplot(121)

df_t.plot(kind='hist', bins=15, alpha=0.6, ax=ax, color=['coral', 'darkslateblue', 'mediumseagreen'])



ax.set_xticks(bin_edges)

ax.set_title('Immigration from Denmark, Norway, and Sweden from 1980 - 2013 Matplotlib')

ax.set_ylabel('Number of Years')

ax.set_xlabel('Number of Immigrants') 



ax = fig.add_subplot(122)

sns.distplot(df_t, kde=False, bins=15, hist_kws={'alpha':0.6},

             color=['coral', 'darkslateblue', 'mediumseagreen'], ax=ax,

             label=df_t.columns.tolist()) 



ax.legend()

ax.set_xticks(bin_edges)

ax.set_title('Histogram of Immigration from 195 Countries in 2013 Seaborn') 

ax.set_ylabel('Number of Countries') 

ax.set_xlabel('Number of Immigrants')



plt.tight_layout()

plt.show()
# step 1: get the data

df_iceland = df.loc['Iceland', years]

df_iceland = df_iceland.astype(int)



# step 2: plot data

fig = plt.figure(figsize=(20,8))

ax = fig.add_subplot(121)

df_iceland.plot(kind='bar', ax=ax)



ax.set_xlabel('Year') 

ax.set_ylabel('Number of immigrants') 

ax.set_title('Icelandic immigrants to Canada from 1980 to 2013 Matplotlib') 



ax = fig.add_subplot(122)

sns.barplot(x=df_iceland.index, y=df_iceland.values, palette='deep', ax=ax) 



ax.tick_params(axis='x', rotation=90)

ax.set_xlabel('Year') 

ax.set_ylabel('Number of immigrants') 

ax.set_title('Icelandic immigrants to Canada from 1980 to 2013 Seaborn') 



plt.tight_layout()

plt.show()
fig = plt.figure(figsize=(20,8))

ax = fig.add_subplot(121)

df_iceland.plot(kind='bar', ax=ax)



ax.set_xlabel('Year') 

ax.set_ylabel('Number of immigrants') 

ax.set_title('Icelandic immigrants to Canada from 1980 to 2013 Matplotlib') 



# Annotate arrow

ax.annotate('',                      # s: str. Will leave it blank for no text

             xy=(32, 70),             # place head of the arrow at point (year 2012 , pop 70)

             xytext=(28, 20),         # place base of the arrow at point (year 2008 , pop 20)

             xycoords='data',         # will use the coordinate system of the object being annotated 

             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2))



# Annotate Text

ax.annotate('2008 - 2011 Financial Crisis', # text to display

             xy=(28, 30),                    # start the text at at point (year 2008 , pop 30)

             rotation=77,                  # based on trial and error to match the arrow

             va='bottom',                    # want the text to be vertically 'bottom' aligned

             ha='left',                      # want the text to be horizontally 'left' algned.

            )



ax = fig.add_subplot(122)

sns.barplot(x=df_iceland.index, y=df_iceland.values, palette='deep', ax=ax) 



ax.tick_params(axis='x', rotation=90)

ax.set_xlabel('Year') 

ax.set_ylabel('Number of immigrants') 

ax.set_title('Icelandic immigrants to Canada from 1980 to 2013 Seaborn') 



# Annotate arrow

ax.annotate('',                      # s: str. Will leave it blank for no text

             xy=(32, 70),             # place head of the arrow at point (year 2012 , pop 70)

             xytext=(28, 20),         # place base of the arrow at point (year 2008 , pop 20)

             xycoords='data',         # will use the coordinate system of the object being annotated 

             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2))



# Annotate Text

ax.annotate('2008 - 2011 Financial Crisis', # text to display

             xy=(28, 30),                    # start the text at at point (year 2008 , pop 30)

             rotation=77,                  # based on trial and error to match the arrow

             va='bottom',                    # want the text to be vertically 'bottom' aligned

             ha='left',                      # want the text to be horizontally 'left' algned.

            )



plt.tight_layout()

plt.show()
# sort dataframe on 'Total' column (descending)

df.sort_values(by='Total', ascending=True, inplace=True)



# get top 15 countries

df_top15 = df['Total'].tail(15)



# generate plot

fig = plt.figure(figsize=(15, 20))

ax = fig.add_subplot(211)

df_top15.plot(kind='barh', ax=ax)



ax.set_xlabel('Number of Immigrants')

ax.set_title('Top 15 Countries Contributing to the Immigration to Canada between 1980 - 2013 Matplotlib')



# annotate value labels to each country

for index, value in enumerate(df_top15): 

    label = format(int(value), ',') # format int with commas

    

    # place text at the end of bar (subtracting 57000 from x, and 0.1 from y to make it fit within the bar)

    ax.annotate(label, xy=(value - 51000, index - 0.10), color='white')



ax = fig.add_subplot(212)

sns.barplot(x=df_top15.values, y=df_top15.index, palette='deep', ax=ax)  



ax.set_xlabel('Number of Immigrants')

ax.set_title('Top 15 Countries Contributing to the Immigration to Canada between 1980 - 2013 Seaborn') 



# annotate value labels to each country

for index, value in enumerate(df_top15): 

    label = format(int(value), ',') # format int with commas

    

    # place text at the end of bar (subtracting 47000 from x, and 0.1 from y to make it fit within the bar)

    ax.annotate(label, xy=(value - 47000, index), color='white')



# invert for largest on top 

ax.invert_yaxis()



plt.tight_layout()

plt.show()
# group countries by continents and apply sum() function 

df_continents = df.groupby('Continent', axis=0).sum()



# note: the output of the groupby method is a `groupby' object. 

# we can not use it further until we apply a function (eg .sum())

print(type(df.groupby('Continent', axis=0)))



df_continents.head()
colors_list = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'lightgreen', 'pink']

explode_list = [0.1, 0, 0, 0, 0.1, 0.1] # ratio for each continent with which to offset each wedge.



df_continents['Total'].plot(kind='pie',

                            figsize=(15, 6),

                            autopct='%1.1f%%', 

                            startangle=90,    

                            shadow=True,       

                            labels=None,         # turn off labels on pie chart

                            pctdistance=1.12,    # the ratio between the center of each pie slice and the start of the text generated by autopct 

                            colors=colors_list,  # add custom colors

                            explode=explode_list # 'explode' lowest 3 continents

                            )



# scale the title up by 12% to match pctdistance

plt.title('Immigration to Canada by Continent [1980 - 2013]', y=1.12) 

plt.axis('equal') 



# add legend

plt.legend(labels=df_continents.index, loc='upper left') 



plt.show()
explode_list = [0.1, 0, 0, 0, 0.1, 0.2] # ratio for each continent with which to offset each wedge.



df_continents['2013'].plot(kind='pie',

                            figsize=(15, 6),

                            autopct='%1.1f%%', 

                            startangle=90,    

                            shadow=True,       

                            labels=None,                 # turn off labels on pie chart

                            pctdistance=1.12,            # the ratio between the pie center and start of text label

                            explode=explode_list         # 'explode' lowest 3 continents

                            )



# scale the title up by 12% to match pctdistance

plt.title('Immigration to Canada by Continent in 2013', y=1.12) 

plt.axis('equal') 



# add legend

plt.legend(labels=df_continents.index, loc='upper left') 



# show plot

plt.show()
# to get a dataframe, place extra square brackets around 'Japan'.

df_japan = df.loc[['Japan'], years].T

df_japan.head()
fig = plt.figure(figsize=(20,8))

ax = fig.add_subplot(121)

df_japan.plot(kind='box', ax=ax)



ax.set_title('Box plot of Japanese Immigrants from 1980 - 2013 Matplotlib')

ax.set_ylabel('Number of Immigrants') 



ax = fig.add_subplot(122)

sns.boxplot(data=df_japan, palette='deep', width=0.15, ax=ax) 



ax.set_title('Box plot of Japanese Immigrants from 1980 - 2013 Seaborn')

ax.set_ylabel('Number of Immigrants') 



plt.tight_layout()

plt.show()
df_japan.describe()
data.describe()
fig = plt.figure(figsize=(20,8))

ax = fig.add_subplot(121)

data.plot(kind='box', ax=ax)



ax.set_title('Box plot of Immigrants from China and India 1980 - 2013 Matplotlib')

ax.set_ylabel('Number of Immigrants') 



ax = fig.add_subplot(122)

sns.boxplot(data=data, palette='deep', width=0.15, ax=ax) 



ax.set_title('Box plot of Immigrants from China and India 1980 - 2013 Seaborn')

ax.set_ylabel('Number of Immigrants') 



plt.tight_layout()

plt.show()
# horizontal box plots

fig = plt.figure(figsize=(20,8))

ax = fig.add_subplot(121)

data.plot(kind='box', ax=ax, vert=False)



ax.set_title('Box plot of Immigrants from China and India 1980 - 2013 Matplotlib')

ax.set_xlabel('Number of Immigrants') 



ax = fig.add_subplot(122)

sns.boxplot(data=data, palette='deep', width=0.15, orient='h', ax=ax) 



ax.set_title('Box plot of Immigrants from China and India 1980 - 2013 Seaborn')

ax.set_xlabel('Number of Immigrants') 



plt.tight_layout()

plt.show()
df_top15 = df.sort_values(['Total'], ascending=False, axis=0).head(15)



# create a list of all years in decades 80's, 90's, and 00's

years_80s = list(map(str, range(1980, 1990))) 

years_90s = list(map(str, range(1990, 2000))) 

years_00s = list(map(str, range(2000, 2010))) 



# slice the original dataframe df_can to create a series for each decade

df_80s = df_top15.loc[:, years_80s].sum(axis=1) 

df_90s = df_top15.loc[:, years_90s].sum(axis=1) 

df_00s = df_top15.loc[:, years_00s].sum(axis=1)



# merge the three series into a new data frame

new_df = pd.DataFrame({'1980s': df_80s, '1990s': df_90s, '2000s':df_00s}) 



# display dataframe

new_df.head()
new_df.describe()
fig = plt.figure(figsize=(20,8))

ax = fig.add_subplot(121)

new_df.plot(kind='box', ax=ax)



ax.set_title('Immigration from top 15 countries for decades 80s, 90s and 2000s Matplotlib')

ax.set_ylabel('Number of Immigrants') 



ax = fig.add_subplot(122)

sns.boxplot(data=new_df, palette='deep', width=0.20, ax=ax) 



ax.set_title('Immigration from top 15 countries for decades 80s, 90s and 2000s Seaborn')

ax.set_ylabel('Number of Immigrants') 



plt.tight_layout()

plt.show()
# let's check how many entries fall above the outlier threshold 

new_df[new_df['2000s']> 209611.5]
# we can use the sum() method to get the total population per year

df_tot = pd.DataFrame(df[years].sum(axis=0))



# change the years to type int (useful for regression later on)

df_tot.index = map(int, df_tot.index)



# reset the index to put in back in as a column in the df_tot dataframe

df_tot.reset_index(inplace = True)



# rename columns

df_tot.columns = ['year', 'total']



# view the final dataframe

df_tot.head()
fig = plt.figure(figsize=(20,8))

ax = fig.add_subplot(121)

df_tot.plot(kind='scatter', x='year', y='total', ax=ax)



ax.set_title('Total Immigration to Canada from 1980 - 2013 Matplotlib')

ax.set_xlabel('Year')

ax.set_ylabel('Number of Immigrants')



ax = fig.add_subplot(122)

sns.scatterplot(x=df_tot.year, y=df_tot.total, palette='deep', ax=ax) 



ax.set_title('Total Immigration to Canada from 1980 - 2013 Seaborn')

ax.set_xlabel('Year')

ax.set_ylabel('Number of Immigrants') 



plt.tight_layout()

plt.show()
x = df_tot['year']      # year on x-axis

y = df_tot['total']     # total on y-axis

fit = np.polyfit(x, y, deg=1)

fit
fig = plt.figure(figsize=(20,8))

ax = fig.add_subplot(121)

df_tot.plot(kind='scatter', x='year', y='total', ax=ax)



ax.set_title('Total Immigration to Canada from 1980 - 2013 Matplotlib')

ax.set_xlabel('Year')

ax.set_ylabel('Number of Immigrants')



# plot line of best fit

ax.plot(x, fit[0] * x + fit[1], color='red') # recall that x is the Years



ax = fig.add_subplot(122)

sns.regplot(x=df_tot.year, y=df_tot.total, ax=ax) 



ax.set_title('Total Immigration to Canada from 1980 - 2013 Seaborn')

ax.set_xlabel('Year')

ax.set_ylabel('Number of Immigrants') 



plt.tight_layout()

plt.show()
df_countries = df.loc[['Denmark', 'Norway', 'Sweden'], years].T



# create df_total by summing across three countries for each year

df_total = pd.DataFrame(df_countries.sum(axis=1))



# reset index in place

df_total.reset_index(inplace=True)



# rename columns

df_total.columns = ['year', 'total']



# change column year from string to int to create scatter plot

df_total['year'] = df_total['year'].astype(int)



# show resulting dataframe

df_total.head()



# generate scatter plot

fig = plt.figure(figsize=(20,8))

ax = fig.add_subplot(121)

df_total.plot(kind='scatter', x='year', y='total', ax=ax)



ax.set_title('Immigration from Denmark, Norway, and Sweden to Canada from 1980 - 2013 Matplotlib')

ax.set_xlabel('Year')

ax.set_ylabel('Number of Immigrants')



ax = fig.add_subplot(122)

sns.scatterplot(x=df_total.year, y=df_total.total, palette='deep', ax=ax) 



ax.set_title('Immigration from Denmark, Norway, and Sweden to Canada from 1980 - 2013 Seaborn')

ax.set_xlabel('Year')

ax.set_ylabel('Number of Immigrants')



plt.tight_layout()

plt.show()
df_t = df[years].T



# cast the Years (the index) to type int

df_t.index = map(int, df_t.index)



# let's label the index. This will automatically be the column name when we reset the index

df_t.index.name = 'Year'



# reset index to bring the Year in as a column

df_t.reset_index(inplace=True)



# view the changes

df_t.head()
from sklearn.preprocessing import MinMaxScaler



scale_bra = MinMaxScaler()

scale_arg = MinMaxScaler()

norm_brazil = scale_bra.fit_transform(df_t['Brazil'].values.reshape(-1, 1))

norm_arg = scale_arg.fit_transform(df_t['Argentina'].values.reshape(-1, 1))
df_t['weight_arg'] = norm_arg

df_t['weight_brazil'] = norm_brazil



fig = plt.figure(figsize=(20,9))

ax = fig.add_subplot(121)



# Brazil

df_t.plot(kind='scatter', x='Year', y='Brazil',

            alpha=0.5,                  # transparency

            s=norm_brazil * 2000 + 10,  # pass in weights 

            ax=ax)



# Argentina

df_t.plot(kind='scatter', x='Year', y='Argentina',

            alpha=0.5,

            color="blue",

            s=norm_arg * 2000 + 10,

            ax=ax)



ax.set_ylabel('Number of Immigrants')

ax.set_title('Immigration from Brazil and Argentina from 1980 - 2013 Matplotlib')

ax.legend(['Brazil', 'Argentina'], loc='upper left', fontsize='x-large')



ax = fig.add_subplot(122)

sns.scatterplot(x=df_t['Year'], y=df_t['Argentina'], palette='deep', 

                size=df_t['weight_arg'], sizes=(10, 2000), alpha=0.5, ax=ax)

sns.scatterplot(x=df_t['Year'], y=df_t['Brazil'], palette='deep', 

                size=df_t['weight_brazil'], sizes=(10, 2000), alpha=0.5, ax=ax) 



ax.set_ylabel('Number of Immigrants')

ax.set_title('Immigration from Brazil and Argentina from 1980 - 2013 Seaborn')

ax.legend(['Brazil', 'Argentina'], loc='upper left', fontsize='x-large')



plt.tight_layout()

plt.show()