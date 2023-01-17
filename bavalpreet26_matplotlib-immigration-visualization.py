# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from PIL import Image # converting images into arrays

%matplotlib inline



import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches # needed for waffle Charts



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_can = pd.read_excel('/kaggle/input/immigration-to-canada-ibm-dataset/Canada.xlsx',

                       sheet_name='Canada by Citizenship',

                       skiprows=range(20),

                       skipfooter=2)



print ('Data read into a pandas dataframe!')
df_can.head()

# tip: You can specify the number of rows you'd like to see as follows: df_can.head(10) 
df_can.tail()
df_can.info()
df_can.columns.values 
df_can.index.values
print(type(df_can.columns))

print(type(df_can.index))
df_can.columns.tolist()

df_can.index.tolist()



print (type(df_can.columns.tolist()))

print (type(df_can.index.tolist()))
# size of dataframe (rows, columns)

df_can.shape    
# in pandas axis=0 represents rows (default) and axis=1 represents columns.

df_can.drop(['AREA','REG','DEV','Type','Coverage'], axis=1, inplace=True)

df_can.head(2)
df_can.rename(columns={'OdName':'Country', 'AreaName':'Continent', 'RegName':'Region'}, inplace=True)

df_can.columns
df_can['Total'] = df_can.sum(axis=1)
df_can.isnull().sum()
df_can.describe()
df_can.Country  # returns a series
df_can[['Country', 1980, 1981, 1982, 1983, 1984, 1985]] # returns a dataframe

# notice that 'Country' is string, and the years are integers. 

# for the sake of consistency, we will convert all column names to string later on.
df_can.set_index('Country', inplace=True)

# tip: The opposite of set is reset. So to reset the index, we can use df_can.reset_index()
df_can.head(3)
# optional: to remove the name of the index

df_can.index.name = None

df_can.head(3)
# 1. the full row data (all columns)

print(df_can.loc['Japan'])



# alternate methods

print(df_can.iloc[87])

print(df_can[df_can.index == 'Japan'].T.squeeze())
# 2. for year 2013

print(df_can.loc['Japan', 2013])



# alternate method

print(df_can.iloc[87, 36]) # year 2013 is the last column, with a positional index of 36
# 3. for years 1980 to 1985

print(df_can.loc['Japan', [1980, 1981, 1982, 1983, 1984, 1984]])

print(df_can.iloc[87, [3, 4, 5, 6, 7, 8]])
df_can.columns = list(map(str, df_can.columns))

# [print (type(x)) for x in df_can.columns.values] #<-- uncomment to check type of column headers
# useful for plotting later on

years = list(map(str, range(1980, 2014)))

years
# 1. create the condition boolean series

condition = df_can['Continent'] == 'Asia'

print(condition)
# 2. pass this condition into the dataFrame

df_can[condition]
# we can pass mutliple criteria in the same line. 

# let's filter for AreaNAme = Asia and RegName = Southern Asia



df_can[(df_can['Continent']=='Asia') & (df_can['Region']=='Southern Asia')]



# note: When using 'and' and 'or' operators, pandas requires we use '&' and '|' instead of 'and' and 'or'

# don't forget to enclose the two conditions in parentheses
print('data dimensions:', df_can.shape)

print(df_can.columns)

df_can.head(2)
# we are using the inline backend

%matplotlib inline 



import matplotlib as mpl

import matplotlib.pyplot as plt
print ('Matplotlib version: ', mpl.__version__)
print(plt.style.available)

mpl.style.use(['ggplot']) 
haiti = df_can.loc['Haiti', years] # passing in years 1980 - 2013 to exclude the 'total' column

haiti.head()

haiti.plot()
haiti.index = haiti.index.map(int) # let's change the index values of Haiti to type integer for plotting

haiti.plot(kind='line')



plt.title('Immigration from Haiti')

plt.ylabel('Number of immigrants')

plt.xlabel('Years')



plt.show() # need this line to show the updates made to the figure
haiti.plot(kind='line')



plt.title('Immigration from Haiti')

plt.ylabel('Number of Immigrants')

plt.xlabel('Years')



# annotate the 2010 Earthquake. 

# syntax: plt.text(x, y, label)

plt.text(2000, 6000, '2010 Earthquake') # see note below



plt.show() 
df_can.head(80)


df = df_can.loc[['India','China'],years]

df = df.transpose()

### type your answer here

df.index = df.index.map(int)

df.plot(kind='line')

plt.title('Immigration from China and India')

plt.ylabel('Number of Immigrations')

plt.xlabel('Years')

plt.show()
### type your answer here

df_can.sort_values(by='Total', ascending=False, axis=0, inplace=True)

df_top5 = df_can.head(5)

df_top5 = df_top5[years].transpose()

 

# print(df_top5)

 

df_top5.index = df_top5.index.map(int)

df_top5.plot(kind='line', figsize=(14,8))

 

plt.title('Immigration Trend of Top 5 Countries')

plt.ylabel('Number of Immigration')

plt.xlabel('Years')

plt.show()
df_can.sort_values(['Total'], ascending = False, axis = 0, inplace = True)



#get the top5 entries

df_top5 = df_can.head()



#transpose the dataframe

df_top5 = df_top5[years].transpose()



df_top5.head()
df_top5.index = df_top5.index.map(int)#change the index value to type integer

df_top5.plot(kind = 'area',stacked=False, figsize = (20,10),#pass a tuple (x,y) size

            )

plt.title('Immigration Trend of Top5 Countries')

plt.ylabel('Number of Immigrants')

plt.xlabel('Years')

plt.show()
df_top5.plot(kind='area', 

             alpha=0.25, # 0-1, default value a= 0.5

             stacked=False,

             figsize=(20, 10),

            )



plt.title('Immigration Trend of Top 5 Countries')

plt.ylabel('Number of Immigrants')

plt.xlabel('Years')



plt.show()
# option 2: preferred option with more flexibility

ax = df_top5.plot(kind='area', alpha=0.35, figsize=(20, 10))



ax.set_title('Immigration Trend of Top 5 Countries')

ax.set_ylabel('Number of Immigrants')

ax.set_xlabel('Years')
df_can.sort_values(['Total'], ascending = True, axis = 0, inplace = True)



#get the top5 entries

df_least5 = df_can.head()



#transpose the dataframe

df_least5 = df_least5[years].transpose()



df_least5.head()


df_least5.index = df_least5.index.map(int)#change the index value to type integer

df_least5.plot(kind='area', 

             alpha=0.45, # 0-1, default value a= 0.5

             stacked=True,

             figsize=(20, 10),

            )



plt.title('Immigration Trend of 5 Countries contributed the least to Canada')

plt.ylabel('Number of Immigrants')

plt.xlabel('Years')



plt.show()
# option 2: preferred option with more flexibility

ax = df_least5.plot(kind='area',stacked=False, alpha=0.55, figsize=(20, 10))



ax.set_title('Immigration Trend of 5 Countries contributed the least to Canada')

ax.set_ylabel('Number of Immigrants')

ax.set_xlabel('Years')
# let's quickly view the 2013 data

df_can.sort_values(['Total'], ascending = False, axis = 0, inplace = True)



df_can['2013'].head()
# np.histogram returns 2 values

count, bin_edges = np.histogram(df_can['2013'])



print(count) # frequency count

print(bin_edges) # bin ranges, default = 10 bins
df_can['2013'].plot(kind='hist', figsize=(8, 5))



plt.title('Histogram of Immigration from 195 Countries in 2013') # add a title to the histogram

plt.ylabel('Number of Countries') # add y-label

plt.xlabel('Number of Immigrants') # add x-label



plt.show()
# 'bin_edges' is a list of bin intervals

count, bin_edges = np.histogram(df_can['2013'])



df_can['2013'].plot(kind='hist', figsize=(8, 5), xticks=bin_edges)



plt.title('Histogram of Immigration from 195 countries in 2013') # add a title to the histogram

plt.ylabel('Number of Countries') # add y-label

plt.xlabel('Number of Immigrants') # add x-label



plt.show()
# let's quickly view the dataset 

df_can.loc[['Denmark', 'Norway', 'Sweden'], years]
# generate histogram

df_can.loc[['Denmark', 'Norway', 'Sweden'], years].plot.hist()
# transpose dataframe

df_t = df_can.loc[['Denmark', 'Norway', 'Sweden'], years].transpose()

df_t.head()
# generate histogram

df_t.plot(kind='hist', figsize=(10, 6))



plt.title('Histogram of Immigration from Denmark, Norway, and Sweden from 1980 - 2013')

plt.ylabel('Number of Years')

plt.xlabel('Number of Immigrants')



plt.show()
# let's get the x-tick values

count, bin_edges = np.histogram(df_t, 15)



# un-stacked histogram

df_t.plot(kind ='hist', 

          figsize=(10, 6),

          bins=15,

          alpha=0.6,

          xticks=bin_edges,

          color=['coral', 'darkslateblue', 'mediumseagreen']

         )



plt.title('Histogram of Immigration from Denmark, Norway, and Sweden from 1980 - 2013')

plt.ylabel('Number of Years')

plt.xlabel('Number of Immigrants')



plt.show()
count, bin_edges = np.histogram(df_t, 15)

xmin = bin_edges[0] - 10   #  first bin value is 31.0, adding buffer of 10 for aesthetic purposes 

xmax = bin_edges[-1] + 10  #  last bin value is 308.0, adding buffer of 10 for aesthetic purposes



# stacked Histogram

df_t.plot(kind='hist',

          figsize=(10, 6), 

          bins=15,

          xticks=bin_edges,

          color=['coral', 'darkslateblue', 'mediumseagreen'],

          stacked=True,

          xlim=(xmin, xmax)

         )



plt.title('Histogram of Immigration from Denmark, Norway, and Sweden from 1980 - 2013')

plt.ylabel('Number of Years')

plt.xlabel('Number of Immigrants') 



plt.show()
df_cof = df_can.loc[['Greece', 'Albania', 'Bulgaria'], years]

df_cof = df_cof.transpose()

count, bin_edges = np.histogram(df_cof, 15)

df_cof.plot(kind ='hist',

            figsize=(10, 6),

            bins=15,

            alpha=0.35,

            xticks=bin_edges,

            color=['coral', 'darkslateblue', 'mediumseagreen']

            )

plt.title('Histogram of Immigration from Greece, Albania, and Bulgaria from 1980 - 2013')

plt.ylabel('Number of Years')

plt.xlabel('Number of Immigrants')

plt.show()
# step 1: get the data

df_iceland = df_can.loc['Iceland', years]

df_iceland.head()
# step 2: plot data

df_iceland.plot(kind='bar', figsize=(10, 6))



plt.xlabel('Year') # add to x-label to the plot

plt.ylabel('Number of immigrants') # add y-label to the plot

plt.title('Icelandic immigrants to Canada from 1980 to 2013') # add title to the plot



plt.show()
df_iceland.plot(kind='bar', figsize=(10, 6), rot=90) # rotate the bars by 90 degrees



plt.xlabel('Year')

plt.ylabel('Number of Immigrants')

plt.title('Icelandic Immigrants to Canada from 1980 to 2013')



# Annotate arrow

plt.annotate('',                      # s: str. Will leave it blank for no text

             xy=(32, 70),             # place head of the arrow at point (year 2012 , pop 70)

             xytext=(28, 20),         # place base of the arrow at point (year 2008 , pop 20)

             xycoords='data',         # will use the coordinate system of the object being annotated 

             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2)

            )



plt.show()
df_iceland.plot(kind='bar', figsize=(10, 6), rot=90) 



plt.xlabel('Year')

plt.ylabel('Number of Immigrants')

plt.title('Icelandic Immigrants to Canada from 1980 to 2013')



# Annotate arrow

plt.annotate('',                      # s: str. will leave it blank for no text

             xy=(32, 70),             # place head of the arrow at point (year 2012 , pop 70)

             xytext=(28, 20),         # place base of the arrow at point (year 2008 , pop 20)

             xycoords='data',         # will use the coordinate system of the object being annotated 

             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2)

            )



# Annotate Text

plt.annotate('2008 - 2011 Financial Crisis', # text to display

             xy=(28, 30),                    # start the text at at point (year 2008 , pop 30)

             rotation=72.5,                  # based on trial and error to match the arrow

             va='bottom',                    # want the text to be vertically 'bottom' aligned

             ha='left',                      # want the text to be horizontally 'left' algned.

            )



plt.show()
df_can.sort_values(by='Total', ascending=True, inplace=True)



# get top 15 countries

df_top15 = df_can['Total'].tail(15)

df_top15
 # generate plot

df_top15.plot(kind='barh', figsize=(12, 12), color='steelblue')

plt.xlabel('Number of Immigrants')

plt.title('Top 15 Conuntries Contributing to the Immigration to Canada between 1980 - 2013')



# annotate value labels to each country

for index, value in enumerate(df_top15): 

    label = format(int(value), ',') # format int with commas

    

    # place text at the end of bar (subtracting 47000 from x, and 0.1 from y to make it fit within the bar)

    plt.annotate(label, xy=(value - 47000, index - 0.10), color='white')

    

plt.show()    
# group countries by continents and apply sum() function 

df_continents = df_can.groupby('Continent', axis=0).sum()



# note: the output of the groupby method is a `groupby' object. 

# we can not use it further until we apply a function (eg .sum())

print(type(df_can.groupby('Continent', axis=0)))



df_continents.head()
# autopct create %, start angle represent starting point

df_continents['Total'].plot(kind='pie',

                            figsize=(5, 6),

                            autopct='%1.1f%%', # add in percentages

                            startangle=90,     # start angle 90° (Africa)

                            shadow=True,       # add shadow      

                            )



plt.title('Immigration to Canada by Continent [1980 - 2013]')

plt.axis('equal') # Sets the pie chart to look like a circle.



plt.show()
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



plt.title('Immigration to Canada by Continent in 2013', y=1.12) 

plt.axis('equal') 



plt.legend(labels=df_continents.index, loc='upper left') 



plt.show()
# to get a dataframe, place extra square brackets around 'Japan'.

df_japan = df_can.loc[['Japan'], years].transpose()

df_japan.head()
df_japan.plot(kind='box', figsize=(8, 6))



plt.title('Box plot of Japanese Immigrants from 1980 - 2013')

plt.ylabel('Number of Immigrants')



plt.show()
df_japan.describe()
# to get a dataframe, place extra square brackets around 'Japan'.

df_CI = df_can.loc[['China','India'], years].transpose()

df_CI.head()
df_CI.plot(kind='box', figsize=(8, 6))



plt.title('Box plots of Immigrants from China and India (1980 - 2013)')

plt.ylabel('Number of Immigrants')



plt.show()
df_CI.describe()
# horizontal box plots

df_CI.plot(kind='box', figsize=(10, 7), color='blue', vert=False)



plt.title('Box plots of Immigrants from China and India (1980 - 2013)')

plt.xlabel('Number of Immigrants')



plt.show()
fig = plt.figure() # create figure



ax0 = fig.add_subplot(1, 2, 1) # add subplot 1 (1 row, 2 columns, first plot)

ax1 = fig.add_subplot(1, 2, 2) # add subplot 2 (1 row, 2 columns, second plot). See tip below**



# Subplot 1: Box plot

df_CI.plot(kind='box', color='blue', vert=False, figsize=(20, 6), ax=ax0) # add to subplot 1

ax0.set_title('Box Plots of Immigrants from China and India (1980 - 2013)')

ax0.set_xlabel('Number of Immigrants')

ax0.set_ylabel('Countries')



# Subplot 2: Line plot

df_CI.plot(kind='line', figsize=(20, 6), ax=ax1) # add to subplot 2

ax1.set_title ('Line Plots of Immigrants from China and India (1980 - 2013)')

ax1.set_ylabel('Number of Immigrants')

ax1.set_xlabel('Years')



plt.show()
df_top15 = df_can.sort_values(['Total'], ascending=False, axis=0).head(15)

df_top15
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
new_df.plot(kind='box', figsize=(10, 6))



plt.title('Immigration from top 15 countries for decades 80s, 90s and 2000s')



plt.show()
# let's check how many entries fall above the outlier threshold 

new_df[new_df['2000s']> 209611.5]
# we can use the sum() method to get the total population per year

df_tot = pd.DataFrame(df_can[years].sum(axis=0))



# change the years to type int (useful for regression later on)

df_tot.index = map(int, df_tot.index)



# reset the index to put in back in as a column in the df_tot dataframe

df_tot.reset_index(inplace = True)



# rename columns

df_tot.columns = ['year', 'total']



# view the final dataframe

df_tot.head()
df_tot.plot(kind='scatter', x='year', y='total', figsize=(10, 6), color='darkblue')



plt.title('Total Immigration to Canada from 1980 - 2013')

plt.xlabel('Year')

plt.ylabel('Number of Immigrants')



plt.show()
x = df_tot['year']      # year on x-axis

y = df_tot['total']     # total on y-axis

fit = np.polyfit(x, y, deg=1)



fit
df_tot.plot(kind='scatter', x='year', y='total', figsize=(10, 6), color='darkblue')



plt.title('Total Immigration to Canada from 1980 - 2013')

plt.xlabel('Year')

plt.ylabel('Number of Immigrants')



# plot line of best fit

plt.plot(x, fit[0] * x + fit[1], color='red') # recall that x is the Years

plt.annotate('y={0:.0f} x + {1:.0f}'.format(fit[0], fit[1]), xy=(2000, 150000))



plt.show()



# print out the line of best fit

'No. Immigrants = {0:.0f} * Year + {1:.0f}'.format(fit[0], fit[1]) 
df_countries = df_can.loc[['Denmark', 'Norway', 'Sweden'],years].transpose()



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

df_total.plot(kind='scatter', x='year', y='total', figsize=(10, 6), color='darkblue')



 # add title and label to axes

plt.title('Immigration from Denmark, Norway, and Sweden to Canada from 1980 - 2013')

plt.xlabel('Year')

plt.ylabel('Number of Immigrants')



 # show plot

plt.show()
df_can_t = df_can[years].transpose() # transposed dataframe



# cast the Years (the index) to type int

df_can_t.index = map(int, df_can_t.index)



# let's label the index. This will automatically be the column name when we reset the index

df_can_t.index.name = 'Year'



# reset index to bring the Year in as a column

df_can_t.reset_index(inplace=True)



# view the changes

df_can_t.head()
# normalize Brazil data

norm_brazil = (df_can_t['Brazil'] - df_can_t['Brazil'].min()) / (df_can_t['Brazil'].max() - df_can_t['Brazil'].min())



# normalize Argentina data

norm_argentina = (df_can_t['Argentina'] - df_can_t['Argentina'].min()) / (df_can_t['Argentina'].max() - df_can_t['Argentina'].min())
# Brazil

ax0 = df_can_t.plot(kind='scatter',

                    x='Year',

                    y='Brazil',

                    figsize=(14, 8),

                    alpha=0.5,                  # transparency

                    color='green',

                    s=norm_brazil * 2000 + 10,  # pass in weights 

                    xlim=(1975, 2015)

                   )



# Argentina

ax1 = df_can_t.plot(kind='scatter',

                    x='Year',

                    y='Argentina',

                    alpha=0.5,

                    color="blue",

                    s=norm_argentina * 2000 + 10,

                    ax = ax0

                   )



ax0.set_ylabel('Number of Immigrants')

ax0.set_title('Immigration from Brazil and Argentina from 1980 - 2013')

ax0.legend(['Brazil', 'Argentina'], loc='upper left', fontsize='x-large')
 # normalize China data

norm_china = (df_can_t['China'] - df_can_t['China'].min()) / (df_can_t['China'].max() - df_can_t['China'].min())



# normalize India data

norm_india = (df_can_t['India'] - df_can_t['India'].min()) / (df_can_t['India'].max() - df_can_t['India'].min())
 # China

ax0 = df_can_t.plot(kind='scatter',

                    x='Year',

                    y='China',

                    figsize=(14, 8),

                    alpha=0.5,                  # transparency

                    color='green',

                    s=norm_china * 2000 + 10,  # pass in weights 

                    xlim=(1975, 2015)

                   )



# India

ax1 = df_can_t.plot(kind='scatter',

                    x='Year',

                    y='India',

                    alpha=0.5,

                    color="blue",

                    s=norm_india * 2000 + 10,

                    ax = ax0

                   )



ax0.set_ylabel('Number of Immigrants')

ax0.set_title('Immigration from China and India from 1980 - 2013')

ax0.legend(['China', 'India'], loc='upper left', fontsize='x-large')
# let's create a new dataframe for these three countries 

df_dsn = df_can.loc[['Denmark', 'Norway', 'Sweden'], :]



# let's take a look at our dataframe

df_dsn
# compute the proportion of each category with respect to the total

total_values = sum(df_dsn['Total'])

category_proportions = [(float(value) / total_values) for value in df_dsn['Total']]



# print out proportions

for i, proportion in enumerate(category_proportions):

    print (df_dsn.index.values[i] + ': ' + str(proportion))
width = 40 # width of chart

height = 10 # height of chart



total_num_tiles = width * height # total number of tiles



print ('Total number of tiles is ', total_num_tiles)
# compute the number of tiles for each catagory

tiles_per_category = [round(proportion * total_num_tiles) for proportion in category_proportions]



# print out number of tiles per category

for i, tiles in enumerate(tiles_per_category):

    print (df_dsn.index.values[i] + ': ' + str(tiles))
# initialize the waffle chart as an empty matrix

waffle_chart = np.zeros((height, width))



# define indices to loop through waffle chart

category_index = 0

tile_index = 0



# populate the waffle chart

for col in range(width):

    for row in range(height):

        tile_index += 1



        # if the number of tiles populated for the current category is equal to its corresponding allocated tiles...

        if tile_index > sum(tiles_per_category[0:category_index]):

            # ...proceed to the next category

            category_index += 1       

            

        # set the class value to an integer, which increases with class

        waffle_chart[row, col] = category_index

        

print ('Waffle chart populated!')
waffle_chart
# instantiate a new figure object

fig = plt.figure()



# use matshow to display the waffle chart

colormap = plt.cm.coolwarm

plt.matshow(waffle_chart, cmap=colormap)

plt.colorbar()
# instantiate a new figure object

fig = plt.figure()



# use matshow to display the waffle chart

colormap = plt.cm.coolwarm

plt.matshow(waffle_chart, cmap=colormap)

plt.colorbar()



# get the axis

ax = plt.gca()



# set minor ticks

ax.set_xticks(np.arange(-.5, (width), 1), minor=True)

ax.set_yticks(np.arange(-.5, (height), 1), minor=True)

    

# add gridlines based on minor ticks

ax.grid(which='minor', color='w', linestyle='-', linewidth=2)



plt.xticks([])

plt.yticks([])
# instantiate a new figure object

fig = plt.figure()



# use matshow to display the waffle chart

colormap = plt.cm.coolwarm

plt.matshow(waffle_chart, cmap=colormap)

plt.colorbar()



# get the axis

ax = plt.gca()



# set minor ticks

ax.set_xticks(np.arange(-.5, (width), 1), minor=True)

ax.set_yticks(np.arange(-.5, (height), 1), minor=True)

    

# add gridlines based on minor ticks

ax.grid(which='minor', color='w', linestyle='-', linewidth=2)



plt.xticks([])

plt.yticks([])



# compute cumulative sum of individual categories to match color schemes between chart and legend

values_cumsum = np.cumsum(df_dsn['Total'])

total_values = values_cumsum[len(values_cumsum) - 1]



# create legend

legend_handles = []

for i, category in enumerate(df_dsn.index.values):

    label_str = category + ' (' + str(df_dsn['Total'][i]) + ')'

    color_val = colormap(float(values_cumsum[i])/total_values)

    legend_handles.append(mpatches.Patch(color=color_val, label=label_str))



# add legend to chart

plt.legend(handles=legend_handles,

           loc='lower center', 

           ncol=len(df_dsn.index.values),

           bbox_to_anchor=(0., -0.2, 0.95, .1)

          )
def create_waffle_chart(categories, values, height, width, colormap, value_sign=''):



    # compute the proportion of each category with respect to the total

    total_values = sum(values)

    category_proportions = [(float(value) / total_values) for value in values]



    # compute the total number of tiles

    total_num_tiles = width * height # total number of tiles

    print ('Total number of tiles is', total_num_tiles)

    

    # compute the number of tiles for each catagory

    tiles_per_category = [round(proportion * total_num_tiles) for proportion in category_proportions]



    # print out number of tiles per category

    for i, tiles in enumerate(tiles_per_category):

        print (df_dsn.index.values[i] + ': ' + str(tiles))

    

    # initialize the waffle chart as an empty matrix

    waffle_chart = np.zeros((height, width))



    # define indices to loop through waffle chart

    category_index = 0

    tile_index = 0



    # populate the waffle chart

    for col in range(width):

        for row in range(height):

            tile_index += 1



            # if the number of tiles populated for the current category 

            # is equal to its corresponding allocated tiles...

            if tile_index > sum(tiles_per_category[0:category_index]):

                # ...proceed to the next category

                category_index += 1       

            

            # set the class value to an integer, which increases with class

            waffle_chart[row, col] = category_index

    

    # instantiate a new figure object

    fig = plt.figure()



    # use matshow to display the waffle chart

    colormap = plt.cm.coolwarm

    plt.matshow(waffle_chart, cmap=colormap)

    plt.colorbar()



    # get the axis

    ax = plt.gca()



    # set minor ticks

    ax.set_xticks(np.arange(-.5, (width), 1), minor=True)

    ax.set_yticks(np.arange(-.5, (height), 1), minor=True)

    

    # add dridlines based on minor ticks

    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)



    plt.xticks([])

    plt.yticks([])



    # compute cumulative sum of individual categories to match color schemes between chart and legend

    values_cumsum = np.cumsum(values)

    total_values = values_cumsum[len(values_cumsum) - 1]



    # create legend

    legend_handles = []

    for i, category in enumerate(categories):

        if value_sign == '%':

            label_str = category + ' (' + str(values[i]) + value_sign + ')'

        else:

            label_str = category + ' (' + value_sign + str(values[i]) + ')'

            

        color_val = colormap(float(values_cumsum[i])/total_values)

        legend_handles.append(mpatches.Patch(color=color_val, label=label_str))



    # add legend to chart

    plt.legend(

        handles=legend_handles,

        loc='lower center', 

        ncol=len(categories),

        bbox_to_anchor=(0., -0.2, 0.95, .1)

    )
width = 40 # width of chart

height = 10 # height of chart



categories = df_dsn.index.values # categories

values = df_dsn['Total'] # correponding values of categories



colormap = plt.cm.coolwarm # color map class
create_waffle_chart(categories, values, height, width, colormap)
# # install wordcloud

# !conda install -c conda-forge wordcloud==1.4.1 --yes



# import package and its set of stopwords

from wordcloud import WordCloud, STOPWORDS



print ('Wordcloud is installed and imported!')
STOPWORDS =  ['were', 'the', 'amp', 'dont', 'got', 'know', 'gon', 'na', 'wan', 'like', 'im', 'hers', 'why', 'over', "'d",'our', 'these', 'nevertheless', 'its', 'them', 'empty', 'how', 'whereas', 'whether', 'fifteen', 'about', 'four', 'give', 'otherwise', 'move', 'do', 'say', '‘ve', 'hence', 'n‘t', 'between', 'bottom', 'some', 'against', 'whole', 'i', 'into', 'they', 'already', 'she', 'either', 'an', 'both', 'him', 'due', 'using', 'five', 'across', 'front', 'in', 'off', 'only', 'really', 'twelve', 'twenty', 'show', 'whereupon', '‘m', 'n’t', 'himself', '’m', 'from', 'often', 'three', 'various', 'thereupon', 'should', 'put', 'take', 'who', 'above', 'their', 'been', 'towards', 'however', "n't", 'her', 'go', 'thereby', 'just', 'yourselves', 'become', 'thru', 'while', 'nowhere', 'neither', 'anyway', 'because', 'ca', 'which', 'moreover', 'forty', 'besides', 'us', 'more', 'third', 'wherein', 'whoever', 'used', 'every', 'whose', 'onto', 'your', 'hereafter', 'itself', 'sometimes', 'name', 'too', 'own', 'somewhere', 'there', 'we', 'you', '’ve', 'ourselves', 'sixty', 'would', 'first', 'must', 'whereafter', 'wherever', 'his', 'around', 'has', 'yours', 'became', 'doing','the', 'below', 'then', 'everyone', 'else', 'any', 'latterly', 'noone', 'part', 'might', "'ve", 'becoming', 'same', 'top', 'yourself', 'he', 'each', 'anyone', 'my', 'seeming', 'six', 'the', 'during', 'afterwards', 'throughout', 'formerly', 'seem', 'therefore', 'another', 'keep', 'without', 'being', 'can', 'had', 'per', "'s", 'other', 'side', '’s', 'also', 'herself', '’ll', 'eight', 'what', 'please', 'a', 'therein', 'back', 'me', 'never', 'not', 'does', 'enough', 'meanwhile', 'toward', 'even', 'get', 'and', 'it', 'perhaps', 'this', 'regarding', 'somehow', 'cannot', 'anyhow', 'through', 'whenever', 'thereafter', 'rather', 'by', 'still', 'where', 'than', 'made', 'of', 'will', 'within', 'are', 'amongst', 'although', 'former', 'full', 'nobody', 'was', 'to', 'is', 'at', 'hundred', 'all', 'on', 'such', 'after', 'almost', 'most', 'no', 'our', 'see', 'thus', 'upon', "'ll", 'whence', 'make', '‘s', 'could', 'quite', 'or', 'beyond', 'thence', 'mostly', 'though', 'alone', 'for', 'under', 'seemed', 'until', 'much', 'nine', 'least', 'that', 'nor', 'further', 'themselves', 'whatever', 'whom', 'anywhere', 'myself', 'eleven', 'none', 'with', 'as', 'have', '‘ll', "'m", 'up', 'if', 'several', 'whereby', 'now', 'always', 'amount', 'done', 'hereupon', 'others', 'may', 'one', 'everything', 'so', 'hereby', 'anything', 'fifty', 'last', 'am', 'beforehand', 'few', 'ever', 'together', 'unless', 'ten', 'behind', 'when', 'those', 'mine', 'everywhere', 'be', 'less', 'nothing', 'something', 'very', "'re", 'here', '‘re', 'since', 'seems', 'down', 'did', 'before', 'serious', '‘d', '’d', 'many', 'call', 'along', 'once', 'herein', 'out', 'namely', 'someone', 'becomes', 'whither', 're', 'two', 'but', 'again', 'elsewhere', 'well', 'next', 'sometime', 'indeed', 'ours', 'yet', '’re', 'via', 'latter', 'except', 'among', 'beside']

stopwords = set(STOPWORDS)
# instantiate a word cloud object

text = " ".join(str(each) for each in df_can.Region)

alice_wc = WordCloud(

    background_color='white',

    max_words=2000,

    stopwords=stopwords

)



# generate the word cloud

alice_wc.generate(text)
# display the word cloud

plt.imshow(alice_wc, interpolation='bilinear')

plt.axis('off')

plt.show()
fig = plt.figure()

fig.set_figwidth(14) # set width

fig.set_figheight(18) # set height



# display the cloud

plt.imshow(alice_wc, interpolation='bilinear')

plt.axis('off')

plt.show()
import seaborn as sns

# we can use the sum() method to get the total population per year

df_tot = pd.DataFrame(df_can[years].sum(axis=0))



# change the years to type float (useful for regression later on)

df_tot.index = map(float, df_tot.index)



# reset the index to put in back in as a column in the df_tot dataframe

df_tot.reset_index(inplace=True)



# rename columns

df_tot.columns = ['year', 'total']



# view the final dataframe

df_tot.head()
import seaborn as sns

ax = sns.regplot(x='year', y='total', data=df_tot)
import seaborn as sns

ax = sns.regplot(x='year', y='total', data=df_tot, color='green')
import seaborn as sns

ax = sns.regplot(x='year', y='total', data=df_tot, color='green', marker='+')
plt.figure(figsize=(15, 10))

ax = sns.regplot(x='year', y='total', data=df_tot, color='green', marker='+')
plt.figure(figsize=(15, 10))

ax = sns.regplot(x='year', y='total', data=df_tot, color='green', marker='+', scatter_kws={'s': 200})



ax.set(xlabel='Year', ylabel='Total Immigration') # add x- and y-labels

ax.set_title('Total Immigration to Canada from 1980 - 2013') # add title
plt.figure(figsize=(15, 10))



sns.set(font_scale=1.5)



ax = sns.regplot(x='year', y='total', data=df_tot, color='green', marker='+', scatter_kws={'s': 200})

ax.set(xlabel='Year', ylabel='Total Immigration')

ax.set_title('Total Immigration to Canada from 1980 - 2013')
plt.figure(figsize=(15, 10))



sns.set(font_scale=1.5)

sns.set_style('ticks') # change background to white background



ax = sns.regplot(x='year', y='total', data=df_tot, color='green', marker='+', scatter_kws={'s': 200})

ax.set(xlabel='Year', ylabel='Total Immigration')

ax.set_title('Total Immigration to Canada from 1980 - 2013')
plt.figure(figsize=(15, 10))



sns.set(font_scale=1.5)

sns.set_style('whitegrid')



ax = sns.regplot(x='year', y='total', data=df_tot, color='green', marker='+', scatter_kws={'s': 200})

ax.set(xlabel='Year', ylabel='Total Immigration')

ax.set_title('Total Immigration to Canada from 1980 - 2013')
### type your answer here



# create df_countries dataframe

df_countries = df_can.loc[['Denmark', 'Norway', 'Sweden'], years].transpose()



 # create df_total by summing across three countries for each year

df_total = pd.DataFrame(df_countries.sum(axis=1))



 # reset index in place

df_total.reset_index(inplace=True)



 # rename columns

df_total.columns = ['year', 'total']



 # change column year from string to int to create scatter plot

df_total['year'] = df_total['year'].astype(int)



# define figure size

plt.figure(figsize=(15, 10))



# define background style and font size

sns.set(font_scale=1.5)

sns.set_style('whitegrid')



# generate plot and add title and axes labels

ax = sns.regplot(x='year', y='total', data=df_total, color='green', marker='+', scatter_kws={'s': 200})

ax.set(xlabel='Year', ylabel='Total Immigration')

ax.set_title('Total Immigrationn from Denmark, Sweden, and Norway to Canada from 1980 - 2013')
# !conda install -c conda-forge folium=0.5.0 --yes

import folium



print('Folium installed and imported!')
# define the world map

world_map = folium.Map()



# display world map

world_map
# define the world map centered around Canada with a low zoom level

world_map = folium.Map(location=[56.130, -106.35], zoom_start=4)



# display world map

world_map
# define the world map centered around Canada with a higher zoom level

world_map = folium.Map(location=[56.130, -106.35], zoom_start=8)



# display world map

world_map
# define Mexico's geolocation coordinates

mexico_latitude = 23.6345 

mexico_longitude = -102.5528



# define the world map centered around mexico with a higher zoom level

mexico_map = folium.Map(location=[mexico_latitude, mexico_longitude], zoom_start=4)



# display world map

mexico_map
# create a Stamen Toner map of the world centered around Canada

world_map = folium.Map(location=[56.130, -106.35], zoom_start=4, tiles='Stamen Toner')



# display map

world_map
# create a Stamen Toner map of the world centered around Canada

world_map = folium.Map(location=[56.130, -106.35], zoom_start=4, tiles='Stamen Terrain')



# display map

world_map
# create a world map with a Mapbox Bright style.

world_map = folium.Map(tiles='Mapbox Bright')



# display the map

world_map
# define Mexico's geolocation coordinates

mexico_latitude = 23.6345 

mexico_longitude = -102.5528



# define the world map centered around mexico with a higher zoom level

mexico_map = folium.Map(location=[mexico_latitude, mexico_longitude], zoom_start=6, tiles='Stamen Terrain')



# display world map

mexico_map


df_can = pd.read_excel('/kaggle/input/canadian-immigration-from-1980-to-2013/Canada.xlsx',

                     sheet_name='Canada by Citizenship',

                     skiprows=range(20),

                     skipfooter=2)



print('Data downloaded and read into a dataframe!')
df_can.head()
# print the dimensions of the dataframe

print(df_can.shape)
# clean up the dataset to remove unnecessary columns (eg. REG) 

df_can.drop(['AREA','REG','DEV','Type','Coverage'], axis=1, inplace=True)



# let's rename the columns so that they make sense

df_can.rename(columns={'OdName':'Country', 'AreaName':'Continent','RegName':'Region'}, inplace=True)



# for sake of consistency, let's also make all column labels of type string

df_can.columns = list(map(str, df_can.columns))



# add total column

df_can['Total'] = df_can.sum(axis=1)



# years that we will be using in this lesson - useful for plotting later on

years = list(map(str, range(1980, 2014)))

print ('data dimensions:', df_can.shape)
df_can.head()
# download countries geojson file

!wget --quiet https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DV0101EN/labs/Data_Files/world_countries.json -O world_countries.json

    

print('GeoJSON file downloaded!')
import folium



world_geo = r'world_countries.json' # geojson file



# create a plain world map

world_map = folium.Map(location=[0, 0], zoom_start=2, tiles='Mapbox Bright')


# world_geo = r'world_countries.json' # geojson file

# # create a plain world map

# world_map = folium.Map(location=[0, 0], zoom_start=2, tiles='Mapbox Bright')



# # generate choropleth map using the total immigration of each country to Canada from 1980 to 2013

# folium.Choropleth(

#     geo_data=world_geo,

#     data=df_can,

#     columns=['Country', 'Total'],

#     key_on='feature.properties.name',

#     fill_color='YlOrRd', 

#     fill_opacity=0.7, 

#     line_opacity=0.2,

#     legend_name='Immigration to Canada'

# ).add_to(world_map)



# # display map



# world_map
# # generate choropleth map using the total immigration of each country to Canada from 1980 to 2013



# world_map.choropleth(

#     geo_data=world_geo,

#     data=df_can,

#     columns=['Country', 'Total'],

#     key_on='feature.properties.name',

#     fill_color='YlOrRd', 

#     fill_opacity=0.7, 

#     line_opacity=0.2,

#     legend_name='Immigration to Canada'

# )



# # display map

# world_map