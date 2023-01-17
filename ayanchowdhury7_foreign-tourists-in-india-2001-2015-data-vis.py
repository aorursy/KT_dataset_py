# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
path = '/kaggle/input/trst1.xls'

df = pd.read_excel(path)
df
df.rename(columns={'Name of Countries': 'Country'},inplace = True)

df
# let's examine the types of the column labels

all(isinstance(column, str) for column in df.columns)
# change the column label to string

df.columns = list(map(str, df.columns))

# types of column labels

all(isinstance(column, str) for column in df.columns)
# Set the country name as index - useful for quick;y looking up countries using .loc method.

df.set_index('Country', inplace = True)

# let's view the first five elements and see how the dataframe was changed

df.head()
# add total column

df.loc[:,'Total'] = df.sum(axis=1)

df
# finally let's create a list of years from 2001 - 2015

# this will come in handy when we start plotting the data



years = list(map(str, range(2001, 2016)))



years
%matplotlib inline



import matplotlib as mpl

import matplotlib.pyplot as plt



mpl.style.use('ggplot')  # optional for ggplot-like style



# check for the latest version of Matplotlib

print('Matpolotlib version: ', mpl.__version__)
df.sort_values(by = 'Total', ascending = False, axis = 0,inplace = True)
df_tourist5 = df.iloc[2:7]

df_tourist5 = df_tourist5.drop(columns = 'Total')
# transpose the data frame

df_tourist5 = df_tourist5[years].transpose()
df_tourist5
# Plotting the data



df_tourist5.index = df_tourist5.index.map(int)

df_tourist5.plot(kind = 'line',

             figsize = (14,8))

plt.title('Top 5 visiting nationalities to India')

plt.xlabel('Years')

plt.ylabel('Number of Tourists')



plt.show()
df_tourist5.plot(kind = 'area',

              stacked = False,

              alpha = 0.35, # transparency coefficient, default 0.5, can set between 0 - 1

              figsize =(20,10),

             )

plt.title('Tourists Trend of Top 5 Countries to India')

plt.ylabel('Number of Tourists')

plt.xlabel('Years')



plt.show()
# selecting the dataset

df_jct = df.loc[['JAPAN', 'CHINA (MAIN)', 'Thailand'], years]
# Transpose dataframe

df_jct = df_jct.transpose()

df_jct.head()

df_jct
# lets get the x-tick values

count, bin_edges = np.histogram(df_jct, 15)

# Unstacked histogram

df_jct.plot(kind = 'hist',

            figsize = (15,8),

            bins = 15,

            alpha = 0.35,

            stacked = False,

            xticks = bin_edges,

            color = ['coral', 'darkslateblue', 'mediumseagreen']

           )

plt.title('Histogram of Tourists from Japan, China and Thailand to India from 2001-2015')

plt.ylabel('Number of Years')

plt.xlabel('Number of Tourists')



plt.show()
# let's check the data set

df_tourist5
# getting the x-tick values

count, bin_edges = np.histogram(df_tourist5, 10)

df_tourist5.plot(kind = 'hist',

              figsize = (15,8),

              bins = 10,

              alpha = 0.65,

              xticks = bin_edges,)



plt.title('Histogram of Tourists from Top 5 countries to India from 2001-2015')

plt.ylabel('No of Years')

plt.xlabel('Number of Tourists')

plt.show()
# Extracting the data for Pakistan

df_pak = df.loc['Pakistan', years]

df_pak
# plot data



df_pak.plot(kind='bar', figsize = (10,6),rot = 90)



plt.xlabel('Year')

plt.ylabel('Number of Tourists')

plt.title('Pakistani Tourists to India from 2001 to 2015')





# Annotate arrow

plt.annotate('', # s: str. will leave it blank for no text

             xy =(10,50000), # place head of the arrow at point(year 2008, tourists 90000)

             xytext = (7,90000), # place base of the arrow

             xycoords = 'data', # will use the coordinate system of the object being annotated

             arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3', color = 'blue', lw = 2 )



            )



# Annotate Text

plt.annotate('2008 Mumbai Attack', # text to display

             xy=(7.5,60000),                    # start the text at at point (year 2008 , pop 30)

             rotation=-42,                  # based on trial and error to match the arrow

             va='bottom',                    # want the text to be vertically 'bottom' aligned

             ha='left',                      # want the text to be horizontally 'left' algned.

            )

plt.show()
# Extracting the data



# sort the data

df.sort_values(by = 'Total', ascending = True, axis = 0,inplace = True)



# slice out last two rows and create a new dataframe

df_total = df.iloc[:-2]



# selecting only the total column

df_total = df_total['Total']



# see the data frame

df_total
# plotting the data

df_total.plot(kind = 'barh', figsize=(12,12), color = 'steelblue')

plt.xlabel('Number of Tourists')

plt.title('Foriegn tourists to India between 2001 - 2015')



# annonate value labels to each country

for index, value in enumerate(df_total):

    label = format(int(value),',') # format int with commas

    

    plt.annotate(label, xy=(value-990000, index-.1), color = 'White')

plt.show()
df
df_total = df.iloc[:-2]
# Selecting the total column

df_tour5 = df_total['Total'].tail(5)



df_tour5
# plotting the data



colors_list = ['maroon', 'crimson', 'dodgerblue', 'aqua', 'darkgreen'] # colors list

explode_list = [0, 0, 0, 0.04, 0.04]

df_tour5.plot(kind = 'pie',

              figsize = (15,6),

              autopct = '%1.1f%%',

              startangle = 90,

              shadow = True,

              labels = None,

              pctdistance = 1.12,

              colors = colors_list,

              explode = explode_list

             )

plt.title('Tourists to India from Top 5 countries [2001 - 2015]', y = 1.12)

plt.axis('equal')



plt.legend(labels = df_tour5.index, loc = 'upper left')



plt.show()
df_jct
df_jct.plot(kind = 'box', figsize = (8,6))

plt.title('Box plot of Tourists from Japan, China and Thailand')

plt.ylabel('Number of Tourists')

plt.xlabel('Country')

plt.show()
# Horizontal Box plot

df_jct.plot(kind = 'box', figsize = (8,6), color = 'blue', vert = False)

plt.title('Box plot of Tourists from Japan, China and Thailand')

plt.xlabel('Number of Tourists')

plt.ylabel('Country')

plt.show()
# sub-plot



fig = plt.figure()



ax0 = fig.add_subplot(1, 2, 1)

ax1 = fig.add_subplot(1, 2, 2)



# subplot 1: Box Plot

df_jct.plot(kind = 'box', color = 'blue', vert = False, figsize=(20,6), ax = ax0)

ax0.set_title('Toruists from Japan. China, and Thailand to India [2001 - 2015]')

ax0.set_xlabel('Number of Tourists')

ax0.set_ylabel('Country')

# subplot 2: Line Plot

df_jct.plot(kind = 'line', figsize = (20,6), ax = ax1)

ax1.set_title('Toruists from Japan. China, and Thailand to India [2001 - 2015]')

ax1.set_xlabel('Number of Tourists')

ax1.set_ylabel('Year')

plt.show()
# extracting dataset

df
# drop the total column and save as df_tot

df_tot = df.drop(columns = 'Total')

df_tot
# Now select the row named 'total'

df_tot = df_tot.iloc[[-1], :]

df_tot
# Transposing the data

df_tot = df_tot.transpose()
df_tot
# reseting the index

df_tot.reset_index(inplace = True)
# set columns name

df_tot.columns = ['year', 'total']
df_tot
# Change the str value to numeric

df_tot["year"] = pd.to_numeric(df_tot["year"])

df_tot["total"] = pd.to_numeric(df_tot["total"])
# plotting the data



df_tot.plot(kind = 'scatter' , x = 'year', y = 'total', figsize = (10, 6), color = 'darkblue')



plt.title('Total Tourists to India [2001 - 2015]')

plt.xlabel('Year')

plt.ylabel('Number of Tourists')

plt.show()
x = df_tot['year']

y = df_tot['total']



fit = np.polyfit(x, y, deg = 1)



fit
df_tot.plot(kind = 'scatter', x = 'year', y = 'total', figsize = (10,6), color = 'darkblue')



plt.title('Total Tourists to India [2001 - 2015]')

plt.xlabel('Year')

plt.ylabel('Number of Tourists')



# plot line of best fit

plt.plot(x, fit[0] * x + fit[1], color = 'red')

plt.annotate('y={0:.0f} x + {1:.0f}'.format(fit[0], fit[1]), xy=(2004,6000000))



plt.show()



# print out the line of best fit

'No. Tourists = {0:.0f} * Year + {1:.0f}'.format(fit[0], fit[1]) 
df
df_UK_USA = df.loc[['UK' , 'USA'], :]
df_UK_USA
# drop the total column

df_UK_USA = df_UK_USA.drop(columns = 'Total')
# transposing the dataset

df_UK_USA = df_UK_USA.transpose()
# normalized USA data

norm_usa = (df_UK_USA['USA'] - df_UK_USA['USA'].min()) / (df_UK_USA['USA'].max() - df_UK_USA['USA'].min())



# normalized UK data

norm_uk = (df_UK_USA['UK'] - df_UK_USA['UK'].min()) / (df_UK_USA['UK'].max() - df_UK_USA['UK'].min())
# let's label the index. This will automatically be the column name when we reset the index

df_UK_USA.index.name = 'Year'

# reset index to bring the Year in as a column

df_UK_USA.reset_index(inplace=True)

df_UK_USA
# Change the str value to numeric

df_UK_USA["Year"] = pd.to_numeric(df_UK_USA["Year"])

df_UK_USA["USA"] = pd.to_numeric(df_UK_USA["USA"])

df_UK_USA["UK"] = pd.to_numeric(df_UK_USA["UK"])
# Plotting the data



# USA

ax0 = df_UK_USA.plot(kind = 'scatter',

                     x = 'Year',

                     y = 'USA',

                     figsize = (14,8),

                     alpha = 0.5,

                     color = 'cyan',

                     s = norm_usa * 2000 + 10, # pass in weights

                     xlim = (1999,2017)

                    )



# UK

ax0 = df_UK_USA.plot(kind = 'scatter',

                     x = 'Year',

                     y = 'UK',

                     figsize = (14,8),

                     alpha = 0.5,

                     color = 'maroon',

                     s = norm_uk * 2000 + 10, # pass in weights

                     xlim = (1999,2017),

                     ax = ax0

                    )



ax0.set_ylabel('Number of Tourists')

ax0.set_title('Tourists from USA and UK from 2001 to 2015')

ax0.legend(['USA', 'UK'], loc = 'upper left', fontsize = 'x-large')
# Extracting the dataset for total tourists

df_tot
# Plotting the data



# importing seaborn

import seaborn as sns



plt.figure(figsize = (15,10))



sns.set(font_scale = 1.5) # chaning the font scale

sns.set_style('whitegrid') # change the background to white grid

ax = sns.regplot(x = 'year', y = 'total', data = df_tot, color = 'green')



ax.set(xlabel = 'Year', ylabel = 'Total Tourists')

ax.set_title('Total Tourists to India [2001 - 2015]')