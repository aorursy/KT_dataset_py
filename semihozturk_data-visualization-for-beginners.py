# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as mpl
import matplotlib.pyplot as plt # standard python visualization library
%matplotlib inline

import matplotlib.patches as mpatches # needed for waffle Charts

# import library
import seaborn as sns


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/new-zealand-citizenships-19492019/granted-citizenship-1949-2019.csv")
data.head() # view the first 5 rows
data.tail() # view the last 5 rows
data = pd.read_csv("../input/new-zealand-citizenships-19492019/granted-citizenship-1949-2019.csv",
                   skipfooter=1) # i skipped last one row
data.tail() # view the first 5 rows
data.info()
data["%"] = data["%"].str[:4]
data.head()
data["%"] = data["%"].astype(float)
data.rename(columns={"%":"percentage_distribution"}, inplace=True)
data.columns.values # they all are object which I'd like so
data.index.values
data.shape # size of data frame(rows,columns)
years = list(map(str,range(1949, 2020))) # there are years from 1949 to 2019, I want to get it as a list
# since range() function returns integer, i should use it with map() function to make them string
data.loc[0,years].sum() # sums the 0 indexed row values
data.isnull().sum().sum() # we have this many NaN values in dataframe
data.fillna(0, inplace=True) # filled with zero
data.head()
data.isnull().sum().sum() # there is no NaN value anymore
# quick review of each column
data.describe() 
data.Total # filter on Total column
data[["Country of Birth","Total","2019"]] # filter on multiple columns
data.set_index("Country of Birth",inplace=True) # changing the index of dataframe
data.head(2)
data.index.name = None
data.head(2)
data[years].head(2)
data[years] = data[years].astype(int)
data[years].head(2)
data.loc["Russia","2019"]
years00_15 = list(map(str,range(2000,2016)))
data.loc["Russia",years00_15]
condition1 = data["2015"]>500 # returns False or True
data[condition1] # show me countries of people that got new zealand citizenship more than 500 in 2015
condition2 = data["2010"]>1000
data[condition1 & condition2]
india = data.loc["India",years] 
india.plot() # as default, it plots like this
plt.show()
mpl.style.use(["ggplot"]) # mpl is matplotlib library
india.plot() 
plt.show()
# we can see how many types we can plot
plt.style.available # you can play around with them
india.index.values # they are object,i want to change them to str
india.index = india.index.map(int)
india.plot(kind="line") # its line by default

plt.title("New Zealand Citizenship for Indian people")
plt.xlabel("Years")
plt.ylabel("Number of citizenship")
plt.show()
india.index = india.index.map(int)
india.plot(kind="line") 

plt.title("New Zealand Citizenship for Indian people")
plt.xlabel("Years")
plt.ylabel("Number of citizenship")
plt.text(1985,2200, "1990 Year") # x axis, y axis
plt.show()
years80_10 = list(map(str,range(1980,2011)))
data_IE = data.loc[["India","England"],years80_10]
data_IE
data_IE = data_IE.transpose()
data_IE.head()
data_IE.index = data_IE.index.map(int) # changing str to int
data_IE.plot(kind="line")
plt.title("Citizenship from India, and England")
plt.xlabel("Years")
plt.ylabel("Number of citizenship")
plt.show()
# sort the data
data.sort_values(by="Total", ascending=False, axis=0, inplace=True)

# get the top 4 countries
data_top4 = data.head(4)
data_top4
# if i want only Total column, data.head(4).loc[:,"Total"]
# transpose the dataframe by years
data_top4 = data_top4[years].transpose()
data_top4
# plot the dataframe
data_top4.index = data_top4.index.map(int) # change them to int from str
data_top4.plot(kind="line",figsize=(12,6))

plt.title("top 4 countries that got New Zealand citizenship")
plt.xlabel("Years")
plt.ylabel("Number or Citizenship")
plt.show()
data_top4.head()
data_top4.plot(kind="area",
               stacked = False,
               figsize=(17,8))
plt.title("top 4 countries that got New Zealand citizenship")
plt.xlabel("Years")
plt.ylabel("Number or Citizenship")
plt.show()
ax = data_top4.plot(kind="area",stacked = False, figsize=(17,8))
ax.set_title("top 4 countries that got New Zealand citizenship")
ax.set_xlabel("Years")
ax.set_ylabel("Number or Citizenship")
plt.show()
data["2019"].head()
count, bin_edges = np.histogram(data["2019"]) # returns 2 values
print(count) # frequency
print(bin_edges) # bin ranges, default=10 bins
data["2019"].plot(kind="hist",figsize=(7,4))
plt.title("histogram of 324 countries that got new zealand citizenship in 2019")
plt.xlabel("Number of Citizenships")
plt.ylabel("Number of Countries")
plt.show()
count, bin_edges = np.histogram(data["2019"])
data["2019"].plot(kind="hist",figsize=(10,4), xticks=bin_edges)
plt.title("histogram of 324 countries that got new zealand citizenship in 2019")
plt.xlabel("Number of Citizenships")
plt.ylabel("Number of Countries")
plt.show()
data.loc[["Sweden","Russia","Turkey"]]
# transpose the dataframe
data_srt = data.loc[["Sweden","Russia","Turkey"], years].transpose()
data_srt.head()
data_srt.plot(kind="hist",figsize=(7,4))
plt.title("Citizenship from Sweden, Russia, Turkey")
plt.xlabel("Number of Citizenship")
plt.ylabel("Years") # from 1949 to 2019 -> 70 years
plt.show()
count,bin_edges = np.histogram(data_srt,15)

# unstacked histogram
data_srt.plot(kind="hist",
              figsize=(15,5),
              bins=bin_edges,
              alpha=0.5,
              xticks=bin_edges,
              color=['coral', 'darkslateblue', 'mediumseagreen'])

plt.title("Citizenship from Sweden, Russia, Turkey")
plt.xlabel("Number of Citizenship")
plt.ylabel("Years")
plt.show()
count, bin_edges = np.histogram(data_srt, 15)
xmin = bin_edges[0]    
xmax = bin_edges[-1]  

# stacked Histogram
data_srt.plot(kind='hist',
          figsize=(14, 6), 
          bins=15,
          xticks=bin_edges,
          color=['coral', 'darkslateblue', 'mediumseagreen'],
          stacked=True, # it adds up
          xlim=(xmin, xmax)
         )

plt.title("Citizenship from Sweden, Russia, Turkey")
plt.xlabel("Number of Citizenship")
plt.ylabel("Years")
plt.show()
# getting the data
years77_19 = list(map(str, range(1977,2020)))
data_england = data.loc["England",years77_19]
data_england.head()
# plotting the data
data_england.plot(kind="bar",figsize=(12,6))
plt.xlabel("Year")
plt.ylabel("Number of Citizenships")
plt.title("New Zealand Citizenship of people from England from 1977 to 2019")
plt.show()

# sort dataframe on "Total" column
data.sort_values(by="Total", ascending=True, inplace=True)
# get top 10 countries
data_top10 = data["Total"].tail(10)

# plot the data
data_top10.plot(kind="barh", figsize=(10,10), color="steelblue")
plt.xlabel("Number of Citizenships")
plt.title("Top 10 Countries that got New Zealand Citizenship between 1977-2019")

# annotate value labels to each country
for index, value in enumerate(data_top10):
    label = format(int(value),",") # format int with commas
    
    # place text at the end of the bar(subtracting 14000 from x, and 0.1 from y to make it fit within the bar)
    plt.annotate(label, xy=(value - 14000, index - 0.10), color='white')
    
plt.show()
# last time, we sorted our data in ascending order, lets change it in descending
data.sort_values(by="Total", ascending=False, inplace=True)
data.head()
data.head()["Total"].plot(kind="pie",
                   figsize=(5,6),
                   autopct='%1.1f%%',# add in percentages
                   startangle=90,     # start angle 90° (England)
                   shadow=True,       # add shadow      
                  )
plt.title('top 5 countries')
plt.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()
colors_list = ["aliceblue","burlywood","lemonchiffon","skyblue","slategrey"]
explode_list = [0, 0, 0, 0.1, 0.1] # ratio for each continent with which to offset, lets explode the lowest 2 countries
data.head()["Total"].plot(kind="pie",
                   figsize=(5,6),
                   autopct='%1.1f%%',    # add in percentages
                   startangle=90,        # start angle 90° (England)
                   shadow=True,          # add shadow      
                   labels=None,          # turn off labels on pie chart
                   pctdistance =1.12,    # the ratio between the center of each pie slice and the start of the text generated by autopct 
                   colors=colors_list,   # add custom colors
                   explode=explode_list  # 'explode' lowest 2 countries
                   )
# scale the title up by 10% to match pctdistance
plt.title("top 5 countries", y=1.1) # you can change "y" value

plt.axis('equal')

# add legend
plt.legend(labels=data.head().index, loc='upper left') 
plt.show()
years05_19 = list(map(str,range(2005,2020)))
data.loc[["Turkey"],years05_19]
data_turkey = data.loc[["Turkey"],years05_19].transpose()
data_turkey.head()
data_turkey.plot(kind="box", figsize=(5,5))

plt.title("people from Turkey that got New Zealand citizenship between 2005-2019")
plt.ylabel("Number of Citizenships")
plt.show()
data_turkey.describe()
data_TI = data.loc[["Turkey","Italy"],years05_19].transpose()
data_TI.head()
data_TI.describe()
data_TI.plot(kind='box', figsize=(8, 6))
plt.title('Box plots of New Zealand Citizenships from Turkey and Italy(2005 - 2019)')
plt.xlabel('Number of Citizenships')
plt.show()
data_TI.plot(kind='box', figsize=(8, 6), color="red", vert=False)
plt.title('Box plots of New Zealand Citizenships from Turkey and Italy(2005 - 2019)')
plt.xlabel('Number of Citizenships')
plt.show()
fig = plt.figure() # create figure
ax0 = fig.add_subplot(1,2,1) # add 1 row, 2 columns, and this is the first plot
ax1 = fig.add_subplot(1,2,2) # this is the second plot

# subplot1-> box plot
data_TI.plot(kind='box', figsize=(17, 6), color="red", vert=False,ax=ax0) # add to subplot1
ax0.set_title('Box plots of New Zealand Citizenships from Turkey and Italy(2005 - 2019)')
ax0.set_xlabel('Number of Citizenships')
ax0.set_ylabel("Country")

# subplot2 -> line plot
data_TI.plot(kind='line', figsize=(20, 6), ax=ax1) # add to subplot 2
ax1.set_title ('Line Plots of New Zealand Citizenships from Turkey and Italy(2005 - 2019)')
ax1.set_ylabel('Number of Citizenships')
ax1.set_xlabel('Year')

plt.show()
data_top10 = data.sort_values(by="Total",ascending=False,axis=0).head(10) # get the top 10 countries by Total

# create list of years
years_90s = list(map(str,range(1990,2000)))
years_00s = list(map(str,range(2000,2010)))
years_10s = list(map(str,range(2010,2020)))

# get the value for each decades as series
data_90s = data_top10.loc[:,years_90s].sum(axis=1)
data_00s = data_top10.loc[:,years_00s].sum(axis=1)
data_10s = data_top10.loc[:,years_10s].sum(axis=1)

# merge series to dataframe
new_data = pd.DataFrame({"1990s":data_90s, "2000s":data_00s, "2010s":data_10s})

new_data.head()
new_data.describe()
# Plot the box plots
new_data.plot(kind='box', figsize=(10, 6))
plt.title('Citizenships from top 10 countries for decades 90s, 2000s and 2010s')
plt.show()
# let's check how many entries fall above the outlier threshold 
new_data[new_data['1990s'] > 29476.25] # England is the outlier in 1990s 
data[years].head()
data[years].sum(axis=0).head()
pd.DataFrame(data[years].sum(axis=0).head())
# we can use the sum() method to get the total population per year
data_total = pd.DataFrame(data[years].sum(axis=0))

# change the years to type int (it will be useful for regression later on)
data_total.index = map(int, data_total.index)

# reset the index to put in back in as a column in the data_total dataframe
data_total.reset_index(inplace = True)

# rename columns
data_total.columns = ['year', 'total'] 

data_total.head()
data_total.plot(kind="scatter", x="year", y="total", figsize=(10,6),color="blueviolet")

plt.title('Total New Zealand Citizenships from other countries for 1949 - 2019')
plt.xlabel('Year')
plt.ylabel('Number of Citizenships')

plt.show()
x = data_total['year']      # year on x-axis
y = data_total['total']     # total on y-axis
fit = np.polyfit(x, y, deg=1)

fit
data_total.plot(kind="scatter",x="year",y="total",figsize=(10,6),color="blueviolet")

plt.title('Total New Zealand Citizenships from other countries for 1949 - 2019')
plt.xlabel('Year')
plt.ylabel('Number of Citizenships')

# plot line of best fit
plt.plot(x, fit[0] * x + fit[1], color='red') # x is the Years, and y = a x + b
plt.annotate('y={0:.0f} x + {1:.0f}'.format(fit[0], fit[1]), xy=(2000, 10000)) # y equation label

plt.show()

# print out the line of best fit
'Number of Citizenships = {0:.0f} * Year + {1:.0f}'.format(fit[0], fit[1]) 
# lets transpose our data to get country list as column
data_t = data[years].transpose()

# cast the Years (the index) to type int
data_t.index = map(int,data_t.index)

# label the index. This will automatically be the column name when we reset the index
data_t.index.name = 'Year'

# reset index to bring the Year in as a column
data_t.reset_index(inplace=True)

data_t.head()
# normalize Italy data
norm_italy = (data_t['Italy'] - data_t['Italy'].min()) / (data_t['Italy'].max() - data_t['Italy'].min())

# normalize Spain data
norm_spain = (data_t['Spain'] - data_t['Spain'].min()) / (data_t['Spain'].max() - data_t['Spain'].min())
# Italy
ax0 = data_t.plot(kind='scatter',
                    x='Year',
                    y='Italy',
                    figsize=(14, 8),
                    alpha=0.5,                  # transparency
                    color='green',
                    s=norm_italy * 1000,  # pass in weights 
                    xlim=(1949, 2019) # x axis
                   )

# Argentina
ax1 = data_t.plot(kind='scatter',
                    x='Year',
                    y='Spain',
                    alpha=0.5,
                    color="blue",
                    s=norm_spain * 1000,
                    ax = ax0
                   )

ax0.set_ylabel('Number of Citizenships')
ax0.set_title('Citizenship of people from Italy and Spain from 1949 - 2019')
ax0.legend(['Italy', 'Spain'], loc='upper left', fontsize='x-large')
plt.show()
# lets get data for three countries
data_fgs = data.loc[["France","Germany","Singapore"],:]
data_fgs
# compute the proportion of each category with respect to the total
total_values = sum(data_fgs['Total'])
category_proportions = [(value / total_values) for value in data_fgs['Total']]

# print out proportions
for i, proportion in enumerate(category_proportions):
    print (data_fgs.index.values[i] + ': ' + str(proportion))
# defining the overall size of the waffle chart
width = 40 # width of chart
height = 10 # height of chart
total_num_tiles = width * height # total number of tiles
# compute the number of tiles for each category
tiles_per_category = [round(proportion * total_num_tiles) for proportion in category_proportions]

# print out number of tiles per category
for i, tiles in enumerate(tiles_per_category):
    print (data_fgs.index.values[i] + ': ' + str(tiles))
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
waffle_chart
# lets find if we did right
unique, counts = np.unique(waffle_chart, return_counts=True)
dict(zip(unique, counts)) 
# result is true, so France:81, Germany:186, Singapore:133
# Map the waffle chart matrix into a visual
fig = plt.figure()

# use matshow to display the waffle chart
colormap = plt.cm.coolwarm
plt.matshow(waffle_chart, cmap=colormap)
plt.colorbar()
plt.show()
# Lets pretify this

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
values_cumsum = np.cumsum(data_fgs['Total'])
total_values = values_cumsum[len(values_cumsum) - 1]

# create legend
legend_handles = []
for i, category in enumerate(data_fgs.index.values):
    label_str = category + ' (' + str(data_fgs['Total'][i]) + ')'
    color_val = colormap(float(values_cumsum[i])/total_values)
    legend_handles.append(mpatches.Patch(color=color_val, label=label_str))

# add legend to chart
plt.legend(handles=legend_handles,
           loc='lower center', 
           ncol=len(data_fgs.index.values),
           bbox_to_anchor=(0., -0.2, 0.95, .1)
          )
plt.show()
# import package and its set of stopwords
from wordcloud import WordCloud, STOPWORDS

import urllib.request
response = urllib.request.urlopen("https://www.w3.org/TR/PNG/iso_8859-1.txt")
text_file = response.read().decode('utf-8')
# use the stopwords that we imported from word_cloud
stopwords = set(STOPWORDS)

# instantiate a word cloud object
text_wc = WordCloud(
    background_color='white',
    max_words=2500, # used the first 250 letters in text file
    stopwords=stopwords
)

# generate the word cloud
text_wc.generate(text_file)
fig = plt.figure()
fig.set_figwidth(8)    # set width
fig.set_figheight(10)  # set height

# display the word cloud
plt.imshow(text_wc, interpolation='bilinear')
plt.axis('off')
plt.show()
stopwords.add('letter') # add the words said to stopwords

# re-generate the word cloud
text_wc.generate(text_file)

# display the cloud
fig = plt.figure()
fig.set_figwidth(8) # set width
fig.set_figheight(10) # set height

plt.imshow(text_wc, interpolation='bilinear')
plt.axis('off')
plt.show()
data.head()
total_citizenship = data["Total"].sum()
total_citizenship
max_words = 70
word_string = ''
for country in data.index.values:
    # check if country's name is a single-word 
    if len(country.split(' ')) == 1:
        repeat_num_times = int(data.loc[country, 'Total']/float(total_citizenship)*max_words)
        word_string = word_string + ((country + ' ') * repeat_num_times)
                                     
# display the generated text
word_string
# create the word cloud
wordcloud = WordCloud(background_color='white').generate(word_string)

# display the cloud
fig = plt.figure()
fig.set_figwidth(10)
fig.set_figheight(12)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
data_total.head()
ax = sns.regplot(x='year', y='total', data=data_total)
# changing the color and adding marker
ax = sns.regplot(x='year', y='total', data=data_total,color="green",marker="+")
plt.figure(figsize=(12, 7))
ax = sns.regplot(x='year', y='total', data=data_total,color="green",marker="+", scatter_kws={'s': 150})
sns.set(font_scale=1.5)
sns.set_style('ticks') # change background to white background

ax.set(xlabel='Year', ylabel='Total Citizenships') # add x- and y-labels
ax.set_title('New Total Citizenship of New Zealand from 1949 - 2019') # add title
plt.show()