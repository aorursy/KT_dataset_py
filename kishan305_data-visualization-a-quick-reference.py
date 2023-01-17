import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# importing libraries required

import numpy as np

import matplotlib.pyplot as plt



# defining figuresize and theme for plot

plt.rcParams['figure.figsize'] = (18, 10)

plt.style.use('ggplot')



# plotting the area graph

turnover = [2, 7, 14, 17, 20, 27, 30, 38, 25, 18, 6, 1]

plt.fill_between(np.arange(12), turnover,color="skyblue", alpha=0.4)

plt.plot(np.arange(12), turnover, color="darkblue",alpha=0.6, linewidth=2)



# customizing the plot

plt.tick_params(labelsize=15)

plt.xticks(np.arange(12), np.arange(1,13))

plt.title('Ice-Cream sells Distribution over an Year', size=20, color='k')

plt.xlabel('Month', size=18)

plt.ylabel('Turn-over of ice-cream', size=18)

plt.ylim(bottom=0)



plt.show()
# importing libraries

import numpy as np 

import matplotlib.pyplot as plt  



# defining figuresize and theme for the plot

plt.rcParams['figure.figsize'] = (18, 8)

plt.style.use('ggplot')



# creating the dataset 

data = {'C':20, 'C++':15, 'Java':30,  

        'Python':35} 

courses = list(data.keys()) 

values = list(data.values())  

  

# creating the bar plot 

plt.bar(courses, values, color ='maroon',  

        width = 0.4) 



# customizing the bar plot

plt.tick_params(labelsize=15)  

plt.xlabel("Courses offered", size=18) 

plt.ylabel("No. of students enrolled", size=18) 

plt.title("Students enrolled in different courses", size=20, color='k') 

plt.show()
# importing libraries 

import matplotlib.pyplot as plt 

import numpy as np 



# defining figuresize and theme for the plot

plt.rcParams['figure.figsize'] = (18, 10)

plt.style.use('ggplot')



# creating dataset 

np.random.seed(10) 

  

data_1 = np.random.normal(100, 10, 200) 

data_2 = np.random.normal(90, 20, 200) 

data_3 = np.random.normal(80, 30, 200) 

data_4 = np.random.normal(70, 40, 200) 

data = [data_1, data_2, data_3, data_4]  





# creating Box plot 

plt.boxplot(data) 

plt.tick_params(labelsize=15) 

plt.show() 
# importing libraries

import matplotlib.pyplot as plt

import numpy as np



# defining figuresize and theme for the plot

plt.rcParams['figure.figsize'] = (15, 8)

plt.style.use('fast')



# create data

x = np.random.rand(20)

y = np.random.rand(20)

z = np.random.rand(20)

colors = np.random.rand(20)



# use the scatter function to plot the bubble chart

plt.scatter(x, y, s=z*1000,c=colors)



# customizing the plot

plt.tick_params(labelsize=15)

plt.xlabel("X", size=18)

plt.ylabel("y", size=18)

plt.title("Bubble Plot with Matplotlib", size=20, color='k')

plt.show()
# importing libraries

import folium

import pandas as pd



# defining san-francisco coordinates

SF_COORDINATES = (37.76, -122.45)



# reading san-francisco crime data

crimedata = pd.read_csv('/kaggle/input/sanfranciso-crime-dataset/Police_Department_Incidents_-_Previous_Year__2016_.csv')

 

# for speed purposes choose only 10 crime records

MAX_RECORDS = 10

  

# create empty map zoomed in on San Francisco

map = folium.Map(location=SF_COORDINATES, zoom_start=12)

 

# add a marker for every record in the filtered data, use a clustered view

for each in crimedata[0:MAX_RECORDS].iterrows():

    folium.CircleMarker(

        location = [each[1]['Y'],each[1]['X']], 

        clustered_marker = True,

        color='crimson',

        fill=True,

        fill_color='crimson'

    ).add_to(map)

  

display(map)
# importing libraries

import calendar

import numpy as np

from matplotlib.patches import Rectangle

import matplotlib.pyplot as plt



# defining calendar plot function

def plot_calendar(days, months):

    # non days are grayed

    ax = plt.gca().axes

    ax.add_patch(Rectangle((29, 2), width=.8, height=.8, 

                           color='gray', alpha=.3))

    ax.add_patch(Rectangle((30, 2), width=.8, height=.8,

                           color='gray', alpha=.5))

    ax.add_patch(Rectangle((31, 2), width=.8, height=.8,

                           color='gray', alpha=.5))

    ax.add_patch(Rectangle((31, 4), width=.8, height=.8,

                           color='gray', alpha=.5))

    ax.add_patch(Rectangle((31, 6), width=.8, height=.8,

                           color='gray', alpha=.5))

    ax.add_patch(Rectangle((31, 9), width=.8, height=.8,

                           color='gray', alpha=.5))

    ax.add_patch(Rectangle((31, 11), width=.8, height=.8,

                           color='gray', alpha=.5))

    for d, m in zip(days, months):

        ax.add_patch(Rectangle((d, m), 

                               width=.8, height=.8, color='coral'))

    plt.yticks(np.arange(1, 13)+.5, list(calendar.month_abbr)[1:])

    plt.xticks(np.arange(1,32)+.5, np.arange(1,32))

    plt.xlim(1, 32)

    plt.ylim(1, 13)

    plt.gca().invert_yaxis()

    # remove borders and ticks

    for spine in plt.gca().spines.values():

        spine.set_visible(False)

    plt.tick_params(top=False, bottom=False, left=False, right=False)

    plt.title('Full Moons in 2018', size=20)

    plt.show()
# defining figuresize and theme for the plot

plt.rcParams['figure.figsize'] = (18, 8)

plt.style.use('grayscale')



# defining data

full_moon_day = [2, 31, 2, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22]

full_moon_month = [1, 1, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]



# plotting the Calendar

plot_calendar(full_moon_day, full_moon_month)
import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.dates as mdates



datafile = '/kaggle/input/week4data/SPY.csv'

data = pd.read_csv(datafile, index_col = 'Date')

data.index = pd.to_datetime(data.index) # Converting the dates from string to datetime format



# We need to exctract the OHLC prices into a list of lists:

dvalues = data[['Open', 'High', 'Low', 'Close']].values.tolist()



# Dates in our index column are in datetime format, we need to comvert them 

# to Matplotlib date format

pdates = mdates.date2num(data.index)



# We prepare a list of lists where each single list is a [date, open, high, low, close] sequence:

ohlc = [ [pdates[i]] + dvalues[i] for i in range(len(pdates)) ]
! pip install mpl_finance
# importing libraries

import mpl_finance as mpf

import warnings

warnings.filterwarnings("ignore")



# defining figuresize and theme for the plot

plt.style.use('fivethirtyeight')

fig, ax = plt.subplots(figsize = (18,8))



# plotting the candelstick chart

mpf.candlestick_ohlc(ax, ohlc[-50:], width=0.4)



# customizing the plot

plt.tick_params(labelsize=15)

ax.set_xlabel('Date', size=18)

ax.set_ylabel('Price ($)', size=18)

ax.set_title('SPDR S&P 500 ETF Trust - Candlestick Chart', size=20, color='k')

ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))



fig.autofmt_xdate()

plt.show()
# importing library

import plotly.express as px



# reading the data

df = px.data.gapminder().query("year==2007")



# plotting the choropleth map

fig = px.choropleth(df, locations="iso_alpha",

                    color="lifeExp", # lifeExp is a column of gapminder

                    hover_name="country", # column to add to hover information

                    color_continuous_scale=px.colors.sequential.Plasma)

fig.show()
# required libraries

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



# reading the data

data = pd.read_csv('/kaggle/input/la-liga-results-19952020/LaLiga Complete Matches 1995-2020.csv')



# setting figuresize and theme for the plot

plt.rcParams['figure.figsize'] = (20,10)

plt.style.use('dark_background')



# creating the countplot

sns.countplot(data['Season'], palette = 'gnuplot')



# customizing the plot

plt.title('Number of Matches Played in each Season', fontweight = 30, fontsize =20)

plt.tick_params(labelsize=15)

plt.xticks(rotation = 90)

plt.show()
# import libraries

import pandas as pd

import matplotlib.pyplot as plt



# read the data

diamonds = pd.read_csv('/kaggle/input/diamonds/diamonds.csv')



# defining figuresize and theme for the plot

plt.rcParams['figure.figsize'] = (15, 10)

plt.style.use('ggplot')



# plotting the density plot

diamonds["carat"].plot(kind="density",  # Create density plot

                      xlim= (0,5));     # Limit x axis values



# customizing the plot

plt.tick_params(labelsize=15)
# library required

import matplotlib.pyplot as plt

 

# create data

size_of_groups=[12,11,3,30]



# Create a pieplot

plt.pie(size_of_groups)

#plt.show()

 

# add a circle at the center

my_circle=plt.Circle( (0,0), 0.7, color='white')

p=plt.gcf()

p.gca().add_artist(my_circle)

plt.show()

# libraries required

import numpy as np 

import matplotlib.pyplot as plt 



# defining figuresize and theme for the plot

plt.rcParams['figure.figsize'] = (18, 12)

plt.style.use('ggplot')



# example data 

xval = np.arange(0.1, 4, 0.5) 

yval = np.exp(-xval) 



# plot the error-bar

plt.errorbar(xval, yval, xerr = 0.4, yerr = 0.5) 



# customize the plot

plt.tick_params(labelsize=15)

plt.xlabel("X", size=18)

plt.ylabel("y", size=18)

plt.title("Error Bar with Matplotlib", size=20)

plt.show() 

# Importing the matplotlb.pyplot 

import matplotlib.pyplot as plt 



plt.rcParams['figure.figsize'] = (18, 10)

plt.style.use('ggplot')



# Declaring a figure "gnt" 

fig, gnt = plt.subplots() 



# Setting Y-axis limits 

gnt.set_ylim(0, 50) 



# Setting X-axis limits 

gnt.set_xlim(0, 160) 



# Setting labels for x-axis and y-axis 

gnt.set_xlabel('seconds since start', size=18) 

gnt.set_ylabel('Processor', size=18) 



# Setting ticks on y-axis 

gnt.set_yticks([15, 25, 35]) 

# Labelling tickes of y-axis 

gnt.set_yticklabels(['1', '2', '3']) 



# Setting graph attribute 

gnt.grid(True) 



# Declaring a bar in schedule 

gnt.broken_barh([(40, 50)], (30, 9), facecolors =('tab:orange')) 



# Declaring multiple bars in at same level and same width 

gnt.broken_barh([(110, 10), (150, 10)], (10, 9), 

						facecolors ='tab:blue') 



gnt.broken_barh([(10, 50), (100, 20), (130, 10)], (20, 9), 

								facecolors =('tab:red')) 

plt.title("Gantt Chart for task sheduling in CPU", size=20)



plt.show()

# libraries required

import seaborn as sns



# reading and cleaning data

diamonds = pd.read_csv('/kaggle/input/diamonds/diamonds.csv')

diamonds = diamonds.drop(columns=['Unnamed: 0'])



# defining plot size

plt.figure(figsize=(14,12))



# plotting the heatmap

sns.heatmap(diamonds.corr(), linewidth=0.2, cmap="YlGnBu", annot=True)

plt.tick_params(labelsize=15)

plt.show() 
# libraries required

import matplotlib.pyplot as plt



# defining figuresize and theme for the plot

plt.rcParams['figure.figsize'] = (18, 8)

plt.style.use('bmh')



# plotting the bar graph

plt.bar([1,3,5,7,9],[5,2,7,8,2], label="Example one")

plt.bar([2,4,6,8,10],[8,6,2,5,6], label="Example two", color='g')



# customizing the plot

plt.legend()

plt.tick_params(labelsize=15)

plt.xlabel('Bar number', size=18)

plt.ylabel('Height', size=18)

plt.title('Histogram',size=20, color='k')



plt.show()
# libraries required

import matplotlib.pyplot as plt



# defining figuresize and theme for the plot

plt.rcParams['figure.figsize'] = (15, 7)

plt.style.use('ggplot')



# using the 'diamond dataset'

diamond = diamonds.head(10)



# plotting the line plot

plt.plot(diamond['price'], diamond['carat'])



# customizing the plot

plt.tick_params(labelsize=15)

plt.xlabel('Price', size=18)

plt.ylabel('Carat', size=18)

plt.title('Diamonds Price vs Carat - Line Plot',size=20, color='k')

plt.show()
# libraries required

from statsmodels.graphics.mosaicplot import mosaic

import matplotlib.pyplot as plt

import pandas



# defining figuresize and theme for the plot

plt.rcParams['figure.figsize'] = (15, 12)

plt.style.use('ggplot')



# creating the dataframe

gender = ['male', 'male', 'male', 'female', 'female', 'female']

pet = ['cat', 'dog', 'dog', 'cat', 'dog', 'cat']

data = pandas.DataFrame({'gender': gender, 'pet': pet})



# plotting the data

mosaic(data, ['pet', 'gender'])



plt.title('MOSAIC Plot / Marimekko Chart',size=20)



plt.show()
# libraries required

import numpy as np

import matplotlib.pyplot as plt



# defining figuresize and theme for the plot

plt.rcParams['figure.figsize'] = (15, 8)

plt.style.use('grayscale')

 

# set width of bar

barWidth = 0.25

 

# set height of bar

bars1 = [12, 30, 1, 8, 22]

bars2 = [28, 6, 16, 5, 10]

bars3 = [29, 3, 24, 25, 17]

 

# Set position of bar on X axis

r1 = np.arange(len(bars1))

r2 = [x + barWidth for x in r1]

r3 = [x + barWidth for x in r2]

 

# Make the plot

plt.bar(r1, bars1, color='#7f6d5f', width=barWidth, edgecolor='white', label='var1')

plt.bar(r2, bars2, color='#557f2d', width=barWidth, edgecolor='white', label='var2')

plt.bar(r3, bars3, color='#2d7f5e', width=barWidth, edgecolor='white', label='var3')

 

# Add xticks on the middle of the group bars

plt.tick_params(labelsize=15)

plt.xlabel('Group', fontweight='bold',size=18)

plt.xticks([r + barWidth for r in range(len(bars1))], ['A', 'B', 'C', 'D', 'E'])

plt.title('Grouped Bar Plot, size=20')

 

# Create legend & Show graphic

plt.legend()

plt.show()

# libraries required

import matplotlib.pyplot as plt



# defining figuresize for the plot

plt.rcParams['figure.figsize'] = (18, 10)



# Pie chart, where the slices will be ordered and plotted counter-clockwise:

labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'

sizes = [15, 30, 45, 10]

explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')



fig1, ax1 = plt.subplots()

ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()
# libraries required

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



# defining figuresize and theme for the plot

plt.rcParams['figure.figsize'] = (18, 12)

plt.style.use('dark_background')



# creating data for plot

df = pd.DataFrame({'Age': ['0-4','5-9','10-14','15-19','20-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-64','65-69','70-74','75-79','80-84','85-89','90-94','95-99','100+'], 

                    'Male': [-49228000, -61283000, -64391000, -52437000, -42955000, -44667000, -31570000, -23887000, -22390000, -20971000, -17685000, -15450000, -13932000, -11020000, -7611000, -4653000, -1952000, -625000, -116000, -14000, -1000], 

                    'Female': [52367000, 64959000, 67161000, 55388000, 45448000, 47129000, 33436000, 26710000, 25627000, 23612000, 20075000, 16368000, 14220000, 10125000, 5984000, 3131000, 1151000, 312000, 49000, 4000, 0]})





AgeClass = ['100+','95-99','90-94','85-89','80-84','75-79','70-74','65-69','60-64','55-59','50-54','45-49','40-44','35-39','30-34','25-29','20-24','15-19','10-14','5-9','0-4']



# creating the population pyramid

bar_plot = sns.barplot(x='Male', y='Age', data=df, order=AgeClass)

bar_plot = sns.barplot(x='Female', y='Age', data=df, order=AgeClass)



# customizing the plot

plt.tick_params(labelsize=15)

bar_plot.set(xlabel="Population (hundreds of millions)", ylabel="Age-Group", title = "Population Pyramid")

plt.show()
# Libraries required

import matplotlib.pyplot as plt

import pandas as pd

from math import pi



# defining figuresize and theme for the plot

plt.rcParams['figure.figsize'] = (18, 12)

plt.style.use('ggplot')

 

# DataFrame for plot

df = pd.DataFrame({

'group': ['A','B','C','D'],

'var1': [38, 1.5, 30, 4],

'var2': [29, 10, 9, 34],

'var3': [8, 39, 23, 24],

'var4': [7, 31, 33, 14],

'var5': [28, 15, 32, 14]

})

 

# number of variable

categories=list(df)[1:]

N = len(categories)

 

# We are going to plot the first line of the data frame.

# But we need to repeat the first value to close the circular graph:

values=df.loc[0].drop('group').values.flatten().tolist()

values += values[:1]

values

 

# What will be the angle of each axis in the plot? (we divide the plot / number of variable)

angles = [n / float(N) * 2 * pi for n in range(N)]

angles += angles[:1]

 

# Initialise the spider plot

ax = plt.subplot(111, polar=True)

 

# Draw one axe per variable + add labels labels yet

plt.xticks(angles[:-1], categories, color='grey', size=8)

 

# Draw ylabels

ax.set_rlabel_position(0)

plt.yticks([10,20,30], ["10","20","30"], color="grey", size=7)

plt.ylim(0,40)





plt.tick_params(labelsize=15)

plt.title('Radar Chart',size=20)



# Plot data

ax.plot(angles, values, linewidth=1, linestyle='solid')

 

# Fill area

ax.fill(angles, values, 'b', alpha=0.1)



plt.show()
# libraries required

import matplotlib.pyplot as plt

from matplotlib import cm

from math import log10



# defining figuresize and theme for the plot

plt.rcParams['figure.figsize'] = (18, 12)

plt.style.use('ggplot')



labels = list("ABCDEFG")

data = [21, 57, 88, 14, 76, 91, 26]

#number of data points

n = len(data)

#find max value for full ring

k = 10 ** int(log10(max(data)))

m = k * (1 + max(data) // k)



#radius of donut chart

r = 1.5

#calculate width of each ring

w = r / n 



#create colors along a chosen colormap

colors = [cm.terrain(i / n) for i in range(n)]



#create figure, axis

fig, ax = plt.subplots()

ax.axis("equal")



#create rings of donut chart

for i in range(n):

    #hide labels in segments with textprops: alpha = 0 - transparent, alpha = 1 - visible

    innerring, _ = ax.pie([m - data[i], data[i]], radius = r - i * w, startangle = 90, labels = ["", labels[i]], labeldistance = 1 - 1 / (1.5 * (n - i)), textprops = {"alpha": 0}, colors = ["white", colors[i]])

    plt.setp(innerring, width = w, edgecolor = "white")



plt.legend()

plt.show()
# libraries required

import matplotlib.pyplot as plt 



# defining figuresize and theme for the plot

plt.rcParams['figure.figsize'] = (15, 10)

plt.style.use('ggplot')



# creating data points

x =[5, 7, 8, 7, 2, 17, 2, 9, 

    4, 11, 12, 9, 6]  

y =[99, 86, 87, 88, 100, 86,  

    103, 87, 94, 78, 77, 85, 86] 



# creating the plot

plt.scatter(x, y, c ="blue") 



plt.tick_params(labelsize=15)

# To show the plot 

plt.show()
# libraries required

import matplotlib.pyplot as plt

import seaborn as sns



# defining figuresize and theme for the plot

plt.rcParams['figure.figsize'] = (20, 9)

plt.style.use('ggplot')

 

# Data

x=range(1,6)

y=[ [1,4,6,8,9], [2,2,7,10,12], [2,8,5,10,6] ]

 

# Plot

plt.stackplot(x,y, labels=['A','B','C'])

plt.legend(loc='upper left')

plt.show()
# libraries required

import numpy as np

import matplotlib.pyplot as plt

from matplotlib import rc

import pandas as pd



# defining figuresize and theme for the plot

plt.rcParams['figure.figsize'] = (15, 10)

plt.style.use('seaborn-pastel')

 

# Data

r = [0,1,2,3,4]

raw_data = {'greenBars': [20, 1.5, 7, 10, 5], 'orangeBars': [5, 15, 5, 10, 15],'blueBars': [2, 15, 18, 5, 10]}

df = pd.DataFrame(raw_data)

 

# From raw value to percentage

totals = [i+j+k for i,j,k in zip(df['greenBars'], df['orangeBars'], df['blueBars'])]

greenBars = [i / j * 100 for i,j in zip(df['greenBars'], totals)]

orangeBars = [i / j * 100 for i,j in zip(df['orangeBars'], totals)]

blueBars = [i / j * 100 for i,j in zip(df['blueBars'], totals)]

 

# plot

barWidth = 0.85

names = ('A','B','C','D','E')

# Create green Bars

plt.bar(r, greenBars, color='#b5ffb9', edgecolor='white', width=barWidth)

# Create orange Bars

plt.bar(r, orangeBars, bottom=greenBars, color='#f9bc86', edgecolor='white', width=barWidth)

# Create blue Bars

plt.bar(r, blueBars, bottom=[i+j for i,j in zip(greenBars, orangeBars)], color='#a3acff', edgecolor='white', width=barWidth)

 

# Custom x axis

plt.xticks(r, names)

plt.xlabel("group")

 

# Show graphic

plt.show()

#libraries required

import matplotlib.pyplot as plt

import squarify # !pip install squarify (algorithm for treemap)



# defining figuresize

plt.rcParams['figure.figsize'] = (15, 10)



# plotting the figure

squarify.plot(sizes=[13,22,35,5], label=["group A", "group B", "group C", "group D"], color=["red","green","blue", "grey"], alpha=.4)

plt.axis('off')

plt.show()
# libraries required

import matplotlib.pyplot as plt

from matplotlib_venn import venn2

 

# First way to call the 2 group Venn diagram:

venn2(subsets = (10, 5, 2), set_labels = ('Group A', 'Group B'))

plt.show()
# libraries required

import matplotlib.pyplot as plt



# defining figuresize and theme for the plot

plt.rcParams['figure.figsize'] = (18, 12)

plt.style.use('Solarize_Light2')



# creating the data

np.random.seed(10)

collectn_1 = np.random.normal(100, 10, 200)

collectn_2 = np.random.normal(80, 30, 200)

collectn_3 = np.random.normal(90, 20, 200)

collectn_4 = np.random.normal(70, 25, 200)



# combine these different collections into a list

data_to_plot = [collectn_1, collectn_2, collectn_3, collectn_4]



# Create a figure instance

fig = plt.figure()



# Create an axes instance

ax = fig.add_axes([0,0,1,1])



# Create the boxplot

bp = ax.violinplot(data_to_plot)

plt.tick_params(labelsize=15)

plt.show()
# libraries required

from wordcloud import WordCloud



# defining figuresize and theme for the plot

plt.rcParams['figure.figsize'] = (15, 8)

plt.style.use('fast')



# create the word cloud

wc = WordCloud(background_color = 'white', width = 1500, height = 800).generate(str(crimedata['Descript']))



plt.imshow(wc)

plt.axis('off')

plt.show()