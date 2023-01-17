!pip install autoviz





import numpy as np

import pandas as pd



import missingno as msno

import matplotlib.pyplot as plt

import seaborn as sns



from plotly.offline import iplot

import plotly.express as px



from wordcloud import WordCloud

from PIL import Image

import requests

from io import BytesIO

import textwrap



from autoviz.AutoViz_Class import AutoViz_Class

from pandas_profiling import ProfileReport



import warnings

warnings.filterwarnings("ignore") 



sns.set_style('dark')

%matplotlib inline
data = pd.read_csv('../input/videogamesales/vgsales.csv', sep = ',')

data.sample(10)
print("Shape: ", data.shape)

msno.matrix(data, labels = True)

plt.show()
print("Count of null values:-\n", data.isnull().sum())
new_data = data.dropna(axis = 0, inplace = False)

new_data = new_data.drop(data[data['Year'] >= 2017].index)
print("Data types:-\n", new_data.dtypes)

print("Shape: ", new_data.shape)

new_data['Year'] = new_data['Year'].astype('int')
new_data.describe().transpose()
response = requests.get('https://mpng.subpng.com/20180915/jzo/kisspng-computer-icons-bird-hunt-lite-game-controllers-sca-5b9d77bf746252.7340208815370464634767.jpg')

image = Image.open(BytesIO(response.content))

crop_image = image.crop(box = (180, 90, 720, 425))  

controller_mask = np.array(crop_image)



for i in range(len(controller_mask)):

    for j in range(len(controller_mask[i])):

        for k in range(3):

            if controller_mask[i][j][k] <= 50:

                controller_mask[i][j][k] = 0

            else:

                controller_mask[i][j][k] = 255



wordcloud_data = pd.concat([new_data.groupby('Platform').sum(), 

                            new_data.groupby('Genre').sum()])

wordcloud_data = wordcloud_data['Global_Sales'].sort_values(ascending = False).index



plt.figure(figsize = (15, 8))

wordcloud = WordCloud(background_color = 'black', repeat = True, 

                      max_words = 100, mask = controller_mask, 

                      contour_color = 'white', contour_width = 3, 

                      min_font_size = 10, 

                      max_font_size = 150).generate(' '.join(wordcloud_data))



plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout()

plt.show()
year_data = new_data.groupby(by = 'Year').count()['Rank']

plt.figure(figsize = (15, 8))

plt.plot(year_data.index, year_data.values, marker = 'o', 

         markerfacecolor = 'red', linewidth = 2)

plt.title('Games Released in Each Year')

plt.xlabel('Year')

plt.ylabel('Count')

plt.grid(True)

plt.show()
def count_plot(column_name, x_text):

    plt.style.use('fivethirtyeight')

    color = ['#4878d0', '#ee854a', '#6acc64', '#d65f5f', '#956cb4', 

             '#8c613c', '#dc7ec0', '#797979', '#d5bb67', '#82c6e2']

    column_data = (new_data.groupby(column_name).count()['Rank']).sort_values(axis = 0)

    

    column_plot = plt.barh(y = column_data.index, width = column_data.values,

                           color = color)

    for bar in column_plot:

        plt.text(bar.get_height() + bar.get_width() + x_text, bar.get_y() + 0.5, 

                 bar.get_width(), horizontalalignment = 'center')

    axes_subplot = plt.subplot()

    axes_subplot.invert_yaxis()

    plt.yticks(column_data.index)

    plt.xticks([])

    plt.title('No. of Games Released On Each %s'%(column_name))

    plt.ylabel(column_name)

    plt.xlabel('No. of games')

    plt.grid(False)
plt.figure(figsize = (15, 15))

count_plot('Platform', 50)

plt.show()
plt.figure(figsize = (13, 7))

count_plot('Genre', 85)

plt.show()
publisher_data = pd.DataFrame(data = new_data.groupby(by = 'Publisher').count()['Rank'])

publisher_data = publisher_data.sort_values(by = 'Rank', ascending = False)

publisher_data = publisher_data.rename(columns = {'Rank' : 'Count'}).head(n = 10)



plt.figure(figsize = (15, 7))

publisher_globalsale_plot = sns.barplot(x = publisher_data.index, 

                                        y = publisher_data.Count, 

                                        edgecolor = 'black', 

                                        linewidth = 2, 

                                        palette = 'colorblind')

publisher_globalsale_plot.set_xticklabels(textwrap.fill(name.get_text(), 

                                                        width = 10) for name in publisher_globalsale_plot.get_xticklabels())

publisher_globalsale_plot.set_xticklabels(publisher_globalsale_plot.get_xticklabels(),

                                          fontdict = {'fontsize' : 12, 'color' : 'black'})

plt.xlabel('Publisher')

plt.ylabel('Sales in Millions')

plt.title('Top 10 Publisher by Global Sales')

plt.grid(False)

plt.show()
year_sales_data = new_data.groupby(by = 'Year')['NA_Sales', 'EU_Sales', 

                                                'JP_Sales', 'Other_Sales', 'Global_Sales'].sum()

plt.figure(figsize = (15, 8))

sns.lineplot(x = year_sales_data.index, 

             y = year_sales_data.iloc[:, 4], 

             color = 'purple', 

             linewidth = 3)

plt.title('Gloabal Sales in Millions')

plt.xlabel('Year')

plt.ylabel('Amount in Millions')

plt.grid(False)

plt.show()
plt.figure(figsize = (15, 8))

sns.lineplot(x = year_sales_data.index, y = year_sales_data.iloc[:, 0], label = 'North America')

sns.lineplot(x = year_sales_data.index, y = year_sales_data.iloc[:, 1], label = 'Europe')

sns.lineplot(x = year_sales_data.index, y = year_sales_data.iloc[:, 2], label = 'Japan')

sns.lineplot(x = year_sales_data.index, y = year_sales_data.iloc[:, 3], label = 'Other')

plt.title('Sales in Each Year')

plt.xlabel('Year')

plt.ylabel('Amount in Millions')

plt.legend(loc = 2, ncol = 1, shadow = True, fancybox = True, frameon = True)

plt.grid(False)

plt.show()
def first_plot(column_name):

    first_column_data = new_data.groupby(by = column_name)['Year'].min()

    first_column_data = first_column_data.sort_values(axis = 0)

    first_column_plot = sns.stripplot(x = first_column_data.index, 

                                      y = first_column_data.values, palette = 'bright')

    first_column_plot.set_xticklabels(first_column_data.index, 

                                      rotation=45, horizontalalignment='right')

    plt.title('First Game Reales on %s' %(column_name))

    plt.xlabel(column_name)

    plt.ylabel('Year')

    plt.grid(True)

    plt.show() 
plt.figure(figsize = (15, 8))

first_plot('Platform')
plt.figure(figsize = (15, 6))

first_plot('Genre')
def type_sales_pie(column_name, sale_name):

    type_sales_data = new_data.groupby(by = column_name)['NA_Sales', 'EU_Sales',

                                                         'JP_Sales', 'Global_Sales']

    type_sales_data = type_sales_data.sum().sort_values(by = 'Global_Sales', 

                                                        ascending = False)

    type_sales_pie = {'data': [{'values': type_sales_data[sale_name], 

                                'labels': type_sales_data.index, 

                                'domain': {'x': [0, .5]}, 

                                'hoverinfo': 'label + percent', 

                                'hole': 0.2, 

                                'type': 'pie'},],

                      'layout': {'title': '%s sales by %s' %(sale_name, column_name), 

                                 'annotations': [{

                                     'font': { 'size': 15}, 

                                     'showarrow': False, 

                                     'text': 'Amount in Millions', 

                                     'x': 0.1, 

                                     'y': 1.1},]}}



    iplot(figure_or_data = type_sales_pie)
type_sales_pie('Genre', 'NA_Sales')
type_sales_pie('Genre', 'EU_Sales')
type_sales_pie('Genre', 'JP_Sales') 
type_sales_pie('Genre', 'Global_Sales')
type_sales_pie('Platform', 'NA_Sales')
type_sales_pie('Platform', 'EU_Sales')
type_sales_pie('Platform', 'JP_Sales')
type_sales_pie('Platform', 'Global_Sales')
no_sales_platform = new_data.groupby(by = 'Platform')['NA_Sales', 'EU_Sales', 

                                                      'JP_Sales', 'Other_Sales', 'Global_Sales']

no_sales_platform = no_sales_platform.sum().sort_values(by = 'Global_Sales', 

                                                        ascending = True).head(n = 10)

no_sales_platform
platform_genre_data = new_data.groupby(by = ['Platform', 'Genre']).size()

platform_genre_data = platform_genre_data.reset_index().rename(columns = {0: "Count"})



fig = px.scatter(platform_genre_data, x = 'Platform', y = 'Genre', 

                 size = 'Count', color = 'Count')

fig.show()
def globalsale_data(column_name):

    column_globalsale_data = new_data.groupby(by = column_name)['Global_Sales']

    column_globalsale_data = column_globalsale_data.sum().sort_values(ascending = False).head(n = 10)

    

    plt.figure(figsize = (15, 7))

    name_globalsale_plot = sns.barplot(x = column_globalsale_data.index, 

                                       y = column_globalsale_data.values, 

                                       edgecolor = 'black', 

                                       linewidth = 2, 

                                       palette = 'colorblind')

    name_globalsale_plot.set_xticklabels(textwrap.fill(name.get_text(), width = 10) for name in name_globalsale_plot.get_xticklabels())

    name_globalsale_plot.set_xticklabels(name_globalsale_plot.get_xticklabels(), 

                                         fontdict = {'fontsize' : 12, 'color' : 'black'})

    plt.ylabel('Sales in Millions')

    plt.grid(False)
globalsale_data('Name')

plt.title('Top 10 Games by Global Sales')

plt.xlabel('Game')

plt.show()
globalsale_data('Publisher')

plt.title('Top 10 Publisher by Global Sales')

plt.xlabel('Publisher')

plt.show()
name_count_data = pd.pivot_table(new_data, index = ['Name'],

                                 aggfunc = {'Name': 'count', 'Global_Sales' : np.sum})

name_count_data = name_count_data.rename(columns = {'Name' : 'Count'})

name_count_data = name_count_data.sort_values(by = ['Count', 'Global_Sales'], 

                                              ascending = False)['Count'].head(n = 10)



plt.figure(figsize = (15, 7))

name_globalsale_plot = sns.barplot(x = name_count_data.index, 

                                   y = name_count_data.values, 

                                   edgecolor = 'black', 

                                   linewidth = 2, 

                                   palette = 'colorblind')

name_globalsale_plot.set_xticklabels(textwrap.fill(name.get_text(), width = 10) for name in name_globalsale_plot.get_xticklabels())

name_globalsale_plot.set_xticklabels(name_globalsale_plot.get_xticklabels(), 

                                     fontdict = {'fontsize' : 12, 'color' : 'black'})

plt.xlabel('Game')

plt.ylabel('Count')

plt.title('Top 10 Games released on different platforms')

plt.grid(False)

plt.show()
plt.figure(figsize = (8, 8))

cmap = sns.diverging_palette(250, 230, 90, 60, as_cmap=True)

sns.heatmap(data = new_data.corr(), annot = True, cmap = cmap, 

            linewidths = 2, square = True, 

            linecolor = 'White', fmt = '.2f')

plt.title('Correlation \n', fontdict = {'fontsize': 20,  

                                        'fontweight' : 'normal',

                                        'color' : 'black'})

plt.show()
ProfileReport(data)
AV = AutoViz_Class()

df = AV.AutoViz(filename = '../input/videogamesales/vgsales.csv', 

                sep = ',', chart_format = 'svg', verbose = 2)
from IPython.display import HTML

style = "<style> div.changeh { background-color: #FF5733; width: 350px; height: 40px;} </style>"

HTML(style)