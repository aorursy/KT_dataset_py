# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from __future__ import print_function

import matplotlib

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #plot graphs to analyze the data

import csv

from PIL import Image # converting images into arrays

from matplotlib.colors import LinearSegmentedColormap

import matplotlib.patches as mpatches

from wordcloud import WordCloud, STOPWORDS

import seaborn as sns

import folium



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output
df_crime_report=pd.read_csv("../input/Crime_Data_from_2010_to_Present.csv")

print("Data Loaded")
df_crime_report.head()
df_crime_report.shape
df_crime_report.columns = list(map(str, df_crime_report.columns))



# let's check the column labels types now

all(isinstance(column, str) for column in df_crime_report.columns)
df_crime_report_area=df_crime_report.set_index('Area Name')



# let's view the first five elements and see how the dataframe was changed

df_crime_report_area.head()
df_crime_report["Area Name"].drop_duplicates()
criminal_in_areas = np.zeros(21)

criminal_in_areas
arearows=["77th Street",

"Central",

"Devonshire",

"Foothill",

"Harbor",

"Hollenbeck",

"HollyWood",

"Mission",

"N HollyWood",

"Newton",

"Northeast",

"Olympic",

"Pacific",

"Rampart",

"Southeast",

"SouthWest",

"Topanga",

"Van Nuys",

"West LA",

"West Valley",

"Wilshire",]
df_areanames=df_crime_report["Area Name"]

df_areanames.head()
for index,item in enumerate(df_areanames):

    if item=="77th Street":

        criminal_in_areas[0]+=1

    elif item=="Central":

        criminal_in_areas[1]+=1

    elif item=="Devonshire":

        criminal_in_areas[2]+=1

    elif item=="Foothill":

        criminal_in_areas[3]+=1

    elif item=="Harbor":

        criminal_in_areas[4]+=1

    elif item=="Hollenbeck":

        criminal_in_areas[5]+=1

    elif item=="Hollywood":

        criminal_in_areas[6]+=1

    elif item=="Mission":

        criminal_in_areas[7]+=1

    elif item=="N Hollywood":

        criminal_in_areas[8]+=1

    elif item=="Newton":

        criminal_in_areas[9]+=1

    elif item=="Northeast":

        criminal_in_areas[10]+=1

    elif item=="Olympic":

        criminal_in_areas[11]+=1

    elif item=="Pacific":

        criminal_in_areas[12]+=1

    elif item=="Rampart":

        criminal_in_areas[13]+=1

    elif item=="Southeast":

        criminal_in_areas[14]+=1

    elif item=="Southwest":

        criminal_in_areas[15]+=1

    elif item=="Topanga":

        criminal_in_areas[16]+=1

    elif item=="Van Nuys":

        criminal_in_areas[17]+=1

    elif item=="West LA":

        criminal_in_areas[18]+=1

    elif item=="West Valley":

        criminal_in_areas[19]+=1

    elif item=="Wilshire":

        criminal_in_areas[20]+=1

    
df_crime_counter=pd.DataFrame(criminal_in_areas,index=arearows)

df_crime_counter
df_crime_counter.plot(kind='Bar',stacked=False,figsize=(15,10))

plt.title('Crime Numbers in Areas')

plt.ylabel('Number Crimes')

plt.xlabel('Areas')



plt.show()
df_crime_report["Victim Age"].fillna(df_crime_report["Victim Age"].mean(),inplace=True)
df_mean_by_Area=df_crime_report.groupby(["Area Name"]).mean()

df_mean_by_Area
df_mean_by_Area["Victim Age"].plot(kind="Bar",stacked=False,figsize=(15,10))

plt.title('Victims Average Ages in Different Areas')

plt.ylabel('Age')

plt.xlabel('Areas')



plt.show()
explode_list = [0.1, 0, 0.1, 0, 0.1, 0,0.1, 0, 0.1, 0, 0.1, 0,0.1, 0, 0.1, 0, 0.1, 0,0.1, 0, 0.1]

df_crime_counter.plot(kind="pie",figsize=(15,10),autopct='%1.1f%%',startangle=90, shadow=True,subplots=True,explode=explode_list)
crime_in_years=np.zeros(10)

crime_in_years
for item,index in enumerate(df_crime_report["Date Occurred"]):

    if "2010" in index:

        crime_in_years[0]+=1

    elif "2011" in index:

        crime_in_years[1]+=1

    elif "2012" in index:

        crime_in_years[2]+=1

    elif "2013" in index:

        crime_in_years[3]+=1

    elif "2014" in index:

        crime_in_years[4]+=1

    elif "2015" in index:

        crime_in_years[5]+=1

    elif "2016" in index:

        crime_in_years[6]+=1

    elif "2017" in index:

        crime_in_years[7]+=1

    elif "2018" in index:

        crime_in_years[8]+=1

    elif "2019" in index:

        crime_in_years[9]+=1

    
yearscolumn=["2010","2011","2012","2013","2014","2015","2016","2017","2018","2019"]
crime_in_years=pd.DataFrame(crime_in_years,index=yearscolumn)

crime_in_years
crime_in_years.plot(kind="line",figsize=(10,5),stacked=False)

plt.title("Crime numbers in Years")

plt.xlabel("Years")

plt.ylabel("Crime Numbers")



plt.show()
df_dene=np.zeros((21,10))


i=0

for area,date in zip(df_crime_report["Area Name"],df_crime_report["Date Occurred"]):

  

    if "2010" in date:

        if area=="77th Street":

            df_dene[0][0]+=1

        elif area=="Central":

            df_dene[1][0]+=1

        elif  area=="Devonshire":

            df_dene[2][0]+=1

        elif  area=="Foothill":

            df_dene[3][0]+=1

        elif  area=="Harbor":

            df_dene[4][0]+=1

        elif  area=="Hollenbeck":

            df_dene[5][0]+=1

        elif  area=="Hollywood":

            df_dene[6][0]+=1

        elif  area=="Newton":

            df_dene[7][0]+=1

        elif  area=="Mission":

            df_dene[8][0]+=1

        elif  area=="N Hollywood":

            df_dene[9][0]+=1

        elif  area=="Northeast":

            df_dene[10][0]+=1

        elif  area=="Olympic":

            df_dene[11][0]+=1

        elif  area=="Pacific":

            df_dene[12][0]+=1

        elif  area=="Rampart":

            df_dene[13][0]+=1

        elif  area=="Southeast":

            df_dene[14][0]+=1

        elif  area=="Southwest":

            df_dene[15][0]+=1

        elif  area=="Topanga":

            df_dene[16][0]+=1

        elif  area=="Van Nuys":

            df_dene[17][0]+=1

        elif  area=="West LA":

            df_dene[18][0]+=1

        elif  area=="West Valley":

            df_dene[19][0]+=1

        elif  area=="Wilshire":

            df_dene[20][0]+=1

    elif  "2011" in date:

        i=1

        if area=="77th Street":

            df_dene[0][i]+=1

        elif area=="Central":

            df_dene[1][i]+=1

        elif  area=="Devonshire":

            df_dene[2][i]+=1

        elif  area=="Foothill":

            df_dene[3][i]+=1

        elif  area=="Harbor":

            df_dene[4][i]+=1

        elif  area=="Hollenbeck":

            df_dene[5][i]+=1

        elif  area=="Hollywood":

            df_dene[6][i]+=1

        elif  area=="Newton":

            df_dene[7][i]+=1

        elif  area=="Mission":

            df_dene[8][i]+=1

        elif  area=="N Hollywood":

            df_dene[9][i]+=1

        elif  area=="Northeast":

            df_dene[10][i]+=1

        elif  area=="Olympic":

            df_dene[11][i]+=1

        elif  area=="Pacific":

            df_dene[12][i]+=1

        elif  area=="Rampart":

            df_dene[13][i]+=1

        elif  area=="Southeast":

            df_dene[14][i]+=1

        elif  area=="Southwest":

            df_dene[15][i]+=1

        elif  area=="Topanga":

            df_dene[16][i]+=1

        elif  area=="Van Nuys":

            df_dene[17][i]+=1

        elif  area=="West LA":

            df_dene[18][i]+=1

        elif  area=="West Valley":

            df_dene[19][i]+=1

        elif  area=="Wilshire":

            df_dene[20][i]+=1

    elif "2012" in date:

        i=2

        if area=="77th Street":

            df_dene[0][i]+=1

        elif area=="Central":

            df_dene[1][i]+=1

        elif  area=="Devonshire":

            df_dene[2][i]+=1

        elif  area=="Foothill":

            df_dene[3][i]+=1

        elif  area=="Harbor":

            df_dene[4][i]+=1

        elif  area=="Hollenbeck":

            df_dene[5][i]+=1

        elif  area=="Hollywood":

            df_dene[6][i]+=1

        elif  area=="Newton":

            df_dene[7][i]+=1

        elif  area=="Mission":

            df_dene[8][i]+=1

        elif  area=="N Hollywood":

            df_dene[9][i]+=1

        elif  area=="Northeast":

            df_dene[10][i]+=1

        elif  area=="Olympic":

            df_dene[11][i]+=1

        elif  area=="Pacific":

            df_dene[12][i]+=1

        elif  area=="Rampart":

            df_dene[13][i]+=1

        elif  area=="Southeast":

            df_dene[14][i]+=1

        elif  area=="Southwest":

            df_dene[15][i]+=1

        elif  area=="Topanga":

            df_dene[16][i]+=1

        elif  area=="Van Nuys":

            df_dene[17][i]+=1

        elif  area=="West LA":

            df_dene[18][i]+=1

        elif  area=="West Valley":

            df_dene[19][i]+=1

        elif  area=="Wilshire":

            df_dene[20][i]+=1

    elif "2013" in date:

        i=3

        if area=="77th Street":

            df_dene[0][i]+=1

        elif area=="Central":

            df_dene[1][i]+=1

        elif  area=="Devonshire":

            df_dene[2][i]+=1

        elif  area=="Foothill":

            df_dene[3][i]+=1

        elif  area=="Harbor":

            df_dene[4][i]+=1

        elif  area=="Hollenbeck":

            df_dene[5][i]+=1

        elif  area=="Hollywood":

            df_dene[6][i]+=1

        elif  area=="Newton":

            df_dene[7][i]+=1

        elif  area=="Mission":

            df_dene[8][i]+=1

        elif  area=="N Hollywood":

            df_dene[9][i]+=1

        elif  area=="Northeast":

            df_dene[10][i]+=1

        elif  area=="Olympic":

            df_dene[11][i]+=1

        elif  area=="Pacific":

            df_dene[12][i]+=1

        elif  area=="Rampart":

            df_dene[13][i]+=1

        elif  area=="Southeast":

            df_dene[14][i]+=1

        elif  area=="Southwest":

            df_dene[15][i]+=1

        elif  area=="Topanga":

            df_dene[16][i]+=1

        elif  area=="Van Nuys":

            df_dene[17][i]+=1

        elif  area=="West LA":

            df_dene[18][i]+=1

        elif  area=="West Valley":

            df_dene[19][i]+=1

        elif  area=="Wilshire":

            df_dene[20][i]+=1

    elif "2014" in date:

        i=4

        if area=="77th Street":

            df_dene[0][i]+=1

        elif area=="Central":

            df_dene[1][i]+=1

        elif  area=="Devonshire":

            df_dene[2][i]+=1

        elif  area=="Foothill":

            df_dene[3][i]+=1

        elif  area=="Harbor":

            df_dene[4][i]+=1

        elif  area=="Hollenbeck":

            df_dene[5][i]+=1

        elif  area=="Hollywood":

            df_dene[6][i]+=1

        elif  area=="Newton":

            df_dene[7][i]+=1

        elif  area=="Mission":

            df_dene[8][i]+=1

        elif  area=="N Hollywood":

            df_dene[9][i]+=1

        elif  area=="Northeast":

            df_dene[10][i]+=1

        elif  area=="Olympic":

            df_dene[11][i]+=1

        elif  area=="Pacific":

            df_dene[12][i]+=1

        elif  area=="Rampart":

            df_dene[13][i]+=1

        elif  area=="Southeast":

            df_dene[14][i]+=1

        elif  area=="Southwest":

            df_dene[15][i]+=1

        elif  area=="Topanga":

            df_dene[16][i]+=1

        elif  area=="Van Nuys":

            df_dene[17][i]+=1

        elif  area=="West LA":

            df_dene[18][i]+=1

        elif  area=="West Valley":

            df_dene[19][i]+=1

        elif  area=="Wilshire":

            df_dene[20][i]+=1

    elif "2015" in date:

        i=5

        if area=="77th Street":

            df_dene[0][i]+=1

        elif area=="Central":

            df_dene[1][i]+=1

        elif  area=="Devonshire":

            df_dene[2][i]+=1

        elif  area=="Foothill":

            df_dene[3][i]+=1

        elif  area=="Harbor":

            df_dene[4][i]+=1

        elif  area=="Hollenbeck":

            df_dene[5][i]+=1

        elif  area=="Hollywood":

            df_dene[6][i]+=1

        elif  area=="Newton":

            df_dene[7][i]+=1

        elif  area=="Mission":

            df_dene[8][i]+=1

        elif  area=="N Hollywood":

            df_dene[9][i]+=1

        elif  area=="Northeast":

            df_dene[10][i]+=1

        elif  area=="Olympic":

            df_dene[11][i]+=1

        elif  area=="Pacific":

            df_dene[12][i]+=1

        elif  area=="Rampart":

            df_dene[13][i]+=1

        elif  area=="Southeast":

            df_dene[14][i]+=1

        elif  area=="Southwest":

            df_dene[15][i]+=1

        elif  area=="Topanga":

            df_dene[16][i]+=1

        elif  area=="Van Nuys":

            df_dene[17][i]+=1

        elif  area=="West LA":

            df_dene[18][i]+=1

        elif  area=="West Valley":

            df_dene[19][i]+=1

        elif  area=="Wilshire":

            df_dene[20][i]+=1

    elif "2016" in date:

        i=6

        if area=="77th Street":

            df_dene[0][i]+=1

        elif area=="Central":

            df_dene[1][i]+=1

        elif  area=="Devonshire":

            df_dene[2][i]+=1

        elif  area=="Foothill":

            df_dene[3][i]+=1

        elif  area=="Harbor":

            df_dene[4][i]+=1

        elif  area=="Hollenbeck":

            df_dene[5][i]+=1

        elif  area=="Hollywood":

            df_dene[6][i]+=1

        elif  area=="Newton":

            df_dene[7][i]+=1

        elif  area=="Mission":

            df_dene[8][i]+=1

        elif  area=="N Hollywood":

            df_dene[9][i]+=1

        elif  area=="Northeast":

            df_dene[10][i]+=1

        elif  area=="Olympic":

            df_dene[11][i]+=1

        elif  area=="Pacific":

            df_dene[12][i]+=1

        elif  area=="Rampart":

            df_dene[13][i]+=1

        elif  area=="Southeast":

            df_dene[14][i]+=1

        elif  area=="Southwest":

            df_dene[15][i]+=1

        elif  area=="Topanga":

            df_dene[16][i]+=1

        elif  area=="Van Nuys":

            df_dene[17][i]+=1

        elif  area=="West LA":

            df_dene[18][i]+=1

        elif  area=="West Valley":

            df_dene[19][i]+=1

        elif  area=="Wilshire":

            df_dene[20][i]+=1

    elif "2017" in date:

        i=7

        if area=="77th Street":

            df_dene[0][i]+=1

        elif area=="Central":

            df_dene[1][i]+=1

        elif  area=="Devonshire":

            df_dene[2][i]+=1

        elif  area=="Foothill":

            df_dene[3][i]+=1

        elif  area=="Harbor":

            df_dene[4][i]+=1

        elif  area=="Hollenbeck":

            df_dene[5][i]+=1

        elif  area=="Hollywood":

            df_dene[6][i]+=1

        elif  area=="Newton":

            df_dene[7][i]+=1

        elif  area=="Mission":

            df_dene[8][i]+=1

        elif  area=="N Hollywood":

            df_dene[9][i]+=1

        elif  area=="Northeast":

            df_dene[10][i]+=1

        elif  area=="Olympic":

            df_dene[11][i]+=1

        elif  area=="Pacific":

            df_dene[12][i]+=1

        elif  area=="Rampart":

            df_dene[13][i]+=1

        elif  area=="Southeast":

            df_dene[14][i]+=1

        elif  area=="Southwest":

            df_dene[15][i]+=1

        elif  area=="Topanga":

            df_dene[16][i]+=1

        elif  area=="Van Nuys":

            df_dene[17][i]+=1

        elif  area=="West LA":

            df_dene[18][i]+=1

        elif  area=="West Valley":

            df_dene[19][i]+=1

        elif  area=="Wilshire":

            df_dene[20][i]+=1

    elif "2018" in date:

        i=8

        if area=="77th Street":

            df_dene[0][i]+=1

        elif area=="Central":

            df_dene[1][i]+=1

        elif  area=="Devonshire":

            df_dene[2][i]+=1

        elif  area=="Foothill":

            df_dene[3][i]+=1

        elif  area=="Harbor":

            df_dene[4][i]+=1

        elif  area=="Hollenbeck":

            df_dene[5][i]+=1

        elif  area=="Hollywood":

            df_dene[6][i]+=1

        elif  area=="Newton":

            df_dene[7][i]+=1

        elif  area=="Mission":

            df_dene[8][i]+=1

        elif  area=="N Hollywood":

            df_dene[9][i]+=1

        elif  area=="Northeast":

            df_dene[10][i]+=1

        elif  area=="Olympic":

            df_dene[11][i]+=1

        elif  area=="Pacific":

            df_dene[12][i]+=1

        elif  area=="Rampart":

            df_dene[13][i]+=1

        elif  area=="Southeast":

            df_dene[14][i]+=1

        elif  area=="Southwest":

            df_dene[15][i]+=1

        elif  area=="Topanga":

            df_dene[16][i]+=1

        elif  area=="Van Nuys":

            df_dene[17][i]+=1

        elif  area=="West LA":

            df_dene[18][i]+=1

        elif  area=="West Valley":

            df_dene[19][i]+=1

        elif  area=="Wilshire":

            df_dene[20][i]+=1

    elif "2019" in date:

        i=9

        if area=="77th Street":

            df_dene[0][i]+=1

        elif area=="Central":

            df_dene[1][i]+=1

        elif  area=="Devonshire":

            df_dene[2][i]+=1

        elif  area=="Foothill":

            df_dene[3][i]+=1

        elif  area=="Harbor":

            df_dene[4][i]+=1

        elif  area=="Hollenbeck":

            df_dene[5][i]+=1

        elif  area=="Hollywood":

            df_dene[6][i]+=1

        elif  area=="Newton":

            df_dene[7][i]+=1

        elif  area=="Mission":

            df_dene[8][i]+=1

        elif  area=="N Hollywood":

            df_dene[9][i]+=1

        elif  area=="Northeast":

            df_dene[10][i]+=1

        elif  area=="Olympic":

            df_dene[11][i]+=1

        elif  area=="Pacific":

            df_dene[12][i]+=1

        elif  area=="Rampart":

            df_dene[13][i]+=1

        elif  area=="Southeast":

            df_dene[14][i]+=1

        elif  area=="Southwest":

            df_dene[15][i]+=1

        elif  area=="Topanga":

            df_dene[16][i]+=1

        elif  area=="Van Nuys":

            df_dene[17][i]+=1

        elif  area=="West LA":

            df_dene[18][i]+=1

        elif  area=="West Valley":

            df_dene[19][i]+=1

        elif  area=="Wilshire":

            df_dene[20][i]+=1
df_dene
df_areas_years=pd.DataFrame(df_dene, columns=yearscolumn,index=arearows)

df_areas_years['Total'] = df_areas_years.sum (axis = 1)

years = list(map(str, range(2010, 2020)))

years
df_Newton=df_areas_years.loc[["Newton"],years].transpose()

df_Newton.describe()
df_Newton.plot(kind='box',vert=True,figsize=(5,5))

plt.title('Crimes in Newton from 2011 to 2019')

plt.ylabel('Number of Crimes')



plt.show()
df_Van_Nuys_Newton=df_areas_years.loc[["Van Nuys","Newton"],years].transpose()

df_Van_Nuys_Newton.describe()
df_Van_Nuys_Newton.plot(kind='box',vert=True,figsize=(5,5))

plt.title('Crimes in Newton and Van Nuys from 2011 to 2019')

plt.ylabel('Number of Crimes')



plt.show()
df_all_box=df_areas_years.loc[:,years].transpose()

df_all_box.describe()
df_all_box.plot(kind='box',vert=True,figsize=(20,10))

plt.title('Crimes in all areas from 2011 to 2019')

plt.ylabel('Number of Crimes')



plt.show()
df_Newton.index=df_Newton.index.astype(int)

df_Newton.index.name="year"

df_Newton.reset_index("year",inplace=True)

df_Newton
df_Newton.plot(kind="scatter",x='year',y='Newton')
df_trio=df_areas_years.loc[["Foothill","Pacific","Topanga"],years].transpose()

count, bin_edges = np.histogram(df_trio)

df_trio.head()
df_trio.plot(kind='hist',xticks=bin_edges,figsize=(8,5))

plt.title('Histogram of Crimes from Foothill,Pacific,Topanga between 2010 and 2019')

plt.ylabel('Number of Aras') 

plt.xlabel('Number of Crimes') 



plt.show()
def create_waffle_chart(df_chart,categories, values, height, width, colormap, value_sign='',):

    

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

        print (df_chart.index.values[i] + ': ' + str(tiles))

    

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

        if i==0:

            colors=[color_val]

        if i!=0:

            colors.append(color_val)

        legend_handles.append(mpatches.Patch(color=color_val, label=label_str))

    

    cmap_name = 'my_list'

    cm = LinearSegmentedColormap.from_list(

        cmap_name, colors,)

    # instantiate a new figure object

    fig = plt.figure()



    # use matshow to display the waffle chart

    plt.matshow(waffle_chart, cmap=cm)

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



    



    # add legend to chart

    plt.legend(

        handles=legend_handles,

        loc='lower center', 

        ncol=len(categories),

        bbox_to_anchor=(0., -0.2, 0.95, .1)

    )
df_trio=  df_trio.transpose()
df_trio['Total'] =  df_trio.sum (axis = 1)

df_trio
width = 40 # width of chart

height = 10 # height of chart

categories = df_trio.index.values # categories

values = df_trio['Total'] # correponding values of categories



colormap = plt.cm.coolwarm # color map class
create_waffle_chart(df_trio,categories, values, height, width, colormap)
df_quartio=df_areas_years.loc[["Foothill","Pacific","Topanga","Harbor"],years]

df_quartio['Total'] =  df_quartio.sum (axis = 1)

categories = df_quartio.index.values # categories

values = df_quartio['Total'] # correponding values of categories
create_waffle_chart(df_quartio,categories, values, height, width, colormap)
height=20

width=100

categories = df_areas_years.index.values # categories

values = df_areas_years['Total'] # correponding values of categories

create_waffle_chart(df_areas_years,categories, values, height, width, colormap)
total_crime=df_areas_years['Total'].sum()

total_crime
df_areas_years.index
df_areas_years.index = list(map(str, df_areas_years.index))



# let's check the column labels types now

all(isinstance(index, str) for column in df_areas_years.index)
max_words = 100

word_string = ''

for area in df_areas_years.index.values:

    # check if country's name is a single-word name

    if len(area.split(' ')) == 1:

        repeat_num_times = int(df_areas_years.loc[area, 'Total']/float(total_crime)*max_words)

        word_string = word_string + ((area + ' ') * repeat_num_times)

                                     

# display the generated text

word_string
# create the word cloud

wordcloud = WordCloud(background_color='white').generate(word_string)



print('Word cloud created!')
# display the cloud

fig = plt.figure()

fig.set_figwidth(14)

fig.set_figheight(18)



plt.imshow(wordcloud, interpolation='bilinear')

plt.axis('off')

plt.show()
for area in df_areas_years.index.values:

    # check if country's name is a single-word name

    if len(area.split(' ')) != 1:

        print(area)
# we can use the sum() method to get the total crime per year

df_tot = pd.DataFrame(df_areas_years[years].sum(axis=0))



# change the years to type float (useful for regression later on)

df_tot.index = map(float, df_tot.index)



# reset the index to put in back in as a column in the df_tot dataframe

df_tot.reset_index(inplace=True)



# rename columns

df_tot.columns = ['year', 'total']



# view the final dataframe

df_tot.head()
df_areas_years.index.name="area"

df_areas_years
ax = sns.regplot(x='year', y='total', data=df_tot)
df_crime_report.head()
df_location_2010=[]

df_location_2011=[]

df_location_2012=[]

df_location_2013=[]

df_location_2014=[]

df_location_2015=[]

df_location_2016=[]

df_location_2017=[]

df_location_2018=[]

df_location_2019=[]

df_location_2010_crime_code=list()

df_location_2011_crime_code=list()

df_location_2012_crime_code=list()

df_location_2013_crime_code=list()

df_location_2014_crime_code=list()

df_location_2015_crime_code=list()

df_location_2016_crime_code=list()

df_location_2017_crime_code=list()

df_location_2018_crime_code=list()

df_location_2019_crime_code=list()


for location,index,category in zip(df_crime_report["Location "],df_crime_report["Date Occurred"],df_crime_report["Crime Code Description"]):

    

    if "2010" in index:

        df_location_2010.extend(location.replace('(',' ').replace(')',' ').split(',',1))

        df_location_2010_crime_code.append(category)

    elif "2011" in index:

        df_location_2011.extend(location.replace('(',' ').replace(')',' ').split(',',1))

        df_location_2010_crime_code.append(category)

    elif "2012" in index:

        df_location_2012.extend(location.replace('(',' ').replace(')',' ').split(',',1))

        df_location_2010_crime_code.append(category)

    elif "2013" in index:

        df_location_2013.extend(location.replace('(',' ').replace(')',' ').split(',',1))

        df_location_2010_crime_code.append(category)

    elif "2014" in index:

        df_location_2014.extend(location.replace('(',' ').replace(')',' ').split(',',1))

        df_location_2010_crime_code.append(category)

    elif "2015" in index:

        df_location_2015.extend(location.replace('(',' ').replace(')',' ').split(',',1))

        df_location_2010_crime_code.append(category)

    elif "2016" in index:

        df_location_2016.extend(location.replace('(',' ').replace(')',' ').split(',',1))

        df_location_2010_crime_code.append(category)

    elif "2017" in index:

        df_location_2017.extend(location.replace('(',' ').replace(')',' ').split(',',1))

        df_location_2010_crime_code.append(category)

    elif "2018" in index:

        df_location_2018.extend(location.replace('(',' ').replace(')',' ').split(',',1))

        df_location_2010_crime_code.append(category)

    elif "2019" in index:

        df_location_2019.extend(location.replace('(',' ').replace(')',' ').split(',',1))

        df_location_2010_crime_code.append(category)

    

#df_location=df_crime_report["Location "].str.replace('(',' ').str.replace(')',' ').str.split(",", n = 1, expand = True) 
print(len(df_location_2010)==len(df_location_2011))

df_location_2010_x=[]

df_location_2010_y=[]

i=0
for element in df_location_2010:

    if i==0:

        df_location_2010_y.append(element)

        i=1

    elif i==1:

        df_location_2010_x.append(element)

        i=0
location_columns=["x","y","category"]

df_locations_2010=pd.DataFrame(list(zip(df_location_2010_x,df_location_2010_y,df_location_2010_crime_code)),columns=location_columns)
df_locations_2010.head()
df_location_1000=df_locations_2010.head(2000)
 # define usa's geolocation coordinates

usa_latitude =   34.052235

usa_longitude = -118.243683



 # define the world map centered around Usa with a higher zoom level

usa_map = folium.Map(location=[usa_latitude, usa_longitude], zoom_start=10)





 # display world map

usa_map
# instantiate a feature group for the incidents in the dataframe

incidents = folium.map.FeatureGroup()



# loop through the 100 crimes and add each to the incidents feature group

for lat, lng, in zip(df_location_1000.y, df_location_1000.x):

    incidents.add_child(

        folium.CircleMarker(

            [lat, lng],

            radius=5, # define how big you want the circle markers to be

            color='yellow',

            fill=True,

            fill_color='blue',

            fill_opacity=0.6

        )

    )



# add incidents to map

usa_map.add_child(incidents)
from folium import plugins



# let's start again with a clean copy of the map of San Francisco

usa_map = folium.Map(location = [usa_latitude, usa_longitude], zoom_start = 10)



# instantiate a mark cluster object for the incidents in the dataframe

incidents = plugins.MarkerCluster().add_to(usa_map)



# loop through the dataframe and add each data point to the mark cluster

for lat, lng, label, in zip(df_location_1000.y, df_location_1000.x, df_location_1000.category):

    folium.Marker(

        location=[lat, lng],

        icon=None,

        popup=label,

    ).add_to(incidents)



# display map

usa_map