# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import bokeh.io

from bokeh.charts import Donut, HeatMap, Histogram, Line, Scatter, show, output_notebook, output_file

from bokeh.plotting import figure



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
bokeh.io.output_notebook()
#read from the csv file

import codecs

with codecs.open("../input/albumlist.csv", "r", "ASCII", "ignore") as file:

    album_list = pd.read_table(file, delimiter=",")

album_list.head()
album_list.dtypes
#add Genre_Refined and Subgenre_Refined

for lab, row in album_list.iterrows():

    album_list.loc[lab, "Genre_Refined"] = row["Genre"].split(',')[0]

    album_list.loc[lab, "Subgenre_Refined"] = row["Subgenre"].split(',')[0]
album_list.head()
album_list.dtypes
#get a count of how many times each artist made it into the list

artists_count = album_list.groupby(['Artist'], as_index=False).count()
#find the top 10 artists

top_artists = artists_count.sort_values(by='Number', ascending=False).head(10)

top_artists = top_artists.reset_index().drop(['index', 'Album','Year','Genre','Subgenre', 'Genre_Refined', 'Subgenre_Refined'], axis=1)

top_artists.head(10)
#get the artists and corresponding counts into two lists for plotting

top_artists_list = top_artists.Artist.values.tolist()

top_artists_count = top_artists.Number.astype(float).values.tolist()
#visualize the data using bokeh

#output_file("top_artists.html", title="top artists")

p = figure(x_range=top_artists_list, plot_height = 500, plot_width = 500)



#set x-axis properties

p.xgrid.visible = False

p.xaxis.major_label_orientation = 3.14/4

p.xaxis.axis_label = 'Artist Name'



#set y-axis properties

p.ygrid.visible = False

p.yaxis.axis_label = 'Album Count'



#draw circles

p.circle(y=top_artists_count, x=top_artists_list, size=15, fill_color="black")

show(p)
#get count of albums in each year

yearwise_albums = album_list.groupby(['Year'], as_index=False).count()

yearwise_albums = yearwise_albums.sort_values(by='Year').reset_index().drop(['index', 'Album', 'Artist','Genre','Subgenre','Subgenre_Refined','Genre_Refined'], axis=1)

yearwise_albums.head(5)
#visulaizing the data using bokeh line graphs

#output_file("yearwise_albums.html", title="yearwise_albums")

line = Line(data=yearwise_albums, x='Year', y='Number')

line.yaxis.axis_label = 'Number of Albums'

show(line)
#pivot the data and get a subset of the pivoted data where each subgenre has a count of more than 5

pivoted = pd.pivot_table(album_list, index=['Genre_Refined', 'Subgenre_Refined'], values=['Number'], aggfunc='count')

pivoted_subset = pivoted[pivoted['Number'] > 5]

pivoted_subset = pivoted_subset.reset_index()

pivoted_subset
#visualizing the data using the bokeh donut chart

#output_file("donut.html", title="donut")

from bokeh.palettes import Purples9 as palette1

palette1 = palette1[::-1]

d = Donut(pivoted_subset, label=['Genre_Refined', 'Subgenre_Refined'], values='Number', 

          text_font_size='10pt', plot_height=800, plot_width=800, palette=palette1)

show(d)
#getting yearwise data for each genre

yearwise_data = album_list.groupby(['Year', 'Genre_Refined'], as_index=False).count()

yearwise_data = yearwise_data.sort_values(by='Year').reset_index().drop(['index', 'Album', 'Artist','Genre','Subgenre','Subgenre_Refined'], axis=1)

yearwise_data.head(25)
#visualizing the data using a bokeh heatmap

#output_file("yearwise_genre.html", title="yearwise_subgenre")

from bokeh.palettes import Reds9 as palette2

palette2 = palette2[::-1]

hm_year = HeatMap(yearwise_data, x='Year', y='Genre_Refined', values='Number', stat=None,

               width=750, plot_height=500, palette=palette2)

#y-axis properties

hm_year.yaxis.axis_label = 'Genre'

hm_year.yaxis.major_label_orientation = 'horizontal'

show(hm_year)
#count subgenres yearwise and subset it for rock music

yearwise_subgenres = album_list.groupby(['Year', 'Genre_Refined', 'Subgenre_Refined'], as_index=False).count()

rock_subgenres_yearwise = yearwise_subgenres[yearwise_subgenres['Genre_Refined'] == 'Rock'].reset_index().drop(['index', 'Album', 'Artist','Genre','Subgenre'], axis=1)

rock_subgenres_yearwise.head()
#visualizing the data using bokeh scatterplot

#output_file("rock_subgenres_yearwise.html", title="rock_subgenres_yearwise")

hm_rock_subgenres = Scatter(rock_subgenres_yearwise, x='Year', y='Subgenre_Refined', width=800, plot_height=800)

#x-axis properties

hm_rock_subgenres.xgrid.visible = False

#y-axis properties

hm_rock_subgenres.yaxis.major_label_orientation = 'horizontal'

hm_rock_subgenres.yaxis.axis_label = 'Subgenres of Rock'

hm_rock_subgenres.ygrid.visible = False

show(hm_rock_subgenres)
#top 10 albums

top_albums = album_list.head(10)

#Get artists and albums into a new data frame

top_albums_a = top_albums['Artist']

top_albums_b = top_albums['Album']

top_albums_final = pd.concat([top_albums_a, top_albums_b], axis=1)

#groupby and summarize

top_albums_chart = top_albums_final.groupby(['Artist', 'Album']).count()

top_albums_chart