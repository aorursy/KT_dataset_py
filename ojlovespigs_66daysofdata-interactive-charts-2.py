#Importing data analytics tools 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



#importing graph interactivity tools 

from bokeh.plotting import figure, output_file, output_notebook #so it renders inline

from bokeh.io import show

from bokeh.models import ColumnDataSource #This connects pandas with bokeh

from bokeh.models.tools import HoverTool



#This is kinda so that Kaggle can work with data sets that I've added to the notebook

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

#Bringing in the work I've done previously. I worked with ramen. Let's continue doing that. 



ratings_data = "../input/ramen-ratings/ramen-ratings.csv"



ratings = pd.read_csv(ratings_data)

ratings = ratings.set_index("Review #")



#I keep calling the stars ratings so I'm going to make life easier for me

ratings = ratings.rename(columns = {"Stars": "Ratings"})



#These were ratings without numbers.

ratings = ratings.drop([2458, 2548, 1587])



#One funny thing about these ratings is that they weren't numeric values!

ratings.Ratings = pd.to_numeric(ratings.Ratings)



#Just to check everything is alright...

ratings.head()
#Let's organise the data the way we want it first for the bar chart. 

#We want to plot the average rating of ramen (regardless of type) by country. 

#I've made it slightly simpler from my previous notebook. 



ratingsc = ratings.loc[:,["Country", "Ratings"]]

ratingsc = ratingsc.sort_values(by=["Country"], ascending=True)

#ratingsc.head()



#Now for the Mean for each country. 



m_country = ratings.groupby("Country")["Ratings"].mean()

m_country = m_country.reset_index()

m_country = m_country.sort_values(by=["Ratings"], ascending=False)
plt.figure(figsize=(45,20))

plt.title("Countries versus rating")

sns.set_style("white")

sns.barplot(y=m_country["Ratings"], x=m_country["Country"])



#Now this looks a bit better. 
#Now for some interactivity 



#So that it renders 

output_notebook()



x = m_country.Country.tolist() #Country as x-axis

y = m_country.Ratings.tolist() #Country as y-axis



source = ColumnDataSource(data=m_country) # connecting it to the table used above.



#Creating the plot

p = figure(plot_width=1000, plot_height=400, x_range=x)



#Adding the circles to the plot

p.circle("Country","Ratings",

         source=source, 

         size = 10, color="red")



#Adding labels



p.title.text = 'Average ramen ratings by country'

p.xaxis.axis_label = 'Country'

p.yaxis.axis_label = 'Average rating out of 5'

p.xaxis.visible = False #The countries were overlapping too much



#Showing the graph

show(p)



#I needed to import "output_notebook" so that it runs within the notebook!
#setting the axis using a dictionary - maybe they aren't connecting together? | This doesn't work either...



x = m_country.Country.tolist() #Country as x-axis

y = m_country.Ratings.tolist() #Country as y-axis



#Just testing this works.
#So it renders in the notebook

output_notebook()



#Let's try something else - loading the graph information directly into the figure() method



f = figure(title='Average ramen ratings by Country',

           plot_height=500, plot_width=700,

           x_axis_label='Rating', y_axis_label='Country')



#Everything else as normal



f.square(x="Country", y="Ratings",

        source=source,

        size=10,color="red")



#showing the graph

#show(f)



#This doesn't work either...
#This is taken directly from the documentation - this works then... 



output_notebook()



p = figure(plot_width=400, plot_height=400)



# add a square renderer with a size, color, and alpha

c = m_country.Country.tolist()

r = m_country.Ratings.tolist()

p.square(c,r, size=20, color="olive", alpha=0.5)



# show the results

#show(p)
output_notebook()



x = m_country.Country.tolist() #Country as x-axis

y = m_country.Ratings.tolist() #Country as y-axis



source = ColumnDataSource(data=m_country) # connecting it to the table used above.



#Creating the plot

p = figure(plot_width=1000, plot_height=400, x_range=x)



#Adding the circles to the plot

p.circle("Country","Ratings",

         source=source, 

         size = 20, color="red")



#Adding labels



p.title.text = 'Average ramen ratings by country'

p.xaxis.axis_label = 'Country'

p.yaxis.axis_label = 'Average rating out of 5'

p.xaxis.visible = False #The countries were overlapping too much



hover = HoverTool()

hover.tooltips=[

    ("Country","@Country"),("Rating out of 5", "@Ratings")

]

p.add_tools(hover)



#Showing the graph

show(p)