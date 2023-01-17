# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import altair as alt #imports altair

from altair import Chart, X, Y, Axis, SortField #imports other things from altair that I needed for later on. 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
endorsement_primary = pd.read_csv("../input/endorsements-2020.csv") #reads my CSV file using pandas

endorsement_primary = endorsement_primary[pd.notnull(endorsement_primary['endorsee'])] #drops null values from the endorsee column so my results aren't crazy

endorsement_primary.head() #reads out my top 5 rows so I can see what my headers are. 
#altair's chart object let's me determine the type of chart I want and how to encode it.

chart = alt.Chart(endorsement_primary).mark_bar().encode( #here I am creating bar chart with the mark_bar function, encoding starts a block of instructions to create the chart.

        alt.X("endorsee"), #made my x axis the endorsee column

        alt.Y("count()") #counts the individual records in the endorsee column to arrive at a total. 

)  

chart
#altair's chart object let's me determine the type of chart I want and how to encode it.

chart = alt.Chart(endorsement_primary).mark_bar().encode( #creates a bar chart

        alt.X("endorsee"), #encodes X axis as endorsee

        alt.Y("count()"), #encodes Y axis as a count of endorsements. 

        color = "category" #I added this to make a stacked bar chart so we could see where endorsements were coming from!

)

chart 
chart = Chart(endorsement_primary).mark_bar().encode( #same altair syntax to create a bar chart. 

    y=Y('endorsee', sort=SortField(field='points', order='descending'),#shows the endorsee as the y axis and sorts descending on points.

        axis=Axis(title='Endorsee')), #i made this slightly fancier by making it horizontal by placing y first. 

    x=X('sum(points)', #sums the point column for each endorsee and displays it as the x axis. 

        axis=Axis(title='endorsee points')), 

    color = "category" #reveals the models inner-workings by splitting it out by category, obviously is weighting federal-reps and senators more than other categories.



)

    



chart
