# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#plotting libraries

from plotly import __version__

#import plotly

#import plotly.graph_objs as go

import plotly.offline as ply

ply.init_notebook_mode(connected=True)





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/multipleChoiceResponses.csv',encoding = "ISO-8859-1")
newdata=[]



#There can be multiple past job titles, and they are comma separated. 

#Let's split them into separate titles and join them with the current titles and create a new dataframe.

for past,current in zip(data['PastJobTitlesSelect'], data['CurrentJobTitleSelect']):

    if type(past)==str and type(current)==str:

        for j in past.split(','):

            newdata.append({'Current':'To '+current, 'Past':'From '+j})



newdf=pd.DataFrame(newdata)
#In order to create a sankey diagram we need the pairs and the corresponding counts.

counts=newdf.groupby(['Past','Current']).size().reset_index(name='counts')
#Plotly's sankey diagram API needs a list of all labels, and links connecting the sources to the targets.

#Converting them to lists so that we can pass the indices to the API

past=counts['Past'].tolist()

current=counts['Current'].tolist()

unique_values=pd.unique(pd.concat((counts['Past'],counts['Current']),axis=0)).tolist()

past_indices=[unique_values.index(i) for i in past]

current_indices=[unique_values.index(i) for i in current]

#type(past)
np.random.randn(len(unique_values))+10
data_trace = dict(

    type='sankey',

    #width = 1118,

    #height = 772,

    domain = dict(

      x =  [0,1],

      y =  [0,1]

    ),

    orientation = "h",

    node = dict(

#      pad = 15,

#      thickness = 15,

#      line = dict(

#        color = "black",

#        width = 0.5

#      ),

      label=unique_values

      #color=np.random.randn(len(unique_values))+10,

      #colorscale='Jet'

    ),

    link=dict(

        source=past_indices,

        target=current_indices,

        value=counts['counts']

    ))



layout =  dict(

    height=1500,

    #width=80,

    title = "Past and current job titles of data professionals",

    font = dict(

      size = 10

    )

)





fig = dict(data=[data_trace], layout=layout)

ply.iplot(fig, validate=False)

      