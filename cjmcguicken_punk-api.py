import pandas as pd



#Grab data from CSV - Github code uses the Punk API (via requests) but this isn't allowed within Kaggle for obvious reasons

#Please see Github for full code - https://github.com/emperorcal/punk-ipython



beer_df = pd.read_csv('../input/punk_data.csv')
import pandas as pd

import ast



#Create new dataframe to understand hops brewed by year

hops_df = pd.DataFrame(columns=['name', 'year'])



#Iterate each row in the raw beer dataframe

for index, row in beer_df.iterrows():

    

    #As multiple hops can be in one brew, second loop to iterate through list within beer dataframe row

    #As taking from CSV need to conver string representation of list into actual list

    for hop_name in ast.literal_eval(row['hops']):

        

        #Create temporary dictionary to be used to append to the hops dataframe

        temp_dict = {}



        #Get name of hop

        temp_dict['name'] = hop_name

        

        #Get year of brew, difficulty here is inconsistent formatting in data (sometimes MM/YYYY or YYYY)

        #Therefore search for "/" and split if required 

        temp_dict['year'] = beer_df.loc[index, 'first_brewed']

        if "/" in temp_dict['year']:

            temp_dict['year'] = temp_dict['year'].split("/")[1]

            

        #Append temporary dictionary to hops dataframe

        temp_df  = pd.DataFrame([temp_dict], columns=hops_df.keys())

        hops_df = hops_df.append(temp_df)



#Reset index of Dataframe (ensuring that old index doesn't get added as a column)

hops_df = hops_df.reset_index(drop=True)  



#Change hops dataframe structure - years as columns and hop name as index

hops_df = pd.crosstab(index=hops_df['name'], columns=hops_df['year'])



#Reset index of dataframe to ensure name is a column (as well as index name)

hops_df = hops_df.reset_index()



#Sort and select top 20 hops

top_20_hops_df = hops_df.sort_values(by=hops_df.columns[len(hops_df.columns)-1], ascending=False).head(20)



#Print progress

print("Hops dataframe completed!")
from plotly.offline import init_notebook_mode, iplot

from IPython.display import display, HTML



#Use offline mode of Plotly - injecting the plotly.js source files into the notebook

init_notebook_mode(connected = True)



#Get list of years required for slider (removing first item as it is 'Name')

years = list(top_20_hops_df)[1:]



#Get list of hops names

hops = top_20_hops_df['name'].tolist()



#Get list of first year

first_year = top_20_hops_df.iloc[:,1].tolist()
#Create figure with custom formatting

#Formatting includes BrewDog logo and play/end animation buttons

figure = {

  'data': [{

    'type': 'bar',

    'x': hops,

    'y': first_year

  }],

  'layout': {

    'images': [{

      'source': 'https://raw.githubusercontent.com/emperorcal/punk-ipython/master/brewdog-logo.jpg',

      'xref': 'paper',

      'yref': 'paper',

      'x': 1,

      'y': 1.05,

      'sizex': 0.3,

      'sizey': 0.3,

      'xanchor': 'right',

      'yanchor': 'bottom'

    }],

    'title': "Brewdog's Top 20 Choice of Hops in a Brew Per Year",

    'xaxis': {

      'title': 'Hop',

      'gridcolor': '#FFFFFF',

      'linecolor': '#000',

      'linewidth': 1,

      'tickangle': -45,

      'zeroline': False,

      'autorange': True

    },

    'yaxis': {

      'title': 'Number of Times in a Brew',

      'gridcolor': '#FFFFFF',

      'linecolor': '#000',

      'linewidth': 1,

      'range': [0, top_20_hops_df.iloc[:,-1].max()+2],

      'autorange': False

    },

    'title': 'Hopping Over the Years<br><i>Brewdog hop choice per year</i>',

    'hovermode': 'closest',

    'updatemenus': [{

      'type': 'buttons',

      'buttons': [{

          'label': 'Play',

          'method': 'animate',

          'args': [None, {

            'frame': {

              'duration': 500,

              'redraw': True

            },

            'fromcurrent': True,

            'transition': {

              'duration': 300,

              'easing': 'quadratic-in-out'

            }

          }]

        },

        {

          'label': 'End',

          'method': 'animate',

          'args': [None, {

            'frame': {

              'duration': 0,

              'redraw': True

            },

            'fromcurrent': True,

            'mode': 'immediate',

            'transition': {

              'duration': 0

            }

          }]

        }

      ],

      'direction': 'left',

      'pad': {

        'r': 10,

        't': 87

      },

      'showactive': False,

      'type': 'buttons',

      'x': 0.1,

      'xanchor': 'right',

      'y': -0.3,

      'yanchor': 'top'

    }]

  },

  'frames': []

}
#Create additional part of the layout - sliders

#To be used to slice data by years



sliders_dict = {

  'active': 0,

  'yanchor': 'top',

  'xanchor': 'left',

  'currentvalue': {

    'font': {

      'size': 20

    },

    'prefix': 'Year:',

    'visible': True,

    'xanchor': 'right'

  },

  'transition': {

    'duration': 300,

    'easing': 'cubic-in-out'

  },

  'pad': {

    'b': 10,

    't': 50

  },

  'len': 0.9,

  'x': 0.1,

  'y': -0.3,

  'steps': []

}



#Loop through years, creating a different figure for each year through a 'frame'

for year in years:

    frame = {

        'data': [{

          'type': 'bar',

          'x': hops,

          'y': top_20_hops_df[year].tolist(),

          'marker': {

            'color': '#cbcfd6',

            'line': {

              'color': 'rgb(0, 0, 0)',

              'width': 2

            }

          }

        }],

        'name': str(year)

        }

    figure['frames'].append(frame)



    slider_step = {

      'args': [

        [year],

        {

          'frame': {

            'duration': 300,

            'redraw': True

          },

          'mode': 'immediate',

          'transition': {

            'duration': 300

          }

        }

      ],

      'label': year,

      'method': 'animate'

    }

    sliders_dict['steps'].append(slider_step)



#Add sliders data to Figure

figure['layout']['sliders'] = [sliders_dict]
#Plot figure

iplot(figure)