#Importing required packages for the analysis.

import numpy as np

import pandas as pd

import pandas_profiling

import plotly.graph_objects as go



#Reading the data file.

df = pd.read_csv('../input/covid19-egypt-cases/Egy-COVID-19.csv')



#Change the dates to the proper formatting

df['date'] = pd.to_datetime(df['date'])
report = df.profile_report(title= 'Egypt COVID-19 Data')

report.to_notebook_iframe()
#plotly configuration

config = {

    'renderer': 'kaggle',

    'displayModeBar': True,

    'modeBarButtonsToRemove': ['hoverClosestCartesian','toggleSpikelines']

}





#plotly layout configuration

layout = {

    'template': 'plotly_dark',

    'margin': {

        'pad': 5

    },

    'height': 1000,

    'xaxis': {

        'title': {

            'text': 'Date',

            'font': {

                'size': 16

            }

        },

        'tickformat': '%d %B <br>%Y'

    },

    'yaxis': {

        'title': {

            'text': 'Number of cases',

            'font': {

                'size': 16

            }

        }

    },

    'title': {

        'text': 'Egypt COVID-19 Cases',

        'font': {

            'size': 20

        },

        'x': 0.5,

        'xanchor': 'center',

        'yanchor': 'top'

    },

    'legend': {

        'bordercolor': '#fff',

        'borderwidth': 2,

        'y': 0.5,

        'yanchor': 'middle',

        'title': {

            'text': 'Status',

            'font': {

                'size': 14

            }

        },

        'font': {

            'size': 12

        }

    },

    'showlegend': True,

    'modebar': {

        'color': '#fff',

        'activecolor': '#00f'

    },

    'hoverlabel': {

        'namelength': -1,

        'bordercolor': '#fff'

    },

    'hovermode': 'x'

}
#initiate the figure

fig = go.Figure(

    layout = go.Layout(layout))



#add confirmed trace to the figure

fig.add_scatter(

    x= df['date'],

    y= df['new_confirmed'],

    mode= 'lines',

    name= 'Confirmed',

    line= {

        'color': 'blue',

        'width': 4

    }

)



#show the figure

fig.show(config = config)
#add deaths trace to the figure

fig.add_scatter(

    x= df['date'],

    y= df['new_deaths'],

    mode= 'lines',

    name= 'Deaths',

    line= {

        'color': 'red',

        'width': 4

    }

)



#hide the confirmed trace

fig.update_traces(visible= 'legendonly',

                  selector= {

                      'name': 'Confirmed'

                  }

)



#show the figure

fig.show(config = config)
#add recovered trace to the figure

fig.add_scatter(

    x= df['date'],

    y= df['new_recovered'],

    mode= 'lines',

    name= 'Recovered',

    line= {

        'color': 'green',

        'width': 4

    }

)



#hide the deaths trace

fig.update_traces(visible= 'legendonly',

                  selector= {

                      'name': 'Deaths'

                  }

)



#show the figure

fig.show(config = config)
#add positive to negative trace to the figure

fig.add_scatter(

    x= df['date'],

    y= df['new_pos_to_neg'],

    mode= 'lines',

    name= 'Positive to Negative',

    line= {

        'color': 'yellow',

        'width': 4

    },

)



#hide the recovred trace

fig.update_traces(visible= 'legendonly',

                  selector= {

                      'name': 'Recovered'

                  }

)



#show the figure

fig.show(config = config)
#hide the positive to negative trace

fig.update_traces(visible= 'legendonly',

                  selector= {

                      'name': 'Positive to Negative'

                  }

)



#show confirmed trace

fig.update_traces(visible= True,

                  selector= {

                      'name': 'Confirmed'

                  }

)



#show recovered trace

fig.update_traces(visible= True,

                  selector= {

                      'name': 'Recovered'

                  }

)



#show the figure

fig.show(config = config)
#hide the confirmed trace

fig.update_traces(visible= 'legendonly',

                  selector= {

                      'name': 'Confirmed'

                  }

)



#show deaths trace

fig.update_traces(visible= True,

                  selector= {

                      'name': 'Deaths'

                  }

)



#show recovered trace

fig.update_traces(visible= True,

                  selector= {

                      'name': 'Recovered'

                  }

)



#show the figure

fig.show(config = config)
#show all traces

fig.update_traces(visible= True)



#update the layout for the y-axis to increase ticks range

fig.update_layout({

    'yaxis': {

        'tickmode': 'linear',

        'tick0': 0,

        'dtick': 20,

        'rangemode': 'nonnegative'

    }

})



#show the figure

fig.show(config = config)
#initiate the figure

fig = go.Figure(

    layout= go.Layout(layout)

)        



#add a trace to the figure

fig.add_scatter(

    x= df['date'],

    y= df['total_recovered'],

    mode= 'lines',

    name= 'Exits',

    line= {

        'color': 'green',

        'width': 4

    },

)



#add a trace to the figure

fig.add_scatter(

    x= df['date'],

    y= df['active_pos'],

    mode= 'lines',

    name= 'Positive In Hospitals',

    line= {

        'color': 'red',

        'width': 4

    },

)



#add a trace to the figure

fig.add_scatter(

    x= df['date'],

    y= df['cases_in_hospital'] - df['active_pos'],

    mode= 'lines',

    name= 'Negative In Hospitals',

    line= {

        'color': 'yellow',

        'width': 4

    },

)





#show the figure

fig.show(config = config)