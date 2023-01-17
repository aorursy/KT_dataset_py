# Import Python Modules

import pandas as pd

import plotly.express as px



# Define Variables

which_months = ['2015-10-01 00:00:00', 

                '2015-11-01 00:00:00',

                '2015-12-01 00:00:00', 

                '2016-01-01 00:00:00',

                '2016-02-01 00:00:00',

                '2016-03-01 00:00:00',

                '2016-04-01 00:00:00',

                '2016-05-01 00:00:00',

                '2016-06-01 00:00:00',

                '2016-07-01 00:00:00',

                '2016-08-01 00:00:00',

                '2016-09-01 00:00:00']

which_regions = ['Colorado',

                 'Washington',

                 'Oregon',

                 'Puerto Rico']



# Load and Reformat the Data

total_women = pd.read_csv('/kaggle/input/publicassistance/WICAgencies2016ytd/Total_Women.csv')

total_women = total_women.melt(id_vars='State Agency or Indian Tribal Organization',value_vars=which_months)

total_women = total_women[total_women['State Agency or Indian Tribal Organization'].isin(which_regions)]



# Plot the Data

fig = px.line(data_frame=total_women, 

              x="variable", 

              y="value", 

              color='State Agency or Indian Tribal Organization',

              title='Supplemental Nutrition Program for Women, Infants, and Children (WIC)')

fig.update(layout=dict(xaxis_title='Date',

                       yaxis_title='Number of Women',

                       legend_orientation="h",

                       showlegend=True))

fig.show()