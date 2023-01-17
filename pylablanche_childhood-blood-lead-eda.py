# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sb
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode, plot
init_notebook_mode(connected=True)

plt.rcParams['figure.figsize'] = (8,6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 15
sb.set_style("whitegrid")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# reading the csv file and displaying some information
data_o = pd.read_csv('../input/childhood-lead.csv')
data_o.info()
# Just giving some columns a slightly easier name to use
dict_rename = {"Population  < 72 months old":"Population",
               "# of Children Tested":"N_children",
               "Total  #  of Children with Confirmed BLL ≥10 µg/dL":"N_confirmed",
               "% of Children Tested with Confirmed BLLs ≥10 µg/dL":"A_confirmed",
               "≥5 µg/dL":"<5 µg/dL"}

data = data_o.rename(index=str, columns=dict_rename)

# little bit of extra cleaning and setting years to be of type `int32`
data.replace('2000 §','2000', inplace=True)
data['Year'] = np.int32(data.Year.values)
# checking the resulting dataframe
data.info()
# Selecting entries regarding the whole country only
data_us = data[data.State=='U.S. Totals'].set_index('Year')
fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10,8))

ax[0].plot(data_us.index, data_us['N_confirmed'], c='C0', marker='|')
ax[0].set_ylabel('Number of confirmed cases')
ax[0].set_ylim((0.0,1.3e5))
ax[0].set_title('Blood Lead Cases', fontsize=20)
           
ax[1].plot(data_us.index, 100*data_us['A_confirmed'], c='C1', marker='|')
ax[1].set_ylabel('% of confirmed cases')
ax[1].set_ylim((0.0,10.0))
ax[1].set_xticks(data_us.index)
ax[1].set_xlabel('Year')
plt.setp( ax[1].xaxis.get_majorticklabels(), rotation=60 )

plt.subplots_adjust(hspace=0.05);
ax = data_us[["<5 µg/dL","5-9 µg/dL","10-14 µg/dL","15-19 µg/dL",
            "20-24 µg/dL","25-44 µg/dL","45-69 µg/dL","≥70 µg/dL"]].plot(xticks=data_us.index,
                                                                         marker='|',
                                                                         figsize=(12,7))

plt.ylabel('Number of Cases')
plt.title('Number of Cases per Blood Lead Levels vs Year', fontsize=20)
plt.setp( ax.xaxis.get_majorticklabels(), rotation=60 );
# defining a temporary DataFrame for the bar plot
data_tmp = data_us[["10-14 µg/dL","15-19 µg/dL","20-24 µg/dL","25-44 µg/dL","45-69 µg/dL","≥70 µg/dL"]].copy()
# comuting the percentage each range represent
data_perc = data_tmp.apply(lambda x : 100 * x/data_tmp.sum(axis=1), axis=0)
# Visualizing the result using a stacked barplot
fig = plt.figure(figsize=(10,6))

data_perc.plot.bar(stacked=True, width=0.9, ax=fig.gca())
plt.yticks(np.arange(0.0,110,10))
plt.ylabel('Cases Percentage')
plt.title('Blood Lead Cases Distribution', fontsize=20)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5));
# In order to visualize cases per state we need to define the dictionary
# giving 2-letters codes for each state
states = {'AK': 'Alaska', 'AL': 'Alabama', 'AR': 'Arkansas', 'AS': 'American Samoa',
          'AZ': 'Arizona', 'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut',
          'DC': 'District of Columbia', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia',
          'GU': 'Guam', 'HI': 'Hawaii', 'IA': 'Iowa', 'ID': 'Idaho', 'IL': 'Illinois',
          'IN': 'Indiana', 'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana',
          'MA': 'Massachusetts', 'MD': 'Maryland', 'ME': 'Maine', 'MI': 'Michigan',
          'MN': 'Minnesota', 'MO': 'Missouri', 'MP': 'Northern Mariana Islands',
          'MS': 'Mississippi', 'MT': 'Montana', 'NA': 'National', 'NC': 'North Carolina',
          'ND': 'North Dakota', 'NE': 'Nebraska', 'NH': 'New Hampshire', 'NJ': 'New Jersey',
          'NM': 'New Mexico', 'NV': 'Nevada', 'NY': 'New York', 'OH': 'Ohio', 'OK': 'Oklahoma',
          'OR': 'Oregon', 'PA': 'Pennsylvania', 'PR': 'Puerto Rico', 'RI': 'Rhode Island',
          'SC': 'South Carolina', 'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas',
          'UT': 'Utah', 'VA': 'Virginia', 'VI': 'Virgin Islands', 'VT': 'Vermont',
          'WA': 'Washington', 'WI': 'Wisconsin', 'WV': 'West Virginia', 'WY': 'Wyoming'}
# and the inverted dictionary
us_states = {states[state]:state for state in states}
# We'll only look at the high values of blood lead > 70 µg/dL
data.replace({"New York (Excl NYC)":"New York", "New York City":"New York"}, inplace=True)
tmp_df = data.replace(us_states)
data_state = tmp_df[tmp_df.State != 'U.S. Totals'].groupby(['State','Year']).sum()
data70 = data_state["≥70 µg/dL"].unstack()
plt.figure(figsize=(14,10))
ax = sb.heatmap(data70, vmin=0, vmax=50, cmap='Reds')
plt.setp( ax.yaxis.get_majorticklabels(), rotation=0 )
plt.title('Blood Lead cases > 70µg/dL per State and Year', fontsize=20);
# Choropleth Visualization
data_choro = []
steps_choro = []

# Note that I fix the range to be between -5000 and 5000 people
# otherwise we can't see much on the map
for year in list(range(1997,2015)):
    data_choro.append(dict(type = 'choropleth',
                      visible=False,
                      locations = data70.index.values,
                      locationmode='USA-states',
                      z = data70[year].values,
                      text = data70.index.values,
                      colorscale='Reds',
                      zmin=0.0,
                      zmax=50.0,
                      marker = dict(line = dict (color = 'rgb(180,180,180)', width = 0.5))))
    steps_choro.append({'label':year,
                        'method':'update',
                        'args':[{'visible': np.arange(1997,2015) == year}]}),

layout = dict(title = 'US Blood Lead Cases ≥70 µg/dL',
              geo = dict(scope='usa',
                         showframe = False,
                         showcoastlines = False),
              sliders=[dict(steps=steps_choro)])

data_choro[0]['visible']=True

iplot(dict(data=data_choro, layout=layout))
