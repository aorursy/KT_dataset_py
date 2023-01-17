# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode(connected=True)

terror_data = pd.read_csv('../input/globalterrorismdb_0617dist.csv', encoding='ISO-8859-1',
                          usecols=[0, 1, 2, 3, 8, 11, 13, 14, 35, 84, 100, 103])
terror_data = terror_data.rename(
    columns={'eventid':'id', 'iyear':'year', 'imonth':'month', 'iday':'day',
             'country_txt':'country', 'provstate':'state', 'targtype1_txt':'target',
             'weapsubtype1_txt':'weapon', 'nkillter':'fatalities', 'nwoundte':'injuries'})
terror_data['fatalities'] = terror_data['fatalities'].fillna(0).astype(int)
terror_data['injuries'] = terror_data['injuries'].fillna(0).astype(int)

attacks_france = terror_data[terror_data.country == 'France']
attacks_france[attacks_france.year == 2016]
# terrorist attacks by year
terror_peryear = np.asarray(attacks_france.groupby('year').year.count())
terror_years = np.arange(1972, 2017)
# terrorist attacks in 1993 missing from database
terror_years = np.delete(terror_years, [23])

trace = [go.Scatter(
         x = terror_years,
         y = terror_peryear,
         mode = 'lines',
         line = dict(
             color = 'rgb(240, 140, 45)',
             width = 3)
         )]

layout = go.Layout(
         title = 'Terrorist Attacks by Year in France (1970-2016)',
         xaxis = dict(
             rangeslider = dict(thickness = 0.05),
             showline = True,
             showgrid = False
         ),
         yaxis = dict(
             range = [0.1, 425],
             showline = True,
             showgrid = False)
         )

figure = dict(data = trace, layout = layout)
iplot(figure)