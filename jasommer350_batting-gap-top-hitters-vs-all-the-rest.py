#Importing the packages I will need for analysis 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#Using Bokeh for charting, I like the charts and the interaction they provide.
from bokeh.io import push_notebook
from bokeh.plotting import figure, show, output_notebook
import bokeh.charts as bcharts
import bokeh.charts.utils as bchartsU
#Focusing on just the batting, have some ideas on the project to add data from some other areas.
full_frame = pd.read_csv("../input/batting.csv")
frame = full_frame[full_frame['year']>=1969]

frame = frame.assign(AVG = frame['h'] / frame['ab'])
frameYearTripleCrown = frame.groupby(['year'], as_index=False)[['rbi', 'hr', 'AVG']].mean()

frameLessCols_AVG = frame.loc[:, ['year', 'player_id', 'AVG']]
yearPlayerAVGSorted = frameLessCols_AVG.sort_values(by=['year', 'AVG'], ascending=[True, False])
topPlayersDetails_AVG = yearPlayerAVGSorted.groupby(['year'], as_index=False).head(50)
topPlayersAVG = topPlayersDetails_AVG.groupby(['year'], as_index=False)['AVG'].mean()
x = frameYearTripleCrown['year']
y = frameYearTripleCrown['rbi']
y2 = frameYearTripleCrown['hr'] 
y3 = frameYearTripleCrown['AVG']
y3_1 = topPlayersAVG['AVG']
output_notebook()
p = figure(title="Mean RBI by Year", plot_height=450, plot_width=800)
p2 = figure(title="Mean HR's by Year", plot_height=450, plot_width=800)
p3 = figure(title="Mean Batting Average by Year Top Players vs All Players", plot_height=450, plot_width=800)
p3_A = figure(title="Mean Batting Average by Year", plot_height=450, plot_width=800)

c = p.circle(x, y, radius=0.8, alpha=0.5)
c2 = p2.circle(x, y2, radius=0.8, alpha=0.5)
c3 = p3.circle(x, y3, radius=0.8, alpha=0.5)
c3_1 = p3.circle(x, y3_1, radius=0.8)
c3_A = p3_A.circle(x, y3, radius=0.8, alpha=0.5)
show(p3_A)
show(p3)