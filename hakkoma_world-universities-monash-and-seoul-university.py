import numpy as np
import pandas as pd
import requests
import sys
import json
import time
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
init_notebook_mode()
#read_csv ARWU (Shanghai data)
data = pd.read_csv('../input/shanghaiData.csv', skiprows=[3897])
data[data['university_name'] == "Seoul National University"]
data[data['university_name'] == "Monash University"]
data[data['university_name'] == "Harvard University"]
data[data['university_name'] == "University of Oxford"]
#lamda (https://wikidocs.net/64)
data['new_total_score'] = data.apply(lambda x: 0.1 * x[4] + 0.2 * x[5] + 0.2 * x[6] + 0.2 * x[7] + 0.2 * x[8] + 0.1 * x[9], axis=1)
#to check the correlation between old total score and NEW total score
data[['total_score', 'new_total_score']].corr()
data.drop('total_score', 1, inplace=True)
data.rename(columns={'new_total_score': 'total_score'}, inplace=True)
#To see the top 10 universities by total_score
data[:10]
#total_score for Monash and Seoul University
data[data.university_name == 'Seoul National University']
data[data.university_name == 'Monash University']
#last year of the data exploration which is 2015
year = 2015
data_byy = data.groupby('year').get_group(year)
corr = data_byy[['alumni', 'award', 'hici', 'ns', 'pub', 'pcp']].corr()
iplot([
    go.Heatmap(
        z=corr.values[::-1],
        x=['alumni', 'award', 'hici', 'ns', 'pub', 'pcp'],
        y=['alumni', 'award', 'hici', 'ns', 'pub', 'pcp'][::-1]
    )
])
#correlation plot setting
def plot_corr(x_name, y_name, year):
    data_sc = [go.Scatter(
        x = data_byy[x_name],
        y = data_byy[y_name],
        text = data_byy['university_name'],
        mode = 'markers',
        marker = dict(
            size = 10,
            color = data_byy['total_score'],
            colorscale = 'Rainbow',
            showscale = True,
            colorbar = dict(
                title = 'total score',
            ),
        ),
    )]

    layout = go.Layout(
        title = '%s World University Rankings' % year,
        hovermode = 'closest',
        xaxis = dict(
            title = x_name,
        ),
        yaxis = dict(
            title = y_name,
        ),
        showlegend = False
    )

    iplot(go.Figure(data=data_sc, layout=layout))
#HiCi and N&S
plot_corr('hici', 'ns', year)
#alumni and award
plot_corr('alumni', 'award', year)
#pub and hici
plot_corr('pub', 'hici', year)
#award and pub
plot_corr('award', 'pub', year)
%matplotlib inline
import matplotlib.pylab as plt
from matplotlib import colors as mcol

class Radar(object):

    def __init__(self, fig, titles, labels, rect=None):
        if rect is None:
            rect = [0.05, 0.05, 0.95, 0.95]

        self.n = len(titles)
        self.angles = np.arange(0, 360, 360.0/self.n)
        self.axes = [fig.add_axes(rect, projection="polar", label="axes%d" % i) 
                         for i in range(self.n)]

        self.ax = self.axes[0]
        self.ax.set_thetagrids(self.angles, labels=titles, fontsize=14)

        for ax in self.axes[1:]:
            ax.patch.set_visible(False)
            ax.grid("off")
            ax.xaxis.set_visible(False)

        for ax, angle, label in zip(self.axes, self.angles, labels):
            ax.set_rgrids(range(1, 102, 10), angle=angle, labels=label)
            ax.spines["polar"].set_visible(False)
            ax.set_ylim(0, 101)

    def plot(self, values, *args, **kw):
        angle = np.deg2rad(np.r_[self.angles, self.angles[0]])
        values = np.r_[values, values[0]]
        self.ax.plot(angle, values, *args, **kw)
top_univers = ['Harvard University',
               'Stanford University',
               'Massachusetts Institute of Technology (MIT)',
               'University of California, Berkeley', 'University of California-Berkeley',
               'University of Cambridge',
               'Princeton University',
               'California Institute of Technology',
               'Columbia University',
               'University of Chicago',
               'University of Oxford',
               'Monash University',
               'Seoul National University']
Comparison = []

years = list(set(data['year']))
for i, year in enumerate(years):
    tmp = data.groupby('year').get_group(year)
    
    ind = np.where(tmp['university_name'] == top_univers[0])[0]
    univers = tmp.iloc[ind].values
    for un in top_univers[1:]:
        ind = np.where(tmp['university_name'] == un)[0]
        univers = np.append(univers, tmp.iloc[ind].values, axis=0)
    
    Comparison += [univers]
    
Comparison = np.array(Comparison)
#2005 - 2015 data comparison in the radar
titles = ['alumni', 'award', 'hici', 'ns', 'pub', 'pcp']
labels = ['' if i != 1 else range(0, 101, 10) for i in range(len(titles) - 1)]
colors = np.asarray(list(mcol.cnames))
colors = colors[np.random.randint(0, len(mcol.cnames), Comparison[0].shape[0])]

for d in Comparison:
    fig = plt.figure(figsize=(5, 5))
    radar = Radar(fig, titles, labels)
    for i, univ in enumerate(d[d[:, 0].argsort()]):
        radar.plot(univ[3:9], lw=2, c=colors[i], alpha=1, label=univ[1] + ' (' + univ[0] + ')')

    radar.ax.legend(bbox_to_anchor=(1, 1), loc=2);
    plt.show()
