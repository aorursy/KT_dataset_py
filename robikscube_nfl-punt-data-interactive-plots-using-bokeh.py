import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

import bokeh
from bokeh.io import show, output_notebook, push_notebook
from bokeh.layouts import row, column, widgetbox
from bokeh.plotting import figure
from bokeh.models import CustomJS, ColumnDataSource, Slider
from bokeh.models import ColumnDataSource, Range1d, LabelSet, Label
from bokeh.models import BoxAnnotation
from bokeh.models.glyphs import Text
from bokeh.models.widgets import PreText, Div

output_notebook()
# Read play data
ngs = pd.read_csv('../input/NGS-2016-pre.csv')
ngs.columns = [col.lower() for col in ngs.columns]
pprd = pd.read_csv('../input/play_player_role_data.csv')
pprd.columns = [col.lower() for col in pprd.columns]
vr = pd.read_csv('../input/video_review.csv')
vr.columns = [col.lower() for col in vr.columns]
# Take a play as an example
vr[['season_year','gamekey','playid']].sort_values(['season_year','gamekey','playid']).loc[1]
# Filter a sepcific play
season_year=2016
gamekey=21
playid=2587

play = ngs[(ngs['season_year'] == season_year) &
           (ngs['gamekey'] == gamekey) &
           (ngs['playid'] == playid)]
# Pull just one play
play = pd.merge(play, pprd)
playxy = play.drop(['season_year','gamekey'], axis=1) \
    .pivot(index='time',
           columns='role',
           values=['x','y'])
# Create player colors
colors = ['red' if x in(vr['gsisid'].values) or x in(vr['primary_partner_gsisid'].values) else 'blue' for x in play.gsisid.unique()]

# Setup Figure
## Football Field Figure
fig = figure(plot_width=800, plot_height=400, x_range=(0,120), y_range=(0, 53.3))
fig.xgrid.grid_line_color = None
fig.ygrid.grid_line_color = None
# Green Field
box = BoxAnnotation(left=0, right=120, fill_color='green', fill_alpha=0.5)
endzone1 = BoxAnnotation(left=0, right=10, fill_color='grey', fill_alpha=0.5)
endzone2 = BoxAnnotation(left=110, right=120, fill_color='grey', fill_alpha=0.5)
fig.add_layout(box)
fig.add_layout(endzone1)
fig.add_layout(endzone2)
# Add lines|
fig.line([10,10,20,20,30,30,40,40,50,50,60,60,70,70,80,80,90,90,
          100,100,110,110,120,120],
         [55.5,0,0,55.5,55.5,0,0,55.5,55.5,0,0,55.5,55.5,0,0,55.5,
         55.5,0,0,55.5,55.5,0,0,0], line_color='grey')
# Line numbers

line_nums = ColumnDataSource(dict(x=[20, 30, 40, 50, 60, 70, 80, 90, 100],
                                  y=[5, 5, 5, 5, 5, 5, 5, 5, 5],
                                  text=['10', '20', '30','40','50','40','30','20','10']))
glyph = Text(x="x", y="y", text="text", angle=0, text_color="black", text_align='center')
fig.add_glyph(line_nums, glyph)

# Create values used in the initial states
x_values = playxy.loc[playxy.index[1]]['x'].values
y_values = playxy.loc[playxy.index[1]]['y'].values
gsisid = playxy['x'].columns.values
source = ColumnDataSource(data=dict(x=x_values,
                                    y=y_values,
                                    gsisid=gsisid))
labels = LabelSet(x='x', y='y', text='gsisid', level='glyph',
              x_offset=5, y_offset=5, source=source, render_mode='canvas')

# Add players as circles
plt = fig.circle(x_values, y_values, size=20, alpha=0.5, color=colors, radius=1)
fig.add_layout(labels)

div = Div(text="""....""",width=200, height=100)

def plot_time_pos(time_pos):
    x_values = playxy.loc[playxy.index[time_pos]]['x'].values
    y_values = playxy.loc[playxy.index[time_pos]]['y'].values
    gsisid = playxy['x'].columns.values
    # p = figure(plot_width=800, plot_height=400)
    # p.circle(x_values, y_values, size=20, color="navy", alpha=0.5)
    source = ColumnDataSource(data=dict(x=x_values,
                                    y=y_values,
                                    gsisid=gsisid))
    labels = LabelSet(x='x', y='y', text='gsisid', level='glyph',
               x_offset=5, y_offset=5, source=source, render_mode='canvas')
    fig.renderers.pop()
    fig.add_layout(labels)
    plt.data_source.data['x'] = x_values
    plt.data_source.data['y'] = y_values
    div.text = playxy.loc[playxy.index[time_pos]].name

    push_notebook(handle=bokeh_handle)


source = ColumnDataSource(data=dict(x=x_values, y=y_values))

# Using javascript custom code
callback = CustomJS(code="""
if (IPython.notebook.kernel !== undefined) {
    var kernel = IPython.notebook.kernel;
    cmd = "plot_time_pos(" + cb_obj.value + ")";
    kernel.execute(cmd, {}, {});
}
""")


slider = Slider(start=1,
                end=len(playxy),
                value=1,
                step=1,
                title="position within play",
                callback=callback)

layout = column(
    widgetbox(slider),
    fig,
    div
)

bokeh_handle = show(layout, notebook_handle=True)