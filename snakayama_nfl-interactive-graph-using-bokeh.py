# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import gc

import seaborn as sns; sns.set()

import matplotlib.pyplot as plt

import matplotlib.patches as patches

import warnings

from bokeh.layouts import row, column, widgetbox

from bokeh.io import output_file, show

from bokeh.plotting import figure, output_file, show,output_notebook

from bokeh.models.annotations import Span, BoxAnnotation, Label, LabelSet

from bokeh.models.glyphs import Text

from bokeh.models import CustomJS, ColumnDataSource, Slider

from bokeh.models import HoverTool

from ipywidgets import interact, Select



output_notebook()

%matplotlib inline



warnings.filterwarnings("ignore")
df_train = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)
df_train.shape
df_train.head()
game_id_list = df_train['GameId'].unique().tolist()

print('train games:{}'.format(len(game_id_list)))
play_id_list = df_train['PlayId'].unique().tolist()

print('play games:{}'.format(len(play_id_list)))
pd.DataFrame(df_train.groupby(by='GameId').size(),columns=['ct']).reset_index().describe()
df_train.columns.tolist()
def create_bokeh_football_field(df,fifty_is_los=False):

    x = [10, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80,80, 90, 90, 100, 100, 110, 110, 120, 0, 0, 120, 120]

    y = [0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3,53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 53.3, 0, 0, 53.3]

    #set object

    f = figure()



    box_2=BoxAnnotation(bottom=0, top=53.3,left=0,right=-9,fill_color="darkgreen",fill_alpha=0.7)

    f.add_layout(box_2)

    box_3=BoxAnnotation(bottom=0, top=53.3,left=120,right=129,fill_color="darkgreen",fill_alpha=0.7)

    f.add_layout(box_3)

    

    f.line(x,y,color='white',line_width=2,)

    

    if fifty_is_los:

        f.line([60, 60], [0, 53.3], color='gold')

        

    f.line([0, 120], [0.5, 0.5], color='white',line_width=3,line_dash="dotted")

    f.line([0, 120], [52.8, 52.8], color='white',line_width=3,line_dash="dotted")

    f.line([0, 120], [22.91, 22.91], color='white',line_width=3,line_dash="dotted")

    f.line([0, 120], [29.73, 29.73], color='white',line_width=3,line_dash="dotted")

    

    #Style the plot area

    f.plot_width = 1200

    f.plot_height = 633

    f.background_fill_color = "Forestgreen"

    

    home_source = ColumnDataSource(data=df[df['Team']=='home'])

    away_source = ColumnDataSource(data=df[df['Team']=='away'])

    f.circle(x="X", y="Y", source=home_source, line_width=2,color='Yellow',alpha=1)

    f.circle(x="X", y="Y", source=away_source, line_width=2,color='blue',alpha=1)

    

    line_nums = ColumnDataSource(dict(x=[20, 30, 40, 50, 60, 70, 80, 90, 100],

                                  y=[5, 5, 5, 5, 5, 5, 5, 5, 5],

                                  text=['10', '20', '30','40','50','40','30','20','10']))

    glyph = Text(x="x", y="y", text="text", angle=0, text_color="white", text_align='center')

    f.add_glyph(line_nums, glyph)

    

    hover=HoverTool()

    hover.tooltips=[("Team","@Team")]

    f.add_tools(hover)



    #別ページにてHTMLファイルを開く

    show(f)
def create_game_graph(id):  

    df_temp = df_train[df_train['GameId']==id]

    @interact(gameid=df_temp['PlayId'].unique().tolist())

    def h(gameid):

        create_bokeh_football_field(df_train[df_train['PlayId']==gameid])
create_game_graph(game_id_list[0])
# x = [10, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80,80, 90, 90, 100, 100, 110, 110, 120, 0, 0, 120, 120]

# y = [0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3,53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 53.3, 0, 0, 53.3]

# #set object

# f = figure()



# box_2=BoxAnnotation(bottom=0, top=53.3,left=0,right=-9,fill_color="darkgreen",fill_alpha=0.7)

# f.add_layout(box_2)

# box_3=BoxAnnotation(bottom=0, top=53.3,left=120,right=129,fill_color="darkgreen",fill_alpha=0.7)

# f.add_layout(box_3)



# f.line(x,y,color='white',line_width=2,)

# f.line([60, 60], [0, 53.3], color='gold')

# f.line([0, 120], [0.5, 0.5], color='white',line_width=3,line_dash="dotted")

# f.line([0, 120], [52.8, 52.8], color='white',line_width=3,line_dash="dotted")

# f.line([0, 120], [22.91, 22.91], color='white',line_width=3,line_dash="dotted")

# f.line([0, 120], [29.73, 29.73], color='white',line_width=3,line_dash="dotted")



# #Style the plot area

# f.plot_width = 1200

# f.plot_height = 633

# f.background_fill_color = "Forestgreen"

# line_nums = ColumnDataSource(dict(x=[20, 30, 40, 50, 60, 70, 80, 90, 100],

#                               y=[5, 5, 5, 5, 5, 5, 5, 5, 5],

#                               text=['10', '20', '30','40','50','40','30','20','10']))

# glyph = Text(x="x", y="y", text="text", angle=0, text_color="white", text_align='center')

# f.add_glyph(line_nums, glyph)



# def plot_time_pos(time_pos):

#     df = df_train[df_train.index==time_pos]

#     home_source = ColumnDataSource(data=df[df['Team']=='home'])

#     away_source = ColumnDataSource(data=df[df['Team']=='away'])

#     f.circle(x="X", y="Y", source=home_source, line_width=2,color='Yellow',alpha=1)

#     f.circle(x="X", y="Y", source=away_source, line_width=2,color='blue',alpha=1)

#     hover=HoverTool()

#     hover.tooltips=[("Team","@Team")]

#     f.add_tools(hover)

#     push_notebook(handle=bokeh_handle)



# #別ページにてHTMLファイルを開く

# #show(f)
# callback = CustomJS(code="""

# if (IPython.notebook.kernel !== undefined) {

#     var kernel = IPython.notebook.kernel;

#     cmd = "plot_time_pos(" + cb_obj.value + ")";

#     kernel.execute(cmd, {}, {});

# }

# """)





# slider = Slider(start=1, end=10, step=1, 

#                 title="position within play",

#                 callback=callback)

# layout = column(

#     widgetbox(slider),

#     f

# )



# bokeh_handle = show(layout, notebook_handle=True)