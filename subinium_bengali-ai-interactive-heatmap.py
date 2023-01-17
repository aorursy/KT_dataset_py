import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



import seaborn as sns

from PIL import Image



from bokeh.plotting import figure, show, output_notebook

from bokeh.models import HoverTool, ColumnDataSource, LinearColorMapper, BasicTicker, PrintfTickFormatter, ColorBar

from bokeh.palettes import Purples256



output_notebook()
train = pd.read_csv('/kaggle/input/bengaliai-cv19/train.csv')

train.head()
def plot_count_heatmap(feature1, feature2, train):  

    count = train.groupby([feature1, feature2])['grapheme'].count().reset_index()

    return count.pivot(feature1, feature2, "grapheme").fillna(0)
def plot_heatmap(f1, f2, width, height, cbar=True):

    f1_len, f2_len = len(train[f1].unique()), len(train[f2].unique())

    

    # index list

    x = [str(i) for i in range(f2_len)] * f1_len

    y = [str(i) for i in range(f1_len) for _ in range(f2_len)]

    

    # count list

    tmp = plot_count_heatmap(f1,f2, train)

    value = [tmp[int(a)][int(b)] for a, b in zip(x, y)]

    

    # example letter list

    letter = train.groupby([f1, f2])['grapheme'].unique().unstack().fillna('')

    lst = [','.join(letter[int(a)][int(b)]) for a, b in zip(x, y)]

    

    # processing for bokeh

    df = pd.DataFrame({f2 : x, f1 : y, 'count' : value, 'example': lst})

    source = ColumnDataSource(df)    

    

    # make continuous color palette

    colors = list(reversed(Purples256))

    mapper = LinearColorMapper(

        palette= colors,

        low=min(value),

        high=max(value)

    )



    # make figure

    p = figure(title=f"{f1} & {f2} Count Heatmap", tools="hover", 

               toolbar_location=None,

               x_range=list(map(str, tmp.columns)), y_range=list(map(str, tmp.index))[::-1],

               plot_width=width, plot_height=height

    )



    # heatmap

    p.rect(f2, f1, 0.95, 0.95, source=source,

          fill_color={'field': 'count', 'transform': mapper}, 

           line_color=None)

    

    # tooltips

    p.hover.tooltips = [

        ("Count", "@count"),

        (f1, f"@{f1}"),

        (f2, f"@{f2}"),

        ("Example", "@example")

    ]



    # detail setting

    p.grid.grid_line_color = None

    p.axis.axis_line_color = None

    p.axis.major_tick_line_color = None

    p.axis.major_label_text_font_size = "5pt"

    p.axis.major_label_standoff = 0



    # colorbar

    

    color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="5pt",

                         ticker=BasicTicker(desired_num_ticks=len(colors)),

                         formatter=PrintfTickFormatter(format="%d"),

                         label_standoff=6, border_line_color=None, location=(0, 0))

    if cbar : p.add_layout(color_bar, 'right')

    

    show(p)
plot_heatmap('grapheme_root', 'vowel_diacritic', 600, 3000)
plot_heatmap('grapheme_root', 'consonant_diacritic', 400, 3000)
plot_heatmap('vowel_diacritic', 'consonant_diacritic', 400, 500, False)