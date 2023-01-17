import pandas as pd
decathlon = pd.read_csv("../input/decathlon.csv")
decathlon.shape
decathlon.columns[0]
decathlon.head()
decathlon.Javeline.head()
decathlon.Discus.head()
from bokeh.io import push_notebook, show, output_notebook
from bokeh.layouts import row 
from bokeh.plotting import figure

output_notebook()
p = figure(title = "Decathlon: Discus x Javeline")
p.circle('Discus','Javeline',source=decathlon,fill_alpha=0.2, size=10)
show(p)
from bokeh.transform import factor_cmap
decathlon.Competition.unique()
index_cmap = factor_cmap('Competition', palette=['red', 'blue'], 
                         factors=sorted(decathlon.Competition.unique()))



p = figure(plot_width=600, plot_height=450, title = "Decathlon: Discus x Javeline")
p.scatter('Discus','Javeline',source=decathlon,fill_alpha=0.6, fill_color=index_cmap,size=10,legend='Competition')
p.xaxis.axis_label = 'Discus'
p.yaxis.axis_label = 'Javeline'
p.legend.location = "top_left"
show(p)
decathlon['Athlets'].head()
p = figure(plot_width=600, plot_height=450, 
           title = "Decathlon: Discus x Javeline",
           toolbar_location=None,
          tools="hover", 
           tooltips="@Athlets: (@Discus,@Javeline)")
p.scatter('Discus','Javeline',source=decathlon,fill_alpha=0.6, fill_color=index_cmap,size=10,legend='Competition')
p.xaxis.axis_label = 'Discus'
p.yaxis.axis_label = 'Javeline'
p.legend.location = "top_left"
show(p)
from bokeh.models import  ColumnDataSource,Range1d, LabelSet, Label

decath=ColumnDataSource(data=decathlon)
p = figure(plot_width=700, plot_height=450, title = "Decathlon: Discus x Javeline")
p.scatter('Discus','Javeline',source=decath,fill_alpha=0.6, fill_color=index_cmap,size=10,legend='Competition')
p.xaxis.axis_label = 'Discus'
p.yaxis.axis_label = 'Javeline'
p.legend.location = "top_left"

labels = LabelSet(x='Discus', y='Javeline', text='Athlets', level='glyph',text_font_size='9pt',
              text_color=index_cmap,x_offset=5, y_offset=5, source=decath, render_mode='canvas')

p.add_layout(labels)
show(p)
from bokeh.models import BoxAnnotation

p = figure(plot_width=700, plot_height=450, title = "Decathlon: Discus x Javeline")
p.scatter('Discus','Javeline',source=decath,fill_alpha=0.6, fill_color=index_cmap,size=10,legend='Competition')
p.xaxis.axis_label = 'Discus'
p.yaxis.axis_label = 'Javeline'
p.legend.location = "top_left"
low_box = BoxAnnotation(top=55, fill_alpha=0.1, fill_color='red')
mid_box = BoxAnnotation(bottom=55, top=65, fill_alpha=0.1, fill_color='green')
high_box = BoxAnnotation(bottom=65, fill_alpha=0.1, fill_color='red')

p.add_layout(low_box)
p.add_layout(mid_box)
p.add_layout(high_box)

p.xgrid[0].grid_line_color=None
p.ygrid[0].grid_line_alpha=0.5
show(p)
p = figure(plot_width=600, plot_height=450, title = "Decathlon: Discus x Javeline")
p.title.text = 'Click on legend entries to hide the corresponding lines'

decathlon.loc[(decathlon.Competition=='OlympicG')].head()
x=['OlympicG','Decastar']
x
for i in x:
    df=decathlon.loc[(decathlon.Competition==i)]
    p.scatter('Discus','Javeline',source=df,fill_alpha=0.6, fill_color=index_cmap,size=10,legend='Competition')

p.legend.location = "top_left"
p.legend.click_policy="hide"
show(p)

    
from bokeh.models import ColumnDataSource, ColorBar
from bokeh.palettes import Spectral6
from bokeh.transform import linear_cmap

Spectral6
mapper = linear_cmap(field_name='Points', palette=Spectral6 ,low=min(decathlon['Points']) ,high=max(decathlon['Points']))

p = figure(plot_width=700, plot_height=450, title = "Decathlon: Discus x Javeline")
p.scatter('Discus','Javeline',source=decath,fill_alpha=0.6, line_color=mapper,color=mapper,size=10)
p.xaxis.axis_label = 'Discus'
p.yaxis.axis_label = 'Javeline'
color_bar = ColorBar(color_mapper=mapper['transform'], width=8,  location=(0,0),title="Points")

p.add_layout(color_bar, 'right')

show(p)

p = figure(plot_width=700, plot_height=450, title = "Decathlon: Discus x Javeline")
p.scatter('Discus','Javeline',source=decath,fill_alpha=0.6, line_color=mapper,color=mapper,size='Points')
p.xaxis.axis_label = 'Discus'
p.yaxis.axis_label = 'Javeline'
show(p)
from bokeh.models import LinearInterpolator
decath
size_mapper=LinearInterpolator(
    x=[decathlon.Points.min(),decathlon.Points.max()],
    y=[5,50]
)
p = figure(plot_width=700, plot_height=450, title = "Decathlon: Discus x Javeline",
          toolbar_location=None,
          tools="hover", tooltips="@Athlets: @Points")
p.scatter('Discus','Javeline',
          source=decathlon,
          fill_alpha=0.6, 
          fill_color=index_cmap,
          size={'field':'Points','transform': size_mapper},
          legend='Competition'
         )
p.xaxis.axis_label = 'Discus'
p.yaxis.axis_label = 'Javeline'
p.legend.location = "top_left"
show(p)