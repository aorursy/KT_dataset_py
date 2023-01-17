import ipywidgets as widgets

from IPython.display import display,HTML
file='../input/html-recipes/sklearn_clusters.html'

with open(file,'r') as f:

    html_str=f.read()

    f.close()

HTML(html_str)
progress=widgets.IntProgress()

text=widgets.IntText(min=0,max=100)

widgets.jslink((progress,'value'),

               (text,'value'))

display(text,progress)
l=[]

def on_value_change(change):

    l.append(change['new'])

slider=widgets.IntSlider(min=1,max=100,step=1,

                         continuous_update=False)

play=widgets.Play(min=1,interval=1000,

                  description="Click Play")

widgets.jslink((play,'value'),(slider,'value'))

slider.observe(on_value_change,names='value')

display(play,slider)
out=widgets.Output()

def value_change(change):

    with out: change['new']

progress=widgets.IntProgress(min=1,max=100,step=1,

                             continuous_update=False)

play=widgets.Play(min=1,interval=1000,

                  description="Click Play")

widgets.jslink((play,'value'),(progress,'value'))

progress.observe(value_change,names='value')

widgets.VBox([play,progress,out])
out=widgets.Output()

def value_change(change):

    with out: change['new']

slider=widgets.IntSlider(min=1,max=10)

whtml=widgets.HTML(value='%d'%1,

                   description='HTML')

widgets.jslink((whtml,'value'),(slider,'value'))

slider.observe(value_change,names='value')

display(slider,whtml,out)