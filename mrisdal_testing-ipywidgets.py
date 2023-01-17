import ipywidgets as widgets
widgets.IntSlider()
play = widgets.Play(

    value=50,

    min=0,

    max=100,

    step=1,

    interval=500,

    description="Press play",

    disabled=False

)

slider = widgets.IntSlider()

widgets.jslink((play, 'value'), (slider, 'value'))

widgets.HBox([play, slider])
widgets.DatePicker(

    description='Pick a Date',

    disabled=False

)
widgets.FileUpload(

    accept='',  # Accepted file extension e.g. '.txt', '.pdf', 'image/*', 'image/*,.pdf'

    multiple=False  # True to accept multiple files upload else False

)
from IPython.display import YouTubeVideo



out = widgets.Output(layout={'border': '1px solid black'})



with out:

    display(YouTubeVideo('eWzY2nGfkXk'))