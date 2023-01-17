import ipywidgets as widgets
widgets.FileUpload(

    accept='',  # Accepted file extension e.g. '.txt', '.pdf', 'image/*', 'image/*,.pdf'

    multiple=False  # True to accept multiple files upload else False

)
from IPython.display import YouTubeVideo



out = widgets.Output(layout={'border': '1px solid black'})



with out:

    display(YouTubeVideo('eWzY2nGfkXk'))