!pip install plotly==4.7
import plotly.express as px

from skimage import data



img = data.chelsea() # or any image represented as a numpy array

fig = px.imshow(img)



# Define dragmode, newshape parameters, amd add modebar buttons

fig.update_layout(

    dragmode='drawrect', # define dragmode

    newshape=dict(line_color='black', fillcolor='black'))



# Add modebar buttons

fig.show(config={

    'modeBarButtonsToAdd': [

        'drawline',

        'drawopenpath',

        'drawclosedpath',

        'drawcircle',

        'drawrect',

        'eraseshape'

    ]

})