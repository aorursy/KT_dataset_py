import matplotlib.pyplot as plt

import numpy as np



import numpy as np

import matplotlib.pyplot as plt



from matplotlib import animation, rc

from IPython.display import HTML
import numpy as np

import ipywidgets as widgets



a = widgets.FloatText()

b = widgets.FloatSlider(min=10, max=90, step=1, value=50)

display(a,b)



mylink = widgets.jslink((a,'value'), (b, 'value'))
def create_circular_mask(h, w, center=None, radius=None):



    if center is None: # use the middle of the image

        center = (int(w/2), int(h/2))

    if radius is None: # use the smallest distance between the center and image walls

        radius = min(center[0], center[1], w-center[0], h-center[1])



    Y, X = np.ogrid[:h, :w]

    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)



    mask = dist_from_center <= radius

    return mask
def lightcurve(star_radius, planet_radius, imsize=(200, 200)):

    area_star_fractional = []

    field = np.zeros(imsize)

    star = create_circular_mask(imsize[0], imsize[1], radius=star_radius)

    field[star] = 1.0

    area_star_total = np.sum(star)



    for x in np.arange(imsize[0]):

        planet = create_circular_mask(imsize[0], imsize[1], center=(x, imsize[1]/2), radius=planet_radius)

        field[star] = 1.0

        field[planet] = 0.0

        area_star_fractional.append(np.sum(field))



    area_star = np.array(area_star_fractional)/area_star_total

#     plt.imshow(star)



    return np.arange(imsize[0]), area_star

imsize = 800, 400

x, y = lightcurve(100, 70, imsize=imsize)
import pandas as pd

imsize = 400, 400

pr, sr = 80, 100

time, flux = lightcurve(sr, pr, imsize=imsize)

dfs = pd.DataFrame({'time': time, 'flux': flux, 'star_radius': sr, 'planet_radius': pr})

print(pr)
import plotly.express as px

import plotly.graph_objects as go



fig = go.Figure(

    data=[go.Scatter(x=dfs.time.values, y=dfs.flux.values,

                     mode="lines",

                     line=dict(width=2, color="blue")),

          go.Scatter(x=dfs.time.values, y=dfs.flux.values,

                     mode="lines",

                     line=dict(width=2, color="blue"))],

    layout=go.Layout(

        width=1000,

        xaxis=dict(range=[0, imsize[0]], autorange=False, zeroline=False),

        yaxis=dict(range=[0, 1.3], autorange=False, zeroline=False),

        title_text="Exoplanet Transit", hovermode="closest",

        xaxis_title='Time',

        yaxis_title='Flux',

        updatemenus=[dict(type="buttons",

                          buttons=[dict(label="Play",

                                        method="animate",

                                        args=[None])])]),

    frames=[go.Frame(

        data=[go.Scatter(

            x=[dfs.time.values[::10][k]],

            y=[dfs.flux.values[::10][k]],

            mode="markers",

            marker=dict(color="red", size=10))])



        for k in range(dfs.time.values[::10].shape[0])]

)



fig.show()