!apt-get install libxss1

!apt -y install libgconf2-4

!conda install -y -c plotly plotly-orca # for exporting plots
import skimage.io

import numpy as np

import matplotlib.pyplot as plt
d = skimage.io.imread('../input/3d_scenes/3d_scenes/img00003.png')

img = skimage.io.imread('../input/3d_scenes/3d_scenes/img00003.tiff')

d = d.max()-d
fig, ax = plt.subplots(1,2, figsize=(20,10))

ax[0].text(50, 100, 'original image', fontsize=16, bbox={'facecolor': 'white', 'pad': 6})

ax[0].imshow(img)



ax[1].text(50, 100, 'depth map', fontsize=16, bbox={'facecolor': 'white', 'pad': 6})

ax[1].imshow(d)
d = np.flipud(d)

img = np.flipud(img)
from PIL import Image as PImage

import plotly.graph_objects as go

from plotly import tools

import plotly.offline

def create_rgb_surface(rgb_img, depth_img, depth_cutoff=20, **kwargs):

    rgb_img = rgb_img.swapaxes(0, 1)[:, ::-1]

    depth_img = depth_img.swapaxes(0, 1)[:, ::-1]

    eight_bit_img = PImage.fromarray(rgb_img).convert('P', palette='WEB', dither=None)

    idx_to_color = np.array(eight_bit_img.getpalette()).reshape((-1, 3))

    colorscale=[[i/255.0, "rgb({}, {}, {})".format(*rgb)] for i, rgb in enumerate(idx_to_color)]

    depth_map = depth_img.copy().astype('float')

    depth_map[depth_map<depth_cutoff] = np.nan

    return go.Surface(

        z=depth_map,

        surfacecolor=np.array(eight_bit_img),

        cmin=0, 

        cmax=255,

        colorscale=colorscale,

        showscale=False,

        **kwargs

    )
fig = go.Figure(

    data=[create_rgb_surface(img, 

                             d,

                             10,

                             contours_z=dict(show=True, project_z=True, highlightcolor="limegreen"),

                             opacity=1.0

                            )],

    layout_title_text="3D Surface"

)

#fig.update_layout(scene_camera_eye=dict(x=0, y=-1, z=.5), scene_camera_up=dict(x=0,y=0,z=1))

fig
from IPython.display import FileLink

plotly.offline.plot(fig, filename = 'rgbd.html', auto_open=False)

FileLink('rgbd.html')
fig.write_image("figure_{:04d}.png".format(0), height=2048, width=2048)