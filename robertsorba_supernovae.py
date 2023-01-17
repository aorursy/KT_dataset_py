import numpy as np

from PIL import Image



from bokeh.resources import INLINE

import bokeh.io

bokeh.io.output_notebook(INLINE)

from bokeh.layouts import column, row, gridplot

from bokeh.models import CustomJS, Slider, Button, Label

from bokeh.plotting import ColumnDataSource, figure, output_file, show
# Open image, and make sure it's RGB*A*

jpgimg = Image.open('/kaggle/input/supernova-images/Crab_nebula.jpg').convert('RGBA')

xdim, ydim = jpgimg.size



# Create an array representation for the image `img`, and an 8-bit "4

# layer/RGBA" version of it `view`.

img = np.empty((ydim, xdim), dtype=np.uint32)

view = img.view(dtype=np.uint8).reshape((ydim, xdim, 4))

# Copy the RGBA image into view, flipping it so it comes right-side up

# with a lower-left origin

view[:,:,:] = np.flipud(np.asarray(jpgimg))



# Display the 32-bit RGBA image

aspectratio = ydim/xdim

p = figure(x_range=(0,xdim), y_range=(0,ydim), plot_width=640, plot_height=int(640 * aspectratio), 

           max_width=640, sizing_mode='scale_width', tooltips=[('x', '$x{int}'), ('y', '$y{int}')])

p.image_rgba(image=[img], x=0, y=0, dw=xdim, dh=ydim)



bokeh.io.show(p)
# Open image, and make sure it's RGB*A*

jpgimg = Image.open('/kaggle/input/supernova-images/SN1987a.tif').convert('RGBA')

xdim, ydim = jpgimg.size



# Create an array representation for the image `img`, and an 8-bit "4

# layer/RGBA" version of it `view`.

img = np.empty((ydim, xdim), dtype=np.uint32)

view = img.view(dtype=np.uint8).reshape((ydim, xdim, 4))

# Copy the RGBA image into view, flipping it so it comes right-side up

# with a lower-left origin

view[:,:,:] = np.flipud(np.asarray(jpgimg))



# Display the 32-bit RGBA image

aspectratio = ydim/xdim

p = figure(x_range=(0,xdim), y_range=(0,ydim), plot_width=640, plot_height=int(640 * aspectratio), 

           max_width=640, sizing_mode='scale_width', tooltips=[('x', '$x{int}'), ('y', '$y{int}')])

p.image_rgba(image=[img], x=0, y=0, dw=xdim, dh=ydim)



bokeh.io.show(p)