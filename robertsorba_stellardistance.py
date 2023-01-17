import numpy as np

from PIL import Image



from bokeh.resources import INLINE

import bokeh.io

bokeh.io.output_notebook(INLINE)

from bokeh.layouts import column, row, gridplot

from bokeh.models import CustomJS, Slider, Rect, Button, Label, VArea, Arrow, OpenHead, Ellipse

from bokeh.plotting import ColumnDataSource, figure, output_file, show
p = figure(plot_width=640, plot_height=480, max_width=640, sizing_mode='scale_width',

          x_range=[-5,5])







sun = ColumnDataSource(dict(x=[0], y=[0]))

p.circle(x='x', y='y', source=sun, fill_color='yellow', line_color='orange', size=40,

        legend_label='Sun')



earth = ColumnDataSource(dict(x=[-1], y=[0]))

p.circle(x='x', y='y', source=earth, fill_color='blue', line_color='darkblue', size=15,

        legend_label='Earth')



r = 20

theta = np.arange(50, 131) * 2 * np.pi / 360

theta2 = np.arange(50, 131, 2.5) * 2 * np.pi / 360

background = ColumnDataSource(dict(x=r * np.cos(theta), y=r * np.sin(theta)))

p.line(x='x', y='y', source=background, color='black')

bgstars = ColumnDataSource(dict(x=r * np.cos(theta2), y=r * np.sin(theta2)))

p.asterisk(x='x', y='y', source=bgstars, color='black', size=20, line_width=2,

        legend_label='Background Stars')



m = 10

b = 10

y = np.arange(0, 18.8, 0.1)

x = (y - b) / m

lineofsight = ColumnDataSource(dict(x=x, y=y))

p.line(x='x', y='y', source=lineofsight, line_dash='dashed', color='black',

      legend_label='Line of Sight')



star = ColumnDataSource(dict(x=[0], y=[10]))

p.asterisk(x='x', y='y', source=star, fill_color='magenta', line_color='magenta', size=20,

        legend_label='Foreground Star', line_width=2)



image = ColumnDataSource(dict(x=[(19 - b) / m], y=[19]))

p.asterisk(x='x', y='y', source=image, fill_color='magenta', line_color='magenta', size=20, alpha=0.5,

        legend_label='Projected Image', line_width=2)



p.axis.visible = False

p.legend.location = 'bottom_right'



slider = Slider(start=-1, end=1, value=-1, step=0.01, title="Earth's Orbital Position [AU]", 

                max_width=620)



callback = CustomJS(args=dict(slider=slider, earth=earth, image=image, los=lineofsight),

                        code="""

    const pos = slider.value;

    earth.data.x[0] = pos;

    const m = 10.0 / (-pos);

    const b = 10.0;

    const data = los.data;

    

    for (var i = 0; i < data['x'].length; i++){

        data['x'][i] = (data['y'][i] - b) / m; 

    }

    

    image.data['x'][0] = (19.0 - b) / m;

    

    earth.change.emit();

    los.change.emit();

    image.change.emit();

    """)



slider.js_on_change('value', callback)



layout = column(p, slider, sizing_mode='scale_width')

bokeh.io.show(layout)

p = figure(plot_width=640, plot_height=480, max_width=640, sizing_mode='scale_width',

          x_range=[-5,5])







sun = ColumnDataSource(dict(x=[0], y=[0]))

p.circle(x='x', y='y', source=sun, fill_color='yellow', line_color='orange', size=40,

        legend_label='Sun')



earth = ColumnDataSource(dict(x=[-1,1], y=[0,0]))

p.circle(x='x', y='y', source=earth, fill_color='blue', line_color='darkblue', size=15,

        legend_label='Earth (6 months apart)')



r = 20

theta = np.arange(50, 131) * 2 * np.pi / 360

theta2 = np.arange(50, 131, 2.5) * 2 * np.pi / 360

background = ColumnDataSource(dict(x=r * np.cos(theta), y=r * np.sin(theta)))

p.line(x='x', y='y', source=background, color='black')

bgstars = ColumnDataSource(dict(x=r * np.cos(theta2), y=r * np.sin(theta2)))

p.asterisk(x='x', y='y', source=bgstars, color='black', size=20, line_width=2,

        legend_label='Background Stars')



m = 10

b = 10

y = np.arange(0, 18.8, 0.1)

x = (y - b) / m

lineofsight1 = ColumnDataSource(dict(x=x, y=y))

p.line(x='x', y='y', source=lineofsight1, line_dash='dashed', color='black',

      legend_label='Line of Sight')

lineofsight2 = ColumnDataSource(dict(x=-x, y=y))

p.line(x='x', y='y', source=lineofsight2, line_dash='dashed', color='black')



star = ColumnDataSource(dict(x=[0], y=[10]))

p.asterisk(x='x', y='y', source=star, fill_color='magenta', line_color='magenta', size=20,

        legend_label='Foreground Star', line_width=2)



images = ColumnDataSource(dict(x=[(19 - b) / m, -(19-b)/m], y=[19, 19]))

p.asterisk(x='x', y='y', source=images, fill_color='magenta', line_color='magenta', size=20, alpha=0.5,

        legend_label='Projected Images', line_width=2)



p.axis.visible = False

p.legend.location = 'bottom_right'



slider = Slider(start=2, end=18, value=10, step=0.1, title="Distance to Star [Arbitrary Units]", 

                max_width=620)



callback = CustomJS(args=dict(slider=slider, star=star, images=images, los1=lineofsight1,

                             los2=lineofsight2),

                        code="""

    const pos = slider.value;

    star.data.y[0] = pos;

    const m = pos - 0.0;

    const b = pos - 0.0;

    const data = los1.data;

    const data2 = los2.data;

    

    for (var i = 0; i < data['x'].length; i++){

        data['x'][i] = (data['y'][i] - b) / m;

        data2['x'][i] = -data['x'][i]

    }

    

    images.data['x'][0] = (19.0 - b) / m;

    

    images.data['x'][1] = -(19.0 - b) / m;

    

    star.change.emit();

    los1.change.emit();

    los2.change.emit();

    images.change.emit();

    """)



slider.js_on_change('value', callback)



layout = column(p, slider, sizing_mode='scale_width')

bokeh.io.show(layout)

# Open image, and make sure it's RGB*A*

jpgimg = Image.open('/kaggle/input/flag-images/Flag_0m.JPG').convert('RGBA')

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

           max_width=640, sizing_mode='scale_width', tooltips=[('x', '$x{int}'), ('y', '$y{int}')],

          title='O meters')

p.image_rgba(image=[img], x=0, y=0, dw=xdim, dh=ydim)



bokeh.io.show(p)

# Open image, and make sure it's RGB*A*

jpgimg = Image.open('/kaggle/input/flag-images/Flag_10m.JPG').convert('RGBA')

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

           max_width=640, sizing_mode='scale_width', tooltips=[('x', '$x{int}'), ('y', '$y{int}')],

          title='10 meters')

p.image_rgba(image=[img], x=0, y=0, dw=xdim, dh=ydim)



bokeh.io.show(p)
