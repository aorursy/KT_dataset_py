import numpy as np

from bokeh.resources import INLINE

import bokeh.io

bokeh.io.output_notebook(INLINE)

from bokeh.layouts import column, row

from bokeh.models import CustomJS, Slider, Text, Label, Arrow, OpenHead

from bokeh.plotting import ColumnDataSource, figure, output_file, show

from bokeh.palettes import Colorblind



# read in data

massdata = np.loadtxt("/kaggle/input/phys1401-shm/SHM_MassData.csv", delimiter=",",

                      unpack=True)

massdata[2] += np.mean(massdata[1]) - np.mean(massdata[2])

massdata[3] += np.mean(massdata[1]) - np.mean(massdata[3])



# set up plot

l1source = ColumnDataSource(data=dict(x=massdata[0], y=massdata[1]))

l2source = ColumnDataSource(data=dict(x=massdata[0], y=massdata[2]))

l3source = ColumnDataSource(data=dict(x=massdata[0], y=massdata[3]))

dt1source = ColumnDataSource(data=dict(x=[0,0], y=[0.29,0.5]))

dt2source = ColumnDataSource(data=dict(x=[16,16], y=[0.29,0.5]))



plot = figure(y_range=(0.29, 0.5), plot_width=640, plot_height=480, 

              sizing_mode='scale_width')

l1 = plot.line('x', 'y', source=l1source, line_width=3, 

          legend_label='150g', color=Colorblind[4][0])

l2 = plot.line('x', 'y', source=l2source, line_width=3, 

          legend_label='250g', color=Colorblind[4][1], line_dash='dashed')

l3 = plot.line('x', 'y', source=l3source, line_width=3, 

          legend_label='350g', color=Colorblind[4][3], line_dash='dotdash')

plot.line('x', 'y', source=dt1source, line_width=3, color='black')

plot.line('x', 'y', source=dt2source, line_width=3, color='black')

measure = Label(x=7, y=0.47, text='16.00 s', render_mode='css',

               text_color='gray')

plot.add_layout(measure)

arrow1 = Arrow(end=OpenHead(line_color="gray", line_width=2),

                   x_start=8, y_start=0.47, x_end=0.1, y_end=0.47,

                  line_width=2, line_color='gray')

arrow2 = Arrow(end=OpenHead(line_color="gray", line_width=2),

                   x_start=8, y_start=0.47, x_end=15.9, y_end=0.47,

                  line_width=2, line_color='gray')

plot.add_layout(arrow1)

plot.add_layout(arrow2)

l2.visible = False

l3.visible = False

plot.legend.location = "top_right"

plot.legend.click_policy="hide"



# set up interactivity with some javascript

dt1_slider = Slider(start=0, end=7.8, value=0, step=.05, title="Slider 1")

dt2_slider = Slider(start=8.2, end=20, value=16, step=.05, title="Slider 2")



callback = CustomJS(args=dict(source1=dt1source, slider1=dt1_slider, 

                              source2=dt2source, slider2=dt2_slider, 

                              m=measure, arr1=arrow1, arr2=arrow2),

                    code="""

    const data1 = source1.data;

    const pos1 = slider1.value;

    const data2 = source2.data;

    const pos2 = slider2.value;

    

    const x = data1['x'];

    for (var i = 0; i < x.length; i++) {

        x[i] = pos1;

    }

    

    const p = data2['x'];

    for (var i = 0; i < p.length; i++) {

        p[i] = pos2;

    }

    

    var deltatime = pos2 - pos1;

    var avgtime = (pos1 + pos2) * 0.5;

    m.text = deltatime.toFixed(2) + " s";

    m.x = avgtime - 0.5

    arr1.x_end = pos1 + 0.1;

    arr2.x_end = pos2 - 0.1;

    arr1.x_start = avgtime;

    arr2.x_start = avgtime;

    source1.change.emit();

    source2.change.emit();

    m.change.emit();

    arr1.change.emit();

    arr2.change.emit();

""")



dt1_slider.js_on_change('value', callback)

dt2_slider.js_on_change('value', callback)

#phase_slider.js_on_change('value', callback)

#offset_slider.js_on_change('value', callback)



layout = column(

    plot,

    dt1_slider,

    dt2_slider,

    sizing_mode="scale_width"

)



bokeh.io.show(layout)



# read in data

ampdata = np.loadtxt("/kaggle/input/phys1401-shm/SHM_AmpData.csv", delimiter=",",

                      unpack=True)



# set up plot

l1source = ColumnDataSource(data=dict(x=ampdata[0][ampdata[1] > 0], 

                                      y=ampdata[1][ampdata[1] > 0]))

l2source = ColumnDataSource(data=dict(x=ampdata[0][ampdata[1] > 0], 

                                      y=ampdata[2][ampdata[1] > 0]))



dt1source = ColumnDataSource(data=dict(x=[0,0], y=[0.24,0.57]))

dt2source = ColumnDataSource(data=dict(x=[20,20], y=[0.24,0.57]))



plot = figure(y_range=(0.24, 0.57), plot_width=640, plot_height=480, 

              sizing_mode='scale_width')

l1 = plot.line('x', 'y', source=l1source, line_width=3, 

          legend_label='Low Amp', color=Colorblind[4][0])

l2 = plot.line('x', 'y', source=l2source, line_width=3, 

          legend_label='Hi Amp', color=Colorblind[4][1], line_dash='dashed')



plot.line('x', 'y', source=dt1source, line_width=3, color='black')

plot.line('x', 'y', source=dt2source, line_width=3, color='black')

measure = Label(x=9, y=0.54, text='20.00 s', render_mode='css',

               text_color='gray')

plot.add_layout(measure)

arrow1 = Arrow(end=OpenHead(line_color="gray", line_width=2),

                   x_start=10, y_start=0.54, x_end=0.1, y_end=0.54,

                  line_width=2, line_color='gray')

arrow2 = Arrow(end=OpenHead(line_color="gray", line_width=2),

                   x_start=10, y_start=0.54, x_end=19.9, y_end=0.54,

                  line_width=2, line_color='gray')

plot.add_layout(arrow1)

plot.add_layout(arrow2)

l2.visible = False



plot.legend.location = "top_right"

plot.legend.click_policy="hide"



# set up interactivity with some javascript

dt1_slider = Slider(start=0, end=9.8, value=0, step=.05, title="Slider 1")

dt2_slider = Slider(start=10.2, end=25, value=20, step=.05, title="Slider 2")



callback = CustomJS(args=dict(source1=dt1source, slider1=dt1_slider, 

                              source2=dt2source, slider2=dt2_slider, 

                              m=measure, arr1=arrow1, arr2=arrow2),

                    code="""

    const data1 = source1.data;

    const pos1 = slider1.value;

    const data2 = source2.data;

    const pos2 = slider2.value;

    

    const x = data1['x'];

    for (var i = 0; i < x.length; i++) {

        x[i] = pos1;

    }

    

    const p = data2['x'];

    for (var i = 0; i < p.length; i++) {

        p[i] = pos2;

    }

    

    var deltatime = pos2 - pos1;

    var avgtime = (pos1 + pos2) * 0.5;

    m.text = deltatime.toFixed(2) + " s";

    m.x = avgtime - 0.5

    arr1.x_end = pos1 + 0.1;

    arr2.x_end = pos2 - 0.1;

    arr1.x_start = avgtime;

    arr2.x_start = avgtime;

    source1.change.emit();

    source2.change.emit();

    m.change.emit();

    arr1.change.emit();

    arr2.change.emit();

""")



dt1_slider.js_on_change('value', callback)

dt2_slider.js_on_change('value', callback)

#phase_slider.js_on_change('value', callback)

#offset_slider.js_on_change('value', callback)



layout = column(

    plot,

    dt1_slider,

    dt2_slider,

    sizing_mode="scale_width"

)



bokeh.io.show(layout)
# set up plot



source = ColumnDataSource(data=dict(x=ampdata[0], y=ampdata[2]))



dt1source = ColumnDataSource(data=dict(x=[800,800], y=[0.24,0.57]))

dt2source = ColumnDataSource(data=dict(x=[820,820], y=[0.24,0.57]))



plot = figure(y_range=(0.24, 0.57), plot_width=640, plot_height=480, 

              sizing_mode='scale_width')

l = plot.line('x', 'y', source=source, line_width=3, 

          legend_label='Hi Amp', color=Colorblind[4][1], line_dash='dashed')



plot.line('x', 'y', source=dt1source, line_width=3, color='black')

plot.line('x', 'y', source=dt2source, line_width=3, color='black')

measure = Label(x=809, y=0.54, text='20.00 s', render_mode='css',

               text_color='gray')

plot.add_layout(measure)

arrow1 = Arrow(end=OpenHead(line_color="gray", line_width=2),

                   x_start=810, y_start=0.54, x_end=800.1, y_end=0.54,

                  line_width=2, line_color='gray')

arrow2 = Arrow(end=OpenHead(line_color="gray", line_width=2),

                   x_start=810, y_start=0.54, x_end=819.9, y_end=0.54,

                  line_width=2, line_color='gray')

plot.add_layout(arrow1)

plot.add_layout(arrow2)

plot.legend.location = "top_right"

plot.legend.click_policy="hide"



# set up interactivity with some javascript

dt1_slider = Slider(start=800, end=809.8, value=800, step=.05, title="Slider 1")

dt2_slider = Slider(start=810.2, end=825, value=820, step=.05, title="Slider 2")



callback = CustomJS(args=dict(source1=dt1source, slider1=dt1_slider, 

                              source2=dt2source, slider2=dt2_slider, 

                              m=measure, arr1=arrow1, arr2=arrow2),

                    code="""

    const data1 = source1.data;

    const pos1 = slider1.value;

    const data2 = source2.data;

    const pos2 = slider2.value;

    

    const x = data1['x'];

    for (var i = 0; i < x.length; i++) {

        x[i] = pos1;

    }

    

    const p = data2['x'];

    for (var i = 0; i < p.length; i++) {

        p[i] = pos2;

    }

    

    var deltatime = pos2 - pos1;

    var avgtime = (pos1 + pos2) * 0.5;

    m.text = deltatime.toFixed(2) + " s";

    m.x = avgtime - 0.5

    arr1.x_end = pos1 + 0.1;

    arr2.x_end = pos2 - 0.1;

    arr1.x_start = avgtime;

    arr2.x_start = avgtime;

    source1.change.emit();

    source2.change.emit();

    m.change.emit();

    arr1.change.emit();

    arr2.change.emit();

""")



dt1_slider.js_on_change('value', callback)

dt2_slider.js_on_change('value', callback)

#phase_slider.js_on_change('value', callback)

#offset_slider.js_on_change('value', callback)



layout = column(

    plot,

    dt1_slider,

    dt2_slider,

    sizing_mode="scale_width"

)



bokeh.io.show(layout)
# read in data

springdata = np.loadtxt("/kaggle/input/phys1401-shm/SHM_SpringData.csv", 

                        delimiter=",",unpack=True)

springdata[1] += np.mean(massdata[1]) - np.mean(springdata[1])

springdata[2] += np.mean(massdata[1]) - np.mean(springdata[2])



# set up plot

l1source = ColumnDataSource(data=dict(x=springdata[0], 

                                      y=springdata[1]))

l2source = ColumnDataSource(data=dict(x=springdata[0], 

                                      y=springdata[2]))

l3source = ColumnDataSource(data=dict(x=massdata[0], 

                                      y=massdata[2]))



dt1source = ColumnDataSource(data=dict(x=[0,0], y=[0.24,0.57]))

dt2source = ColumnDataSource(data=dict(x=[20,20], y=[0.24,0.57]))



plot = figure(y_range=(0.24, 0.57), plot_width=640, plot_height=480, 

              sizing_mode='scale_width', x_range=(-1, 25))

l1 = plot.line('x', 'y', source=l1source, line_width=3, 

          legend_label='Series', color=Colorblind[4][0])

l2 = plot.line('x', 'y', source=l2source, line_width=3, 

          legend_label='Parallel', color=Colorblind[4][1], line_dash='dashed')

l2 = plot.line('x', 'y', source=l3source, line_width=3, 

          legend_label='Normal', color=Colorblind[4][3], line_dash='dotdash')



plot.line('x', 'y', source=dt1source, line_width=3, color='black')

plot.line('x', 'y', source=dt2source, line_width=3, color='black')

measure = Label(x=9, y=0.54, text='20.00 s', render_mode='css',

               text_color='gray')

plot.add_layout(measure)

arrow1 = Arrow(end=OpenHead(line_color="gray", line_width=2),

                   x_start=10, y_start=0.54, x_end=0.1, y_end=0.54,

                  line_width=2, line_color='gray')

arrow2 = Arrow(end=OpenHead(line_color="gray", line_width=2),

                   x_start=10, y_start=0.54, x_end=19.9, y_end=0.54,

                  line_width=2, line_color='gray')

plot.add_layout(arrow1)

plot.add_layout(arrow2)

l2.visible = False



plot.legend.location = "top_right"

plot.legend.click_policy="hide"



# set up interactivity with some javascript

dt1_slider = Slider(start=0, end=9.8, value=0, step=.05, title="Slider 1")

dt2_slider = Slider(start=10.2, end=25, value=20, step=.05, title="Slider 2")



callback = CustomJS(args=dict(source1=dt1source, slider1=dt1_slider, 

                              source2=dt2source, slider2=dt2_slider, 

                              m=measure, arr1=arrow1, arr2=arrow2),

                    code="""

    const data1 = source1.data;

    const pos1 = slider1.value;

    const data2 = source2.data;

    const pos2 = slider2.value;

    

    const x = data1['x'];

    for (var i = 0; i < x.length; i++) {

        x[i] = pos1;

    }

    

    const p = data2['x'];

    for (var i = 0; i < p.length; i++) {

        p[i] = pos2;

    }

    

    var deltatime = pos2 - pos1;

    var avgtime = (pos1 + pos2) * 0.5;

    m.text = deltatime.toFixed(2) + " s";

    m.x = avgtime - 0.5

    arr1.x_end = pos1 + 0.1;

    arr2.x_end = pos2 - 0.1;

    arr1.x_start = avgtime;

    arr2.x_start = avgtime;

    source1.change.emit();

    source2.change.emit();

    m.change.emit();

    arr1.change.emit();

    arr2.change.emit();

""")



dt1_slider.js_on_change('value', callback)

dt2_slider.js_on_change('value', callback)

#phase_slider.js_on_change('value', callback)

#offset_slider.js_on_change('value', callback)



layout = column(

    plot,

    dt1_slider,

    dt2_slider,

    sizing_mode="scale_width"

)



bokeh.io.show(layout)