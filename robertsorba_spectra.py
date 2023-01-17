from astropy.io import fits

import numpy as np

from bokeh.resources import INLINE

import bokeh.io

bokeh.io.output_notebook(INLINE)

from bokeh.layouts import column, row

from bokeh.models import CustomJS, Slider, RadioButtonGroup, Button, Label, VArea, Arrow, OpenHead

from bokeh.plotting import ColumnDataSource, figure, output_file, show

#from bokeh.palettes import Colorblind
h = 6.62606896e-34 # planck constant

c = 2.99792458e8 # speed of light

k = 1.3806504e-23 #boltzmann constant




step = 0.01

xlim = [0.8, 1.61]

lamb = np.arange(xlim[0], xlim[1], step) * 10**(-6) 



B = []

temps = [2300, 2400, 2500, 2600, 2700]

for T in temps:

    flux = 2 * h * c**2 / lamb**5 * 1 / (np.expm1(h * c / k / T / lamb))

    B.append(flux / 10**10)



plot = figure(plot_width=640, plot_height=480, sizing_mode='scale_width', x_range=[0.75, 1.65], 

             y_range=[14, 61])



hsource = ColumnDataSource(data=dict(x=[0.4, 2.6], y=[B[0][0]] * 2))

hline = plot.line('x', 'y', source=hsource,

                 color='orange', line_dash='dashed', line_width=3)



vsource = ColumnDataSource(data=dict(x=[lamb[0] * 10**6] * 2, y=[0, 61]))

vline = plot.line('x', 'y', source=vsource,

                 color='orange', line_dash='dashed', line_width=3)



bbsource = ColumnDataSource(data=dict(x=lamb * 10**6, y=B[0]))

blackbody = plot.square('x', 'y', source=bbsource, size=4)



hlsource = ColumnDataSource(data=dict(x=[lamb[0] * 10**6], y=[B[0][0]]))

highlight = plot.square('x', 'y', source=hlsource, size=20, fill_color='orange', 

                        fill_alpha=0.5)



coordstring = "x,y = (%.2f, %.3f)" % (lamb[0] * 10**6, B[0][0]) 

point = Label(x=1.21, y=16, text=coordstring, render_mode='css',

               text_color='black')

plot.add_layout(point)



plot.xaxis.axis_label = "Wavelength [μm]"

plot.yaxis.axis_label = "Intensity [Scaled Units]"

plot.axis.axis_label_text_font_size = "12pt"

plot.axis.axis_label_text_font_style = 'bold'



slider = Slider(start=xlim[0], end=xlim[1], value=xlim[0], step=step, title="Wavelength")



callback = CustomJS(args=dict(hlsource=hlsource, slider=slider, bbsource=bbsource, bottomx=xlim[0],

                              step=step, hsource=hsource, vsource=vsource, p=point),

                    code="""

    const hldata = hlsource.data;

    const pos = slider.value;

    const bbdata = bbsource.data;

    const hdata = hsource.data;

    const vdata = vsource.data;

    var index = 0;

    

    

    index = Math.round(pos / step - bottomx / step);

    for (var i = 0; i < vdata['x'].length; i++) {

        vdata['x'][i] = pos;

    }

    

    

    const x = hldata['x'];

    x[0] = pos;

    

    const y = hldata['y'];

    y[0] = bbdata['y'][index];

    

    p.text = 'x,y = (' + pos.toFixed(2) + ', ' + y[0].toFixed(3) + ')';

    

    for (var i = 0; i < hdata['y'].length; i++) {

        hdata['y'][i] = bbdata['y'][index];

    }

    

    hlsource.change.emit();

    hsource.change.emit();

    vsource.change.emit();

""")



slider.js_on_change('value', callback)



TempLabels = ["2300 K", "2400 K", "2500 K", "2600 K", "2700 K"]



radio = RadioButtonGroup(labels=TempLabels, active=0, sizing_mode="scale_width")

radioCallback = CustomJS(args=dict(radio=radio, bbsource=bbsource, B=B, step=step, bottomx=xlim[0],

                                  hlsource=hlsource, hsource=hsource, slider=slider, p=point),

                        code="""

    const data = bbsource.data;

    const Bindex = radio.active;

    const pos = slider.value;

    const lamdex = Math.round(pos / step - bottomx / step);

    

    for (var i = 0; i < data['y'].length; i++){

        data['y'][i] = B[Bindex][i]; 

    }

    

    for (var i = 0; i < hsource.data['y'].length; i++){

        hsource.data['y'][i] = B[Bindex][lamdex]; 

    }

    

    for (var i = 0; i < hlsource.data['y'].length; i++){

        hlsource.data['y'][i] = B[Bindex][lamdex]; 

    }

    

    const y = hlsource.data['y'];

    p.text = 'x,y = (' + pos.toFixed(2) + ', ' + y[0].toFixed(3) + ')';

    

    bbsource.change.emit();

    hlsource.change.emit();

    hsource.change.emit();

                        """)

radio.js_on_click(radioCallback)



plot2 = figure(plot_width=640, plot_height=480, sizing_mode='scale_width',

              x_range=[3.5e-4, 4.6e-4], y_range=[0.5, 2])



plot2.xaxis.axis_label = "1 / T[K]"

plot2.yaxis.axis_label = "Max Wavelength [μm]"

plot2.axis.axis_label_text_font_size = "12pt"

plot2.axis.axis_label_text_font_style = 'bold'



measure = Label(x=3.8e-4, y=1.8, text="λmax = ???? [mK] / T", render_mode='css',

               text_color='gray')

plot2.add_layout(measure)



fitsource = ColumnDataSource(data=dict(x=[],y=[]))

plot2.line('x', 'y', source=fitsource, line_width=4, color='gray', line_dash='dashed')



circlesource = ColumnDataSource(data=dict(x=[],y=[]))

plot2.circle('x', 'y', source=circlesource, size=15, fill_color='red')



button = Button(label="Record Max Wavelength!", button_type="success", sizing_mode="scale_width")



buttonCallback = CustomJS(args=dict(radio=radio, vsource=vsource, temps=temps, m=measure,

                                    csource=circlesource, fitsource=fitsource),

                         code="""

    const index = radio.active;

    const lammax = vsource.data['x'][0];

    const x = csource.data['x'];

    const y = csource.data['y'];

    

    x.push(1.0 / temps[index]);

    y.push(lammax);

    

    csource.change.emit();

    

    if (x.length > 1){

        var sum_x = 0;

        var sum_y = 0;

        var sum_xy = 0;

        var sum_xx = 0;

        var sum_yy = 0;

        

        for (var i = 0; i < y.length; i++) {

            sum_x += x[i];

            sum_y += y[i];

            sum_xy += (x[i]*y[i]);

            sum_xx += (x[i]*x[i]);

            sum_yy += (y[i]*y[i]);

        } 

        

        var slope = (y.length * sum_xy - sum_x * sum_y) / (y.length * sum_xx - sum_x * sum_x);

        var intercept = (sum_y - slope * sum_x) / y.length;

        

        fitsource.data['x'] = [3.5e-4, 4.6e-4];

        fitsource.data['y'] = [slope * 3.5e-4 + intercept, slope * 4.6e-4 + intercept];

        

        var formattedSlope = slope / 1000.0;

        m.text = 'λmax = ' + formattedSlope.toFixed(1) + 'e-3 [mK] / T';

    }

    else{

        fitsource.data['x'] = [];

        fitsource.data['y'] = [];

        m.text = 'λmax = ???? / T';

    }

    

    fitsource.change.emit();  

    

                         """)

button.js_on_click(buttonCallback)



backbutton = Button(label="Remove Last Datapoint", button_type="danger", sizing_mode="scale_width")



backCallback = CustomJS(args=dict(csource=circlesource, fitsource=fitsource, m=measure),

                         code=""" 

    const x = csource.data['x'];

    const y = csource.data['y'];

    x.pop();

    y.pop();

    

    csource.change.emit()

    

    if (x.length > 1) {

        var sum_x = 0;

        var sum_y = 0;

        var sum_xy = 0;

        var sum_xx = 0;

        var sum_yy = 0;

        

        for (var i = 0; i < y.length; i++) {

            sum_x += x[i];

            sum_y += y[i];

            sum_xy += (x[i]*y[i]);

            sum_xx += (x[i]*x[i]);

            sum_yy += (y[i]*y[i]);

        } 

        

        var slope = (y.length * sum_xy - sum_x * sum_y) / (y.length * sum_xx - sum_x * sum_x);

        var intercept = (sum_y - slope * sum_x) / y.length;

        

        fitsource.data['x'] = [3.5e-4, 4.6e-4];

        fitsource.data['y'] = [slope * 3.5e-4 + intercept, slope * 4.6e-4 + intercept];

        var formattedSlope = slope / 1000.0;

        m.text = 'λmax = ' + formattedSlope.toFixed(1) + 'e-3 [mK] / T';

    } else {

        fitsource.data['x'] = [];

        fitsource.data['y'] = [];

        m.text = 'λmax = ???? / T';

    }

    

    fitsource.change.emit(); 

                         """)

backbutton.js_on_click(backCallback)





layout = row(column(plot, slider,radio), column(plot2, button, backbutton),sizing_mode="scale_width")



bokeh.io.show(layout)
#from bokeh.models import HoverTool



#hover_tool = HoverTool(tooltips=[('name', '$name'),('Energy', '$x')], renderers=[])  # instantiate HoverTool without its renderers

#print(hover_tool.renderers)



plot = figure(plot_width=400, plot_height=400, 

              y_range=[-15, 1], x_range=[0, 10],

             min_border=0, toolbar_location=None)



Eo = -13.6

for n in range(10):

    lvl = Eo / (n + 1)**2

    #print(lvl)

    if n < 4:

        plot.line([0, 10], [lvl, lvl], color='black')

        label = Label(x=8.5, y=lvl, text="n = %i" % (n + 1), render_mode='css',

               text_color='black')

        plot.add_layout(label)

        

    else:

        plot.line([0, 8], [lvl, lvl], color='black')



#for n in [3,4,5]:

#    deltaE = (Eo / n**2 - Eo / 4) * 1.602e-19

#    wl = h * c / deltaE

#    print(wl * 10**9)

    

#for n in [2,3,4,5]:

#    deltaE = (Eo / n**2 - Eo / 1) * 1.602e-19

#    wl = h * c / deltaE

#    print(deltaE)

    

    

    



source = ColumnDataSource(dict(x=[0,8], y1=[0,0], y2=[1,1]))

glyph = VArea(x='x', y1='y1', y2='y2', hatch_color="#f46d43", hatch_pattern='x',

             fill_color='white', hatch_scale=20)

plot.add_glyph(source, glyph)

     

label = Label(x=2.5, y=0.1, text="Free Electrons", render_mode='css',

               text_color='black', text_font_style='bold')

plot.add_layout(label)





for i in range(7):

    greek = 'αβγδ'

    if i < 4:

        y_start = Eo

        y_end = Eo/(i+2)**2

        name = 'Ly' + greek[i]

    else:

        j = i - 4

        y_start = Eo / 4

        y_end = Eo/(j+3)**2

        name= 'H' + greek[j]

    a = Arrow(end=OpenHead(line_color="dodgerblue", line_width=2, size=15),

                   x_start=i+1, y_start=y_start+0.2, x_end=i+1, y_end=y_end,

                  line_width=2, line_color='dodgerblue')

    lbl = Label(x=i+0.75, y=y_start-1, text=name, render_mode='css',

               text_color='black')

    plot.add_layout(a)

    plot.add_layout(lbl)

#    line = plot.line([i+1, i+1], [y_start, y_end], line_width=10, color=None, name=name)

#    hover_tool.renderers.append(line)

    

#print(hover_tool.renderers)

#plot.add_tools(hover_tool)



plot.circle(np.arange(7)+1, [Eo,Eo,Eo,Eo, Eo/4, Eo/4, Eo/4], 

            fill_color='white', line_color='black', size=10)







plot.axis.visible = False

plot.xgrid.visible = False

plot.ygrid.visible = False



bokeh.io.show(plot)



spec = fits.open("/kaggle/input/a6-empirical-stellar-spectrum/A6_0.0_Dwarf.fits")

flux = spec[1].data.field('Flux')

wavelength = 10.0**spec[1].data.field('LogLam') # Angstroms

wavelength = wavelength / 10 # nm

err = spec[1].data.field('PropErr')





plot = figure(x_range=(365, 700), plot_width=640, plot_height=480, 

              sizing_mode='scale_width', tooltips=[('wavelngth [nm]', '$x')])

plot.line(wavelength, flux, color='black')



plot.xaxis.axis_label = "Wavelength [nm]"

plot.yaxis.axis_label = "Flux [Scaled Units]"

plot.axis.axis_label_text_font_size = "12pt"

plot.axis.axis_label_text_font_style = 'bold'



layout = column(plot, sizing_mode='scale_width')



bokeh.io.show(layout)


spec = fits.open("/kaggle/input/a6-empirical-stellar-spectrum/A6_0.0_Dwarf.fits")

flux = spec[1].data.field('Flux')

wavelength = 10.0**spec[1].data.field('LogLam') # Angstroms

wavelength = wavelength / 10 # nm

err = spec[1].data.field('PropErr')



hotsource = ColumnDataSource(dict(x=wavelength, y=flux))



spec = fits.open("/kaggle/input/a6-empirical-stellar-spectrum/K4_0.0_Dwarf.fits")

flux = spec[1].data.field('Flux')

wavelength = 10.0**spec[1].data.field('LogLam') # Angstroms

wavelength = wavelength / 10 # nm

err = spec[1].data.field('PropErr')



coolsource = ColumnDataSource(dict(x=wavelength, y=flux * 2.5))



plot = figure(x_range=(365, 700), plot_width=640, plot_height=480, 

              sizing_mode='scale_width', y_range=(0, 6))

plot.line(x='x', y='y', source=hotsource, color='black', legend_label='A6')

plot.line(x='x', y='y', source=coolsource, color='black', legend_label='K4', visible=False)



plot.legend.location = "top_right"

plot.legend.click_policy="hide"



B = []

lamb = np.arange(350.0, 701.0, 2) # in nm

start = 3000.0

end = 15001.0

step = 200.0

temps = np.arange(start, end, step)

for T in temps:

    flux = 2 * h * c**2 / (lamb / 10**9)**5 * 1 / (np.expm1(h * c / k / T / (lamb / 10**9)))

    flux = flux / np.max(flux)

    B.append(flux)



    

bbsource = ColumnDataSource(data=dict(x=lamb, y=B[6]))

blackbody = plot.line('x', 'y', source=bbsource, line_width=3)



peakslider = Slider(start=start, end=end, value=start, step=step, title="Temperature [K]")

scaleslider = Slider(start=0.5, end=10, value=1, step=0.1, title="Scaling Factor")



callback = CustomJS(args=dict(peakslider=peakslider, bbsource=bbsource, B=B, step=step, 

                              start=start, scaleslider=scaleslider),

                        code="""

    const data = bbsource.data;

    const pos = peakslider.value;

    const scale = scaleslider.value;

    const index = Math.round(pos / step - start / step);

    

    for (var i = 0; i < data['y'].length; i++){

        data['y'][i] = B[index][i] * scale; 

    }

    

    bbsource.change.emit();

    """)



peakslider.js_on_change('value', callback)

scaleslider.js_on_change('value', callback)



plot.xaxis.axis_label = "Wavelength [nm]"

plot.yaxis.axis_label = "Flux [Scaled Units]"

plot.axis.axis_label_text_font_size = "12pt"

plot.axis.axis_label_text_font_style = 'bold'



layout = column(plot, peakslider, scaleslider, sizing_mode='scale_width')



bokeh.io.show(layout)