import numpy as np

from IPython.display import Audio

from scipy.io import wavfile



from bokeh.resources import INLINE

import bokeh.io

bokeh.io.output_notebook(INLINE)

from bokeh.layouts import column, row, gridplot

from bokeh.models import CustomJS, Slider, Rect, Button, Label, VArea, Arrow, OpenHead, Ellipse

from bokeh.plotting import ColumnDataSource, figure, output_file, show
from IPython.core.display import display,HTML

from IPython.display import YouTubeVideo

display(HTML('<style>.prompt{width: 0px; min-width: 0px; visibility: collapse}</style>'))

YouTubeVideo('m5gCwT6m5M4', width=560, height=315)

#https://youtu.be/m5gCwT6m5M4
samplerate, openopen = wavfile.read("/kaggle/input/glass-tubes/openopen3.wav")

Audio(openopen, rate=samplerate)
size = 1024

window = np.blackman(size)

step = samplerate / size

freqs = np.arange(size / 2 + 1) * step

#plotfreqs = freqs[np.logical_and(freqs>500, freqs<5000)]



allamps = []

startdex =  2 * samplerate

for i in range(int(openopen.size)):

    startdex = (i + 1) * size

    startdex = i

    if startdex + size > openopen.size:

        break

    dataSlice = openopen[startdex:startdex + size] * window

    



    F = np.fft.rfft(dataSlice)

    amps = np.abs(F) / F.size * 2

    allamps.append(amps)

    

allamps = np.array(allamps)

avgamps = np.average(allamps, axis=0)

    



p = figure(plot_width=640, plot_height=480, max_width=640, sizing_mode='scale_width',

          x_range=[0, 10000])

p.xaxis.axis_label = "Frequency [Hz]"

p.yaxis.axis_label = "Amplitude [Scaled Units]"

p.axis.axis_label_text_font_size = "12pt"

p.axis.axis_label_text_font_style = 'bold'



hsource = ColumnDataSource(data=dict(x=[0, 10000], y=[avgamps[20]] * 2))

hline = p.line('x', 'y', source=hsource,

                 color='orange', line_dash='dashed', line_width=3)



vsource = ColumnDataSource(data=dict(x=[freqs[20]] * 2, y=[0, 300]))

vline = p.line('x', 'y', source=vsource,

                 color='orange', line_dash='dashed', line_width=3)



src = ColumnDataSource(dict(x=freqs, y=avgamps))

p.line(x='x', y='y', source=src, line_width=3)



hlsource = ColumnDataSource(data=dict(x=[freqs[20]], y=[avgamps[20]]))

highlight = p.square('x', 'y', source=hlsource, size=20, fill_color='orange', 

                        fill_alpha=0.5)



slider = Slider(start=0, end=10000, value=freqs[20], step=step, title="Freq [Hz]",

               max_width=620)



callback = CustomJS(args=dict(hlsource=hlsource, slider=slider, src=src, bottomx=0,

                              step=step, hsource=hsource, vsource=vsource),

                    code="""

    const hldata = hlsource.data;

    const pos = slider.value;

    const data = src.data;

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

    y[0] = data['y'][index];

    

    //p.text = 'x,y = (' + pos.toFixed(2) + ', ' + y[0].toFixed(3) + ')';

    

    for (var i = 0; i < hdata['y'].length; i++) {

        hdata['y'][i] = data['y'][index];

    }

    

    hlsource.change.emit();

    hsource.change.emit();

    vsource.change.emit();

""")



slider.js_on_change('value', callback)



##################



p2 = figure(plot_width=640, plot_height=480, sizing_mode='scale_width',

              x_range=[0, 9], y_range=[0, 10000])



p2.xaxis.axis_label = "Harmonic Number"

p2.yaxis.axis_label = "Frequency [Hz]"

p2.axis.axis_label_text_font_size = "12pt"

p2.axis.axis_label_text_font_style = 'bold'

p2.xaxis.ticker = np.arange(10)



measure = Label(x=4, y=8000, text="f = ?n", render_mode='css',

               text_color='gray')

p2.add_layout(measure)



fitsource = ColumnDataSource(data=dict(x=[],y=[]))

p2.line('x', 'y', source=fitsource, line_width=4, color='gray', line_dash='dashed')



circlesource = ColumnDataSource(data=dict(x=[],y=[], freqs=[]))

p2.circle('x', 'y', source=circlesource, size=15, fill_color='red')



button = Button(label="Record Peak!", button_type="success", sizing_mode="scale_width")



buttonCallback = CustomJS(args=dict(vsource=vsource, m=measure,

                                    csource=circlesource, fitsource=fitsource),

                         code="""

    const f = vsource.data['x'][0];

    const x = csource.data['x'];

    const y = csource.data['y'];

    const freqs = csource.data['freqs'];

    

    x.push(x.length + 1);

    freqs.push(f);

    for (var i =0; i < freqs.length; i++){

        y[i] = freqs[i];

    }

    y.sort(function(a, b){return a-b});

    

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

        

        fitsource.data['x'] = [0, 9];

        fitsource.data['y'] = [intercept, slope * 9 + intercept];

        

        //var formattedSlope = slope / 1000.0;

        m.text = 'f = ' + slope.toFixed(1) + 'n';

    }

    else{

        fitsource.data['x'] = [];

        fitsource.data['y'] = [];

        m.text = 'f = ?n';

    }

    

    fitsource.change.emit();  

    

                         """)

button.js_on_click(buttonCallback)



backbutton = Button(label="Remove Last Datapoint", button_type="danger", sizing_mode="scale_width")



backCallback = CustomJS(args=dict(csource=circlesource, fitsource=fitsource, m=measure),

                         code=""" 

    const x = csource.data['x'];

    const y = csource.data['y'];

    const freqs = csource.data['freqs'];

    

    x.pop();

    y.pop();

    freqs.pop();

    for (var i = 0; i < freqs.length; i++){

        y[i] = freqs[i];

    }

    y.sort(function(a, b){return a-b});

    

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

        

        fitsource.data['x'] = [0, 9];

        fitsource.data['y'] = [intercept, slope * 9 + intercept];

        m.text = 'f = ' + slope.toFixed(1) + 'n';

    } else {

        fitsource.data['x'] = [];

        fitsource.data['y'] = [];

        m.text = 'f = ?n';

    }

    

    fitsource.change.emit(); 

                         """)

backbutton.js_on_click(backCallback)







layout = column(p, slider, row(button, backbutton), p2, max_width=640, sizing_mode='scale_width')



bokeh.io.show(layout)
samplerate, openopen = wavfile.read("/kaggle/input/glass-tubes/openclosed2.wav")

Audio(openopen, rate=samplerate)
size = 1024

window = np.blackman(size)

step = samplerate / size

freqs = np.arange(size / 2 + 1) * step

#plotfreqs = freqs[np.logical_and(freqs>500, freqs<5000)]



allamps = []

startdex =  2 * samplerate

for i in range(int(openopen.size)):

    startdex = (i + 1) * size

    startdex = i

    if startdex + size > openopen.size:

        break

    dataSlice = openopen[startdex:startdex + size] * window

    



    F = np.fft.rfft(dataSlice)

    amps = np.abs(F) / F.size * 3.5

    allamps.append(amps)

    

allamps = np.array(allamps)

avgamps = np.average(allamps, axis=0)

    



p = figure(plot_width=640, plot_height=480, max_width=640, sizing_mode='scale_width',

          x_range=[0, 10000])

p.xaxis.axis_label = "Frequency [Hz]"

p.yaxis.axis_label = "Amplitude [Scaled Units]"

p.axis.axis_label_text_font_size = "12pt"

p.axis.axis_label_text_font_style = 'bold'



hsource = ColumnDataSource(data=dict(x=[0, 10000], y=[avgamps[20]] * 2))

hline = p.line('x', 'y', source=hsource,

                 color='orange', line_dash='dashed', line_width=3)



vsource = ColumnDataSource(data=dict(x=[freqs[20]] * 2, y=[0, 300]))

vline = p.line('x', 'y', source=vsource,

                 color='orange', line_dash='dashed', line_width=3)



src = ColumnDataSource(dict(x=freqs, y=avgamps))

p.line(x='x', y='y', source=src, line_width=3)



hlsource = ColumnDataSource(data=dict(x=[freqs[20]], y=[avgamps[20]]))

highlight = p.square('x', 'y', source=hlsource, size=20, fill_color='orange', 

                        fill_alpha=0.5)



slider = Slider(start=0, end=10000, value=freqs[20], step=step, title="Freq [Hz]",

               max_width=620)



callback = CustomJS(args=dict(hlsource=hlsource, slider=slider, src=src, bottomx=0,

                              step=step, hsource=hsource, vsource=vsource),

                    code="""

    const hldata = hlsource.data;

    const pos = slider.value;

    const data = src.data;

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

    y[0] = data['y'][index];

    

    //p.text = 'x,y = (' + pos.toFixed(2) + ', ' + y[0].toFixed(3) + ')';

    

    for (var i = 0; i < hdata['y'].length; i++) {

        hdata['y'][i] = data['y'][index];

    }

    

    hlsource.change.emit();

    hsource.change.emit();

    vsource.change.emit();

""")



slider.js_on_change('value', callback)



##################



p2 = figure(plot_width=640, plot_height=480, sizing_mode='scale_width',

              x_range=[0, 9], y_range=[0, 10000])



p2.xaxis.axis_label = "Harmonic Number"

p2.yaxis.axis_label = "Frequency [Hz]"

p2.axis.axis_label_text_font_size = "12pt"

p2.axis.axis_label_text_font_style = 'bold'

p2.xaxis.ticker = np.arange(10)



measure = Label(x=4, y=8000, text="f = ?n", render_mode='css',

               text_color='gray')

p2.add_layout(measure)



fitsource = ColumnDataSource(data=dict(x=[],y=[]))

p2.line('x', 'y', source=fitsource, line_width=4, color='gray', line_dash='dashed')



circlesource = ColumnDataSource(data=dict(x=[],y=[], freqs=[]))

p2.circle('x', 'y', source=circlesource, size=15, fill_color='red')



button = Button(label="Record Peak!", button_type="success", sizing_mode="scale_width")



buttonCallback = CustomJS(args=dict(vsource=vsource, m=measure,

                                    csource=circlesource, fitsource=fitsource),

                         code="""

    const f = vsource.data['x'][0];

    const x = csource.data['x'];

    const y = csource.data['y'];

    const freqs = csource.data['freqs'];

    

    x.push(2 * x.length + 1);

    freqs.push(f);

    for (var i =0; i < freqs.length; i++){

        y[i] = freqs[i];

    }

    y.sort(function(a, b){return a-b});

    

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

        

        fitsource.data['x'] = [0, 9];

        fitsource.data['y'] = [intercept, slope * 9 + intercept];

        

        //var formattedSlope = slope / 1000.0;

        m.text = 'f = ' + slope.toFixed(1) + 'n';

    }

    else{

        fitsource.data['x'] = [];

        fitsource.data['y'] = [];

        m.text = 'f = ?n';

    }

    

    fitsource.change.emit();  

    

                         """)

button.js_on_click(buttonCallback)



backbutton = Button(label="Remove Last Datapoint", button_type="danger", sizing_mode="scale_width")



backCallback = CustomJS(args=dict(csource=circlesource, fitsource=fitsource, m=measure),

                         code=""" 

    const x = csource.data['x'];

    const y = csource.data['y'];

    const freqs = csource.data['freqs'];

    

    x.pop();

    y.pop();

    freqs.pop();

    for (var i = 0; i < freqs.length; i++){

        y[i] = freqs[i];

    }

    y.sort(function(a, b){return a-b});

    

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

        

        fitsource.data['x'] = [0, 9];

        fitsource.data['y'] = [intercept, slope * 9 + intercept];

        m.text = 'f = ' + slope.toFixed(1) + 'n';

    } else {

        fitsource.data['x'] = [];

        fitsource.data['y'] = [];

        m.text = 'f = ?n';

    }

    

    fitsource.change.emit(); 

                         """)

backbutton.js_on_click(backCallback)







layout = column(p, slider, row(button, backbutton), p2, max_width=640, sizing_mode='scale_width')



bokeh.io.show(layout)
samplerate, green = wavfile.read("/kaggle/input/glass-tubes/flute.wav")

size = 16384

window = np.blackman(size)

step = samplerate / size

freqs = np.arange(size / 2 + 1) * step

#plotfreqs = freqs[np.logical_and(freqs>500, freqs<5000)]



#allamps = []

startdex =  2 * samplerate

#for i in range(int(openopen.size)):

#    startdex = (i + 1) * size

#    startdex = i

#    if startdex + size > openopen.size:

#        break

dataSlice = green[startdex:startdex + size] * window

    



F = np.fft.rfft(dataSlice)

amps = np.abs(F) / F.size

#    allamps.append(amps)

    

#allamps = np.array(allamps)

#avgamps = np.average(allamps, axis=0)

    



p = figure(plot_width=640, plot_height=480, max_width=640, sizing_mode='scale_width',

          x_range=[0, 4000])

p.xaxis.axis_label = "Frequency [Hz]"

p.yaxis.axis_label = "Amplitude [Scaled Units]"

p.axis.axis_label_text_font_size = "12pt"

p.axis.axis_label_text_font_style = 'bold'





src = ColumnDataSource(dict(x=freqs, y=amps))

p.line(x='x', y='y', source=src, line_width=3)



bokeh.io.show(p)