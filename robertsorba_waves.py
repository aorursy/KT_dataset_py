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
YouTubeVideo('v8PxXCM4BXE', width=560, height=315)
samplerate, triangledata = wavfile.read("/kaggle/input/waves-recordings/triangle.wav")

freq = 3000 # Hz

size = samplerate * 30

x = np.arange(size)

puretone = np.sin(2 * np.pi * freq * x / samplerate)

print("Pure Tone at 3000 Hz")

Audio(puretone, rate=samplerate)
print("Interference Example")

Audio(triangledata[:,0], rate=samplerate)
plotamps = []

freqs = np.arange(513) * samplerate / 1024

#plotfreqs = freqs[np.logical_and(freqs>500, freqs<5000)]



for i in range(150):

    startdex = 4410 * i

    dataSlice = triangledata[:,0][startdex:startdex + 1024]



    F = np.fft.rfft(dataSlice)

    amps = np.abs(F) / F.size    

    plotamps.append(amps)#[np.logical_and(freqs>500, freqs<5000)])

    



p = figure(plot_width=640, plot_height=480, max_width=640, sizing_mode='scale_width',

          x_range=[0, 6000], y_range=[0,1500])

p.xaxis.axis_label = "Frequency [Hz]"

p.yaxis.axis_label = "Amplitude [Scaled Units]"

p.axis.axis_label_text_font_size = "12pt"

p.axis.axis_label_text_font_style = 'bold'



src = ColumnDataSource(dict(x=freqs, y=plotamps[0]))

p.line(x='x', y='y', source=src, line_width=3)



#duration = triangledata[:,0].size / samplerate



slider = Slider(start=0, end=15, value=0, step=0.1, title="Time [s]",

               max_width=620)



callback = CustomJS(args=dict(slider=slider, src=src, amps=plotamps),

                   code="""

    const time = slider.value;

    const index = Math.round(time * 10)

    src.data['y'] = amps[index];

    src.change.emit();

                   

                   """)



slider.js_on_change('value', callback)



layout = column(p, slider, max_width=640, sizing_mode='scale_width')



bokeh.io.show(layout)
stp = 0.005

x = np.arange(0, 1.001, stp) - 0.5

f = 3000.0 # Hz

v = 343.0 # spped of sound m/s

wl = v/f

y1 = np.cos(2 * np.pi * f / v * (x + 0.5))

y2 = np.cos(-2 * np.pi * f / v * (x - 0.5))



speakerplot = figure(plot_width=640, plot_height=240, max_width=640, sizing_mode='scale_width', 

                     x_range=[-0.6, 0.6], toolbar_location=None)

speakerplot.xaxis.axis_label = "Distance [m]"

speakerplot.yaxis.axis_label = "Amplitude"

speakerplot.axis.axis_label_text_font_size = "12pt"

speakerplot.axis.axis_label_text_font_style = 'bold'

speakerplot.yaxis.major_label_text_color = None

speakerplot.yaxis.major_tick_line_color = None

speakerplot.yaxis.minor_tick_line_color = None



leftsource = ColumnDataSource(data=dict(x=x, y=y1))

speakerplot.line('x', 'y', source=leftsource, line_width=3, alpha=0.8)



rightsource = ColumnDataSource(data=dict(x=x, y=y2))

speakerplot.line('x', 'y', source=rightsource, line_width=3, color='orange', alpha=0.8)



powerplot = figure(plot_width=640, plot_height=240, max_width=640, sizing_mode='scale_width', 

                     x_range=speakerplot.x_range, y_range=[0,4], toolbar_location=None)

powerplot.yaxis.axis_label = "Power (Amp. squared)"

powerplot.axis.axis_label_text_font_size = "12pt"

powerplot.axis.axis_label_text_font_style = 'bold'

powerplot.yaxis.major_label_text_color = None

powerplot.yaxis.major_tick_line_color = None

powerplot.yaxis.minor_tick_line_color = None

powerplot.xaxis.ticker = [-4*wl, -3*wl, -2*wl, -wl, 0, wl, 2*wl, 3*wl, 4*wl]

powerplot.xaxis.major_label_overrides = {0:'0', wl:"λ", -wl:"-λ", 2*wl:"2λ", -2*wl:"-2λ",

                                        3*wl:"3λ", -3*wl:"-3λ", 4*wl:"4λ", -4*wl:"-4λ"}



sumsource = ColumnDataSource(data=dict(x=x, y=(y1 + y2)**2))

powerplot.line('x', 'y', source=sumsource,legend_label='Combined Signal',

                 color='green', line_width=3)

powerplot.legend.location = "top_right"







diaphramsource = ColumnDataSource(data=dict(x=[-0.51,-0.505, 0.51, 0.505], y=[-0.4, 0.5, -0.4, 0.5], 

                                            h=[0.9, 0.5, 0.9, 0.5], w=[0.04, 0.025, 0.04, 0.025]))

glyph = Ellipse(x="x", y="y", width='w', height="h", fill_color="darkgray")

speakerplot.add_glyph(diaphramsource, glyph)



speakersource = ColumnDataSource(data=dict(x=[-0.55, 0.55], y=[0, 0]))

glyph = Rect(x="x", y="y", width=0.1, height=2, fill_color="#cab2d6")

speakerplot.add_glyph(speakersource, glyph)



slider = Slider(start=0.2, end=1.001, value=1.0, step=stp*2, title="Speaker Distance [m]",

               max_width=620)



callback = CustomJS(args=dict(rsrc=rightsource, lsrc=leftsource, slider=slider, x=x, 

                              sumsrc=sumsource, step=stp, y1=y1, y2=y2, spkrsrc=speakersource,

                             dsrc=diaphramsource),

                    code="""

    const pos = slider.value;

    const lastdex = Math.round(pos / step) + 1;

    const edge = (x.length - lastdex) / 2;

    const offset = pos / 2.0 + 0.05;

    

    lsrc.data['y'] = y1.slice(1, lastdex + 1);

    rsrc.data['y'] = y2.slice(x.length - lastdex + 1, x.length+1);

    lsrc.data['x'] = x.slice(edge + 1, x.length - edge + 1);

    rsrc.data['x'] = x.slice(edge + 1, x.length - edge + 1);

    

    sumsrc.data['x'] = [];

    sumsrc.data['y'] = [];

    for (var i = 0; i < lsrc.data['x'].length; i++){

        sumsrc.data['x'][i] = lsrc.data['x'][i];

        sumsrc.data['y'][i] = Math.pow((lsrc.data['y'][i] + rsrc.data['y'][i]), 2);

    }

    

    spkrsrc.data['x'] = [-offset, offset];

    dsrc.data['x'] = [-offset + 0.04, -offset + 0.045, offset - 0.04, offset - 0.045];

    

    lsrc.change.emit();

    rsrc.change.emit();

    sumsrc.change.emit();

    spkrsrc.change.emit();

    dsrc.change.emit();

""")



slider.js_on_change('value', callback)



layout = column(gridplot([powerplot, speakerplot], ncols=1, toolbar_location=None), 

                slider, sizing_mode="scale_width")



bokeh.io.show(layout)
YouTubeVideo('kqrwr_Jbnso', width=560, height=315)

# https://youtu.be/kqrwr_Jbnso
samplerate, data = wavfile.read("/kaggle/input/waves-recordings/speed_cropped.wav")

plotamps = []

freqs = np.arange(513) * samplerate / 1024

#print(freqs.size)

plotfreqs = freqs[np.logical_and(freqs>500, freqs<5000)]



def findIndexFromPos(x, samplerate=48000):

    time = -x + 50 - 22.2

    frame = samplerate * time

    return frame.astype(int)



probex = np.arange(24.75, 6.75, -0.1 / 2.54)

probedex = findIndexFromPos(probex)

maxpeak = []



for i in probedex:

    dataSlice = data[i:i + 1024]



    F = np.fft.rfft(dataSlice)

    amps = np.abs(F) / F.size 

    #print(amps.size)

    plotamps.append(amps[np.logical_and(freqs>500, freqs<5000)])

    maxpeak.append(np.max(amps[np.logical_and(freqs>500, freqs<5000)]))

    



p = figure(plot_width=640, plot_height=480, max_width=640, sizing_mode='scale_width',

          x_range=[500, 5000], y_range=[0,6000])

p.xaxis.axis_label = "Frequency [Hz]"

p.yaxis.axis_label = "Amplitude [Scaled Units]"

p.axis.axis_label_text_font_size = "12pt"

p.axis.axis_label_text_font_style = 'bold'



src = ColumnDataSource(dict(x=plotfreqs, y=plotamps[206]))

p.line(x='x', y='y', source=src, line_width=3)



slider = Slider(start=20, end=60, value=40, step=0.1, title="Mic Position [cm]",

               max_width=620)



callback = CustomJS(args=dict(slider=slider, src=src, amps=plotamps),

                   code="""

    const pos = slider.value;

    const index = Math.round(10 * pos - 194)

    src.data['y'] = amps[index];

    src.change.emit();

                   

                   """)



slider.js_on_change('value', callback)



#junk = figure(plot_width=640, plot_height=480, max_width=640, sizing_mode='scale_width')

#junk.line((probex - 15.75) * 2.54, maxpeak)



layout = column(p, slider, max_width=640, sizing_mode='scale_width')



bokeh.io.show(layout)
YouTubeVideo('m7Ijr1kPLuI', width=560, height=315)

# https://youtu.be/m7Ijr1kPLuI
samplerate, data = wavfile.read("/kaggle/input/waves-recordings/diffraction.wav")

freqs = np.arange(513) * samplerate / 1024

times = [2, 8, 16, 24] # 500 blocked, 500 unblocked, 5000 blocked, 5000 unblocked

plotamps = []

for t in times:

    index = samplerate * t

    dataSlice = data[index:index + 1024]



    F = np.fft.rfft(dataSlice)

    amps = np.abs(F) / F.size 

    #print(amps.size)

    plotamps.append(amps)

    

p = figure(plot_width=640, plot_height=480, max_width=640, sizing_mode='scale_width',

          x_range=[0, 6000], y_range=[0,3000], tooltips=[('Amplitude', '$y{int}')])

p.xaxis.axis_label = "Frequency [Hz]"

p.yaxis.axis_label = "Amplitude [Scaled Units]"

p.axis.axis_label_text_font_size = "12pt"

p.axis.axis_label_text_font_style = 'bold'



src = ColumnDataSource(dict(x=freqs, y=plotamps[1]))

p.line(x='x', y='y', source=src, line_width=3, legend_label='500 Hz Unblocked', color='orange')





src2 = ColumnDataSource(dict(x=freqs, y=plotamps[0]))

p.line(x='x', y='y', source=src2, line_width=3, legend_label='500 Hz Blocked', line_dash='dashed', color='orange')



src3 = ColumnDataSource(dict(x=freqs, y=plotamps[3]))

p.line(x='x', y='y', source=src3, line_width=3, legend_label='5000 Hz Unblocked')





src4 = ColumnDataSource(dict(x=freqs, y=plotamps[2]))

p.line(x='x', y='y', source=src4, line_width=3, legend_label='5000 Hz Blocked', line_dash='dashed')



p.legend.location = "top_right"

p.legend.click_policy="hide"



bokeh.io.show(p)



    
YouTubeVideo('GDm_KZcb29c', width=560, height=315)

# https://youtu.be/GDm_KZcb29c
f = 300 # Hz

sep = 10

samplerate = 44100

size = samplerate * 30

x = np.arange(size)

beat = np.sin(2 * np.pi * f * x / samplerate) + np.sin(2 * np.pi * (f+sep) * x / samplerate)

print("Separation of %i Hz" % sep)

Audio(beat, rate=samplerate)
f = 300 # Hz

sep = 8

samplerate = 44100

size = samplerate * 30

x = np.arange(size)

beat = np.sin(2 * np.pi * f * x / samplerate) + np.sin(2 * np.pi * (f+sep) * x / samplerate)

print("Separation of %i Hz" % sep)

Audio(beat, rate=samplerate)
f = 300 # Hz

sep = 6

samplerate = 44100

size = samplerate * 30

x = np.arange(size)

beat = np.sin(2 * np.pi * f * x / samplerate) + np.sin(2 * np.pi * (f+sep) * x / samplerate)

print("Separation of %i Hz" % sep)

Audio(beat, rate=samplerate)
f = 300 # Hz

sep = 4

samplerate = 44100

size = samplerate * 30

x = np.arange(size)

beat = np.sin(2 * np.pi * f * x / samplerate) + np.sin(2 * np.pi * (f+sep) * x / samplerate)

print("Separation of %i Hz" % sep)

Audio(beat, rate=samplerate)
f = 300 # Hz

sep = 2

samplerate = 44100

size = samplerate * 30

x = np.arange(size)

beat = np.sin(2 * np.pi * f * x / samplerate) + np.sin(2 * np.pi * (f+sep) * x / samplerate)

print("Separation of %i Hz" % sep)

Audio(beat, rate=samplerate)