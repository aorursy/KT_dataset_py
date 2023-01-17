import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



%matplotlib inline

import matplotlib.pyplot as plt
sampling_frequency = 4e6

time_step = 1./sampling_frequency



chunksize = 150_000



train = pd.read_csv('../input/train.csv', iterator=True, 

                    chunksize=chunksize, 

                    dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})
from scipy import signal



def get_spectrum(input_signal):

    """

    Get a pandas Series with the fourier power spectum for a given signal segment.

    """

    input_signal = np.asarray(input_signal.values, dtype='float64')

    

    # Remove the mean  

    input_signal -= input_signal.mean()  

    

    # Estimate power spectral density using a periodogram.

    frequencies , power_spectrum = signal.periodogram(input_signal, sampling_frequency, scaling='spectrum')    

    

    # Run a running windows average of 10-points to smooth the signal.

    power_spectrum = pd.Series(power_spectrum, index=frequencies).rolling(window=10).mean()        

    

    return pd.Series(power_spectrum)



def get_segment_spectrum(segment_df):

    """

    Get the fourier power spectrum of a given segment.

    

    Returns the quake_time, frequencies, and power_spectrum

    """

    

    quake_time =segment_df['time_to_failure'].values[-1]

    

    _power_spectrum = get_spectrum(segment_df['acoustic_data']).dropna() 

    

    # Keep only frequencies < 450khz (larger frequencies have a negligible contribution).

    _power_spectrum = _power_spectrum[_power_spectrum.index<450_000]

    

    # Keep one every 10 samples

    power_spectrum=_power_spectrum.values[::10]

    frequencies=_power_spectrum.index.values[::10]    

    

    return quake_time, frequencies, power_spectrum
quake_times = list()       

power_spectrums = list()



for df in train:    

    quake_time, _frequencies, power_spectrum = get_segment_spectrum(df)    

    if df.shape[0]<chunksize:

        continue

    

    frequencies=_frequencies

    quake_times.append(quake_time)    

    power_spectrums.append(power_spectrum)



power_spectrums = np.stack(power_spectrums, axis=0)

quake_times = pd.Series(quake_times)    
print("power_spectrums.shape:",power_spectrums.shape)

print("power_spectrums size:",power_spectrums.nbytes/(1024**2.),"[mb]")
from bokeh.layouts import column, row

from bokeh.plotting import Figure, show

from bokeh.io import output_notebook

from bokeh.models import PrintfTickFormatter

from bokeh.models import LinearAxis, Range1d



output_notebook() # Display Bokeh plots inline in a classic Jupyter notebooks



# Compute the average spectrum

average_power_spectrum = pd.Series(power_spectrums.mean(axis=0))

average_power_spectrum = average_power_spectrum.rolling(window=10).mean()



# "@foo{(0.00 a)}" # formats 1230974 as: 1.23 m

TOOLTIPS = [ ("x", "@x{(0.00 a)}Hz"), ("y", "$y")   ]



pl = Figure(plot_width=800, plot_height=400,title="Average power spectrum",tooltips=TOOLTIPS)

pl.line(frequencies, average_power_spectrum, line_color="navy")

pl.xaxis[0].formatter = PrintfTickFormatter(format="%d")

pl.xaxis.axis_label = "Frequency [hz]"

pl.yaxis.axis_label = "Power [V**2]"

show(pl)
normalized_spectrums = power_spectrums/(power_spectrums.sum(axis=1)[:,np.newaxis])

average_power_spectrum = pd.Series(normalized_spectrums.mean(axis=0))

average_power_spectrum = average_power_spectrum.rolling(window=10).mean()



TOOLTIPS = [ ("x", "@x{(0.00 a)}Hz"), ("y", "$y")   ]

pl = Figure(plot_width=800, plot_height=400, title="Average of normalized power-spectrums",

            tooltips=TOOLTIPS)

pl.line(frequencies, average_power_spectrum, line_color="navy")

pl.xaxis[0].formatter = PrintfTickFormatter(format="%d")

pl.xaxis.axis_label = "Frequency [hz]"

pl.yaxis.axis_label = "Normalized Power []"

show(pl)
# Create normalized spectrum composites for mean, std, min and max.



dt = 0.2 # Interval of time to aggregate a composite

max_quake_time = quake_times.max()

quake_times_intervals = np.arange(dt/2, max_quake_time+dt/2+0.01, dt)    



composite_spectrums_mean = np.zeros((quake_times_intervals.size, frequencies.size))

composite_spectrums_std = np.zeros((quake_times_intervals.size, frequencies.size))

composite_spectrums_min = np.zeros((quake_times_intervals.size, frequencies.size))

composite_spectrums_max = np.zeros((quake_times_intervals.size, frequencies.size))



normalized_spectrums = power_spectrums/(power_spectrums.sum(axis=1)[:,np.newaxis])



for n , quake_time in enumerate(quake_times_intervals):

    t0 = quake_time-dt/2

    t1 = quake_time+dt/2

    segments = quake_times[(quake_times>=t0) & (quake_times<t1)].index.values

    if len(segments) > 0:

        composite_spectrums_mean[n] = normalized_spectrums[segments,:].mean(axis=0)

        composite_spectrums_std[n] = normalized_spectrums[segments,:].std(axis=0)

        composite_spectrums_min[n] = normalized_spectrums[segments,:].min(axis=0)

        composite_spectrums_max[n] = normalized_spectrums[segments,:].max(axis=0)



print("composite_spectrums_mean.shape:",composite_spectrums_mean.shape)

print("composite_spectrums_mean size:",composite_spectrums_mean.nbytes/(1024**2.),"[mb]")
from bokeh.models import CustomJS, Slider

from bokeh.plotting import figure, output_file, show, ColumnDataSource



_quake_time = quake_times_intervals[0]



TOOLTIPS = [ ("x", "@frequencies{(0.00 a)}Hz"), ("y", "$y")   ]



pl = Figure(plot_width=800, plot_height=400, title= f"{_quake_time-dt/2} <= Quake time < {_quake_time+dt/2}",

           y_range=(0, 4e-3), tooltips=TOOLTIPS)



data_dict = dict()

for i in range(quake_times_intervals.size):    

    data_dict[str(i)] = composite_spectrums_mean[i]

all_data = ColumnDataSource(data=data_dict)

source = ColumnDataSource(data=dict(frequencies=frequencies, composite_mean=composite_spectrums_mean[0]))



pl.line('frequencies', 'composite_mean', line_color="navy", source=source)

pl.xaxis[0].formatter = PrintfTickFormatter(format="%d")



                            

callback = CustomJS(args=dict(source=source, 

                              all_data=all_data,plot=pl,

                              quake_times_intervals=quake_times_intervals,

                              dt=dt), 

                    code="""

    var data = source.data;

    var interval = slider.value;

    var composite_means = all_data.data[String(interval)];    

    var y = data['composite_mean']

    for (var i = 0; i < y.length; i++) {

        y[i] = composite_means[i];

    }

    var _quake_time = quake_times_intervals[interval]

    console.log(String(_quake_time-dt/2));

    console.log(String(_quake_time+dt/2));

    plot.title.text = String((_quake_time-dt/2).toFixed(1)) + " <= Quake time < " + String((_quake_time+dt/2).toFixed(1));

    source.change.emit();

""")



time_slider = Slider(start=0, end=composite_spectrums_mean.shape[0], value=0, step=1,

                     title="Quaketime interval number", callback=callback)

callback.args["slider"] = time_slider

                            

layout = column(time_slider,  pl)

show(layout)