import ble2lsl
from ble2lsl.devices import muse2016
from wizardhat import acquire, plot, transform

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
device = muse2016
dummy_outlet = ble2lsl.Dummy(device)
receiver = acquire.Receiver()
lo_cut = 20
hi_cut = 50
notch_freq = None
filter = transform.Bandpass(receiver.buffers['EEG'],lo_cut,hi_cut)
channel_to_view = 'TP9'
samples_to_view = 2000
raw = filter.buffer_in.data[channel_to_view][-samples_to_view:]
time_raw = filter.buffer_in.get_timestamps()[-samples_to_view:]
filt = filter.buffer_out.data[channel_to_view][-samples_to_view:]
time_filt = filter.buffer_out.get_timestamps()[-samples_to_view:]
plt.subplots(figsize=(20,5))
plt.plot(time_raw,raw)
plt.plot(time_filt,filt)
plt.xlabel('time (s)',fontsize=20)
plt.ylabel('voltage (mV)',fontsize=20)
plt.legend(['Raw signal','Filtered Signal'],fontsize=20)
pre_filter = transform.PSD(receiver.buffers['EEG'])
post_filter = transform.PSD(filter.buffer_out)
# timestamp of most recent psd
channel_to_view = 'TP9'
timestamp_to_view = pre_filter.buffer_out.get_timestamps(1) # last n timestamps

# grab psd pre- and post-filter
pre_filter_data = pre_filter.buffer_out.data[['time',channel_to_view]]
psd_raw = pre_filter_data[pre_filter_data['time']==timestamp_to_view]

post_filter_data = post_filter.buffer_out.data[['time',channel_to_view]]
psd_filt = post_filter_data[post_filter_data['time']==timestamp_to_view]

len(psd_raw[channel_to_view].T)
x = np.arange(0,len(psd_raw[channel_to_view].T))
plt.subplots(figsize=(20,5))
plt.plot(x,psd_raw[channel_to_view].T)
plt.plot(x,psd_filt[channel_to_view].T)

plt.axvline(x=lo_cut,color='red',linestyle='--')
plt.axvline(x=hi_cut,color='red',linestyle='--')

plt.xlabel('Frequency (Hz)',fontsize=20)
plt.ylabel('Power (Watts/Hz)',fontsize=20)
plt.legend(['Raw signal','Filtered Signal'],fontsize=20)
plt.title(f'Bandpass from {lo_cut} Hz to {hi_cut} Hz',fontsize=20)