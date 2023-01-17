import ble2lsl
from ble2lsl.devices import muse2016
from wizardhat import acquire, transform

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
#if we had a device with us, we would use:
#streamer = ble2lsl.Streamer(muse2016)
#but if you're debugging or learning, use the dummy streamer with the command below:
streamer = ble2lsl.Dummy(muse2016)
#After writing streamer. you can use the tab key to see a list of properties and methods that streamer has,
#for example, streamer.subscriptions shows all the subscribed data streams that the streamer object has picked up
#from the device.
streamer.subscriptions

receiver = acquire.Receiver()
receiver.buffers
receiver.ch_names
receiver.buffers['EEG'].data
receiver.buffers['EEG'].data.shape
#the default window for seeing data is 10 seconds. You can change that when you call acquire.Receiver(window=15) etc
receiver.buffers['EEG'].unstructured
#this version of the data has no labels and is just a pure numpy matrix
receiver.buffers['EEG'].get_timestamps()
our_first_recording = receiver.record(5)
#wait 5 seconds after running this command 
our_first_recording.buffers['EEG'].data
#notice it only goes up to 5 seconds
channel_to_view = 'TP9'
samples_to_view = 2000

raw = receiver.buffers['EEG'].data[channel_to_view][-samples_to_view:]
time_raw = receiver.buffers['EEG'].data['time'][-samples_to_view:]

plt.subplots(figsize=(20,5))
plt.plot(time_raw,raw)


lo_cut = 20
hi_cut = 50

filter = transform.Bandpass(receiver.buffers['EEG'],lo_cut,hi_cut)


raw = receiver.buffers['EEG'].data[channel_to_view][-samples_to_view:]
time_raw = receiver.buffers['EEG'].data['time'][-samples_to_view:]
filt = filter.buffer_out.data[channel_to_view][-samples_to_view:]
time_filt = filter.buffer_out.data['time'][-samples_to_view:]
plt.subplots(figsize=(20,5))
plt.plot(time_raw,raw)
plt.plot(time_filt,filt)


plt.xlabel('time (s)',fontsize=20)
plt.ylabel('voltage (mV)',fontsize=20)
plt.legend(['Raw signal','Filtered Signal'],fontsize=20)
pre_filter = transform.PSD(receiver.buffers['EEG'])
post_filter = transform.PSD(filter.buffer_out)
timestamp_to_view = pre_filter.buffer_out.get_timestamps(1)
pre_filter_data = pre_filter.buffer_out.data[['time',channel_to_view]]
post_filter_data = post_filter.buffer_out.data[['time',channel_to_view]]

psd_raw = pre_filter_data[pre_filter_data['time']==timestamp_to_view]
psd_filt = post_filter_data[post_filter_data['time']==timestamp_to_view]
psd_time = np.arange(0,len(psd_raw[channel_to_view].T))
plt.subplots(figsize=(20,5))
plt.plot(psd_time,psd_raw[channel_to_view].T)
plt.plot(psd_time,psd_filt[channel_to_view].T)


plt.xlabel('Freq (Hz)',fontsize=20)
plt.ylabel('Power',fontsize=20)
plt.axvline(x=lo_cut,color='red',linestyle='--')
plt.axvline(x=hi_cut,color='red',linestyle='--')
plt.legend(['Raw signal','Filtered Signal'],fontsize=20)
plt.title(f'Bandpass from {lo_cut} Hz to {hi_cut} Hz',fontsize=20)