import pylab as plt

import numpy as np

import PIL

import os

from scipy import ndimage, misc

import pandas as pd

import matplotlib

import matplotlib.pylab as plt

from blimpy import Filterbank

from blimpy import Waterfall

from blimpy.utils import db, lin, rebin, closest
file = '../input/hip4436/HIP4436/spliced_blc0001020304050607_guppi_57803_80733_HIP4436_0032.gpuspec.0000.h5'

file2 = '../input/hip4436/HIP4436/spliced_blc0001020304050607_guppi_57803_81079_HIP3333_0033.gpuspec.0000.h5'

file3 = '../input/hip4436/HIP4436/spliced_blc0001020304050607_guppi_57803_81424_HIP4436_0034.gpuspec.0000.h5'

file4 = '../input/hip4436/HIP4436/spliced_blc0001020304050607_guppi_57803_81768_HIP3597_0035.gpuspec.0000.h5'

file5 = '../input/hip4436/HIP4436/spliced_blc0001020304050607_guppi_57803_82111_HIP4436_0036.gpuspec.0000.h5'

file6 = '../input/hip4436/HIP4436/spliced_blc0001020304050607_guppi_57803_82459_HIP3677_0037.gpuspec.0000.h5'



filenames_list = [file,file2,file3,file4,file5,file6]
filterbank_head = Waterfall(file, load_data=False)

filterbank_head.info()
filterbank = Waterfall(file)
filterbank.container.f_stop
correction = filterbank.container.f_stop - 1380.87763 #know for HIP4436

filterbank.container.f_start =  filterbank.container.f_start - correction

filterbank.container.f_stop =  filterbank.container.f_stop - correction
print('Min freq: ' + str(filterbank.container.f_start))

print('Max freq: ' + str(filterbank.container.f_stop))
filterbank.plot_waterfall()
window = 0.002

correction = filterbank.container.f_stop - 1380.877634 #know for HIP4436

filterbank.container.f_start =  filterbank.container.f_start - correction

filterbank.container.f_stop =  filterbank.container.f_stop - correction

f_start = (filterbank.container.f_start + (filterbank.container.f_stop - filterbank.container.f_start)/2) - (window/2)

f_stop = f_start + window

filterbank.plot_waterfall(f_start=f_start, f_stop=f_stop)
#Below is a stripped down version of the author's (J. Emilio Enriquez) code here: https://github.com/jeenriquez/Lband_seti/blob/master/analysis/plot_candidates.py



pd.options.mode.chained_assignment = None  # To remove pandas warnings: default='warn'



#------

#Hardcoded values

MAX_DRIFT_RATE = 2.0

OBS_LENGHT = 300.



MAX_PLT_POINTS      = 65536                  # Max number of points in matplotlib plot

MAX_IMSHOW_POINTS   = (8192, 4096)           # Max number of points in imshow plot

MAX_DATA_ARRAY_SIZE = 1024 * 1024 * 1024     # Max size of data array to load into memory

MAX_HEADER_BLOCKS   = 100                    # Max size of header (in 512-byte blocks)



#------





def plot_waterfall(fil, f_start=None, f_stop=None, if_id=0, logged=True,cb=False,freq_label=False,MJD_time=False, **kwargs):

    """ Plot waterfall of data

    Args:

        f_start (float): start frequency, in MHz

        f_stop (float): stop frequency, in MHz

        logged (bool): Plot in linear (False) or dB units (True),

        cb (bool): for plotting the colorbar

        kwargs: keyword args to be passed to matplotlib imshow()

    """





    fontsize=18



    font = {'family' : 'serif',

            'size'   : fontsize}



    matplotlib.rc('font', **font)



    plot_f, plot_data = fil.grab_data(f_start, f_stop, if_id)



    # Make sure waterfall plot is under 4k*4k

    dec_fac_x, dec_fac_y = 1, 1

    if plot_data.shape[0] > MAX_IMSHOW_POINTS[0]:

        dec_fac_x = plot_data.shape[0] / MAX_IMSHOW_POINTS[0]



    if plot_data.shape[1] > MAX_IMSHOW_POINTS[1]:

        dec_fac_y =  plot_data.shape[1] /  MAX_IMSHOW_POINTS[1]



    print(dec_fac_x)

        

    plot_data = rebin(plot_data, int(dec_fac_x), int(dec_fac_y))



    if MJD_time:

        extent=(plot_f[0], plot_f[-1], fil.timestamps[-1], fil.timestamps[0])

    else:

        extent=(plot_f[0], plot_f[-1], (fil.timestamps[-1]-fil.timestamps[0])*24.*60.*60, 0.0)



    this_plot = plt.imshow(plot_data,

        aspect='auto',

        rasterized=True,

        interpolation='nearest',

        extent=extent,

        cmap='viridis_r',

        **kwargs

    )

    if cb:

        plt.colorbar()



    if freq_label:

        plt.xlabel("Frequency [Hz]",fontdict=font)

    if MJD_time:

        plt.ylabel("Time [MJD]",fontdict=font)

    else:

        plt.ylabel("Time [s]",fontdict=font)



    return this_plot



def make_waterfall_plots(filenames_list,target,f_start,f_stop,ion = False,correction_in = 0,**kwargs):

    ''' Makes waterfall plots per group of ON-OFF pairs (up to 6 plots.)

    '''



    fontsize=18



    font = {'family' : 'serif',

            'size'   : fontsize}



    matplotlib.rc('font', **font)



    if ion:

        plt.ion()



    n_plots = len(filenames_list)

    fig = plt.subplots(n_plots, sharex=True, sharey=True,figsize=(10, 2*n_plots))



    fil = Waterfall(filenames_list[0], f_start=f_start, f_stop=f_stop)

    

    A1_avg = np.median(fil.data)

    A1_max = fil.data.max()

    A1_std = np.std(fil.data)



    labeling = ['A','B','A','C','A','D']



    delta_f = np.abs(f_start-f_stop)

    mid_f = np.abs(f_start+f_stop)/2.

    

    #Adjust the incorrect header data

    correction = mid_f - correction_in

    mid_f_text = mid_f - correction



    for i,filename in enumerate(filenames_list):

        plt.subplot(n_plots,1,i+1)



        fil = Waterfall(filename, f_start=f_start, f_stop=f_stop)



        this_plot = plot_waterfall(fil,f_start=f_start, f_stop=f_stop,vmin=A1_avg-A1_std*0,vmax=A1_avg+5.*A1_std,**kwargs)



        if i == 0:

            plt.title(target.replace('HIP','HIP '))



        if i < len(filenames_list)-1:

            plt.xticks(np.arange(f_start, f_stop, delta_f/4.), ['','','',''])



    #Some plot formatting.

    ax = plt.gca()

    ax.get_xaxis().get_major_formatter().set_useOffset(False)

    

    if target == 'HIP7981':

        #f_start -= 0.3

        #f_stop  += 0.3

        factor = 1e3

        units = 'kHz'

    else:

        factor = 1e6

        units = 'Hz'

    

    plt.xticks(np.arange(f_start, f_stop, delta_f/4.),[round(loc_freq) for loc_freq in np.arange((f_start-mid_f), (f_stop-mid_f), delta_f/4.)*factor])

    plt.xlabel("Relative Frequency [%s] from %f MHz"%(units,mid_f_text),fontdict=font)



    #to plot color bar. for now.

    cax = fig[0].add_axes([0.9, 0.11, 0.03, 0.77])

    fig[0].colorbar(this_plot,cax=cax,label='Power [Arbitrary Units]')



    # Fine-tune figure; make subplots close to each other and hide x ticks for

    # all but bottom plot.

    plt.subplots_adjust(hspace=0,wspace=0)
window = 0.002

filterbank_head = Waterfall(file, load_data=False)

f_start = (filterbank_head.container.f_start + (filterbank_head.container.f_stop - filterbank_head.container.f_start)/2) - (window/2)

f_stop = f_start + window



make_waterfall_plots(filenames_list=filenames_list, target='HIP4436', f_start=f_start, f_stop=f_stop, ion = False, correction_in=1380.87763)
file = '../input/hip7981/HIP7981/spliced_blc0001020304050607_guppi_57680_15520_HIP7981_0039.gpuspec.0000.h5'

file2 = '../input/hip7981/HIP7981/spliced_blc0001020304050607_guppi_57680_15867_HIP6917_0040.gpuspec.0000.h5'

file3 = '../input/hip7981/HIP7981/spliced_blc0001020304050607_guppi_57680_16219_HIP7981_0041.gpuspec.0000.h5'

file4 = '../input/hip7981/HIP7981/spliced_blc0001020304050607_guppi_57680_16566_HIP6966_0042.gpuspec.0000.h5'

file5 = '../input/hip7981/HIP7981/spliced_blc0001020304050607_guppi_57680_16915_HIP7981_0043.gpuspec.0000.h5'

file6 = '../input/hip7981/HIP7981/spliced_blc0001020304050607_guppi_57680_17264_HIP6975_0044.gpuspec.0000.h5'



filenames_list = [file,file2,file3,file4,file5,file6]



window = 0.2

filterbank_head = Waterfall(file, load_data=False)

f_start = (filterbank_head.container.f_start + (filterbank_head.container.f_stop - filterbank_head.container.f_start)/2) - (window/2)

f_stop = f_start + window



make_waterfall_plots(filenames_list=filenames_list, target='HIP7981', f_start=f_start, f_stop=f_stop, ion = False, correction_in=1621.24028)
file = '../input/hip65352/HIP65352/spliced_blc02030405_2bit_guppi_57459_34297_HIP65352_0027.gpuspec.0000.h5'

file2 = '../input/hip65352/HIP65352/spliced_blc02030405_2bit_guppi_57459_34623_HIP65352_OFF_0028.gpuspec.0000.h5'

file3 = '../input/hip65352/HIP65352/spliced_blc02030405_2bit_guppi_57459_34949_HIP65352_0029.gpuspec.0000.h5'

file4 = '../input/hip65352/HIP65352/spliced_blc02030405_2bit_guppi_57459_35275_HIP65352_OFF_0030.gpuspec.0000.h5'

file5 = '../input/hip65352/HIP65352/spliced_blc02030405_2bit_guppi_57459_35601_HIP65352_0031.gpuspec.0000.h5'



filenames_list = [file,file2,file3,file4,file5]



window = 0.002

filterbank_head = Waterfall(file, load_data=False)

f_start = (filterbank_head.container.f_start + (filterbank_head.container.f_stop - filterbank_head.container.f_start)/2) - (window/2)

f_stop = f_start + window



make_waterfall_plots(filenames_list=filenames_list, target='HIP65352', f_start=f_start, f_stop=f_stop, ion = False, correction_in=1522.181016)
file = '../input/hip17147/HIP17147/spliced_blc0001020304050607_guppi_57523_69379_HIP17147_0015.gpuspec.0000.h5'

file2 = '../input/hip17147/HIP17147/spliced_blc0001020304050607_guppi_57523_69728_HIP16229_0016.gpuspec.0000.h5'

file3 = '../input/hip17147/HIP17147/spliced_blc0001020304050607_guppi_57523_70077_HIP17147_0017.gpuspec.0000.h5'

file4 = '../input/hip17147/HIP17147/spliced_blc0001020304050607_guppi_57523_70423_HIP16299_0018.gpuspec.0000.h5'

file5 = '../input/hip17147/HIP17147/spliced_blc0001020304050607_guppi_57523_70769_HIP17147_0019.gpuspec.0000.h5'

file6 = '../input/hip17147/HIP17147/spliced_blc0001020304050607_guppi_57523_71116_HIP16341_0020.gpuspec.0000.h5'



filenames_list = [file,file2,file3,file4,file5,file6]



window = 0.002

filterbank_head = Waterfall(file, load_data=False)

f_start = (filterbank_head.container.f_start + (filterbank_head.container.f_stop - filterbank_head.container.f_start)/2) - (window/2)

f_stop = f_start + window



make_waterfall_plots(filenames_list=filenames_list, target='HIP17147', f_start=f_start, f_stop=f_stop, ion = False, correction_in=1379.27751)