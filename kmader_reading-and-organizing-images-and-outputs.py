import numpy as np

import os

import matplotlib

import matplotlib.pyplot as plt

from glob import glob

import pandas as pd

import seaborn as sns

plt.style.use('ggplot')

sns.set_style("whitegrid", {'axes.grid' : False})

import matplotlib.animation as mpl_animation

matplotlib.rc('animation', html='html5')

%matplotlib inline
def read_header(infile):

    """Read image header (first 512 bytes)

    """

    h = dict()

    with open(infile, 'r+b') as fid:

        h['filename'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 20))

        h['parent_filename'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 20))

        h['comments1'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 80))

        h['comments2'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 80))

        h['energy_type'] = np.fromfile(fid, dtype = np.int16, count = 1)

        h['config_type'] = np.fromfile(fid, dtype = np.int16, count = 1)

        h['file_type'] = np.fromfile(fid, dtype = np.int16, count = 1)

        h['trans_type'] = np.fromfile(fid, dtype = np.int16, count = 1)

        h['scan_type'] = np.fromfile(fid, dtype = np.int16, count = 1)

        h['data_type'] = np.fromfile(fid, dtype = np.int16, count = 1)

        h['date_modified'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 16))

        h['frequency'] = np.fromfile(fid, dtype = np.float32, count = 1)

        h['mat_velocity'] = np.fromfile(fid, dtype = np.float32, count = 1)

        h['num_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)

        h['num_polarization_channels'] = np.fromfile(fid, dtype = np.int16, count = 1)

        h['spare00'] = np.fromfile(fid, dtype = np.int16, count = 1)

        h['adc_min_voltage'] = np.fromfile(fid, dtype = np.float32, count = 1)

        h['adc_max_voltage'] = np.fromfile(fid, dtype = np.float32, count = 1)

        h['band_width'] = np.fromfile(fid, dtype = np.float32, count = 1)

        h['spare01'] = np.fromfile(fid, dtype = np.int16, count = 5)

        h['polarization_type'] = np.fromfile(fid, dtype = np.int16, count = 4)

        h['record_header_size'] = np.fromfile(fid, dtype = np.int16, count = 1)

        h['word_type'] = np.fromfile(fid, dtype = np.int16, count = 1)

        h['word_precision'] = np.fromfile(fid, dtype = np.int16, count = 1)

        h['min_data_value'] = np.fromfile(fid, dtype = np.float32, count = 1)

        h['max_data_value'] = np.fromfile(fid, dtype = np.float32, count = 1)

        h['avg_data_value'] = np.fromfile(fid, dtype = np.float32, count = 1)

        h['data_scale_factor'] = np.fromfile(fid, dtype = np.float32, count = 1)

        h['data_units'] = np.fromfile(fid, dtype = np.int16, count = 1)

        h['surf_removal'] = np.fromfile(fid, dtype = np.uint16, count = 1)

        h['edge_weighting'] = np.fromfile(fid, dtype = np.uint16, count = 1)

        h['x_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)

        h['y_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)

        h['z_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)

        h['t_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)

        h['spare02'] = np.fromfile(fid, dtype = np.int16, count = 1)

        h['x_return_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)

        h['y_return_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)

        h['z_return_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)

        h['scan_orientation'] = np.fromfile(fid, dtype = np.int16, count = 1)

        h['scan_direction'] = np.fromfile(fid, dtype = np.int16, count = 1)

        h['data_storage_order'] = np.fromfile(fid, dtype = np.int16, count = 1)

        h['scanner_type'] = np.fromfile(fid, dtype = np.int16, count = 1)

        h['x_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)

        h['y_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)

        h['z_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)

        h['t_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)

        h['num_x_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)

        h['num_y_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)

        h['num_z_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)

        h['num_t_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)

        h['x_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)

        h['y_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)

        h['z_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)

        h['x_acc'] = np.fromfile(fid, dtype = np.float32, count = 1)

        h['y_acc'] = np.fromfile(fid, dtype = np.float32, count = 1)

        h['z_acc'] = np.fromfile(fid, dtype = np.float32, count = 1)

        h['x_motor_res'] = np.fromfile(fid, dtype = np.float32, count = 1)

        h['y_motor_res'] = np.fromfile(fid, dtype = np.float32, count = 1)

        h['z_motor_res'] = np.fromfile(fid, dtype = np.float32, count = 1)

        h['x_encoder_res'] = np.fromfile(fid, dtype = np.float32, count = 1)

        h['y_encoder_res'] = np.fromfile(fid, dtype = np.float32, count = 1)

        h['z_encoder_res'] = np.fromfile(fid, dtype = np.float32, count = 1)

        h['date_processed'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 8))

        h['time_processed'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 8))

        h['depth_recon'] = np.fromfile(fid, dtype = np.float32, count = 1)

        h['x_max_travel'] = np.fromfile(fid, dtype = np.float32, count = 1)

        h['y_max_travel'] = np.fromfile(fid, dtype = np.float32, count = 1)

        h['elevation_offset_angle'] = np.fromfile(fid, dtype = np.float32, count = 1)

        h['roll_offset_angle'] = np.fromfile(fid, dtype = np.float32, count = 1)

        h['z_max_travel'] = np.fromfile(fid, dtype = np.float32, count = 1)

        h['azimuth_offset_angle'] = np.fromfile(fid, dtype = np.float32, count = 1)

        h['adc_type'] = np.fromfile(fid, dtype = np.int16, count = 1)

        h['spare06'] = np.fromfile(fid, dtype = np.int16, count = 1)

        h['scanner_radius'] = np.fromfile(fid, dtype = np.float32, count = 1)

        h['x_offset'] = np.fromfile(fid, dtype = np.float32, count = 1)

        h['y_offset'] = np.fromfile(fid, dtype = np.float32, count = 1)

        h['z_offset'] = np.fromfile(fid, dtype = np.float32, count = 1)

        h['t_delay'] = np.fromfile(fid, dtype = np.float32, count = 1)

        h['range_gate_start'] = np.fromfile(fid, dtype = np.float32, count = 1)

        h['range_gate_end'] = np.fromfile(fid, dtype = np.float32, count = 1)

        h['ahis_software_version'] = np.fromfile(fid, dtype = np.float32, count = 1)

        h['spare_end'] = np.fromfile(fid, dtype = np.float32, count = 10)

    return h
from collections import namedtuple

from warnings import warn

ScanData = namedtuple('ScanData', ['header', 'data', 'real', 'imag', 'extension'])

def read_data(infile):

    """Read any of the 4 types of image files, returns a numpy array of the image contents

    """

    _, extension = os.path.splitext(infile)

    sd_dict = {'header': None, 'data': None, 'real': None, 'imag': None, 'extension': extension}

    

    h = read_header(infile)

    sd_dict['header'] = h

    nx = int(h['num_x_pts'])

    ny = int(h['num_y_pts'])

    nt = int(h['num_t_pts'])

    with open(infile, 'rb') as fid:

        fid.seek(512) #skip header

        if extension == '.aps' or extension == '.a3daps':

            if(h['word_type']==7): #float32

                data = np.fromfile(fid, dtype = np.float32, count = nx * ny * nt)

            elif(h['word_type']==4): #uint16

                data = np.fromfile(fid, dtype = np.uint16, count = nx * ny * nt)

            data = data * h['data_scale_factor'] #scaling factor

            data = data.reshape(nx, ny, nt, order='F').copy() #make N-d image

        elif extension == '.a3d':

            if(h['word_type']==7): #float32

                data = np.fromfile(fid, dtype = np.float32, count = nx * ny * nt)

            elif(h['word_type']==4): #uint16

                data = np.fromfile(fid, dtype = np.uint16, count = nx * ny * nt)

            data = data * h['data_scale_factor'] #scaling factor

            data = data.reshape(nx, nt, ny, order='F').copy() #make N-d image

        elif extension == '.ahi':

            data = np.fromfile(fid, dtype = np.float32, count = 2* nx * ny * nt)

            data = data.reshape(2, ny, nx, nt, order='F').copy()

            real = data[0,:,:,:].copy()

            imag = data[1,:,:,:].copy()

            sd_dict['real'] = real

            sd_dict['imag'] = imag

        else:

            warn('Extension not really supported: {}'.format(extension), RuntimeWarning)

            data = None

        sd_dict['data'] = data

        

    return ScanData(**sd_dict)
from skimage.util.montage import montage2d

def plot_montage(sdata):

    if sdata.data is not None:

        print('input data shape', sdata.data.shape)

        fig = plt.figure(figsize = (16,16))

        ax = fig.add_subplot(111)

        ax.imshow(montage2d(np.flipud(sdata.data).swapaxes(0,2)), cmap = 'viridis')

        return fig

def plot_mip(sdata, mip_func = np.max):

    if sdata.data is not None:

        print('input data shape', sdata.data.shape)

        fig, m_axs = plt.subplots(1, 3, figsize = (16,16))

        n_data = np.flipud(sdata.data).swapaxes(0,2)

        for i, (c_name, c_ax)  in enumerate(zip(['xy', 'yz', 'xz'], m_axs)):

            c_ax.imshow(mip_func(n_data,i), cmap = 'viridis')

            c_ax.set_title('%s MIP Projection' % (c_name))

        return fig

            

def plot_animation(sdata):

    if sdata.data is not None:

        print('input data shape', sdata.data.shape)

        fig = plt.figure(figsize = (16,16))

        ax = fig.add_subplot(111)

        def animate(i):

            im = ax.imshow(np.flipud(sdata.data[:,:,i].transpose()), cmap = 'viridis')

            return [im]

        return mpl_animation.FuncAnimation(fig, animate, frames=range(0,sdata.data.shape[2]), interval=200, blit=True)
from keras.utils.np_utils import to_categorical

base_dir = os.path.join('..', 'input')

label_df = pd.read_csv(os.path.join(base_dir, 'stage1_labels.csv'))

label_df['ImageId'] = label_df['Id'].map(lambda x: x.split('_')[0])

label_df['ImageZoneId'] = label_df['Id'].map(lambda x: int(x.split('_')[1][4:]))

# create a vector with each category being an image zone

label_df['ImageZoneVec'] = label_df.apply(

    lambda c_row: [c_row['Probability']*to_categorical(c_row['ImageZoneId']-1, 

                                                      num_classes = label_df['ImageZoneId'].max())],1)

label_df

label_df.sample(3)
def vec_agg(x):

    rslt = dict()

    for col in x.columns:

        rslt[col]=x[col].tolist()

    return pd.Series(rslt)



agg_label_df = label_df[['ImageId','ImageZoneVec']].groupby('ImageId').apply(vec_agg).drop('ImageId', axis = 1)

agg_label_df = agg_label_df.reset_index()

agg_label_df['ImageZoneVec'] = agg_label_df['ImageZoneVec'].map(lambda x: np.sum(np.hstack(x)[0],0))

agg_label_df['AverageProbability'] = agg_label_df['ImageZoneVec'].map(lambda x: np.mean(x))

agg_label_df.sample(2)
files_df = pd.DataFrame([dict(ImageId = os.path.splitext(os.path.basename(x))[0], 

                              path = x,

                             extension = os.path.splitext(x)[1]) 

                         for x in glob(os.path.join(base_dir, 'sample', '*'))])

files_df.sample(2)
def path_agg(x):

    rslt = dict()

    for c_ext, c_path in zip(x['extension'], x['path']):

        rslt[c_ext.replace('.', '')] = c_path

        if c_ext == '.a3d':

            # read the full header for the 3d files

            for i,j in read_header(c_path).items():

                rslt[i] = j

    return pd.Series(rslt)

full_files_df = files_df.groupby('ImageId').apply(path_agg).reset_index()

full_files_df.sample(2)
if False:

    for _, c_row in full_files_df.sample(1).iterrows():

        full_3d = read_data(c_row['a3d'])

        fig = plot_mip(full_3d, np.sum)
if False:

    for _, c_row in full_files_df.sample(1).iterrows():

        full_aps = read_data(c_row['aps'])

        plot_montage(full_aps)
comb_df = pd.merge(full_files_df, agg_label_df)

comb_df.sample(2)