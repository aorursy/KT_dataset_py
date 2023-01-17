import os

import numpy as np

import pandas as pd

from struct import unpack
#PATH = '/data/passenger-screening/data/interim/'

#FILES = [file for file in os.listdir(PATH) if file != '.gitkeep']
def read_header(file, path): # good default value path=PATH

    """Read image header (first 512 bytes)"""

    h = dict()

    with open(path+file, 'rb') as f:

        header = f.read(512)

    h['filename'], h['parent_filename'], h['comments1'], h['comments2'] = unpack('20s20s80s80s', header[:200])

    h['energy_type'], h['config_type'], h['file_type'], h['trans_type'], h['scan_type'], h['data_type'] = unpack('6h', header[200:212])

    h['date_modified'], h['frequency'], h['mat_velocity'], h['num_pts'] = unpack('16s2fi', header[212:240])

    h['num_polarization_channels'], h['spare00'] = unpack('2h', header[240:244])

    h['adc_min_voltage'], h['adc_max_voltage'], h['band_width'] = unpack('3f', header[244:256])

    h['spare01'], h['spare02'], h['spare03'], h['spare04'], h['spare05'] = unpack('5h', header[256:266])

    h['polar_t1'], h['polar_t2'], h['polar_t3'], h['polar_t4'] = unpack('4h', header[266:274])

    h['record_header_size'], h['word_type'], h['word_precision'] = unpack('3h', header[274:280])

    h['min_data_value'], h['max_data_value'], h['avg_data_value'], h['data_scale_factor'] = unpack('4f', header[280:296])

    h['data_units'] = unpack('h', header[296:298])

    h['surf_removal'], h['edge_weighting'], h['x_units'], h['y_units'], h['z_units'], h['t_units'] = unpack('6H', header[298:310])

    h['spare06'] = unpack('h', header[310:312])

    h['x_return_speed'], h['y_return_speed'], h['z_return_speed'] = unpack('3f', header[312:324])

    h['scan_orientation'], h['scan_direction'], h['data_storage_order'], h['scanner_type'] = unpack('4h', header[324:332])

    h['x_inc'], h['y_inc'], h['z_inc'], h['t_inc'] = unpack('4f', header[332:348])

    h['num_x_pts'], h['num_y_pts'], h['num_z_pts'], h['num_t_pts'] = unpack('4i', header[348:364])

    h['x_speed'], h['y_speed'], h['z_speed'] = unpack('3f', header[364:376])

    h['x_acc'], h['y_acc'], h['z_acc'] = unpack('3f', header[376:388])

    h['x_motor_res'], h['y_motor_res'], h['z_motor_res'] = unpack('3f', header[388:400])

    h['x_encoder_res'], h['y_encoder_res'], h['z_encoder_res'] = unpack('3f', header[400:412])

    h['date_processed'], h['time_processed'] = unpack('8s8s', header[412:428])

    h['depth_recon'], h['x_max_travel'], h['y_max_travel'], h['elevation_offset_angle'] = unpack('4f', header[428:444])

    h['roll_offset_angle'], h['z_max_travel'], h['azimuth_offset_angle'] = unpack('3f', header[444:456])

    h['adc_type'], h['spare06'], h['scanner_radius'] = unpack('2hf', header[456:464])

    h['x_offset'], h['y_offset'], h['z_offset'], h['t_delay'] = unpack('4f', header[464:480])

    h['range_gate_start'], h['range_gate_end'], h['ahis_software_version'] = unpack('3f', header[480:492])

    h['spare07'], h['spare08'], h['spare09'], h['spare10'], h['spare11'], h['spare12'], \

    h['spare13'], h['spare14'], h['spare15'], h['spare16'] = unpack('10h', header[492:])

    return h
def read_header_orig(file, path): # good default value path=PATH

    """Read image header (first 512 bytes)"""

    h = dict()

    with open(path+file, 'rb') as fid:

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

        h['spare01'], h['spare02'], h['spare03'], h['spare04'], h['spare05'] = np.fromfile(fid, dtype = np.int16, count = 5)

        h['polar_t1'], h['polar_t2'], h['polar_t3'], h['polar_t4'] = np.fromfile(fid, dtype = np.int16, count = 4)

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

        h['spare06'] = np.fromfile(fid, dtype = np.int16, count = 1)

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

        h['spare07'], h['spare08'], h['spare09'], h['spare10'], h['spare11'], h['spare12'], \

        h['spare13'], h['spare14'], h['spare15'], h['spare16'] = np.fromfile(fid, dtype = np.int16, count = 10)

    return h
#%timeit [read_header(file) for file in FILES]

#%timeit pd.DataFrame([read_header_orig(file) for file in FILES])

#%timeit pd.DataFrame([read_header(file) for file in FILES])
#df = pd.DataFrame([read_header_orig(file) for file in FILES])

#df.head()
#df = pd.DataFrame([read_header(file) for file in FILES])

#df.head()
def read_data(file, path): # good default value path=PATH

    """Read any of the 4 types of image files, returns a numpy array of the image contents"""

    extension = file.split('.')[-1]

    with open(path+file, 'rb') as f:

        header = f.read(512)

        word_type = unpack('h', header[276:278])

        scale_factor = unpack('f', header[292:296])

        nx, ny, nz, nt = unpack('4i', header[348:364])

        if extension == 'ahi':

            data = np.fromfile(f, dtype=np.float32, count=2*nx*ny*nt)

            data = data.reshape(2, ny, nx, nt, order='F')

            return data[0,:,:,:], data[1,:,:,:]

        else:

            if word_type == 7: #float32

                data = np.fromfile(f, dtype=np.float32, count=nx*ny*nt)

            else:

                data = np.fromfile(f, dtype=np.uint16, count=nx*ny*nt)

            data = data * scale_factor

            if extension == 'a3d':

                return data.reshape(nx, nt, ny, order='F')

            return data.reshape(nx, ny, nt, order='F')
#img = read_data(FILES[0])

#img.shape