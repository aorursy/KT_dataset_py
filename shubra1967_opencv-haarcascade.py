from __future__ import print_function

from __future__ import division

import numpy as np

import os

from matplotlib import pyplot as plt

import cv2

import pandas as pd

import seaborn as sns

import scipy.stats as stats

import matplotlib

import datetime

import glob

import csv, io 

%matplotlib inline

print("Package Imported..")

def read_header(infile):

    """Read image header (first 512 bytes)

    """

    h = dict()

    fid = open(infile, 'r+b')

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

print("Read Header Completed")
def read_data(infile):

    """Read any of the 4 types of image files, returns a numpy array of the image contents

    """

    extension = os.path.splitext(infile)[1]

    h = read_header(infile)

    nx = int(h['num_x_pts'])

    ny = int(h['num_y_pts'])

    nt = int(h['num_t_pts'])

    fid = open(infile, 'rb')

    fid.seek(512) #skip header

    if extension == '.aps' or extension == '.a3daps':

        if(h['word_type']==7): #float32

            data = np.fromfile(fid, dtype = np.float32, count = nx * ny * nt)

        elif(h['word_type']==4): #uint16

            data = np.fromfile(fid, dtype = np.uint16, count = nx * ny * nt)

#        data = data * h['data_scale_factor'] #scaling factor

        data = data.reshape(nx, ny, nt, order='F').copy() #make N-d image

    elif extension == '.a3d':

        if(h['word_type']==7): #float32

            data = np.fromfile(fid, dtype = np.float32, count = nx * ny * nt)

        elif(h['word_type']==4): #uint16

            data = np.fromfile(fid, dtype = np.uint16, count = nx * ny * nt)

#        data = data * h['data_scale_factor'] #scaling factor

        data = data.reshape(nx, nt, ny, order='F').copy() #make N-d image

    elif extension == '.ahi':

        data = np.fromfile(fid, dtype = np.float32, count = 2* nx * ny * nt)

        data = data.reshape(2, ny, nx, nt, order='F').copy()

        real = data[0,:,:,:].copy()

        imag = data[1,:,:,:].copy()

    fid.close()

    if extension != '.ahi':

        return data

    else:

        return real, imag
def aps_full_body_coord():

    

    csvfilename='Full_Body_coordinates.csv'

    #outdirname='output' # Commented outputdirname as the dir doesn't exists for this notebook

    outdirname='../working'

    with io.open(outdirname + "/" + csvfilename,'w',encoding='ascii',errors='replace') as out_file:   

        writer = csv.writer(out_file)

        writer.writerow(('Image_Name','File_Extension','x','y','w','h')) 

        

        i =0

        for path in glob.glob("stage1_aps/*.aps"):

            ff=path.split(os.sep)[1]

            img_name = os.path.splitext(ff)[0]

            img_file_extn = os.path.splitext(ff)[1]

            """

            Comment the below mentioned 3 lines to get coordinates of all 1247 files

            """

            i +=1 # Pl comment for all 1247 .aps files

            if i ==20: # Pl comment for all 1247 .aps files

                break # Pl comment for all 1247 .aps files

            image = read_data(path)



            image = image.transpose(1,0,2)

            #print("Image ",image.shape,image[:,:,0].shape, image[:,:,0].dtype, type(image[:,:,0]))





            test1 = (image / 255).round().astype(np.uint8)

            test1 = np.array(test1)



            test_c = np.copy(cv2.cvtColor(test1[:,:,0], cv2.COLOR_GRAY2RGB))

            test_c = cv2.cvtColor(test_c, cv2.COLOR_RGB2GRAY)



            test_d = np.copy(cv2.cvtColor(test1[:,:,0], cv2.COLOR_GRAY2RGB))

            test_d = cv2.cvtColor(test_d, cv2.COLOR_RGB2GRAY)



            test_e = np.copy(test_d)



            face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_fullbody.xml')

            faces1 = face_cascade.detectMultiScale(test_c,scaleFactor=1.001)#,minNeighbors=1,minSize=(100,100),maxSize=(600,600),flags=cv2.CASCADE_SCALE_IMAGE)



            if len(faces1)==0:

                faces1=[[160,100,200,550]]



            elif  len(faces1)>= 2:

                faces1 = [faces1[0]]

            else:

                faces1 = faces1



            for (x,y,w,h) in faces1:

                #print("x,y,w,h ",x,y,w,h)

                if x > 150 or x < 100 :

                    x = 160

                if y > 200   :

                    y = 100

                if w < 150  :

                    w =200

                if h < 400  :

                    h = 550



                cv2.rectangle(test_e, (x-70,0), (x+w+70,y+h), (255,0,0), 3)

                (x,y,w,h)= (x-70,0,w+70,h) 



            writer.writerow((img_name,img_file_extn,x,y,w,h))

            plt.imshow(np.flipud(test_e),cmap='gray')

            #plt.imshow(test_e,cmap='gray')

            plt.show()

            #print("Face ",faces1)

    out_file.close()

    

""" TEST"""   

aps_full_body_coord()
from subprocess import check_output

#print(check_output(["ls", "-l", "../../kaggle/input"]).decode("utf8"))

#print(check_output(["ls", "/"]).decode("utf8"))

#print(check_output(["ls","-l",  "../working"]).decode("utf8"))



"""

Check if the Full Body coordinate has been created and how many rows it has ....

"""

print(check_output(["head","-2",  "../working/Full_Body_coordinates.csv"]).decode("utf8"))

print(check_output(["wc","-l",  "../working/Full_Body_coordinates.csv"]).decode("utf8"))