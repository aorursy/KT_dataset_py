%matplotlib inline

import SimpleITK as sitk

import numpy as np

import csv

from glob import glob

import pandas as pd

import os

import matplotlib

import matplotlib.pyplot as plt



        

def get_filename(file_list, case):      

    for f in file_list:

        if case in f:

            return(f)

            

def extract_cubir_from_mhd(dcim_path,annatation_file,output_path):

    '''

      @param: dcim_path :       the path contains all mhd file

      @param: annatation_file:  the annatation csv file,contains every nodules' coordinate

      @param: output_path       the extracted cubic of size 20x20x6,30x30x10,40x40x26 npy file,every nodule end up withs three size

    '''

    file_list=glob(dcim_path+"*.mhd")

    # The locations of the nodes

    df_node = pd.read_csv(annatation_file)

    df_node["file"] = df_node["seriesuid"].map(lambda file_name: get_filename(file_list, file_name))

    df_node = df_node.dropna()



    for img_file in file_list:

        mini_df = df_node[df_node["file"]==img_file] #get all nodules associate with file

        file_name = str(img_file).split("/")[-1]

        if mini_df.shape[0]>0: # some files may not have a nodule--skipping those 

            # load the data once

            itk_img = sitk.ReadImage(img_file) 

            img_array = sitk.GetArrayFromImage(itk_img) # indexes are z,y,x (notice the ordering)

            num_z, height, width = img_array.shape        #heightXwidth constitute the transverse plane

            origin = np.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)

            spacing = np.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)

            # go through all nodes 

            print("begin to process nodules...")

            print(img_array.shape)

            for node_idx, cur_row in mini_df.iterrows():       

                node_x = cur_row["coordX"]

                node_y = cur_row["coordY"]

                node_z = cur_row["coordZ"]             

                # every nodules saved into size of 20x20x6,30x30x10,40x40x26 

                imgs1 = np.ndarray([20,20,6],dtype=np.float32)

                imgs2 = np.ndarray([30,30,10],dtype=np.float32)

                imgs3 = np.ndarray([40,40,26],dtype=np.float32)



                center = np.array([node_x, node_y, node_z])   # nodule center

                v_center = np.rint((center-origin)/spacing)  # nodule center in voxel space (still x,y,z ordering)

                # take care on the sequence of axis of v_center ,is z,y,x not x,y,z

                imgs1[:,:,:]=img_array[int(v_center[2]-10):int(v_center[2]+10),int(v_center[1]-10):int(v_center[1]+10),int(v_center[0]-3):int(v_center[0]+3)]

                imgs2[:,:,:]=img_array[int(v_center[2]-15):int(v_center[2]+15),int(v_center[1]-15):int(v_center[1]+15),int(v_center[0]-5):int(v_center[0]+5)]

                imgs3[:,:,:]=img_array[int(v_center[2]-20):int(v_center[2]+20),int(v_center[1]-20):int(v_center[1]+20),int(v_center[0]-13):int(v_center[0]+13)]         

                np.save(os.path.join(output_path,"images_%s_%d_size10x10.npy" % (str(file_name), node_idx)),imgs1)

                np.save(os.path.join(output_path,"images_%s_%d_size20x20.npy" % (str(file_name), node_idx)),imgs2)

                np.save(os.path.join(output_path,"images_%s_%d_size40x40.npy" % (str(file_name), node_idx)),imgs3)    

                print("nodules %s from image %s extracted finished!..."%(node_idx,str(file_name)))

# a plot function to check the extraction

def plot_cubic(npy_file):

    cubic_array = np.load(npy_file)

    f, plots = plt.subplots(int(cubic_array.shape[2]/3), 3, figsize=(50, 50))

    for i in range(1, cubic_array.shape[2]+1):

        plots[int(i / 3), int((i % 3) )].axis('off')

        plots[int(i / 3), int((i % 3) )].imshow(cubic_array[:,:,i], cmap=plt.cm.bone)
# let's take a look

dcim_path = '/data/LUNA2016/lung_imgs/subset0/'

annatation_file = '/data/LUNA2016/lung_imgs/evaluationScript/annotations/annotations.csv'

output_path = '/data/LUNA2016/cubic_npy'

extract_cubir_from_mhd(dcim_path,annatation_file,output_path)

print("finished!...")