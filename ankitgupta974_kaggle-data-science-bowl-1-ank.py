import dicom # read the dicom files
import os # do directory operations
import pandas as pd #nice for data annalysis

data_dir = '../input/sample_images/'
patients = os.listdir(data_dir)
labels_df = pd.read_csv('../input/stage1_labels.csv',index_col=0)

labels_df 

for patient in patients[:10]:
    label = labels_df.get_value(patient, 'cancer')
    path  = data_dir + patient
    slices = [dicom.read_file(path + '/' + s ) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
    print(len(slices), label)
    print(slices[0])
     
print(len(slices), slices[0].pixel_array.shape)
print(len(slices),slices[0].PixelData)
len(patients)
import matplotlib.pyplot as plt
import cv2
import numpy as np
IMG_PX_SIZE =150

for patient in patients[:10]:
    label =labels_df.get_value(patient,'cancer')
    path = data_dir + patient
    slices = [dicom.read_file(path + '/' + s ) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
    
    fig=plt.figure()
    for num,each_slice in enumerate(slices[:12]):
        y =fig.add_subplot(3,4,num+1)
        new_image = cv2.resize(np.array(each_slice.pixel_array),(IMG_PX_SIZE,IMG_PX_SIZE))
        y.imshow(new_image)
        plt.show()
    
 

