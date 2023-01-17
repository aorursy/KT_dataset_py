%matplotlib inline



import dicom # for reading dicom files

import os # for doing directory operations 

import pandas as pd # for some simple data analysis (right now, just to load in the labels data and quickly reference it)



# Change this to wherever you are storing your data:

# IF YOU ARE FOLLOWING ON KAGGLE, YOU CAN ONLY PLAY WITH THE SAMPLE DATA, WHICH IS MUCH SMALLER



data_dir = '../input/sample_images/'

patients = os.listdir(data_dir)

labels_df = pd.read_csv('../input/stage1_labels.csv', index_col=0)





#image[image == -2000] = 0





for patient in patients[:4]:

    label = labels_df.get_value(patient, 'cancer')

    path = data_dir + patient

    

    # a couple great 1-liners from: https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial

    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]

    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))

    print(slices[0].pixel_array.shape, len(slices))



#dcm = dicom.read_file(dcm)

#image=dcm.pixel_array

#dcm.pixel_array.shape
dcm.rows