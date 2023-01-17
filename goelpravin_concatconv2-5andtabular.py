# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#print("test")

#for dirname, _, filenames in os.walk('/kaggle/input'):

    #for filename in filenames:

        #print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pydicom

import plotly

from plotly.graph_objs import *

#import chart_studio.plotly as py

import matplotlib.pyplot as plt

import numpy as np

#import tensorflow_io as tfio

import tensorflow as tf

import scipy.ndimage

from skimage import measure, morphology

#from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import keras_preprocessing

from keras_preprocessing import image

from keras_preprocessing.image import ImageDataGenerator

import keras

from tensorflow.keras import layers

from tensorflow.keras import Model

from tensorflow.keras.mixed_precision import experimental as mixed_precision

import sys

import pandas as pd

from pandas import DataFrame

#from tensorflow import Session





def get_data(filename):

    with open(filename) as file_to_read:

        #first_line = file_to_read.readline()

        row = file_to_read.readline()

        count = 0

        while (row):

            #print(row)

            row = file_to_read.readline()

            count = count+1

    file_to_read.close()

    return count
#total number of files under train = 33026, number of patients = 173, on average 9 weeks training data per patient, so each image will be used 9 times

#so total training paths = 33026 times 9 approx althought image cropping will likely work 90% of times only

#for steps per epoch calc, assume images = 33026 times 9 times 0.9 = approx 250k 

#to avoid memory allocation error,assume batch size of 32 so may be 8000 steps per epoch!



def get_images_patient_count(path):

    patient_count=0

    images_count=0

    patients_to_filter_out_while_setting_pipeline = ['ID00010637202177584971671','ID00011637202177653955184','ID00012637202177665765362','ID00014637202177757139317','ID00015637202177877247924',

                                                     'ID00019637202178323708467','ID00020637202178344345685','ID00023637202179104603099','ID00025637202179541264076','ID00026637202179561894768','ID00027637202179689871102','ID00030637202181211009029',

                                                     'ID00032637202181710233084','ID00035637202182204917484','ID00038637202182690843176','ID00042637202184406822975','ID00047637202184938901501','ID00048637202185016727717','ID00051637202185848464638',

                                                     'ID00052637202186188008618','ID00060637202187965290703','ID00061637202188184085559','ID00062637202188654068490','ID00067637202189903532242','ID00068637202190879923934','ID00072637202198161894406',

                                                     'ID00073637202198167792918','ID00075637202198610425520','ID00076637202199015035026','ID00077637202199102000916','ID00078637202199415319443','ID00082637202201836229724','ID00086637202203494931510',

                                                     'ID00089637202204675567570','ID00090637202204766623410','ID00093637202205278167493','ID00094637202205333947361','ID00099637202206203080121','ID00102637202206574119190','ID00104637202208063407045',

                                                     'ID00105637202208831864134','ID00108637202209619669361','ID00109637202210454292264','ID00110637202210673668310','ID00111637202210956877205','ID00115637202211874187958','ID00117637202212360228007',

                                                     'ID00119637202215426335765','ID00122637202216437668965','ID00123637202217151272140','ID00124637202217596410344','ID00125637202218590429387','ID00126637202218610655908','ID00127637202219096738943',

                                                     'ID00128637202219474716089','ID00129637202219868188000','ID00130637202220059448013','ID00131637202220424084844','ID00132637202222178761324','ID00133637202223847701934','ID00134637202223873059688',

                                                     'ID00135637202224630271439','ID00136637202224951350618','ID00138637202231603868088','ID00139637202231703564336','ID00140637202231728595149','ID00149637202232704462834','ID00161637202235731948764',

                                                     'ID00165637202237320314458','ID00167637202237397919352','ID00168637202237852027833','ID00169637202238024117706','ID00170637202238079193844','ID00172637202238316925179','ID00173637202238329754031',

                                                     'ID00180637202240177410333','ID00183637202241995351650','ID00184637202242062969203','ID00186637202242472088675','ID00190637202244450116191','ID00192637202245493238298','ID00196637202246668775836',

                                                     'ID00197637202246865691526','ID00199637202248141386743','ID00202637202249376026949','ID00207637202252526380974','ID00210637202257228694086','ID00213637202257692916109','ID00214637202257820847190',

                                                     'ID00216637202257988213445','ID00218637202258156844710','ID00219637202258203123958','ID00221637202258717315571','ID00222637202259066229764','ID00224637202259281193413','ID00225637202259339837603',

                                                     'ID00228637202259965313869','ID00229637202260254240583','ID00232637202260377586117','ID00233637202260580149633','ID00234637202261078001846','ID00235637202261451839085','ID00240637202264138860065',

                                                     'ID00241637202264294508775','ID00242637202264759739921','ID00248637202266698862378','ID00249637202266730854017','ID00251637202267455595113','ID00255637202267923028520','ID00264637202270643353440',

                                                     'ID00267637202270790561585','ID00273637202271319294586','ID00275637202271440119890','ID00276637202271694539978','ID00279637202272164826258','ID00283637202278714365037','ID00285637202278913507108',

                                                     'ID00288637202279148973731','ID00290637202279304677843','ID00291637202279398396106','ID00294637202279614924243','ID00296637202279895784347','ID00298637202280361773446','ID00299637202280383305867',

                                                     'ID00305637202281772703145','ID00307637202282126172865','ID00309637202282195513787','ID00312637202282607344793','ID00317637202283194142136','ID00319637202283897208687','ID00322637202284842245491',

                                                     'ID00323637202285211956970','ID00329637202285906759848','ID00331637202286306023714','ID00335637202286784464927','ID00336637202286801879145','ID00337637202286839091062','ID00339637202287377736231',

                                                     'ID00340637202287399835821','ID00341637202287410878488','ID00342637202287526592911','ID00343637202287577133798','ID00344637202287684217717','ID00351637202289476567312','ID00355637202295106567614',

                                                     'ID00358637202295388077032','ID00360637202295712204040','ID00364637202296074419422','ID00365637202296085035729','ID00367637202296290303449','ID00368637202296470751086','ID00370637202296737666151',

                                                     'ID00371637202296828615743','ID00376637202297677828573','ID00378637202298597306391','ID00381637202299644114027','ID00383637202300493233675','ID00388637202301028491611','ID00392637202302319160044',

                                                     'ID00393637202302431697467','ID00398637202303897337979','ID00400637202305055099402','ID00401637202305320178010','ID00405637202308359492977','ID00407637202308788732304','ID00408637202308839708961',

                                                     'ID00411637202309374271828',

                                                     'ID00414637202310318891556','ID00417637202310901214011','ID00419637202311204720264','ID00421637202311550012437','ID00422637202311677017371','ID00423637202312137826377','ID00426637202313170790466']



    for dirname, _, filenames in os.walk(path):

        patient_id = dirname[len(path)+1:]

        #if(patient_id in patients_to_filter_out_while_setting_pipeline):

           #patient_count = patient_count-1

        if(patient_id.find("ID0") == 0):

           patient_count = patient_count+1 

        #dir_name_minus_path = 

        #print("directory:", dirname)

        #for filename in filenames:

            #images_count = images_count+1

            #if(patient_id in patients_to_filter_out_while_setting_pipeline):

                #images_count = images_count-1

            

    return patient_count

#print(count)

#patient_count, images_count = get_images_patient_count('/kaggle/input/osic-pulmonary-fibrosis-progression/test')

#print("patients, images", patient_count,images_count )

def load_dcm_img_bytes(path):

    image_bytes = [tf.io.read_file(path+"/"+ slice) for slice in os.listdir(path)]

    return image_bytes



        

    #patient_dicoms = load_dcmFiles(patient_dicom_path)

     #   patient_images = convert_dcm_slices_to_HU_images(patient_dicoms)

def load_dcmFiles(path):

    slices = [pydicom.dcmread(path + "/" + s) for s in               

              os.listdir(path)]

    #for oneslice in slices:

        #print(oneslice.dir())

        #print(oneslice)

        #print("Pixel Spacing",oneslice.PixelSpacing)

    

    slices = [s for s in slices ]

    slices.sort(key = lambda x: int(x.InstanceNumber))

    try: 

        if "ImagePositionPatient" in slices[0]:

            slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])

        if "SliceThickness" in slices[0]:

            slice_thickness = slices[0].SliceThickness

    except:

        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:

        s.SliceThickness = slice_thickness

    #for s in slices:

        #print(s.dir())

        #print(s.ImagePositionPatient)

    #print("Pixel Spacing",slices[1].PixelSpacing)

    return slices



def load_dcm_Files_file_list(path, file_list):

    slices = [pydicom.dcmread(path + "/" + s) for s in         

              file_list]

    #for oneslice in slices:

        #print(oneslice.dir())

        #print(oneslice)

        #print("Pixel Spacing",oneslice.PixelSpacing)

    

    slices = [s for s in slices ]

    slices.sort(key = lambda x: int(x.InstanceNumber))

    try: 

        if "ImagePositionPatient" in slices[0]:

            slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])

        if "SliceThickness" in slices[0]:

            slice_thickness = slices[0].SliceThickness

    except:

        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:

        s.SliceThickness = slice_thickness

    #for s in slices:

        #print(s.dir())

        #print(s.ImagePositionPatient)

    #print("Pixel Spacing",slices[1].PixelSpacing)

    return slices



def load_dcm_single_file(single_file):

    slices = []

    slices.append(pydicom.dcmread(single_file)) 

    return slices



def convert_dcm_slices_to_HU_images(scans):

    image = np.stack([s.pixel_array for s in scans])

    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0

    # The intercept is usually -1024, so air is approximately 0

    image[image == -2000] = 0

    

    # Convert to Hounsfield units (HU)

    intercept = scans[0].RescaleIntercept

    slope = scans[0].RescaleSlope

    

    if slope != 1:

        image = slope * image.astype(np.float64)

        image = image.astype(np.int16)

        

    image += np.int16(intercept)

    

    return np.array(image, dtype=np.int16)



def resample(image, scan, new_spacing=[1,1,1]):

    # Determine current pixel spacing

    #spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)

    #changed spacing method to be compatible with newer pydicom versions

    spacing = np.array([scan[0].SliceThickness, scan[0].PixelSpacing[0], scan[0].PixelSpacing[1]], dtype=np.float32)

    resize_factor = spacing / new_spacing

    new_real_shape = image.shape * resize_factor

    new_shape = np.round(new_real_shape)

    real_resize_factor = new_shape / image.shape

    new_spacing = spacing / real_resize_factor

    

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    

    return image, new_spacing



def plot_3d(image, threshold=-300):

    

    # Position the scan upright, 

    # so the head of the patient would be at the top facing the camera

    p = image.transpose(2,1,0)

    

    #verts, faces = measure.marching_cubes(p, threshold)

    #new version for skimage.measure does not have marching_cubes, using marching_cubes_lewiner instead

    verts, faces, extraxx1, extraxx2 = measure.marching_cubes_lewiner(p, threshold)

    



    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(111, projection='3d')



    # Fancy indexing: `verts[faces]` to generate a collection of triangles

    mesh = Poly3DCollection(verts[faces], alpha=0.70)

    face_color = [0.45, 0.45, 0.75]

    mesh.set_facecolor(face_color)

    ax.add_collection3d(mesh)



    ax.set_xlim(0, p.shape[0])

    ax.set_ylim(0, p.shape[1])

    ax.set_zlim(0, p.shape[2])



    plt.show()



def largest_label_volume(im, bg=-1):

    vals, counts = np.unique(im, return_counts=True)



    counts = counts[vals != bg]

    vals = vals[vals != bg]



    if len(counts) > 0:

        return vals[np.argmax(counts)]

    else:

        return None



def segment_lung_mask(image, fill_lung_structures=True):

    

    # not actually binary, but 1 and 2. 

    # 0 is treated as background, which we do not want

    binary_image = np.array(image > -320, dtype=np.int8)+1

    labels = measure.label(binary_image)

    

    # Pick the pixel in the very corner to determine which label is air.

    #   Improvement: Pick multiple background labels from around the patient

    #   More resistant to "trays" on which the patient lays cutting the air 

    #   around the person in half

    background_label = labels[0,0,0]

    

    #Fill the air around the person

    binary_image[background_label == labels] = 2

    

    

    # Method of filling the lung structures (that is superior to something like 

    # morphological closing)

    if fill_lung_structures:

        # For every slice we determine the largest solid structure

        for i, axial_slice in enumerate(binary_image):

            axial_slice = axial_slice - 1

            labeling = measure.label(axial_slice)

            l_max = largest_label_volume(labeling, bg=0)

            

            if l_max is not None: #This slice contains some lung

                binary_image[i][labeling != l_max] = 1



    

    binary_image -= 1 #Make the image actual binary

    binary_image = 1-binary_image # Invert it, lungs are now 1

    

    # Remove other air pockets insided body

    labels = measure.label(binary_image, background=0)

    l_max = largest_label_volume(labels, bg=0)

    if l_max is not None: # There are air pockets

        binary_image[labels != l_max] = 0

 

    return binary_image



MIN_BOUND = -1000.0

MAX_BOUND = 400.0

    

def normalize(image):

    #print("normalize start")

    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)

    image[image>1] = 1.

    image[image<0] = 0.

    #print("normalized")

    return image



#PIXEL_MEAN = 0.25

PIXEL_MEAN = -676.4

#-676.4050579921004 as pixel_mean for this training dataset = does that sound right? Lungs are in -500 space and air in -1000, so maybe it is right



def zero_center(image):

    image = image - PIXEL_MEAN

    #print("zero centered")

    return image



def create_zerocenter_norm_images(image):

    return normalize(zero_center(image))

def cropped_masked_lung_images_with_status(masks, images,x_shape=200,y_shape=300 ):

    return_status = False

    imglist=[]

    for i in range(0, len(masks)):

        segmented_lung_slice_mask = masks[i]

        shape1, shape2 = segmented_lung_slice_mask.shape

        x,y=np.nonzero(segmented_lung_slice_mask)

        masked_image = segmented_lung_slice_mask*images[i]

        

        try:

            x_start = x[0]

            y_start = y[0]

            start_offset = shape1-(x_start+x_shape)

            y_offset = shape2-20-y_shape

            if(y_offset>0):

                y_offset=0

            if(start_offset>0):

                start_offset=0

            #print(x_start)

            crop_image = masked_image[x_start+start_offset:x_start+x_shape,y_offset+20:y_offset+20+y_shape]

            #print("crop image shape\t",crop_image.shape)

            #print(x[0],y[0],x[28672],y[28672])

            #print(x,y,x.shape,y.shape)

            totalshape=shape1*shape2

            mask_ratio = np.sum(segmented_lung_slice_mask)/totalshape

            imglist.append(crop_image)

            #if(mask_ratio>0.03 and mask_ratio<0.3):

                #imglist.append(crop_image)

                #return_status = True

                #print("good lung image found")

            #else:

                #return_status = False

                #print("cant see much of lungs here, skipping")

        except:

            pass

    return np.array(imglist, dtype=np.int16), True



def cropped_masked_lung_images(masks, images,x_shape=200,y_shape=300 ):

    imglist=[]

    for i in range(0, len(masks)):

        segmented_lung_slice_mask = masks[i]

        shape1, shape2 = segmented_lung_slice_mask.shape

        x,y=np.nonzero(segmented_lung_slice_mask)

        masked_image = segmented_lung_slice_mask*images[i]

        

        try:

            x_start = x[0]

            y_start = y[0]

            start_offset = shape1-(x_start+x_shape)

            if(start_offset>0):

                start_offset=0

            y_offset = shape2-20-y_shape

            if(y_offset>0):

                y_offset=0



            #print(x_start)

            crop_image = masked_image[x_start+start_offset:x_start+x_shape,y_offset+20:y_offset+20+y_shape]

            #print("crop image shape\t",crop_image.shape)

            #print(x[0],y[0],x[28672],y[28672])

            #print(x,y,x.shape,y.shape)

            totalshape=shape1*shape2

            mask_ratio = np.sum(segmented_lung_slice_mask)/totalshape

            imglist.append(crop_image)

            #if(mask_ratio>0.03 and mask_ratio<0.3):

                #if(crop_image.shape == (200,300)):

                    #imglist.append(crop_image)

                #print("good lung image found")

            #else:

                #print("cant see much of lungs here, skipping")

        except:

            pass

    return np.array(imglist, dtype=np.int16)

        

# paitent id = ID00123637202217151272140 - no good lung seg

# paitent id = ID00405637202308359492977 - works for plot3d; no good lung seg

# patient id = ID00012637202177665765362

#ID00014637202177757139317 - works for plot3d, rest have memory issues

#ID00398637202303897337979 - works for plot3d and has good lung segmentation

#ID00011637202177653955184 - GDCM issue

#ID00329637202285906759848 - no good lung seg

#'ID00323637202285211956970','','ID00331637202286306023714','ID00335637202286784464927','ID00336637202286801879145','ID00337637202286839091062','ID00339637202287377736231'



#patient_dicoms = load_dcmFiles("/kaggle/input/osic-pulmonary-fibrosis-progression/test/ID00422637202311677017371")

#full_file_list = os.listdir("/kaggle/input/osic-pulmonary-fibrosis-progression/test/ID00419637202311204720264")

#first_X_files = full_file_list[0:4]

#patient_dicoms = load_dcm_Files_file_list(patient_dicom_path,full_file_list)

#patient_images = convert_dcm_slices_to_HU_images(patient_dicoms)

#pix_resampled, spacing = resample(patient_images, patient_dicoms, [1,1,1])

#print("Shape before resampling\t", patient_images.shape)

#print("Shape after resampling\t", pix_resampled.shape)

#print ("patient_images shape=" ,patient_images.shape)

#print ("patient_dicom len=" ,len(patient_dicoms))

#segmented_lungs = segment_lung_mask(pix_resampled, False)

#segmented_lungs_fill = segment_lung_mask(pix_resampled, True)

#print("Shape after segmentation\t", segmented_lungs_fill.shape)

#cropped_images = cropped_masked_lung_images(segmented_lungs_fill,pix_resampled)

#zerocenter_norm_ima = create_zerocenter_norm_images(cropped_images)

#print("Shape after cropping\t", zerocenter_norm_ima.shape)

#plt.imshow(cropped_images[1], cmap=plt.cm.bone)

#plt.imshow(zerocenter_norm_ima[1], cmap=plt.cm.bone)



#print("crop image shape\t",cropped_images[1].shape)
#find mean pixel value for the whole training dataset

def get_mean_pixel_value():

    from matplotlib import pyplot as plt

    import pandas as pd

    import numpy as np

    import numpy.ma as ma

    returnDF = None

    filename = "/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv"

    pixelvaluestotal = 0

    pixelcount = 0

    training_data = pd.read_csv(filename)

    training_data.drop_duplicates(keep=False, inplace=True, subset=['Patient', 'Weeks'])

    tr_data_per_patient = training_data.groupby('Patient')

    #TODO: figure out how to do GDCM binding - it seems it is a C++ library and not that direct outside Conda

    patients_needing_GDCM_binding = ['ID00011637202177653955184','ID00052637202186188008618']

    #print(tr_data_per_patient)

    for patient, patient_df in tr_data_per_patient:

        #print(patient)

        if (patient in patients_needing_GDCM_binding):

            continue

        #if (patient in patients_to_filter_out_while_setting_pipeline):

            #continue

        sorted_patient_df = patient_df.sort_values('Weeks')

        sorted_patient_df.fillna(0, inplace=True)

        #let's now load patient images

        #print("loading images")

        patient_dicom_path = "/kaggle/input/osic-pulmonary-fibrosis-progression/train/"+patient

        patient_dicoms = load_dcmFiles(patient_dicom_path)

        patient_images = convert_dcm_slices_to_HU_images(patient_dicoms)

        #print(len(patient_images))

        imglist=[]

        for img in patient_images:

            pixelvaluestotal = pixelvaluestotal+np.sum(img)

            pixelcount = pixelcount+ma.count(img)

    return pixelvaluestotal/pixelcount



#print(get_mean_pixel_value())


TRAINING_DIR = "/kaggle/input/osic-pulmonary-fibrosis-progression/train/"

MODEL_CONFIDENCE=0.025

MAX_SCANS_PER_PATIENT_FOR_TRAINING = 15



policy = mixed_precision.Policy('mixed_float16')

mixed_precision.set_policy(policy)



# Note the input shape is the desired size of the image 200x300 with 1 bytes color

input_images = tf.keras.layers.Input(shape=[200, 300, 1],name='image')

#input_images = Input(shape=[512, 512,1],name='image')

# This is the first convolution

#conv2point5_model = tf.keras.layers.Conv2D(16, (3,3), activation='relu')(input_images)

conv2point5_model = layers.Conv2D(16, (3,3), activation='relu')(input_images)

conv2point5_model = layers.MaxPooling2D(2, 2)(conv2point5_model)

# The second convolution

conv2point5_model = layers.Conv2D(32, (3,3), activation='relu')(conv2point5_model)

conv2point5_model = layers.MaxPooling2D(2, 2)(conv2point5_model)

# The third convolution

conv2point5_model = layers.Conv2D(64, (3,3), activation='relu')(conv2point5_model)

conv2point5_model = layers.MaxPooling2D(2, 2)(conv2point5_model)

# The fourth convolution

conv2point5_model = layers.Conv2D(64, (3,3), activation='relu')(conv2point5_model)

conv2point5_model = layers.MaxPooling2D(2, 2)(conv2point5_model)

# The fifth convolution

conv2point5_model = layers.Conv2D(64, (3,3), activation='relu')(conv2point5_model)

conv2point5_model = layers.GlobalMaxPooling2D()(conv2point5_model)

conv2point5_model = layers.Dense(3, activation="relu")(conv2point5_model)





conv2point5_model = Model(inputs=input_images,outputs=conv2point5_model)



#print(conv2point5_model.summary())

#Asumme tabular inputs will be 5 features(normalized?)  Age, Sex, SmokingStatus, weeksSinceLastFVC, lastFVC 

input_tabular = tf.keras.layers.Input(shape=[4,],name='metadata_and_lastFVC')

#input_tabular = Input(shape=[3,],name='metadata_and_lastFVC')

tabular_model = layers.Dense(20, activation="relu")(input_tabular)

#tabular_model = tf.keras.layers.GlobalMaxPooling1D()(tabular_model)

tabular_model = layers.Dense(3, activation="relu")(tabular_model)



#tabular_model = tf.keras.layers.Conv2D(64, (3,3), activation='relu')(tabular_model)

#tabular_model = tf.keras.layers.MaxPooling2D(2, 2)(tabular_model)



tabular_model = Model(inputs=input_tabular,outputs=tabular_model)



#print(tabular_model.summary())

# combine the output of the two branches

concat = layers.Concatenate()

img_tabular_combined_out = concat([conv2point5_model.output, tabular_model.output])

#img_tabular_combined_out = layers.concatenate([conv2point5_model.output, tabular_model.output])



# apply a FC # 512 neuron hidden layer and then a regression prediction on the combined outputs

z = layers.Dense(512, activation="relu")(img_tabular_combined_out)

#quantiles 0, 1 and 2

#q0 = layers.Dense(1, activation="linear",name='q0')(z)

q1 = layers.Dense(1, activation="linear",dtype='float32',name='q1')(z)

#q2 = layers.Dense(1, activation="linear",name='q2')(z)

# our model will accept the inputs of the two branches and then output a single value

#conv_and_tabular_model = tf.keras.Model((inputs=[conv2point5_model.input, tabular_model.input], outputs=z))

#inputs = {"image": shape=[512,512,3],"metadata_and_lastFVC": shape=[3]}

#conv_and_tabular_model = Model(inputs=[conv2point5_model.input, tabular_model.input], outputs=[q0,q1,q2])

conv_and_tabular_model = Model(inputs=[conv2point5_model.input, tabular_model.input], outputs=q1)

#conv_and_tabular_model = tf.keras.Model(inputs=inputs, outputs=z))

#print("so far so good")

#print(conv_and_tabular_model.summary())

#print(tabular_model.input.shape)



#import torch

def quantile_loss2(target, preds):

    target_c = tf.keras.backend.cast(target, 'float32') 

    preds_c = tf.keras.backend.cast(preds, 'float32')  

    quantiles = (0.2, 0.5, 0.8)

    losses = []

    for i, q in enumerate(quantiles):

        errors = target_c - preds_c[:, i]

        errors1 = (q - 1) * errors

        errors1_c = tf.keras.backend.cast(errors1, 'int32')

        errors2 = q  * errors

        errors2_c = tf.keras.backend.cast(errors2, 'int32')

        maxerror = torch.max(errors1_c,errors2_c)

        #print("errors2_c shape",errors2_c.shape)

        #print("errors2 shape",errors2.shape)

        #losses.append(tf.math.reduce_max(errors1_c, errors2_c))

        #max1 = torch.max((q - 1) * errors)

        #max2 = torch.max((q ) * errors)

        print("max1=",max1)

        print("max2=",max2)

        

        losses.append(maxerror.unsqueeze(1))

        #losses.append(tf.math.reduce_max((q - 1) * errors, q * errors))

        #losses.append(tf.keras.backend.max((q - 1) * errors, q * errors))

        

    #loss = tf.math.reduce_mean(tf.math.reduce_sum(tf.concat(losses, 1), dim=1))

    loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))

    return loss

def quantile_loss(y,output):

    y_c = tf.keras.backend.cast(y, 'float32') 

    output_c = tf.keras.backend.cast(output, 'float32') 

    error = tf.subtract(y_c, output_c)

    q0=0.2

    q1=0.5

    q2=0.8

    losses = []

    lossq0 = tf.reduce_mean(tf.maximum(q0*error, (q0-1)*error), axis=-1)

    lossq1 = tf.reduce_mean(tf.maximum(q1*error, (q1-1)*error), axis=-1)

    lossq2 = tf.reduce_mean(tf.maximum(q2*error, (q2-1)*error), axis=-1)

    losses.append(lossq0)

    losses.append(lossq1)

    losses.append(lossq2)

    

    loss = tf.reduce_mean(tf.add_n(losses))

    return loss

def minus_laplace_log_loss(fvc_true,fvc_pred):

    #cast both fvc_true and fvc_pred to either both int or both float

    fvc_true_c = tf.keras.backend.cast(fvc_true, 'float32') 

    fvc_pred_c = tf.keras.backend.cast(fvc_pred, 'float32') 



    confidence = MODEL_CONFIDENCE * fvc_pred_c

    confidence_clipped = confidence

    confidence_clipped = tf.math.maximum(confidence_clipped, tf.constant(70.0))

    #if(confidence>70):

        #confidence_clipped = 70.0

    delta = tf.abs(tf.subtract(fvc_true_c,fvc_pred_c))

    delta = tf.math.maximum(delta, tf.constant(1000.0))

    #if(delta>1000):

        #delta = 1000.0

    metric_part1 = tf.math.divide_no_nan(tf.math.multiply_no_nan(tf.sqrt(2.0),delta),confidence_clipped)

    

    metric_part2 = tf.math.log(tf.math.multiply_no_nan(tf.sqrt(2.0),confidence_clipped))

    metric =  tf.add(metric_part1,metric_part2)

    return metric

        

    

#from tensorflow.keras.optimizers import RMSprop

conv_and_tabular_model.compile(

    optimizer=keras.optimizers.RMSprop(1e-3),

    metrics=[tf.keras.metrics.MeanSquaredError()],

    loss='mse'

#    loss=minus_laplace_log_loss

#    loss=quantile_loss

#    loss={'q0':'mae',

#           'q1':'mae',

#           'q2':'mae'},

#    loss_weights={'q0':0.2,'q1':0.5,'q2':0.8}

    #loss_weights={'q0':1,'q1':1,'q2':1}

        )

keras.utils.plot_model(conv_and_tabular_model, "multi_input_and_output_model.png", show_shapes=True)
@tf.function

def load_training_datapoint(datapoint1,datapoint2):

    image = datapoint1['image']

    metadata_and_lastFVC = datapoint1['metadata_and_lastFVC']

    fvc = datapoint2

    x = {"image": image, "metadata_and_lastFVC":metadata_and_lastFVC }

    y = fvc

    return x,y





def get_training_data_gen():

  # The first line contains the column headers

  # Each successive line contians 7 columns: Patient,Weeks, FVC, PercentAge, Sex, SmokingStatus

  # We also need to add two extra columns : lastFVC and weeksSinceLastFVC  

  # We should groupby for a patient, sort by weeks and then for the minimum week, copy FVC value into lastFVC 

  # also and weeksSinceLastFVC = 0 for the minimum week. For next week value copy lastFVC from previous week and 

  # calculate weeksSinceLastFVC as difference between this weeks's value and previous one. 

  # Then FVC value is the label

  # The function will return labels, dicom_slices and metadata_and_lastFVC

  # 

    from matplotlib import pyplot as plt

    import pandas as pd

    import numpy as np

    labels_list = []

    dicom_slices_list = []

    metadata_and_lastFVC = []

    returnDF = None

    filename = "/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv"

    training_data = pd.read_csv(filename)



    training_data.drop_duplicates(keep=False, inplace=True, subset=['Patient', 'Weeks'])

    #Initialize lastFVC as FVC and weeksSinceLastFVC as Weeks

    training_data['lastFVC'] = training_data['FVC']

    training_data['weeksSinceLastFVC'] = training_data['Weeks']

    training_data['MaleFemale'] = training_data['Weeks']

    training_data['codedSmokingStatus'] = training_data['Weeks']

    tr_data_per_patient = training_data.groupby('Patient')



    #training_complete_copy = training_data.copy()

    #training_data = training_complete_copy.sample(frac=(1-validation_split), random_state=0)

    #testing_data = training_complete_copy.drop(training_data.index)

    

    #if(validation_data):

        #tr_data_per_patient = training_data.groupby('Patient')

    #else:

        #tr_data_per_patient = testing_data.groupby('Patient')

        

    #TODO: figure out how to do GDCM binding - it seems it is a C++ library and not that direct outside Conda

    patients_needing_GDCM_binding = ['ID00011637202177653955184','ID00052637202186188008618']

    #only to restrict data while setting and testing pipeline

    patients_to_filter_out_while_setting_pipeline = ['ID00010637202177584971671','ID00011637202177653955184','ID00012637202177665765362','ID00014637202177757139317','ID00015637202177877247924',

                                                     'ID00019637202178323708467','ID00020637202178344345685','ID00023637202179104603099','ID00025637202179541264076','ID00026637202179561894768','ID00027637202179689871102','ID00030637202181211009029',

                                                     'ID00032637202181710233084','ID00035637202182204917484','ID00038637202182690843176','ID00042637202184406822975','ID00047637202184938901501','ID00048637202185016727717','ID00051637202185848464638',

                                                     'ID00052637202186188008618','ID00060637202187965290703','ID00061637202188184085559','ID00062637202188654068490','ID00067637202189903532242','ID00068637202190879923934','ID00072637202198161894406',

                                                     'ID00073637202198167792918','ID00075637202198610425520','ID00076637202199015035026','ID00077637202199102000916','ID00078637202199415319443','ID00082637202201836229724','ID00086637202203494931510',

                                                     'ID00089637202204675567570','ID00090637202204766623410','ID00093637202205278167493','ID00094637202205333947361','ID00099637202206203080121','ID00102637202206574119190','ID00104637202208063407045',

                                                     'ID00105637202208831864134','ID00108637202209619669361','ID00109637202210454292264','ID00110637202210673668310','ID00111637202210956877205','ID00115637202211874187958','ID00117637202212360228007',

                                                     'ID00119637202215426335765','ID00122637202216437668965','ID00123637202217151272140','ID00124637202217596410344','ID00125637202218590429387','ID00126637202218610655908','ID00127637202219096738943',

                                                     'ID00128637202219474716089','ID00129637202219868188000','ID00130637202220059448013','ID00131637202220424084844','ID00132637202222178761324','ID00133637202223847701934','ID00134637202223873059688',

                                                     'ID00135637202224630271439','ID00136637202224951350618','ID00138637202231603868088','ID00139637202231703564336','ID00140637202231728595149','ID00149637202232704462834','ID00161637202235731948764',

                                                     'ID00165637202237320314458','ID00167637202237397919352','ID00168637202237852027833','ID00169637202238024117706','ID00170637202238079193844','ID00172637202238316925179','ID00173637202238329754031',

                                                     'ID00180637202240177410333','ID00183637202241995351650','ID00184637202242062969203','ID00186637202242472088675','ID00190637202244450116191','ID00192637202245493238298','ID00196637202246668775836',

                                                     'ID00197637202246865691526','ID00199637202248141386743','ID00202637202249376026949','ID00207637202252526380974','ID00210637202257228694086','ID00213637202257692916109','ID00214637202257820847190',

                                                     'ID00216637202257988213445','ID00218637202258156844710','ID00219637202258203123958','ID00221637202258717315571','ID00222637202259066229764','ID00224637202259281193413','ID00225637202259339837603',

                                                     'ID00228637202259965313869','ID00229637202260254240583','ID00232637202260377586117','ID00233637202260580149633','ID00234637202261078001846','ID00235637202261451839085','ID00240637202264138860065',

                                                     'ID00241637202264294508775','ID00242637202264759739921','ID00248637202266698862378','ID00249637202266730854017','ID00251637202267455595113','ID00255637202267923028520','ID00264637202270643353440',

                                                     'ID00267637202270790561585','ID00273637202271319294586','ID00275637202271440119890','ID00276637202271694539978','ID00279637202272164826258','ID00283637202278714365037','ID00285637202278913507108',

                                                     'ID00288637202279148973731','ID00290637202279304677843','ID00291637202279398396106','ID00294637202279614924243','ID00296637202279895784347','ID00298637202280361773446','ID00299637202280383305867',

                                                     'ID00305637202281772703145','ID00307637202282126172865','ID00309637202282195513787','ID00312637202282607344793','ID00317637202283194142136','ID00319637202283897208687','ID00322637202284842245491',

                                                     'ID00323637202285211956970','ID00329637202285906759848','ID00331637202286306023714','ID00335637202286784464927','ID00336637202286801879145','ID00337637202286839091062','ID00339637202287377736231',

                                                     'ID00340637202287399835821','ID00341637202287410878488','ID00342637202287526592911','ID00343637202287577133798','ID00344637202287684217717','ID00351637202289476567312','ID00355637202295106567614',

                                                     'ID00358637202295388077032','ID00360637202295712204040','ID00364637202296074419422','ID00365637202296085035729','ID00367637202296290303449','ID00368637202296470751086','ID00370637202296737666151',

                                                     'ID00371637202296828615743','ID00376637202297677828573','ID00378637202298597306391','ID00381637202299644114027','ID00383637202300493233675','ID00388637202301028491611','ID00392637202302319160044',

                                                     'ID00393637202302431697467','ID00398637202303897337979','ID00400637202305055099402','ID00401637202305320178010','ID00405637202308359492977','ID00407637202308788732304','ID00408637202308839708961',

                                                     'ID00411637202309374271828',

                                                     'ID00414637202310318891556','ID00417637202310901214011','ID00419637202311204720264','ID00421637202311550012437','ID00422637202311677017371','ID00423637202312137826377','ID00426637202313170790466']



    #print(tr_data_per_patient)

    df_row_count = 0

    rows=[]

    rows_image=[]

    rows_fvc=[]

    dtype = np.float32

    for patient, patient_df in tr_data_per_patient:

        #print(patient)

        if (patient in patients_needing_GDCM_binding):

            continue

        #TODO comment once pipeline has been tested

        #if (patient in patients_to_filter_out_while_setting_pipeline):

            #continue

        sorted_patient_df = patient_df.sort_values('Weeks')

        sorted_patient_df.lastFVC =  sorted_patient_df.FVC.shift(1)

        sorted_patient_df.weeksSinceLastFVC =  sorted_patient_df.Weeks -sorted_patient_df.Weeks.shift(1)

        try:

            sorted_patient_df.lastFVC[0] = sorted_patient_df.FVC[0]

        except:

            pass

        sorted_patient_df.fillna(0, inplace=True)

        #add column MaleFemale with 1 where Male and 2 where Female

        sorted_patient_df.MaleFemale = np.where(sorted_patient_df['Sex']=='Male', 1, 2)

        #Currently smokes=1 Ex-smoker=2 else 3

        sorted_patient_df.codedSmokingStatus = np.where(sorted_patient_df['SmokingStatus']=='Currently smokes', 1, np.where(sorted_patient_df['SmokingStatus']=='Ex-smoker',2,3))

        #normalize age simply by divide-by-100 and storing that in column called AgeNorm

        sorted_patient_df['AgeNorm'] = (sorted_patient_df['Age']) / 100

         

        #let's now load patient images

        #print("loading images")

        patient_dicom_path = "/kaggle/input/osic-pulmonary-fibrosis-progression/train/"+patient

        full_file_list = os.listdir(patient_dicom_path)

        if(len(full_file_list)>MAX_SCANS_PER_PATIENT_FOR_TRAINING):

            first_X_files = full_file_list[0:MAX_SCANS_PER_PATIENT_FOR_TRAINING]

            #print("will get zero center images for", patient)

            #patient_zerocenter_norm_im = get_fixed_number_of_cropped_images(patient_dicom_path,NUM_CROPS)

            #patient_dicoms = load_dcmFiles(patient_dicom_path)

            patient_dicoms = load_dcm_Files_file_list(patient_dicom_path,first_X_files)

        else:

            patient_dicoms = load_dcmFiles(patient_dicom_path)

        #patient_dicoms = load_dcmFiles(patient_dicom_path)

        patient_images = convert_dcm_slices_to_HU_images(patient_dicoms)

        pix_resampled, spacing = resample(patient_images, patient_dicoms, [1,1,1])

        segmented_lungs_fill = segment_lung_mask(pix_resampled, True)

        #print("Shape after segmentation\t", segmented_lungs_fill.shape)

        cropped_images = cropped_masked_lung_images(segmented_lungs_fill,pix_resampled)

        zerocenter_norm_im = create_zerocenter_norm_images(cropped_images)



        

        if(len(zerocenter_norm_im)>MAX_SCANS_PER_PATIENT_FOR_TRAINING):

            zerocenter_norm_im_trim = zerocenter_norm_im[0:MAX_SCANS_PER_PATIENT_FOR_TRAINING]

        

        for row in sorted_patient_df.itertuples():

            for img in zerocenter_norm_im_trim:

                image = np.expand_dims(img,2)#first add the single channel at the end

                tabular_input = [row.Weeks,row.AgeNorm,row.MaleFemale,row.codedSmokingStatus]

                label = row.FVC

                yield (np.asarray(image,dtype=dtype),np.asarray(tabular_input,dtype=dtype)),np.asarray(label,dtype=dtype)

    

 #--------END of train_gen       

 



    

class myCallback (tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):

        print("epoch callback")

        #if(logs.get('accuracy')>0.8):

            #print("\nReached 80% accuracy, so stopping further epochs")

            #self.model.stop_training = True



callbacks = myCallback()

#conv_and_tabular_model.summary()



BATCH_SIZE = 32

BUFFER_SIZE = 1000

VALIDATION_SPLIT = 0.2



training_dataset_full = tf.data.Dataset.from_generator(get_training_data_gen, output_types=((tf.float32,tf.float32),tf.float32))

train_patient_count = get_images_patient_count('/kaggle/input/osic-pulmonary-fibrosis-progression/train')

validation_size = int((VALIDATION_SPLIT)*MAX_SCANS_PER_PATIENT_FOR_TRAINING*train_patient_count*8) #assuming 8 weeks per patient



training_size = int((1-VALIDATION_SPLIT)*MAX_SCANS_PER_PATIENT_FOR_TRAINING*train_patient_count*8 )



train_dataset = training_dataset_full.take(training_size)

validation_dataset = training_dataset_full.skip(training_size)

validation_dataset = validation_dataset.take(validation_size)

#training_dataset = training_dataset.map(load_training_datapoint)

train_dataset_cached = train_dataset.cache().batch(BATCH_SIZE).repeat()

train_dataset = train_dataset_cached.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

validation_dataset = validation_dataset.batch(BATCH_SIZE)

#Steps per epoch = number of training images/batch-size in training generator. Likewise validation steps

if(training_size==0):

    training_size = 36000 #dummy number to avoid issues



if(validation_size==0):

    validation_size = 200 #dummy number to avoid issues



number_of_epochs = 30



steps_per_epoch = training_size//BATCH_SIZE

validation_steps = validation_size//BATCH_SIZE



model_history = conv_and_tabular_model.fit(train_dataset, epochs=number_of_epochs,

                         steps_per_epoch=steps_per_epoch,

                         validation_steps=validation_steps,

                         validation_data=validation_dataset,

                         #verbose=2,

                         callbacks=[callbacks])



import matplotlib.pyplot as plt

loss = model_history.history['loss']

#val_loss = model_history.history['val_loss']



epochs = range(len(loss))





plt.figure()



plt.plot(epochs, loss, 'r', label='Training Loss')

#plt.plot(epochs, val_loss, 'b', label='Validation Loss')

plt.title('Training and validation loss')

plt.legend()



plt.show()
conv_and_tabular_model.save("/kaggle/working/osic_model")
for dirname, _, filenames in os.walk('/kaggle/working'):

   for filename in filenames:

        path = os.path.join(dirname, filename)

        size = os.stat(path).st_size

        print(filename, size)

#multi_input_and_output_model.png 142881

#__notebook_source__.ipynb 263

#saved_model.pb 276973

#variables.index 3689

#variables.data-00000-of-00001 825341

#Looks like it is taking about 1MB to save model

def update_submission(patient,week,fvc, confidence):

    patient_week_to_updt = patient + "_" + str(week)

    #print(patient_week_to_updt)

    #subs_df.FVC = np.where(subs_df['Patient_Week']=='patient_week_to_updt', fvc, 2000)

    #subs_df.Confidence = np.where(subs_df['Patient_Week']=='patient_week_to_updt', confidence, 100)

    subs_df.loc[subs_df['Patient_Week'] == patient_week_to_updt, 'FVC'] = fvc

    subs_df.loc[subs_df['Patient_Week'] == patient_week_toupdt, 'Confidence'] = confidence

    #subs_df['Patient_Week'==patient_week_to_updt].FVC = fvc

    #subs_df['Patient_Week'==patient_week_to_updt].Confidence = confidence



def nextFVC_and_confidence(patient_df, zerocenter_norm_im, lastFVC,weekincrement):



    AgeNorm = patient_df.AgeNorm

    MaleFemale = patient_df.MaleFemale

    codedSmokingStatus = patient_df.codedSmokingStatus

    Percent = patient_df.Percent

    inputs2 = np.asarray([[weekincrement,lastFVC,AgeNorm,MaleFemale,codedSmokingStatus]])

    q0FVC=0

    q1FVC=0

    q2FVC=0

    q1min=0

    q1max=0

    #print(inputs2)

    for img in zerocenter_norm_im:

        #print(inputs2)

        image = np.asarray([img])

        weekFVC = conv_and_tabular_model.predict({"image": image, "metadata_and_lastFVC": inputs2})

        #q0FVC = q0FVC+weekFVC[0]

        #q1FVC = q1FVC+weekFVC[1]

        q1FVC = q1FVC+weekFVC

        #q2FVC = q2FVC+weekFVC[2]

        if(weekFVC>q1max):

            q1max=weekFVC

        if(weekFVC<q1min):

            q1min=weekFVC

    datapoints = len(zerocenter_norm_im)

    if(datapoints>0):

        predictedFVC = q1FVC/datapoints

        #confidence = (q2FVC-q0FVC)/datapoints

        confidence = MODEL_CONFIDENCE*predictedFVC

        lower_confidence = 0.015*predictedFVC

        upper_confidence = 0.04*predictedFVC

        if(confidence<lower_confidence):

            confidence = lower_confidence

        if(confidence>upper_confidence):

            confidence = upper_confidence

            

    else:

        #dummy data if we could not extract ANY lung images

        predictedFVC = 2000

        confidence = 100

        return predictedFVC, confidence

        

    #print(type(predictedFVC))

    return predictedFVC.item(), confidence.item()



def get_fvc_and_confidence(patient_df,zerocenter_norm_im, week):



    AgeNorm = patient_df.AgeNorm

    MaleFemale = patient_df.MaleFemale

    codedSmokingStatus = patient_df.codedSmokingStatus

    Percent = patient_df.Percent

    inputs2 = np.asarray([[week,AgeNorm,MaleFemale,codedSmokingStatus]])

    q0FVC=0

    q1FVC=0

    q2FVC=0

    q1min=0

    q1max=0

    #print(inputs2)

    for img in zerocenter_norm_im:

        #print(inputs2)

        #print("image found:",img)

        image = np.asarray([img])

        weekFVC = conv_and_tabular_model.predict({"image": image, "metadata_and_lastFVC": inputs2})

        #q0FVC = q0FVC+weekFVC[0]

        #q1FVC = q1FVC+weekFVC[1]

        q1FVC = q1FVC+weekFVC

        #q2FVC = q2FVC+weekFVC[2]

        if(weekFVC>q1max):

            q1max=weekFVC

        if(weekFVC<q1min):

            q1min=weekFVC

    datapoints = len(zerocenter_norm_im)

    if(datapoints>0):

        predictedFVC = q1FVC/datapoints

        #confidence = (q2FVC-q0FVC)/datapoints

        confidence = MODEL_CONFIDENCE*predictedFVC

        lower_confidence = 0.015*predictedFVC

        upper_confidence = 0.04*predictedFVC

        if(confidence<lower_confidence):

            confidence = lower_confidence

        if(confidence>upper_confidence):

            confidence = upper_confidence

            

    else:

        #dummy data if we could not extract ANY lung images

        predictedFVC = 2000

        confidence = 100

        return predictedFVC, confidence

        

    #print(type(predictedFVC))

    return predictedFVC.item(), confidence.item()

#prepare submissions file - move this block after nextFVC_and_conf method

#The number of patients should be read from test.csv

def get_fixed_number_of_cropped_images(patient_dicom_path,NUM_CROPS):

    

    imglist = []

    crop_count = 0

    for single_file in os.listdir(patient_dicom_path):

        #print("single_file", single_file)

        single_file_path = patient_dicom_path + "/" + single_file

        patient_dicoms = load_dcm_single_file(single_file_path)

        #print("dicom for single file loaded",single_file_path)



        patient_images = convert_dcm_slices_to_HU_images(patient_dicoms)

        #print("hu for single file done")

        pix_resampled, spacing = resample(patient_images, patient_dicoms, [1,1,1])

        #print("Shape before resampling\t", patient_images.shape)

        #print("Shape after resampling\t", pix_resampled.shape)

        #print ("patient_images shape=" ,patient_images.shape)

        segmented_lungs_fill = segment_lung_mask(pix_resampled, True)

        #print("Shape after segmentation\t", segmented_lungs_fill.shape)

        cropped_image, success_in_cropping = cropped_masked_lung_images_with_status(segmented_lungs_fill,pix_resampled)

        #print("Shape after cropping\t", cropped_image.shape)

        zerocenter_image = create_zerocenter_norm_images(cropped_image)

        #print("Shape after zero center\t", zerocenter_image.shape)

        imglist.append(zerocenter_image)

        #print("crop success:",success_in_cropping)

        if(success_in_cropping):

            crop_count = crop_count +1

            #plt.imshow(cropped_image, cmap=plt.cm.bone)

        if(crop_count>NUM_CROPS):

            print("breaking crop count",crop_count)

            break

    #patient_zerocenter_norm_im = np.array(imglist, dtype=np.int16)

    return imglist

    



filename = "/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv"

test_data = pd.read_csv(filename)

test_data['MaleFemale'] = test_data['Weeks']

test_data['codedSmokingStatus'] = test_data['Weeks']

#add column MaleFemale with 1 where Male and 2 where Female

test_data.MaleFemale = np.where(test_data['Sex']=='Male', 1, 2)

#Currently smokes=1 Ex-smoker=2 else 3

test_data.codedSmokingStatus = np.where(test_data['SmokingStatus']=='Currently smokes', 1, np.where(test_data['SmokingStatus']=='Ex-smoker',2,3))

#normalize age simply by divide-by-100 and storing that in column called AgeNorm

test_data['AgeNorm'] = (test_data['Age']) / 100

#total_patients_to_predict_on = len(test_data.index)

total_patients_to_predict_on = 2

submission_row_list_for_one_row = []

submission_data_list = []

sub_patient_week_list = []

sub_fvc_list = []

sub_conf_list = []

sub_patient_list=[]

sub_week_list=[]

NUM_CROPS = 3

NUM_DCM = 3

for patient_number in range(0,total_patients_to_predict_on,1):

    patientdf = test_data.iloc[patient_number]

    patient = patientdf.Patient

    try:

        patient_dicom_path = "/kaggle/input/osic-pulmonary-fibrosis-progression/test/"+patient

        #attempt to speed up prediction - instead of predicting on all slices, restrict to first NUM_CROPS OR NUM_DCM

        full_file_list = os.listdir(patient_dicom_path)

        if(len(full_file_list)>NUM_DCM):

            first_X_files = full_file_list[0:NUM_DCM]

            #print("will get zero center images for", patient)

            #patient_zerocenter_norm_im = get_fixed_number_of_cropped_images(patient_dicom_path,NUM_CROPS)

            #patient_dicoms = load_dcmFiles(patient_dicom_path)

            patient_dicoms = load_dcm_Files_file_list(patient_dicom_path,first_X_files)

            

            patient_images = convert_dcm_slices_to_HU_images(patient_dicoms)

            pix_resampled, spacing = resample(patient_images, patient_dicoms, [1,1,1])

            segmented_lungs_fill = segment_lung_mask(pix_resampled, True)

            cropped_images = cropped_masked_lung_images(segmented_lungs_fill,pix_resampled)

            #print("Shape after cropping\t", cropped_images.shape)

            #print("length=",len(cropped_images))

            

            patient_zerocenter_norm_im = create_zerocenter_norm_images(cropped_images)

            start_index = NUM_DCM

            

            while(len(patient_zerocenter_norm_im)==0):

                #in this case keep trying with 4 extra DCMs till the length is not zero or files exhausted

                end_index = start_index + NUM_DCM

                if end_index>(len(full_file_list)-1):

                    break

                first_X_files = full_file_list[start_index:end_index]

                #print("will get zero center images for", patient)

                #patient_zerocenter_norm_im = get_fixed_number_of_cropped_images(patient_dicom_path,NUM_CROPS)

                #patient_dicoms = load_dcmFiles(patient_dicom_path)

                patient_dicoms = load_dcm_Files_file_list(patient_dicom_path,first_X_files)

                patient_images = convert_dcm_slices_to_HU_images(patient_dicoms)

                pix_resampled, spacing = resample(patient_images, patient_dicoms, [1,1,1])

                segmented_lungs_fill = segment_lung_mask(pix_resampled, True)

                cropped_images = cropped_masked_lung_images(segmented_lungs_fill,pix_resampled)

                #print("Shape after cropping\t", cropped_images.shape)

                #print("length=",len(cropped_images))

                patient_zerocenter_norm_im = create_zerocenter_norm_images(cropped_images)

                start_index = end_index

                    

                

                

        print("patient images for a patient prepared for prediction----------")



        #print("patient"+ patient_number+" images loaded---------------------------")

    except:

        imglist = []

        #print("exception happened")

        patient_zerocenter_norm_im = np.array(imglist, dtype=np.int16)

    

    #if(len(patient_zerocenter_norm_im)>0):

        #print("A sample image being used while predicting:")

        #plt.imshow(patient_zerocenter_norm_im[1], cmap=plt.cm.bone)

    

    starting_week_fwd = patientdf.Weeks+1

    starting_week_bwd = patientdf.Weeks-1

    #MAX_WEEK=9

    #MIN_WEEK=5

    MIN_WEEK=-13

    MAX_WEEK=134



    predictedFVC =patientdf.FVC

    patient_week_given = patient + "_" + str(patientdf.Weeks)

    confidence_given = 1#dont know if there is a divide by zero issue in eval script and hence

    sub_patient_week_list.append(patient_week_given)

    sub_fvc_list.append(predictedFVC)

    sub_conf_list.append(confidence_given)

    sub_patient_list.append(patient)

    sub_week_list.append(patientdf.Weeks)

    for week in range(starting_week_fwd,MAX_WEEK,1):

        patient_week = patient + "_" + str(week)

        #print(patient_week)

        #predictedFVC, confidence = nextFVC_and_confidence(patientdf,patient_zerocenter_norm_im,predictedFVC,1)

        predictedFVC, confidence = get_fvc_and_confidence(patientdf,patient_zerocenter_norm_im, week)

        print(patient_week,predictedFVC,confidence)

        sub_patient_week_list.append(patient_week)

        sub_fvc_list.append(int(predictedFVC))

        sub_conf_list.append(int(confidence))

        sub_patient_list.append(patient)

        sub_week_list.append(week)



        #submission_row_list_for_one_row.append([patient_week,predictedFVC,confidence])

        #submission_data_list.append(submission_row_list_for_one_row)

        #update_submission(patient0df.Patient,week,predictedFVC, confidence)

    #now walk back from week 5 to week -12

    predictedFVC=patientdf.FVC

    for week in range(starting_week_bwd,MIN_WEEK,-1):

        patient_week = patient + "_" + str(week)

        #print(patient_week)

        #predictedFVC, confidence = nextFVC_and_confidence(patientdf,patient_zerocenter_norm_im,predictedFVC,-1)

        predictedFVC, confidence = get_fvc_and_confidence(patientdf,patient_zerocenter_norm_im, week)

        print(patient_week,predictedFVC,confidence)

        sub_patient_week_list.append(patient_week)

        sub_fvc_list.append(int(predictedFVC))

        sub_conf_list.append(int(confidence))

        sub_patient_list.append(patient)

        sub_week_list.append(week)



        #submission_row_list_for_one_row.append([patient_week,predictedFVC,confidence])

        #submission_data_list.append(submission_row_list_for_one_row)

        #update_submission(patient0df.Patient,week,predictedFVC, confidence)

subs_data = {'Patient_Week':sub_patient_week_list, 'FVC':sub_fvc_list,'Confidence':sub_conf_list,'Patient':sub_patient_list,'Week':sub_week_list} 

submission_df = pd.DataFrame(subs_data)

submission_df = submission_df.sort_values(by=['Patient', 'Week'], ascending=True)

submission_df = submission_df.drop(columns=['Patient', 'Week'])



submission_df

submission_df.to_csv('submission.csv', index=False)
submission_df