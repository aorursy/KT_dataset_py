import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import pydicom



import os

import time
import imageio

from IPython.display import Image

from skimage.measure import label, regionprops
INPUT_FOLDER = '/kaggle/input/osic-pulmonary-fibrosis-progression/train/'



patients = os.listdir(INPUT_FOLDER)

patients.sort()



print('some patients: \n', '\n'.join(patients[:5]))
def load_scan(path):

    '''

    Loads scans from folder into a list

    Args path of images to load

    Returns images path in a list

    '''

    slices = [pydicom.read_file(path +'/'+ s) for s in os.listdir(path)]

    slices.sort(key = lambda x: int(x.InstanceNumber))

    

    try:

        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[0].ImagePositionPatient[2])

    except:

        slice[0].slice_location - slice[1].slice_location

    for s in slices:

        s.SliceThickness = slice_thickness

    

    return slices

        
#helper function to convert dicom images to houndsfield



def get_pixels_hu(scans):

    '''

    converts raw image files into hounsfield unit

    Arguments: raw images

    Returns: images numpy array

    '''

    image = np.stack([s.pixel_array for s in scans])

    image = image.astype(np.int16)

    # Since the scanning equipment is cylindrical in nature and image output is square,

    # we set the out-of-scan pixels to 0

    

    image[image ==-2000] = 0

    intercept = scans[0].RescaleIntercept

    slope = scans[0].RescaleSlope

    

    if slope != 1:

        image = slope * image.astype(np.float64)

        image = image.astype(np.int16)

        

    image += np.int16(intercept)

    

    return np.array(image, dtype=np.int16)

path = INPUT_FOLDER + patients[24]

test_patient_scan = load_scan(path)

test_patient_images = get_pixels_hu(test_patient_scan)
path = INPUT_FOLDER + patients[24]

slices = [pydicom.read_file(path +'/'+ s) for s in os.listdir(path)]
plt.imshow(test_patient_images[12]) ;

plt.title('original image slice 12');
from skimage import measure, morphology, segmentation

import scipy.ndimage as ndimage
def generate_markers(image):

    '''

    Generate markers for a given image

    Arguments: image

    returns : Internal marker, external marker, watershed marker

    '''

    #creation the internal marker

    marker_internal = image < -400

    marker_internal = segmentation.clear_border(marker_internal)

    marker_internal_labels = measure.label(marker_internal)

    

    areas = [r.area for r in measure.regionprops(marker_internal_labels)]

    areas.sort()

    

    if len(areas) > 2:

        for region in measure.regionprops(marker_internal_labels):

            if region.area < areas[-2]:

                for coordinates in region.coords:

                    marker_internal_labels[coordinates[0], coordinates[1]] == 0

                    

    marker_internal = marker_internal_labels > 0

    

    #creation of external marker

    external_a = ndimage.binary_dilation(marker_internal, iterations=10)

    external_b = ndimage.binary_dilation(marker_internal, iterations=55)

    marker_external = external_b ^ external_a

    

    #creation of watershed marker

    marker_watershed = np.zeros((512,512), dtype=np.int)

    marker_watershed += marker_internal * 255

    marker_watershed += marker_external * 128

    

    return marker_internal, marker_external, marker_watershed

    
test_patient_internal, test_patient_external, test_patient_watershed = generate_markers(

                                                                        test_patient_images[12])

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize= (15,15))



ax1.imshow(test_patient_internal)#, cmap='gray')

ax1.set_title('internal_marker')

ax1.axis('off')



ax2.imshow(test_patient_external)

ax2.set_title('external_marker')



ax3.imshow(test_patient_watershed)

ax3.set_title('watershed')
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(15,15))



ax1.imshow(test_patient_internal, cmap='gray')

ax1.set_title("Internal Marker")

ax1.axis('off')



ax2.imshow(test_patient_external, cmap='gray')

ax2.set_title("External Marker")

ax2.axis('off')



ax3.imshow(test_patient_watershed, cmap='gray')

ax3.set_title("Watershed Marker")

ax3.axis('off')
#list to store computation times and iterations

compute_time = []

iter_titles = []
def seperate_lungs(image, iterations=1):

    """

    Segments lungs using various techniques

    Parameters: image (scan images) iteration (number of iteration)

    returns:  -Segmented Lung, -Lung Filter, -Outline Lung, 

              -watershed Lung, -Sobel Gradient

    """

    # Store the start time

    start = time.time()

    

    marker_internal, marker_external, marker_watershed = generate_markers(image)

    

    """

    creation of sobel gradient

    """

    # Sobel Gradient

    sobel_filtered_dx = ndimage.sobel(image, 1)

    sobel_filtered_dy = ndimage.sobel(image, 0)

    sobel_gradient = np.hypot(sobel_filtered_dx, sobel_filtered_dy)

    sobel_gradient *= 255.0 / np.max(sobel_gradient)

    

    """

    using algorithm watershed

    

    we pass the image convoluted by sobel and watershed marker

    to morphology watershed and get a matrix matrix labelled using 

    the watershed segmentation algorithm

    """

    watershed = morphology.watershed(sobel_gradient, marker_watershed)

    """

    Reducing the image to outlines after watershed algorithm

    """

    outline = ndimage.morphological_gradient(watershed, size=(3,3))

    outline = outline.astype(bool)

    """

    Black Top-Hart morphology:

    

    

    """

    #structuring element used for filter

    blackhat_struct = [[0, 0, 1, 1, 1, 0, 0],

                        [0, 1, 1, 1, 1, 1, 0],

                        [1, 1, 1, 1, 1, 1, 1],

                        [1, 1, 1, 1, 1, 1, 1],

                        [1, 1, 1, 1, 1, 1, 1],

                        [0, 1, 1, 1, 1, 1, 0],

                        [0, 0, 1, 1, 1, 0, 0]]

    blackhat_struct = ndimage.iterate_structure(blackhat_struct, iterations)

    #Perform black top-hat filter

    outline = ndimage.black_tophat(outline, structure= blackhat_struct)

    """

    Generate internal filter using internal marker and outline

    """

    lung_filter = np.bitwise_or(marker_internal, outline)

    lung_filter = ndimage.morphology.binary_closing(lung_filter, structure=np.ones((3,3)), iterations=1)

    """

    Segment lung using lungfilter and the image

    """

    segmented = np.where(lung_filter==1, image, -2000* np.ones((512, 512)))

    

    #Append Computation time

    end = time.time()

    compute_time.append(end - start)

    iter_titles.append("{num} iterations".format(num=iterations))

    

    return segmented, lung_filter, outline, watershed, sobel_gradient
for itr in range(1,9):

    (test_segmented, test_lung_filter, test_outline,

    test_watershed, test_sobel_gradient) = seperate_lungs(test_patient_images[12], itr)#test_patient_images[12]
itr_dict = {'Iterations' : iter_titles, 'computation time (in seconds)' : compute_time}

colors = ['#30336b',] * 8

colors[0] = '#ed4d4b'



import plotly.express as px

import plotly.graph_objects as go



fig = go.Figure(data=[go.Bar(

            x= itr_dict['Iterations'],

            y= itr_dict['computation time (in seconds)'],

            marker_color = colors

            )])



fig.update_traces(texttemplate= '%{y:.3s}', textposition= 'outside')



fig.update_layout(

            title = 'Iterations vs computation times',

            yaxis = dict(

                    title='Computation time (in seconds)',

                    titlefont_size= 16,

                    tickfont_size= 14,

                        ),

            autosize= False,

            width= 800,

            height= 700,)



fig.show()
f, ax = plt.subplots(1,2, sharey=True, figsize=(12,12))

ax[0].imshow(test_sobel_gradient)

ax[0].set_title('test sobel gradient')

ax[0].axis('off')



ax[1].imshow(test_watershed)

ax[1].set_title('test_watershed')

ax[1].axis('off')



plt.show()
f, (ax1, ax2, ax3) = plt.subplots(1,3, sharex=True, figsize=(12,12))



ax1.imshow(test_outline)

ax1.set_title('test_outline')

ax2.imshow(test_lung_filter)

ax2.set_title('test_lung_filter')

ax2.axis('off')

ax3.imshow(test_segmented)

ax3.set_title('test_segment')

ax3.axis('off')



plt.show()
f, ax = plt.subplots(1,2, sharey=True, figsize=(14, 12))

ax[0].imshow(test_patient_images[12])

ax[0].set_title('original image')

ax[0].axis('off')



ax[1].imshow(test_segmented)

ax[1].set_title('segmented image')

ax[1].axis('off')



plt.show()
def set_lungwin(img, hu=[-1200., 600.]):

    lungwin = np.array(hu)

    newimg = (img-lungwin[0]) / (lungwin[1]-lungwin[0])

    newimg[newimg < 0] = 0

    newimg[newimg > 1] = 1

    newimg = (newimg * 255).astype('uint8')

    return newimg





scans = load_scan('../input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430/')

scan_array = set_lungwin(get_pixels_hu(scans))



imageio.mimsave("/tmp/gif.gif", scan_array, duration=0.0001)

Image(filename="/tmp/gif.gif", format='png')
slices = [path +'/'+ s for s in os.listdir(path)]

slices[5]
#sample_image = pydicom.dcmread(scans[7])

sample_image = pydicom.dcmread(slices[7])

img = sample_image.pixel_array



plt.imshow(img, cmap='gray') ;

#print(img.value)
img = (img + sample_image.RescaleIntercept) / sample_image.RescaleSlope

img = img < -400 #HU unit range for lung CT scans

f, (ax1, ax2) = plt.subplots(1,2, figsize=(10, 10))



ax1.imshow(img, cmap='gray')

ax1.set_title('Binary mask image')



img = segmentation.clear_border(img)

ax2.imshow(img, cmap = 'gray')

ax2.set_title('cleaned border image')
img = label(img)

plt.imshow(img, cmap='gray') ;
len([r for r in regionprops(img)])

areas = [r.area for r in regionprops(img)]

areas.sort()

if len(areas) > 2:

    for region in regionprops(img):

        if region.area < areas[-2]:

            for coordinates in region.coords:

                img[coordinates[0], coordinates[1]] = 0

img = img > 0

plt.imshow (img)
import tensorflow as tf

import random



from tensorflow.keras import Model

import tensorflow.keras.backend as K

import tensorflow.keras.layers as L

import tensorflow.keras.models as M

from tensorflow.keras.optimizers import Nadam

from tensorflow.keras.utils import Sequence



import seaborn as sns

from PIL import Image
from tqdm.notebook import tqdm

import cv2
def seed_everything(seed= 2020):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    tf.random.set_seed(seed)

    tf.random.set_seed(seed)

    

seed_everything(42)
#tf.compact allows us to write code that works both in tf 1.x and 2.x e.g tf.compact.v2 allows us

#to use things introduced in 2.x from 1.x

config = tf.compat.v1.ConfigProto()

config.gpu_options.allow_growth = True

session = tf.compat.v1.Session(config= config)
train = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
train.sample(5)
def get_tab(df):

    vector = [(df.Age.values[0] - 30) / 30]

    

    if df.Sex.values[0] == 'male':

        vector.append(0)

    else:

        vector.append(1)

        

    if df['SmokingStatus'].values[0] == 'Never Smoked':

        vector.extend([0, 0])

    elif df['SmokingStatus'].values[0] == 'Ex-smoker':

        vector.extend([1,1])

    elif df['SmokingStatus'].values[0] == 'Currently Smokes':

        vector.extend([0, 1])

    else:

        vector.extend([1, 0])

    

    return np.array(vector)

!pip install ../input/kerasapplications/keras-team-keras-applications-3b180cb -f ./ --no-index

!pip install ../input/efficientnet/efficientnet-1.1.0/ -f ./ --no-index
A = {} 

TAB = {} 

P = [] 

for i, p in tqdm(enumerate(train.Patient.unique())):

    sub = train.loc[train.Patient == p, :] 

    fvc = sub.FVC.values

    weeks = sub.Weeks.values

    c = np.vstack([weeks, np.ones(len(weeks))]).T

    a, b = np.linalg.lstsq(c, fvc)[0]

    

    A[p] = a

    TAB[p] = get_tab(sub)

    P.append(p)
def get_img(path):

    d = pydicom.dcmread(path)

    return cv2.resize(d.pixel_array / 2**11, (512, 512))
from tensorflow.keras.utils import Sequence



class IGenerator(Sequence):

    BAD_ID = ['ID00011637202177653955184', 'ID00052637202186188008618']

    def __init__(self, keys, a, tab, batch_size=32):

        self.keys = [k for k in keys if k not in self.BAD_ID]

        self.a = a

        self.tab = tab

        self.batch_size = batch_size

        

        self.train_data = {}

        for p in train.Patient.values:

            self.train_data[p] = os.listdir(f'../input/osic-pulmonary-fibrosis-progression/train/{p}/')

    

    def __len__(self):

        return 1000

    

    def __getitem__(self, idx):

        x = []

        a, tab = [], [] 

        keys = np.random.choice(self.keys, size = self.batch_size)

        for k in keys:

            try:

                i = np.random.choice(self.train_data[k], size=1)[0]

                img = get_img(f'../input/osic-pulmonary-fibrosis-progression/train/{k}/{i}')

                x.append(img)

                a.append(self.a[k])

                tab.append(self.tab[k])

            except:

                print(k, i)

       

        x,a,tab = np.array(x), np.array(a), np.array(tab)

        x = np.expand_dims(x, axis=-1)

        return [x, tab] , a
from tensorflow.keras.layers import (

    Dense, Dropout, Activation, Flatten, Input, BatchNormalization, GlobalAveragePooling2D, Add, Conv2D, AveragePooling2D, 

    LeakyReLU, Concatenate 

)

import efficientnet.tfkeras as efn



def get_efficientnet(model, shape):

    models_dict = {

        'b0': efn.EfficientNetB0(input_shape=shape,weights=None,include_top=False),

        'b1': efn.EfficientNetB1(input_shape=shape,weights=None,include_top=False),

        'b2': efn.EfficientNetB2(input_shape=shape,weights=None,include_top=False),

        'b3': efn.EfficientNetB3(input_shape=shape,weights=None,include_top=False),

        'b4': efn.EfficientNetB4(input_shape=shape,weights=None,include_top=False),

        'b5': efn.EfficientNetB5(input_shape=shape,weights=None,include_top=False),

        'b6': efn.EfficientNetB6(input_shape=shape,weights=None,include_top=False),

        'b7': efn.EfficientNetB7(input_shape=shape,weights=None,include_top=False)

    }

    return models_dict[model]



def build_model(shape=(512, 512, 1), model_class=None):

    inp = Input(shape=shape)

    base = get_efficientnet(model_class, shape)

    x = base(inp)

    x = GlobalAveragePooling2D()(x)

    inp2 = Input(shape=(4,))

    x2 = tf.keras.layers.GaussianNoise(0.2)(inp2)

    x = Concatenate()([x, x2]) 

    x = Dropout(0.5)(x) 

    x = Dense(1)(x)

    model = Model([inp, inp2] , x)

    

    weights = [w for w in os.listdir('../input/osic-model-weights') if model_class in w][0]

    model.load_weights('../input/osic-model-weights/' + weights)

    return model



model_classes = ['b5'] #['b0','b1','b2','b3',b4','b5','b6','b7']

models = [build_model(shape=(512, 512, 1), model_class=m) for m in model_classes]

print('Number of models: ' + str(len(models)))
from sklearn.model_selection import train_test_split 



tr_p, vl_p = train_test_split(P, 

                              shuffle=True, 

                              train_size= 0.8) 

sns.distplot(list(A.values()))
def score(fvc_true, fvc_pred, sigma):

    sigma_clip = np.maximum(sigma, 70) # changed from 70, trie 66.7 too

    delta = np.abs(fvc_true - fvc_pred)

    delta = np.minimum(delta, 1000)

    sq2 = np.sqrt(2)

    metric = (delta / sigma_clip)*sq2 + np.log(sigma_clip* sq2)

    return np.mean(metric)
subs = []

for model in models:

    metric = []

    for q in tqdm(range(1, 10)):

        m = []

        for p in vl_p:

            x = [] 

            tab = [] 



            if p in ['ID00011637202177653955184', 'ID00052637202186188008618']:

                continue



            ldir = os.listdir(f'../input/osic-pulmonary-fibrosis-progression/train/{p}/')

            for i in ldir:

                if int(i[:-4]) / len(ldir) < 0.8 and int(i[:-4]) / len(ldir) > 0.15:

                    x.append(get_img(f'../input/osic-pulmonary-fibrosis-progression/train/{p}/{i}')) 

                    tab.append(get_tab(train.loc[train.Patient == p, :])) 

            if len(x) < 1:

                continue

            tab = np.array(tab) 



            x = np.expand_dims(x, axis=-1) 

            _a = model.predict([x, tab]) 

            a = np.quantile(_a, q / 10)



            percent_true = train.Percent.values[train.Patient == p]

            fvc_true = train.FVC.values[train.Patient == p]

            weeks_true = train.Weeks.values[train.Patient == p]



            fvc = a * (weeks_true - weeks_true[0]) + fvc_true[0]

            percent = percent_true[0] - a * abs(weeks_true - weeks_true[0])

            m.append(score(fvc_true, fvc, percent))

        print(np.mean(m))

        metric.append(np.mean(m))



    q = (np.argmin(metric) + 1)/ 10



    sub = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv') 

    test = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv') 

    A_test, B_test, P_test,W, FVC= {}, {}, {},{},{} 

    STD, WEEK = {}, {} 

    for p in test.Patient.unique():

        x = [] 

        tab = [] 

        ldir = os.listdir(f'../input/osic-pulmonary-fibrosis-progression/test/{p}/')

        for i in ldir:

            if int(i[:-4]) / len(ldir) < 0.8 and int(i[:-4]) / len(ldir) > 0.15:

                x.append(get_img(f'../input/osic-pulmonary-fibrosis-progression/test/{p}/{i}')) 

                tab.append(get_tab(test.loc[test.Patient == p, :])) 

        if len(x) <= 1:

            continue

        tab = np.array(tab) 



        x = np.expand_dims(x, axis=-1) 

        _a = model.predict([x, tab]) 

        a = np.quantile(_a, q)

        A_test[p] = a

        B_test[p] = test.FVC.values[test.Patient == p] - a*test.Weeks.values[test.Patient == p]

        P_test[p] = test.Percent.values[test.Patient == p] 

        WEEK[p] = test.Weeks.values[test.Patient == p]



    for k in sub.Patient_Week.values:

        p, w = k.split('_')

        w = int(w) 



        fvc = A_test[p] * w + B_test[p]

        sub.loc[sub.Patient_Week == k, 'FVC'] = fvc

        sub.loc[sub.Patient_Week == k, 'Confidence'] = (

            P_test[p] - A_test[p] * abs(WEEK[p] - w) 

    ) 



    _sub = sub[["Patient_Week","FVC","Confidence"]].copy()

    subs.append(_sub)
N = len(subs)

sub = subs[0].copy() # ref

sub["FVC"] = 0

sub["Confidence"] = 0

for i in range(N):

    sub["FVC"] += subs[0]["FVC"] * (1/N)

    sub["Confidence"] += subs[0]["Confidence"] * (1/N)
sub.head()
sub.to_csv('submission.csv', index= False)