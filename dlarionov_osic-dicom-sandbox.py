!conda install -c conda-forge gdcm -y

import gdcm # required for ID00011637202177653955184
import os

import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import matplotlib.animation as animation

from IPython.display import HTML



import pydicom

import cv2

from skimage import measure, morphology, segmentation

import scipy.ndimage as ndimage



from multiprocessing import Pool

from tqdm.notebook import tqdm



DCOM_DIR = '../input/osic-pulmonary-fibrosis-progression/train'
train = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')

train = train.drop_duplicates(keep=False, subset=['Patient','Weeks'])

train
# copypaste https://www.kaggle.com/carlossouza/osic-autoencoder-training



def seperate_lungs(image, min_hu, iterations):

    h, w = image.shape[0], image.shape[1]



    marker_internal, marker_external, marker_watershed = generate_markers(image)



    # Sobel-Gradient

    sobel_filtered_dx = ndimage.sobel(image, 1)

    sobel_filtered_dy = ndimage.sobel(image, 0)

    sobel_gradient = np.hypot(sobel_filtered_dx, sobel_filtered_dy)

    sobel_gradient *= 255.0 / np.max(sobel_gradient)



    watershed = morphology.watershed(sobel_gradient, marker_watershed)



    outline = ndimage.morphological_gradient(watershed, size=(3,3))

    outline = outline.astype(bool)



    # Structuring element used for the filter

    blackhat_struct = [[0, 0, 1, 1, 1, 0, 0],

                       [0, 1, 1, 1, 1, 1, 0],

                       [1, 1, 1, 1, 1, 1, 1],

                       [1, 1, 1, 1, 1, 1, 1],

                       [1, 1, 1, 1, 1, 1, 1],

                       [0, 1, 1, 1, 1, 1, 0],

                       [0, 0, 1, 1, 1, 0, 0]]



    blackhat_struct = ndimage.iterate_structure(blackhat_struct, iterations)



    # Perform Black Top-hat filter

    outline += ndimage.black_tophat(outline, structure=blackhat_struct)



    lungfilter = np.bitwise_or(marker_internal, outline)

    lungfilter = ndimage.morphology.binary_closing(lungfilter, structure=np.ones((5,5)), iterations=3)



    segmented = np.where(lungfilter == 1, image, min_hu * np.ones((h, w)))



    return segmented  #, lungfilter, outline, watershed, sobel_gradient



def generate_markers(image, threshold=-400):

    h, w = image.shape[0], image.shape[1]



    marker_internal = image < threshold

    marker_internal = segmentation.clear_border(marker_internal)

    marker_internal_labels = measure.label(marker_internal)



    areas = [r.area for r in measure.regionprops(marker_internal_labels)]

    areas.sort()



    if len(areas) > 2:

        for region in measure.regionprops(marker_internal_labels):

            if region.area < areas[-2]:

                for coordinates in region.coords:

                    marker_internal_labels[coordinates[0], coordinates[1]] = 0



    marker_internal = marker_internal_labels > 0



    # Creation of the External Marker

    external_a = ndimage.binary_dilation(marker_internal, iterations=10)

    external_b = ndimage.binary_dilation(marker_internal, iterations=55)

    marker_external = external_b ^ external_a



    # Creation of the Watershed Marker

    marker_watershed = np.zeros((h, w), dtype=np.int)

    marker_watershed += marker_internal * 255

    marker_watershed += marker_external * 128



    return marker_internal, marker_external, marker_watershed
def _parts(pid:int, dcom_dir=DCOM_DIR)->list:

    return sorted([int(i.split('.')[0]) for i in os.listdir(os.path.join(dcom_dir, pid))])



def _crop(s, size=512):

    if (s.shape[0]<=size):

        return s

    s_cropped = s[~np.all(s == 0, axis=1)]

    s_cropped = s_cropped[:, ~np.all(s_cropped == 0, axis=0)]

    return s_cropped



def _resize(s, size=512):

    if (s.shape[0]<=size):

        return s

    s_resized = cv2.resize(s, (0,0), fx=size/s.shape[0], fy=size/s.shape[1], interpolation=cv2.INTER_AREA)        

    return s_resized



def _clip(s, lo=-1000, hi=500):

    s[s<lo]=lo

    s[s>hi]=hi

    return s



def _hu(s, slope:float, intercept:int):     

    s_hu = (s * slope + intercept).astype(np.int16)

    return s_hu



def _norm(s, lo=-1000, hi=500):

    s_normed= s.astype(np.float)

    s_normed = (s_normed-lo)*255.0/(hi-lo)

    return s_normed.astype(np.int16)



def _meta(dcom):

    return {

        #'Modality': dcom.Modality, # const

        'Manufacturer' : dcom.Manufacturer,

        'ManufacturerModelName' : dcom.ManufacturerModelName,

        #'BodyPartExamined': dcom.BodyPartExamined, # const

        'SliceThickness' : dcom.SliceThickness,

        'KVP' : dcom.KVP,

        'TableHeight' : dcom.TableHeight,

        #'RotationDirection' : dcom.RotationDirection, # const       

        'ConvolutionKernel' : dcom.ConvolutionKernel,

        'PatientPosition' : dcom.PatientPosition,             

        #'PhotometricInterpretation': dcom.PhotometricInterpretation, # const

        #'SamplesPerPixel': dcom.SamplesPerPixel, # const

        #'BitsAllocated': dcom.BitsAllocated, # const

        'BitsStored': dcom.BitsStored,

        'HighBit': dcom.HighBit,

        'PixelRepresentation': dcom.PixelRepresentation,

        'PixelSpacing0': dcom.PixelSpacing[0],

        'PixelSpacing1': dcom.PixelSpacing[1],

        'WindowCenter': dcom.WindowCenter,

        'WindowWidth': dcom.WindowWidth,

        'RescaleIntercept': dcom.RescaleIntercept,

        #'RescaleSlope': dcom.RescaleSlope # const

    }
%%time



def dicom_cube(pid):

    arr = []

    for part in _parts(pid)[200:250]:

        dicom = pydicom.dcmread(os.path.join(DCOM_DIR, pid, f'{part}.dcm'))

        img = dicom.pixel_array        

        meta = _meta(dicom)

        img = _crop(img, size=512)

        img = _hu(img, slope=1., intercept=meta['RescaleIntercept'])

        img = _resize(img, size=512)

        img = _clip(img, lo=-1000, hi=500)

        img = seperate_lungs(img, min_hu=-1000, iterations=1) # time consuming operation

        img = _norm(img, lo=-1000, hi=500)

        arr.append(img)

    return np.array(arr)



cube = dicom_cube('ID00067637202189903532242')



fig = plt.figure(figsize=(8,8))



ims = []

for i in cube:

    im = plt.imshow(i, animated=True, cmap='gray')

    plt.axis('off')

    ims.append([im])



ani = animation.ArtistAnimation(fig, ims, interval=50)

plt.close()



HTML(ani.to_jshtml())
OUT_DIR = '/tmp/osic'



trash = [

    os.path.join(DCOM_DIR, 'ID00105637202208831864134', '1.dcm'), # black

    os.path.join(DCOM_DIR, 'ID00052637202186188008618', '4.dcm'), # corrupted

]



def process_one(pid):

    os.makedirs(os.path.join(OUT_DIR, pid), exist_ok=True)    

    metas = []

    parts = _parts(pid)

    for part in parts:

        path = os.path.join(DCOM_DIR, pid, f'{part}.dcm')

        if path in trash:

            continue

        

        dicom = pydicom.dcmread(path)

        img = dicom.pixel_array        

        meta = _meta(dicom)

        img = _crop(img, size=512)

        img = _hu(img, slope=1., intercept=meta['RescaleIntercept'])

        img = _resize(img, size=512)

        img = _clip(img, lo=-1000, hi=500)

        img = seperate_lungs(img, min_hu=-1000, iterations=1)

        img = _norm(img, lo=-1000, hi=500)     

        

        meta['Patient'] = pid

        meta['Part'] = part

        meta['Min'] = np.min(img)

        meta['Max'] = np.max(img)

        meta['Mean'] = np.mean(img)

        meta['Std'] = np.std(img)

        metas.append(meta)

        

        res = cv2.imwrite(os.path.join(OUT_DIR, pid, f'{part}.png'), img)

        

    return metas



#metas = process_one('ID00067637202189903532242')

#metas[0]
batch = train.Patient.unique()#[:5]

with Pool(processes=4) as pool:

    res = list(

        tqdm(pool.imap(process_one, list(batch)), total = len(batch))

    )
stack = []

for i in range(len(res)):

    for j in range(len(res[i])):

        stack.append(res[i][j])



df = pd.DataFrame(stack)

df.loc[df.Manufacturer == 'GE MEDICAL SYSTEMS', 'Manufacturer'] = 'GE'

df.loc[df.Manufacturer == 'Hitachi Medical Corporation', 'Manufacturer'] = 'Hitachi'

df.ManufacturerModelName.fillna('TOSHIBA', inplace=True)

df
!mkdir -p /root/.kaggle/

!cp ../input/mykaggleapi/kaggle.json /root/.kaggle/

!chmod 600 /root/.kaggle/kaggle.json
DS_DIR = '/tmp/dataset'

!mkdir -p {DS_DIR}

!tar czf {DS_DIR}/osic.tar.gz -C {OUT_DIR} .

df.to_csv(os.path.join(DS_DIR, f'CT.csv'), index=False)

!kaggle datasets init -p {DS_DIR}
import json



with open(f'{DS_DIR}/dataset-metadata.json', 'r+') as f:

    data = json.load(f)

    data['title'] = f'osic sandbox'

    data['id'] = f'dlarionov/osic-sandbox'

    f.seek(0)

    json.dump(data, f, indent=4)

    f.truncate()



!cat {DS_DIR}/dataset-metadata.json
!kaggle datasets create -p {DS_DIR} -q -r tar

!rm -rf {OUT_DIR}

!rm -rf {DS_DIR}