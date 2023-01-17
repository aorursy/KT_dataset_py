import numpy as np

import pandas as pd



# path management

from pathlib import Path



# DICOM file handling

import pydicom



# image processing

import cv2

from skimage import measure 

from skimage.morphology import disk, opening, closing



# progress bars

from tqdm import tqdm



# zip files

import zipfile
basepath = Path('../input/rsna-str-pulmonary-embolism-detection/')

for p in basepath.iterdir():

    print(str(p))
train = pd.read_csv(basepath / 'train.csv')

train['dcmpath'] = str(basepath) + '/train' + '/' + train.StudyInstanceUID + '/' + train.SeriesInstanceUID
def open_files(dcmpath):

    '''loads all scans in the given path and orders them'''

    filelist = np.array(list(Path(dcmpath).iterdir()))

    scans = np.array([pydicom.dcmread(str(file)) for file in filelist])

    sortkey = np.argsort([float(x.ImagePositionPatient[2]) for x in scans])

    return scans[sortkey], filelist[sortkey]





def transform_to_hu(scans):

    '''transform all scans from raw data into Hounsfield units'''

    # stack scans

    imglist = []

    for file in scans:

        try:

            tmp = file.pixel_array

        except:

            tmp = np.zeros((512,512))

        imglist.append(tmp)

    images = np.stack(imglist)

    images = images.astype(np.int16)

    

    # threshold between air (0) and default mask value (-2000) using -1000 [raw values]

    images[images <= -1000] = 0

    

    # convert to HU

    for nnn in range(len(scans)):        

        intercept = scans[nnn].RescaleIntercept

        slope = scans[nnn].RescaleSlope        

        if slope != 1:

            images[nnn] = slope * images[nnn].astype(np.float64)

            images[nnn] = images[nnn].astype(np.int16)            

        images[nnn] += np.int16(intercept)    

    return np.array(images, dtype=np.int16)





def segment_lung_mask(scans):

    '''mask image and retain only lung sections'''

    segmented = np.zeros(scans.shape)

    for nnn in range(scans.shape[0]):



        # segment into water-like (2) and air-like (1) parts

        slice_binary = np.array((scans[nnn]>-320), dtype=np.int8) + 1

        slice_label = measure.label(slice_binary)

        

        bad_labels = np.unique([

            slice_label[0,:],

            slice_label[-1,:],

            slice_label[:,0],

            slice_label[:,-1]

        ])

        for bbb in bad_labels:

            slice_binary[slice_label == bbb] = 2



        # invert, air-like is now 1

        slice_binary -= 1

        slice_binary = 1 - slice_binary

        

        segmented[nnn] = slice_binary.copy() * scans[nnn]

    return segmented





def resize_scans(scans, NSCANS, NPX):

    '''resize collections of scans to a common size'''

    resized_scans = np.zeros((NSCANS, NPX, NPX))



    split = np.linspace(0, scans.shape[0], num=NSCANS+1).astype(int)

    for sss in range(NSCANS):

        scan_selection = np.mean(scans[split[sss]:split[sss+1]], axis=0)

        resized_scans[sss] = cv2.resize(scan_selection, (NPX, NPX))

    return resized_scans





def load_scans(dcmpath, NSCANS, NPX):

    '''load all files of a scan through the full pipeline'''

    scans, filelist = open_files(dcmpath)

    hu_scans = transform_to_hu(scans)

    segmented_scans = segment_lung_mask(hu_scans)

    resized_scans = resize_scans(segmented_scans, NSCANS, NPX)

    return resized_scans, filelist





def preproc_scans(dcmpathlist, NSCANS, NPX, outdir):

    '''preprocess list of scans through the full pipeline and save result in zipped output file'''

    ziparchive = zipfile.ZipFile(Path(outdir), 'w', zipfile.ZIP_DEFLATED)

    for dcmpath in tqdm(dcmpathlist):

        # load and preprocess

        scans, filelist = load_scans(dcmpath, NSCANS, NPX)

        # get identifiers

        series = Path(dcmpath).name

        study = Path(dcmpath).parent.name

        # save processed data

        filename = (study + '_' + series + '_data.npy')

        np.save(filename, scans)

        ziparchive.write(filename)

        Path(filename).unlink()

        # save ordered filelist

        filename = (study + '_' + series + '_list.npy')

        np.save(filename, filelist)

        ziparchive.write(filename)

        Path(filename).unlink()

    ziparchive.close()
scanlist = np.unique(train.dcmpath.values)

preproc_scans(scanlist[:5], 20, 128, 'proc_20_128_train.zip')
import matplotlib.pyplot as plt



sample_scans, filelist = load_scans(scanlist[150], 20, 128)



fig, ax = plt.subplots(5, 4, figsize=(20,20))

ax = ax.flatten()

for m in range(20):

    ax[m].imshow(sample_scans[m], cmap='Blues_r')