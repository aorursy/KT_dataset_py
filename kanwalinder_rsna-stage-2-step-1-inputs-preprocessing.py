import os, random
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import pydicom
import glob
import h5py
# flags
AT_KAGGLE=True
QUICK_PROCESS=True
QUICK_PROCESS_SIZE=16 # used if QUICK_PROCESS is True
PROCESS="All" #["All", "Images", "Yolov3Labels","MetaData"]
# global variables (stage 2 variables do not have a stage prefix)
TRAIN_DIR="../input/rsna-pneumonia-detection-challenge/stage_2_train_images"
STAGE1_DETAILED_CLASSES_CSV_FILE="../input/rsna-stage1-archived-inputs/stage_1_detailed_class_info.csv"
DETAILED_CLASSES_CSV_FILE="../input/rsna-pneumonia-detection-challenge/stage_2_detailed_class_info.csv"
DETAILED_CLASSES_CSV_COLUMN_NAMES=['patientId' , 'class']
# dictionary to map string classes to numerical
CLASSES_DICT={'Normal': 0, 'Lung Opacity' : 1, 'No Lung Opacity / Not Normal' : 2}

STAGE1_TRAIN_LABELS_CSV_FILE="../input/rsna-stage1-archived-inputs/stage_1_train_labels.csv"
TRAIN_LABELS_CSV_FILE="../input/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv"
# pedantic nit: we are changing 'Target' to 'label' on the way in
TRAIN_LABELS_CSV_COLUMN_NAMES=['patientId', 'x1', 'y1', 'bw', 'bh', 'label']

# saved test ids from stage1
STAGE1_TEST_IDS_FILE="../input/rsna-stage1-archived-inputs/stage1_test_ids.npy"

TEST_DIR="../input/rsna-pneumonia-detection-challenge/stage_2_test_images"
# list of test images
TEST_LIST=sorted(os.listdir(TEST_DIR))

SAVED_KEYS_FILE="rsna-stage1-and-stage2-keys.npz"
SAVED_PATIENTDICT_FILE="rsna-patientdict.h5"

DICOM_IMAGE_SIZE=1024
# setup output directories
# directory where we will put processed inputs
if not AT_KAGGLE:
    processedtraininputsdirectory="../input/rsna-stage2-processed-train-inputs"
    processedtestinputsdirectory="../input/rsna-stage2-processed-test-inputs"
    processedmetadatadirectory="../input/rsna-stage2-processed-metadata-inputs"
else: # we are at Kaggle (have to write to the current directory to keep the Commit engine happy)
    processedtraininputsdirectory="./rsna-stage2-processed-train-inputs"
    processedtestinputsdirectory="./rsna-stage2-processed-test-inputs"
    processedmetadatadirectory="./rsna-stage2-processed-metadata-inputs"
    
# create directories (one-time)
os.makedirs(processedtraininputsdirectory, exist_ok=False)
os.makedirs(processedtestinputsdirectory, exist_ok=False)
os.makedirs(processedmetadatadirectory, exist_ok=False)

print ("Preprocessing training inputs into directory: {}".format(processedtraininputsdirectory))
print ("Preprocessing test inputs into directory: {}".format(processedtestinputsdirectory))
print ("Preprocessing meta data into directory: {}".format(processedmetadatadirectory))
# read STAGE1_DETAILED_CLASSES_CSV_FILE into a pandas dataframe
classesdf = pd.read_csv(STAGE1_DETAILED_CLASSES_CSV_FILE,
                        names=DETAILED_CLASSES_CSV_COLUMN_NAMES,
                        # skip the header line
                        header=0,
                        # index the dataframe on patientId
                        index_col='patientId')
#print (classesdf.shape)
#print (classesdf.head(n=10))

# remove duplicates
classesdf=classesdf.groupby(['patientId'])['class'].first()
# make classes numerical based on CLASSES_DICT
classesdf=pd.DataFrame(classesdf.replace(to_replace=CLASSES_DICT), columns=['class'])
print ("Stage 1:: {} lines read from {}".format(len(classesdf), STAGE1_DETAILED_CLASSES_CSV_FILE))
# read list of stage1 test images
stage1testkeys=sorted(list(np.load(STAGE1_TEST_IDS_FILE)))
# capture stage1 patientids for different classes
stage1allkeys=classesdf.index.tolist()
stage1lungopacitykeys=classesdf.index[classesdf['class']==1].tolist()
stage1normalkeys=classesdf.index[classesdf['class']==0].tolist()
stage1otherabnormalkeys=classesdf.index[classesdf['class']==2].tolist()
print ("################STAGE 1 SUMMARY################")
print ("Total Training Samples: {}".format(len(stage1allkeys)))
print (">>Lung Opacity Samples: {}".format(len(stage1lungopacitykeys)))
print (">>Normal Samples: {}".format(len(stage1normalkeys)))
print (">>Not Normal / No Lung Opacity Samples: {}".format(len(stage1otherabnormalkeys)))
print ("Test Samples: {}".format(len(stage1testkeys)))
print ("##############################################")
# read stage2 DETAILED_CLASSES_CSV_FILE into a pandas dataframe
classesdf = pd.read_csv(DETAILED_CLASSES_CSV_FILE,
                        names=DETAILED_CLASSES_CSV_COLUMN_NAMES,
                        # skip the header line
                        header=0,
                        # index the dataframe on patientId
                        index_col='patientId')
#print (classesdf.shape)
#print (classesdf.head(n=10))

# remove duplicates
classesdf=classesdf.groupby(['patientId'])['class'].first()
# make classes numerical based on CLASSES_DICT
classesdf=pd.DataFrame(classesdf.replace(to_replace=CLASSES_DICT), columns=['class'])
print ("Stage 2:: {} lines read from {}".format(len(classesdf), DETAILED_CLASSES_CSV_FILE))
# capture stage2 test keys
testkeys=[]
for filename in TEST_LIST:
    key=filename.split(".")[0]
    testkeys.append(key)
# capture stage2 patientids for different classes
allkeys=classesdf.index.tolist()
lungopacitykeys=classesdf.index[classesdf['class']==1].tolist()
normalkeys=classesdf.index[classesdf['class']==0].tolist()
otherabnormalkeys=classesdf.index[classesdf['class']==2].tolist()
print ("################STAGE 2 SUMMARY################")
print ("Total Training Samples: {}".format(len(allkeys)))
print (">>Lung Opacity Samples: {}".format(len(lungopacitykeys)))
print (">>Normal Samples: {}".format(len(normalkeys)))
print (">>Not Normal / No Lung Opacity Samples: {}".format(len(otherabnormalkeys)))
print ("Test Samples: {}".format(len(testkeys)))
print ("##############################################")
print ("{} test samples from Stage 1 were distributed into Stage 2 as:".format(len(stage1testkeys)))
print (">>{} additional Lung Opacity Samples".format(len(lungopacitykeys)-len(stage1lungopacitykeys)))
print (">>{} additional Normal Samples".format(len(normalkeys)-len(stage1normalkeys)))
print (">>{} additional Not Normal / No Lung Opacity Samples".format(len(otherabnormalkeys)-len(stage1otherabnormalkeys)))
# check stage2 vs stage1 keys
assert sorted(allkeys)==sorted(stage1normalkeys+stage1lungopacitykeys+stage1otherabnormalkeys+stage1testkeys), "Keys Mismatch"
# read TRAIN_LABELS_CSV_FILE into a pandas dataframe
labelsbboxdf = pd.read_csv(TRAIN_LABELS_CSV_FILE,
                           names=TRAIN_LABELS_CSV_COLUMN_NAMES,
                           # skip the header line
                           header=0,
                           # index the dataframe on patientId
                           index_col='patientId')

labelsbboxdf.head(10)
# compute and store bounding box centers
bx=labelsbboxdf['x1']+labelsbboxdf['bw']/2
by=labelsbboxdf['y1']+labelsbboxdf['bh']/2
labelsbboxdf=labelsbboxdf.assign(bx=bx, by=by)
labelsbboxdf.head(10)
# drop labels and rearrange dataframe so we have bounding boxes in rsna format,
# dropping all rows other than lungopacity ones
rsnabboxesdf=labelsbboxdf[['x1', 'y1', 'bw', 'bh']].dropna()
rsnabboxesdf.head(10)
# drop labels and top left coordinates and rearrange dataframe in yolov3 format,
# dropping all rows other than lungopacity ones
yolov3bboxesdf=labelsbboxdf[['bx', 'by', 'bw', 'bh']].dropna()
yolov3bboxesdf.head(10)
# yolov3 requires bounding box dimensions to be between 0 and 1
# normalize to DICOM_IMAGE_SIZE
yolov3bboxesdf=yolov3bboxesdf/DICOM_IMAGE_SIZE
yolov3bboxesdf.head(10)
# save a copy of keys we are going to munge up when running quick checks
if QUICK_PROCESS==True:
    savedallkeys=allkeys
    savedtestkeys=testkeys
    savedlungopacitykeys=lungopacitykeys
# setup a quick test to make sure everything is working before heading off to GCP
if QUICK_PROCESS == True:
    allkeys=random.sample(allkeys, QUICK_PROCESS_SIZE)
    testkeys=random.sample(testkeys, QUICK_PROCESS_SIZE)
    lungopacitykeys=random.sample(lungopacitykeys, QUICK_PROCESS_SIZE)
    print ("Quick check by preprocessing {} samples".format(QUICK_PROCESS_SIZE))
def loadDicomImage (directory, patientid, mode="metadata"):
    imagergb=np.zeros([DICOM_IMAGE_SIZE, DICOM_IMAGE_SIZE, 3])
    attributes=[]
    filename="{}.dcm".format(patientid)
    
    if mode=="metadata":
        # load patient meta-data only from file
        patientdata = pydicom.dcmread(os.path.join(directory, filename), stop_before_pixels=True)
    elif mode=="image":
        # load patient meta-data and image from file
        patientdata = pydicom.dcmread(os.path.join(directory, filename))
        imagegray=patientdata.pixel_array
        # convert grayscale to rgb
        imagegray=imagegray/imagegray.max()
        imagegray = (255*imagegray).clip(0, 255).astype(np.uint8)
        imagergb=np.stack([imagegray]*3, -1)
    # make sure there isn't a mismatch
    assert patientid==patientdata.PatientID, "PatientId Mismatch"
    # grab attributes
    attributes.append(patientdata.PatientSex)
    attributes.append(patientdata.PatientAge)
    attributes.append(patientdata.ViewPosition)
    
    #print (imagergb)
    return attributes, imagergb
# save jpg images for train samples in original size
if PROCESS == "All"  or PROCESS=="Images":
    for patientid in tqdm(allkeys):
        imagefilename="{}.jpg".format(patientid)
        imagepathname=os.path.join(processedtraininputsdirectory, imagefilename)
        #print (imagepathname)
        _, imagergb = loadDicomImage (TRAIN_DIR, patientid, mode="image")
        image=Image.fromarray(imagergb)
        assert image.size==(DICOM_IMAGE_SIZE, DICOM_IMAGE_SIZE), "Input Image Size Mismatch"
        image.save(imagepathname)
        
    # make sure all images made it through correctly
    processedtrainkeys=[]
    for filename in glob.glob(processedtraininputsdirectory+'/*.jpg'):
        key=os.path.split(filename)[1].split(".")[0]
        processedtrainkeys.append(key)
    assert sorted(processedtrainkeys)==sorted(allkeys), "Train Samples Missed"
    print ("Preprocessed {} train images".format(len(processedtrainkeys)))
# save jpg images for test inputs in original size
if PROCESS == "All"  or PROCESS=="Images":
    for patientid in tqdm(testkeys):
        imagefilename="{}.jpg".format(patientid)
        imagepathname=os.path.join(processedtestinputsdirectory, imagefilename)
        #print (imagepathname)
        _, imagergb = loadDicomImage (TEST_DIR, patientid, mode="image")
        image=Image.fromarray(imagergb)
        assert image.size==(DICOM_IMAGE_SIZE, DICOM_IMAGE_SIZE), "Input Image Size Mismatch"
        image.save(imagepathname)
        
    # make sure all images made it through correctly
    processedtestkeys=[]
    for filename in glob.glob(processedtestinputsdirectory+'/*.jpg'):
        key=os.path.split(filename)[1].split(".")[0]
        processedtestkeys.append(key)
    assert sorted(processedtestkeys)==sorted(testkeys), "Test Samples Missed"
    print ("Preprocessed {} test images".format(len(processedtestkeys)))
# get yolov3 bounding boxes by patientid
def getyolov3BoundingBoxes (bboxesdf, key):
    bboxarray=bboxesdf.loc[key][['bx', 'by', 'bw', 'bh']].values
    # hack to detect and fix single bounding box case which
    # comes in with shape (4,)
    #print (bboxarray.shape)
    bboxarray=np.expand_dims(bboxarray, -1)
    if bboxarray.shape[1]==1:
        bboxarray=bboxarray.reshape(1, bboxarray.shape[0])
    else:
        bboxarray=np.squeeze(bboxarray, axis=-1)
    #print (bboxarray.shape)
    return bboxarray
# write bounding box information for lungopacity cases in yolov3 format
if PROCESS == "All"  or PROCESS=="Yolov3labels":
    for patientid in tqdm(lungopacitykeys):
        bboxarray=getyolov3BoundingBoxes(yolov3bboxesdf, patientid)
        assert len(bboxarray) > 0, "Missing Bounding Boxes for {}".format(patientid)
        bboxfilename="{}.txt".format(patientid)
        bboxpathname=os.path.join(processedtraininputsdirectory, bboxfilename)
        #print (bboxpathname)
        file=open(bboxpathname,'w')
        for i in range(len(bboxarray)):
            bx, by, bw, bh = bboxarray[i]
            #print(bx, by, bw, bh)
            boxrecord="0 {} {} {} {}\n".format(bx, by, bw, bh)
            file.write(boxrecord)
        file.close()
        
    # make sure all boxes made it through correctly
    processedlungopacitykeys=[]
    for filename in glob.glob(processedtraininputsdirectory+'/*.txt'):
        key=os.path.split(filename)[1].split(".")[0]
        processedlungopacitykeys.append(key)
    assert sorted(processedlungopacitykeys)==sorted(lungopacitykeys), "Lung Opacity Samples Missed"
    print ("Saved bounding boxes for {} Lung Opacity cases".format(len(processedlungopacitykeys)))
!ls -al "./rsna-stage2-processed-train-inputs"
!ls -al "./rsna-stage2-processed-test-inputs"
# we are going to run the RSNA Metadata for all samples so we don't clutter up the code
# reset the keys we munged up for quick check
if QUICK_PROCESS==True:
    allkeys=savedallkeys
    testkeys=savedtestkeys
    lungopacitykeys=savedlungopacitykeys
# get rsna bounding boxes by patientid
def getrsnaBoundingBoxes (bboxesdf, key):
    bboxarray=bboxesdf.loc[key][['x1', 'y1', 'bw', 'bh']].values
    # hack to detect and fix single bounding box case which
    # comes in with shape (4,)
    #print (bboxarray.shape)
    bboxarray=np.expand_dims(bboxarray, -1)
    if bboxarray.shape[1]==1:
        bboxarray=bboxarray.reshape(1, bboxarray.shape[0])
    else:
        bboxarray=np.squeeze(bboxarray, axis=-1)
    #print (bboxarray.shape)
    return bboxarray
# save RSNA metadata (will take some time, you can kill the kernel if you have seen enough)
if PROCESS == "All"  or PROCESS=="MetaData":
    rsnapatientdict=pd.DataFrame()
    oneboundingboxkeys=[]
    twoboundingboxkeys=[]
    threeboundingboxkeys=[]
    fourboundingboxkeys=[]
    
    for patientid in tqdm(allkeys):
        rsnaattributes, _ = loadDicomImage (TRAIN_DIR, patientid, mode="metadata")
        bboxlist=[]
        if patientid in lungopacitykeys:
            bboxarray=getrsnaBoundingBoxes(rsnabboxesdf, patientid)
            assert len(bboxarray) > 0, "Missing Bounding Boxes for {}".format(patientid)
            bboxlist=list(bboxarray)
            
            if len(bboxarray) == 1:
                oneboundingboxkeys.append(patientid)
            elif len(bboxarray) == 2:
                twoboundingboxkeys.append(patientid)
            elif len(bboxarray) == 3:
                threeboundingboxkeys.append(patientid)
            elif len(bboxarray) == 4:
                fourboundingboxkeys.append(patientid)
        
        patientrecord=pd.DataFrame({
            'patientId': [patientid],
            'patientSex': [rsnaattributes[0]],
            'patientAge': [rsnaattributes[1]],
            'patientViewPosition': [rsnaattributes[2]],
            'BoundingBoxes': [bboxlist]})
        rsnapatientdict=rsnapatientdict.append(patientrecord)
        
    print ("################STAGE 2 BOUNDING BOX SUMMARY################")
    print ("Total Lung Opacity Samples: {}".format(len(lungopacitykeys)))
    print (">>Samples with 1 Bounding Box: {}".format(len(oneboundingboxkeys)))
    print (">>Samples with 2 Bounding Boxes: {}".format(len(twoboundingboxkeys)))
    print (">>Samples with 3 Bounding Boxes: {}".format(len(threeboundingboxkeys)))
    print (">>Samples with 4 Bounding Boxes: {}".format(len(fourboundingboxkeys)))
    print ("#############################################################")
        
    # save all stage1 and stage2 keys
    np.savez(os.path.join(processedmetadatadirectory, SAVED_KEYS_FILE),
             np.array(allkeys),
             np.array(normalkeys),
             np.array(lungopacitykeys),
             np.array(otherabnormalkeys),
             np.array(testkeys),
             np.array(oneboundingboxkeys),
             np.array(twoboundingboxkeys),
             np.array(threeboundingboxkeys),
             np.array(fourboundingboxkeys),
             np.array(stage1allkeys),
             np.array(stage1normalkeys),
             np.array(stage1lungopacitykeys),
             np.array(stage1otherabnormalkeys),
             np.array(stage1testkeys))
    
    # save RSNA patient dictionary (work in progress, hdf5 is creaky about strings, may not be working )
    rsnapatientdict.to_hdf(os.path.join(processedmetadatadirectory, SAVED_PATIENTDICT_FILE),
                           key='rsnapatientdict',
                           mode='w')
    print (">>>Saved RSNA patient dictionary to to {}".format(os.path.join(processedmetadatadirectory, SAVED_PATIENTDICT_FILE)))
        
    # make sure we can get everything back
    npzfile=np.load(os.path.join(processedmetadatadirectory, SAVED_KEYS_FILE))

    assert allkeys==sorted(list(npzfile['arr_0'])), "All Keys Mismatch"
    assert normalkeys==sorted(list(npzfile['arr_1'])), "Normal Keys Mismatch"
    assert lungopacitykeys==sorted(list(npzfile['arr_2'])), "Lung Opacity Keys Mismatch"
    assert otherabnormalkeys==sorted(list(npzfile['arr_3'])), "Not Normal / No Lung Opacity Keys Mismatch"
    assert testkeys==sorted(list(npzfile['arr_4'])), "Test Keys Mismatch"

    assert oneboundingboxkeys==sorted(list(npzfile['arr_5'])), "One Bounding Box Keys Mismatch"
    assert twoboundingboxkeys==sorted(list(npzfile['arr_6'])), "Two Bounding Box Keys Mismatch"
    assert threeboundingboxkeys==sorted(list(npzfile['arr_7'])), "Three Bounding Box Keys Mismatch"
    assert fourboundingboxkeys==sorted(list(npzfile['arr_8'])), "Four Bounding Box Keys Mismatch"

    assert stage1allkeys==sorted(list(npzfile['arr_9'])), "Stage1 All Keys Mismatch"
    assert stage1normalkeys==sorted(list(npzfile['arr_10'])), "Stage1 Normal Keys Mismatch"
    assert stage1lungopacitykeys==sorted(list(npzfile['arr_11'])), "Stage1 Lung Opacity Keys Mismatch"
    assert stage1otherabnormalkeys==sorted(list(npzfile['arr_12'])), "Stage1 Not Normal / No Lung Opacity Keys Mismatch"
    assert stage1testkeys==sorted(list(npzfile['arr_13'])), "Stage1 Test Keys Mismatch"
        
    