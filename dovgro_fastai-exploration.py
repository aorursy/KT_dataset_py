%load_ext autoreload
%autoreload 2
# Patch dataset.py until my pull request https://github.com/fastai/fastai/pull/777
# has been incorporated.
import fastai
import fastai.dataset
import pydicom

def isdicom(fn):
    '''True if the fn points to a DICOM image'''
    if fn.endswith('.dcm'):
        return True
    # Dicom signature from the dicom spec.
    with open(fn) as fh:
        fh.seek(0x80)
        return fh.read(4)=='DICM'


def open_image(fn):
    """ Opens an image using OpenCV given the file path.

    Arguments:
        fn: the file path of the image

    Returns:
        The image in RGB format as numpy array of floats normalized to range between 0.0 - 1.0
    """
    flags = cv2.IMREAD_UNCHANGED+cv2.IMREAD_ANYDEPTH+cv2.IMREAD_ANYCOLOR
    if not os.path.exists(fn) and not str(fn).startswith("http"):
        raise OSError('No such file or directory: {}'.format(fn))
    elif os.path.isdir(fn) and not str(fn).startswith("http"):
        raise OSError('Is a directory: {}'.format(fn))
    elif isdicom(fn):
        slice = pydicom.read_file(fn)
        if slice.PhotometricInterpretation.startswith('MONOCHROME'):
            # Make a fake RGB image
            im = np.stack([slice.pixel_array]*3,-1)
            return im / ((1 << slice.BitsStored)-1)
        else:
            # No support for RGB yet, as it involves various color spaces.
            # It shouldn't be too difficult to add though, if needed.
            raise OSError('Unsupported DICOM image with PhotometricInterpretation=={}'.format(slice.PhotometricInterpretation))
    else:
        #res = np.array(Image.open(fn), dtype=np.float32)/255
        #if len(res.shape)==2: res = np.repeat(res[...,None],3,2)
        #return res
        try:
            if str(fn).startswith("http"):
                req = urllib.urlopen(str(fn))
                image = np.asarray(bytearray(req.read()), dtype="uint8")
                im = cv2.imdecode(image, flags).astype(np.float32)/255
            else:
                im = cv2.imread(str(fn), flags).astype(np.float32)/255
            if im is None: raise OSError(f'File not recognized by opencv: {fn}')
            return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        except Exception as e:
            raise OSError('Error handling image at: {}'.format(fn)) from e

fastai.dataset.isdicom=isdicom
fastai.dataset.pydicom=pydicom
fastai.dataset.open_image=open_image

from fastai.conv_learner import *
# Rewrite the train csv file to contain only two columns as expected by fastai
label_csv = '../input/rsna-pneumonia-detection-challenge/stage_1_train_labels.csv'
df = pd.read_csv(label_csv)
df[['patientId','Target']].to_csv('train_target_labels.csv',index=False)
label_csv = 'train_target_labels.csv'
df = pd.read_csv(label_csv)
df.head()
PATH = '../input/rsna-pneumonia-detection-challenge'
TMP_PATH = "/tmp/tmp"
MODEL_PATH = "/tmp/model/"
sz = 224
arch = resnet34
bs = 64

n = len(df)
val_idxs = get_cv_idxs(n) # random 20% data for validation set

aug_tfms=transforms_side_on # Use None for faster testing
tfms = tfms_from_model(arch, sz, aug_tfms=aug_tfms, max_zoom=1.1)
data = ImageClassifierData.from_csv(PATH, 'stage_1_train_images', label_csv, test_name='stage_1_test_images', 
                                    val_idxs=val_idxs, 
                                    suffix='.dcm', 
                                    tfms=tfms,  # just for fast testing
                                    bs=bs)
learn = ConvLearner.pretrained(arch, data, tmp_name=TMP_PATH, models_name=MODEL_PATH)
learn.fit(0.02,3)
