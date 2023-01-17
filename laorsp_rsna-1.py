import glob, pylab, pandas as pd

import pydicom , numpy as np

from os import listdir

from os.path import isfile, join
!ls '../input'
train = pd.read_csv('../input/rsna-intracranial-hemorrhage-detection/stage_1_train.csv')
train.head()
new_table = train.copy()
train_images_dir = '../input/rsna-intracranial-hemorrhage-detection/stage_1_train_images'

#train_images = [f for f in listdir(train_images_dir) if isfile(join(train_images_dir, f))]

test_images_dir = '../input/rsna-intracranial-hemorrhage-detection/stage_1_test_images/'

#test_images = [f for f in listdir(test_images_dir) if isfile(join(test_images_dir, f))]

#print('5 Training images', train_images[:5]) 
train['Patient_ID'] = train['ID'].str.split(pat='_',n=3,expand=True)[1]

train.head()
train['Sub_type'] = train['ID'].str.split(pat='_',n=3,expand=True)[2]

train.head()
lbls = { i : "" for i in train.Patient_ID.unique()}



i=0

for name, group in train[train.Label==1].groupby("Patient_ID"):

    lbls[name] = " ".join(group.Sub_type)

    if i % 10000 == 0: print(lbls[name])

    i += 1
df = pd.DataFrame(np.array([list(lbls.keys()), list(lbls.values())]).transpose(), columns=["id", "labels"])
from fastai.vision import *
def window_image(img, window_center,window_width, intercept, slope):



    img = (img*slope +intercept)

    img_min = window_center - window_width//2

    img_max = window_center + window_width//2

    img[img<img_min] = img_min

    img[img>img_max] = img_max

    return img



def get_first_of_dicom_field_as_int(x):

    #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)

    if type(x) == pydicom.multival.MultiValue:

        return int(x[0])

    else:

        return int(x)



def get_windowing(data):

    dicom_fields = [data[('0028','1050')].value, #window center

                    data[('0028','1051')].value, #window width

                    data[('0028','1052')].value, #intercept

                    data[('0028','1053')].value] #slope

    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]
def new_open_image(path, div=True, convert_mode=None, after_open=None):

    dcm = pydicom.dcmread(str(path))

    window_center, window_width, intercept, slope = get_windowing(dcm)

    im = window_image(dcm.pixel_array, window_center, window_width, intercept, slope)

    im = np.stack((im,)*3, axis=-1)

    im -= im.min()

    im_max = im.max()

    if im_max != 0: im = im / im.max()

    x = Image(pil2tensor(im, dtype=np.float32))

    #if div: x.div_(2048)  # ??

    return x

vision.data.open_image = new_open_image
df.id = "ID_"+ df.id

df = df[df.id!="ID_6431af929"] #remove corrupted image

bs = 16

tfms = get_transforms()

data = (ImageList.from_df(path=train_images_dir,df=df,suffix='.dcm')

.split_by_rand_pct(0.2)

.label_from_df(label_delim=" ")

.transform(tfms,size=128)

.databunch(bs=bs).normalize(imagenet_stats))
data.show_batch(3)
arch = models.resnet34

learn = cnn_learner(data,arch,metrics=[accuracy_thresh])
models_path = Path("/kaggle/working/models")

if not models_path.exists(): models_path.mkdir() 

learn.model_dir = models_path

learn.lr_find()
learn.recorder.plot()
lr = slice(1e-2,1e-1)
learn.fit_one_cycle(1,lr)