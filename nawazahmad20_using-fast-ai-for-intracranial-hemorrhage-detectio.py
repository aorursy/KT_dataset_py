%reload_ext autoreload

%autoreload 2

%matplotlib inline
import pydicom

import os

import numpy

from matplotlib import pyplot, cm

from fastai.vision import *

import fastai
from os import walk

f = []

#for (dirpath, dirnames, filenames) in walk("../input/rsna-intracranial-hemorrhage-detection/stage_1_train_images"):

#    f.extend(filenames)

#    break
# function to open dcm images

def open_dcm_image(fn:PathOrStr, div:bool=True, convert_mode:str='L', cls:type=Image,

        after_open:Callable=None)->Image:

    "Return `Image` object created from image in file `fn`."

    with warnings.catch_warnings():

        warnings.simplefilter("ignore", UserWarning) # EXIF warning from TiffPlugin

        #x= PIL.Image.open(fn).convert(convert_mode)

        # code added for opening dcm images

        dicom_file = pydicom.dcmread(str(fn))

        arr = dicom_file.pixel_array.copy() 

        arr = arr * int(dicom_file.RescaleSlope) + int(dicom_file.RescaleIntercept) 

        level = 40; window = 80

        arr = np.clip(arr, level - window // 2, level + window // 2)

        x = PIL.Image.fromarray(arr).convert(convert_mode)

    if after_open: x = after_open(x)

    x = pil2tensor(x,np.float32)

    if div: x.div_(255)

    return cls(x)
fastai.vision.data.open_image = open_dcm_image
dirpath = "../input/rsna-intracranial-hemorrhage-detection/"

!ls ../input/rsna-intracranial-hemorrhage-detection
df_train = pd.read_csv('../input/rsna-intracranial-hemorrhage-detection/stage_1_train.csv')

df_train['fn'] = df_train.ID.apply(lambda x: '_'.join(x.split('_')[:2]) + '.dcm')

df_train['label'] = df_train.ID.apply(lambda x: x.split('_')[-1])

df_train.head()

# remove one corrupted file

df_train = df_train[df_train.fn != 'ID_000039fa0.dcm']
open_dcm_image( dirpath +"stage_1_train_images/" + df_train.fn.values[5] , convert_mode= 'L').show(cmap= 'gray')
print(df_train.shape)

df_train.drop_duplicates(inplace = True)

print(df_train.shape)
pivot = df_train.pivot(index='fn', columns='label', values='Label')

pivot.reset_index(inplace=True)

# chcek if there are only two types of values in any

assert pivot[pivot['any'] == 0].shape[0] + pivot[pivot['any'] == 1].shape[0] == pivot.shape[0] 

pivot['any'].value_counts()
mask = pivot['any'] == 0

pivot['None'] = ""

pivot.loc[mask, 'None'] = 'None'



label_cols = ['any', 'epidural', 'intraparenchymal', 'intraventricular','subarachnoid', 'subdural', 'None'] 

for col in label_cols:

    print(col, end= ", ")

    pivot[col] = pivot[col].replace({0:"", 1:col})

    

pivot['MultiLabel'] = pivot[label_cols].apply(lambda x: " ".join((' '.join(x)).split()), axis=1)
df_test = pd.read_csv('../input/rsna-intracranial-hemorrhage-detection/stage_1_sample_submission.csv')

df_test['fn'] = df_test.ID.apply(lambda x: '_'.join(x.split('_')[:2]) + '.dcm')

df_test['label'] = df_test.ID.apply(lambda x: x.split('_')[-1])
pivot_test = df_test.pivot(index='fn', columns='label', values='Label')

pivot_test.reset_index(inplace=True)

pivot_test['MultiLabel'] = " "
tfms = get_transforms(do_flip = False)
path = "../input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/"

np.random.seed(42)

data_train = (ImageList

.from_df(path=path,df= pivot[['fn', 'MultiLabel']])

.split_by_rand_pct()

.label_from_df(cols=1, label_delim = " ")

.transform(size=(128,128))

.databunch()

.normalize(imagenet_stats))
path = "../input/rsna-intracranial-hemorrhage-detection/stage_1_test_images/"

data_test = (ImageList

.from_df(path=path,df= pivot_test[['fn', 'MultiLabel']])

.split_by_rand_pct(valid_pct= 0)

.label_from_df(cols=1, label_delim = " ")

.transform(size=(128,128))

.databunch()

.normalize(imagenet_stats))
assert len(data_test.train_ds.y) == pivot_test.shape[0]
path = '/output/'

acc_02 = partial(accuracy_thresh, thresh=0.2)

f_score = partial(fbeta, thresh=0.2)
learn = cnn_learner( data_train, models.resnet34, path = path, metrics = [acc_02, f_score] )
learn.lr_find()

learn.recorder.plot()
lr = 3e-3

learn.freeze()

learn.fit_one_cycle(2, slice(lr))
learn.save('stage-1')

learn.load('stage-1')
data_classes = learn.data.classes
learn.data = data_test
y_predict, _ = learn.get_preds(ds_type=DatasetType.Fix)
assert len(y_predict) == pivot_test.shape[0]
#y_predict = [learn.predict(data_test.train_ds.x[i])[2].numpy() for i in range(62836)]
pivot_test['MultiLabel']  = y_predict
learn.data.classes
#data_classes = ['None'] + label_cols[:-1]
for i, col in enumerate(data_classes):

    print(col, end= ", ")

    pivot_test[col] = pivot_test['MultiLabel'].apply(lambda x: x[i].numpy())
cols_to_consider = [col for col in pivot_test.columns if not col in ['None', 'MultiLabel']]

print(cols_to_consider)

df_temp = pd.melt(pivot_test[cols_to_consider], id_vars= ['fn'])
df_temp['ID'] = df_temp['fn'].apply(lambda x: x[:-4])

df_temp['ID'] = df_temp[['ID', 'label']].apply(lambda x: "_".join(x), axis = 1)
assert len(set(df_test.ID.unique()).intersection(set(df_temp.ID.unique()))) == df_test.shape[0]
df_temp.rename(columns={'value':'Label'}, inplace = True)

df_temp[['ID', 'Label']].to_csv("submission1.gz", compression = 'gzip' , index = False)
from IPython.display import FileLink

FileLink("submission1.gz")