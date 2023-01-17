import glob, pylab, pandas as pd

import pydicom, numpy as np

from os import listdir

from os.path import isfile, join

import matplotlib.pylab as plt



import seaborn as sns



from tqdm import tqdm_notebook as tqdm

from fastai.vision import *
DATA = Path("../input/rsna-intracranial-hemorrhage-detection")
df = pd.read_csv('../input/rsna-intracranial-hemorrhage-detection/stage_1_train.csv')
newtable = df.copy()
new = newtable["ID"].str.split("_", n = 1, expand = True)

newX = new[1].str.split("_", n = 1, expand = True)

newX[1]

newtable['Image_ID'] = newX[0]

newtable['Sub_type'] = newX[1]
image_ids = newtable.Image_ID.unique()

labels = ["" for _ in range(len(image_ids))]

new_df = pd.DataFrame(np.array([image_ids, labels]).transpose(), columns=["id", "labels"])
lbls = {i : "" for i in image_ids}
newtable = newtable[newtable.Label == 1]

newtable = newtable[newtable.Sub_type != "any"]



i = 0

for name, group in newtable.groupby("Image_ID"):

    lbls[name] = " ".join(group.Sub_type)

    if i % 10000 == 0: print(i)

    i += 1
new_df = pd.DataFrame(np.array([list(lbls.keys()), list(lbls.values())]).transpose(), columns=["id", "labels"])
del lbls

del newtable

del newX

del new

gc.collect()
#https://www.kaggle.com/omission/eda-view-dicom-images-with-correct-windowing



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
new_df.id = "ID_" + new_df.id + ".dcm"
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
df_train = pd.concat([new_df[new_df.labels == ""][:15000], new_df[new_df.labels != ""][:15000]])
bs = 128



im_list = ImageList.from_df(df_train, path=DATA/"stage_1_train_images")

test_fnames = pd.DataFrame("ID_" + pd.read_csv(DATA/"stage_1_sample_submission.csv")["ID"].str.split("_", n=2, expand = True)[1].unique() + ".dcm")

test_im_list = ImageList.from_df(test_fnames, path=DATA/"stage_1_test_images")



tfms = get_transforms(do_flip=False)



data = (im_list.split_by_rand_pct(0.2)

               .label_from_df(label_delim=" ")

               .transform(tfms, size=512)

               .add_test(test_im_list)

               .databunch(bs=bs, num_workers=0)

               .normalize())
data.show_batch(3)
learn = cnn_learner(data, models.resnet18)



models_path = Path("/kaggle/working/models")

if not models_path.exists(): models_path.mkdir()

    

learn.model_dir = models_path

learn.metrics = [accuracy_thresh]
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(4, 5e-2)
learn.unfreeze()

learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(12, slice(1e-3))
submission = pd.read_csv(DATA/"stage_1_sample_submission.csv")
preds = learn.get_preds(ds_type=DatasetType.Test)
preds = np.array(preds[0])
any_probs = 1 - np.prod(1 - preds, axis=1)
any_probs.shape
submission.Label = np.hstack([preds, np.expand_dims(any_probs, -1)]).reshape(-1)
submission.head()
submission.to_csv("submission.csv", index=False)