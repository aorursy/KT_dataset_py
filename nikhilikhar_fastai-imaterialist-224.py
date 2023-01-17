# import required packages

import numpy as np

import pandas as pd



from pathlib import Path

from fastai.vision import *

from fastai.callbacks.hooks import *

from fastai.utils.mem import *



from itertools import groupby

from progressbar import ProgressBar

import cv2

import os

import json

import torchvision



category_num = 46 + 1



class ImageMask():

    masks = {}

    

    def make_mask_img(self, segment_df):

        seg_width = segment_df.iloc[0].Width

        seg_height = segment_df.iloc[0].Height

        

        seg_img = np.copy(self.masks.get((seg_width, seg_height)))

        try:

            if not seg_img:

                seg_img = np.full(seg_width*seg_height, category_num-1, dtype=np.int32)

                self.masks[(seg_width, seg_height)] = np.copy(seg_img)

        except:

            # seg_img exists

            pass

        for encoded_pixels, class_id in zip(segment_df["EncodedPixels"].values, segment_df["ClassId"].values):

            pixel_list = list(map(int, encoded_pixels.split(" ")))

            for i in range(0, len(pixel_list), 2):

                start_index = pixel_list[i] - 1

                index_len = pixel_list[i+1] - 1

                if int(class_id.split("_")[0]) < category_num - 1:

                    seg_img[start_index:start_index+index_len] = int(class_id.split("_")[0])

        seg_img = seg_img.reshape((seg_height, seg_width), order='F')

        return seg_img

        



def create_label(images, path_lbl):

    """

    img_name = "53d0ee82b3b7200b3cec8c3c1becead9.jpg"

    img_df = df[df.ImageId == img_name]

    img_mask = ImageMask()

    mask = img_mask.make_mask_img(img_df)

    """

    img_mask = ImageMask()



    print("Start creating label")

    for i,img in enumerate(images):

        fname = path_lbl.joinpath(os.path.splitext(img)[0] + '_P.png').as_posix()

        if os.path.isfile(fname): # skip

            continue

        img_df = df[df.ImageId == img]

        mask = img_mask.make_mask_img(img_df)

        img_mask_3_chn = np.dstack((mask, mask, mask))

        cv2.imwrite(fname, img_mask_3_chn)

        if i % 40 ==0 : print(i, end=" ")

    print("Finish creating label")

            

        

def get_predictions(path_test, learn, size):

    # predicts = get_predictions(path_test, learn)

    learn.model.cuda()

    files = list(path_test.glob("**/*.jpg"))    #<---------- HERE

    test_count = len(files)

    results = {}

    for i, img in enumerate(files):

        results[img.stem] = learn.predict(open_image(img).resize(size))[1].data.numpy().flatten()

    

        if i%20==0:

            print("\r{}/{}".format(i, test_count), end="")

    return results       

        



# https://www.kaggle.com/go1dfish/u-net-baseline-by-pytorch-in-fgvc6-resize

def encode(input_string):

    return [(len(list(g)), k) for k,g in groupby(input_string)]



def run_length(label_vec):

    encode_list = encode(label_vec)

    index = 1

    class_dict = {}

    for i in encode_list:

        if i[1] != category_num-1:

            if i[1] not in class_dict.keys():

                class_dict[i[1]] = []

            class_dict[i[1]] = class_dict[i[1]] + [index, i[0]]

        index += i[0]

    return class_dict



    

def get_submission_df(predicts):

    sub_list = []

    for img_name, mask_prob in predicts.items():

        class_dict = run_length(mask_prob)

        if len(class_dict) == 0:

            sub_list.append([img_name, "1 1", 1])

        else:

            for key, val in class_dict.items():

                sub_list.append([img_name + ".jpg", " ".join(map(str, val)), key])

        # sub_list

    jdf = pd.DataFrame(sub_list, columns=['ImageId','EncodedPixels', 'ClassId'])

    return jdf

        

        

def test_mask_to_img(segment_df):

    """

    plt.imshow(test_mask_to_img(jdf))

    """

    seg_img = np.full(size*size, category_num-1, dtype=np.int32)

    for encoded_pixels, class_id in zip(segment_df["EncodedPixels"].values, segment_df["ClassId"].values):

        encoded_pixels= str(encoded_pixels)

        class_id = str(class_id)

        

        pixel_list = list(map(int, encoded_pixels.split(" ")))

        for i in range(0, len(pixel_list), 2):

            start_index = pixel_list[i] - 1

            index_len = pixel_list[i+1] - 1

            if int(class_id.split("_")[0]) < category_num - 1:

                seg_img[start_index:start_index+index_len] = int(class_id.split("_")[0])

        seg_img = seg_img.reshape((size, size), order='F')

    return seg_img

    

path = Path("../input/imaterialist-fashion-2019-FGVC6/")

path_img = path/'train'

path_lbl = Path("../input/imaterialist2019labels/labels/labels/")

path_test = path/'test'

# if  not os.path.isdir(path_lbl):

#     os.makedirs(path_lbl)



# create a folder for the mask images

if  not os.path.isdir(path_lbl):

    os.makedirs(path_lbl)

    

# category is in util

size = 224               # <---------------- HERE



# train dataframe

df = pd.read_csv(path/'train.csv')



# get and show categories

with open(path/"label_descriptions.json") as f:

    label_descriptions = json.load(f)



label_names = [x['name'] for x in label_descriptions['categories']]

print(label_names)
img_name = "53d0ee82b3b7200b3cec8c3c1becead9.jpg"



img_df = df[df.ImageId == img_name]

img_mask = ImageMask()

mask = img_mask.make_mask_img(img_df)



# img_mask = make_mask_img(img_df)

plt.imshow(mask)
images = df.ImageId.unique()[:10000]        # <---------------- HERE
# Already created

# create_label(images, path_lbl)
get_y_fn = lambda x: path_lbl/f'{Path(x).stem}_P.png'



bs = 16

#classes = label_names

classes = list(range(category_num))

wd = 1e-2



images_df = pd.DataFrame(images)





src = (SegmentationItemList.from_df(images_df, path, folder='train')

       .split_by_rand_pct()

       .label_from_func(get_y_fn, classes=classes)

       .add_test_folder('test')

     )


def no_tfms(self, x, size, resize_method): return x

EmptyLabel.apply_tfms = no_tfms



data = (src.transform(tfms=get_transforms(), size=size, resize_method=ResizeMethod.SQUISH, tfm_y=True)

       .databunch(bs=bs,num_workers=4)

        .normalize(imagenet_stats)

       )
def acc_fashion(input, target):

    target = target.squeeze(1)

    mask = target != (category_num - 1)

    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()



learn = unet_learner(data, models.resnet34, metrics=acc_fashion, wd=wd, model_dir="/kaggle/working/models")



# look at a batch

data.show_batch(3, figsize=(10,10))
# find lr



# lr_find(learn)

# learn.recorder.plot()
lr=1e-4

learn.fit_one_cycle(5, slice(lr), pct_start=0.9) #<------------ HERE
#  take a look at some results

learn.show_results()
# unfreeze earlier weights

# decrease the learning rate

# train for 10 more cycles unfrozen



learn.unfreeze()

lrs = slice(lr/400,lr/4)



learn.fit_one_cycle(5, lrs, pct_start=0.8)          #<------------ HERE
# more results



learn.show_results()
# check can we save model

learn.save('stage-1-224')
predicts = get_predictions(path_test, learn, size)

len(predicts)
submission_df = get_submission_df(predicts)

submission_df.to_csv("./submission.csv", index=False)
test_df = df[df.ImageId=="53d0ee82b3b7200b3cec8c3c1becead9.jpg"]

pred = learn.predict(open_image((path_img/img_name).as_posix()).resize(size))

pred[0]