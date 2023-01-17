%reload_ext autoreload

%autoreload 2

%matplotlib inline



from fastai.vision import *

from fastai.callbacks.hooks import *

from fastai.utils.mem import *



import shutil

import pathlib

import pandas as pd
path = pathlib.Path('../input')



path_img = path/'train_v2'

path_label = path/'train_ship_segmentations_v2.csv'

path_img_test = path/'test_v2'



print(len(path_img.ls()))

print(len(path_img_test.ls()))

fnames = get_image_files(path_img)



img_f = fnames[44]

img = open_image(img_f)

img.show(figsize=(5,5))

img.shape
! mkdir small_train



fnames_small = fnames[:200]

small_train_path = pathlib.Path('small_train/')



for fn in fnames_small:

    to_file = small_train_path/fn.name

    shutil.copy(str(fn), str(to_file))



fnames_small = get_image_files(small_train_path)



img_f = fnames_small[42]

img = open_image(img_f)

img.show(figsize=(5,5))

img.shape
label_df = pd.read_csv(path_label)

label_df[:10]
def get_seg(x_path, x_size=(768, 768)):

    rte_list = label_df.loc[label_df['ImageId'] == x_path.name]['EncodedPixels'].tolist()

    if len(rte_list) == 1 and not isinstance(rte_list[0], str):

        return open_mask_rle("", x_size)

    else:

        mask = FloatTensor(rle_decode(" ".join(rte_list), x_size).astype(np.uint8))

        mask = mask.view(x_size[1], x_size[0], -1)

        return ImageSegment(mask.permute(2,1,0))



    



img_f = fnames_small[3]



mask = get_seg(img_f)

img = open_image(img_f)

img.show(figsize=(5,5), y=mask, title='masked')

src_size = array([768, 768])

size = src_size//4



free = gpu_mem_get_free_no_cache()

# the max size of bs depends on the available GPU RAM

if free > 8200: bs=8

else:           bs=4

print(f"using bs={bs}, have {free}MB of GPU RAM free")
src = (SegmentationItemList.from_folder(small_train_path)

       .split_by_rand_pct(0.2)

       .label_from_func(get_seg, classes=array(['Backgroun', 'Vessel'], dtype='<U17')))

data = (src.transform(get_transforms(), size=size, tfm_y=True)

        .databunch(bs=bs)

        .normalize(imagenet_stats))
data.show_batch(2, figsize=(10,7))