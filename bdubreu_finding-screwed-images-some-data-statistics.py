!pip install torch torchvision feather-format pyarrow --upgrade   > /dev/null

!pip install git+https://github.com/fastai/fastai_dev             > /dev/null



from fastai2.basics           import *

from fastai2.medical.imaging  import *



np.set_printoptions(linewidth=120)
path_inp = Path('../input')

path = path_inp/'rsna-intracranial-hemorrhage-detection'

path_trn = path/'stage_1_train_images'

path_tst = path/'stage_1_test_images'
# This is the crappy file we found: 

dcmread('../input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/ID_b79eed528.dcm').show(figsize=(6,6))



# the image is really noisy. That's a 600K+ images dataset we're looking at, so there might be more of those. Can we find them ?
# This pic has two interesting features:

# - a std of 11269!

# - a lower quartile of -2000

pd.Series((dcmread('../input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/ID_b79eed528.dcm').pixel_array.flatten())).describe()
# To understand what's going on here, please refer to https://www.kaggle.com/jhoward/some-dicom-gotchas-to-be-aware-of-fastai

path_df = path_inp/'creating-a-metadata-dataframe'



df_lbls = pd.read_feather(path_df/'labels.fth')

df_tst = pd.read_feather(path_df/'df_tst.fth')

df_trn = pd.read_feather(path_df/'df_trn.fth')



comb = df_trn.join(df_lbls.set_index('ID'), 'SOPInstanceUID')

assert not len(comb[comb['any'].isna()])
# So, a std of 11269 was indeed a very weird value, considering the 99th percentile is 1340 

# (that means 99% of picture have a std of 1340 or lower)

comb['img_std'].quantile([0.5, 0.7, 0.9, 0.99, 0.999, 0.9999])
# Indeed, this is actually the only image with that kind of standard deviation

# The second largest standard dev (in terms of pixel values) is 1513

comb['img_std'].sort_values(ascending=False)[:5]
# other images with a large std (> 1500) show no signs of being corrupt

f_name = comb[ comb['img_std'] > 1500 ].sample()['fname'].values[0]

print(f_name)

dcmread(f_name).show(figsize=(6,6))

pd.Series(dcmread(f_name).pixel_array.flatten()).describe()
# I used matplotlib to get a sense of which line started to get crappy

import matplotlib.pyplot as plt

plt.imshow(dcmread('../input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/ID_b79eed528.dcm').pixel_array)

# Around 300-350
# Doing some bisection search, I found the limit:

# line 332 (indexing starts at 0) has 864 std

dcmread('../input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/ID_b79eed528.dcm').pixel_array[331].std()
# but then at line 333, std in terms of pixel values starts to skyrocket:

dcmread('../input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/ID_b79eed528.dcm').pixel_array[333].std()
df_trn['img_std'].drop(640545).describe()
df_tst['img_std'].describe()
# we find back the image noted in this kernel : https://www.kaggle.com/tonyyy/corrupted-pixeldata-in-id-6431af929-dcm

df_trn[df_trn['img_std'] == 0]['fname']

# in fact, we can't plot any of those pics because they are all corrupt
# in fact, we can't plot any of those pics because they are all corrupt

# although only the ID_6431af929.dcm image will raise a value error

# the four other raise index Errors, and I don't clearly know why

# select the three following lines and then CTRL+/ to uncomment them all at once (amazing, isn't it?)



# fname = df_trn[df_trn['img_std'] == 0].sample()['fname'].values[0]

# print(fname)

# dcmread(fname).show(figsize=(6,6))
# Images with std between 0 and 1 are really spherical. I don't know what we should do with those ?! 

# You can press shift-enter several times here to see a bunch of examples

fname = df_trn[(df_trn['img_std'] > 0) & (df_trn['img_std'] < 1)].sample()['fname'].values[0]

print(fname)

dcmread(fname).show(figsize=(6,6))
# These is (only) one similar picture in the test set

fname = df_tst[(df_tst['img_std'] > 0) & (df_tst['img_std'] < 1)].sample()['fname'].values[0]

print(fname)

dcmread(fname).show(figsize=(6,6))
fig, axes = plt.subplots(2, 4, figsize=(20,10))

for i, img in enumerate(comb[ comb['img_std'] < 300 ].sample(8)['fname'].values):

    dcmread(img).show(ax=axes[i%2, i//2])
fig, axes = plt.subplots(2, 4, figsize=(20,10))

for i, img in enumerate(comb[ comb['img_std'] > 600 ].sample(8)['fname'].values):

    dcmread(img).show(ax=axes[i%2, i//2])
fig, axes = plt.subplots(2, 4, figsize=(20,10))

for i, img in enumerate(comb[ comb['img_std'] > 1200 ].sample(8)['fname'].values):

    dcmread(img).show(ax=axes[i%2, i//2])
# Problem: we have low detail images in the test set as well

fig, axes = plt.subplots(2, 4, figsize=(20,10))

for i, img in enumerate(df_tst[ df_tst['img_std'] < 300 ].sample(8)['fname'].values):

    dcmread(img).show(ax=axes[i%2, i//2])