import numpy as np

import glob

import pandas as pd

import matplotlib.pyplot as plt

import os

os.listdir('../input')
df = pd.read_csv('../input/stage1_labels.csv')

df.head()
basedir = '../input/sample_images'

baseids = os.listdir(basedir)
# From https://www.kaggle.com/anokas/data-science-bowl-2017/exploratory-data-analysis

import dicom 



def get_slice_location(dcm):

    return float(dcm[0x0020, 0x1041].value)



def load_patient(patient_id):

    files = glob.glob(basedir + '/{}/*.dcm'.format(patient_id))

    imgs = {}

    for f in files:

        dcm = dicom.read_file(f)

        img = dcm.pixel_array

        img[img == -2000] = 0

        sl = get_slice_location(dcm)

        imgs[sl] = img



    # Not a very elegant way to do this

    sorted_imgs = [x[1] for x in sorted(imgs.items(), key=lambda x: x[0])]

    return sorted_imgs



def draw_patient(id_i):

    pat = load_patient(id_i)

    fig, axs = plt.subplots(11, 10, sharex='all', sharey='all', figsize=(10, 11))

    # matplotlib is drunk

    #plt.title('Sorted Slices of Patient 0a38e7597ca26f9374f8ea2770ba870d - No cancer')

    for i in range(110):

        axs[i // 10, i % 10].axis('off')

        axs[i // 10, i % 10].imshow(pat[i], cmap='jet')

    fig.suptitle('cancer' if df['cancer'][df['id'] == id_i].values[0] else 'no cancer')
draw_patient(baseids[0])
dicom.read_file(

    glob.glob(basedir + '/{}/*.dcm'.format(baseids[0]))[0])
def get_pixel_data(baseids):

    vals = []

    for id_i in baseids:

        this_val = []

        files = glob.glob(basedir + '/{}/*.dcm'.format(id_i))

        for f in files:

            dcm = dicom.read_file(f)

            img = dcm.pixel_array

            img[img == -2000] = 0

            this_val.append(img)

        vals.append(this_val)

    return vals





def get_data(baseids, format1, format2, **kwargs):

    vals = []

    for id_i in baseids:

        files = glob.glob(basedir + '/{}/*.dcm'.format(id_i))

        for f in files:

            dcm = dicom.read_file(f, **kwargs)

            vals.append(dcm[format1, format2].value)

    return vals





def get_nested_data(baseids, format1, format2, **kwargs):

    vals = []

    for id_i in baseids:

        this_val = []

        files = glob.glob(basedir + '/{}/*.dcm'.format(id_i))

        for f in files:

            dcm = dicom.read_file(f, **kwargs)

            this_val.append(dcm[format1, format2].value)

        vals.append(this_val)

    return vals





def get_pixel_sizes(baseids, nested=False):

    if nested:

        return [[(float(el[0]), float(el[1])) for el in sub]

                for sub in get_nested_data(baseids, 0x0028, 0x0030, stop_before_pixels=True)]

    else:

        return [(float(el[0]), float(el[1]))

               for el in get_data(baseids, 0x0028, 0x0030, stop_before_pixels=True)]





def get_orientations(baseids, nested=False):

    if nested:

        return [[tuple(el) for el in sub]

                for sub in get_nested_data(baseids, 0x0020, 0x0037, stop_before_pixels=True)]

    else:

        return [tuple(el) for el in get_data(baseids, 0x0020, 0x0037, stop_before_pixels=True)]

        

        

def get_slice_locations(baseids, nested=False):

    if nested:

        return [[float(el) for el in sub]

                for sub in get_nested_data(baseids, 0x0020, 0x1041, stop_before_pixels=True)]

    else:

        return [float(el) for el in get_data(baseids, 0x0020, 0x1041, stop_before_pixels=True)]





def get_positions(baseids, nested=False):

    if nested:

        return [[tuple(el) for el in sub]

                for sub in get_nested_data(baseids, 0x0020, 0x0032, stop_before_pixels=True)]

    else:

        return [tuple(el) for el in get_data(baseids, 0x0020, 0x0032, stop_before_pixels=True)]





def get_dimensions(baseids):

    return zip(

        get_data(baseids, 0x0028, 0x0010, stop_before_pixels=True),

        get_data(baseids, 0x0028, 0x0011, stop_before_pixels=True))
res_pxs = set(get_pixel_sizes(baseids))

res_ori = set(get_orientations(baseids))

res_pos = set(get_positions(baseids))

res_dim = set(get_dimensions(baseids))

res_loc = set(get_slice_locations(baseids))
res_pxs
res_ori
res_pos
res_dim
res_loc
imgs = get_pixel_data(baseids)
len(imgs)
imgs_3d = [np.concatenate([img_j[..., np.newaxis] for img_j in img_i], 2) for img_i in imgs]
for px_i, shp_i in zip(res_pxs, [imgs_3d_i.shape for imgs_3d_i in imgs_3d]):

    print('pixel res: {}, shape: {}'.format(px_i, shp_i))
locs = get_slice_locations(baseids, nested=True)

fig, axs = plt.subplots(2, 1)

for sub in locs:

    sorted_pos = sorted([float(pos_i) for pos_i in sub])

    axs[0].plot(sorted_pos)

    axs[1].plot(sorted_pos - np.mean(sorted_pos))
pos = get_positions(baseids, nested=True)

fig, axs = plt.subplots(2, 1)

for sub in pos:

    sorted_pos = sorted([float(pos_i[2]) for pos_i in sub])

    axs[0].plot(sorted_pos)

    axs[1].plot(sorted_pos - np.mean(sorted_pos))
mean = np.zeros((512, 512))

m2 = np.zeros((512, 512))

s = np.zeros((512, 512))

n_imgs = 3604

for sub in imgs:

    for img_i in sub:

        delta = img_i - mean

        mean += delta / n_imgs;

        m2 += delta * (img_i - mean)

        s += np.sqrt(np.maximum(1e-10, m2)) / (n_imgs - 1)
fig, axs = plt.subplots(1, 2, figsize=(10,5))

axs[0].imshow(mean, cmap='bone')

axs[1].imshow(s, cmap='bone')