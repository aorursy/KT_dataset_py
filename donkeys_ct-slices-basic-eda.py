import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import pydicom

import matplotlib.pyplot as plt

from tqdm import tqdm

from PIL import Image



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import pydicom

import matplotlib.pyplot as plt

from tqdm.auto import tqdm

from PIL import Image



tqdm.pandas()

!ls
!ls /kaggle/input/osic-pulmonary-fibrosis-progression
!ls /kaggle/input/osic-pulmonary-fibrosispreprocessed/dataset
df_train = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv")

df_train.head()
df_test = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv")

df_test.head()
df_train_meta = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosispreprocessed/dataset/meta_files_train.csv")

df_train_meta.head()
df_test_meta = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosispreprocessed/dataset/meta_files_test.csv")

df_test_meta.head()
df_train["SmokingStatus"].value_counts()
df_train["Sex"].value_counts()
df_train["Age"].value_counts().sort_index()
df_train["Patient"].value_counts()
df_train[df_train["Patient"] == "ID00229637202260254240583"]
!ls /kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430 | wc -l
    !ls /kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00229637202260254240583/
import glob



files = glob.glob("/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00229637202260254240583/*.dcm")

files
import re



files.sort(key=lambda f: int(re.sub('\D', '', f)))

files
def show_img(img_path):

    ds = pydicom.dcmread(img_path)

    im = Image.fromarray(ds.pixel_array)

    #im = im.resize((DESIRED_SIZE,DESIRED_SIZE)) 

    #im.show()

    plt.imshow(im, cmap=plt.cm.bone)

    plt.show()

    



show_img(files[0])

#for file in files:

#    show_img(file)

import types



#ds = pydicom.dcmread('/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00229637202260254240583/1.dcm')

ds = pydicom.dcmread('/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00421637202311550012437/1.dcm')

attrs = dir(ds)

for attr in attrs:

    if attr.startswith("_"):

        continue

    if attr == "PixelData" or attr == "pixel_array": 

        continue

#            print(f"{attr}")

    var_type = type(getattr(ds,attr))

    if var_type == types.MethodType: 

        continue

        #    print(f"{attr}: {type(attr)}")

    #print(f"{attr}: {var_type}")

    print(f"{attr}: {getattr(ds,attr)}")

    
import math



def plot_images(images, cols=3):

#    plt.clf()

#    plt.figure(figsize(14,8))

    rows = math.ceil(len(images)/cols)

    fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(14,5*rows))



    idx = 0

    for row in ax:

        for col in row:

            if idx > len(images)-1:

                break

            img_path= images[idx]

            ds = pydicom.dcmread(img_path)

            im = Image.fromarray(ds.pixel_array)

            col.imshow(im)

            col.title.set_text(ds.ImagePositionPatient)

            idx += 1

    plt.show()



plot_images(files)
import math



def plot_image_diffs(images, cols=3):

#    plt.clf()

#    plt.figure(figsize(14,8))

    rows = math.ceil(len(images)/cols)

    fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(14,5*rows))



    prev_img = None

    idx = 0

    for row in ax:

        for col in row:

            if idx > len(images)-1:

                break

            img_path= images[idx]

            ds = pydicom.dcmread(img_path)

            if prev_img is not None:

                print(f"no diff {idx}")

                diff = prev_img - ds.pixel_array

                #diff = ds.pixel_array - prev_img

                im = Image.fromarray(diff)

            else:

                print(f"diff {idx}")

                im = Image.fromarray(ds.pixel_array)

            col.imshow(im)

            col.title.set_text(ds.ImagePositionPatient)

            prev_img = ds.pixel_array

            idx += 1

    plt.show()



plot_image_diffs(files)
import glob



files = glob.glob("/kaggle/input/osic-pulmonary-fibrosispreprocessed/dataset/png/train/ID00229637202260254240583/*.png")

files
import re



files.sort(key=lambda f: int(re.sub('\D', '', f)))

files
def show_png(img_path):

    im = Image.open(img_path)

    plt.imshow(im, cmap=plt.cm.bone)

    plt.show()    



show_png(files[0])

import math



def plot_images(images, cols=3):

    rows = math.ceil(len(images)/cols)

    fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(14,5*rows))



    idx = 0

    for row in ax:

        for col in row:

            if idx > len(images)-1:

                break

            img_path= images[idx]

            im = Image.open(img_path)

            col.imshow(im, cmap=plt.cm.bone)

            #col.title.set_text()

            idx += 1

    plt.show()



plot_images(files)
import math



def plot_image_diffs(images, cols=3):

    rows = math.ceil(len(images)/cols)

    fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(14,5*rows))



    prev_img = None

    idx = 0

    for row in ax:

        for col in row:

            if idx > len(images)-1:

                break

            img_path= images[idx]

            img = Image.open(img_path)

            if prev_img is not None:

                print(f"no diff {idx}")

                img_data = np.asarray(img, dtype=np.uint8)

                diff = prev_img - img_data

                im = Image.fromarray(diff)

            else:

                print(f"diff {idx}")

                im = img

            col.imshow(im, cmap=plt.cm.bone)

            #col.title.set_text(ds.ImagePositionPatient)

            prev_img = np.asarray(img, dtype=np.uint8)

            idx += 1

    plt.show()



plot_image_diffs(files)
df_train_meta.describe()
def show_train_png(img_id, idx):

    plt.figure(figsize=(12,8))

    im = Image.open(f"/kaggle/input/osic-pulmonary-fibrosispreprocessed/dataset/png/train/{img_id}/{idx}.png")

    plt.imshow(im, cmap=plt.cm.bone)

    plt.show()    



show_train_png("ID00012637202177665765362", 1)
show_train_png("ID00012637202177665765362", 7)
show_train_png("ID00019637202178323708467", 2)
show_train_png("ID00019637202178323708467", 7)
show_train_png("ID00020637202178344345685", 1)

show_train_png("ID00047637202184938901501", 1)
show_train_png("ID00122637202216437668965", 1)
show_train_png("ID00076637202199015035026", 1)
show_train_png("ID00126637202218610655908", 1)
show_train_png("ID00133637202223847701934", 1)
show_train_png("ID00123637202217151272140", 1)
show_train_png("ID00134637202223873059688", 1)
show_train_png("ID00139637202231703564336", 1)
show_train_png("ID00264637202270643353440", 1)
show_train_png("ID00196637202246668775836", 1)