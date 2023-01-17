import numpy as np 

import pandas as pd

from glob import glob

from PIL import Image

import matplotlib.pyplot as plt

%matplotlib inline
def defeat_position(img_id, class_id):

    

    train_s = train[train.ImageId_ClassId==f"{img_id}_{class_id}"]

    #print(train_s.EncodedPixels.values[0])

    

    if not isinstance(train_s.EncodedPixels.values[0], str) and np.isnan(train_s.EncodedPixels.values[0]):

        return np.array([]), np.array([])

    

    encoded_pixels = [int(i) for i in train_s.EncodedPixels.values[0].split(" ")]



    pixcels = []

    for pos, offset in zip(encoded_pixels[0:len(encoded_pixels):2], encoded_pixels[1:len(encoded_pixels):2]):

        pixcels.extend(list(range(pos, pos+offset)))

    pixcels = np.array(pixcels)    

    x = pixcels // 256

    y = pixcels % 256  

    return x, y



def show_segmented_image(img_file, x, y, class_id):

    im = np.array(Image.open(img_file))

    

    if class_id == 4:

        im[y, x, 1] += 50

        im[y, x, 2] += 50

    else:

        im[y, x, class_id-1] += 50

        

    im = np.clip(im, 0, 255)



    plt.figure(figsize=(25,5))

    plt.imshow(im)

    plt.xticks([]);plt.yticks([]);

    plt.show()

    

def visualize_defect(class_id, n_show=20):

    cnt = 0

    for img_file in train_files:

        img_id = img_file.split('/')[-1]

        x, y = defeat_position(img_id, class_id)

        if len(x)==0:

            continue

        print(img_file, class_id)

        show_segmented_image(img_file, x, y, class_id)

        cnt += 1

        if cnt > n_show: break
train = pd.read_csv("../input/severstal-steel-defect-detection/train.csv")



train_files = np.sort(glob("../input/severstal-steel-defect-detection/train_images/*"))

test_files  = np.sort(glob("../input/severstal-steel-defect-detection/test_images/*"))

print(f"number of train images: {len(train_files)}")

print(f"number of test images: {len(test_files)}")

na_cut = train.EncodedPixels.isna()

count_na = (~na_cut).astype(int)
# The number of defect

count_na.value_counts()
# The number of each defect class

pd.Series(train.loc[~na_cut].ImageId_ClassId.str.split("_", expand=True).values[:, 1]).value_counts().sort_index()
visualize_defect(1, n_show=20)
visualize_defect(2, n_show=20)
visualize_defect(3, n_show=20)
visualize_defect(4, n_show=20)