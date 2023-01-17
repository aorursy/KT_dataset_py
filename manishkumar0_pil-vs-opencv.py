import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import PIL

import cv2

from PIL import ImageOps
print(cv2.__version__, cv2.__spec__)

print(cv2.getBuildInformation())
PIL.__version__, PIL.__spec__
!cat /proc/cpuinfo | egrep "model name"
!lsblk -o name,rota,type,mountpoint
import os

this_path = '.'

INPUT_PATH = os.path.abspath(os.path.join(this_path, '..', 'input'))

TRAIN_DATA = os.path.join(INPUT_PATH,"manish")



from glob import glob

filenames = glob(os.path.join(TRAIN_DATA, "*.jpg"))

len(filenames)
import matplotlib.pylab as plt

%matplotlib inline
import numpy as np

from PIL import Image, ImageOps,ImageFilter



def stage_1_PIL(filename):

    img_pil = Image.open(filename)

    img_pil = img_pil.filter(ImageFilter.BoxBlur(1))

    img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)

    return np.asarray(img_pil)



def stage_1_cv2(filename):

    img = cv2.imread(filename)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.blur(img, ksize=(3, 3))

    img = cv2.flip(img, flipCode=1)

    return img
f = filenames[0]

r1 = stage_1_PIL(f) 

r2 = stage_1_cv2(f)



plt.figure(figsize=(16, 16))

plt.subplot(131)

plt.imshow(r1)

plt.subplot(132)

plt.imshow(r2)

plt.subplot(133)

plt.imshow(np.abs(r1 - r2))
%timeit -n5 -r3 [stage_1_PIL(f) for f in filenames[:100]]
%timeit -n5 -r3 [stage_1_cv2(f) for f in filenames[:100]]
def stage_1b_PIL(img_pil):

    img_pil = img_pil.filter(ImageFilter.BoxBlur(1))

    img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)

    return np.asarray(img_pil)



def stage_1b_cv2(img):    

    img = cv2.blur(img, ksize=(3, 3))

    img = cv2.flip(img, flipCode=1)

    return img
imgs_PIL = [Image.open(filename) for filename in filenames[:100]]
def cv2_open(filename):

    img = cv2.imread(filename)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)



imgs_cv2 = [cv2_open(filename) for filename in filenames[:100]]
%timeit -n5 -r3 [stage_1b_PIL(img_pil) for img_pil in imgs_PIL]
%timeit -n5 -r3 [stage_1b_cv2(img) for img in imgs_cv2]













import numpy as np

from PIL import Image, ImageOps





def stage_2_PIL(filename):

    img_pil = Image.open(filename)

    img_pil = img_pil.resize((512, 512), Image.CUBIC)

    img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)

    img_pil = img_pil.transpose(Image.FLIP_TOP_BOTTOM)

    return np.asarray(img_pil)



def stage_2_cv2(filename):

    img = cv2.imread(filename)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    

    img = cv2.resize(img, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)

    img = cv2.flip(img, flipCode=1)

    img = cv2.flip(img, flipCode=0)

    return img
f = filenames[0]

r1 = stage_2_PIL(f) 

r2 = stage_2_cv2(f)



plt.figure(figsize=(16, 16))

plt.subplot(131)

plt.imshow(r1)

plt.subplot(132)

plt.imshow(r2)

plt.subplot(133)

plt.imshow(np.abs(r1 - r2))
%timeit -n5 -r3 [stage_2_PIL(f) for f in filenames[:200]]
%timeit -n5 -r3 [stage_2_cv2(f) for f in filenames[:200]]