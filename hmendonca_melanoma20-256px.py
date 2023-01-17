target_size = 256



# clean up

!rm -rf test train
import os, cv2, random

from glob import glob

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from joblib import Parallel, delayed

from tqdm.notebook import tqdm



files = glob('../input/*/jpeg/*/*.jpg')

random.shuffle(files)



def crop(f, debug=False):

#     filename = os.path.basename(f)

    filename = f.split('/')[-2:]

    os.makedirs(filename[0], exist_ok=True)

    filename = '/'.join(f.split('/')[-2:])



    img = cv2.imread(f, cv2.IMREAD_UNCHANGED)

    size = min(img.shape[:2])

    offset0 = (img.shape[0] - size) // 2

    offset1 = (img.shape[1] - size) // 2

    if debug:

        print(filename, img.shape, size, offset0, offset1)

        plt.imshow(img) # in cv2 BGR

        plt.show()

    # center crop

    img = img[offset0:offset0+size, offset1:offset1+size, :]

    img = cv2.resize(img, (target_size, target_size))

    cv2.imwrite(filename, img, [cv2.IMWRITE_JPEG_QUALITY, 99])

    if debug:

        plt.imshow(mpimg.imread(filename))



crop(files[0], debug=True)
with Parallel(n_jobs=os.cpu_count()) as parallel:

    parallel(delayed(crop)(i) for i in tqdm(files))
!du -sh *
!zip -rmq images{target_size}px.zip test train

!ls -sh *.zip