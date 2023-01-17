%pylab inline
!find ../input/train/ -type f -name '*.jpg' | cut -f4 -d/ | sort | uniq -c
!find ../input/train -type f -name '*.jpg' | wc -l
import glob, os

from skimage import io

from skimage import exposure

all_image_files = glob.glob("../input/train/*/*.jpg")

image_heights = []

image_widths = []

image_sizes = []

rgb_intensities = []

is_low_contrast = []



for image_file in np.random.choice(all_image_files, 500, replace=False):

    img = io.imread(image_file)

    image_widths.append(img.shape[0])

    image_heights.append(img.shape[1])

    image_sizes.append(os.path.getsize(image_file))

    rgb_intensities.append(np.mean(img, axis=(0,1)))

    is_low_contrast.append(exposure.is_low_contrast(img))
import pandas as pd

image_stats = pd.DataFrame(

    {'image heights': image_heights,

     'image widths': image_widths,

     'image sizes': [v/1000.0 for v in image_sizes], 

     'red intensity': [v[0] for v in rgb_intensities], 

     'green intensity': [v[1] for v in rgb_intensities], 

     'blue intensity': [v[2] for v in rgb_intensities], 

     'is low contrast': is_low_contrast, 

    })
image_stats.describe()
from collections import Counter

Counter(is_low_contrast)
alb_images = [v for v in all_image_files if "/ALB/" in v]

sample = np.random.choice(alb_images, 4, replace=False)

fig, axes = plt.subplots(2,2, figsize=(8,8))

idx = 0

for axrow in axes:

    for axcell in axrow:

        img = io.imread(alb_images[idx])

        idx+=1

        axcell.imshow(img)

plt.tight_layout()

plt.show()