%matplotlib inline

import os



import numpy as np

import openslide

from matplotlib import pyplot as plt
images_dir = '/kaggle/input/prostate-cancer-grade-assessment/train_images'

images_filenames = os.listdir(images_dir)

len(images_filenames)
class PatchSlicer:

    height: int

    width: int



    def __init__(self, path_to_wsi: str, step_size: int = 256):

        self.path_to_wsi = path_to_wsi

        self.step_size = step_size

        self.patch_size = (self.step_size, self.step_size)

        self.x = 0

        self.y = 0



    def patch_generator(self):

        with openslide.OpenSlide(self.path_to_wsi) as wsi :

            self.width = wsi.level_dimensions[0][0]

            self.height = wsi.level_dimensions[0][1]

            while self.y + self.step_size < self.height:

                while self.x + self.step_size < self.width:

                    coords = (self.x, self.y)

                    yield wsi.read_region(coords, 0, self.patch_size), coords

                    self.x += self.step_size

                self.x = 0

                self.y += self.step_size
step_size = 512

image_index = 0

ps = PatchSlicer(os.path.join(images_dir, images_filenames[image_index]), step_size)

ps_generator = ps.patch_generator()



mean_pixel_value = []

coords = []

for patch, coord in ps_generator:

    mean_pixel_value.append(np.array(patch).mean())

    coords.append(coord)

len(coords)
xy = zip(mean_pixel_value, coords)

image = openslide.OpenSlide(os.path.join(images_dir, images_filenames[image_index]))

cols = 20

rows = 20

offset = 0

plt.figure(figsize=(20, 20))

plt.subplots_adjust(wspace=0, hspace=0)



for chart, (_, coord) in enumerate(sorted(xy)[offset:], 1):

    ax = plt.subplot(rows, cols, chart)

    ax.imshow(np.array(image.read_region(coord, 0, (step_size, step_size))))

    ax.axis('off')

    if chart == cols * rows:

        break