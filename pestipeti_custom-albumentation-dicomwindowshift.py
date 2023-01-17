import albumentations as A

import cv2

import numpy as np

import pydicom

import random



from albumentations.core.transforms_interface import ImageOnlyTransform

from matplotlib import pyplot as plt



IMAGE_DIR = '/kaggle/input/rsna-intracranial-hemorrhage-detection/stage_1_test_images/'
def apply_window(image, center, width):

    image = image.copy()



    min_value = center - width // 2

    max_value = center + width // 2



    image[image < min_value] = min_value

    image[image > max_value] = max_value



    return image





def dicom_window_shift(img, windows, min_max_normalize=True):

    image = np.zeros((img.shape[0], img.shape[1], 3))



    if img.ndim == 2:

        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)



    for i in range(3):

        ch = apply_window(img[:, :, i], windows[i][0], windows[i][1])



        if min_max_normalize:

            image[:, :, i] = (ch - ch.min()) / (ch.max() - ch.min())

        else:

            image[:, :, i] = ch



    return image





class DicomWindowShift(ImageOnlyTransform):

    """Randomly shift the DICOM window (per channel) between min and max values.

    

    Note: It won't work for preprocessed png or jpg images. Please use the dicom's HU values

    (rescaled width slope/intercept!)

    

    Args:

        window_width_mins (int, int, int): minimun window width per channel

        window_width_maxs (int, int, int): maximum window width per channel

        window_center_mins (int, int, int): minimum value for window center per channel

        window_center_maxs (int, int, int): maximum value for window center per channel

        min_max_normalize: (bool) Apply min-max normalization

    Targets:

        image

    Image types:

        uint8 (shape: HxW | HxWxC)

    """

    def __init__(

            self,

            window_width_mins=(80, 200, 380),

            window_width_maxs=(80, 200, 380),

            window_center_mins=(40, 80, 40),

            window_center_maxs=(40, 80, 40),

            min_max_normalize=True,

            always_apply=False,

            p=0.5,

    ):

        super(DicomWindowShift, self).__init__(always_apply, p)

        self.window_width_mins = window_width_mins

        self.window_width_maxs = window_width_maxs

        self.window_center_mins = window_center_mins

        self.window_center_maxs = window_center_maxs

        self.min_max_normalize = min_max_normalize



        assert len(self.window_width_mins) == 3

        assert len(self.window_width_maxs) == 3

        assert len(self.window_center_mins) == 3

        assert len(self.window_center_maxs) == 3



    def apply(self, image, windows=(), min_max_normalize=True, **params):

        return dicom_window_shift(image, windows, min_max_normalize)



    def get_params_dependent_on_targets(self, params):

        windows = []



        for i in range(3):

            window_width = random.randint(self.window_width_mins[i], self.window_width_maxs[i])

            window_center = random.randint(self.window_center_mins[i], self.window_center_maxs[i])



            windows.append([window_center, window_width])



        return {"windows": windows, "min_max_normalize": self.min_max_normalize}



    @property

    def targets_as_params(self):

        return ["image"]



    def get_transform_init_args_names(self):

        return "window_width_mins", "window_width_maxs", "window_center_mins", "window_center_maxs", "min_max_normalize"

sample_id = 'ID_687b495b6'



dicom = pydicom.read_file(IMAGE_DIR + sample_id + '.dcm')

image = dicom.pixel_array

image = image * dicom.RescaleSlope + dicom.RescaleIntercept



plt.imshow(apply_window(image, 40, 80), cmap='gray')
transform = DicomWindowShift(

    # (brain_width_min, subdural_width_min, bones_width_min)

    window_width_mins=(75, 190, 360),

    

    # (brain_width_max, subdural_width_min, bones_width_min)

    window_width_maxs=(85, 210, 400),

    

    # (brain_center_min, subdural_center_min, bones_center_min)

    window_center_mins=(15, 75, 35),

    

    # (brain_center_max, subdural_center_max, bones_center_max)

    window_center_maxs=(85, 85, 45),



    min_max_normalize=True,

    p=1.0

)
f, ax = plt.subplots(2, 5, figsize=(16, 8))

ax = ax.flatten()



for i in range(10):

    tr = transform(image=image)

    ax[i].imshow(tr['image'][:,:,0], cmap='gray')
transform = A.Compose([

    A.Rotate(p=1.0),

    DicomWindowShift(window_width_mins=(75, 190, 360),

                     window_width_maxs=(85, 210, 400),

                     window_center_mins=(15, 75, 35),

                     window_center_maxs=(85, 85, 45),

                     min_max_normalize=True,

                     p=1.0)

])
f, ax = plt.subplots(2, 5, figsize=(16, 8))

ax = ax.flatten()



for i in range(10):

    tr = transform(image=image)

    ax[i].imshow(tr['image'][:,:,0], cmap='gray')