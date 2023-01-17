import imageio
import os
os.listdir('../input')
first_image = os.listdir('../input/stage_1_train_images')[0]
img = imageio.imread('../input/stage_1_train_images/4ba3e640-eb0a-4f4f-900c-af7405bc1790.dcm')
im  = imageio.imread('body-001.dcm')
