!ls '../input/cnn-part-1-create-subslices-for-each-sound/output/'
!pip install split_folders
import split_folders



import os
os.makedirs('output')

os.makedirs('output/train')

os.makedirs('output/val')
audio_loc = '../input/cnn-part-1-create-subslices-for-each-sound/output/'



split_folders.ratio(audio_loc, output='output', seed=1337, ratio=(0.8, 0.2))