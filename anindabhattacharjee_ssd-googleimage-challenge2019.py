# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



#import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

 #   for filename in filenames:

  #      print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import torch 

precision = 'fp32'

ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math=precision)
import numpy

import scipy

import skimage 

import matplotlib as mt

from skimage import io,transform
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')
ssd_model.to('cuda')

ssd_model.eval()
!pip install torchsummary 
from torchsummary import summary



summary(ssd_model, (3, 300, 300))
uris = [

    '/kaggle/input/open-images-2019-object-detection/test/c744be039ce8b59f.jpg',

    '/kaggle/input/open-images-2019-object-detection/test/6ced51a34b3e6bb5.jpg',

    '/kaggle/input/open-images-2019-object-detection/test/5a3215a639ea3308.jpg',

    '/kaggle/input/open-images-2019-object-detection/test/827376834a225c73.jpg',

    '/kaggle/input/open-images-2019-object-detection/test/9a3de6cd6c83f1e0.jpg'

]
inputs = [utils.prepare_input(uri) for uri in uris]

tensor = utils.prepare_tensor(inputs, precision == 'fp16')
with torch.no_grad():

    detections_batch = ssd_model(tensor)
results_per_input = utils.decode_results(detections_batch)

best_results_per_input = [utils.pick_best(results, 0.40) for results in results_per_input]
classes_to_labels = utils.get_coco_object_dictionary()
from matplotlib import pyplot as plt

import matplotlib.patches as patches



for image_idx in range(len(best_results_per_input)):

    fig, ax = plt.subplots(1)

    # Show original, denormalized image...

    image = inputs[image_idx] / 2 + 0.5

    ax.imshow(image)

    # ...with detections

    bboxes, classes, confidences = best_results_per_input[image_idx]

    for idx in range(len(bboxes)):

        left, bot, right, top = bboxes[idx]

        x, y, w, h = [val * 300 for val in [left, bot, right - left, top - bot]]

        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')

        ax.add_patch(rect)

        ax.text(x, y, "{} {:.0f}%".format(classes_to_labels[classes[idx] - 1], confidences[idx]*100), bbox=dict(facecolor='white', alpha=0.5))

plt.show()