# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import os

import sys

import random



import torch

import torchvision.transforms as transforms



print("Using PyTorch", torch.__version__)
# import module we'll need to import our custom module

from shutil import copyfile



# copy our file into the working directory (make sure it has .py suffix)

copyfile(src = "../input/object-detection-with-yolov3-pytorch/yolo_pytorch/utils/datasets.py", dst = "../working/datasets.py")

copyfile(src = "../input/object-detection-with-yolov3-pytorch/yolo_pytorch/utils/torch_utils.py", dst = "../working/torch_utils.py")

copyfile(src = "../input/object-detection-with-yolov3-pytorch/yolo_pytorch/utils/utils.py", dst = "../working/utils.py")

copyfile(src = "../input/object-detection-with-yolov3-pytorch/yolo_pytorch/utils/parse_config.py", dst = "../working/parse_config.py")

copyfile(src = "../input/object-detection-with-yolov3-pytorch/yolo_pytorch/models.py", dst = "../working/models.py")





# Custom functions 

print('Uploading custom functions...')

import models as models

from utils import *

print('Done.')

# Set up model

print('Setting up the model...')

model_config = '../input/object-detection-with-yolov3-pytorch/yolo_pytorch/yolov3.cfg'

img_size = 416

home = '../input/object-detection-with-yolov3-pytorch'

print('...Uploading the weights...')

weights = os.path.join(home, "yolov3.weights")

print('Upload done. Continuing the set up...\n')



print('Informations about the model: \n')

model = models.Darknet(model_config, img_size)

models.load_darknet_weights(model, weights)

print(model)

print('Set up done. \n')
print('Initializing function 1...')



def detect_objects(model, img):

    

    # Use GPU if available

    if torch.cuda.is_available():

        model.cuda()

        Tensor = torch.cuda.FloatTensor

    else:

        Tensor = torch.FloatTensor

    

    # Set the model to evaluation mode

    model.eval()

    

    # Get scaled width and height

    ratio = min(img_size/img.size[0], img_size/img.size[1])

    imw = round(img.size[0] * ratio)

    imh = round(img.size[1] * ratio)



    # Transform the image for prediction

    img_transforms = transforms.Compose([

         transforms.Resize((imh, imw)),

         transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),

                        (128,128,128)),

         transforms.ToTensor(),

         ])

    

    # convert image to a Tensor

    image_tensor = img_transforms(img).float()

    image_tensor = image_tensor.unsqueeze_(0)

    

    # Use the model to detect objects in the image

    with torch.no_grad():

        detections = model(image_tensor)

        # Eliminate duplicates with non-max suppression

        detections = non_max_suppression(detections, 0.8, 0.4)

    return detections[0]



print('Function 1 initialized. \,')



print('Initializing function 2...')



def show_objects(img, detections):

    import random

    import matplotlib.patches as patches

    import matplotlib.pyplot as plt

    

    # Get bounding-box colors

    cmap = plt.get_cmap('tab20b')

    colors = [cmap(i) for i in np.linspace(0, 1, 20)]



    img = np.array(img)

    plt.figure()

    fig, ax = plt.subplots(1, figsize=(12,9))

    ax.imshow(img)



    pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))

    pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))

    unpad_h = img_size - pad_y

    unpad_w = img_size - pad_x



    if detections is not None:

        # process each instance of each class that was found

        classes = load_classes('../input/object-detection-with-yolov3-pytorch/yolo_pytorch/coco.names')

        unique_labels = detections[:, -1].cpu().unique()

        n_cls_preds = len(unique_labels)

        bbox_colors = random.sample(colors, n_cls_preds)

        # browse detections and draw bounding boxes

        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

            # Get the class name

            predicted_class = classes[int(cls_pred)]

            

            #We'll display the class name and probability

            label = '{} {:.2f}'.format(predicted_class, cls_conf)

            

            # Set the box dimensions

            box_h = ((y2 - y1) / unpad_h) * img.shape[0]

            box_w = ((x2 - x1) / unpad_w) * img.shape[1]

            y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]

            x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]

            

            # Add a box with the color for this class

            color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]

            bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor='none')

            ax.add_patch(bbox)

            plt.text(x1, y1, s=label, color='white', verticalalignment='top',

                    bbox={'color': color, 'pad': 0})

    plt.axis('off')



    plt.show()



print('Function 2 initialized. \,')



print("Functions ready")
import os

from PIL import Image



test_dir = "../input/object-detection-with-yolov3-pytorch/data/object_detection"

for image_file in os.listdir(test_dir):

    

    # Load image

    img_path = os.path.join(test_dir, image_file)

    image = Image.open(img_path)



    # Detect objects in the image

    detections = detect_objects(model, image)



    # How many objects did we detect?

    print('Found {} objects in {}'.format(len(detections), image_file))



    # Display the image with bounding boxes

    show_objects(image, detections)