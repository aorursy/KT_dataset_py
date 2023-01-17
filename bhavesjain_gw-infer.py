import pandas as pd

import numpy as np

import cv2

import os



import torch

import torchvision

from torchvision import models,transforms

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchvision.models.detection import FasterRCNN

from torchvision.models.detection.rpn import AnchorGenerator

from torch.utils.data import DataLoader, Dataset

from torch.utils.data.sampler import SequentialSampler



from PIL import Image

from matplotlib import pyplot as plt

# import imgaug as ia

# import imageio

# from imgaug import augmenters as iaa

# from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
dir_test = "/kaggle/input/global-wheat-detection/test"
model = torch.load("/kaggle/input/model-colab/fasterrcnn_resnet50_fpn0.pth",map_location='cpu')
# model = torch.load("/kaggle/input/gwd-augs-out/fasterrcnn_resnet50_fpn_new0.pth",map_location='cpu')
# model = torch.load("/kaggle/input/gwd-augmentations/fasterrcnn_resnet50_fpn_new.pth",map_location='cpu')
# model = torch.load("/kaggle/input/gwd-train/fasterrcnn_resnet50_fpn_new.pth",map_location = 'cpu')
preprocess = transforms.Compose([transforms.ToTensor()])
from PIL import Image



kelvin_table = {

    1000: (255,56,0),

    1500: (255,109,0),

    2000: (255,137,18),

    2500: (255,161,72),

    3000: (255,180,107),

    3500: (255,196,137),

    4000: (255,209,163),

    4500: (255,219,186),

    5000: (255,228,206),

    5500: (255,236,224),

    6000: (255,243,239),

    6500: (255,249,253),

    7000: (245,243,255),

    7500: (235,238,255),

    8000: (227,233,255),

    8500: (220,229,255),

    9000: (214,225,255),

    9500: (208,222,255),

    10000: (204,219,255),

    15000: (179, 204, 255),

    20000: (168 ,197, 255)}





def convert_temp(image, temp):

    r, g, b = kelvin_table[temp]

    matrix = ( r / 255.0, 0.0, 0.0, 0.0,

               0.0, g / 255.0, 0.0, 0.0,

               0.0, 0.0, b / 255.0, 0.0 )

    return image.convert('RGB', matrix)



# for file in os.listdir(dir_test):

#     a = cv2.imread(os.path.join(dir_test,file))

#     a = cv2.cvtColor(a,cv2.COLOR_BGR2RGB).astype(np.float32)

# #     a = Image.open(os.path.join(dir_test,file))

#     fig,(ax1,ax2) = plt.subplots(1,2,figsize = (16,8))

#     ax1.imshow(a)

#     img = Image.fromarray(np.uint8(a))

#     b = convert_temp(img,20000)

#     ax2.imshow(b)

#     plt.show()
model.eval()

color = (220,0,0)

results = []

for img_file in os.listdir(dir_test):

    result = []

    img = cv2.imread(os.path.join(dir_test,img_file))

    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB).astype(np.float32)

    img = Image.fromarray(np.uint8(img))

    img = convert_temp(img,15000)

    img = np.array(img).astype(np.float32)

    img = img/255.0

    img_t = preprocess(img)

    img_t = img_t.unsqueeze(0)

    pred = model(img_t)

    bboxes = pred[0]['boxes'].cpu().detach().numpy()

    bscores = pred[0]['scores'].cpu().detach().numpy()

    img_name = img_file.split('.')[:-1]

    for i in range(len(bboxes)):

        if bscores[i]>0.5:

            result.append((bscores[i],bboxes[i]))

    results.append((str(img_name[0]),result))

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    for i in result:

        if i[0]>0.5:

            box = i[1]

            cv2.rectangle(img,(int(box[0]), int(box[1])),(int(box[2]), int(box[3])),color, 5)

    ax.set_axis_off()

    ax.imshow(img)

    plt.show()
res = []

for result in results:

#     print(result[0],end='')

    pred_str = []

    for box in result[1]:

        pred_str.append(box[0])

        pred_str.append(box[1][0])

        pred_str.append(box[1][1])

        pred_str.append(box[1][2]-box[1][0])

        pred_str.append(box[1][3]-box[1][1])

    pred = {}

    pred['image_id'] = str(result[0])

    pred['PredictionString'] = ' '.join(str(i) for i in pred_str)

    res.append(pred)
test_df = pd.DataFrame(res, columns=['image_id', 'PredictionString'])

print(test_df)
test_df.to_csv("/kaggle/working/submission.csv",index=False)
!cat submission.csv