import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

import cv2
import pydicom
! pip install efficientnet_pytorch
def get_img(path):
    d = pydicom.dcmread(path)
    # normlize and resize the image
    return cv2.resize((d.pixel_array - d.pixel_array.min()) / (d.pixel_array.max() - d.pixel_array.min()), (512, 512))
ROOT = "../input/osic-pulmonary-fibrosis-progression"
os.listdir(ROOT)
# load a demo image
demo_img_path = os.path.join(ROOT, "train", "ID00007637202177411956430", "1.dcm")
demo_img = get_img(demo_img_path)
demo_img.shape
# get the efficientNet model
from efficientnet_pytorch import EfficientNet
model = EfficientNet.from_pretrained('efficientnet-b6')
model.cuda()
import torch
# convert the image to the input tensor, the input must be a 3-channel tensor with float dtype
demo_img_ts = torch.from_numpy(demo_img).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(dtype=torch.float).cuda()
demo_img_ts.size()
demo_feature = model.extract_features(demo_img_ts)
demo_feature.size()