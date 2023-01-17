!pip install /kaggle/input/packages/webcolors-1.11.1-py3-none-any.whl
import os, sys

sys.path.insert(0, "/kaggle/input/yaefficientdetpytorch/Yet-Another-EfficientDet-Pytorch")

import torch

import numpy as np 

import cv2

import matplotlib.pyplot as plt

import pandas as pd

from backbone import EfficientDetBackbone

from efficientdet.utils import BBoxTransform, ClipBoxes

from utils.utils import preprocess, postprocess, invert_affine

from torch.backends import cudnn

from torch.utils.data import Dataset, DataLoader

from glob import glob
DATA_ROOT_PATH = '../input/global-wheat-detection/test'
compound_coef = 4

force_input_size = None

test_images = [path.split('/')[-1][:-4] for path in glob(f'{DATA_ROOT_PATH}/*.jpg')]

test_images_paths = [os.path.join(DATA_ROOT_PATH, f"{img}.jpg") for img in test_images]

# img_path = [os.path.join(INPUT_DIR, "test", img_path) for img_path in os.listdir(os.path.join(INPUT_DIR, "test"))]

# IMG_PATH = os.path.join(CONFIG.CFG.DATA.BASE, "test", "2fd875eaa.jpg")



threshold = 0.2

iou_threshold = 0.2



use_cuda = True

use_float16 = False

cudnn.fastest = True

cudnn.benchmark = True



obj_list = ['wheat']
test_images_paths
%%capture

# tf bilinear interpolation is different from any other's, just make do

input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]

input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size



model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),



                             # replace this part with your project's anchor config

                             ratios=[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)],

                             scales=[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])



model.load_state_dict(torch.load('/kaggle/input/efficientdet4/efficientdet-d4_14_20235.pth'))

model.requires_grad_(False)

model.eval()

if use_cuda:

    model = model.cuda()

if use_float16:

    model = model.half()
class WheatDataset(Dataset):

    def __init__(self, img_paths):

        self.img_paths = img_paths

    

    def __len__(self):

        return len(self.img_paths)

    

    def __getitem__(self, idx):

        curr_img = self.img_paths[idx]

        _, framed_img, framed_meta = preprocess(curr_img, max_size=input_size)

        return {

            "image": torch.from_numpy(framed_img[0]).to(torch.float32 if not use_float16 else torch.float16),

            "framed_meta": torch.tensor(list(framed_meta[0])),

            "image_id": curr_img.split("/")[-1][:-4]

        }
test_dataset = WheatDataset(test_images_paths)
test_data_loader = DataLoader(

    test_dataset,

    batch_size=2,

    drop_last=False,

    shuffle=False,

)
results = []

with torch.no_grad():

    for img in test_data_loader:

        images = img['image'].permute(0, 3, 1, 2).cuda()

        framed_meta = img['framed_meta']

        image_id = img['image_id']

        

        features, regression, classification, anchors = model(images)



        regressBoxes = BBoxTransform()

        clipBoxes = ClipBoxes()



        out = postprocess(images,

                          anchors, regression, classification,

                          regressBoxes, clipBoxes,

                          threshold, iou_threshold)

        out = invert_affine(framed_meta, out)

        

        for i, out_dict in enumerate(out):

            if len(out_dict['rois']) == 0:

                result = {

                    'image_id': image_id[i],

                    'PredictionString': ""

                }

                results.append(result)

            else:            

                result = {

                    'image_id': image_id[i],

                    'PredictionString': ""

                }

                

                pred_strings = []

                

                for j in range(len(out_dict['rois'])):

                    score = float(out_dict['scores'][j])

                    (x1, y1, x2, y2) = out_dict['rois'][j].astype(np.int)

                    pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(score, x1, y1, x2-x1, y2-y1))

                result['PredictionString'] = " ".join(pred_strings)

            results.append(result)
results
test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])

test_df.head()
test_df.to_csv("submission.csv", index=False)