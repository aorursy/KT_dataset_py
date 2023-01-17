from tqdm import tqdm

import numpy as np

import torch

import pandas as pd

from torchvision import transforms

from PIL import Image

import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import os



device = torch.device('cuda:0')
def mask_to_rle(img, width, height):

    rle = []

    lastColor = 0

    currentPixel = 0

    runStart = -1

    runLength = 0



    for x in range(width):

        for y in range(height):

            currentColor = img[x][y]

            if currentColor != lastColor:

                if currentColor == 1:

                    runStart = currentPixel

                    runLength = 1

                else:

                    rle.append(str(runStart))

                    rle.append(str(runLength))

                    runStart = -1

                    runLength = 0

                    currentPixel = 0

            elif runStart > -1:

                runLength += 1

            lastColor = currentColor

            currentPixel+=1

    return " " + " ".join(rle)
num_classes = 2



sample_df = pd.read_csv("../input/siim-acr-pneumothorax-segmentation/sample_submission.csv")

sample_df = sample_df.drop_duplicates('ImageId', keep='last').reset_index(drop=True)



model_ft = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

in_features = model_ft.roi_heads.box_predictor.cls_score.in_features

model_ft.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

in_features_mask = model_ft.roi_heads.mask_predictor.conv5_mask.in_channels

hidden_layer = 256

model_ft.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)



model_ft.load_state_dict(torch.load("../input/train-your-own-mask-rcnn/model.bin"))

model_ft = model_ft.to(device)



for param in model_ft.parameters():

    param.requires_grad = False



model_ft.eval()
sub = pd.read_csv('/kaggle/input/siim-acr-pneumothorax-segmentation/sample_submission.csv')



tmp = sub.groupby('ImageId')['ImageId'].count().reset_index(name='N')

tmp = tmp.loc[tmp.N > 1] #find image id's with more than 1 row -> has pneumothorax mask!



possible_patients = tmp.ImageId.values
tt = transforms.ToTensor()

sublist = []

counter = 0

for index, row in tqdm(sample_df.iterrows(), total=len(sample_df)):

    rle = " -1"

    image_id = row['ImageId']

    if image_id in possible_patients:

        img_path = os.path.join('../input/siim-png-images/input/test_png', image_id + '.png')



        img = Image.open(img_path).convert("RGB")

        width, height = img.size

        img = img.resize((256, 256), resample=Image.BILINEAR)

        img = tt(img)

        result = model_ft([img.to(device)])[0]

        if len(result["masks"]) > 0:

            if result['scores'][0] > 0.7:

                counter += 1

                res = transforms.ToPILImage()(result["masks"][0].permute(1, 2, 0).cpu().numpy())

                res = np.asarray(res.resize((width, height), resample=Image.BILINEAR))

                res = (res[:, :] * 255. > 127).astype(np.uint8).T

                rle = mask_to_rle(res, width, height)

    sublist.append([image_id, rle])



submission_df = pd.DataFrame(sublist, columns=sample_df.columns.values)

submission_df.to_csv("submission.csv", index=False)

print(counter)