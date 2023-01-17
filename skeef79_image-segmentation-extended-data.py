# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

file_list = []

import os

for dirname, _, filenames in os.walk('/kaggle/input/plantpathology-apple-dataset'):

    for filename in filenames:

        file_list.append(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install -U torch==1.5 torchvision==0.6 -f https://download.pytorch.org/whl/cu101/torch_stable.html 

!pip install cython pyyaml==5.1

!pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

import torch, torchvision

print(torch.__version__, torch.cuda.is_available())

!gcc --version
# install detectron2:

!pip install detectron2==0.1.3 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/index.html
!pip list
import sys

#sys.path.append('/content/detectron2_repo')

import os

import numpy as np



import math  

import detectron2 

from detectron2.engine import DefaultTrainer

from detectron2.config import get_cfg

from detectron2 import model_zoo

from detectron2.engine import DefaultPredictor

from detectron2.config import get_cfg

from detectron2.utils.visualizer import Visualizer

from detectron2.data import MetadataCatalog

from PIL import Image

import cv2



import requests



def download_file_from_google_drive(id, destination):

    URL = "https://docs.google.com/uc?export=download"



    session = requests.Session()



    response = session.get(URL, params = { 'id' : id }, stream = True)

    token = get_confirm_token(response)



    if token:

        params = { 'id' : id, 'confirm' : token }

        response = session.get(URL, params = params, stream = True)



    save_response_content(response, destination)    



def get_confirm_token(response):

    for key, value in response.cookies.items():

        if key.startswith('download_warning'):

            return value



    return None



def save_response_content(response, destination):

    CHUNK_SIZE = 32768



    with open(destination, "wb") as f:

        for chunk in response.iter_content(CHUNK_SIZE):

            if chunk: # filter out keep-alive new chunks

                f.write(chunk)
## download the pretrained weights for leaf segmentation.



file_id = '17AHanttKcR9B4A0m7QZqwAvaWxGrYYQp'

destination = './model.pth'

download_file_from_google_drive(file_id, destination)
#get the predictor

def get_predictor():

    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

    cfg.DATASETS.TRAIN = ()

    cfg.DATALOADER.NUM_WORKERS = 16



    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (8)  # faster, and good enough for this toy dataset

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # 3 classes (data, fig, hazelnut)



    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "/kaggle/working/model.pth")

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model

    predictor = DefaultPredictor(cfg)

    return predictor
#function to get the leaf image.



def get_cropped_leaf(img,predictor,return_mapping=False,resize=None):

    #convert to numpy    

    img = np.array(img)[:,:,::-1]

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    

    #get prediction

    outputs = predictor(img)

    

    #get boxes and masks

    ins = outputs["instances"]

    pred_masks = ins.get_fields()["pred_masks"]

    boxes = ins.get_fields()["pred_boxes"]    

    

    #get main leaf mask if the area is >= the mean area of boxes and is closes to the centre 

    

    masker = pred_masks[np.argmin([calculateDistance(x[0], x[1], int(img.shape[1]/2), int(img.shape[0]/2)) for i,x in enumerate(boxes.get_centers()) if (boxes[i].area()>=torch.mean(boxes.area()).to("cpu")).item()])].to("cpu").numpy().astype(np.uint8)



    #mask image

    mask_out = cv2.bitwise_and(img, img, mask=masker)

    

    #find contours and boxes

    contours, hierarchy = cv2.findContours(masker.copy() ,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contour = contours[np.argmax([cv2.contourArea(x) for x in contours])]

    rotrect = cv2.minAreaRect(contour)

    box = cv2.boxPoints(rotrect)

    box = np.int0(box)

    



    #crop image

    cropped = get_cropped(rotrect,box,mask_out)



    #resize

    rotated = MakeLandscape()(Image.fromarray(cropped))

    

    if not resize == None:

        resized = ResizeMe((resize[0],resize[1]))(rotated)

    else:

        resized = rotated

        

    if return_mapping:

        img = cv2.drawContours(img, [box], 0, (0,0,255), 10)

        img = cv2.drawContours(img, contours, -1, (255,150,), 10)

        return resized, ResizeMe((int(resize[0]),int(resize[1])))(Image.fromarray(img))

    

    return resized



#function to crop the image to boxand rotate



def get_cropped(rotrect,box,image):

    

    width = int(rotrect[1][0])

    height = int(rotrect[1][1])



    src_pts = box.astype("float32")

    # corrdinate of the points in box points after the rectangle has been

    # straightened

    dst_pts = np.array([[0, height-1],

                        [0, 0],

                        [width-1, 0],

                        [width-1, height-1]], dtype="float32")



    # the perspective transformation matrix

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)



    # directly warp the rotated rectangle to get the straightened rectangle

    warped = cv2.warpPerspective(image, M, (width, height))

    return warped



def calculateDistance(x1,y1,x2,y2):  

    dist = math.hypot(x2 - x1, y2 - y1)

    return dist  

#image manipulations 



class ResizeMe(object):

    #resize and center image in desired size 

    def __init__(self,desired_size):

        

        self.desired_size = desired_size

        

    def __call__(self,img):

    

        img = np.array(img).astype(np.uint8)

        

        desired_ratio = self.desired_size[1] / self.desired_size[0]

        actual_ratio = img.shape[0] / img.shape[1]



        desired_ratio1 = self.desired_size[0] / self.desired_size[1]

        actual_ratio1 = img.shape[1] / img.shape[0]



        if desired_ratio < actual_ratio:

            img = cv2.resize(img,(int(self.desired_size[1]*actual_ratio1),self.desired_size[1]),None,interpolation=cv2.INTER_AREA)

        elif desired_ratio > actual_ratio:

            img = cv2.resize(img,(self.desired_size[0],int(self.desired_size[0]*actual_ratio)),None,interpolation=cv2.INTER_AREA)

        else:

            img = cv2.resize(img,(self.desired_size[0], self.desired_size[1]),None, interpolation=cv2.INTER_AREA)

            

        h, w, _ = img.shape



        new_img = np.zeros((self.desired_size[1],self.desired_size[0],3))

        

        hh, ww, _ = new_img.shape



        yoff = int((hh-h)/2)

        xoff = int((ww-w)/2)

        

        new_img[yoff:yoff+h, xoff:xoff+w,:] = img



        

        return Image.fromarray(new_img.astype(np.uint8))



class MakeLandscape():

    #flip if needed

    def __init__(self):

        pass

    def __call__(self,img):

        

        if img.height > img.width:

            img = np.rot90(np.array(img))

            img = Image.fromarray(img)

        return img

len(file_list)
predictor = get_predictor()
img, img1 = get_cropped_leaf(Image.open("/kaggle/input/plantpathology-apple-dataset/images/Train_128.jpg"),predictor,return_mapping=True,resize = (800,int(800)))
img
if img1.height > img1.width:

    img1 = np.rot90(np.array(img1))

    img1 = Image.fromarray(img1)

img1
import matplotlib.pyplot as plt

kek = np.array(img)

kek_mask = np.array(img1)

plt.imshow(kek)

plt.show()

plt.imshow(kek_mask)

plt.show()



print(kek.dtype)

print(kek_mask.shape)
final_image = []



for x in range(75):

    #select random image

    file_loc = file_list[np.random.randint(0,len(file_list))]

    #get outputs from predictor

    img, img1 = get_cropped_leaf(Image.open(file_loc),predictor,return_mapping=True,resize = (600,int(600*.65)))

    #stack horizontally

    stacked = np.hstack([img,img1])

    #append images

    final_image.append(stacked)

import matplotlib.pyplot as plt

%matplotlib inline
for x in final_image:

    fig = plt.figure(figsize=(20,10))

    plt.imshow(x)
os.makedirs('images')
!rmdir 'output'
path = 'images/'



from tqdm import tqdm



for i in range(1,len(file_list)):

    img,img1 = get_cropped_leaf(Image.open(file_list[i]),predictor,return_mapping=True,resize = (800,int(800)))

    kek = os.path.split(file_list[i])[1]

    img.save(path+kek) 

    
im = Image.open('images/train_add_127.jpg')

im
print(os.path.split(file_list[1])[1])
