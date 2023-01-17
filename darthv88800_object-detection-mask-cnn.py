import matplotlib.pyplot as plt

from PIL import Image

import torch

import torchvision.transforms as T

import numpy as np

import cv2

import time

import random

import torchvision

import random

import cv2

import gc

import os
print(os.listdir())

if not os.path.exists("HEMA"):

    os.mkdir("HEMA")



if not os.path.exists("HEMA/data"):

    os.mkdir("HEMA/data")



if not os.path.exists("HEMA/frames"):

    os.mkdir("HEMA/frames")

if not os.path.exists("HEMA/vision"):

    os.mkdir("HEMA/vision")



print(os.listdir("HEMA"))



FRAMES_DIR = "HEMA/data/"

PREDICTED_FRAMES_DIR = "HEMA/frames/"

SAVE_VID = 'HEMA/vision/'



VIDEO = "/kaggle/input/West Virginia(720P_HD).mp4"



model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)



COCO_INSTANCE_CATEGORY_NAMES = [

    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',

    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',

    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',

    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',

    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',

    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',

    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',

    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',

    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',

    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',

    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',

    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'

]
model.eval()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = model.to(device)
count = 0

cap = cv2.VideoCapture(VIDEO)

framerate = cap.get(5)



x = 1

while(cap.isOpened()):

    frameId = cap.get(1)

    ret, frame = cap.read()

    if(ret != True):

        break

    #if(frameId % math.floor(framerate) == 0):

    filename = "%d.jpg" % count;count +=1

    cv2.imwrite(FRAMES_DIR+filename,frame)

fps = cap.get(cv2.CAP_PROP_FPS)

print ("FPS: {0}".format(fps))

cap.release()

print("Number of frames",len(os.listdir(FRAMES_DIR)))
def randomColor(image):

    colours = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]

    r = np.zeros_like(image).astype(np.uint8)

    g = np.zeros_like(image).astype(np.uint8)

    b = np.zeros_like(image).astype(np.uint8)

    r[image == 1], g[image == 1], b[image == 1] = colours[random.randrange(0,10)]

    coloured_mask = np.stack([r, g, b], axis=2)

    return coloured_mask



class prediction():

    def __init__(self,image_path,threshold,mask=True):

        

        

        self.mask       = mask

        self.image      = image_path

        self.threshold  = threshold

        self.img        = Image.open(self.image)

        self.transform  = T.Compose([T.ToTensor()])

        self.img        = self.transform(self.img).to(device)

        self.start      = time.time()

        self.pred       = model([self.img])

        self.end        = time.time() - self.start

        self.pred_score = list(self.pred[0]['scores'].detach().cpu().numpy())

        #print("-"*50)

        #print(self.pred_score)

        

        #print("-"*50)

        try:

            self.pred_t     = [self.pred_score.index(x) for x in self.pred_score if x> threshold][-1]

            self.shortlist  = [x for x in self.pred_score if x > threshold]

        except IndexError:

            self.pred_t = self.pred_score.index(max(self.pred_score))

            self.shortlist = [max(self.pred_score)]

            #print("DD:",self.pred_t)

        self.pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(self.pred[0]['labels'].cpu().numpy())]

        if(self.mask == True):

            self.masks  = (self.pred[0]['masks']>threshold).squeeze().detach().cpu().numpy()

            self.masks  = self.masks[:self.pred_t+1]

        

        

        self.pred_boxes = [[(i[0],i[1]),(i[2],i[3])] for i in list(self.pred[0]['boxes'].detach().cpu().numpy())]

        self.pred_boxes = self.pred_boxes[:self.pred_t+1]

        self.pred_class = self.pred_class[:self.pred_t+1]



    def combine(self,show_image=False,rect_th=3,text_size=1.5,text_th=3,save_img=False):

        

        self.cv2_img   = cv2.imread(self.image)

        self.cv2_img   = cv2.cvtColor(self.cv2_img,cv2.COLOR_BGR2RGB)

        if(self.mask== True):

            

            for i in range(len(self.masks)):

                rgb_mask = randomColor(self.masks[i])

                self.cv2_img = cv2.addWeighted(self.cv2_img,1,rgb_mask,0.5,0)

                boxes = self.pred_boxes[i]

                pred_name = self.pred_class[i]

                score = (self.shortlist)[i]

                # img, start point, end point, color, thickness

                cv2.rectangle(self.cv2_img, boxes[0], boxes[1],color=(0, 255, 0), thickness=rect_th)

                cv2.putText(self.cv2_img,pred_name+": "+str(score)+"%", boxes[0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)

            

            

        else:

            self.cv2_img = self.cv2_img

            for i in range(len(pred.pred_class)):

                #label_num = pred.pred_class.index(i)

                boxes = self.pred_boxes[i]

                pred_name = self.pred_class[i]

                score = (self.shortlist)[i]

                

                # img, start point, end point, color, thickness

                cv2.rectangle(self.cv2_img, boxes[0], boxes[1],color=(0, 255, 0), thickness=rect_th)

                cv2.putText(self.cv2_img,pred_name+": "+str(score)+"%", boxes[0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)

                

        if(show_image==True):

            print(self.end)

            plt.figure(figsize=(20,30))

            plt.imshow(self.cv2_img)

            plt.xticks([])

            plt.yticks([])

            plt.show()



        return self.cv2_img

    
! wget https://brooklynreporter.com/wp-content/uploads/2019/11/News-DOT-plans-traffic-direction-change-By-Paula-Katinas-1024x768.jpg lmao.jpg
#pred = prediction('News-DOT-plans-traffic-direction-change-By-Paula-Katinas-1024x768.jpg',0.7)

#pred.combine(show_image=True)

#plt.imshow(pred.combine())
"""device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

model.eval()

model = model.to(device)

img = Image.open("News-DOT-plans-traffic-direction-change-By-Paula-Katinas-1024x768.jpg")

transforms = T.Compose([T.ToTensor()])

img = transforms(img).to(device)



t0 = time.time()

pred = model([img])

print(time.time() - t0)"""
import tqdm

class vision():

    def __init__(self,frames_dir):

        self.frames_dir = frames_dir        

        self.dirc = os.listdir(self.frames_dir)

        for i in tqdm.tqdm(range(len(self.dirc))):

            ID = str(i)+".jpg"

            full_path = os.path.join(self.frames_dir,ID)

            pred = prediction(full_path,0.65).combine()

            name = os.path.join(PREDICTED_FRAMES_DIR,ID)

            self.save(name,pred)

        

    

    def save(self,name,img):

        cv2.imwrite(name,img)
vision(FRAMES_DIR)
img_array = []

vision_path = PREDICTED_FRAMES_DIR

frames = os.listdir(vision_path)

for frame in range(len(frames)):

    ID = str(frame)+".jpg"

    full_vision_path = os.path.join(vision_path,ID)

    print(full_vision_path)

    img = cv2.imread(full_vision_path)

    height, width, layers = img.shape

    size = (width, height)

    img_array.append(img)

vid = cv2.VideoWriter("AI_virginia.avi",cv2.VideoWriter_fourcc(*"DIVX"),29.885057471264368,size)

for i in range(len(img_array)):

    vid.write(img_array[i])

cv2.destroyAllWindows()

vid.release()
img = Image.open("HEMA/frames/779.jpg")

plt.imshow(img)
from IPython.display import Video



Video("")