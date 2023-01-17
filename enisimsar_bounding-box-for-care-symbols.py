import sys

sys.path.append('../input/yolov3-for-care-symbols/')
from yolo import YOLO

from PIL import Image



import matplotlib.pyplot as plt



%matplotlib inline
FLAGS = {

        "model_path": '../input/yolov3-for-care-symbols/best_model.h5',

        "anchors_path": '../input/yolov3-for-care-symbols/yolo_anchors.txt',

        "classes_path": '../input/yolov3-for-care-symbols/classes.txt',

        "score" : 0.3,

        "iou" : 0.45,

        "model_image_size" : (416, 416),

        "gpu_num" : 1,

    }



yolo_obj = YOLO(**(FLAGS))
IMG_PATH = '../input/identification-care-symbols/test/Test/test_147.jpg'



img = yolo_obj.detect_image(Image.open(IMG_PATH), return_image=True)



plt.figure(figsize=(20,10))



plt.imshow(img)
import glob



from tqdm import tqdm



import numpy as np



import os
file_dict = {}

for v, k in zip(glob.glob('../input/identification-care-symbols/train/Train/*'), map(lambda x: x.split('/')[-1].split('.')[0], glob.glob('../input/identification-care-symbols/train/Train/*'))):

    file_dict[k] = v
c = 0



os.makedirs('train_cropped/', exist_ok=True)



for f in tqdm(list(file_dict.values())[:20]):

    img = Image.open(f)

    coor = yolo_obj.detect_image(img)

    f_n = f"train_cropped/{f.split('/')[-1]}"

    

    try:

        coord = coor[0][0].astype(int)



        coord[0] -= 30

        coord[2] += 30



        coord[1] -= 30

        coord[3] += 30

        

        coord[coord < 0] = 0

        

        img = np.array(img)

        

        img = img[coord[0]:coord[2], coord[1]:coord[3]]

        

    except Exception as e:

        c += 1

        img = np.array(img)

        pass

        

    plt.imsave(f_n, img)
print(f"total images: {len(file_dict)}, {c} of them cannot be cropped")
file_dict = {}

for v, k in zip(glob.glob('../input/identification-care-symbols/test/Test/*'), map(lambda x: x.split('/')[-1].split('.')[0], glob.glob('../input/identification-care-symbols/test/Test/*'))):

    file_dict[k] = v
c = 0



os.makedirs('test_cropped/', exist_ok=True)



for f in tqdm(list(file_dict.values())[:20]):

    img = Image.open(f)

    coor = yolo_obj.detect_image(img)

    f_n = f"test_cropped/{f.split('/')[-1]}"

    

    try:

        coord = coor[0][0].astype(int)



        coord[0] -= 30

        coord[2] += 30



        coord[1] -= 30

        coord[3] += 30

        

        coord[coord < 0] = 0

        

        img = np.array(img)

        

        img = img[coord[0]:coord[2], coord[1]:coord[3]]

        

    except Exception as e:

        c += 1

        img = np.array(img)

        pass

        

    plt.imsave(f_n, img)
print(f"total images: {len(file_dict)}, {c} of them cannot be cropped")