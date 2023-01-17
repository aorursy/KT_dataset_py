import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
!pip install mtcnn
!pip install tensorflow
from mtcnn import MTCNN
import cv2
import glob
import os.path as path
print(os.listdir("../input"))
data_dir = os.path.join('..','input')
paths_img = glob.glob(os.path.join(data_dir,'newphoto','*.jpg'))
out_dir = "../output"
#paths_img = glob.glob(os.path.join(data_dir,'*.jpg'))
#list_of_images = os.listdir(paths_img)
#os.path.join(data_dir,'2015-11-26-16-25-34-278.jpg')
def generate_image(path_label, out_name):
    image = cv2.cvtColor(cv2.imread(path_label), cv2.COLOR_BGR2RGB)
    detector = MTCNN()
    detector.detect_faces(image)
    [
        {
            'box': [277, 90, 48, 63],
            'keypoints':
            {
                'nose': (303, 131),
                'mouth_right': (313, 141),
                'right_eye': (314, 114),
                'left_eye': (291, 117),
                'mouth_left': (296, 143)
            },
            'confidence': 0.99851983785629272
        }
    ]

    result = detector.detect_faces(image)

    bounding_box = result[0]['box']
    keypoints = result[0]['keypoints']

    cv2.rectangle(image,
                (bounding_box[0], bounding_box[1]),
                (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                (0,155,255),
                2)

    cv2.circle(image,(keypoints['left_eye']), 2, (0,155,255), 2)
    cv2.circle(image,(keypoints['right_eye']), 2, (0,155,255), 2)
    cv2.circle(image,(keypoints['nose']), 2, (0,155,255), 2)
    cv2.circle(image,(keypoints['mouth_left']), 2, (0,155,255), 2)
    cv2.circle(image,(keypoints['mouth_right']), 2, (0,155,255), 2)

    cv2.imwrite(out_dir+out_name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    print(result)
from tqdm import tqdm as tq

for i in tq(range(len(paths_img))):
    try:
        path = paths_img[i]
        args = (path.split("/"))
        out_name = args[len(args)-1]
        generate_image(path, out_name)
    except:
        print("cannot find path or generate image")
        None
for path in enumerate(paths_img):
    #path = '../input/newphoto/3f.jpg'
    print(path)
    img = cv2.imread(os.path.join(paths_img,path))
    image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    detector = MTCNN()
    detector.detect_faces(image)
    [
        {
            'box': [277, 90, 48, 63],
            'keypoints':
            {
                'nose': (303, 131),
                'mouth_right': (313, 141),
                'right_eye': (314, 114),
                'left_eye': (291, 117),
                'mouth_left': (296, 143)
            },
            'confidence': 0.99851983785629272
        }
    ]
import os    
import cv2
path_of_images = "../input/newphoto"
list_of_images = os.listdir(path_of_images)
j = 0
for image in list_of_images:
    img = cv2.imread(os.path.join(path_of_images, image))
    image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    detector = MTCNN()
    result = detector.detect_faces(image)

# Result is an array with all the bounding boxes detected. We know that for 'ivan.jpg' there is only one.
    bounding_box = result[0]['box']
    keypoints = result[0]['keypoints']

    cv2.rectangle(image,
                  (bounding_box[0], bounding_box[1]),
                  (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                  (0,155,255),
                  2)

    cv2.circle(image,(keypoints['left_eye']), 2, (0,155,255), 2)
    cv2.circle(image,(keypoints['right_eye']), 2, (0,155,255), 2)
    cv2.circle(image,(keypoints['nose']), 2, (0,155,255), 2)
    cv2.circle(image,(keypoints['mouth_left']), 2, (0,155,255), 2)
    cv2.circle(image,(keypoints['mouth_right']), 2, (0,155,255), 2)

    cv2.imwrite("%d_drawn.jpg", j, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    j = j + 1
    print(result)