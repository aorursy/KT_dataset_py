# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os

os.chdir("/kaggle/input/pycoco")
!pip install pycocotools-2.0.1.tar
os.chdir("/kaggle/input/efficientdet-repo")
from PIL import Image

import os

import inference

import tensorflow.compat.v1 as tf

import numpy as np

tf.reset_default_graph()



tf.disable_v2_behavior()



MODEL='efficientdet-d3'

MODEL_DIR='/kaggle/input/effiecientdetd3-10k-epoch-checkpoints'

driver = inference.ServingDriver(MODEL, MODEL_DIR)

driver.build()

def preprocess_predictions(prediction):

    boxes = prediction[0][:, 1:5].astype(int)

    classes = prediction[0][:, 6].astype(int)

    scores = prediction[0][:, 5]

    final_cords=[]

    

    for ind in range(len(boxes)):

        try:

            ymin, xmin ,ymax ,xmax = boxes[ind]



            w,h=int(ymax-ymin), int(xmax-xmin)



            final_cords.append("{:.2f} {} {} {} {}".format(scores[ind],xmin,ymin,w,h))

            

                               

        except:

            

            final_cords.append("")

            

            

    return final_cords

                        



        

    
import cv2

from PIL import Image

import time

import matplotlib.pyplot as plt

%matplotlib inline





data_dir = '/kaggle/input/global-wheat-detection/test'



submission = pd.read_csv('/kaggle/input/global-wheat-detection/sample_submission.csv')





root_image = "/kaggle/input/global-wheat-detection/test/"

test_images = [root_image + f"{img}.jpg" for img in submission.image_id]





submission = []



for imagepath in test_images:

    start=time.time()

    image = np.array(Image.open(imagepath))

    predictions = driver.serve_images([image])

    

    boxes = predictions[0][:, 1:5].astype(int)

    

    for box in boxes:

        ymin, xmin, ymax , xmax=box

        cv2.rectangle(image, (xmin, ymin), (xmax,ymax), (0,0,255),3)

    plt.figure(figsize=(10,10))

    plt.imshow(image)

    plt.show()

        

    

  

    cords=preprocess_predictions(predictions)

#     print(time.time()-start)

    

    prediction_string = " ".join(cords)

    

    submission.append([os.path.basename(imagepath)[:-4],prediction_string])



sample_submission = pd.DataFrame(submission, columns=["image_id","PredictionString"])

sample_submission.to_csv("/kaggle/working/submission.csv", index=False)
sample_submission["PredictionString"]