# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

# #         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import time

import cv2

import os







def load_network(weightsPath, configPath):



    print("[INFO] loading YOLO from disk...")

    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    return net



def get_predictions(labelsPath,img, net):

  

    LABELS = open(labelsPath).read().strip().split("\n")





    a=0

    image = img

    (H, W) = image.shape[:2]



    # determine only the *output* layer names that we need from YOLO

    ln = net.getLayerNames()

    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]



    # construct a blob from the input image and then perform a forward

    # pass of the YOLO object detector, giving us our bounding boxes and

    # associated probabilities

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),

    swapRB=True, crop=False)

    net.setInput(blob)

    start = time.time()

    layerOutputs = net.forward(ln)

    end = time.time()



    # show timing information on YOLO

    #   print("[INFO] YOLO took {:.6f} seconds".format(end - start))





    # initialize our lists of detected bounding boxes, confidences, and

    # class IDs, respectively

    boxes = []

    confidences = []

    classIDs = []





    # loop over each of the layer outputs

    for output in layerOutputs:

        # loop over each of the detections

        for detection in output:

            

            

          # extract the class ID and confidence (i.e., probability) of

          # the current object detection

            scores = detection[5:]

            classID = np.argmax(scores)

            confidence = scores[classID]



          # filter out weak predictions by ensuring the detected

          # probability is greater than the minimum probability

            if confidence > 0.5:

                # scale the bounding box coordinates back relative to the

                # size of the image, keeping in mind that YOLO actually

                # returns the center (x, y)-coordinates of the bounding

                # box followed by the boxes' width and height

                box = detection[0:4] * np.array([W, H, W, H])

                (centerX, centerY, width, height) = box.astype("int")



                # use the center (x, y)-coordinates to derive the top and

                # and left corner of the bounding box

                x = int(centerX - (width / 2))

                y = int(centerY - (height / 2))



                # update our list of bounding box coordinates, confidences,

                # and class IDs

                boxes.append([x, y, int(width), int(height)])

                confidences.append(float(confidence))

                classIDs.append(classID)



        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,0.5)





        cords=[]

        find_labels=[]

        # ensure at least one detection exists







    if len(idxs)>0:

        

        # loop over the indexes we are keepin

        for i in idxs.flatten():

            



            (x,y)=(boxes[i][0], boxes[i][1])

            (w,h)=(boxes[i][2], boxes[i][3])

            score=confidences[i]



            text = "{}:{:.2f}".format(LABELS[classIDs[i]],score)

            a,b=x,y

            c,d=x+w, y+h



            cords.append("{:.2f} {} {} {} {}".format(score,abs(int(a)),abs(int(b)),abs(int(w)),abs(int(h))))





            cv2.rectangle(image, (a,b),(c,d),(255,0,0),2)

            cv2.putText(image, text, (a, b - 5), cv2.FONT_HERSHEY_SIMPLEX,1, (255,255,0), 3)

            find_labels.append(text)



        return image,find_labels,cords



    else:



        return [],[],[]

      





net=load_network("../input/trained-weights-and-cfg/yolov3_2000.weights", "../input/trained-weights-and-cfg/yolov3.cfg")



from tqdm import tqdm

import cv2

import matplotlib.pyplot as plt



data_dir = '../input/global-wheat-detection/test'



submission = pd.read_csv('../input/global-wheat-detection/sample_submission.csv')





root_image = "../input/global-wheat-detection/test/"

test_images = [root_image + f"{img}.jpg" for img in submission.image_id]





submission = []



for imagepath in test_images:

    im=cv2.imread(imagepath)

    image, labels , cords= get_predictions("../input/trained-weights-and-cfg/labels.name",im,net)

    prediction_string = " ".join(cords)

#     plt.figure(figsize=(10,10))

#     plt.imshow(image)

#     plt.show()

    

    submission.append([os.path.basename(imagepath)[:-4],prediction_string])



sample_submission = pd.DataFrame(submission, columns=["image_id","PredictionString"])



sample_submission.to_csv('submission.csv', index=False)