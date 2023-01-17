import os

import shutil

import pandas as pd

import numpy as np

import cv2

import glob

import time

import matplotlib.pyplot as plt
photos = glob.glob('../input/global-wheat-detection/train/*.jpg')

len(photos)
df = pd.read_csv('../input/wheat-data/train_2.csv')

df.head()
len(df['image_id'].unique())
df = pd.read_csv('../input/wheat-data/labels.csv')

df.head()

full_path_to_csv = '../input/wheat-data'

full_path_to_train_images = '../input/global-wheat-detection/train'

full_path_to_test_images = '../input/global-wheat-detection/test'
classes = pd.read_csv('../input/wheat-data/labels.csv',usecols=[0,1],header=None)

classes
labels = ['wheat']

encrypted_strings = []



for v in labels:

  sub_classes = classes.loc[classes[1]==v]

  print(sub_classes)



  e = sub_classes.iloc[0][0]

  print(e)



  encrypted_strings.append(e)



print()

print(labels)

print(encrypted_strings)
annotations = pd.read_csv('../input/wheat-data/train_2.csv',usecols=['image_id',

                                                                    'label_name',

                                                                    'xmin',

                                                                    'ymin',

                                                                    'width.1',

                                                                    'height.1'])

annotations.head()
sub_ann = annotations.loc[annotations['label_name'].isin(encrypted_strings)].copy()

print(sub_ann.head())
sub_ann['class_number'] = ''

sub_ann['center x'] = ''

sub_ann['center y'] = ''

sub_ann['xmax'] = ''

sub_ann['ymax'] = ''



for i in range(len(encrypted_strings)):

  sub_ann.loc[sub_ann['label_name']==encrypted_strings[i], 'class_number'] = i



sub_ann['xmax'] = sub_ann['width.1'] + sub_ann['xmin']

sub_ann['ymax'] = sub_ann['height.1'] + sub_ann['ymin']



sub_ann['center x'] = (sub_ann['xmax']+sub_ann['xmin'])/2

sub_ann['center y'] = (sub_ann['ymax']+sub_ann['ymin'])/2



r = sub_ann.loc[:, ['image_id',

                    'class_number',

                    'center x',

                    'center y',

                    'width.1',

                    'height.1']].copy()

print(r.head())

pwd()
os.chdir(full_path_to_train_images)

print(os.getcwd())
pwd()
cnt = 0

for current_dir, dirs, files in os.walk('.'):

  for f in files:

    if f.endswith('.jpg'):

      image_name = f[:-4]



      sub_r = r.loc[r['image_id'] == image_name]



      resulted_frame = sub_r.loc[:,['class_number',

                                    'center x',

                                    'center y',

                                    'width.1',

                                    'height.1']].copy()



      path_to_save = '/'+ image_name + '.txt'



      resulted_frame.to_csv(path_to_save, header = False, index = False, sep=' ')
%cd ..
%cd ..
%cd ..
%cd ..
os.chdir('kaggle/working')

print(os.getcwd())
test = glob.glob('../input/global-wheat-detection/test/*')
test[0][37:-4]
image = []

string = []



for i in range(len(test)):

    img = test[i]

    

    image_BGR = cv2.imread(img)

    image.append(str(img[37:-4]))

    

    #cv2.namedWindow('Original Image',cv2.WINDOW_NORMAL)

    #cv2.imshow('Original Image',image_BGR)



    #cv2.waitKey(0)



    #cv2.destroyWindow('Original Image')



    #check point

    #print('Image Shape:',image_BGR.shape)



    h,w = image_BGR.shape[:2]



    #check point

    #print(f'Image height {h} and width {w}')



    blob = cv2.dnn.blobFromImage(image_BGR, 1/255, (416,416), swapRB =True, crop= False)



    #check point

    #print('Image shape: ',image_BGR.shape)

    #print('Blob shape: ',blob.shape)



    #check point

    blob_to_show = blob[0,:,:,:].transpose(1,2,0)

    #print('blob_to_show shape',blob_to_show.shape)



    #cv2.namedWindow('Blob Image',cv2.WINDOW_NORMAL)



    #cv2.imshow('Blob Image', cv2.cvtColor(blob_to_show , cv2.COLOR_RGB2BGR))

    #cv2.waitKey(0)



    #cv2.destroyWindow('Blob Image')



    with open('../input/wheat-data/classes.names') as f:

        labels = [line.strip() for line in f]



    #print(List with labels names)    

    #print(labels)



    network = cv2.dnn.readNetFromDarknet('../input/wheat4/yolov3_custom_train.cfg',

                                         '../input/wheat4/yolov3_custom_train_4000.weights')



    #chech point

    layers_names_all = network.getLayerNames()

    #print(layers_names_all)



    layers_names_output = [layers_names_all[i[0] - 1] for i in network.getUnconnectedOutLayers()]



    #check points

    #print(layers_names_output)



    probability_minimum = 0.1



    threshold = 0.5



    colours = np.random.randint(0,255, size = (len(labels) ,3), dtype = 'uint8')



    #check point

    #print()

    #print(type(colours))

    #print(colours.shape)

    #print(colours[0])



    network.setInput(blob)

    start = time.time()

    output_from_network = network.forward(layers_names_output)

    end = time.time()



    #print(f'object detection took {end - start} seconds')





    bounding_boxes = []

    confidences = []

    class_numbers = []



    for result in output_from_network:

        for detected_objects in result:

            scores = detected_objects[5:]

            

            class_current = np.argmax(scores)

            confidence_current = scores[class_current]

        

            #check point

            #print(detected_objects.shape)

        

            if  confidence_current > probability_minimum:

                box_current = detected_objects[0:4] * np.array([w,h,w,h])

                x_center, y_center, box_width, box_height = box_current

                x_min = int(x_center - (box_width/2))

                y_min = int(y_center - (box_height/2))

                

                bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])

                confidences.append(float(confidence_current))

                class_numbers.append(class_current)

            

        



        

    result = cv2.dnn.NMSBoxes(bounding_boxes, confidences, probability_minimum, threshold)



    pred_strings = []



    counter = 1

    if len(result)>0:

    

        for i in result.flatten():

            #print(f'object {counter}: {labels[int(class_numbers[i])]}')

            counter +=1

        

            x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]

        

            box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

        

            colour_box_current = colours[class_numbers[i]].tolist()

            #print(x_min, y_min, box_width, box_height)

        

            pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(confidences[i], x_min, y_min, box_width, box_height))

            

        

            #check point

            #print(type(colour_box_current)) List

            #print(colour_box_current) [172,10,127]

        

            cv2.rectangle(image_BGR, (x_min,y_min),

                         (x_min + box_width, y_min+box_height),

                         colour_box_current,2)

        

            text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])],

                                                  confidences[i])

        

            cv2.putText(image_BGR, text_box_current, (x_min, y_min-5),

                       cv2.FONT_HERSHEY_COMPLEX, 0.7, colour_box_current, 2)

    

    pred_strings = str(pred_strings).replace(',','')

    pred_strings = str(pred_strings).replace('[','')

    pred_strings = str(pred_strings).replace(']','')

    pred_strings = str(pred_strings).replace("'",'')

    string.append(pred_strings)



    print()

    print('Image:',img[16:])

    print('Total objects been detected: ',len(bounding_boxes))

    print('Number of objects left after non-maximum supprssion: ',counter-1)

    #plt.show(image_BGR)



"""

cv2.namedWindow('Detections',cv2.WINDOW_NORMAL)

cv2.imshow('Detections',image_BGR)

cv2.waitKey(0)

cv2.destroyWindow('Detections')"""
df = pd.DataFrame()

df['image_id'] = image

df['PredictionString'] = string
df.to_csv('submission.csv',index = False)

df.head()