# Install necessary packages one by one

!cat /kaggle/input/odfkosteruw1/koster-uw-workshop-1/requirements.txt | xargs -n 1 pip install
from pathlib import Path

kaggle_path = Path('/kaggle/input/odfkosteruw1/koster-uw-workshop-1')
# Imports

import matplotlib.pyplot as plt

import cv2

import numpy as np
%pylab inline 

import cv2

from IPython.display import clear_output

from google.colab.patches import cv2_imshow
# video_file = Path(kaggle_path, "videos/TjarnoROV1-990813_3-1122.mov")

# video = cv2.VideoCapture(video_file)



# try:

#     while True:

#         (grabbed, frame) = video.read()



#         if not grabbed:

#             break



#         # The important part - Correct BGR to RGB channel

#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)



#         axis('off')

#         # Title of the window

#         title("Input Stream")

#         # Display the frame

#         imshow(frame)

#         show()

#         # Display the frame until new frame is available

#         clear_output(wait=True)

    

#     cv2.destroyAllWindows()

#     video.release()

    

# except KeyboardInterrupt: video.release()

# video_file = Path(kaggle_path, "videos/TjarnoROV1-990813_3-1122.mov")

# video = cv2.VideoCapture(video_file)



# try:

#     while True:

#         (grabbed, frame) = video.read()



#         if not grabbed:

#             break



#         blur = cv2.GaussianBlur(frame, (21, 21), 0)

#         hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)



#         lower = np.array([0,120,70])

#         upper = np.array([180,255,255])

#         lower = np.array(lower, dtype="uint8")

#         upper = np.array(upper, dtype="uint8")

#         mask = cv2.inRange(hsv, lower, upper)



#         frame = cv2.bitwise_and(frame, hsv, mask=mask)

#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)





#         axis('off')

#         # Title of the window

#         title("Input Stream")

#         # Display the frame

#         imshow(frame)

#         show()

#         # Display the frame until new frame is available

#         clear_output(wait=True)

    

#     cv2.destroyAllWindows()

#     video.release()

    

# except KeyboardInterrupt: video.release()

def clearImage(image):

    # Convert the image from BGR to gray

    dark_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)



    channels = cv2.split(image)



    # Get the maximum value of each channel

    # and get the dark channel of each image

    # record the maximum value of each channel

    a_max_dst = [ float("-inf") ]*len(channels)

    for idx in range(len(channels)):

        a_max_dst[idx] = channels[idx].max()



    dark_image = cv2.min(channels[0],cv2.min(channels[1],channels[2]))



    # Gaussian filtering the dark channel

    dark_image = cv2.GaussianBlur(dark_image,(25,25),0)



    image_t = (255.-0.95*dark_image)/255.

    image_t = cv2.max(image_t,0.5)



    # Calculate t(x) and get the clear image

    for idx in range(len(channels)):

        channels[idx] = cv2.max(cv2.add(cv2.subtract(channels[idx].astype(np.float32), int(a_max_dst[idx]))/image_t,

                                                        int(a_max_dst[idx])),0.0)/int(a_max_dst[idx])*255

        channels[idx] = channels[idx].astype(np.uint8)



    return cv2.merge(channels)
# video_file = Path(kaggle_path, "videos/TjarnoROV1-990813_3-1122.mov")

# video = cv2.VideoCapture(video_file)



# try:

#     while True:

#         (grabbed, frame) = video.read()



#         if not grabbed:

#             break



#         # The important part - Correct BGR to RGB channel

#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         n_frame = clearImage(frame)

#         axis('off')

#         # Title of the window

#         title("Input Stream")

#         # Display the frame

#         imshow(n_frame)

#         show()

#         # Display the frame until new frame is available

#         clear_output(wait=True)

    

#     cv2.destroyAllWindows()

#     video.release()

    

# except KeyboardInterrupt: video.release()

# Reference in C++: 

# https://answers.opencv.org/question/26280/background-color-similar-to-object-color-how-isolate-it/

 

#video_file = Path(kaggle_path, "videos/TjarnoROV1-990813_3-1122.mov")

# video_file = Path(kaggle_path, "videos/000114 TMBL-ROV 2000 Säckenrevet EJ numrerade band_1440.mp4")

# video = cv2.VideoCapture(video_file)



# blur_size = 20

# grid_size = 500



# try:

#     while True:

#         (grabbed, frame) = video.read()



#         if frame is None: break



#         # Reduce the size that we observe to reduce noise from corners of the frame

#         origin = frame[100:500, 100:500]



#         if not grabbed:

#             break



#         # Clean up our image

#         new_img = clearImage(frame)



#         new_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)



#         new_img = cv2.split(frame)[2]



#         # Cut to the most important segment

#         new_img = new_img[100:500, 100:500]



#         blur_size += (1 - blur_size % 2)



#         blur = cv2.GaussianBlur(new_img, (blur_size, blur_size), 0)



#         # equalise the histogram

#         equal = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(5,5)).apply(blur)



#         grid_size += (1 - grid_size % 2)



#         # create a binary mask using an adaptive thresholding technique

#         binimage = cv2.adaptiveThreshold(equal, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\

#                                          cv2.THRESH_BINARY, grid_size, -30)



#         #cv2.imshow("bin", binimage)



#         contours, _ = cv2.findContours(binimage.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)



#         # Cycle through contours and add area to array

#         areas = []

#         for c in contours:

#             areas.append(cv2.contourArea(c))



#         # Sort array of areas by size

#         try:

#             largest = np.argmax(areas)

#         except:

#             largest = None



#         if largest is not None:

#             fishMask = np.zeros(new_img.shape, dtype = np.uint8)

#             # Choose our largest contour to be the object we wish to detect

#             fishContours = contours[largest]

#             cv2.polylines(origin,  [fishContours],  True,  (0, 0, 255),  2)

#             # Draw these contours we detect

#             cv2.drawContours(fishMask, contours, -1, 255, -1);

#             #cv2.imshow("fish_mask", fishMask)



            

#         origin = cv2.cvtColor(origin, cv2.COLOR_BGR2RGB)

        

#         axis('off')

#         # Title of the window

#         title("Input Stream")

#         # Display the frame

#         imshow(origin)

#         show()

#         # Display the frame until new frame is available

#         clear_output(wait=True)



#     cv2.destroyAllWindows()

#     video.release()

    

# except KeyboardInterrupt: video.release()

    

            

 

### Save frames as images

# import cv2

# import numpy as np

# import scipy.io as sio

 

# video_file = Path(kaggle_path, "videos/TjarnoROV1-990813_3-1122.mov")

# video = cv2.VideoCapture(video_file)



# total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)



# frame_id = 0

# i = 0

# while True:

#     (grabbed, frame) = video.read()

    

#     if not grabbed:

#         break

 

#     new_img = clearImage(frame)

#     new_img = cv2.resize(new_img, (416, 416))

#     assert(new_img.shape == (416, 416, 3))

    

#     frame_id += 1

    

#     if frame_id % 100 == 0:

#         print("Saved", frame_id)

#         cv2.imwrite(Path(kaggle_path, "img/odf_video_frames/{:s}".format(str(i)+'.jpg'), new_img))

#         i += 1

#     if cv2.waitKey(1) & 0xFF == ord('q'):

#         break

# print('Saved images')

# cv2.destroyAllWindows()

# video.release()
import glob, os



dataset_path = Path(kaggle_path, "img/odf_video_frames")



# Percentage of images to be used for the test set

percentage_test = 10;



# Create and/or truncate train.txt and test.txt

file_train = open('./Data/img/train.txt', 'w')  

file_test = open('./Data/img/test.txt', 'w')



# Populate train.txt and test.txt

counter = 1

index_test = int(percentage_test / 100 * len(os.listdir(dataset_path)))

for pathAndFilename in glob.iglob(os.path.join(dataset_path, "*.jpg")):  

    title, ext = os.path.splitext(os.path.basename(pathAndFilename))



    if counter == index_test+1:

        counter = 1

        file_test.write(os.path.basename(title) + '.jpg' + "\n")

    else:

        file_train.write(os.path.basename(title) + '.jpg' + "\n")

        counter = counter + 1
# !labelImg ./Data/img/odf_video_frames/ ./Data/img/odf_classes.txt
lines = []

for line in open(Path(kaggle_path, "logs/train_log_example.log")):

    if "avg" in line:

        lines.append(line)



iterations = []

avg_loss = []



print('Retrieving data and plotting training loss graph...')

for i in range(len(lines)):

    lineParts = lines[i].split(',')

    iterations.append(int(lineParts[0].split(':')[0]))

    avg_loss.append(float(lineParts[1].split()[0]))



fig = plt.figure(figsize=(15,10))

for i in range(0, len(lines)):

    plt.plot(iterations[i:i+2], avg_loss[i:i+2], 'r.-')



plt.xlabel('Batch Number')

plt.ylabel('Avg Loss')

fig.savefig('training_loss_plot.png', dpi=1000)



print('Done! Plot saved as training_loss_plot.png')
## Visualize predictions using OpenCV



import argparse

import sys

import numpy as np

import os.path



# Initialize the parameters

confThreshold = 0.1 #Confidence threshold

nmsThreshold = 0.4 #Non-maximum suppression threshold



inpWidth = 416  #608     #Width of network's input image

inpHeight = 416 #608     #Height of network's input image

        

# Load names of classes

classesFile = Path(kaggle_path, "models/sweden_yolo/odf_classes.names");



classes = None

with open(classesFile, 'rt') as f:

    classes = f.read().rstrip('\n').split('\n')



# Give the configuration and weight files for the model and load the network using them.



modelConfiguration = Path(kaggle_path, "models/sweden_yolo/sweden_yolo.cfg");

modelWeights = Path(kaggle_path, "models/sweden_yolo/sweden_yolo.backup");



net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)



# Get the names of the output layers

def getOutputsNames(net):

    # Get the names of all the layers in the network

    layersNames = net.getLayerNames()

    # Get the names of the output layers, i.e. the layers with unconnected outputs

    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]



# Draw the predicted bounding box

def drawPred(classId, conf, left, top, right, bottom):

    # Draw a bounding box.

    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)



    label = '%.2f' % conf

        

    # Get the label for the class name and its confidence

    if classes:

        assert(classId < len(classes))

        label = '%s:%s' % (classes[classId], label)



    #Display the label at the top of the bounding box

    labelSize, baseLine = cv2.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)

    top = max(top, labelSize[1])

    cv2.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), 

                 (0, 0, 255), cv2.FILLED)

    

    cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2)



# Remove the bounding boxes with low confidence using non-maxima suppression

def postprocess(frame, outs):

    frameHeight = frame.shape[0]

    frameWidth = frame.shape[1]



    classIds = []

    confidences = []

    boxes = []

    # Scan through all the bounding boxes output from the network and keep only the

    # ones with high confidence scores. Assign the box's class label as the class with the highest score.

    classIds = []

    confidences = []

    boxes = []

    for out in outs:

        print("out.shape : ", out.shape)

        for detection in out:

            #if detection[4]>0.001:

            scores = detection[5:]

            classId = np.argmax(scores)

            #if scores[classId]>confThreshold:

            confidence = scores[classId]

            if detection[4]>confThreshold:

                print(detection[4], " - ", scores[classId], " - th : ", confThreshold)

                print(detection)

            if confidence > confThreshold:

                center_x = int(detection[0] * frameWidth)

                center_y = int(detection[1] * frameHeight)

                width = int(detection[2] * frameWidth)

                height = int(detection[3] * frameHeight)

                left = int(center_x - width / 2)

                top = int(center_y - height / 2)

                classIds.append(classId)

                confidences.append(float(confidence))

                boxes.append([left, top, width, height])



    # Perform non maximum suppression to eliminate redundant overlapping boxes with

    # lower confidences.

    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)

    for i in indices:

        i = i[0]

        box = boxes[i]

        left = box[0]

        top = box[1]

        width = box[2]

        height = box[3]

        drawPred(classIds[i], confidences[i], left, top, left + width, top + height)



# Process inputs

winName = 'ODF - Sweden Demo'

cv2.namedWindow(winName, cv2.WINDOW_NORMAL)



outputFile = Path(kaggle_path, "models/sweden_yolo/yolo_out_py.avi");



video_path = Path(kaggle_path, "models/sweden_yolo/crabs.mov")

cap = cv2.VideoCapture(video_path)

vid_writer = cv2.VideoWriter(outputFile, cv2.VideoWriter_fourcc('M','J','P','G'),

                            30, (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))





count = 0

    

while cv2.waitKey(1) < 0:

    

    # get frame from the video

    hasFrame, frame = cap.read()

    if frame is None: break

    #frame = frame[100:516, 100:516]

    frame = clearImage(frame)

    frame = cv2.resize(frame, (inpWidth, inpHeight))

    

    # Stop the program if reached end of video

    if not hasFrame:

        print("Done processing !!!")

        print("Output file is stored as ", outputFile)

        cv2.waitKey(3000)

        break



    # Create a 4D blob from a frame.

    blob = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

    

    # Sets the input to the network

    net.setInput(blob)



    # Runs the forward pass to get output of the output layers

    outs = net.forward(getOutputsNames(net))



    # Remove the bounding boxes with low confidence

    postprocess(frame, outs)



    # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)

    t, _ = net.getPerfProfile()

    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())

    vid_writer.write(frame.astype(np.uint8))

    

    count += 30 # i.e. at 30 fps, this advances one second

    cap.set(1, count)

    #cv2.imshow(winName, frame)

    