'''



Advantages of my solution: 



CPU only

Total time taken to process  400  video files is  1667.5433096885681  seconds, that's 28 minutes

Logloss is  0.21584579832806053  for  400  videos



I have tried a very simple and holistic approach to a complex challenge driven primarily by limited computing resources and

time. Thus my focus has been to extract valuable information and simplify the input data and at the same time apply 

proven tested methods to train a NN so that the results can be accurate and fast. 



To that end I have used Transfer Learning leveraging the Inceptionv3 CNN model. Instead of using an RNN/LSTM to deal with the temporal aspects of Video frames,

I bunched together N frames from a video and coalesced them to include the temporal aspect which I then analyze with the CNN. Instead of the full 

N frames, I extracted relevant sections of frames, faces in this specific case and reduced the input dimensions further. 



Pre-processing for Training uses exactly the same modules and functions as the ones below except with different set of parameters. I haven't included the trianing module

here but it is as simple as an agglomeration of calls like this - in the case of training I store the concatenated extracted sections from the N frames

in the StoreConcatIn based on this logic :



                            if label.find("FAKE") != -1 : #meaning it  said FAKE

                                StoreConcatIn = "TL/data/train/fake"



                            else:

                                StoreConcatIn = "TL/data/train/real"

                                

                            n = ScoreVideo(filename, StoreConcatIn, 0)

                            

I loop over training folders in the zip folders and collect this collection of N sub-frames concatenated together. 



I feed this to the transfer training NN and train it. The minimum error rate I obtained was around 0.12 and that too with a minor subset of the training

videos as I didn't have enough resources.  The training pre-processing module is at the very bottom and as I ran it primarily from my desktop has references to various

modules that I just left as is. 



I am happy to provide the jupyter notebook for the actual training process as well if needed. 



The testing pre-procerssing part is at the very bottom. 

                         

'''



# the import modules 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import glob

import csv

import datetime as dt

import platform

import os

import os.path

from random import seed, random , randint # temporary score generator

import cv2

import keras

from keras.preprocessing import image

from keras.models import  load_model

from matplotlib import pyplot as plt

import time

import sys

from matplotlib.patches import Rectangle

#from mtcnn.mtcnn import MTCNN

import warnings

warnings.filterwarnings("ignore")

from IPython.display import Image

#Image(filename='test.png') 

import math
#face detection functions

#for Yolo and dnn stuff

min_confidence=0.1 #0.1

classes = None

COLORS = None





cascPath = "/kaggle/input/haarxml/haarcascade_frontalface_default.xml"

setupName = "/kaggle/input/yolov3" # Windows

model = os.path.join(setupName, 'yolo.weights') #https://pjreddie.com/darknet/yolov2/

config = os.path.join(setupName, 'yolo.cfg')

labels = os.path.join(setupName, 'labels.txt')



net = cv2.dnn.readNet(model, config)

#face detection functions

#---------------------------------------------------------------------

#

#   finds number of people in the image and returns the # found and the co-ordinates using caffee

#   need to push this into Kaggle

#---------------------------------------------------------------------

def findFaceCaffe (image, debug = 0):

    #https://github.com/mmilovec/facedetectionOpenCV



    net = cv2.dnn.readNetFromCaffe("/kaggle/input/caffestuff/proto.txt", "/kaggle/input/caffestuff/res10_300x300_ssd_iter_140000.caffemodel")

    #image = cv2.imread(imageName)

    (h, w) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (103.93, 116.77, 123.68))

    net.setInput(blob)

    detections = net.forward()

    boxes = []



    # loop over the detections

    for i in range(0, detections.shape[2]):

        # extract the confidence (i.e., probability) associated with the

        # prediction

        confidence = detections[0, 0, i, 2]



        # filter out weak detections by ensuring the `confidence` is

        # greater than the minimum confidence

        if confidence > 0.4:

            # compute the (x, y)-coordinates of the bounding box for the

            # object

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

            (startX, startY, endX, endY) = box.astype("int")

            boxes.append([startX, startY, endX, endY])



            # draw the bounding box of the face along with the associated

            # probability

            if debug:

                text = "{:.2f}%".format(confidence * 100)

                y = startY - 10 if startY - 10 > 10 else startY + 10

                cv2.rectangle(image, (startX, startY), (endX, endY),

                              (0, 0, 255), 2)

                cv2.putText(image, text, (startX, y),

                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)



    # show the output image

    if debug:

        cv2.imshow("Caffe faces found", image)

        cv2.waitKey(0)



    return len(boxes), boxes



'''

    Function crops an image and does some error checking

    Returns cropped version of img input 

'''

def CropImage (img, rect, debug = 0, name='cropped', dimension = 250):

    

    if debug: print('From inside CropImage function of Detectface.py module with dimension = ', dimension)

    x, y, w, h = rect

    if debug: print('Rect is  ',rect)

    w = dimension

    h = dimension

    x1 = 0

    x2 = 0

    y1 = 0

    y2 = 0



    # cannot have the left or the top outside the original image

    if (y - 0.25 * h < 0 ):

        y1 = 0 #cannot have negative

        y2 = h #need a square of the dimension = dimension

    else:

        y1 = y - 0.25 * h

        y2 = y + 0.75 * h



    if x - 0.25 * w < 0:

        x1 = 0

        x2 =  w #same reason as above for y

    else:

        x1 = x - 0.25 * w

        x2 = x + 0.75 * w



    # cannot have the bottom or right outside the original image



    if y2 > img.shape[0] :

        y2 = img.shape[0]

        y1 = y2 - h



    if x2 > img.shape[1] :

        x2 = img.shape[1]

        x1 = x2 - w



    if debug: print('parameters for cropping are ' , y1, y2, x1,x2 , ' image shape is ', img.shape)

    #crop_img = img[int(y - 0.25 * h):int(y + 0.75 * h), int(x - 0.25 * w):int(x + 0.75 * w)]

    crop_img = img[int(y1):int(y2), int(x1):int(x2)]

    if debug:

        cv2.imshow(name, crop_img)



    return crop_img



'''

    Function returns rectangles that bounds faces found in the image using Haar cascade

'''

def findPersonsInImageHaar( image, debug = 0):

    if debug: print('From inside findPersonsInImage function of HaarDetectface.py module')

    # Create the haar cascade

    #faceCascade = cv2.CascadeClassifier(cascPath)

    faceCascade = cv2.CascadeClassifier(cascPath)



    # Read the image

    #image = cv2.imread(img)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    count = 0

    starttime = time.time()

    faces = faceCascade.detectMultiScale(gray, 1.1, 5)

    ftime = time.time() - starttime



    count = len(faces)



    if debug:

        # Draw a rectangle around the faces

        for (x, y, w, h) in faces:

            #draw bigger rectangles

            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)





    #print(count)

    if debug: 

        print('Haar - it took: ' ,  ftime, " to find {0} ,  faces!".format(count))

        cv2.imshow("Haar Faces found", image)

        cv2.waitKey(0)



    return count, faces



def get_output_layers(net):

    

    layer_names = net.getLayerNames()

    

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]



    return output_layers



'''

    Function returns rectangles that bounds faces found in the image usingYolo and dnn

'''

def findPersonsInImageYolo( image, debug = 0):

    global classes, COLORS



    if debug: print('From inside findPersonsInImage function of YoloAIv3Still.py module')

    



    Width = image.shape[1]

    Height = image.shape[0]

    scale = 0.00392





    with open(labels, 'r') as f:

        classes = [line.strip() for line in f.readlines()]

    #print(classes) # not necessary each time



    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))



    #net = cv2.dnn.readNet(model, config)



    starttime = time.time()



    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)



    net.setInput(blob)



    outs = net.forward(get_output_layers(net))



    class_ids = []

    confidences = []

    boxes = []

    conf_threshold = 0.5

    nms_threshold = 0.4



    count = 0



    for out in outs:

        for detection in out:

            scores = detection[5:]

            class_id = np.argmax(scores)

            confidence = scores[class_id]



            #label = str(classes[class_id])

            #if confidence > min_confidence: #old

            if (confidence > min_confidence)  and (str(classes[class_id]) == 'person' ):

                center_x = int(detection[0] * Width)

                center_y = int(detection[1] * Height)

                w = int(detection[2] * Width)

                h = int(detection[3] * Height)

                x = center_x - w / 2

                y = center_y - h / 2

                class_ids.append(class_id)

                confidences.append(float(confidence))

                boxes.append([x, y, w, h])

                count = count + 1



    totaltime = time.time() - starttime

    if debug: print( 'YoloAIv3Still/findPersonsInImage ', count, ' persons were found by ',setupName, ' in ', totaltime, ' time')



    if count == 0: return count, boxes



    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)



    for i in indices:

        i = i[0]

        box = boxes[i]

        x = box[0]

        if x < 0: box[0] = 0

        y = box[1]

        if y < 0: box[1] = 0

        w = box[2]

        h = box[3]



        '''

        if debug: 

            #draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

            #try cropping the relevant sections of the image

            crop_img = image[round(y):round(y+h), round(x):round(x+w)]

            cropImage = 'detectPerson' + str(i)

            cv2.imshow(cropImage, crop_img)

        '''



    #dont forget why we came here

    return count, boxes



#####################################################################################

#

#   Takes as input a frame of a video and sends back a concatenated set cropped images

#   that can be used instead of the frame for training

#

#   concat = 1 will mean vertical

#

#####################################################################################

def ConcatImages (frame, debug =0, name='concatenated', concat = 0, dims = 250, numFovea=1):



    if debug: 

        print('1. From inside ConcatImages function of YoloAIv3Still.py module with dimension ', dims, 'debug flag ', debug)

    img = []

    boxes = []

    dimension = dims

    retcode = 99

    number = 0



    #cv2.imshow('wierd frame', frame)

    #cv2.waitKey(0)



    number, boxes = findFaceCaffe(frame, debug)

    

    #find the details from the frame by

    if number == 0 : #lets try Yolo

        number, boxes = findPersonsInImageYolo(frame, debug)

    

    if number == 0 : #lets try Haar

        number, boxes = findPersonsInImageHaar(frame, debug)



    if debug: print('2. YoloAI/Concat - Boxes are ', boxes, ' Number is ', number)



    if number == 0:

        if debug: cv2.imshow('YoloAI/Concat - could not find person with Yolo or Haar', frame)

        return [], 0



    concatenatedImage = np.zeros((dimension,dimension,3)) #initialize

    

    for i in range(len(boxes)):

        if debug : print('3. YoloAI/Concat - In for loop inside ConcatImages in YolAIv3Still.py')

        

        #just get 1 people , 2 would mean

        if i == numFovea:

            if debug: print('4. ----YoloAI/Concat - More than 2 persons found - leaving with 2')

            retcode = 4

            break





        #otherwise lets get things done

        if debug: print('5. ----YoloAI/Concat - Box details are :', boxes[i -1])

        img = []

        img = CropImage (frame, boxes[i -1], debug, dimension = dimension)

        if i == 0: #prior to doubling

            concatenatedImage = img

            retcode = 5

            if debug:

                print('6. ----YoloAI/Concat - Printing before concatenating images for first fovea sections - shape = ',img.shape  )

                timee = str(time.time())

                first = 'first'+ timee

                cv2.imshow(first, img)

                second = 'second' + timee

                cv2.imshow(second, concatenatedImage)

                #do the concatenation right here



        else:

            if (concatenatedImage.shape[0] == dimension) and (img.shape[0] == dimension):  # this won't allow concatenation

                if concat == 0:

                    #horizontal stack

                    concatenatedImage = np.hstack((concatenatedImage, img))

                    retcode = 6

                    if debug:

                        print('7.1. ----YoloAI/Concat - concatenating horizontaly')

                else:

                    # vertical stack

                    concatenatedImage = np.vstack((concatenatedImage, img))

                    retcode = 7

                    if debug:

                        print('7.1. ----YoloAI/Concat - concatenating vertically')

            else:

                #one of the components is not right - dont return anything

                return [], 1





        imgName = "img" + str(i)

        frameName = name

        



    #what if there is only one person - then it throws everything off the skelter

    #double it

    if (len(boxes) == 1):

        if debug:

            extra = 'could this be it ?' + str(time.time())

            print('8.YoloAI/Concat - One person found - doubling it, and now see the  img')

            cv2.imshow(extra, concatenatedImage)



        if numFovea == 1: # just ship the image back; no need to double or concatenate

            if img.shape[0] == dimension :

                return img, 2



        else: #numFovea != 1

            if (concatenatedImage.shape[0] == dimension) and (img.shape[0] == dimension):  # this won't allow concatenation

                if concat == 0: concatenatedImage = np.hstack((concatenatedImage, img))

                else: concatenatedImage = np.vstack((concatenatedImage, img))



            else: #didn't meet criteria

                return [], 3



    if debug:

        cv2.waitKey(0)



    

    return concatenatedImage, retcode



def dummyCropImage(): 

    print('Haha')

    image = cv2.imread("/kaggle/input/junkimages/2women.JPG")

    faces, n = findPersonsInImageHaar(image, debug = 0)

    print(n, ' faces found with Haar with following details', faces)

    

def dummyYolov3AITesting (): # this worked well

    print('hello dumy Yolo')

    image = cv2.imread("/kaggle/input/junkimages1/2women.JPG")

    result, retcode = ConcatImages (image, debug=0, dims = 175)

    

    if len(result) : print ('Great', result.shape, 'Return code:', retcode)

    else: print('disaster', 'Return code:', retcode)

    

    

    

    

#Frames related functions

'''



    Function rescales a given frame by a percent 

    

'''  

def rescale_frame(frame, percent=75):

    width = int(frame.shape[1] * percent/ 100)

    height = int(frame.shape[0] * percent/ 100)

    dim = (width, height)

    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)



'''



    Function returns howMany frames from a given Video mp4 with some options

    returns empty arrays if for any reason VideoCapture, or .get does crazy things

    

'''

def CaptureNRadomFrames(videoFile, howMany, debug = 0, seedOption = 1, gap=45):



    #frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    randomFrames = []

    frameNums = []

    if seedOption: seed(1)

    else: seed(time.clock())

    

    if debug: print('From inside CaptureNRadomFrames function of VideoFunctions module')

    

    cap =  cv2.VideoCapture(videoFile)

    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)



    if debug: print('Frame count for video', videoFile, 'is ',frame_count)

    #print('Frame count for video', videoFile, 'is ', frame_count)

    

    if frame_count == 0: return randomFrames, frameNums



    frame_number = randint(0, frame_count -1)

    

    for _ in range(howMany):

        cap.set(1, frame_number)

        res, frame = cap.read()

        if res:

            frame50 = rescale_frame(frame, percent=50)

            #YoloAIv3Still.findPersonsInImage(frame50, 2)

            randomFrames.append(frame50)

            frameNums.append(frame_number)

        #frame_number = randint(frame_number, frame_count -1)

        frame_number = int((frame_number + gap ) % frame_count)



        # When everything done, release the capture

    cap.release()

    cv2.destroyAllWindows()



    return randomFrames, frameNums 



'''

This tested well 

'''

def dummyCaptureNRadomFrames ():

    print('Haha')

    '''

        testing for CaptureNRadomFrames

    '''

    #lets check out one video and the random frame

    videoFile = '/kaggle/input/deepfake-detection-challenge/test_videos/uhakqelqri.mp4'

    RandomFrames, frameNos = CaptureNRadomFrames(videoFile, 5,1,seedOption = 0)

    print('how many :', len(RandomFrames), ' frame numbers :', frameNos)

    plt.figure(figsize=(20,10))

    columns = 5

    for i in range (len(RandomFrames)):

        pic = str(i)

        #cv2.imshow(pic, RandomFrames[i-1] )

        plt.subplot(len(RandomFrames) / columns + 1, columns, i + 1)

        plt.imshow(RandomFrames[i-1])

        plt.title(pic)

        #plt.figure()

    plt.show()

#Analyse video functions - needs ScoreConcat
#Video prep and scoring modules

def ScoreVideo(videoFile, SavePath ,debugValue = 0):



    # first get random frames from the video

    debug = debugValue



    if debug:

        print('From inside ScoreVideo function of AnalyzeVideo.py module')

        print()

    

    RandomFrames = []

    frameNos = []

    boxes = []

    originalWorkingDirectory = os.getcwd()

    score = 0.99 # unique initialization value





    numFrames = 5

    perFrameMaxFovea = 1 #original default 2

    heightConcat = 175

    widthConcat = heightConcat * perFrameMaxFovea

    ww = heightConcat * perFrameMaxFovea * numFrames

    masterShape = (heightConcat, ww , 3)

    masterConcat = []



    if debug:

        print('AnalyzeVideo/ScoreVideo - the master shape should be ', masterShape)

        print()



    RandomFrames, frameNos = CaptureNRadomFrames(videoFile, numFrames, debug)

    if len(RandomFrames) == 0: #no video in container 

        if debug: print('AnalyzeVideo/ScoreVideo - Could not get any frames fom this videofile :' , videoFile)

        return score



    if len(RandomFrames) != numFrames:  # then lets try again

        if debug: print('AnalyzeVideo/ScoreVideo Could not get ',numFrames, ' frames fom this videofile :', videoFile, '; trying again')

        #reinitialise

        RandomFrames,  frameNos = [], []

        #and try again

        RandomFrames, frameNos = CaptureNRadomFrames(videoFile, numFrames, debug, seedOption = 0)

        if len(RandomFrames) == 0:

            return score  # no point in going further 



    if debug:print('ScoreVideo/AnalyzeVideo ; Frames are :', frameNos) #[223, 228, 233, 238], [180, 185, 190, 195],[180, 185, 190, 195]



    masterConcat = []



    # For each of the RandomFrames, extract fovea regions in the frames and concatenate

    concatenatedImage = []

    retcode = 999



    # sometimes we are not getting a full complement of numFrames frames







    for i in range(len(RandomFrames)):



        concatenatedImage, retcode = ConcatImages (RandomFrames[i], debug, str(frameNos[i]), dims = heightConcat, numFovea=perFrameMaxFovea)



        #sometimes the image returned is no good - (250, 0, 3)

        if len(concatenatedImage) == 0 :

            if debug: print('----------------------------> AnalyzeVideo/ScoreVideo Disaster - no good, trying again, Retcode :', retcode)

            #lets try again

            concatenatedImage, retcode = ConcatImages(RandomFrames[i], debug, str(frameNos[i]), dims = heightConcat, numFovea=perFrameMaxFovea)



        if len(concatenatedImage) == 0:

            

            if debug: 

                print()

                print(videoFile, ' ----> AnalyzeVideo/ScoreVideo ----------Disaster----No images returned--------------skip this video, Retcode :', retcode)

                print()

            return 0.001  # Could not find any faces - most likely real - Change #1



        #change this to just check the shape and not the length - anyways if len is 0, there is no need to check shape

        #and would be caught before

        #if len(concatenatedImage) and (concatenatedImage.shape == (heightConcat, widthConcat, 3)): # as if we cannot find any person, we return a null array whose len is 0

        if (concatenatedImage.shape == (heightConcat, widthConcat, 3)):

            #masterConcat = masterConcat + concatenatedImage

            if debug:

                print(' ----> AnalyzeVideo/ScoreVideo ', frameNos[i], 'image shape is :', concatenatedImage.shape, '  Retcode :', retcode)

            if i == 0:

                masterConcat = concatenatedImage

            else:

                masterConcat = np.hstack((masterConcat, concatenatedImage))



            if debug:

                cv2.imshow(' ----> AnalyzeVideo/ScoreVideo  master concat', masterConcat)

                print(' ----> AnalyzeVideo/ScoreVideo MasterConcat shape is ', masterConcat.shape)

            if np.array_equal(masterShape , masterConcat.shape) :

                if debug:

                    print()

                    print('------------------------------- >>>AnalyzeVideo/ScoreVideo   --- lets celebrate , Retcode :', retcode)

                    print()

                



        else:

            if debug:

                print()

                print('i is ', i, ' image shape is ', concatenatedImage.shape)

                print(videoFile, ' ----> AnalyzeVideo/ScoreVideo ---Disaster------No person found-------skip this video, Retcode :', retcode)

                print()

            return 0.001 # Could not find any faces - most likely real Change #2





    cv2.imwrite('temp.jpg', masterConcat)

    score = ScoreImage('temp.jpg', widthConcat, heightConcat, 0)

    

    if debug: print('--------------> AnlyzeVideo/ScoreVideo ', videoFile, ' just scored ', score)



    #lets cleanup

    os.remove('/kaggle/working/temp.jpg')

    

    if debug: cv2.waitKey(0)



    return score



def TestScoreVideo():

    print('ha ha testing testing')

    videoFile = '/kaggle/input/deepfake-detection-challenge/test_videos/uhakqelqri.mp4'

    score = ScoreVideo(videoFile, "", 0)



    print('Score for ', videoFile, ' is ', score)

    
#ScoreConcat functions



filepath = "/kaggle/input/mymodel/bestModel.h5"

mymodel = load_model(filepath)





'''

    Function prepares image prior to using it for prediction

'''

def prepare_image(file, widthConcat,height):

    img_path = ''

    img = image.load_img(img_path + file, target_size=(widthConcat,height))

    img_array = image.img_to_array(img)

    img_array_expanded_dims = np.expand_dims(img_array, axis=0)

    return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)



'''

    Function uses the image prepared by function above in conjunction with the

    model to run a prediction

'''

def ScoreImage (file, widthConcat,height, debug =0):

    score = 0.5

    n = []

    preprocessed_image = prepare_image(file, widthConcat,height)

    n = mymodel.predict(preprocessed_image)

    if debug: print('Real prediction is :', predictions)



    score = n[0][0] #fake

    

    # Leave the scores alone - Change 3

    

    return round(score, 3)

    



def testScoreImage (): #works good

    perFrameMaxFovea = 1

    height = 175

    widthConcat = height * perFrameMaxFovea

    score = 0

    ImageFilename2 ="/kaggle/input/realfake/fake.jpg"

    score = ScoreImage(ImageFilename2, widthConcat,height, 0)

    print('Image ', ImageFilename2, ' has a score of ', score)

    



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# added the findfacewithCaffe function - check it out



'''

    before submission 

        

        1. remove the if count > 7...

        2. uncomment #collectdata(fileName, os.path.basename(videoName), score)

        3. remove the print(os.path.basename(videoName), score)



'''

    

def main():

    

    count = 0

    logloss = 0 

    fileName = 'submission.csv'

    filenames = []

    VideoFilenames = []

    ScoreArray = []

    

    starttime = time.time() # Start the timer

    

    for dirname, _, filenames in os.walk('/kaggle/input/deepfake-detection-challenge/test_videos/'):

        for videoName in filenames:

            score = 0.5

            VideoFullPath = os.path.join(dirname, videoName)

            try :



                #Analyze and score - the ScoreVideo function now checks for the SavePath parameter

                score = ScoreVideo(VideoFullPath, "", 0)

                print(count, os.path.basename(videoName), score)

                count = count + 1

                



            except:

                # predict something for that videoFile

                score = 0.99

                print(videoName, ' Main: Could not Analyze video  ')

                count = count + 1

            

            #store the videoname and the score in arrays to use with pandas

            VideoFilenames.append(os.path.basename(videoName))

            ScoreArray.append(score)

            

            #if count > 20 : break #testing testing

            #logloss = logloss + score * math.log(score) + (1 - score) * math.log(1 - score)

            if score > 0 and score < 1.0: logloss = logloss + score * math.log(score) + (1 - score) * math.log(1 - score)

    

    #do the pandas and csv thing now - https://www.kaggle.com/dansbecker/submitting-from-a-kernel -  much easier 

    my_submission = pd.DataFrame({'filename': VideoFilenames, 'label': ScoreArray})

    # you could use any filename. We choose submission here

    my_submission.to_csv(fileName, index=False)

            

    totalTime = time.time() - starttime # Stop the timer

    logloss = - 1 / (count ) * logloss

    

    

    print('Total time taken to process ' , count, ' video files is ', totalTime, ' seconds')

    print ('Logloss is ', logloss , ' for ', count, ' videos')

    

if __name__ == '__main__':

    main()

    



# Any results you write to the current directory are saved as output.
!more *csv | wc -l

#!more *csv

#!ls *csv

#!more submission.csv 

!ls *csv

#!more submission.csv
'''

Pre-processing for Training  



'''

import zipfile

import json

import pprint

from glob import glob

import cv2

import numpy as np

import os



directory = "/media/saibal/Seagate Backup Plus Drive/DeepFakeData" # this is where I stored the training videos Kaggle provided



def ppmain():



    #zipfileName = 'dfdc_train_part_06' + '.zip'

    #zipfileList = glob('dfdc*.zip') #works

    zipfileList = glob(directory + '/' + 'dfdc*.zip')  # works



    #zipfileList = glob('dfdc*.zip')  # works

    

    try:

        

        for zipfileName in zipfileList:

            print('')

            print('')

            print('For zip directory :', zipfileName)

            count = 0

            junk = 0

            json_data = {}

            with zipfile.ZipFile(zipfileName) as z: # opening the zip file using 'zipfile.ZipFile' class



                listOfiles = z.namelist()

                

                #lets get the metadata stuff out of the way first

                for filename in listOfiles:

                    if filename.endswith('json'):

                        print('--->JSON Filename is : ', filename)



                        #extract the path to the filename here for later

                        #use with the videos

                        pathVar = os.path.dirname(filename)

                        # read the file

                        with z.open(filename) as f:

                            data = f.read()

                            json_data = json.loads(data)

                            #pprint.pprint(json_data)



                                

                #lets deal with video now

                for filename in listOfiles:

                    if filename.endswith('mp4'):

                        z.extract(filename)

                        #what is the video label - Fake or real ?

                        _, tail = os.path.split(filename)

                        #feed the filename only to get label

                        label = json_data[tail]['label']

                        #print('--------------->Filename is : ', tail, ' and label =', label)

                        StoreConcatIn = ''

                        if label.find("FAKE") != -1 : #meaning it  said FAKE

                            StoreConcatIn = "TL/data/train/fake"

                            print('fake - store ',tail,' in ', StoreConcatIn)



                        else:

                            StoreConcatIn = "TL/data/train/real"

                            print('real - store ',tail,' in ', StoreConcatIn)



                        n = ScoreVideo(filename, StoreConcatIn, 0)





                        #now lets do the video reads and representation business - just for debugging

                        '''

                        cap =  cv2.VideoCapture(filename)

                        while True:

                            ret, frame = cap.read()

                            #print(ret, frame)

                            if ret == True:

                                cv2.imshow('Video', frame)

                                if cv2.waitKey(1) & 0xFF == ord('q'):

                                    break

                            else:

                                #print('frame is None')

                                break



                        cap.release()

                        cv2.destroyAllWindows()

                        '''

                        

                    else:

                        print('what file is this ?')

                            

                                

                        

                    



        #list all file names

        '''

        for elem in listOfiles:

            print(elem)

        '''



            



        #list how many files are there

        print('')

        #print(len(listOfiles))

            

    except zipfile.BadZipFile: # if the zip file has any errors then it prints the error message which you wrote under the 'except' block

        print('Error: Zip file is corrupted')



#if __name__ == '__main__':

    #ppmain()
