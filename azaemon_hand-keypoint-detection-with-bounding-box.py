!pip install tensorflow==1.14
import tensorflow as tf
import time
from PIL import Image
print(tf.__version__)
import sys
sys.path.append('../input/handpose/Hand Pose')
root='../input/handpose/Hand Pose/'
import numpy as np
import sys
import os

import detection_rectangles
from utils import draw_util
from matplotlib import pyplot as plt
import cv2
def Reformat_Image(img,im_width, im_height):

    from PIL import Image
    from skimage.transform import resize

    width = img.shape[1]
    height = img.shape[0]
    image = Image.fromarray(img)
    image_size = image.size
    print(image_size)
    bigsidew = im_width
    bigsideh = im_height

    background = Image.new('RGB', (bigsidew, bigsideh), (255, 255, 255))
    offset = (int(round(((bigsidew - width) / 2), 0)), int(round(((bigsideh - height) / 2),0)))
    print("offset: ",offset)
    background.paste(image, offset)
    print(background.size)
    result= np.array(background)
    #result= resize(result, (224,224,3))
    plt.imshow(result)
    plt.savefig('result.png')
    print(result.shape)
    print("Image has been resized !")
    return result
def plot_multiple_img(img_matrix_list, title_list, ncols, main_title="Hand Pose"):
    fig, myaxes = plt.subplots(figsize=(20, 15), nrows=2, ncols=ncols, squeeze=True)
    fig.suptitle(main_title, fontsize = 30)
    #fig.subplots_adjust(wspace=0.3)
    #fig.subplots_adjust(hspace=0.3)
    for i, (img, title) in enumerate(zip(img_matrix_list, title_list)):
        myaxes[i // ncols][i % ncols].imshow(img)
        myaxes[i // ncols][i % ncols].set_title(title, fontsize=15)
    plt.show()
#image_np= Image.open(root+'body.jpg')
file='../input/hand-pose-estimation/Hand Data/Multiple hands/t2.jpg'
image_np= cv2.imread(file)
image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
im_width = image_np.shape[1]
im_height = image_np.shape[0]
print(im_width,im_height)
detection_graph, sess = detection_rectangles.load_inference_graph()
relative_boxes, scores, classes = detection_rectangles.detect_objects(image_np, detection_graph, sess)
box_relative2absolute = lambda box: (box[1] * im_width, box[3] * im_width, box[0] * im_height, box[2] * im_height)
hand_boxes = []

###Take the best 2 boxes for 2 hands
hand_boxes.append(box_relative2absolute(relative_boxes[0]))
hand_boxes.append(box_relative2absolute(relative_boxes[1]))
print(hand_boxes)
image=draw_util.draw_box_on_image(hand_boxes,  image_np)
plt.imshow(image)
protoFile = root+"hand/pose_deploy.prototxt"
weightsFile = root+"hand/pose_iter_102000.caffemodel"
nPoints = 22
POSE_PAIRS = [ [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20] ]
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
count=1
img_matrix_list=[]
for box in hand_boxes:
  xmin= int(box[0])
  xmax= int(box[1])
  ymin= int(box[2])
  ymax= int(box[3])
  print(ymin,ymax,xmin,xmax)
  result= image_np[ymin:ymax,xmin:xmax,:]
  im_width = 2*result.shape[1]
  im_height = 2*result.shape[0]
  frame= Reformat_Image(result,im_width,im_height)

  frameCopy = np.copy(frame)
  print(frame.shape)
  frameWidth = frame.shape[1]
  frameHeight = frame.shape[0]
  print(frameWidth,frameHeight)
  aspect_ratio = frameWidth/frameHeight

  threshold = 0.5

  t = time.time()
  # input image dimensions for the network
  inHeight = 368
  inWidth = int(((aspect_ratio*inHeight)*8)//8)
  inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
  print(inpBlob.shape)
  net.setInput(inpBlob)

  output = net.forward()
  #print(output)
  print("time taken by network : {:.3f}".format(time.time() - t))

    # Empty list to store the detected keypoints
  points = []

  for i in range(nPoints):
    # confidence map of corresponding body's part.
    probMap = output[0, i, :, :]
    probMap = cv2.resize(probMap, (frameWidth, frameHeight))

    # Find global maxima of the probMap.
    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
    #print(probMap)
    #print(point)
    #print()
        
    if prob > threshold :
        point= list(point)
        addx= point[0]+xmin
        addy= point[1]+ymin
        point= tuple(point)
        cv2.circle(frameCopy, (int(point[0]), int(point[1])), 2, (27, 74, 29), thickness=1, lineType=cv2.FILLED)
        cv2.putText(frameCopy, "{}".format(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 255), 2, lineType=cv2.LINE_AA)


        #cv2.circle(image_np, (int(addx), int(addy)), 8, (27, 74, 29), thickness=1, lineType=cv2.FILLED)
        # Add the point to the list if the probability is greater than the threshold
        points.append((int(point[0]), int(point[1])))
        print(int(point[0]),int(point[1]))
    else :
        points.append(None)
  
  img_matrix_list.append(frameCopy)

  # Draw Skeleton
  for pair in POSE_PAIRS:
    partA = pair[0]
    partB = pair[1]

    if points[partA] and points[partB]:
        cv2.line(frame, points[partA], points[partB], (0, 255, 255), 8)
        cv2.circle(frame, points[partA], 2, (0, 0, 255), thickness=2, lineType=cv2.FILLED)
        cv2.circle(frame, points[partB], 2, (0, 0, 255), thickness=2, lineType=cv2.FILLED)

  img_matrix_list.append(frame)
  nymin= frame.shape[0] // 4
  nymax= nymin + frame.shape[0] // 2
  nxmin= frame.shape[1] // 4
  nxmax= nxmin + frame.shape[1] // 2
  #r_imgs.append(frame[nymin:nymax,nxmin:nxmax,:])

  count=count+2

titles_list = ["Hand 1","Hand1","Hand 2","Hand 2"]
plot_multiple_img(img_matrix_list, titles_list, ncols = 2)
#https://stackoverflow.com/questions/20925818/algorithm-to-check-if-two-boxes-overlap
# Overlap check for 1D rectangle
def isOverlapping1D(box1,box2):
    #print(box1,box2)
    xmax1= box1[1]
    xmin1= box1[0]
    xmax2= box2[1]
    xmin2= box2[0]
    return xmax1 >= xmin2 and xmax2 >= xmin1

def isOverlapping2D(box1,box2):
    return isOverlapping1D(box1['x'], box2['x']) and isOverlapping1D(box1['y'], box2['y'])
box1 = {'x':(hand_boxes[0][0],hand_boxes[0][1]),'y':(hand_boxes[0][2],hand_boxes[0][3])}
box2 = {'x':(hand_boxes[1][0],hand_boxes[1][1]),'y':(hand_boxes[1][2],hand_boxes[1][3])}

if isOverlapping2D(box1,box2):
    print("The bounding boxes are overlapping.")
else:
    print("The bounding boxes are not ovelapping")

def adjust_bbox(hand_boxes):
    updated_hand_boxes=[]
    for box in hand_boxes:
        xmin= int(box[0])
        xmax= int(box[1])
        ymin= int(box[2])
        ymax= int(box[3])
        
        xmin= xmin - (xmax-xmin)/2
        xmax= xmax + (xmax-xmin)/2
        
        updated_box= [xmin,xmax,ymin,ymax]
        updated_hand_boxes.append(updated_box)
    
    return updated_hand_boxes
updated_hand_boxes= adjust_bbox(hand_boxes)
print(updated_hand_boxes)
print(hand_boxes)
count=1
img_matrix_list=[]
r_imgs=[]
for box in updated_hand_boxes:
  xmin= int(box[0])
  xmax= int(box[1])
  ymin= int(box[2])
  ymax= int(box[3])
  print(ymin,ymax,xmin,xmax)
  result= image_np[ymin:ymax,xmin:xmax,:]
  #im_width = 2*result.shape[1]
  #im_height = 2*result.shape[0]
  #frame= Reformat_Image(result,im_width,im_height)
  frame= result
  frameCopy = np.copy(frame)
  print(frame.shape)
  frameWidth = frame.shape[1]
  frameHeight = frame.shape[0]
  print(frameWidth,frameHeight)
  aspect_ratio = frameWidth/frameHeight

  threshold = 0.5

  t = time.time()
  # input image dimensions for the network
  inHeight = 368
  inWidth = int(((aspect_ratio*inHeight)*8)//8)
  inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
  print(inpBlob.shape)
  net.setInput(inpBlob)

  output = net.forward()
  #print(output)
  print("time taken by network : {:.3f}".format(time.time() - t))

    # Empty list to store the detected keypoints
  points = []

  for i in range(nPoints):
    # confidence map of corresponding body's part.
    probMap = output[0, i, :, :]
    probMap = cv2.resize(probMap, (frameWidth, frameHeight))

    # Find global maxima of the probMap.
    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
    #print(probMap)
    #print(point)
    #print()
        
    if prob > threshold :
        point= list(point)
        addx= point[0]+xmin
        addy= point[1]+ymin
        point= tuple(point)
        cv2.circle(frameCopy, (int(point[0]), int(point[1])), 2, (27, 74, 29), thickness=1, lineType=cv2.FILLED)
        cv2.putText(frameCopy, "{}".format(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 255), 2, lineType=cv2.LINE_AA)


        #cv2.circle(image_np, (int(addx), int(addy)), 8, (27, 74, 29), thickness=1, lineType=cv2.FILLED)
        # Add the point to the list if the probability is greater than the threshold
        points.append((int(point[0]), int(point[1])))
        print(int(point[0]),int(point[1]))
    else :
        points.append(None)
  
  img_matrix_list.append(frameCopy)

  # Draw Skeleton
  for pair in POSE_PAIRS:
    partA = pair[0]
    partB = pair[1]

    if points[partA] and points[partB]:
        cv2.line(frame, points[partA], points[partB], (0, 255, 255), 8)
        cv2.circle(frame, points[partA], 2, (0, 0, 255), thickness=2, lineType=cv2.FILLED)
        cv2.circle(frame, points[partB], 2, (0, 0, 255), thickness=2, lineType=cv2.FILLED)

  img_matrix_list.append(frame)
  #nymin= frame.shape[0] // 4
  #nymax= nymin + frame.shape[0] // 2
  #nxmin= frame.shape[1] // 4
  #nxmax= nxmin + frame.shape[1] // 2
  r_imgs.append(frame)

  count=count+2

titles_list = ["Hand 1-Keypoint","Hand1-Skeleton","Hand 2-keypoint","Hand 2-skeleton"]
plot_multiple_img(img_matrix_list, titles_list, ncols = 2)
background = Image.fromarray(image_np)
for i in range(len(r_imgs)):
  os2= (int(updated_hand_boxes[i][0]),int(updated_hand_boxes[i][2]))
  print(os2)
  background.paste(Image.fromarray(r_imgs[i]), os2)

result= np.array(background)
plt.imshow(result)
cv2.imwrite('Final_result.jpg',result)
