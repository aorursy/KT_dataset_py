import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import os

print(os.listdir("../input"))
DATA = '../input'

LABELS='train.csv'

TRAIN = os.path.join(DATA, 'train')

TEST = os.path.join(DATA, 'test')
train_paths = [os.path.join(TRAIN,img) for img in os.listdir(TRAIN)]

test_paths = [os.path.join(TEST,img) for img in os.listdir(TEST)]
import cv2

#read image

img=cv2.imread(train_paths[5])

blurred = cv2.GaussianBlur(img, (7,7), 0) # Remove noise

plt.imshow(blurred)
#close the small line gaps using errosion

kernel = np.ones((3,3), np.uint8)

erode = cv2.erode(blurred, kernel, iterations = 3)

plt.imshow(erode)
#cannyedge 

def canny_edge_detector(input_img, threshold1, threshold2, draw=True, save=True):

    canny_img = cv2.cvtColor(np.copy(input_img), cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(canny_img, threshold1, threshold2)

    return edges
#try adding Eroding before edge detection(increase black lines)

canny_edges = canny_edge_detector(input_img=erode, threshold1=100, threshold2=150) 

plt.imshow(canny_edges)
#close the small line gaps using dilation

kernel = np.ones((5,5), np.uint8)

dilation_canny = cv2.dilate(canny_edges, kernel, iterations = 3)

canny_blurred = cv2.GaussianBlur(dilation_canny, (3,3), 0) # Remove noise

plt.imshow(canny_blurred)
from skimage import measure

from shapely.geometry import Polygon,Point

min_contour_size = canny_blurred.size * 5 / 100

print("min size:"+str(min_contour_size))
#box=(x0,y0,x1,t1)

def calc_box_size(box):

    box_width=box[2]-box[0]

    box_hight=box[3]-box[1]

    box_area=box_width*box_hight

    return box_area



def bounding_rectangle(polygon):

  x0=min(polygon[:, 1])

  y0=min(polygon[:, 0])

  x1=max(polygon[:, 1])

  y1=max(polygon[:, 0])

  return x0,y0,x1,y1



def find_max_contour(image):

  contours = measure.find_contours(image.copy(), 0.8)

  max_area=0

  max_x=0

  max_y=0

  min_x=image.shape[0]

  min_y=image.shape[1]

  #get def_box

  for n, contour in enumerate(contours):

    contour[:, 1], contour[:, 0]

    max_c_x=max(contour[:, 1])

    max_c_y=max(contour[:, 0])

    min_c_x=min(contour[:, 1])

    min_c_y=min(contour[:, 0])

    if max_c_x>max_x:

      max_x=max_c_x

    if max_c_y>max_y:

      max_y=max_c_y

    if min_c_x<min_x:

      min_x=min_c_x

    if min_c_y<min_y:

      min_y=min_c_y

    

  def_box=(min_x,min_y,max_x,max_y)

  max_contour=None

  for n, contour in enumerate(contours):

    if contour.shape[0]<3: continue

    box=bounding_rectangle(contour)    

    box_size=calc_box_size(box)



    if max_contour is None:

      max_contour=contour

      max_area=box_size

    if box_size>max_area:

      max_contour=contour

      max_area=box_size

  return max_contour,max_area,def_box
contour,area,def_box=find_max_contour(canny_blurred) 

plt.imshow(img)

plt.plot(contour[:, 1], contour[:, 0], linewidth=1)
import matplotlib.patches as patches

box=bounding_rectangle(contour)

plt.imshow(img)

plt.plot(contour[:, 1], contour[:, 0], linewidth=1)

# Get the current reference

ax = plt.gca()

# Create a Rectangle patch

rect = patches.Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],linewidth=1,edgecolor='r',facecolor='none')

# Add the patch to the Axes

ax.add_patch(rect)

def_rect = patches.Rectangle((def_box[0],def_box[1]),def_box[2]-def_box[0],def_box[3]-def_box[1],linewidth=1,edgecolor='g',facecolor='none')

# Add the patch to the Axes

ax.add_patch(def_rect)

def_box
def get_box_center(box):

    #box polygon

    x0=box[0]

    y0=box[1]

    x1=box[2]

    y1=box[3]

    x2=x1

    y2=y0

    x3=x0

    y3=y1

    in_box=[[x0,y0],[x1,y1],[x2,y2],[x3,y3]]

    polygon_box = Polygon(in_box)

    box_centr=polygon_box.centroid.coords

    return box_centr



def get_serrounding_box_for_p(point,img_width,img_high,margin=0.2):

    x0=point[0]-margin*img_width

    y0=point[1]-margin*img_high

    x1=point[0]+margin*img_width

    y1=point[1]+margin*img_high

    return (x0,y0,x1,y1)



    

def validate_bb(image, box):

    if box is None:

        return False

    #check min size

    box_area=calc_box_size(box)

    min_contour_size = image.size * 5 / 100

    if box_area<min_contour_size:

        return False

    

    #box polygon

    box_centr=get_box_center(box)[0]

    

    #default polygon

    img_centr=get_box_center((0,0,image.shape[1],image.shape[0]))[0]

    srr_box=get_serrounding_box_for_p(img_centr,image.shape[1],image.shape[0],margin=0.2)

    

    #check box centered

    if  box_centr[0]>srr_box[0] and box_centr[0]< srr_box[2] and box_centr[1]>srr_box[1] and box_centr[1]<srr_box[3]:

        return True

    

    return False



print(validate_bb(img,box))
#get BB coordinates

def get_whale_bb(image_path):

    img=cv2.imread(image_path)

    blurred = cv2.GaussianBlur(img, (7,7), 0) # Remove noise

    kernel = np.ones((3,3), np.uint8) 

    erode = cv2.erode(blurred, kernel, iterations = 3)

    

    ##find edges

    canny_edges = canny_edge_detector(input_img=erode, threshold1=100, threshold2=150)   

    kernel = np.ones((5,5), np.uint8)

    dilation_canny = cv2.dilate(canny_edges, kernel, iterations = 3)#close the small line gaps using dilation

    canny_blurred = cv2.GaussianBlur(dilation_canny, (3,3), 0) # Remove noise

    

    ##find contour

    contour,area,def_box=find_max_contour(canny_blurred)

    

    ##find bb

    box=None

    if contour is not None:

        box=bounding_rectangle(contour)

        #check that box is not none, more than min size, with centroid in the center of image

        valid=validate_bb(img, box)

        if valid:

            return box

    

    valid=validate_bb(img, def_box)

    if valid:

        return def_box

    return None
bb_train = pd.DataFrame(columns=['image','x0','y0','x1','y1'])

for i in range(0,25):

    img_path=train_paths[i]

    bb=get_whale_bb(img_path)

    if bb is None:

        continue

    tmpdf=pd.DataFrame([[img_path,bb[0],bb[1],bb[2],bb[3]]],columns=['image','x0','y0','x1','y1'])

    bb_train=bb_train.append(tmpdf)



#look at examples

n=len(bb_train)

imgs_df=bb_train[:n].reset_index()

per_row=5

rows=n//per_row

cols      = min(per_row, n)

fig, axes = plt.subplots(rows,cols, figsize=(24//per_row*cols,24//per_row*rows))

for ax in axes.flatten(): 

    ax.axis('off')

for i,ax in enumerate(axes.flatten()): 

#     print (i)

    image_path=imgs_df.loc[i,'image']

    x0=float(imgs_df.loc[i,'x0'])

    y0=float(imgs_df.loc[i,'y0'])

    x1=float(imgs_df.loc[i,'x1'])

    y1=float(imgs_df.loc[i,'y1'])

    ax.imshow(cv2.imread(image_path))

    

    rect = patches.Rectangle((x0,y0),x1-x0,y1-y0,linewidth=1,edgecolor='r',facecolor='none')

    ax.add_patch(rect) 
#train bb 

print("total train images:"+str(len(train_paths)))

bb_train = pd.DataFrame(columns=['image','x0','y0','x1','y1'])

for i in range(len(train_paths)):

    if i%1000==0:

        print(i)

    img_path=train_paths[i]

    bb=get_whale_bb(img_path)

    if bb is None:

        continue

    tmpbb=pd.DataFrame([[img_path,bb[0],bb[1],bb[2],bb[3]]],columns=['Image','x0','y0','x1','y1'])

    bb_train=bb_train.append(tmpbb)



print("total croped train images:"+str(len(bb_train)))

bb_train.to_csv('boxs_train.csv', header=True, index=False)

print("finished!")
#test bb 

print("total test images:"+str(len(train_paths)))

bb_test = pd.DataFrame(columns=['image','x0','y0','x1','y1'])

for i in range(len(test_paths)):

    if i%1000==0:

        print(i)

    img_path=test_paths[i]

    bb=get_whale_bb(img_path)

    if bb is None:

        continue

    tmpbb=pd.DataFrame([[img_path,bb[0],bb[1],bb[2],bb[3]]],columns=['Image','x0','y0','x1','y1'])

    bb_test=bb_test.append(tmpbb)



print("total croped test images:"+str(len(bb_test)))

bb_test.to_csv('boxs_test.csv', header=True, index=False)

print("finished!")