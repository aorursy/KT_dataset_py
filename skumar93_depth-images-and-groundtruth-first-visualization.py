! pip install --upgrade imutils

import numpy as np

import struct

import os

import array

import cv2

import matplotlib.pylab as plt

from IPython.display import clear_output

from scipy.signal import savgol_filter

from time import sleep

from skimage.feature import peak_local_max

from skimage.morphology import watershed

from scipy import ndimage

import numpy as np

import imutils
1

#w = savgol_filter(y, 10, 2)



#READ GROUNDTRUTH FILE

print("Opening GT")

#gt = open("../input/data2/data/seq-P02-M02-A0001-G02-C00-S0023/seq-P02-M02-A0001-G02-C00-S0023.gt")

path ="../input/data2/data/seq-P04-M04-A0001-G01-C00-S0033/seq-P04-M04-A0001-G01-C00-S0033.gt"

path1="../input/data2/data/seq-P04-M04-A0001-G01-C00-S0033/seq-P04-M04-A0001-G01-C00-S0033.z16"

gt = open(path)

groundtruth = gt.readlines()

gt.close()



person_index=[]

for i in range(0,len(groundtruth)):

    b = groundtruth[i].split(" ")

    person_index.append(int(b[0]))

person_index=np.array(person_index)

frame = 1

counter=0

aux=0

#cv2.namedWindow('image with groundtruth')



#READ BINARY FILE

print("Opening Z16")

num =0;

with open(path1, "rb") as f:

    while(False):

        

        arrx = []

        depthimage = array.array("h")

        depthimage.fromfile(f, 512*424)

        depthimage=np.array(depthimage)# ARRAY WITH POINTS IN MILIMETERS

        depthimage=np.reshape(depthimage,(424,512))# RESIZE TO KINECT V2 DEPTH RESOLUTION

        eight_bit_visualization=np.uint8(depthimage * (255 / np.max(depthimage))) #CONVERSION TO 8 BIT TO VISUALIZE IN A EASIER WAY

        found =0

        if(len(person_index[person_index==frame])>0):

            found = 1

            a=groundtruth[counter].split()#PARSE THE GROUNDTRUTH FILE

            counter=counter+1

            #PLOT GROUNDTRUTH POINTS ONLY FOR 1 PERSON

            temp= np.copy( eight_bit_visualization)

            if(len(a)==15):            

                for j in range(0,12,2):

                    x=int(float(a[j+3]))

                    y=int(float(a[j+4]))

                    #cv2.circle(eight_bit_visualization,(x,y), 3, (255), -1)

            for l in range (20,410):

                arrx.append(255-temp[l][256])

                cv2.circle(eight_bit_visualization,(256,l), 3, (255), -1)

                cv2.putText(eight_bit_visualization,''+str(frame),(50,50),cv2.FONT_HERSHEY_SIMPLEX,1, (255,255,255),2)

            arrx = savgol_filter(arrx, 9, 2) 

            del temp



            #IF YOU WANT THIS FOR MORE THAN ONE USER... YOU HAVE TO CONTINUE THE SEQUENCE OF IFS OR PARSE

            #THE POINTS OF THE GROUNDTRUTH IN ANOTHER WAY



        #PLOT THE SEQUENCE WITH THE GROUNDTRUTH POINTS PLOTTED IN WHITE

        #cv2.imshow('image with groundtruth',eight_bit_visualization)

        #cv2.waitKey(1)

        #if(aux==40):

        #print( "Visualized Frame %d" % (frame))

        if found == 1:

            plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

            plt.subplot(2, 1, 1)

            #f1= plt.figure()

            plt.imshow(eight_bit_visualization,cmap='bone',aspect="auto")

#             plt.subplot(2, 1, 2)

#             #f2 = plt.figure()

#             axes = plt.gca()

#             axes.set_ylim([0,350])

#             plt.bar(np.arange(len(arrx)),arrx)

            plt.show()

            #sleep(0.5)

            clear_output(wait=True)

            num +=1

        #aux=0

        frame=frame+1

        aux=aux+1

           

       
size = 512*424

file = None
def getFrame(frame,size=512*424):

    arr=[]

    global file

    if file is None:

          file= open(path1, "rb")

    file.seek(size*2*frame)

    buffer = file.read()

    outx = np.frombuffer(buffer,dtype=np.int16,count=size).reshape(424,512)

    #temp= np.copy(outx)

    out = np.uint8(outx * (255 / np.max(outx)))

    out = 255-out

    out[out>240] =0

    out[out<80] = 0

#     for l in range (20,410):

#                 val=np.max(temp)-temp[l][256]

#                 if val >= 3000: val=0

# #                 if val<3000 and val> 2000:

#                 arr.append(val)                   

#                 #cv2.circle(out,(256,l), 3, (255), -1)

#                 #cv2.putText(out,''+str(frame),(50,50),cv2.FONT_HERSHEY_SIMPLEX,1, (255,255,255),2)

#     plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

#     plt.subplot(2, 1, 1)

#     plt.imshow(out,cmap='bone',aspect="auto")

#     plt.subplot(2, 1, 2)

#     axes = plt.gca()

#     axes.set_ylim([2000,3000])

#     plt.bar(np.arange(len(arr)),arr)

# #     person = checkPerson(arr)

# #     if person ==1 :

# #         print("person detected")

#     plt.show()

    return out

    #return arr

   

   

    #clear_output(wait=True)

    

    

    
plt.imshow(image)

plt.show()
thresh = cv2.threshold(image, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

thresh[thresh==255]=1

plt.imshow(thresh)

plt.show()
np.max(thresh)
dilated = cv2.dilate(image,kernel,iterations=1)

plt.imshow(dilated)

plt.show()
res= dilated*thresh

plt.imshow(res)

plt.show()
final = res-image

plt.imshow(10*final)

plt.show()
final[thresh==0]=255

plt.imshow(final)

plt.show()
plt.imshow(255-final)

plt.show()
last = 255-final

last[last<248]=0

plt.imshow(last)
def findHead(image,kernel_size=101,thresh_rem=248):

    kernel = np.ones((kernel_size,kernel_size),np.uint8)

    thresh = cv2.threshold(image, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    thresh[thresh==255]=1

    dilated = cv2.dilate(image,kernel,iterations=1)

    res= dilated*thresh

    res= res-image

    res[thresh==0]=255

    res= 255-res

    res[res< thresh_rem] =0

    kernel2 = np.ones((11,11),np.uint8)

    res= cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel2)

    return res
for i in range (72,280):

    #print(i)

    out=getFrame(i)

    final=findHead(out)

    cnts = cv2.findContours(final.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    cnts = imutils.grab_contours(cnts)

    contours=[]

    for cnt in cnts:

        area = cv2.contourArea(cnt)

        if area > 500:

            #print(area)

            contours.append(cnt)

        

    #c = max(cnts, key=cv2.contourArea)

    count = 0

    for c in contours:

        count = count + 1

        ((x, y), r) = cv2.minEnclosingCircle(c)

        cv2.circle(out, (int(x), int(y)), int(r), (255, 255, 255), 2)

        cv2.putText(out, "#{}".format(str(count)), (int(x) - 10, int(y)),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    

    plt.imshow(out)

    plt.show()

    #sleep(0.2)

    clear_output(wait=True)
image = getFrame(377)

#shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)

kernel = np.ones((11,11),np.uint8)

#localMax =0

#def detect (image):

global localMax,D,gray

#image=cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)

plt.imshow(image)

plt.show()

#shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)

shifted = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

plt.imshow(shifted)

plt.show()

gray = shifted#cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)

thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

plt.imshow(thresh)

plt.show()

D = ndimage.distance_transform_edt(gray)

D = cv2.erode(D, kernel, iterations=3)

plt.imshow(D*gray)

plt.show()



localMax = peak_local_max(D*gray, indices=False, min_distance=50,exclude_border=True,labels=thresh)

#localMax = cv2.erode(localMax, kernel)



# perform a connected component analysis on the local peaks,

# using 8-connectivity, then appy the Watershed algorithm

markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]

labels = watershed(-D, markers, mask=thresh)

plt.imshow(labels)

plt.show()

print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))

# loop over the unique labels returned by the Watershed

# algorithm

for label in np.unique(labels):

    # if the label is zero, we are examining the 'background'

    # so simply ignore it

    if label == 0:

        continue



    # otherwise, allocate memory for the label region and draw

    # it on the mask

    mask = np.zeros(gray.shape, dtype="uint8")

    mask[labels == label] = 255



    # detect contours in the mask and grab the largest one

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,

        cv2.CHAIN_APPROX_SIMPLE)

    cnts = imutils.grab_contours(cnts)

    c = max(cnts, key=cv2.contourArea)



    # draw a circle enclosing the object

    ((x, y), r) = cv2.minEnclosingCircle(c)

    cv2.circle(image, (int(x), int(y)), int(r), (255, 255, 255), 2)

    cv2.putText(image, "#{}".format(label), (int(x) - 10, int(y)),

        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

 # show the output image
image = getFrame(55)

def detect(image,image1):

    #shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)

    #kernel = np.ones((11,11),np.uint8)

    #localMax =0

    #def detect (image):

    global localMax,D,gray

    #image=cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)

    #plt.imshow(image)

    #plt.show()

    #shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)

    #shifted = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    #plt.imshow(shifted)

    #plt.show()

    gray = image#cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)

    thresh = gray#cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    #plt.imshow(thresh)

    #plt.show()

    D = ndimage.distance_transform_edt(gray)

    #D = cv2.erode(D, kernel, iterations=3)

    #plt.imshow(D*gray)

    #plt.show()



    localMax = peak_local_max(D, indices=False, min_distance=40,labels=thresh)



    # perform a connected component analysis on the local peaks,

    # using 8-connectivity, then appy the Watershed algorithm

    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]

    labels = watershed(-D, markers, mask=thresh)

    #plt.imshow(labels)

    #plt.show()

    print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))

    # loop over the unique labels returned by the Watershed

    # algorithm

    for label in np.unique(labels):

        # if the label is zero, we are examining the 'background'

        # so simply ignore it

        if label == 0:

            continue



        # otherwise, allocate memory for the label region and draw

        # it on the mask

        mask = np.zeros(gray.shape, dtype="uint8")

        mask[labels == label] = 255



        # detect contours in the mask and grab the largest one

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,

            cv2.CHAIN_APPROX_SIMPLE)

        cnts = imutils.grab_contours(cnts)

        c = max(cnts, key=cv2.contourArea)



        # draw a circle enclosing the object

        ((x, y), r) = cv2.minEnclosingCircle(c)

        cv2.circle(image1, (int(x), int(y)), int(r), (255, 255, 255), 2)

        cv2.putText(image1, "#{}".format(label), (int(x) - 10, int(y)),

            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

     # show the output image

    #sleep(0.2)

    return image1

    #clear_output(wait=True)
for i in range (370,400):

    #print(i)

    out=getFrame(i)

    res= findHead(out)

    final=detect(res,out)

    plt.imshow(final)

    plt.show()

    #sleep(0.2)

    clear_output(wait=True)
from scipy import ndimage as ndi

import matplotlib.pyplot as plt

from skimage.feature import peak_local_max

from skimage import data, img_as_float



im = image = getFrame(578)



# image_max is the dilation of im with a 20*20 structuring element

# It is used within peak_local_max function

image_max = ndi.maximum_filter(im, size=20, mode='constant')



# Comparison between image_max and im to find the coordinates of local maxima

coordinates = peak_local_max(im, min_distance=20)



# display results

fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey=True)

ax = axes.ravel()

ax[0].imshow(im, cmap=plt.cm.gray)

ax[0].axis('off')

ax[0].set_title('Original')



ax[1].imshow(image_max, cmap=plt.cm.gray)

ax[1].axis('off')

ax[1].set_title('Maximum filter')



ax[2].imshow(im, cmap=plt.cm.gray)

ax[2].autoscale(False)

ax[2].plot(coordinates[:, 1], coordinates[:, 0], 'r.')

ax[2].axis('off')

ax[2].set_title('Peak local max')



fig.tight_layout()



plt.show()
im = getFrame(578)

image_max = ndi.maximum_filter(im, size=20, mode='constant')

coordinates = peak_local_max(im, min_distance=50)

# localMax = peak_local_max(image, indices=False, min_distance=50)#labels=gray)

# len(localMax[localMax==True])
plt.imshow(im)
len(coordinates)
coords=coordinates
for coord in coords:

    cv2.circle(im, (int(coord[1]), int(coord[0])), int(2), (255, 255, 255), 2) 
plt.imshow(im)
coords


cv2.circle(gray, (int(coords[0][0]), int(coords[1][0])), int(60), (255, 255, 255), 2)

cv2.circle(gray, (int(coords[0][1]), int(coords[1][1])), int(60), (255, 255, 255), 2)
plt.imshow(gray)
np.min(D)
np.max(gray)
np.min(gray)
test = (gray/230.0)*(gray/230.0)*(gray/230.0)*(gray/230.0)#*(D/35.0)
th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
plt.imshow(gray>120)

plt.show()
hist,bins = np.histogram(gray.ravel(),256,[0,256])
plt.plot(hist[1:])
er = cv2.morphologyEx(gray, cv2.MORPH_ERODE, kernel)
kernel = np.ones((70,70),np.uint8)
dl = cv2.morphologyEx(er, cv2.MORPH_DILATE, kernel)
plt.imshow(dl*1.0)

plt.show()
plt.hist(gray.ravel(),256,[0,256]); plt.show()
np.max(test)
def doVal(SH1,H1,H2,SH2):

    #check left shoulder

    shoulder = H1-SH1

    if shoulder < 15 or shoulder >50:

        return 0

    head = H2-H1

    if head <15 or head > 50:

        return 0

    shoulder2 = SH2-H2

    

    if shoulder2 < 15 or shoulder2 >50:

        return 0

    return 1

    #check head

    #check right shoulder
thresh_up=500

thresh_num =15

thresh_edge =10

def findStep(pos,arr,up):

    

    size = len(arr)

    for i in range(pos,size):

        start = i+thresh_edge

        end = i+thresh_edge + thresh_num

        if end > size:

            end = size

        for j in range(start, end):

            if up*(arr[j] - arr[i])< thresh_up:

                break;

            if j == i+thresh_num :

                return i

    return -1            

                

            
def checkPerson(arr):

    sz = len(arr)

    SH1= findStep(0,arr,1)

    if(SH1!=-1):

        H1 = findStep(SH1,arr,1)

        if H1 != -1:

            H2 = findStep(H1,arr,-1)

            if H2 != -1:

                SH2 = findStep(H2,arr,-1)

                if SH2 != -1:

                    return doValidations(SH1,H1,H2,SH2)

    return 0
final = np.uint8(out * (255 / np.max(out)))  

np.max(final)
len(arr)
final[0][0].dtype
f1.seek(512*424*2*170)

f2=f1.read()

frameNum = 1

a=np.frombuffer(f2,dtype=np.int16, count=512*424).reshape(424,512)
f2 = f1.seek(512*424)

f2
t=a * (255 / np.max(a))

plt.imshow(t)
np.max(a)
#f1= open("../input/data2/data/seq-P01-M02-A0001-G00-C00-S0001/seq-P01-M02-A0001-G00-C00-S0001.z16", "rb")

depthimage = array.array("h")

depthimage.fromfile(f1, 512*424)

depthimage=np.array(depthimage)# ARRAY WITH POINTS IN MILIMETERS

depthimage=np.reshape(depthimage,(424,512))# RESIZE TO KINECT V2 DEPTH RESOLUTION

eight_bit_visualization=np.uint8(depthimage * (255 / np.max(depthimage))) #CONVERSION TO 8 BIT TO VISUALIZE IN A EASIER WAY

plt.imshow(eight_bit_visualization)
np.int16