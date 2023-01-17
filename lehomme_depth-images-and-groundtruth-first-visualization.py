import numpy as np
import struct
import os
import array
import cv2
import matplotlib.pylab as plt

#READ GROUNDTRUTH FILE
print("Opening GT")
gt = open("../input/data2/data/seq-P01-M02-A0001-G00-C00-S0001/seq-P01-M02-A0001-G00-C00-S0001.gt")
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
with open("../input/data2/data/seq-P01-M02-A0001-G00-C00-S0001/seq-P01-M02-A0001-G00-C00-S0001.z16", "rb") as f:
    while(1):
        try:
            
            depthimage = array.array("h")
            depthimage.fromfile(f, 512*424)
            depthimage=np.array(depthimage)# ARRAY WITH POINTS IN MILIMETERS
            depthimage=np.reshape(depthimage,(424,512))# RESIZE TO KINECT V2 DEPTH RESOLUTION
            eight_bit_visualization=np.uint8(depthimage * (255 / np.max(depthimage))) #CONVERSION TO 8 BIT TO VISUALIZE IN A EASIER WAY
            if(len(person_index[person_index==frame])>0):
                a=groundtruth[counter].split()#PARSE THE GROUNDTRUTH FILE
                counter=counter+1
                #PLOT GROUNDTRUTH POINTS ONLY FOR 1 PERSON
                if(len(a)==15):
                    for j in range(0,12,2):
                        x=int(float(a[j+3]))
                        y=int(float(a[j+4]))
                        cv2.circle(eight_bit_visualization,(x,y), 3, (255), -1)
                        

                #IF YOU WANT THIS FOR MORE THAN ONE USER... YOU HAVE TO CONTINUE THE SEQUENCE OF IFS OR PARSE
                #THE POINTS OF THE GROUNDTRUTH IN ANOTHER WAY

            #PLOT THE SEQUENCE WITH THE GROUNDTRUTH POINTS PLOTTED IN WHITE
            #cv2.imshow('image with groundtruth',eight_bit_visualization)
            #cv2.waitKey(1)
            if(aux==40):
                print( "Visualized Frame %d" % (frame))
                plt.figure(1)
                plt.imshow(eight_bit_visualization,cmap='bone')
                plt.show()
                aux=0
            frame=frame+1
            aux=aux+1
        except:
            print("error")
            break
