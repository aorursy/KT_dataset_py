import ssl

ssl._create_default_https_context = ssl._create_unverified_context

!pip install moviepy
import numpy as np

import matplotlib.pyplot as plt

import cv2.cv2 as cv2

import moviepy.editor as mpy
def colour_editor(all_images,images_direct,annotation_direct,segment,coeff,name):

    image_list=[]

    for i in range(all_images):

      

      number='000'+str(i)

      if len(number)<5:

        number='0'+ number

      number=number+'.jpg'

      print(annotation_direct +number.split('.')[0]+'.png')

      bgr_image=cv2.imread(images_direct+ number)

      image=cv2.cvtColor(bgr_image,cv2.COLOR_BGR2RGB)

      seg=cv2.imread(annotation_direct +number.split('.')[0]+'.png',cv2.IMREAD_GRAYSCALE)

      

      #the number of divisions and their values are calculated

      seg_map=list(set(seg.flatten()))

      value=seg_map[segment]      #the segment is choosen arbitrary

      

      #changes are in the colour channels of the choosen segment

      new_seg=(seg==value)*1

      new_seg=np.stack((coeff[0]*new_seg,coeff[1]*new_seg,coeff[2]*new_seg),axis=2)

      

      #the background remains unchanged

      background=(seg!=value)*1

      background=np.stack((background,background,background),axis=2)

      

      #multiply every pixel in the segment with the given coefficients

      image_with_the_segment=np.multiply(image,new_seg)  

      

      #obtain the background by setting to 0 other pixels

      image_without_the_segment=np.multiply(image,background)  

    

      

      img=np.add(image_with_the_segment,image_without_the_segment)

      img=img.astype(np.uint8)

      image_list.append(img)

      

      

      

    

    clip=mpy.ImageSequenceClip(image_list,fps=25)

    clip.write_videofile('part1_'+name +'.mp4',codec='mpeg4')



#colour_editor(40,'../input/jpegimages/','../input/annotations/',2,np.asarray([0.75,0.1,0.25]),'shooting')

#colour_editor(46,'../input/imagesrace/','../input/annot-race/',0,np.asarray([0,0.5,0.25]),'race')

#colour_editor(50,'../input/blackswan/', '../input/annot-blackswan/',0,np.asarray([0.9,1,0.1]),'swan')
def build_cdf(image,size):   #the source for this function are the lecture slides

    r,c,b = image.shape

    

    histogram = np.zeros([256,1,b],dtype=np.uint8)

    

    for g in range(256):

        histogram[g, 0,...] = np.sum(np.sum(image == g, 0), 0)

    

    pdf = histogram/size

    cdf = pdf.cumsum(axis=0)

    #cdf_normalized = cdf * hist.max()/ cdf.max() 

    #plt.plot(cdf[:,0,0],'r',cdf[:,0,1],'g',cdf[:,0,2],'b')

    return cdf

#build_cdf(cv2.imread("../input/jpegimages/00000.jpg"))
def create_lut(img,img_size,target,target_size):

    

    target_cdf=build_cdf(target,target_size)

    img_cdf=build_cdf(img,img_size)

    

    lut=np.zeros((256,1,3))

    for b in range(3):

        j=0

        for i in range(256):

            while target_cdf[j,0,b]< img_cdf[i,0,b] and j<255:

                j+=1

            lut[i,0,b]=j

    return lut



#g=cv2.imread("../input/targetimages/view.jpg")

#tr=cv2.imread("../input/targetimages/sea.jpg")

#l=create_lut(g,tr)

def matching_without_segmentation(path,all_images,target_path,name):

    

    frames=[]

    image_list=[]

    target_bgr=cv2.imread(target_path)

    targetim=cv2.cvtColor(target_bgr,cv2.COLOR_BGR2RGB)

    i=cv2.imread(path+'00000.jpg')

    whole_image=cv2.cvtColor(i,cv2.COLOR_BGR2RGB)

    for i in range(all_images):

        number='000'+str(i)

        if len(number)<5:

            number='0'+ number

        number=number+'.jpg'

        image=cv2.imread(path+ number)

        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        frames.append(image)

        whole_image=np.concatenate((whole_image,image))       #all images are concatenated together forming one single image

    

    t=np.shape(targetim)

    im=np.shape(whole_image)

    lut=create_lut(whole_image,im[0]*im[1],targetim,t[0]*t[1])

    for i in range(len(frames)):

        f=frames[i]

        l,w,c=f.shape

        for k in range(c):

            for j in range(w):

                for m in range(l):

                    pixel=f[m,j,k]

                    f[m,j,k]=lut[pixel,0,k]

        image_list.append(f)

    

    clip=mpy.ImageSequenceClip(image_list,fps=25)

    clip.write_videofile('part2_'+name+'.mp4',codec='mpeg4')



#matching_without_segmentation('../input/jpegimages/',40,'../input/targetimages/view.jpg','shooting')

#matching_without_segmentation('../input/blackswan/',50,'../input/targetimages/sea.jpg','swan')

#matching_without_segmentation('../input/imagesrace/',46,'../input/targetimages/gogh.jpg','race')
targets=[]

i=cv2.imread('../input/targetimages/akdenizheykeli.jpg')

i=cv2.cvtColor(i,cv2.COLOR_BGR2RGB)

targets.append(i)

i=cv2.imread('../input/targetimages/gogh.jpg')

i=cv2.cvtColor(i,cv2.COLOR_BGR2RGB)

targets.append(i)

i=cv2.imread('../input/targetimages/sea.jpg')

i=cv2.cvtColor(i,cv2.COLOR_BGR2RGB)

targets.append(i)

i=cv2.imread('../input/targetimages/view.jpg')

i=cv2.cvtColor(i,cv2.COLOR_BGR2RGB)

targets.append(i)
def matching_segementation(images_path,annot_path,target_list,all_images,name):

    pixel_count={}

    annot={}  

    image_list=[]

    res=(0,0,0)

    for i in range(all_images):

      

      number='000'+str(i)

      if len(number)<5:

        number='0'+ number

      number=number+'.jpg'

      bgr_image=cv2.imread(images_path+ number)

      image=cv2.cvtColor(bgr_image,cv2.COLOR_BGR2RGB)

      res=(image.shape)

      

      seg=cv2.imread(annot_path +number.split('.')[0]+'.png',cv2.IMREAD_GRAYSCALE)

      

      #the number of divisions and their values are calculated

      seg_map=list(set(seg.flatten()))

      for i in range(len(seg_map)):

        if seg_map[i]  not in annot:

            annot[seg_map[i]]=[]   #create empty list to store frames

            pixel_count[seg_map[i]]= 0

      

      for key in annot.keys():                              #division

          new_seg=(seg==key)*1

          pixels=np.count_nonzero(new_seg)

          pixel_count[key]+=pixels

          new_seg=np.stack((new_seg,new_seg,new_seg),axis=2)

          image_with_the_segment=np.multiply(image,new_seg)  

          annot[key].append(image_with_the_segment)

          

    

    for i,key in enumerate(annot):                    #histogram matching

      whole_image=np.concatenate(tuple(annot[key]),)  

      print(whole_image.shape)

      target=target_list[i]

      lut=create_lut(whole_image,pixel_count[key],target,target.shape[0]*target.shape[1])

      for p in range(all_images):

        l,w,c=annot[key][i].shape

        for k in range(c):

            for j in range(w):

                for m in range(l):

                    pixel=annot[key][p][m,j,k]

                    annot[key][p][m,j,k]=lut[pixel,0,k]

    

    

    

    frames=[]

    for i in range(all_images):          #addition

      frame=np.zeros(res)

      for key in annot.keys():

          frame=np.add(frame,annot[key][i])

          frame=frame.astype(np.uint8)

          frames.append(frame)

          #plt.imshow(img)

      

      

    

    clip=mpy.ImageSequenceClip(frames,fps=25)

    clip.write_videofile('part3_'+name +'.mp4',codec='mpeg4')

    

matching_segementation('../input/jpegimages/','../input/annotations/',targets,40,'shooting')

#matching_segementation('../input/imagesrace/','../input/annot-race/',targets,46,'race')

#matching_segementation('../input/blackswan/','../input/annot-blackswan/',targets,50,'swan')