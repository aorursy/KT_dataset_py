# Importing the modules

import cv2

import numpy as np

import matplotlib.image as mpimg

import matplotlib.pyplot as plt

from PIL import Image

from pathlib import Path

import os

import ipywidgets as widgets

from IPython.display import display

from IPython.html.widgets import interactive

import log

from numpy import r_

import scipy

from scipy.fftpack import dct, idct

from numpy import pi

from numpy import sin

from numpy import zeros



import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
path = Path('/kaggle/input/alaska2-image-steganalysis')

folders = ['Cover', 'JMiPOD', 'JUNIWARD', 'UERD']



# Let us plot image histogram for cover and stego images in Gray scale:

img_num = 5



fig, ax = plt.subplots(figsize=(20, 10))



gray_img_cov = cv2.imread(f"{path}/{folders[0]}/0000{img_num}.jpg", cv2.IMREAD_GRAYSCALE)

gray_img_jmi = cv2.imread(f"{path}/{folders[1]}/0000{img_num}.jpg", cv2.IMREAD_GRAYSCALE)

gray_img_juni = cv2.imread(f"{path}/{folders[2]}/0000{img_num}.jpg", cv2.IMREAD_GRAYSCALE)

gray_img_uerd = cv2.imread(f"{path}/{folders[3]}/0000{img_num}.jpg", cv2.IMREAD_GRAYSCALE)





#################################

plt.subplot(141)

plt.title("Cover")

plt.imshow(gray_img_cov, cmap="gray")



plt.subplot(142)

plt.title("JMiPOD")

plt.imshow(gray_img_jmi, cmap="gray")



plt.subplot(143)

plt.title("JUNIWARD")

plt.imshow(gray_img_juni, cmap="gray")



plt.subplot(144)

plt.title("UERD")

plt.imshow(gray_img_uerd, cmap="gray")



plt.show()

#################################



hist_cov = cv2.calcHist([gray_img_cov],[0],None,[256],[0,256])

hist_jmi = cv2.calcHist([gray_img_jmi],[0],None,[256],[0,256])

hist_juni = cv2.calcHist([gray_img_juni],[0],None,[256],[0,256])

hist_uerd = cv2.calcHist([gray_img_uerd],[0],None,[256],[0,256])



#################################

fig, ax = plt.subplots(figsize=(20, 5))



plt.subplot(141)

plt.hist(gray_img_cov.ravel(),256,[0,256])

plt.title('Histogram for Cover')



plt.subplot(142)

plt.hist(gray_img_jmi.ravel(),256,[0,256])

plt.title('Histogram for JMiPOD')



plt.subplot(143)

plt.hist(gray_img_juni.ravel(),256,[0,256])

plt.title('Histogram for JUNIWARD')



plt.subplot(144)

plt.hist(gray_img_uerd.ravel(),256,[0,256])

plt.title('Histogram for UERD')



plt.show()

# Let us plot image histogram for cover and stego images in RGB scale:

img_num = 5



fig, ax = plt.subplots(figsize=(20, 10))



img_cov = cv2.imread(f"{path}/{folders[0]}/0000{img_num}.jpg", -1)

img_jmi = cv2.imread(f"{path}/{folders[1]}/0000{img_num}.jpg", -1)

img_juni = cv2.imread(f"{path}/{folders[2]}/0000{img_num}.jpg", -1)

img_uerd = cv2.imread(f"{path}/{folders[3]}/0000{img_num}.jpg", -1)





#################################

plt.subplot(141)

plt.title("Cover")

plt.imshow(img_cov)



plt.subplot(142)

plt.title("JMiPOD")

plt.imshow(img_jmi)



plt.subplot(143)

plt.title("JUNIWARD")

plt.imshow(img_juni)



plt.subplot(144)

plt.title("UERD")

plt.imshow(img_uerd)



plt.show()

#################################



histb_cov = cv2.calcHist([img_cov],[0],None,[256],[0,256])

histg_cov = cv2.calcHist([img_cov],[1],None,[256],[0,256])

histr_cov = cv2.calcHist([img_cov],[2],None,[256],[0,256])



histb_jmi = cv2.calcHist([img_jmi],[0],None,[256],[0,256])

histg_jmi = cv2.calcHist([img_jmi],[1],None,[256],[0,256])

histr_jmi = cv2.calcHist([img_jmi],[2],None,[256],[0,256])



histb_juni = cv2.calcHist([img_juni],[0],None,[256],[0,256])

histg_juni = cv2.calcHist([img_juni],[1],None,[256],[0,256])

histr_juni = cv2.calcHist([img_juni],[2],None,[256],[0,256])



histb_uerd = cv2.calcHist([img_uerd],[0],None,[256],[0,256])

histg_uerd = cv2.calcHist([img_uerd],[1],None,[256],[0,256])

histr_uerd = cv2.calcHist([img_uerd],[2],None,[256],[0,256])



#################################

fig, ax = plt.subplots(figsize=(20, 5))



plt.subplot(341)

plt.plot(histb_cov,color = 'b')

plt.title('Histogram for Cover channel-0 (B)')



plt.subplot(342)

plt.plot(histb_jmi,color = 'b')

plt.title('Histogram for JMiPOD channel-0 (B)')



plt.subplot(343)

plt.plot(histb_juni,color = 'b')

plt.title('Histogram for JUNIWARD channel-0 (B)')



plt.subplot(344)

plt.plot(histb_uerd,color = 'b')

plt.title('Histogram for UERD channel-0 (B)')





plt.subplot(345)

plt.plot(histg_cov,color = 'g')

plt.title('Histogram for Cover channel-1 (G)')



plt.subplot(346)

plt.plot(histg_jmi,color = 'g')

plt.title('Histogram for JMiPOD channel-1 (G)')



plt.subplot(347)

plt.plot(histg_juni,color = 'g')

plt.title('Histogram for JUNIWARD channel-1 (G)')



plt.subplot(348)

plt.plot(histg_uerd,color = 'g')

plt.title('Histogram for UERD channel-1 (G)')





plt.subplot(349)

plt.plot(histr_cov,color = 'r')

plt.title('Histogram for Cover channel-2 (R)')



plt.subplot(3,4,10)

plt.plot(histr_jmi,color = 'r')

plt.title('Histogram for JMiPOD channel-2 (R)')



plt.subplot(3,4,11)

plt.plot(histr_juni,color = 'r')

plt.title('Histogram for JUNIWARD channel-2 (R)')



plt.subplot(3,4,12)

plt.plot(histr_uerd,color = 'r')

plt.title('Histogram for UERD channel-2 (R)')



plt.tight_layout()

plt.show()

(histr_jmi == histr_cov).all()
# Ipython widget will allow you to select images from the list.

# Will consider only 100 images out of 75k

img_files = [f"0000{i}.jpg" if i<10 else f"000{i}.jpg"  for i in range(1,100)] 

def show_hist_gray(img_file):

    # Let us plot image histogram for cover and stego images in Gray scale:

    fig, ax = plt.subplots(figsize=(20, 10))

    print(f"{path}/{folders[0]}/{img_file}")

    gray_img_cov = cv2.imread(f"{path}/{folders[0]}/{img_file}", cv2.IMREAD_GRAYSCALE)

    gray_img_jmi = cv2.imread(f"{path}/{folders[1]}/{img_file}", cv2.IMREAD_GRAYSCALE)

    gray_img_juni = cv2.imread(f"{path}/{folders[2]}/{img_file}", cv2.IMREAD_GRAYSCALE)

    gray_img_uerd = cv2.imread(f"{path}/{folders[3]}/{img_file}", cv2.IMREAD_GRAYSCALE)





    #################################

    plt.subplot(141)

    plt.title("Cover")

    plt.imshow(gray_img_cov, cmap="gray")



    plt.subplot(142)

    plt.title("JMiPOD")

    plt.imshow(gray_img_jmi, cmap="gray")



    plt.subplot(143)

    plt.title("JUNIWARD")

    plt.imshow(gray_img_juni, cmap="gray")



    plt.subplot(144)

    plt.title("UERD")

    plt.imshow(gray_img_uerd, cmap="gray")



    plt.show()

    #################################



    hist_cov = cv2.calcHist([gray_img_cov],[0],None,[256],[0,256])

    hist_jmi = cv2.calcHist([gray_img_jmi],[0],None,[256],[0,256])

    hist_juni = cv2.calcHist([gray_img_juni],[0],None,[256],[0,256])

    hist_uerd = cv2.calcHist([gray_img_uerd],[0],None,[256],[0,256])



    #################################

    fig, ax = plt.subplots(figsize=(20, 5))



    plt.subplot(141)

    plt.hist(gray_img_cov.ravel(),256,[0,256])

    plt.title('Histogram for Cover')



    plt.subplot(142)

    plt.hist(gray_img_jmi.ravel(),256,[0,256])

    plt.title('Histogram for JMiPOD')



    plt.subplot(143)

    plt.hist(gray_img_juni.ravel(),256,[0,256])

    plt.title('Histogram for JUNIWARD')



    plt.subplot(144)

    plt.hist(gray_img_uerd.ravel(),256,[0,256])

    plt.title('Histogram for UERD')



#     plt.show()

opt = widgets.Select(options=img_files)

interactive(show_hist_gray, img_file=opt)
def show_hist_rgb(img_file):

    # Let us plot image histogram for cover and stego images in RGB scale:



    fig, ax = plt.subplots(figsize=(20, 10))



    img_cov = cv2.imread(f"{path}/{folders[0]}/{img_file}", -1)

    img_jmi = cv2.imread(f"{path}/{folders[1]}/{img_file}", -1)

    img_juni = cv2.imread(f"{path}/{folders[2]}/{img_file}", -1)

    img_uerd = cv2.imread(f"{path}/{folders[3]}/{img_file}", -1)





    #################################

    plt.subplot(141)

    plt.title("Cover")

    plt.imshow(img_cov)



    plt.subplot(142)

    plt.title("JMiPOD")

    plt.imshow(img_jmi)



    plt.subplot(143)

    plt.title("JUNIWARD")

    plt.imshow(img_juni)



    plt.subplot(144)

    plt.title("UERD")

    plt.imshow(img_uerd)



    plt.show()

    #################################



    histb_cov = cv2.calcHist([img_cov],[0],None,[256],[0,256])

    histg_cov = cv2.calcHist([img_cov],[1],None,[256],[0,256])

    histr_cov = cv2.calcHist([img_cov],[2],None,[256],[0,256])



    histb_jmi = cv2.calcHist([img_jmi],[0],None,[256],[0,256])

    histg_jmi = cv2.calcHist([img_jmi],[1],None,[256],[0,256])

    histr_jmi = cv2.calcHist([img_jmi],[2],None,[256],[0,256])



    histb_juni = cv2.calcHist([img_juni],[0],None,[256],[0,256])

    histg_juni = cv2.calcHist([img_juni],[1],None,[256],[0,256])

    histr_juni = cv2.calcHist([img_juni],[2],None,[256],[0,256])



    histb_uerd = cv2.calcHist([img_uerd],[0],None,[256],[0,256])

    histg_uerd = cv2.calcHist([img_uerd],[1],None,[256],[0,256])

    histr_uerd = cv2.calcHist([img_uerd],[2],None,[256],[0,256])



    #################################

    fig, ax = plt.subplots(figsize=(20, 5))



    plt.subplot(341)

    plt.plot(histb_cov,color = 'b')

    plt.title('Histogram for Cover channel-0 (B)')



    plt.subplot(342)

    plt.plot(histb_jmi,color = 'b')

    plt.title('Histogram for JMiPOD channel-0 (B)')



    plt.subplot(343)

    plt.plot(histb_juni,color = 'b')

    plt.title('Histogram for JUNIWARD channel-0 (B)')



    plt.subplot(344)

    plt.plot(histb_uerd,color = 'b')

    plt.title('Histogram for UERD channel-0 (B)')





    plt.subplot(345)

    plt.plot(histg_cov,color = 'g')

    plt.title('Histogram for Cover channel-1 (G)')



    plt.subplot(346)

    plt.plot(histg_jmi,color = 'g')

    plt.title('Histogram for JMiPOD channel-1 (G)')



    plt.subplot(347)

    plt.plot(histg_juni,color = 'g')

    plt.title('Histogram for JUNIWARD channel-1 (G)')



    plt.subplot(348)

    plt.plot(histg_uerd,color = 'g')

    plt.title('Histogram for UERD channel-1 (G)')





    plt.subplot(349)

    plt.plot(histr_cov,color = 'r')

    plt.title('Histogram for Cover channel-2 (R)')



    plt.subplot(3,4,10)

    plt.plot(histr_jmi,color = 'r')

    plt.title('Histogram for JMiPOD channel-2 (R)')



    plt.subplot(3,4,11)

    plt.plot(histr_juni,color = 'r')

    plt.title('Histogram for JUNIWARD channel-2 (R)')



    plt.subplot(3,4,12)

    plt.plot(histr_uerd,color = 'r')

    plt.title('Histogram for UERD channel-2 (R)')



    plt.tight_layout()

#     plt.show()



    
opt = widgets.Select(options=img_files)

interactive(show_hist_rgb, img_file=opt)
# Statistical analysis of an image to detect LSB steganography

def analyse(in_file):

	'''

	- Split the image into blocks.

	- Compute the average value of the LSBs for each block.

	- The plot of the averages should be around 0.5 for zones that contain

	  hidden encrypted messages (random data).

	'''

	BS = 100	# Block size 

	img = Image.open(in_file)

	(width, height) = img.size

	print("[+] Image size: %dx%d pixels." % (width, height))

	conv = img.convert("RGBA").getdata()



	# Extract LSBs

	vr = []	# Red LSBs

	vg = []	# Green LSBs

	vb = []	# LSBs

	for h in range(height):

		for w in range(width):

			(r, g, b, a) = conv.getpixel((w, h))

			vr.append(r & 1)

			vg.append(g & 1)

			vb.append(b & 1)



	# Average colours' LSB per each block

	avgR = []

	avgG = []

	avgB = []

	for i in range(0, len(vr), BS):

		avgR.append(np.mean(vr[i:i + BS]))

		avgG.append(np.mean(vg[i:i + BS]))

		avgB.append(np.mean(vb[i:i + BS]))



	# Nice plot 

	numBlocks = len(avgR)

	blocks = [i for i in range(0, numBlocks)]

	plt.axis([0, len(avgR), 0, 1])

	plt.ylabel('Average LSB per block')

	plt.xlabel('Block number')



#	plt.plot(blocks, avgR, 'r.')

#	plt.plot(blocks, avgG, 'g')

	plt.plot(blocks, avgB, 'bo')

	

# 	plt.show()
img_num = 1

fig, ax = plt.subplots(figsize=(20, 10))



plt.subplot(141)

plt.title("Cover")

analyse(f"{path}/{folders[0]}/0000{img_num}.jpg")



plt.subplot(142)

plt.title("JMiPOD")

analyse(f"{path}/{folders[1]}/0000{img_num}.jpg")



plt.subplot(143)

plt.title("JUNIWARD")

analyse(f"{path}/{folders[2]}/0000{img_num}.jpg")



plt.subplot(144)

plt.title("UERD")

analyse(f"{path}/{folders[3]}/0000{img_num}.jpg")
# source: https://github.com/qll/shit/blob/master/shit/analyse.py

def match_imgs(img1_path, img2_path):

    """Match images together and check properties like mode and dimension."""

    img1 = Image.open(img1_path)

    img2 = Image.open(img2_path)

    if img1.mode != img2.mode:

        log.warning('Unequal image modes (%s vs %s) - converting %s to %s',

                    img1.mode, img2.mode, img1.mode, img2.mode)

        img1 = img1.convert(img2.mode)

    if img1.width != img2.width or img1.height != img2.height:

        log.warning('Dimensions do not match ([%d, %d] vs [%d, %d])',

                    img1.width, img1.height, img2.width, img2.height)

    return img1, img2



def diffed_imgs(img1, img2):

    """Generator over all Pillow Image object pixel differences.

    Will run in the boundaries of min{width, height} of both images. It is the

    caller's responsibility to match image modes (RGB <-> RGBA).

    """

    for y in range(min(img1.height, img2.height)):

        for x in range(min(img1.width, img2.width)):

            img1_pixel = img1.getpixel((x, y))

            img2_pixel = img2.getpixel((x, y))

            if img1_pixel != img2_pixel:

                yield (x, y), img1_pixel, img2_pixel





def find_diffs(orig_path, stego_path, out_path=None):

    """Find differences in pixel values of the original and stego image."""

    orig, stego = match_imgs(orig_path, stego_path)

    out = Image.new('RGB', orig.size, (0, 0, 0)) if out_path else None

    for pos, orig_pixel, stego_pixel in diffed_imgs(orig, stego):

#         log.info('Mismatched pixels at %s: %s vs %s', pos, orig_pixel,

#                  stego_pixel)

        if out:

            out.putpixel(pos, (255, 255, 255))



    if out:

        out.save(out_path)

        

    im = plt.imread('/kaggle/working/output.jpg')

    plt.imshow(im)    

def show_diff_img(img_num):

    fig, ax = plt.subplots(figsize=(20, 20))



    img_cov = cv2.imread(f"{path}/{folders[0]}/0000{img_num}.jpg", -1)

    img_jmi = cv2.imread(f"{path}/{folders[1]}/0000{img_num}.jpg", -1)

    img_juni = cv2.imread(f"{path}/{folders[2]}/0000{img_num}.jpg", -1)

    img_uerd = cv2.imread(f"{path}/{folders[3]}/0000{img_num}.jpg", -1)



    plt.subplot(331)

    plt.title("Cover")

    plt.imshow(img_cov)



    plt.subplot(332)

    plt.title("JMiPOD")

    plt.imshow(img_jmi)



    plt.subplot(333)

    plt.title("Cover-JMiPOD")

    find_diffs(f"{path}/{folders[0]}/0000{img_num}.jpg", f"{path}/{folders[1]}/0000{img_num}.jpg", "output.jpg")





    plt.subplot(334)

    plt.title("Cover")

    plt.imshow(img_cov)



    plt.subplot(335)

    plt.title("JUNIWARD")

    plt.imshow(img_juni)



    plt.subplot(336)

    plt.title("Cover-JUNIWARD")

    find_diffs(f"{path}/{folders[0]}/0000{img_num}.jpg", f"{path}/{folders[2]}/0000{img_num}.jpg", "output.jpg")





    plt.subplot(337)

    plt.title("Cover")

    plt.imshow(img_cov)



    plt.subplot(338)

    plt.title("JUERD")

    plt.imshow(img_juni)



    plt.subplot(339)

    plt.title("Cover-UERD")

    find_diffs(f"{path}/{folders[0]}/0000{img_num}.jpg", f"{path}/{folders[3]}/0000{img_num}.jpg", "output.jpg")
show_diff_img(8)
show_diff_img(5)
img_cov = cv2.imread(f"{path}/{folders[0]}/0000{img_num}.jpg", cv2.IMREAD_GRAYSCALE)

img_jmi = cv2.imread(f"{path}/{folders[1]}/0000{img_num}.jpg", cv2.IMREAD_GRAYSCALE)

img_juni = cv2.imread(f"{path}/{folders[2]}/0000{img_num}.jpg", cv2.IMREAD_GRAYSCALE)

img_uerd = cv2.imread(f"{path}/{folders[3]}/0000{img_num}.jpg", cv2.IMREAD_GRAYSCALE)
def dct2(a):

    return scipy.fftpack.dct( scipy.fftpack.dct( a, axis=0, norm='ortho' ), axis=1, norm='ortho' )



def idct2(a):

    return scipy.fftpack.idct( scipy.fftpack.idct( a, axis=0 , norm='ortho'), axis=1 , norm='ortho')
imsize = img_cov.shape

dct = np.zeros(imsize)



# Do 8x8 DCT on image (in-place)

for i in r_[:imsize[0]:8]:

    for j in r_[:imsize[1]:8]:

        dct[i:(i+8),j:(j+8)] = dct2( img_cov[i:(i+8),j:(j+8)] )


pos = 128



# Extract a block from image

plt.figure()

plt.imshow(img_cov[pos:pos+8,pos:pos+8],cmap='gray')

plt.title( "An 8x8 Image block")



# Display the dct of that block

plt.figure()

plt.imshow(dct[pos:pos+8,pos:pos+8],cmap='gray',vmax= np.max(dct)*0.01,vmin = 0, extent=[0,pi,pi,0])

plt.title( "An 8x8 DCT block")
# Display entire DCT

plt.figure()

plt.imshow(img_cov)

plt.title( "orig image")
# Display entire DCT

plt.figure()

plt.imshow(dct,cmap='gray',vmax = np.max(dct)*0.01,vmin = 0)

plt.title( "DCTs of the image")
def show_dct_gray(img_num):

    # Let us plot image histogram for cover and stego images in Gray scale:

    fig, ax = plt.subplots(figsize=(20, 10))

    print(f"{path}/{folders[0]}/0000{img_num}.jpg")

    gray_img_cov = cv2.imread(f"{path}/{folders[0]}/0000{img_num}.jpg", cv2.IMREAD_GRAYSCALE)

    gray_img_jmi = cv2.imread(f"{path}/{folders[1]}/0000{img_num}.jpg", cv2.IMREAD_GRAYSCALE)

    gray_img_juni = cv2.imread(f"{path}/{folders[2]}/0000{img_num}.jpg", cv2.IMREAD_GRAYSCALE)

    gray_img_uerd = cv2.imread(f"{path}/{folders[3]}/0000{img_num}.jpg", cv2.IMREAD_GRAYSCALE)

    

    imsize = gray_img_cov.shape

    dct_cov = np.zeros(imsize)

    dct_jmi = np.zeros(imsize)

    dct_juni = np.zeros(imsize)

    dct_uerd = np.zeros(imsize)

    

    # Do 8x8 DCT on image (in-place)

    for i in r_[:imsize[0]:8]:

        for j in r_[:imsize[1]:8]:

            dct_cov[i:(i+8),j:(j+8)] = dct2( gray_img_cov[i:(i+8),j:(j+8)] )

            dct_jmi[i:(i+8),j:(j+8)] = dct2( gray_img_jmi[i:(i+8),j:(j+8)] )

            dct_juni[i:(i+8),j:(j+8)] = dct2( gray_img_juni[i:(i+8),j:(j+8)] )

            dct_uerd[i:(i+8),j:(j+8)] = dct2( gray_img_uerd[i:(i+8),j:(j+8)] )

    

 

   #################################

    

#     plt.suptitle('DCT on gray images')       fig, ax = plt.subplots(figsize=(20, 10))

    plt.subplot(141)

    plt.title("Cover")

    plt.imshow(gray_img_cov, cmap="gray")



    plt.subplot(142)

    plt.title("JMiPOD")

    plt.imshow(gray_img_jmi, cmap="gray")



    plt.subplot(143)

    plt.title("JUNIWARD")

    plt.imshow(gray_img_juni, cmap="gray")



    plt.subplot(144)

    plt.title("UERD")

    plt.imshow(gray_img_uerd, cmap="gray")



    plt.show()    

  



    #################################

#     plt.suptitle('Original gray images')

    fig, ax = plt.subplots(figsize=(20, 10))

    plt.subplot(141)

    plt.title("DCT on Cover")

    plt.imshow(dct_cov, cmap="gray", vmax = np.max(dct_cov)*0.01,vmin = 0)



    plt.subplot(142)

    plt.title("DCT on JMiPOD")

    plt.imshow(dct_jmi, cmap="gray", vmax = np.max(dct_jmi)*0.01,vmin = 0)



    plt.subplot(143)

    plt.title("DCT on JUNIWARD")

    plt.imshow(dct_juni, cmap="gray", vmax = np.max(dct_juni)*0.01,vmin = 0)



    plt.subplot(144)

    plt.title("DCT on UERD")

    plt.imshow(dct_uerd, cmap="gray", vmax = np.max(dct_uerd)*0.01,vmin = 0)



    plt.show()

 

    #########################################

    diff_cov_jmi = dct_cov - dct_jmi

    diff_cov_juni = dct_cov - dct_juni

    diff_cov_uerd = dct_cov - dct_uerd

    ########################################

    

    fig, ax = plt.subplots(figsize=(20, 10))

#     plt.suptitle('Difference in DCT coeff')

    plt.subplot(141)

    plt.title("DCT Cover")#.axis('off')

    plt.imshow(dct_cov, cmap="gray", vmax = np.max(dct_cov)*0.01,vmin = 0)



    plt.subplot(142)

    plt.title("DCT Cover - DCT JMiPOD")

    plt.imshow(diff_cov_jmi, cmap="gray", vmax = np.max(diff_cov_jmi)*0.01,vmin = 0)



    plt.subplot(143)

    plt.title("DCT Cover - DCT JUNIWARD")

    plt.imshow(diff_cov_juni, cmap="gray", vmax = np.max(diff_cov_juni)*0.01,vmin = 0)



    plt.subplot(144)

    plt.title("DCT Cover - DCT UERD")

    plt.imshow(diff_cov_uerd, cmap="gray", vmax = np.max(diff_cov_uerd)*0.01,vmin = 0)



    plt.show()

    



show_dct_gray(1)
show_dct_gray(5)
def show_dct_channel_wise(img_num):

    # Let us plot image histogram for cover and stego images in Gray scale:

    fig, ax = plt.subplots(figsize=(20, 10))

    print(f"{path}/{folders[0]}/0000{img_num}.jpg")

    img_cov = cv2.imread(f"{path}/{folders[0]}/0000{img_num}.jpg", -1)

    img_jmi = cv2.imread(f"{path}/{folders[1]}/0000{img_num}.jpg", -1)

    img_juni = cv2.imread(f"{path}/{folders[2]}/0000{img_num}.jpg", -1)

    img_uerd = cv2.imread(f"{path}/{folders[3]}/0000{img_num}.jpg", -1)

    

    imsize = img_cov.shape

    

    dct_cov = np.zeros(imsize)

    dct_jmi = np.zeros(imsize)

    dct_juni = np.zeros(imsize)

    dct_uerd = np.zeros(imsize) 

    

    diff_cov_jmi = np.zeros(imsize)

    diff_cov_juni = np.zeros(imsize)

    diff_cov_uerd = np.zeros(imsize) 

    

    

    # Do 8x8 DCT on image (in-place)

    for i in r_[:imsize[0]:8]:

        for j in r_[:imsize[1]:8]:

            dct_cov[:,:,0][i:(i+8),j:(j+8)] = dct2( img_cov[:,:,0][i:(i+8),j:(j+8)] )

            dct_jmi[:,:,0][i:(i+8),j:(j+8)] = dct2( img_jmi[:,:,0][i:(i+8),j:(j+8)] )

            dct_juni[:,:,0][i:(i+8),j:(j+8)] = dct2( img_juni[:,:,0][i:(i+8),j:(j+8)] )

            dct_uerd[:,:,0][i:(i+8),j:(j+8)] = dct2( img_uerd[:,:,0][i:(i+8),j:(j+8)] )



            dct_cov[:,:,1][i:(i+8),j:(j+8)] = dct2( img_cov[:,:,1][i:(i+8),j:(j+8)] )

            dct_jmi[:,:,1][i:(i+8),j:(j+8)] = dct2( img_jmi[:,:,1][i:(i+8),j:(j+8)] )

            dct_juni[:,:,1][i:(i+8),j:(j+8)] = dct2( img_juni[:,:,1][i:(i+8),j:(j+8)] )

            dct_uerd[:,:,1][i:(i+8),j:(j+8)] = dct2( img_uerd[:,:,1][i:(i+8),j:(j+8)] )

            

            dct_cov[:,:,2][i:(i+8),j:(j+8)] = dct2( img_cov[:,:,2][i:(i+8),j:(j+8)] )

            dct_jmi[:,:,2][i:(i+8),j:(j+8)] = dct2( img_jmi[:,:,2][i:(i+8),j:(j+8)] )

            dct_juni[:,:,2][i:(i+8),j:(j+8)] = dct2( img_juni[:,:,2][i:(i+8),j:(j+8)] )

            dct_uerd[:,:,2][i:(i+8),j:(j+8)] = dct2( img_uerd[:,:,2][i:(i+8),j:(j+8)] )

 

   #################################

    

#     plt.suptitle('DCT on gray images')       fig, ax = plt.subplots(figsize=(20, 10))

    plt.subplot(141)

    plt.title("Cover")

    plt.imshow(img_cov)



    plt.subplot(142)

    plt.title("JMiPOD")

    plt.imshow(img_jmi)



    plt.subplot(143)

    plt.title("JUNIWARD")

    plt.imshow(img_juni)



    plt.subplot(144)

    plt.title("UERD")

    plt.imshow(img_uerd)



    plt.show()    

  



    #################################

#     plt.suptitle('Original gray images')

    fig, ax = plt.subplots(figsize=(20, 10))

    plt.subplot(141)

    plt.title("DCT on Cover=>channel 0")

    plt.imshow(dct_cov[:,:,0], cmap="gray", vmax = np.max(dct_cov[:,:,0])*0.01,vmin = 0)



    plt.subplot(142)

    plt.title("DCT on JMiPOD=>channel 0")

    plt.imshow(dct_jmi[:,:,0], cmap="gray", vmax = np.max(dct_jmi[:,:,0])*0.01,vmin = 0)



    plt.subplot(143)

    plt.title("DCT on JUNIWARD=>channel 0")

    plt.imshow(dct_juni[:,:,0], cmap="gray", vmax = np.max(dct_juni[:,:,0])*0.01,vmin = 0)



    plt.subplot(144)

    plt.title("DCT on UERD=>channel 0")

    plt.imshow(dct_uerd[:,:,0], cmap="gray", vmax = np.max(dct_uerd[:,:,0])*0.01,vmin = 0)



    plt.show()

 

    #########################################

    diff_cov_jmi[:,:,0] = dct_cov[:,:,0] - dct_jmi[:,:,0]

    diff_cov_juni[:,:,0] = dct_cov[:,:,0] - dct_juni[:,:,0]

    diff_cov_uerd[:,:,0] = dct_cov[:,:,0] - dct_uerd[:,:,0]

    ########################################

    

    fig, ax = plt.subplots(figsize=(20, 10))

#     plt.suptitle('Difference in DCT coeff')

    plt.subplot(141)

    plt.title("DCT Cover=>channel 0")#.axis('off')

    plt.imshow(dct_cov[:,:,0], cmap="gray", vmax = np.max(dct_cov[:,:,0])*0.01,vmin = 0)



    plt.subplot(142)

    plt.title("DCT Cover - DCT JMiPOD : Channel 0")

    plt.imshow(diff_cov_jmi[:,:,0], cmap="gray", vmax = np.max(diff_cov_jmi[:,:,0])*0.01,vmin = 0)



    plt.subplot(143)

    plt.title("DCT Cover - DCT JUNIWARD : Channel 0")

    plt.imshow(diff_cov_juni[:,:,0], cmap="gray", vmax = np.max(diff_cov_juni[:,:,0])*0.01,vmin = 0)



    plt.subplot(144)

    plt.title("DCT Cover - DCT UERD : Channel 0")

    plt.imshow(diff_cov_uerd[:,:,0], cmap="gray", vmax = np.max(diff_cov_uerd[:,:,0])*0.01,vmin = 0)



    plt.show()

    

    



    #################################

    fig, ax = plt.subplots(figsize=(20, 10))

    plt.subplot(141)

    plt.title("DCT on Cover=>channel 1")

    plt.imshow(dct_cov[:,:,1], cmap="gray", vmax = np.max(dct_cov[:,:,1])*0.01,vmin = 0)



    plt.subplot(142)

    plt.title("DCT on JMiPOD=>channel 1")

    plt.imshow(dct_jmi[:,:,1], cmap="gray", vmax = np.max(dct_jmi[:,:,1])*0.01,vmin = 0)



    plt.subplot(143)

    plt.title("DCT on JUNIWARD=>channel 1")

    plt.imshow(dct_juni[:,:,1], cmap="gray", vmax = np.max(dct_juni[:,:,1])*0.01,vmin = 0)



    plt.subplot(144)

    plt.title("DCT on UERD=>channel 1")

    plt.imshow(dct_uerd[:,:,1], cmap="gray", vmax = np.max(dct_uerd[:,:,1])*0.01,vmin = 0)



    plt.show()

 

    #########################################

    diff_cov_jmi[:,:,1] = dct_cov[:,:,1] - dct_jmi[:,:,1]

    diff_cov_juni[:,:,1] = dct_cov[:,:,1] - dct_juni[:,:,1]

    diff_cov_uerd[:,:,1] = dct_cov[:,:,1] - dct_uerd[:,:,1]

    ########################################

    

    fig, ax = plt.subplots(figsize=(20, 10))

#     plt.suptitle('Difference in DCT coeff')

    plt.subplot(141)

    plt.title("DCT Cover=>channel 1")#.axis('off')

    plt.imshow(dct_cov[:,:,1], cmap="gray", vmax = np.max(dct_cov[:,:,1])*0.01,vmin = 0)



    plt.subplot(142)

    plt.title("DCT Cover - DCT JMiPOD : Channel 1")

    plt.imshow(diff_cov_jmi[:,:,1], cmap="gray", vmax = np.max(diff_cov_jmi[:,:,1])*0.01,vmin = 0)



    plt.subplot(143)

    plt.title("DCT Cover - DCT JUNIWARD : Channel 1")

    plt.imshow(diff_cov_juni[:,:,1], cmap="gray", vmax = np.max(diff_cov_juni[:,:,1])*0.01,vmin = 0)



    plt.subplot(144)

    plt.title("DCT Cover - DCT UERD : Channel 1")

    plt.imshow(diff_cov_uerd[:,:,1], cmap="gray", vmax = np.max(diff_cov_uerd[:,:,1])*0.01,vmin = 0)



    plt.show()  

    

    #################################

    fig, ax = plt.subplots(figsize=(20, 10))

    plt.subplot(141)

    plt.title("DCT on Cover=>channel 2")

    plt.imshow(dct_cov[:,:,2], cmap="gray", vmax = np.max(dct_cov[:,:,2])*0.01,vmin = 0)



    plt.subplot(142)

    plt.title("DCT on JMiPOD=>channel 2")

    plt.imshow(dct_jmi[:,:,2], cmap="gray", vmax = np.max(dct_jmi[:,:,2])*0.01,vmin = 0)



    plt.subplot(143)

    plt.title("DCT on JUNIWARD=>channel 2")

    plt.imshow(dct_juni[:,:,2], cmap="gray", vmax = np.max(dct_juni[:,:,2])*0.01,vmin = 0)



    plt.subplot(144)

    plt.title("DCT on UERD=>channel 2")

    plt.imshow(dct_uerd[:,:,2], cmap="gray", vmax = np.max(dct_uerd[:,:,2])*0.01,vmin = 0)



    plt.show()

 

    #########################################

    diff_cov_jmi[:,:,2] = dct_cov[:,:,2] - dct_jmi[:,:,2]

    diff_cov_juni[:,:,2] = dct_cov[:,:,2] - dct_juni[:,:,2]

    diff_cov_uerd[:,:,2] = dct_cov[:,:,2] - dct_uerd[:,:,2]

    ########################################

    

    fig, ax = plt.subplots(figsize=(20, 10))

#     plt.suptitle('Difference in DCT coeff')

    plt.subplot(141)

    plt.title("DCT Cover=>channel 2")#.axis('off')

    plt.imshow(dct_cov[:,:,2], cmap="gray", vmax = np.max(dct_cov[:,:,2])*0.01,vmin = 0)



    plt.subplot(142)

    plt.title("DCT Cover - DCT JMiPOD : Channel 2")

    plt.imshow(diff_cov_jmi[:,:,2], cmap="gray", vmax = np.max(diff_cov_jmi[:,:,2])*0.01,vmin = 0)



    plt.subplot(143)

    plt.title("DCT Cover - DCT JUNIWARD : Channel 2")

    plt.imshow(diff_cov_juni[:,:,2], cmap="gray", vmax = np.max(diff_cov_juni[:,:,2])*0.01,vmin = 0)



    plt.subplot(144)

    plt.title("DCT Cover - DCT UERD : Channel 2")

    plt.imshow(diff_cov_uerd[:,:,2], cmap="gray", vmax = np.max(diff_cov_uerd[:,:,2])*0.01,vmin = 0)



    plt.show()  

        

    

    



show_dct_channel_wise(1)
show_dct_channel_wise(2)
def show_dct_bgr(img_num):

    # Let us plot image histogram for cover and stego images in Gray scale:

    fig, ax = plt.subplots(figsize=(20, 10))

    print(f"{path}/{folders[0]}/0000{img_num}.jpg")

    img_cov = cv2.imread(f"{path}/{folders[0]}/0000{img_num}.jpg", -1)

    img_jmi = cv2.imread(f"{path}/{folders[1]}/0000{img_num}.jpg", -1)

    img_juni = cv2.imread(f"{path}/{folders[2]}/0000{img_num}.jpg", -1)

    img_uerd = cv2.imread(f"{path}/{folders[3]}/0000{img_num}.jpg", -1)

    

    imsize = img_cov.shape

    

    dct_cov = np.zeros(imsize)

    dct_jmi = np.zeros(imsize)

    dct_juni = np.zeros(imsize)

    dct_uerd = np.zeros(imsize) 

    

    

    # Do 8x8 DCT on image (in-place)

    for i in r_[:imsize[0]:8]:

        for j in r_[:imsize[1]:8]:

            dct_cov[:,:,0][i:(i+8),j:(j+8)] = dct2( img_cov[:,:,0][i:(i+8),j:(j+8)] )

            dct_jmi[:,:,0][i:(i+8),j:(j+8)] = dct2( img_jmi[:,:,0][i:(i+8),j:(j+8)] )

            dct_juni[:,:,0][i:(i+8),j:(j+8)] = dct2( img_juni[:,:,0][i:(i+8),j:(j+8)] )

            dct_uerd[:,:,0][i:(i+8),j:(j+8)] = dct2( img_uerd[:,:,0][i:(i+8),j:(j+8)] )



            dct_cov[:,:,1][i:(i+8),j:(j+8)] = dct2( img_cov[:,:,1][i:(i+8),j:(j+8)] )

            dct_jmi[:,:,1][i:(i+8),j:(j+8)] = dct2( img_jmi[:,:,1][i:(i+8),j:(j+8)] )

            dct_juni[:,:,1][i:(i+8),j:(j+8)] = dct2( img_juni[:,:,1][i:(i+8),j:(j+8)] )

            dct_uerd[:,:,1][i:(i+8),j:(j+8)] = dct2( img_uerd[:,:,1][i:(i+8),j:(j+8)] )

            

            dct_cov[:,:,2][i:(i+8),j:(j+8)] = dct2( img_cov[:,:,2][i:(i+8),j:(j+8)] )

            dct_jmi[:,:,2][i:(i+8),j:(j+8)] = dct2( img_jmi[:,:,2][i:(i+8),j:(j+8)] )

            dct_juni[:,:,2][i:(i+8),j:(j+8)] = dct2( img_juni[:,:,2][i:(i+8),j:(j+8)] )

            dct_uerd[:,:,2][i:(i+8),j:(j+8)] = dct2( img_uerd[:,:,2][i:(i+8),j:(j+8)] )

 

   #################################

    

#     plt.suptitle('DCT on gray images')       fig, ax = plt.subplots(figsize=(20, 10))

    plt.subplot(141)

    plt.title("Cover")

    plt.imshow(img_cov)



    plt.subplot(142)

    plt.title("JMiPOD")

    plt.imshow(img_jmi)



    plt.subplot(143)

    plt.title("JUNIWARD")

    plt.imshow(img_juni)



    plt.subplot(144)

    plt.title("UERD")

    plt.imshow(img_uerd)



    plt.show()    

  



    #################################

#     plt.suptitle('Original gray images')

    fig, ax = plt.subplots(figsize=(20, 10))

    plt.subplot(141)

    plt.title("DCT on Cover")

    plt.imshow(dct_cov)



    plt.subplot(142)

    plt.title("DCT on JMiPOD")

    plt.imshow(dct_jmi)



    plt.subplot(143)

    plt.title("DCT on JUNIWARD")

    plt.imshow(dct_juni)



    plt.subplot(144)

    plt.title("DCT on UERD")

    plt.imshow(dct_uerd)



    plt.show()

 

    #########################################

    

    diff_cov_jmi = dct_cov - dct_jmi

    diff_cov_juni = dct_cov - dct_juni

    diff_cov_uerd = dct_cov - dct_uerd

    ########################################

    

    fig, ax = plt.subplots(figsize=(20, 10))

#     plt.suptitle('Difference in DCT coeff')

    plt.subplot(141)

    plt.title("DCT Cover")#.axis('off')

    plt.imshow(dct_cov)



    plt.subplot(142)

    plt.title("DCT Cover - DCT JMiPOD")

    plt.imshow(diff_cov_jmi)



    plt.subplot(143)

    plt.title("DCT Cover - DCT JUNIWARD")

    plt.imshow(diff_cov_juni)



    plt.subplot(144)

    plt.title("DCT Cover - DCT UERD")

    plt.imshow(diff_cov_uerd)



    plt.show()

    

    

    

    



show_dct_bgr(1)
show_dct_bgr(2)