import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
def chocochip_count(imgname):
    
    img = cv.imread(imgname, 1) 
    scaled_cookie = cv.resize(img, (255,255)) #scale image

    #make image greyscale -> blur 
    bw_im = cv.cvtColor(scaled_cookie, cv.COLOR_BGR2GRAY)
    blur_im = cv.GaussianBlur(bw_im,(3,3),0)
    #thresholding to make chocochips more prominent
    ret,thresh_im = cv.threshold(blur_im, 85,255,cv.THRESH_BINARY)

    #initializing parameters for blob detection
    params = cv.SimpleBlobDetector_Params()

    #params.minThreshold = 0
    #params.maxThreshold = 100

    #initializing circularity of the chocochips
    params.filterByCircularity = True
    params.minCircularity = 0.18

    #setting the convexivity that is to be allowed
    params.filterByConvexity = True
    params.minConvexity = 0.2

    #setting how much elongation to be allowed
    params.filterByInertia = True
    params.minInertiaRatio = 0.1
    
    ver = (cv.__version__).split('.')
    if int(ver[0]) < 3 :  #if you have opencv ver2.x
        detector = cv.SimpleBlobDetector(params)
    else : 
        detector = cv.SimpleBlobDetector_create(params)   #for opencv 3.x and above

    #list of found blobs
    keypoints = detector.detect(im)

    #length of the list is number of chocochips
    return len(keypoints)
imgname = 'cookies/cookie1.jpg' #input the path of the cookie image to use
chocochip_count(imgname) #predict the number of chocolate chips