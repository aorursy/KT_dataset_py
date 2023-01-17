import numpy as np
import cv2 
import os

def ExtractObject(datafile = "info_nike_demo.data", # annotation file
                  pathtowrite = "./Train/"):

    #open datafile
    f = open(datafile)
    content = f.read()
    i = 1
    for l in content.split('\n'):
        words = l.split()
        if(len(words) >= 6): # coz sometimes the images have no region. 
            img_path = ' '.join(words[:-5]) # path the positive sample file
            img_path = img_path.replace('\\','/') # replace back-slash if you are window user
            img = cv2.imread(img_path,0) # read the read the sample using OpenCV
            logo = [int(w) for w in words[-4:]] #extract the logo
            x,y,w,h = logo
            logograb = img[y:y+h, x:x+w]
            # keep the size small to keep the training time short
            img = cv2.resize(logograb, (60,20), interpolation = cv2.INTER_AREA)
            cv2.imwrite(pathtowrite + str(i) + '.jpg', img)
            i = i + 1
from sklearn.feature_extraction.image import PatchExtractor
from skimage import data, transform

def extract_patches(img, N, scale=1.0, patch_size=(100,100)):
    extracted_patch_size = tuple((scale * np.array(patch_size)).astype(int))
    extractor = PatchExtractor(patch_size=extracted_patch_size,
                               max_patches=N, random_state=0)
    patches = extractor.transform(img[np.newaxis])
    if scale != 1:
        patches = np.array([transform.resize(patch, patch_size)
                            for patch in patches])
    return patches


cokelogo_cascade = "C:/Users/rithanya/Documents/Python/Industrial_Safety/coke/cascade.xml"
cokecascade = cv2.CascadeClassifier(cokelogo_cascade)

#utility function to apply different cascade function on the images at difference scaleFactor
def detect(faceCascade, gray_,  scaleFactor_ = 1.1, minNeighbors = 5):
    faces = faceCascade.detectMultiScale(gray_,
                    scaleFactor= scaleFactor_,
                    minNeighbors=5,
                    minSize= (30,30), #(60, 20),
                    flags = cv2.CASCADE_SCALE_IMAGE
                )
    return faces

def DetectAndShow(imgfolder = 'NegFromAds/coca cola advertisements/'):
    cokelogo_cascade = "./cascade4/cokelogoorigfullds.xml"
    cokecascade = cv2.CascadeClassifier(cokelogo_cascade)
    for i in os.listdir(imgfolder):
        filepath = imgfolder + i
        img = cv2.imread(filepath)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cokelogos = detect(cokecascade, gray, 1.25, 6)
        for (x, y, w, h) in cokelogos:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imshow('positive samples',img)
        k = 0xFF & cv2.waitKey(0)
        if k == 27:         # q to exit
            break
