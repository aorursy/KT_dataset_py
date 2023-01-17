import cv2
import numpy as np
import matplotlib.pyplot as plt 
import os
def imageGaussianBlur():
    
    img = cv2.imread('../input/imagea-001/A_001.png')
    
    matrix = (7,7)
    blur =cv2.GaussianBlur(img,matrix,0)
    
    plt.imshow(blur)
    plt.show()
    
def imageMedianBlur(): 
    
    img = cv2.imread('../input/imagea-001/A_001.png')
    
    kernel = 3
    blur =cv2.medianBlur(img,kernel)
    
    plt.imshow(blur)
    plt.show()
    
def imageBilateralFilter():
    
    img = cv2.imread('../input/imagea-001/A_001.png')
    
    dimensionalpixel =7
    color =100
    space =100 
    filter=cv2.bilateralFilter(img,dimensionalpixel,color,space)
    
    plt.imshow(filter)
    plt.show()
imageGaussianBlur()
imageMedianBlur()
imageBilateralFilter()
def imageThresholding():
    
    '''0 meaning black and 1 meaning black'''
    
    img = cv2.imread('../input/imagea-001/A_001.png',0)
    
    cols = img.shape[1]
    rows = img.shape[0]
    
    '''define the pixels value, if pixel value below that range then should turn in zero meaning blac'''
    '''if greater than this number then return to white'''
    
    threshold_value = 100 
    
    (T_value,binary_threshold) = cv2.threshold(img,threshold_value,255,cv2.THRESH_BINARY)
     
    '''T Value mean number gonna be return, binary threshold is a function'''
    
    plt.imshow(binary_threshold)
    plt.show()
    
imageThresholding()
def imageCannyEdgeDetection():
    
    img = cv2.imread('../input/imagea-001/A_001.png')
    
    '''Canny Edge Detection is used to detect the edges in an image. '''
    '''It accepts a gray scale image as input and it uses a multistage algorithm.'''
    
    '''Canny(image, edges, threshold1, threshold2)'''
    
    '''This method accepts the following parameters −'''

    '''image − A Mat object representing the source (input image) for this operation.'''
    '''edges − A Mat object representing the destination (edges) for this operation.'''
    '''threshold1 − A variable of the type double representing the first threshold for the hysteresis procedure.'''
    '''threshold2 − A variable of the type double representing the second threshold for the hysteresis procedure.'''
    
    threshold1 =300
    threshold2 =300
    
    canny =cv2.Canny(img,threshold1,threshold2)
    
    plt.imshow(canny)
    plt.show()
imageCannyEdgeDetection()
def drawRectangle():
    
    '''mat − A Mat object representing the image on which the rectangle is to be drawn.'''
    '''pt1 and pt2 − Two Point objects representing the vertices of the rectangle that is to be drawn.'''
    '''scalar − A Scalar object representing the color of the rectangle. (BGR)'''
    '''thickness − An integer representing the thickness of the rectangle; by default, the value of thickness is 1.'''
    
    '''how to create an image and draw a rectangle within it'''
    
    pic = np.zeros((500,500,3), dtype='uint8')
    
    '''pic is variable'''
    '''np.zeros is method from numpy that use to create numpy arrays that contain only zeros'''
    '''define an image how big is it ? then you use (500 x 500) = (500,500)'''
    '''you define the channel of the colour too it will be (500,500, 3) 3 is the channel of the colour its contain Red, Green and Blue'''
    
    '''cause of the range between 0 to 255 then you should define the datatype that is U INT 8 dtype =uint8'''
    
    cv2.rectangle(pic,(0,0),(500,150),(123,200,98),3, lineType=8, shift=0)
    
    '''pic = variable of matrix, (0,0) = p1, (500,150) = p2, (123,200,98) = scalar object for colour, 3 thickness of line, shift to resize the rectangle'''
    
    plt.imshow(pic)
    plt.show()
drawRectangle()
def imageRotation():
    
    '''You can perform rotation operation on an image using the warpAffine() method of the imgproc class.'''
    '''src − A Mat object representing the source (input image) for this operation.'''
    '''dst − A Mat object representing the destination (output image) for this operation.'''
    '''rotationMatrix − A Mat object representing the rotation matrix.'''
    '''size − A variable of the type integer representing the size of the output image.'''
    
    img = cv2.imread('../input/imagea-001/A_001.png')
    
    '''first make a matrix'''
    cols = img.shape[1]
    rows = img.shape[0]
    
    '''to define the center of the image, then the master said you should make half of the column and rows'''
    
    center = (cols/2,rows/2)
    
    '''90 degrees'''

    angle = 90 
    
    '''Creating the transformation matrix M'''
    
    M = cv2.getRotationMatrix2D(center,angle,1)
    
    '''Rotating the given image'''
    
    rotate = cv2.warpAffine(img,M,(cols,rows))
    
    plt.imshow(rotate)
    plt.show()
imageRotation()
def imageTransform():
    
    '''Affine Method'''
    '''src − A Mat object representing the source (input image) for this operation.'''
    '''dst − A Mat object representing the destination (output image) for this operation.'''
    '''tranformMatrix − A Mat object representing the transformation matrix.'''
    '''size − A variable of the type integer representing the size of the output image.'''
    
    img = cv2.imread('../input/imagea-001/A_001.png')
    
    '''first make a matrix'''
    cols = img.shape[1]
    rows = img.shape[0]
    
    '''Define the location of the image'''
    '''Here we Use matrix 1x 3 , use float for datatype'''
    
    m = np.float32([[1,0,150],[0,1,70]])
    
    '''shifted'''
    '''Imgproc.warpAffine(src, dst, tranformMatrix, size);'''
    
    s = cv2.warpAffine(img,m,(cols,rows))
    plt.imshow(s)
    plt.show()
imageTransform()
def resizeImage():
    
    img = cv2.imread('../input/imagea-001/A_001.png')
    image_scale = cv2.resize(img,(50,50),interpolation=cv2.INTER_AREA)
    plt.imshow(img)
    plt.show()
    
    plt.imshow(image_scale)
    plt.show()
resizeImage()
def writeImage():
    
    img = cv2.imread('../input/imagea-001/A_001.png')
    img = cv2.imwrite('A_002.png', img)
writeImage()
def writeText():
    
    '''mat − A Mat object representing the image to which the text is to be added.'''
    '''text − A string variable of representing the text that is to be added.'''
    '''org − A Point object representing the bottom left corner text string in the image.'''
    '''fontFace − A variable of the type integer representing the font type.'''
    '''fontScale − A variable of the type double representing the scale factor that is multiplied by the font-specific base size.'''
    '''scalar − A Scalar object representing the color of the text that is to be added. (BGR)'''
    '''thickness − An integer representing the thickness of the line by default, the value of thickness is 1.'''
    
    pic = np.zeros((500,500,3), dtype='uint8')
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(pic, 'Flo',(100,100), font,3,(255,255,255),4,cv2.LINE_8)
    
    '''putText = method, pic = a blank zeros array, Flo = value of the text/string/text, (100,100) cooridnate'''
    '''font = kind of font,3 = size of font, (255,255,255) = scalar of colour, 4 = thickness, cv2.LINE_8 = kind of line on the documentation'''
    
    plt.imshow(pic)
    plt.show()
writeText()
def Contour():
    image = cv2.imread("../input/imagea-001/A_001.png")
    image_scale = cv2.resize(image,(70,50),interpolation=cv2.INTER_AREA)
    dst = cv2.fastNlMeansDenoisingColored(image_scale,None,6,6,7,21)
    gray = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray, 400, 500)
    contours, hierarchy =cv2.findContours(edge, cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(image_scale,contours,-1,(0,255,0),1)

    plt.imshow(image_scale)
    plt.show()
    
    print("Number of Countour" + str(len(contours)))
Contour()
def Dilation():
    img = cv2.imread('../input/imagea-001/A_001.png') 
    kernel = np.ones((1,1), np.uint8) 
    img_dilation = cv2.dilate(img, kernel, iterations=1)
    
    plt.imshow(img)
    plt.show()
    
    plt.imshow(img_dilation)
    plt.show()
Dilation()
def Erosion():
    img = cv2.imread('../input/imagea-001/A_001.png') 
    kernel = np.ones((5,5), np.uint8) 
    img_erosion = cv2.erode(img, kernel, iterations=1)
    
    plt.imshow(img)
    plt.show()
    
    plt.imshow(img_erosion)
    plt.show()
Erosion()
