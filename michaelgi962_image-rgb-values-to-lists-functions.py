import os
from PIL import Image
import numpy
#Get a path for a image file
directory = '../input/memes/memes/'
fileNames = os.listdir(directory) #Store the list of file names as strings in an array
fileName = fileNames[1] #Get the name of a file
path = directory+fileName #Make the path
print(path)
#This function gets a list of the red values from an image
def imageRedValuesList(path):
    #Store the image as a image object 
    img = Image.open(path,'r')
    
    #Find the width, height, and area of the image
    width = img.width 
    height = img.height 
    area = width*height 
    
    #Store the image as a matrix of red, green, and blue tuples
    imgRGB = img.convert('RGB') #Convert the image to RBG
    imgRGBArray = numpy.array(imgRGB) #Convert the image to a numpy array of red, green, and, blue tuples
    #print(imgRGBArray)

    red = numpy.empty([area]) #Create an empty 1D numpy array the length of the image area
    count = -1 #Silly variable

    #Store the red values of an image in a array
    for j in range(height):
        for i in range(width):
            count = count+1 
            #Store the red value of a pixel in a array
            red[count] = imgRGBArray[j,i,0]  
    return red
print(imageRedValuesList(path))
#Get function gets a list of the blue values from an image
def imageBlueValuesList(path):
    #Store the image as a image object 
    img = Image.open(path,'r') 
    
    #Find the width, height, and area of the image
    width = img.width 
    height = img.height 
    area = width*height 
    
    #Store the image as a matrix of red, green, and blue tuples
    imgRGB = img.convert('RGB') #Convert the image to RBG
    imgRGBArray = numpy.array(imgRGB) #Convert the image to a numpy array of red, green, and, blue tuples
    #print(imgRGBArray)
    
    #Create a 1D numpy array
    blue = numpy.empty([area])
    count = -1 #Silly variable

    #Store the blue values of an image in a array
    for j in range(height):
        for i in range(width):
            count = count+1 
            #Store the blue value of the pixel in a array
            blue[count] = imgRGBArray[j,i,1]
    return blue
print(imageBlueValuesList(path))
#This function gets a list of the green values from an image
def imageGreenValuesList(path):
    #Store the image as a image object 
    img = Image.open(path,'r') 
    
    #Find the width, height, and area of the image
    width = img.width 
    height = img.height 
    area = width*height 
    
    #Store the image as a matrix of red, green, and blue tuples
    imgRGB = img.convert('RGB') #Convert the image to RBG
    imgRGBArray = numpy.array(imgRGB) #Convert the image to a numpy array of red, green, and, blue tuples
    #print(imgRGBArray)
    
    #Create a 1D numpy array
    green = numpy.empty([area])
    count = -1 #Silly variable

    #Store the green values of an image in a array
    for j in range(height):
        for i in range(width):
            count = count+1 
            #Store the green value of the pixel in a array
            green[count] = imgRGBArray[j,i,2]
    return green

print(imageGreenValuesList(path))
