import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import SimpleITK as sitk
path = '../input/'
reader = sitk.ImageSeriesReader() #create image reader object
filenamesDICOM = reader.GetGDCMSeriesFileNames(path) #get a series of file address
reader.SetFileNames(filenamesDICOM) #set all the address in image reader object
imgOriginal = reader.Execute() # read all the images
# This is the femous help function which helps you to show your image read by sitk.ReadImage object
# This help function is available everywhere in the internate
def sitk_show(img, title=None, margin=0.05, dpi=40 ):
    nda = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    figsize = (1 + margin) * nda.shape[0] / dpi, (1 + margin) * nda.shape[1] / dpi
    extent = (0, nda.shape[1]*spacing[1], nda.shape[0]*spacing[0], 0)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])

    plt.set_cmap("gray")
    ax.imshow(nda,extent=extent,interpolation=None)
img1 = imgOriginal[:,:,0] # select the first image
#Image smoothing function, to find suitable smooting play with image1,timeStep, numberOfIterations
imgSmooth = sitk.CurvatureFlow(image1=img1,
                                    timeStep=0.125,
                                    numberOfIterations=5)
sitk_show(img1)
sitk_show(imgSmooth)
imgWhiteMatter = sitk.ConnectedThreshold(image1=imgSmooth, 
                                              seedList=[(170,100)], 
                                              lower=170, 
                                              upper=900,
                                              replaceValue=1)
sitk_show(imgWhiteMatter)
imgSmoothInt = sitk.Cast(sitk.RescaleIntensity(imgSmooth), imgWhiteMatter.GetPixelID())
sitk.GetArrayFromImage(imgSmoothInt) 
# converted all pixel values to imgWhiteMatter pixel value type (Int) and re-scalled
# Pixel value type is same -> ready to overlay
overlay_image = sitk.LabelOverlay(imgSmoothInt, imgWhiteMatter)
sitk.GetArrayFromImage(overlay_image) 
# you can see, one image layer as RGB layer has been added. Look at the array start array([[[
sitk_show(overlay_image)
imgWhiteMatterNoHoles = sitk.VotingBinaryHoleFilling(image1=imgWhiteMatter,
                                                          radius=[2]*3,
                                                          majorityThreshold=1,
                                                          backgroundValue=0,
                                                          foregroundValue=1)

sitk_show(sitk.LabelOverlay(imgSmoothInt, imgWhiteMatterNoHoles))



