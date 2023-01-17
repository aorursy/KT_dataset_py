# This kernel imports code from codebase.py, a custom script I made.

# It is available at: https://www.kaggle.com/samuelepino/codebase



import os

import sys

sys.path.append("../usr/lib/codebase/")

from codebase import Segmenter, FigureManager



from pathlib import Path

import numpy as np

from PIL import Image

from skimage.segmentation import mark_boundaries
def loadImage(img, targetSize=None, normalize=False):

	imgData = Image.open(os.path.join(img["dirname"], img["filename"]))

	if (targetSize is not None):

		imgData = imgData.resize(targetSize, resample=Image.BILINEAR)

	if (normalize):

		imgData = np.asarray(imgData) / 255.

	return imgData



def newImgSequence(directory):



    imgInfoList = []

    for filename in sorted(os.listdir(directory)):

        imgInfoList.append({

            "dirname" : directory, 

            "filename" : filename, 

            "class" : 0

            })



    video = np.zeros((len(imgInfoList), 256, 256, 3))

    for idx, imgInfo in enumerate(imgInfoList):

        img = loadImage(imgInfo, targetSize=(256,256), normalize=True)

        video[idx] = img

    

    return video
sequenceName = "kxzgidrqse"

directory = os.path.join("../input/df-face-seq/", sequenceName)

video = newImgSequence(directory)



# show the first frame

FigureManager.saveAndDisplayImages([video[0]])
# SLICE 2D

seg = Segmenter(mode="color", segmentsNumber=100, segCompactness=10)



sliced2D = video.copy()

for idx, frame in enumerate(sliced2D):

    frame_slice = seg.segment(frame)

    sliced2D[idx] = mark_boundaries(frame, frame_slice)



# an example of output from slicing a single frame

seg.segment(video[0])
# SLICE 3D

seg = Segmenter(mode="color", segmentsNumber=100, segCompactness=10)



video_slice = seg.segment(video)

sliced3D = mark_boundaries(video, video_slice)



# an example of output from slicing a sequence

seg.segment(video[0:2])
# SLICE GRID 2D

seg = Segmenter(mode="grid2D", segmentsNumber=100)



video_slice = seg.segment(video)

slicedGrid = mark_boundaries(video, video_slice)



# an example of output from slicing a sequence

seg.segment(video[0])
FigureManager.saveAndDisplayGIF(sliced2D, f"{sequenceName}_sliced2D.gif", fps=10)
FigureManager.saveAndDisplayGIF(sliced3D, f"{sequenceName}_sliced3D.gif", fps=10)
FigureManager.saveAndDisplayGIF(slicedGrid, f"{sequenceName}_slicedGrid.gif", fps=10)