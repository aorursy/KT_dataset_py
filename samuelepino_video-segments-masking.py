# copy mesonet package into our working dir for later import

from shutil import copytree

from os import path

mesonetSourcePath = "../input/mesonet"

mesonetDestPath = "../working/mesonet"

if (not path.exists(mesonetDestPath)):

    copytree(src=mesonetSourcePath, dst=mesonetDestPath)



import tensorflow.keras as keras

from tensorflow.keras.applications.vgg16 import VGG16, decode_predictions # preprocess_input

from tensorflow.keras.preprocessing import image

import requests

from skimage.segmentation import slic, mark_boundaries

import matplotlib.pylab as pl

import numpy as np

import shap

from os.path import join

import os

from PIL import Image

import imageio

from IPython.display import Image as IImg, display



from mesonet.classifiers import *
# Return a list of all the images in a directory

def getImageList(directory, dirClass, samplesLimit=np.infty):



	imgList = []



	for filename in sorted(os.listdir(directory)):

		imgList.append({

			"dirname" : directory, 

			"filename" : filename, 

			"class" : dirClass

			})

	

	return imgList[0 : min(samplesLimit, len(imgList))]



from matplotlib.backends.backend_agg import FigureCanvas



# Convert figure to image as a numpy array

def fig2array(fig):

    canvas = FigureCanvas(fig)

    # Force a draw so we can grab the pixel buffer

    canvas.draw()

    # grab the pixel buffer and dump it into a numpy array

    return np.array(canvas.renderer.buffer_rgba())



# Prepare image array

def preprocess_input(image):

    array = np.array(image, ndmin=4)

    if (np.max(array) > 1):

        array /= 255.

    return array



def saveAndDisplayGIF(sequence, outputName="sequence.gif", FPS=5):

    with imageio.get_writer(outputName, mode='I', fps=FPS) as writer:

        for frame in sequence:

            writer.append_data(frame)

    display(IImg("../working/"+outputName))

    

def predict_sequence(sequence):

    p = []

    for image in sequence:

        p.append(model.predict(preprocess_input(sequence)))

    return np.mean(np.array(p) > 0.5)
# Configuration

#OUTPUT_DIR = "./kxzgidrqse"

SEQUENCE_NAME = "obiwan"  # "kxzgidrqse"



# Load classifier and weights

classifier = MesoInception4()

classifier.load("./mesonet/weights/MesoInception_DF")

model = classifier.model

# output values for the classifier

REAL_CLASS_VAL = 1

FAKE_CLASS_VAL = 0



# Get image list from folder

imgList = getImageList(f"../input/df-face-seq/{SEQUENCE_NAME}/", FAKE_CLASS_VAL)



print("Number of frames in the sequence:", len(imgList))
imageSequence = []

inputSize = 256

# create the image sequence

for idx, img in enumerate(imgList):

    inputDir = img["dirname"]

    imageFilename = img["filename"]

    

    # load an image

    file = join(inputDir, imageFilename)

    img = image.load_img(file, target_size=(inputSize, inputSize))

    imageSequence.append(image.img_to_array(img) / 255.)



imageSequence = np.array(imageSequence)

print("Image sequence shape:", imageSequence.shape)
# define a function that depends on a binary mask representing if an image region is hidden

def mask_image(zs, segmentation, image, background=None):

    if background is None:

        background = image.mean((0,1))

    out = np.zeros((zs.shape[0], image.shape[0], image.shape[1], image.shape[2]))

    for i in range(zs.shape[0]):

        out[i,:,:,:] = image

        for j in range(zs.shape[1]):

            if zs[i,j] == 0:

                out[i][segmentation == j,:] = background

    return out
SEGMENTS = 50

masked_fraction = 0.4
image = imageSequence[0]

segments_slic = slic(image, n_segments=SEGMENTS, compactness=20, sigma=3)



samples = 5

fig_size = 3

fig, axes = pl.subplots(nrows=1, ncols=samples, figsize=(samples*fig_size, fig_size))

axes[0].imshow(image)

axes[0].set_title("original")



print("Real class of original image: {}".format(FAKE_CLASS_VAL))

print("Prediction for original image: {:.4f}".format(model.predict(preprocess_input(image))[0,0]))



for i in range(1, samples):

    zs = np.random.choice([0, 1], size=(1,SEGMENTS), p=[masked_fraction, 1-masked_fraction])

    masked_img = mask_image(zs, segments_slic, image)[0]

    axes[i].imshow(masked_img)

    axes[i].set_title("mask {}".format(i))

    print("Prediction for mask {}: {:.4f}".format(i, model.predict(preprocess_input(masked_img))[0,0]))
# define a function that depends on a binary mask representing if an image region is hidden

def mask_sequence(zs, segmentation, sequence, background=None):

    # zs: matrix having dimensions (masking_patterns, num_segments) - deprecated

    # zs: a binary array of dimension num_segments

    if background is None:

        background = sequence.mean((0,1,2))

    out = np.zeros((zs.shape[0], sequence.shape[0], sequence.shape[1], sequence.shape[2], sequence.shape[3]))

    #print("out.size:", out.shape)

    

    for i in range(zs.shape[0]):

        out[i,:,:,:,:] = sequence

        for j in range(zs.shape[1]):

            if zs[i,j] == 0:

                out[i][segmentation == j,:] = background

    return out
# Original sequence

saveAndDisplayGIF(imageSequence, outputName="original.gif")
# Prediction on the original sequence

prediction = predict_sequence(imageSequence)

print("True class for the original sequence:", FAKE_CLASS_VAL)

print("Prediction for the original sequence:", prediction)
zs = np.random.choice([0, 1], size=(1,SEGMENTS), p=[masked_fraction, 1-masked_fraction])



print("Mask array (1 means on, 0 means off):")

print(zs)

print("Masked segments: {} out of {}.".format(np.count_nonzero(zs==0), zs.size))



segments_slic = slic(imageSequence, n_segments=SEGMENTS, compactness=20, sigma=(0.5,3,3))

masked_seq = mask_sequence(zs, segments_slic, imageSequence)[0]



saveAndDisplayGIF(mark_boundaries(masked_seq, segments_slic), outputName="segmented.gif")
# Prediction on the masked sequence

prediction = predict_sequence(masked_seq)

print("Prediction for the masked sequence:", prediction)