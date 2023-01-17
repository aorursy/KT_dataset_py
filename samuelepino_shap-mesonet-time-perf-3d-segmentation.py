!nvcc --version
!pip install cupy-cuda101
from tensorflow.keras.preprocessing import image

import requests

from skimage.segmentation import slic

import numpy as np

import cupy as cp

import shap

from os.path import join

import os

from PIL import Image

import imageio

from tqdm import tqdm

import time



import sys

sys.path.append("../input/")

from mesonet.classifiers import *
import tensorflow as tf



z = np.zeros((256,256,1000))

start_time = time.time()

for _ in range(100):

    z += 1

print("Elapsed with Numpy: %s seconds" % (time.time() - start_time))

print(np.sum(z))

del z



z = cp.zeros((256,256,1000))

start_time = time.time()

for _ in range(100):

    z += 1

print("Elapsed with Cupy: %s seconds" % (time.time() - start_time))

print(cp.sum(z))

del z



z = tf.zeros((256,256,1000))

start_time = time.time()

for _ in range(100):

    z += 1

print("Elapsed with Tensorflow: %s seconds" % (time.time() - start_time))

print(tf.reduce_sum(z))

del z
# numpy image sequence from directory of images

def npSeqFromDir(directory, targetSize=None, normalize=True, frameLimit=np.infty):

    

    # take only the files for which the extension represents an image

    imageList = [img for img in sorted(os.listdir(directory)) if img[-4:].lower() in [".jpg",".jpeg",".png"] ]

    

    if (not targetSize):

        img = Image.open(os.path.join(directory, imageList[0]))

        targetSize = (img.width, img.height)

    

    sequence = np.zeros(( min(len(imageList),frameLimit), targetSize[0], targetSize[1], 3 ), dtype="uint8")

    

    for idx,filename in enumerate(imageList):

        # check the list length

        if (idx >= frameLimit):

            break

        # load the image with PIL

        img = Image.open(os.path.join(directory, filename))

        # resize if necessary

        imgSize = (img.width, img.height)

        if (imgSize != targetSize):

            img = img.resize(targetSize)

        # add the image to the list

        sequence[idx,:,:,:] = np.array(img)

    

    if (normalize):

        # return a numpy 4D array with values in [0,1]

        return sequence / 255.

    else:

        # return a numpy 4D array with values in [0,255]

        return sequence

    

# to suppress SHAP's warnings on numerical precision

from contextlib import contextmanager

import sys, os



@contextmanager

def suppress_stderr():

    with open(os.devnull, "w") as devnull:

        old_stderr = sys.stderr

        sys.stderr = devnull

        try:  

            yield

        finally:

            sys.stderr = old_stderr

            

@contextmanager

def suppress_stdout():

    with open(os.devnull, "w") as devnull:

        old_stdout = sys.stdout

        sys.stdout = devnull

        try:  

            yield

        finally:

            sys.stdout = old_stdout
# a function that depends on a binary mask representing if an image region is hidden

def mask_sequence(mask_pattern, segmentation : cp.ndarray, sequence : cp.ndarray, background=None) -> np.ndarray:

    # mask_pattern: an array having length 'nSegments'

    if background is None:

        background = sequence.mean((0,1,2))

    

    out = cp.zeros((sequence.shape[0], sequence.shape[1], sequence.shape[2], sequence.shape[3]))

    out[:,:,:,:] = sequence

    

    for j,segm_state in enumerate(mask_pattern):

        if (segm_state == 0):

            out[segmentation==j, :] = background

    return cp.asnumpy(out)
def f_mesonet(model, maskingPatterns, media : cp.ndarray, segments_slic, trackTime=False) -> np.array:



    hideProgressBar = (maskingPatterns.shape[0] <= 1 or trackTime)



    predictions = []



    mask_time = 0

    pred_time = 0



    # if it's an image

    if (len(media.shape)==3):

        avg = media.mean((0,1))

        for maskingPattern in maskingPatterns:

            masked_image = mask_image(maskingPattern, segments_slic, media, avg)

            preds = model.predict(np.array(masked_image, ndmin=4))[0]

            predictions.append(preds)

    

    # if it's a sequence

    elif (len(media.shape)==4):

        avg = media.mean((0,1,2))

        for idx, maskingPattern in tqdm(enumerate(maskingPatterns), disable=hideProgressBar):



            start_mask_time = time.time()

            masked_sequence = mask_sequence(maskingPattern, segments_slic, media, avg)

            mask_time += (time.time() - start_mask_time)



            start_pred_time = time.time()

            frames_preds = model.predict(np.array(masked_sequence, ndmin=4))

            pred_time += (time.time() - start_pred_time)



            video_pred = np.mean(np.array(frames_preds) > 0.5)

            predictions.append(np.array(video_pred, ndmin=1))



        if (trackTime and maskingPatterns.shape[0]>1):

            print("--- Masking:      %s seconds ---" % (mask_time))

            print("--- Predicting:   %s seconds ---" % (pred_time))



    return np.array(predictions, ndmin=2)
def explain(model, media : np.ndarray, shapSamples, nSegments=50, segCompactness=20, trackTime=False) -> tuple:

    

    # segment the image so we don't have to explain every pixel



    segmentation_time = time.time()

    # if it's an image

    if (len(media.shape)==3):

        segments_slic = cp.asarray(slic(media, n_segments=nSegments, compactness=segCompactness, sigma=3))

    # if it's a sequence

    elif (len(media.shape)==4):

        # sigma parameters is the size of the gaussian filter that pre-smooth the data

        # I defined the sigma as a triplet where the dimensions represent (time, image_x, image_y)

        segments_slic = cp.asarray(slic(media, n_segments=nSegments, compactness=segCompactness, sigma=(0.5,3,3)))

    else:

        print("Media shape not recognized:", media.shape)

        return

    if (trackTime):

    	print("--- Segmentation: %s seconds ---" % (time.time() - segmentation_time))



    # prediction function to pass to SHAP

    def f(z):

        # z: feature vectors from the point of view of shap

        #    from our point of view they are binary vectors defining which segments are active in the image.



        # converting the sequence/image to cupy array to use gpu

        return f_mesonet(model, z, cp.asarray(media), segments_slic, trackTime=trackTime)



    # use Kernel SHAP to explain the network's predictions

    background_data = np.zeros((1, nSegments))   # https://shap.readthedocs.io/en/latest/#shap.KernelExplainer

    samples_features = np.ones(nSegments)        # A vector of features on which to explain the model’s output.

    explainer = shap.KernelExplainer(f, background_data)

    shap_values = explainer.shap_values(samples_features, nsamples=shapSamples, l1_reg="aic")

    

    return shap_values, cp.asnumpy(segments_slic)
# Configuration

SEQUENCE_NAME = "obiwan"  # "kxzgidrqse"



# Load classifier and weights

classifier = MesoInception4()

classifier.load("../input/mesonet/weights/MesoInception_DF")

model = classifier.model

# output values for the classifier

REAL_CLASS_VAL = 1

FAKE_CLASS_VAL = 0
frames_tests = [10, 20, 40]

samples_tests = [250, 500, 1000, 2000, 4000]

N_SEGMENTS = 200



print("Performance tests on MesoInception")



for frame_num in frames_tests:

    print(f"\n=== Frames number: {frame_num} ===")

    

    # Get image list from folder

    imageSequence = npSeqFromDir(f"../input/df-face-seq/{SEQUENCE_NAME}/", targetSize=(256,256), frameLimit=frame_num)



    for samples_num in samples_tests:

        print(f"\n    === Samples number: {samples_num} ===\n")



        start_time = time.time()

        with suppress_stderr():

            explain(model, imageSequence, shapSamples=samples_num, nSegments=N_SEGMENTS, trackTime=True)

        print("--- Total time:   %s seconds ---" % (time.time() - start_time))