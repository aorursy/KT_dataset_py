import tensorflow.keras as keras

from tensorflow.keras.preprocessing import image

from PIL import Image

import imageio

import requests

from skimage.segmentation import slic

import numpy as np

import shap

import os

from os.path import join

from tqdm import tqdm

import time

import matplotlib.pyplot as plt



import sys

sys.path.append("../input/")

from mesonet.classifiers import *
# define a function that depends on a binary mask representing if an image region is hidden

def mask_image(zs, segmentation, image, background=None):

    # zs: an array having dimensions (image, segmentsActiveValues)

    if background is None:

        background = image.mean((0,1))

    out = np.zeros((image.shape[0], image.shape[1], image.shape[2]))

    

    out[:,:,:] = image

    for j in range(len(zs)):

        if zs[j] == 0:

            out[segmentation == j,:] = background

    return out
def explainFrame(model, img, runTimes, nSegments=50, segCompactness=20):

    

    # segment the image so we don't have to explain every pixel

    segments_slic = slic(img, n_segments=nSegments, compactness=segCompactness, sigma=3)

    

    # prediction function to pass to SHAP

    def f(z):

        maskingPatterns = z

        predictions = []

        for maskingPattern in maskingPatterns:

            masked_image = mask_image(maskingPattern, segments_slic, img)

            predictions.append(model.predict(np.array(masked_image, ndmin=4))[0])

        return np.array(predictions, ndmin=2)



    # use Kernel SHAP to explain the network's predictions

    background_data = np.zeros((1,N_SEGMENTS))   # https://shap.readthedocs.io/en/latest/#shap.KernelExplainer

    samples_features = np.ones(N_SEGMENTS)       # A vector of features on which to explain the model’s output.

    explainer = shap.KernelExplainer(f, background_data)

    shap_values = explainer.shap_values(samples_features, nsamples=runTimes, l1_reg="aic")

    

    return shap_values, segments_slic
# Configuration

MAX_FRAMES = 1        # how many frames to explain per sequence

# input/output values for the classifier

INPUT_SIZE = 256

REAL_CLASS_VAL = 1

FAKE_CLASS_VAL = 0

# Load classifier and weights

classifier = MesoInception4()

classifier.load("../input/mesonet/weights/MesoInception_DF")

model = classifier.model
file = "../input/mesonet-dataset-sfw/validation/df/val_df_179/179_18.jpg"



img = Image.open(file)

if ((img.width, img.height) != (INPUT_SIZE, INPUT_SIZE)):

    img = img.resize((INPUT_SIZE, INPUT_SIZE))

plt.imshow(img)

img = np.array(img) / 255.



# get the prediction from the model

preds = model.predict(np.array(img, ndmin=4))

print("True class:", FAKE_CLASS_VAL)

print("Predicted:", preds[0,0])
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
VARIANCE_SAMPLES = 30

SAVE_PLOT = True



# number of segments in the frame

segments_tests = [50, 100, 150, 200]

# how many times the frame should be masked and classified

samples_tests = [2**4, 2**5, 2**6, 2**7, 2**8, 2**9]



log = ""



for N_SEGMENTS in segments_tests:

    logLine = f"\n=== Number of segments: {N_SEGMENTS} ===\n\n"

    log += logLine

    print(logLine)



    variances = []

    for SHAP_SAMPLES in samples_tests:



        #print(f"Starting iteration with SHAP_SAMPLES = {SHAP_SAMPLES}")

        log += f"Shap samples: {SHAP_SAMPLES}\n"



        shap_values_list = []

        for _ in tqdm(range(VARIANCE_SAMPLES), desc=f"Iteration with SHAP_SAMPLES={SHAP_SAMPLES} "):

            

            with suppress_stderr():

                shap_values, segments_slic = explainFrame(model, img, SHAP_SAMPLES, nSegments=N_SEGMENTS)

            shap_values_list.append(shap_values)



        segment_variances = np.var(shap_values_list, axis=0)

        variances.append(np.mean(segment_variances))

        log += f"Mean variance: {np.mean(segment_variances)}\n\n"



    if (SAVE_PLOT):

        plt.plot(samples_tests, variances)

        plt.savefig(f"variance_{N_SEGMENTS}_segments.png")
print(log)