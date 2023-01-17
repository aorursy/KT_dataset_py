import tensorflow.keras as keras

from tensorflow.keras.preprocessing import image

import requests

from skimage.segmentation import slic

import matplotlib.pylab as pl

import numpy as np

import shap

from os.path import join

import os

from PIL import Image

import imageio

from IPython.display import Image as IImg, display

from tqdm import tqdm

import time
# copy mesonet package into our working dir and import it

from shutil import copytree

from os import path

mesonetSourcePath = "../input/mesonet"

mesonetDestPath = "../working/mesonet"

if (not path.exists(mesonetDestPath)):

    copytree(src=mesonetSourcePath, dst=mesonetDestPath)

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



def saveAndDisplayGIF(sequence, outputName="sequence.gif", FPS=5):

    with imageio.get_writer(outputName, mode='I', fps=FPS) as writer:

        for frame in sequence:

            writer.append_data(frame)

    display(IImg("../working/"+outputName))
from matplotlib.colors import LinearSegmentedColormap

# Fill every segment with the color relative to the shapley value

def fill_segmentation(values, segmentation):

    out = np.zeros(segmentation.shape)

    for i in range(len(values)):

        out[segmentation == i] = values[i]

    return out



def getExplanationFigure(img, imageTrueClass, preds, shap_values, segments_slic):

    # make a color map

    

    colors = []

    for l in np.linspace(1, 0, 100):

        colors.append((245/255, 39/255, 87/255, l))

    for l in np.linspace(0,1,100):

        colors.append((24/255, 196/255, 93/255, l))

    cm = LinearSegmentedColormap.from_list("shap", colors)



    # set first image (original image)

    fig, axes = pl.subplots(nrows=1, ncols=2, figsize=(6,4))

    axes[0].imshow(img)

    axes[0].set_title("class: {}, pred: {:.3f}".format(imageTrueClass, preds[0][0]))

    axes[0].axis('off')

    

    #max_val = np.max([np.max(np.abs(shap_values[0][:-1])) for i in range(len(shap_values))])

    max_val = 0.3

    # set second image (segmented and colored image)

    m = fill_segmentation(shap_values[0], segments_slic)

    axes[1].set_title("expl. for real")

    axes[1].imshow(image.array_to_img(img).convert('LA'), alpha=0.15)

    im = axes[1].imshow(m, cmap=cm, vmin=-max_val, vmax=max_val)

    axes[1].axis('off')

    

    # horizontal bar

    cb = fig.colorbar(im, ax=axes.ravel().tolist(), label="SHAP value", orientation="horizontal", aspect=60)

    cb.outline.set_visible(False)



    return fig
# define a function that depends on a binary mask representing if an image region is hidden

def mask_image(zs, segmentation, image, background=None):

    # zs: an array having dimensions (image, segmentsActiveValues)

    if background is None:

        background = image.mean((0,1))

    out = np.zeros((zs.shape[0], image.shape[0], image.shape[1], image.shape[2]))

    for i in range(zs.shape[0]):

        out[i,:,:,:] = image

        for j in range(zs.shape[1]):

            if zs[i,j] == 0:

                out[i][segmentation == j,:] = background

    return out
def explainFrame(model, img, imageTrueClass, runTimes):



    N_SEGMENTS = 50       # number of segments in the frame

    SEG_COMPACT = 20      # compactness of segments

    

    # segment the image so we don't have to explain every pixel

    segments_slic = slic(img, n_segments=N_SEGMENTS, compactness=SEG_COMPACT, sigma=3)

    

    # prediction function to pass to SHAP

    def f(z):

        maskingPatterns = z

        masked_images = mask_image(maskingPatterns, segments_slic, img)

        predictions = model.predict(masked_images)

        return predictions



    # use Kernel SHAP to explain the network's predictions

    background_data = np.zeros((1,N_SEGMENTS))   # https://shap.readthedocs.io/en/latest/#shap.KernelExplainer

    samples_features = np.ones(N_SEGMENTS)       # A vector of features on which to explain the model’s output.

    explainer = shap.KernelExplainer(f, background_data)

    shap_values = explainer.shap_values(samples_features, nsamples=runTimes, l1_reg="aic")

    

    # get the top predictions from the model

    preds = model.predict(np.array(img, ndmin=4))

    #top_preds = np.argsort(-preds)



    fig = getExplanationFigure(img, imageTrueClass, preds, shap_values, segments_slic)

    

    return fig
# Configuration

SEQUENCE_NAME = "obiwan"  # "kxzgidrqse"

MAX_FRAMES = 30

SHAP_SAMPLES = 500
# Load classifier and weights

classifier = MesoInception4()

classifier.load("./mesonet/weights/MesoInception_DF")

model = classifier.model

# output values for the classifier

REAL_CLASS_VAL = 1

FAKE_CLASS_VAL = 0

INPUT_SIZE = 256



# Get image list from folder

imgList = getImageList(f"../input/df-face-seq/{SEQUENCE_NAME}/", FAKE_CLASS_VAL, MAX_FRAMES)



print("Number of frames in the sequence:", len(imgList))
# create the image sequence

imageSequence = []

for idx, img in enumerate(imgList):

    inputDir = img["dirname"]

    imageFilename = img["filename"]

    

    # load an image

    file = join(inputDir, imageFilename)

    img = image.load_img(file, target_size=(INPUT_SIZE, INPUT_SIZE))

    imageSequence.append(image.img_to_array(img) / 255.)



#imageSequence = np.array(imageSequence)
# Create the video sequence as list of frames

fig_sequence = []



start_time = time.time()



for img in tqdm(imageSequence):

    imgClassDesc = '0 (fake)'# if img["class"]==FAKE_CLASS_VAL else '1 (real)'

    fig = explainFrame(model, img, imgClassDesc, SHAP_SAMPLES)

    fig_sequence.append(fig2array(fig)[:,:,:3])  # the [:,:,:3] is used to drop the forth color channel (alpha)

    pl.close('all')



print("--- Total time %s seconds ---" % (time.time() - start_time))

      

# Convert the list of frames into a 4D numpy array (frame, width, height, color)

fig_sequence = np.array(fig_sequence)



print("Dimensions of the fig_sequence array:", fig_sequence.shape)
saveAndDisplayGIF(fig_sequence, outputName=f"{SEQUENCE_NAME}_frames_expl.gif")
average_img_array = np.mean(fig_sequence, axis=0)

# Round values in array and cast as 8-bit integer

average_img_array = np.array(np.round(average_img_array), dtype=np.uint8)



average_img = Image.fromarray(average_img_array, mode="RGB")

average_img