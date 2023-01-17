import tensorflow.keras as keras
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
    max_val = np.max([np.max(np.abs(shap_values[0][:-1])) for i in range(len(shap_values))])
    
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
def mask_sequence(mask_pattern, segmentation, sequence, background=None):
    
    if background is None:
        background = sequence.mean((0,1,2))
    
    out = np.zeros((sequence.shape[0], sequence.shape[1], sequence.shape[2], sequence.shape[3]))
    out[:,:,:,:] = sequence
    
    #start_time = time.time()
    for j,segm_state in enumerate(mask_pattern):
        if (segm_state == 0):
            out[segmentation == j,:] = background
    #print("--- M Computing masked out %s seconds ---" % (time.time() - start_time))
    return out
def explainSequence(model, imageSequence, imageTrueClass, runTimes):

    N_SEGMENTS = 50       # number of segments in the frame
    SEG_COMPACT = 20      # compactness of segments

    # segment the image so we don't have to explain every pixel
    # sigma parameters is the size of the gaussian filter that pre-smooth the data
    # I defined the sigma as a triplet where the dimensions represent (time, image_x, image_y)
    start_time = time.time()
    segments_slic = slic(imageSequence, n_segments=N_SEGMENTS, compactness=SEG_COMPACT, sigma=(0.5,3,3))
    print("--- Segmentation %s seconds ---" % (time.time() - start_time))
    
    seq_avg = imageSequence.mean((0,1,2))
    
    # prediction function to pass to SHAP
    def f(z):
        # z: feature vector from the point of view of shap
        #    from our point of view it is a binary vector defining which segments are active in the image.
        #    From how this function is called, z will always be an array containing another array as only element.
        
        maskingPatterns = z
        
        predictions = np.zeros((maskingPatterns.shape[0], 1))
        
        # show a progress bar when the algorithm starts to compute the samples
        # for all the other single runs of f(z): just do them silently
        if (maskingPatterns.shape[0] <= 1):
            hideProgressBar = True
        else:
            hideProgressBar = False
            print("Total samples to predict:", maskingPatterns.shape[0])
        
        mask_time = 0
        pred_time = 0
        for idx,maskingPattern in tqdm(enumerate(maskingPatterns), disable=hideProgressBar):
            start_time = time.time()
            masked_sequence = mask_sequence(maskingPattern, segments_slic, imageSequence, seq_avg)
            mask_time += (time.time() - start_time)
            
            start_time = time.time()
            frames_preds = model.predict(np.array(masked_sequence, ndmin=4))
            video_pred = np.mean(np.array(frames_preds) > 0.5)
            predictions[idx, 0] = video_pred
            pred_time += (time.time() - start_time)
        
        if (maskingPatterns.shape[0] > 1):
            print("--- Masking %s seconds ---" % (mask_time))
            print("--- Predicting %s seconds ---" % (pred_time))

        return predictions
    
    # use Kernel SHAP to explain the network's predictions
    background_data = np.zeros((1,N_SEGMENTS))   # https://shap.readthedocs.io/en/latest/#shap.KernelExplainer
    samples_features = np.ones(N_SEGMENTS)       # A matrix of samples (# samples x # features) on which to explain the model’s output.
    explainer = shap.KernelExplainer(f, background_data)
    shap_values = explainer.shap_values(samples_features, nsamples=runTimes, l1_reg="aic")

    # get the prediction from the model
    frames_preds = model.predict(np.array(imageSequence, ndmin=4))
    video_pred = np.mean(np.array(frames_preds) > 0.5)
    pred = np.array(video_pred, ndmin=2)
    
    figs = []
    for idx,img in enumerate(imageSequence):
        fig = getExplanationFigure(img, imageTrueClass, pred, shap_values, segments_slic[idx])
        figs.append(fig2array(fig)[:,:,:3])
        pl.close('all')
    
    return figs
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
imgList = getImageList(f"../input/df-face-seq/{SEQUENCE_NAME}/", FAKE_CLASS_VAL, 30)

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
start_time = time.time()

figs = explainSequence(model, imageSequence, '0 (fake)', 1000)

print("--- %s seconds ---" % (time.time() - start_time))
saveAndDisplayGIF(figs)