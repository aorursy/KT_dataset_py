!pip install efficientnet-pytorch



from tensorflow.keras.preprocessing import image

import requests

from skimage.segmentation import slic

import numpy as np

import shap

import os

from PIL import Image

import imageio

from tqdm import tqdm

import time



import torch

from torch.utils.model_zoo import load_url

import matplotlib.pyplot as plt

from scipy.special import expit



import sys

sys.path.append('../input/icpr2020')



from blazeface import FaceExtractor, BlazeFace, VideoReader

from architectures import fornet,weights

from isplutils import utils
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



import io

def fig2pil():

    buf = io.BytesIO()

    plt.savefig(buf, format='png')

    buf.seek(0)

    im = Image.open(buf)

    im.show()

    buf.close()

    return im



def fig2arrayRGB():

    return np.array(fig2pil())[:,:,:3]



from IPython.display import Image as IImg, display

def saveAndDisplayGIF(sequence, outputName="sequence.gif", FPS=5, displayOnNotebook=True):

    with imageio.get_writer(outputName, mode='I', fps=FPS) as writer:

        for frame in sequence:

            writer.append_data(frame)

    if (displayOnNotebook):

        display(IImg("../working/"+outputName))



def saveAndDisplayImages(images, outputPrefix="output", displayOnNotebook=True):

    for i,image in enumerate(images):

        outputPath = f"../working/{outputPrefix}_{i}.jpg"

        with imageio.get_writer(outputPath, mode='i') as writer:

            writer.append_data(image)

        if (displayOnNotebook):

            display(IImg(outputPath))

    

def saveAverageSequence(sequence, outputName="avg_sequence.png", displayOnNotebook=False):

    avg_array = np.mean(np.array(sequence), axis=0)

    # Round values in array and cast as 8-bit integer

    avg_array = np.array(np.round(avg_array), dtype=np.uint8)

    avg_img = Image.fromarray(avg_array, mode="RGB")

    avg_img.save(outputName)

    if (displayOnNotebook):

        display(avg_img)

    

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
from matplotlib.colors import LinearSegmentedColormap

# Fill every segment with the color relative to the shapley value

def fill_segmentation(values, segmentation):

    out = np.zeros(segmentation.shape)

    for i in range(len(values)):

        out[segmentation == i] = values[i]

    return out



def getExplanationFigure(img, imageTrueClass, prediction, shap_values, segments_slic, fakeClassValue):

    

    if (img.dtype != "uint8"):

        print("getExplanationFigure(): 'img' numpy array must be of type 'uint8'")

        return None

    

    # make a color map

    colors = []

    for l in np.linspace(1, 0, 100):

        colors.append((245/255, 39/255, 87/255, l))

    for l in np.linspace(0,1,100):

        colors.append((24/255, 196/255, 93/255, l))

    cm = LinearSegmentedColormap.from_list("shap", colors)

    

    # reverse shap values so that red is fake

    if (fakeClassValue==0):

        adjusted_shap_values = shap_values

    else:

        adjusted_shap_values = np.negative(shap_values)



    # set first image (original image)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6,4))

    axes[0].imshow(img)

    axes[0].set_title("class: {}, pred: {:.3f}".format(imageTrueClass, prediction))

    axes[0].axis('off')

    

    #max_val = np.max([np.max(np.abs(adjusted_shap_values[0][:-1])) for i in range(len(shap_values))])

    max_val = 0.3

    # set second image (segmented and colored image)

    m = fill_segmentation(adjusted_shap_values[0], segments_slic)

    axes[1].set_title("green: real - red: fake")

    grayImage = np.mean(img, axis=2).astype("uint8")

    grayImage = np.dstack((grayImage,grayImage,grayImage))

    axes[1].imshow(grayImage, alpha=0.15)

    im = axes[1].imshow(m, cmap=cm, vmin=-max_val, vmax=max_val)

    axes[1].axis('off')

    

    # horizontal bar

    cb = fig.colorbar(im, ax=axes.ravel().tolist(), label="SHAP value", orientation="horizontal", aspect=60)

    cb.outline.set_visible(False)



    return fig
def icrpGetFaceCroppedImages(images):

    faceList = []

    for image in images:

        faceImages = face_extractor.process_image(img=image)

        # take the face with the highest confidence score found by BlazeFace

        if (faceImages['faces']):

            faceList.append(faceImages['faces'][0])

    return np.array(faceList)



def icrpGetFaceCroppedVideo(video):

    INPUT_SIZE = 224

    faceList = face_extractor.process_video(video)

    faceList = [np.array(frame['faces'][0]) for frame in faceList if len(frame['faces'])]

    sequence = np.zeros((len(faceList), INPUT_SIZE, INPUT_SIZE, 3), dtype="uint8")

    for idx,face in enumerate(faceList):

        # resize the image

        sequence[idx,:,:,:] = np.array(Image.fromarray(face).resize((INPUT_SIZE,INPUT_SIZE)))

    return sequence



def icprPredictFaceImages(net, images):

    faces_t = torch.stack( [ transf(image=img)['image'] for img in images] )

    with torch.no_grad():

        faces_pred = torch.sigmoid(net(faces_t.to(device))).cpu().numpy().flatten()

    return faces_pred



def icprPredictImages(net, images):

    faceList = icrpGetFaceCroppedImages(images)

    faces_pred = icprPredictFaceImages(net, faceList)

    return faces_pred
"""

Choose an architecture between

- EfficientNetB4

- EfficientNetB4ST

- EfficientNetAutoAttB4

- EfficientNetAutoAttB4ST

- Xception

"""

net_model = 'EfficientNetAutoAttB4'



"""

Choose a training dataset between

- DFDC

- FFPP

"""

train_db = 'DFDC'



device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')



model_url = weights.weight_url['{:s}_{:s}'.format(net_model,train_db)]

net = getattr(fornet,net_model)().eval().to(device)

net.load_state_dict(load_url(model_url,map_location=device,check_hash=True))



# output values for the classifier

REAL_CLASS_VAL = 0

FAKE_CLASS_VAL = 1
face_policy = 'scale'

FACE_SIZE = 224

FRAMES_PER_VIDEO = 15



transf = utils.get_transformer(face_policy, FACE_SIZE, net.get_normalizer(), train=False)



facedet = BlazeFace().to(device)

facedet.load_weights("../input/icpr2020/blazeface/blazeface.pth")

facedet.load_anchors("../input/icpr2020/blazeface/anchors.npy")

videoreader = VideoReader(verbose=False)

video_read_fn = lambda x: videoreader.read_frames(x, num_frames=FRAMES_PER_VIDEO)

face_extractor = FaceExtractor(video_read_fn=video_read_fn,facedet=facedet)
# a function that depends on a binary mask representing if an image region is hidden

def mask_sequence(mask_pattern, segmentation, sequence, background=None):

    # mask_pattern: an array having length 'nSegments'

    if background is None:

        background = sequence.mean((0,1,2))

    

    out = np.zeros((sequence.shape[0], sequence.shape[1], sequence.shape[2], sequence.shape[3]))

    out[:,:,:,:] = sequence

    

    for j,segm_state in enumerate(mask_pattern):

        if (segm_state == 0):

            out[segmentation==j, :] = background

    return out
def f_icpr(maskingPatterns, media : np.array, segments_slic) -> np.array:



    hideProgressBar = (maskingPatterns.shape[0] <= 1)



    predictions = []



    # if it's an image

    if (len(media.shape)==3):

        avg = media.mean((0,1))

        for maskingPattern in maskingPatterns:

            masked_image = mask_image(maskingPattern, segments_slic, media, avg)

            preds = icprPredictFaceImages(net, [media])

            predictions.append(preds)

    

    # if it's a sequence

    elif (len(media.shape)==4):

        avg = media.mean((0,1,2))

        for idx, maskingPattern in tqdm(enumerate(maskingPatterns), disable=hideProgressBar):

            masked_sequence = mask_sequence(maskingPattern, segments_slic, media, avg)

            frames_preds = icprPredictFaceImages(net, masked_sequence)

            video_pred = np.mean(np.array(frames_preds) > 0.5)

            predictions.append(np.array(video_pred, ndmin=1))



    return np.array(predictions, ndmin=2)





def explain(model, media, runTimes, nSegments=50, segCompactness=10) -> tuple:



    # segment the image so we don't have to explain every pixel



    # if it's an image

    if (len(media.shape)==3):

        spatial_smooth = 1       # variance of gaussian smooth across adiacent pixels

        segments_slic = slic(media, n_segments=nSegments, compactness=segCompactness, sigma=1)



    # if it's a sequence

    elif (len(media.shape)==4):

        # sigma parameters is the size of the gaussian filter that pre-smooth the data

        # I defined the sigma as a triplet where the dimensions represent (time, image_x, image_y)

        temporal_smooth = 0.5    # variance of gaussian smooth across conseutive frame

        spatial_smooth = 1       # variance of gaussian smooth across adiacent pixels

        s = (temporal_smooth, spatial_smooth, spatial_smooth)

        segments_slic = slic(media, n_segments=nSegments, compactness=segCompactness, sigma=s)

    

    else:

        print("Media shape not recognized:", media.shape)

        return

    

    # prediction function to pass to SHAP

    def f(z):

        # z: feature vectors from the point of view of shap

        #    from our point of view they are binary vectors defining which segments are active in the image.

        return f_icpr(z, media, segments_slic)



    # use Kernel SHAP to explain the network's predictions

    background_data = np.zeros((1, nSegments))   # https://shap.readthedocs.io/en/latest/#shap.KernelExplainer

    samples_features = np.ones(nSegments)        # A vector of features on which to explain the model’s output.

    explainer = shap.KernelExplainer(f, background_data)

    shap_values = explainer.shap_values(samples_features, nsamples=runTimes, l1_reg="aic")

    

    return shap_values, segments_slic
# Configuration

SHAP_SAMPLES = 500

N_SEGMENTS = 200

VIDEOS_TO_ANALYZE = 20
# load videos



import json



directory = "../input/deepfake-detection-challenge/train_sample_videos/"

with open(os.path.join(directory, "metadata.json")) as f:

    metadata = json.load(f)



vidList = []

for vidname in sorted(os.listdir(directory))[:VIDEOS_TO_ANALYZE]:

    sequence_value = FAKE_CLASS_VAL if metadata[vidname]["label"]=="FAKE" else REAL_CLASS_VAL

    vidList.append((os.path.join(directory, vidname), sequence_value))

    

print("Total videos:", len(vidList))
for i, (vid, class_value) in enumerate(vidList):

    

    print(f"--- Analizing video: {i+1}/{len(vidList)} ---")

    

    imgClassDesc = str(class_value) + (' (fake)' if class_value==FAKE_CLASS_VAL else ' (real)')

    

    start_time = time.time()

    faceSequence = icrpGetFaceCroppedVideo(vid)

    print("Face cropping: %s seconds" % (time.time() - start_time))

    

    start_time = time.time()

    with suppress_stderr():

        shap_values, segments_slic = explain(net, faceSequence, SHAP_SAMPLES, nSegments=N_SEGMENTS)

    print("Shap time for %d samples: %s seconds" % (SHAP_SAMPLES, time.time() - start_time))

    

    start_time = time.time()

    preds = icprPredictFaceImages(net, faceSequence)

    fig_sequence = []

    for idx,img in enumerate(faceSequence):

        fig = getExplanationFigure(img, imgClassDesc, np.mean(preds), shap_values, segments_slic[idx], FAKE_CLASS_VAL)

        figarray = fig2arrayRGB()

        fig_sequence.append(figarray)

        plt.close('all')

    # Convert the list of frames into a 4D numpy array (frame, width, height, color)

    fig_sequence = np.asarray(fig_sequence)

    print("Create animation for %d frames: %s seconds" % (fig_sequence.shape[0], time.time() - start_time))

    

    start_time = time.time()

    saveAndDisplayGIF(fig_sequence, outputName=f"dfdc_{i}_shap2D.gif")

    print("Save and show animation: %s seconds" % (time.time() - start_time))

    

    start_time = time.time()

    saveAverageSequence(fig_sequence, outputName=f"dfdc_{i}_shap2D_avg.png")

    print("Show average image: %s seconds" % (time.time() - start_time))

print(" ")