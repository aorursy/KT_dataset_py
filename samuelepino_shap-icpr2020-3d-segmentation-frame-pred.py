import os

import numpy as np

from skimage.segmentation import slic

from tqdm import tqdm
from time import time



class Timer:

    def __init__(self, title):

        self.title = title

    def __enter__(self):

        self.start_time = time()

    def __exit__(self, exc_type, exc_val, exc_tb):

        print("{} time: {:.3f} s".format(self.title, time() - self.start_time))
import json



class VideoLoader:

    

    DFDC_metadataPath = "../input/deepfake-detection-challenge/train_sample_videos/metadata.json"

    DFDC_trainVideoDir = "../input/deepfake-detection-challenge/train_sample_videos/"

    DFDC_testVideoDir = "../input/deepfake-detection-challenge/test_videos/"

    MDFD_testVideoDir = "../input/mesonet-dataset-sfw/validation/"

    

    def npSeqFromDir(directory : str, targetSize:tuple=None, normalize=True, frameLimit=np.infty) -> np.ndarray:

    

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

    

    def loadFilenamesDFDC(videoCount=10, fakeClassValue=1, realClassValue=0):

        # Load metadata file containing labels for videos ("REAL" or "FAKE")

        with open(VideoLoader.DFDC_metadataPath) as f:

            metadata = json.load(f)

            

        videosInDirectory = [ vid for vid in sorted(os.listdir(VideoLoader.DFDC_trainVideoDir)) if vid[-4:].lower() == ".mp4" ]

        vidList = []

        for vidname in videosInDirectory[ : min(videoCount, len(videosInDirectory))]:

            sequence_value = fakeClassValue if metadata[vidname]["label"]=="FAKE" else realClassValue

            vidList.append((vidname, sequence_value))

            

        print("Total video names extracted from DFDC:", len(vidList))

        return vidList

    

    def loadDirnamesMDFD(label="df", videoCount=10, fakeClassValue=1, realClassValue=0):

        # label: "real" or "df"

        videosInDirectory = sorted(os.listdir(os.path.join(VideoLoader.MDFD_testVideoDir, label)))

        vidList = []

        for vidname in videosInDirectory[ : min(videoCount, len(videosInDirectory))]:

            sequence_value = fakeClassValue if label=="df" else realClassValue

            vidList.append((vidname, sequence_value))

        print("Total video names extracted from MDFD:", len(vidList))

        return vidList
class Segmenter:

    

    def __init__(self, mode="color", segmentsNumber=100, segCompactness=20):

        # mode : "color", "grid"

        self.mode = mode

        self.segmentsNumber = segmentsNumber

        self.segCompactness = segCompactness

        

    def segment(self, media : np.ndarray) -> np.ndarray:

        

        if (self.mode == "color"):

            # if it's an image

            if (len(media.shape)==3):

                return slic(media, n_segments=self.segmentsNumber, compactness=self.segCompactness, sigma=3)

            # if it's a sequence

            elif (len(media.shape)==4):

                # sigma parameters is the size of the gaussian filter that pre-smooth the data

                # I defined the sigma as a triplet where the dimensions represent (time, image_x, image_y)

                return slic(media, n_segments=self.segmentsNumber, compactness=self.segCompactness, sigma=(0.5,3,3))

            else:

                raise Exception(f"Media shape not recognized: {media.shape}")

        

        elif (self.mode == "grid"):

            pass

        
import shap



# if GPU is available, import the numpy accelerated module

from torch import cuda

GPU = cuda.is_available()

if (GPU):

    try:

        import cupy as cp

    except ImportError:

        ! pip install cupy-cuda101

        import cupy as cp

    print("Using GPU acceleration for Numpy")





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



class Explainer:

    

    def __init__(self, classifier, trackTime=False):

        # modelName : "mesonet" or "icpr"

        self.classifier = classifier

        self.model = classifier.getModel()

        self.classifierName = classifier.NAME

        self.trackTime = trackTime

        

    def normalizePredictions(self, p):

        # the predictions are normalized so that FAKE = -1 and REAL = +1

        a = self.classifier.FAKE_CLASS_VAL

        b = self.classifier.REAL_CLASS_VAL

        return 2 * (p - (a+b)/2) / (b-a)

    

    def explain(self, media : np.ndarray, segmentation : np.ndarray, nSegments, shapSamples : int) -> tuple:

        

        # in order to use the GPU and increase performance, numpy arrays are converted at the beginning to

        # cupy arrays

        if (GPU):

            cp_media = cp.asarray(media)

            cp_segmentation = cp.asarray(segmentation)

        else:

            cp_media = media

            cp_segmentation = segmentation

        

        # prediction function to pass to SHAP

        def f(z):

            # z: feature vectors from the point of view of shap

            #    from our point of view they are binary vectors defining which segments are active in the image.

            p = self.predictSamples(z, cp_media, cp_segmentation)

            return self.normalizePredictions(p)

        

        # use Kernel SHAP to explain the network's predictions

        background_data = np.zeros((1, nSegments))   # https://shap.readthedocs.io/en/latest/#shap.KernelExplainer

        samples_features = np.ones(nSegments)        # A vector of features on which to explain the model’s output.

        explainer = shap.KernelExplainer(f, background_data)

        shap_values = explainer.shap_values(samples_features, nsamples=shapSamples, l1_reg="aic")

        

        return shap_values[0], explainer.expected_value[0]



    def predictSamples(self, maskingPatterns, media, segments_slic):

        hideProgressBar = (maskingPatterns.shape[0] <= 1 or self.trackTime)



        predictions = []



        mask_time = 0

        pred_time = 0



        # if it's an image

        if (len(media.shape)==3):

            avg = media.mean((0,1))

            

            # create batches of masking patterns (for performance reasons)

            batchSize = 50

            batches = []

            i = 0

            while (i < maskingPatterns.shape[0]):

                j = i+batchSize if (i+batchSize < maskingPatterns.shape[0]) else maskingPatterns.shape[0]

                batches.append(maskingPatterns[i:j, :])

                i += batchSize

            

            for batch in batches:

                

                # create masked images for this batch

                start_mask_time = time()

                masked_images_batch = []

                for maskingPattern in batch:

                    masked_images_batch.append(Explainer.mask_image(maskingPattern, segments_slic, media, avg))

                mask_time += (time() - start_mask_time)

                

                # predict masked images for this batch

                start_pred_time = time()

                if (self.classifierName == "mesonet"):

                    preds = self.model.predict(np.array(masked_image, ndmin=4))[0]

                elif (self.classifierName == "icpr"):

                    preds = self.classifier.predictFaceImages(masked_images_batch)

                pred_time += (time() - start_pred_time)

                

                # concatenate this predictions with previous batch predictions

                predictions += list(preds)

                

            if (self.trackTime and maskingPatterns.shape[0]>1):

                print("--- Masking:      %s seconds ---" % (mask_time))

                print("--- Predicting:   %s seconds ---" % (pred_time))



        # if it's a sequence

        elif (len(media.shape)==4):

            avg = media.mean((0,1,2))

            for idx, maskingPattern in tqdm(enumerate(maskingPatterns), disable=hideProgressBar):



                start_mask_time = time()

                masked_sequence = Explainer.mask_sequence(maskingPattern, segments_slic, media, avg)

                mask_time += (time() - start_mask_time)



                start_pred_time = time()

                if (self.classifierName == "mesonet"):

                    frames_preds = model.predict(np.array(masked_sequence, ndmin=4))

                elif (self.classifierName == "icpr"):

                    frames_preds = self.classifier.predictFaceImages(masked_sequence)

                video_pred = np.mean(frames_preds)

                pred_time += (time() - start_pred_time)



                predictions.append(video_pred)



            if (self.trackTime and maskingPatterns.shape[0]>1):

                print("--- Masking:      %s seconds ---" % (mask_time))

                print("--- Predicting:   %s seconds ---" % (pred_time))

        

        return np.array(predictions, ndmin=2)

        # Predictions should be a numpy array like

        # [[0.6044281 ] [0.6797433 ] [0.5042769 ] ... ]

        # or

        # [[0.49638838 0.99638945 0.42781693 ... ]]

    

    # a function that depends on a binary mask representing if an image region is hidden

    def mask_image(mask_pattern, segmentation, image, background=None) -> np.ndarray:

        # mask_pattern : an array having length 'nSegments'

        # segmentation : a 3D cupy or numpy array containing the segmentId of every pixel

        # image        : a 3D cupy or numpy array containing the image

        if background is None:

            background = image.mean((0,1))

        

        if (GPU):

            out = cp.zeros((image.shape[0], image.shape[1], image.shape[2]))

        else:

            out = np.zeros((image.shape[0], image.shape[1], image.shape[2]))

        out[:,:,:] = image



        for j,segm_state in enumerate(mask_pattern):

            if (segm_state == 0):

                out[segmentation==j, :] = background

        if (GPU):

            return cp.asnumpy(out)

        else:

            return out



    # a function that depends on a binary mask representing if an image region is hidden

    def mask_sequence(mask_pattern, segmentation, sequence, background=None) -> np.ndarray:

        # mask_pattern : an array having length 'nSegments'

        # segmentation : a 4D cupy or numpy array containing the segmentId of every pixel

        # image        : a 4D cupy or numpy array containing the image sequence

        if background is None:

            background = sequence.mean((0,1,2))

        

        if (GPU):

            out = cp.zeros((sequence.shape[0], sequence.shape[1], sequence.shape[2], sequence.shape[3]))

        else:

            out = np.zeros((sequence.shape[0], sequence.shape[1], sequence.shape[2], sequence.shape[3]))

        out[:,:,:,:] = sequence



        for j,segm_state in enumerate(mask_pattern):

            if (segm_state == 0):

                out[segmentation==j, :] = background

        if (GPU):

            return cp.asnumpy(out)

        else:

            return out

    

    

    def getExplanationFigure(self, img, imageTrueClass, prediction, shap_values, segments_slic):

        # shap values : a numpy array of length N_SEGMENTS

        

        from matplotlib.colors import LinearSegmentedColormap

        # Fill every segment with the color relative to the shapley value

        def fill_segmentation(values, segmentation):

            out = np.zeros(segmentation.shape)

            for i in range(len(values)):

                out[segmentation == i] = values[i]

            return out



        def subplotWithTitle(ax, img, title="", alpha=1):

            ax.imshow(img)

            ax.set_title(title)

            ax.axis('off')



        if (img.dtype != "uint8"):

            print("getExplanationFigure(): 'img' numpy array must be of type 'uint8'")

            return None



        # make a color map

        colors = []

        fc = (.96, .15, .34) # fake color (RGB): red

        rc = (.09, .77, .36) # real color (RGB): green

        for alpha in np.linspace(1, 0, 100):

            colors.append((fc[0], fc[1], fc[2], alpha))

        for alpha in np.linspace(0, 1, 100):

            colors.append((rc[0], rc[1], rc[2], alpha))

        cm = LinearSegmentedColormap.from_list("shap", colors)



        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6,4))

        

        # set up first image (original image)

        subplotWithTitle(axes[0], img, title = "class: {}, pred: {:.3f}".format(

            self.normalizePredictions(imageTrueClass), self.normalizePredictions(prediction)))

        

        # set up second image (gray image)

        grayImage = np.mean(img, axis=2).astype("uint8")

        grayImage = np.dstack((grayImage,grayImage,grayImage))

        subplotWithTitle(axes[1], grayImage, alpha=0.15, title="green: real - red: fake")

        

        # set up segments color overlay

        m = fill_segmentation(shap_values, segments_slic)

        max_val = np.max( [ np.max(np.abs(shap_values[:-1])) for i in range(len(shap_values)) ] )

        

        im = axes[1].imshow(m, cmap=cm, vmin=-max_val, vmax=max_val)

        

        # horizontal bar

        cb = fig.colorbar(im, ax=axes.ravel().tolist(), label="SHAP value", orientation="horizontal", aspect=60)

        cb.outline.set_visible(False)



        return fig
import io

import imageio

from PIL import Image

from IPython.display import display, Image as InteractiveImage

import matplotlib.pyplot as plt



class FigureManager:

    

    def fig2pil():

        buf = io.BytesIO()

        plt.savefig(buf, format='png')

        buf.seek(0)

        im = Image.open(buf)

        im.show()

        buf.close()

        return im



    def fig2arrayRGB():

        return np.array(FigureManager.fig2pil())[:,:,:3]



    def saveAndDisplayGIF(sequence, outputName="sequence.gif", FPS=5, displayOnNotebook=True):

        outputPath = f"../working/{outputName}"

        with imageio.get_writer(outputName, mode='I', fps=FPS) as writer:

            for frame in sequence:

                writer.append_data(frame)

        if (displayOnNotebook):

            display(InteractiveImage(outputPath))



    def saveAndDisplayImages(images, outputPrefix="output", displayOnNotebook=True):

        for i,image in enumerate(images):

            outputPath = f"../working/{outputPrefix}_{i}.jpg"

            with imageio.get_writer(outputPath, mode='i') as writer:

                writer.append_data(image)

            if (displayOnNotebook):

                display(Image.open(outputPath))



    def saveAverageSequence(sequence, outputName="avg_sequence.png"):

        avg_array = np.mean(np.array(sequence), axis=0)

        # Round values in array and cast as 8-bit integer

        avg_array = np.array(np.round(avg_array), dtype=np.uint8)

        avg_img = Image.fromarray(avg_array, mode="RGB")

        avg_img.save(outputName)
import torch

from torch.utils.model_zoo import load_url

from scipy.special import expit



! pip install efficientnet-pytorch



import sys

sys.path.append('../input/icpr2020')

from blazeface import FaceExtractor, BlazeFace, VideoReader

from architectures import fornet,weights

from isplutils import utils



class ICPR:

    

    NAME = "icpr"

    INPUT_SIZE = 224

    # output values for the classifier

    REAL_CLASS_VAL = 0

    FAKE_CLASS_VAL = 1

    

    def __init__(self, frames_per_video=10):

        # Choose an architecture between:

        # "EfficientNetB4", "EfficientNetB4ST", "EfficientNetAutoAttB4", "EfficientNetAutoAttB4ST", "Xception"

        net_model = "EfficientNetAutoAttB4"



        # Choose a training dataset between:

        # DFDC, FFPP

        train_db = "DFDC"



        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        face_policy = 'scale'



        model_url = weights.weight_url['{:s}_{:s}'.format(net_model,train_db)]

        self.net = getattr(fornet,net_model)().eval().to(self.device)

        self.net.load_state_dict(load_url(model_url, map_location=self.device, check_hash=True))

        

        self.transf = utils.get_transformer(face_policy, self.INPUT_SIZE, self.net.get_normalizer(), train=False)



        facedet = BlazeFace().to(self.device)

        facedet.load_weights("../input/icpr2020/blazeface/blazeface.pth")

        facedet.load_anchors("../input/icpr2020/blazeface/anchors.npy")

        videoreader = VideoReader(verbose=False)

        video_read_fn = lambda x: videoreader.read_frames(x, num_frames=frames_per_video)

        self.face_extractor = FaceExtractor(video_read_fn=video_read_fn, facedet=facedet)

    

    def getModel(self):

        return self.net

    

    def getFaceCroppedImages(self, images):

        faceList = []

        for image in images:

            faceImages = self.face_extractor.process_image(img=image)

            # take the face with the highest confidence score found by BlazeFace

            if (faceImages['faces']):

                faceList.append(faceImages['faces'][0])

        return np.array(faceList)



    def getFaceCroppedVideo(self, videoPath):

        faceList = self.face_extractor.process_video(videoPath)

        faceList = [np.array(frame['faces'][0]) for frame in faceList if len(frame['faces'])]

        sequence = np.zeros((len(faceList), self.INPUT_SIZE, self.INPUT_SIZE, 3), dtype="uint8")

        for idx,face in enumerate(faceList):

            # resize the image

            sequence[idx,:,:,:] = np.array(Image.fromarray(face).resize((self.INPUT_SIZE, self.INPUT_SIZE)))

        return sequence



    def predictFaceImages(self, images):

        faces_t = torch.stack( [ self.transf(image=img)['image'] for img in images] )

        with torch.no_grad():

            raw_preds = self.net(faces_t.to(self.device))

            faces_pred = torch.sigmoid(raw_preds).cpu().numpy().flatten()

        return faces_pred



    def predictImages(self, images):

        faceList = self.getFaceCroppedImages(images)

        faces_pred = self.predictFaceImages(faceList)

        return faces_pred
class Pipeline:

    def __init__(self, classifier, seg : Segmenter, expl : Explainer, segmentationDim : str, explanationMode : str,

                nSegments : int, shapSamples : int):

        # segmentationDim : a string among ["2D", "3D"]

        # explanationMode : a string among ["frame", "video"]

        self.classifier = classifier

        self.seg = seg

        self.expl = expl

        self.segmentationDim = segmentationDim

        self.explanationMode = explanationMode

        self.nSegments = nSegments

        self.shapSamples = shapSamples

        

    def start(self, imageSequence : np.ndarray):

        if (not isinstance(imageSequence, np.ndarray) or len(imageSequence.shape)!=4):

            raise Exception("imageSequence must be a 4-dimensional Numpy array.")

        

        # SEGMENT IMAGE OR VIDEO

        

        with Timer("Segmenting"):

            segmentation = self._segment(imageSequence)

        

        # COMPUTE SHAP VALUES

        

        with Timer("Computing SHAP values"):

            shap_values = self._explain(imageSequence, segmentation)

        

        # COMPUTE CLASSIFIER PREDICTIONS FOR FRAMES OR VIDEO

        

        with Timer("Classifier original predictions"):

            if (self.explanationMode == "video"):

                videoPred = self._predictSequence(imageSequence)

                #print("Normalized pred.: {:.3f} | Sum of shap values: {:.3f}".format(

                #    expl.normalizePredictions(videoPred), np.sum(shap_values)))

                framesPred = np.ones(imageSequence.shape[0]) * videoPred

                framesShapValues = np.tile(shap_values, (imageSequence.shape[0],1) )

            elif (self.explanationMode == "frame"):

                framesPred = np.zeros(imageSequence.shape[0])

                for i in range(imageSequence.shape[0]):

                    framesPred[i] = self._predictSequence([imageSequence[i]])

                framesShapValues = shap_values

        

        # SHOW FIGURES

        

        with Timer("Showing and saving figures"):

            fig_sequence = []

            for i in range(imageSequence.shape[0]):

                fig = self.expl.getExplanationFigure(imageSequence[i], vidClass, framesPred[i], framesShapValues[i], segmentation[i])

                fig_sequence.append(FigureManager.fig2arrayRGB())

                plt.close('all')



            # Convert the list of frames into a 4D numpy array (frame, width, height, color)

            fig_sequence = np.array(fig_sequence)



            FigureManager.saveAndDisplayGIF(fig_sequence, outputName=f"{vidName}_shap2D.gif")

            FigureManager.saveAverageSequence(fig_sequence, outputName=f"{vidName}_shap2D_avg.png")

        

    def _segment(self, imageSequence):

        if (self.segmentationDim == "3D"):

            segmentation = self.seg.segment(imageSequence)

        elif (self.segmentationDim == "2D"):

            segmentation = np.zeros((imageSequence.shape[0], imageSequence.shape[1], imageSequence.shape[2]))

            for i,frame in enumerate(imageSequence):

                segmentation[i,:,:] = self.seg.segment(frame)

        return segmentation

    

    def _explain(self, imageSequence, segmentation):

        if (self.explanationMode == "video"):

            with suppress_stderr():

                shap_values, expected_value = self.expl.explain(imageSequence, segmentation, self.nSegments, self.shapSamples)

            #print("Prediction on all-masked image | expected value: {:.3f}".format(expected_value))

            

        elif (self.explanationMode == "frame"):

            shap_values = np.zeros((imageSequence.shape[0], N_SEGMENTS))

            for i in range(imageSequence.shape[0]):

                print(f"\rAnalizing frame: {i+1}/{imageSequence.shape[0]}",

                      end="" if (i+1)<imageSequence.shape[0] else "\n")

                with suppress_stderr():

                    shap_values[i], _ = self.expl.explain(imageSequence[i], segmentation[i], N_SEGMENTS, SHAP_SAMPLES)

        

        return shap_values

    

    def _predictSequence(self, imageSequence):

        if (self.classifier.NAME == "icpr"):

            framePreds = self.classifier.predictFaceImages(imageSequence)

            videoPred = np.mean(framePreds)

        elif (self.classifier.NAME == "mesonet"):

            raise Exception("mesonet prediction not implemented yet.")

        return videoPred
# Configuration

N_VIDEOS = 10

FRAMES_PER_VIDEO = 20

SHAP_SAMPLES = 1000

N_SEGMENTS = 200



# Initialization

classif = ICPR(frames_per_video=FRAMES_PER_VIDEO)

seg = Segmenter(mode="color", segmentsNumber=N_SEGMENTS)

expl = Explainer(classifier=classif, trackTime=False)
p = Pipeline(classif, seg, expl, segmentationDim="3D", explanationMode="frame", nSegments=N_SEGMENTS, shapSamples=SHAP_SAMPLES)



dfdc_vidlist = VideoLoader.loadFilenamesDFDC(videoCount=N_VIDEOS, 

                                   fakeClassValue=classif.FAKE_CLASS_VAL, realClassValue=classif.REAL_CLASS_VAL)



for (vidName, vidClass) in dfdc_vidlist:

    

    print("Analyzing sequence", vidName)

    

    vidPath = os.path.join(VideoLoader.DFDC_trainVideoDir, vidName)

    # Get image sequence from folder

    imageSequence = classif.getFaceCroppedVideo(vidPath)

    

    p.start(imageSequence)