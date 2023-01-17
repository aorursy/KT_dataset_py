# This kernel imports code from codebase.py, a custom script I made.

# It is available at: https://www.kaggle.com/samuelepino/codebase



import os

import sys

sys.path.append("../usr/lib/codebase/")

from codebase import ICPR, Segmenter, Explainer, Pipeline, VideoLoader, FigureManager
# Configuration

N_VIDEOS = 1

FRAMES_PER_VIDEO = 30

SHAP_SAMPLES = 16000

N_SEGMENTS = 200



# Initialization

classifier = ICPR(frames_per_video=FRAMES_PER_VIDEO, consecutive_frames=True)

seg = Segmenter(mode="color", segmentsNumber=N_SEGMENTS)

expl = Explainer(classifier=classifier, trackTime=False)
# load videos



dfdc_vidlist = VideoLoader.loadFilenamesDFDC(

    videoCount=N_VIDEOS, 

    fakeClassValue=classifier.FAKE_CLASS_VAL,

    realClassValue=classifier.REAL_CLASS_VAL)
dfdc_vidlist
# show first video



vidName = dfdc_vidlist[0][0]

vidClass = dfdc_vidlist[0][1]

vidPath = os.path.join(VideoLoader.DFDC_trainVideoDir, vidName)

imageSequence = classifier.getFaceCroppedVideo(vidPath)



FigureManager.saveAndDisplayGIF(

                imageSequence, vidName+".gif",

                fps=15, displayOnNotebook=True)
# get frame predictions



import numpy as np



framesPred = np.zeros(imageSequence.shape[0])

framesPred = expl.normalizePredictions(classifier.predictFaceImages(imageSequence))

videoPred = np.mean(framesPred)
print("Video's true class is", vidClass)

print("Video has been predicted as", videoPred)

print(" ")

print("Fake frames have value close to:", classifier.FAKE_CLASS_VAL)

print("Real frames have value close to:", classifier.REAL_CLASS_VAL)

print(" ")



highlighted_frames = 3



top_fake_frames = np.argsort(framesPred)[:highlighted_frames]

top_real_frames = np.argsort(-framesPred)[:highlighted_frames]
# show the 'fakest' frames



from matplotlib import pyplot as plt



print("The 'fakest' frames are:")



fig, ax = plt.subplots(1, highlighted_frames, figsize=(4*highlighted_frames, 4))

fig.set_facecolor('white')



for i, frame_id in enumerate(top_fake_frames):

    

    ax[i].imshow(imageSequence[frame_id])

    ax[i].set_title(f"Frame {frame_id}, value {framesPred[frame_id]:.4f}")

    
# show the 'realest' frames



print("The 'realest' frames are:")



fig, ax = plt.subplots(1, highlighted_frames, figsize=(4*highlighted_frames, 4))

fig.set_facecolor('white')



for i, frame_id in enumerate(top_real_frames):

    ax[i].imshow(imageSequence[frame_id])

    ax[i].set_title(f"Frame {frame_id}, value {framesPred[frame_id]:.4f}")

    
# explanation for the 'fakest' frame



p = Pipeline(classifier, seg, expl, segmentationDim="2D", explanationMode="frame",

             nSegments=N_SEGMENTS, shapSamples=SHAP_SAMPLES)



p.start(imageSequence[top_fake_frames[0]:top_fake_frames[0]+1], vidClass, vidName+"-fake-frames")
# explanation for the 'realest' frame



p = Pipeline(classifier, seg, expl, segmentationDim="2D", explanationMode="frame",

             nSegments=N_SEGMENTS, shapSamples=SHAP_SAMPLES)



p.start(imageSequence[top_real_frames[0]:top_real_frames[0]+1], vidClass, vidName[:-4]+"-real-frames")
# Configuration

N_VIDEOS = 10



# Initialization

classifier = ICPR(frames_per_video=FRAMES_PER_VIDEO, consecutive_frames=True)

seg = Segmenter(mode="color", segmentsNumber=N_SEGMENTS)

expl = Explainer(classifier=classifier, trackTime=False)



# load videos



dfdc_vidlist = VideoLoader.loadFilenamesDFDC(

    videoCount=N_VIDEOS, 

    fakeClassValue=classifier.FAKE_CLASS_VAL,

    realClassValue=classifier.REAL_CLASS_VAL)[1:]



for (vidName, vidClass) in dfdc_vidlist:

    

    print(f"\nAnalyzing video {vidName} (class {vidClass})")

    

    vidPath = os.path.join(VideoLoader.DFDC_trainVideoDir, vidName)

    imageSequence = classifier.getFaceCroppedVideo(vidPath)

    

    # get frame predictions

    framesPred = expl.normalizePredictions(classifier.predictFaceImages(imageSequence))

    videoPred = np.mean(framesPred)

    

    print("Video's true class is", vidClass)

    print("Video has been predicted as", videoPred)

    print(" ")

    print("Fake frames have value close to:", classifier.FAKE_CLASS_VAL)

    print("Real frames have value close to:", classifier.REAL_CLASS_VAL)

    print(" ")



    highlighted_frames = 3



    top_fake_frames = np.argsort(framesPred)[:highlighted_frames]

    top_real_frames = np.argsort(-framesPred)[:highlighted_frames]

    

    # show the 'fakest' frames

    print("The 'fakest' frames are:")

    fig, ax = plt.subplots(1, highlighted_frames, figsize=(4*highlighted_frames, 4))

    fig.set_facecolor('white')

    for i, frame_id in enumerate(top_fake_frames):

        ax[i].imshow(imageSequence[frame_id])

        ax[i].set_title(f"Frame {frame_id}, value {framesPred[frame_id]:.4f}")

    plt.show()

    

    # show the 'realest' frames

    print("The 'realest' frames are:")

    fig, ax = plt.subplots(1, highlighted_frames, figsize=(4*highlighted_frames, 4))

    fig.set_facecolor('white')

    for i, frame_id in enumerate(top_real_frames):

        ax[i].imshow(imageSequence[frame_id])

        ax[i].set_title(f"Frame {frame_id}, value {framesPred[frame_id]:.4f}")

    plt.show()

    

    # explanation for the 'fakest' frame

    p = Pipeline(classifier, seg, expl, segmentationDim="2D", explanationMode="frame",

                 nSegments=N_SEGMENTS, shapSamples=SHAP_SAMPLES)

    p.start(imageSequence[top_fake_frames[0]:top_fake_frames[0]+1], vidClass, vidName[:-4]+"-fake-frames")

        

    # explanation for the 'realest' frame

    p = Pipeline(classifier, seg, expl, segmentationDim="2D", explanationMode="frame",

                 nSegments=N_SEGMENTS, shapSamples=SHAP_SAMPLES)

    p.start(imageSequence[top_real_frames[0]:top_real_frames[0]+1], vidClass, vidName[:-4]+"-real-frames")