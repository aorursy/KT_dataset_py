# This kernel imports code from codebase.py, a custom script I made.

# It is available at: https://www.kaggle.com/samuelepino/codebase



import os

import sys

import json

sys.path.append("../usr/lib/codebase/")

from codebase import ICPR, Segmenter, Explainer, Pipeline, VideoLoader
# Configuration

N_VIDEOS = 5

FRAMES_PER_VIDEO = 1

SHAP_SAMPLES = [250, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000]

N_SEGMENTS = 200



# Initialization

classif = ICPR(frames_per_video=FRAMES_PER_VIDEO)

seg = Segmenter(mode="color", segmentsNumber=N_SEGMENTS)

expl = Explainer(classifier=classif, trackTime=False)
dfdc_vidlist = VideoLoader.loadFilenamesDFDC(videoCount=N_VIDEOS, 

                                   fakeClassValue=classif.FAKE_CLASS_VAL, realClassValue=classif.REAL_CLASS_VAL)



for shapSamples in SHAP_SAMPLES:

    

    print(f"\n=== SHAP SAMPLES : {shapSamples} ===\n")

    p = Pipeline(classif, seg, expl, segmentationDim="3D", explanationMode="frame", nSegments=N_SEGMENTS, shapSamples=shapSamples,

                displayIFigures=False)

    

    for (vidName, vidClass) in dfdc_vidlist:



        print("Analyzing sequence", vidName)



        vidPath = os.path.join(VideoLoader.DFDC_trainVideoDir, vidName)

        # Get image sequence from folder

        imageSequence = classif.getFaceCroppedVideo(vidPath)



        p.start(imageSequence, vidClass, vidName)



    with open(f"shap_values_sf_{shapSamples}.json", 'w') as f:

        f.write(json.dumps(p.getShapValuesCollection(), indent=2))
# Configuration

N_VIDEOS = 1

FRAMES_PER_VIDEO = 10

SHAP_SAMPLES = [250, 500, 1000, 2000, 4000, 8000, 16000]

N_SEGMENTS = 200



# Initialization

classif = ICPR(frames_per_video=FRAMES_PER_VIDEO)

seg = Segmenter(mode="color", segmentsNumber=N_SEGMENTS)

expl = Explainer(classifier=classif, trackTime=False)
dfdc_vidlist = VideoLoader.loadFilenamesDFDC(videoCount=N_VIDEOS, 

                                   fakeClassValue=classif.FAKE_CLASS_VAL, realClassValue=classif.REAL_CLASS_VAL)



for shapSamples in SHAP_SAMPLES:

    

    print(f"\n=== SHAP SAMPLES : {shapSamples} ===\n")

    p = Pipeline(classif, seg, expl, segmentationDim="3D", explanationMode="frame", nSegments=N_SEGMENTS, shapSamples=shapSamples,

                displayIFigures=False)

    

    for (vidName, vidClass) in dfdc_vidlist:



        print("Analyzing sequence", vidName)



        vidPath = os.path.join(VideoLoader.DFDC_trainVideoDir, vidName)

        # Get image sequence from folder

        imageSequence = classif.getFaceCroppedVideo(vidPath)



        p.start(imageSequence, vidClass, vidName)

        

    with open(f"shap_values_mf_{shapSamples}.json", 'w') as f:

        f.write(json.dumps(p.getShapValuesCollection(), indent=2))