# This kernel imports code from codebase.py, a custom script I made.

# It is available at: https://www.kaggle.com/samuelepino/codebase



import os

import sys

sys.path.append("../usr/lib/codebase/")

from codebase import ICPR, Segmenter, Explainer, Pipeline, VideoLoader
# Configuration

N_VIDEOS = 10

FRAMES_PER_VIDEO = 10

SHAP_SAMPLES = 2000

N_SEGMENTS = 100



# Initialization

classif = ICPR(frames_per_video=FRAMES_PER_VIDEO)

seg = Segmenter(mode="grid2D", segmentsNumber=N_SEGMENTS)

expl = Explainer(classifier=classif, trackTime=False)
p = Pipeline(classif, seg, expl, segmentationDim="3D", explanationMode="frame", nSegments=N_SEGMENTS, shapSamples=SHAP_SAMPLES)



dfdc_vidlist = VideoLoader.loadFilenamesDFDC(videoCount=N_VIDEOS, 

                                   fakeClassValue=classif.FAKE_CLASS_VAL, realClassValue=classif.REAL_CLASS_VAL)



for (vidName, vidClass) in dfdc_vidlist:

    

    print("Analyzing sequence", vidName)

    

    vidPath = os.path.join(VideoLoader.DFDC_trainVideoDir, vidName)

    # Get image sequence from folder

    imageSequence = classif.getFaceCroppedVideo(vidPath)

    

    p.start(imageSequence, vidClass, vidName)