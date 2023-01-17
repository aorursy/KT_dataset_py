# keras(python, tensorflow, numpy), Ipython, pydot, graphviz, and h5py
import keras as K
print("Keras Version is: ", K.__version__)
import tensorflow as tf
print("Tensorflow Version is: ", tf.__version__)
from keras.utils.vis_utils import model_to_dot, plot_model
from IPython.display import SVG
import os, random, re, h5py
import numpy as np
# keras utility to load saved models
from keras.models import load_model
# keras imports to load images and to convert them to arrays
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
# to convert numeric to categorical labels
from keras.utils import to_categorical
# global variables
# transfer models we want to use
TRANSFER_MODELS=['xception', 'inception_v3', 'resnet50', 'inception_resnet_v2', 'mobilenet']
#TRANSFER_MODELS=['inception_v3']
# classes we are interested in
CLASSES=np.array([['cat'.encode("utf-8")], ['dog'.encode("utf-8")]])
print(os.listdir("../input"))
#PRETRAINED_MODELS_DIRECTORY=os.listdir("../input")[0]
#print (PRETRAINED_MODELS_DIRECTORY)

DOGSVSCATS_DATA_DIRECTORY="../input"
print (DOGSVSCATS_DATA_DIRECTORY)
TRAIN_IMAGELIST=os.listdir(os.path.join(DOGSVSCATS_DATA_DIRECTORY, "train"))
random.shuffle(TRAIN_IMAGELIST)
TEST_IMAGELIST=os.listdir(os.path.join(DOGSVSCATS_DATA_DIRECTORY, "test"))
random.shuffle(TEST_IMAGELIST)
print ("{} training samples".format(len(TRAIN_IMAGELIST)))
print ("{} test samples".format(len(TEST_IMAGELIST)))
"""select model and make model-specific keras imports"""
def SelectTransferModel(modelidentifier='inception_v3'):
    print("=====================================================")
    print("Selected Model: {}...".format(modelidentifier.title()))
    print("=====================================================")
    if modelidentifier=='xception':
        # each transfer model has a method to preprocess inputs
        from keras.applications.xception import Xception, preprocess_input
        transfermodelmethod=Xception
        # xception expects images of size 299x299
        resize=(299, 299)
    if modelidentifier=='inception_v3':
        # each transfer model has a method to preprocess inputs
        from keras.applications.inception_v3 import InceptionV3, preprocess_input
        transfermodelmethod=InceptionV3
        # inception_v3 expects images of size 299x299
        resize=(299, 299)
    if modelidentifier=='resnet50':
        from keras.applications.resnet50 import ResNet50, preprocess_input
        # resnet50 expects images of size 224x224
        transfermodelmethod=ResNet50
        resize=(224, 224)
    if modelidentifier=='inception_resnet_v2':
        from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
        # inception_resnet_v2 expects images of size 224x224
        transfermodelmethod=InceptionResNetV2
        resize=(299, 299)
    if modelidentifier=='mobilenet':
        from keras.applications.mobilenet import MobileNet, preprocess_input
        transfermodelmethod=MobileNet
        # mobilenetv2 expects images of size 224x224
        resize=(224, 224)
    return transfermodelmethod, resize, preprocess_input
def ProcessImagebatch (transfermodel, modelidentifier, resize, preprocessmethod, imagebatch, mode="train"):
    examples=len(imagebatch)
    inputids=np.zeros((examples,), dtype=int)
    inputimages=np.zeros((examples, resize[0], resize[1], 3))
    inputlabels=np.zeros((examples, 1))
    print(">>Loading {} images...".format(examples), end="")
    for i, filename in enumerate(imagebatch):
        # record the corresponding id from filename
        if mode=="train":
            id=filename.split(".")[1]
        elif mode=="test":
            id=filename.split(".")[0]
        inputids[i]=id
        #print (id)
        # load image
        filenamepath=os.path.join(DOGSVSCATS_DATA_DIRECTORY, mode, filename)
        image = load_img(filenamepath, target_size=resize)
        # convert image to array and append to inputimages array
        inputimages[i]=img_to_array(image)
        # record the corresponding label from filename
        if mode=="train":
            #label=os.path.split(filename)[1].split(".")[0]
            label=filename.split(".")[0]
            #print (label)
            if str(label) == "cat":
                inputlabels[i]=0
            elif str(label) == "dog":
                inputlabels[i]=1
    print("done")
    # preprocess images to transfer model requirements
    print (">>Preprocessing {} loaded images for {}...".format(inputimages.shape[0], modelidentifier.title()), end="")
    preprocessedinputimages = preprocessmethod(inputimages.copy())
    print ("done")
    print (">>Converting {} preprocessed images to features...".format(preprocessedinputimages.shape[0]))
    featuresbatch=transfermodel.predict(preprocessedinputimages, verbose=1)
    # to keep the function reusable, the function creates (spurious) labels for test inputs too,
    # but the returned labels will be discarded
    labelsbatch = to_categorical(inputlabels, num_classes=CLASSES.shape[0])
    idsbatch=inputids
    return featuresbatch, labelsbatch, idsbatch, preprocessedinputimages
def Images2Features(transfermodel, modelidentifier, resize, preprocessmethod, imagelist, batchsize=1024, mode="train"):
    examples=len(imagelist)
    ids=np.zeros((examples,), dtype=int)
    features=np.zeros((examples, transfermodel.output.shape[1]))
    labels=np.zeros((examples, CLASSES.shape[0]))
    batches, leftover=divmod(examples,batchsize)
    #print (batches, leftover)
    # process all batches
    for batch in range(batches):
        startindex=batch*batchsize
        endindex=batch*batchsize+batchsize
        print (">>Converting input images {}-{}...".format(startindex, endindex))
        imagebatch=imagelist[startindex:endindex]
        featuresbatch, labelsbatch, idsbatch, _ = ProcessImagebatch(transfermodel, modelidentifier, resize, preprocessmethod, imagebatch, mode=mode)
        features[startindex:endindex]=featuresbatch
        labels[startindex:endindex]=labelsbatch
        ids[startindex:endindex]=idsbatch
    # process leftover
    if leftover:
        startindex=batches*batchsize
        endindex=batches*batchsize+leftover
        print (">>Converting input images {}-{}...".format(startindex, endindex))
        imagebatch=imagelist[startindex:endindex]
        featuresbatch, labelsbatch, idsbatch, _ = ProcessImagebatch(transfermodel, modelidentifier, resize, preprocessmethod, imagebatch, mode=mode)
        features[startindex:endindex]=featuresbatch
        labels[startindex:endindex]=labelsbatch
        ids[startindex:endindex]=idsbatch
    return features, labels, ids
for modelidentifier in TRANSFER_MODELS:
    savedtrainfeaturesfilename="kaggledogsvscatsredux-{}-features-trainsamples-{}.h5".format(modelidentifier,
                                                                                             len(TRAIN_IMAGELIST))
    savedtestfeaturesfilename="kaggledogsvscatsredux-{}-features-testsamples-{}.h5".format(modelidentifier,
                                                                                          len(TEST_IMAGELIST))
    #print (savedtrainfeaturesfilename)
    #print (savedtestfeaturesfilename)
    transfermodelmethod, resize, preprocessmethod = SelectTransferModel(modelidentifier)
    transfermodel = transfermodelmethod(weights='imagenet', include_top=False, pooling='avg')
    # create training features
    print("****************************************************************************")
    print ("...Creating {} Training Features and Labels...".format(len(TRAIN_IMAGELIST)))
    print("****************************************************************************")
    train_x, train_y, train_ids = Images2Features(transfermodel, modelidentifier, resize, preprocessmethod, TRAIN_IMAGELIST,  mode="train")
    # save training feature inputs
    with h5py.File(savedtrainfeaturesfilename, "w") as file:
        file.create_dataset("train_x", data=train_x)
        file.create_dataset("train_y", data=train_y)
        file.create_dataset("train_ids", data=train_ids)
        file.create_dataset("classes", data=CLASSES)
        file.close()
    print (">>>Saved {} training features to {}".format(train_x.shape[0],
                                                        savedtrainfeaturesfilename))
    print("****************************************************************************")
    print ("...Creating {} Test Features...".format(len(TEST_IMAGELIST)))
    print("****************************************************************************")
    test_x, _, test_ids = Images2Features(transfermodel, modelidentifier, resize, preprocessmethod, TEST_IMAGELIST,  mode="test")
    with h5py.File(savedtestfeaturesfilename, "w") as file:
        file.create_dataset("test_x", data=test_x)
        file.create_dataset("test_ids", data=test_ids)
        file.create_dataset("classes", data=CLASSES)
        file.close()
    print (">>>Saved {} test features to {}".format(test_x.shape[0],
                                                    savedtestfeaturesfilename))
    
    