# the code below implements 5 classifiers in sequence, one each for transfer
# features in the companion dataset.  The companion dataset was previously
# generated from transfer models inception_v3, inception_restnet_v2, resnet50,
# xception, and mobilenet using keras.applications.
# basic imports
import os, h5py, datetime
import numpy as np
# keras specific imports
# to create models and add layers
from keras import models, layers
# to add specific types of layers
from keras.layers import Dense, Input
# to use keras optimizers
from keras import optimizers
# to visualize keras models
from keras.utils.vis_utils import model_to_dot
from IPython.display import SVG
# global variables
# directory where feature files are available
FEATURES_DIRECTORY="../input"

# transfer models we want to use
# test on one set of features in the dataset to get familiar with the code
TRANSFER_MODELS=['inception_resnet_v2']
# once ready, let the code rip on all models and produce all results
# in less than 10 minutes.  Remember to turn on GPU.
# TRANSFER_MODELS=['xception', 'inception_v3', 'resnet50', 'inception_resnet_v2', 'mobilenet']

# number of examples of transfer features available
TRAIN_SAMPLES=25000
TEST_SAMPLES=12500

# we will use 80% of the examples for training, 20% for validation
TRAIN_SIZE=int(TRAIN_SAMPLES*0.8)
VALIDATE_SIZE=int(TRAIN_SAMPLES-TRAIN_SIZE)

# fully connected classifer parameters
LAYERS_DIMS=(1024, 512, 256, 128)
DROPOUT=0.5
EPOCHS=10
BATCHSIZE=256

# review parameter
# review predictions that do not have 100% probability
THRESHOLD=1.0
print(os.listdir("../input"))
def LoadFeatures (modelidentifier):
    savedtrainfeaturesfilename="kaggledogsvscatsredux-{}-features-trainsamples-{}.h5".format(modelidentifier,
                                                                                             TRAIN_SAMPLES)
    savedtestfeaturesfilename="kaggledogsvscatsredux-{}-features-testsamples-{}.h5".format(modelidentifier,
                                                                                           TEST_SAMPLES)
    #print (savedtrainfeaturesfilename)
    #print (savedtestfeaturesfilename)
    file=h5py.File(os.path.join(FEATURES_DIRECTORY, savedtrainfeaturesfilename), "r")
    keys=list(file.keys())
    train_x = np.array(file["train_x"][:])
    train_y = np.array(file["train_y"][:])
    train_ids = np.array(file["train_ids"][:])
    classes = np.array(file["classes"][:])
    file.close()
    print("*****************************************************************************************")
    print("{} contains:\n\tDatasets:{}".format(savedtrainfeaturesfilename, keys))
    print ("\tLoaded {} training examples, {} labels, and {} ids for {} classes".format(train_x.shape[0],
                                                                                        train_y.shape[0],
                                                                                        train_ids.shape[0],
                                                                                        classes.shape[0]))
    
    file=h5py.File(os.path.join(FEATURES_DIRECTORY, savedtestfeaturesfilename), "r")
    keys=list(file.keys())
    test_x = np.array(file["test_x"][:])
    test_ids = np.array(file["test_ids"][:])
    file.close()
    print("{} contains:\n\tDatasets:{}".format(savedtestfeaturesfilename, keys))
    print ("\tLoaded {} test examples and {} test ids to predict for {} classes".format(test_x.shape[0],
                                                                                        test_ids.shape[0],
                                                                                        classes.shape[0]))
    return train_x, train_y, train_ids, classes, test_x, test_ids
def CreateClassifier (train_x, layersdims, numclasses):
    model = models.Sequential()
    model.add(layers.Dense(layersdims[0], activation='relu', input_shape=train_x.shape[1:2]))
    model.add(layers.Dropout(DROPOUT))
    for i in range(1, len(LAYERS_DIMS)):
        model.add(layers.Dense(layersdims[i], activation='relu'))
        model.add(layers.Dropout(DROPOUT))
    model.add(layers.Dense(numclasses, activation='softmax'))
    model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return model
def PredictAndSave(modelidentifier, classifier, test_x, classes, test_ids, threshold=1.0):
    examples=test_x.shape[0]
    print ("Making Predictions on {} Images...".format(examples))
    predictions = classifier.predict(test_x)
    
    time=datetime.datetime.now()
    timestamp="{}-{}-{}-{}-{}-{}".format(time.month, time.day, time.year, time.hour, time.minute, time.second)
    
    # change 'kanwalindersingh' to your moniker.  too lazy to make this a variable :)
    submitcsvfilename="dogsvscatsredux-kanwalindersingh-{}-base-submit-{}.csv".format(modelidentifier, timestamp)
    reviewcsvfilename="dogsvscatsredux-kanwalindersingh-{}-base-review-{}.csv".format(modelidentifier, timestamp)    
    
    # write csv header
    csvsubmit="id,label\n"
    # write review header
    # 'dogs vs cats redux' submission format is comma separated
    # id from test set filename and the probability that the id
    # is a dog
    csvreview="id,label,probability\n"

    for i in range(examples):
        classindex=np.argmax(predictions[i])
        id=test_ids[i]
        label=np.squeeze(classes)[classindex].decode("utf-8")
        dogprobability=round(predictions[i][1], 1)
        probability=round(predictions[i][classindex], 1)
        
        # write submit record
        csvsubmit="{}{},{}\n".format(csvsubmit, id, dogprobability)
        # write review record if probability is less than threshold
        if probability < threshold:
            csvreview="{}{},{},{}\n".format(csvreview, str(id), str(label), str(probability))

    # write kaggle submit file
    with open(submitcsvfilename,'w') as file:
        file.write(csvsubmit)
    print ("Saved Kaggle Submission file {}".format (submitcsvfilename))
    # write review file
    with open(reviewcsvfilename,'w') as file:
        file.write(csvreview)
    print ("Saved Review file {}".format (reviewcsvfilename))
    return predictions
print("*****************************************************************************************")
for modelidentifier in TRANSFER_MODELS:
    # load input features for classifier
    print ("Using Transfer Learning Features from model: {}...".format(modelidentifier))
    train_x_input, train_y_input, train_ids, classes, test_x, test_ids = LoadFeatures(modelidentifier)
    # define training and validation sets
    print("Using 80% of input data for Training, 20% for Validation...")
    train_x=train_x_input[0:TRAIN_SIZE]
    train_y=train_y_input[0:TRAIN_SIZE]
    validate_x=train_x_input[TRAIN_SIZE:TRAIN_SIZE+VALIDATE_SIZE]
    validate_y=train_y_input[TRAIN_SIZE:TRAIN_SIZE+VALIDATE_SIZE]
    print ("Training Inputs Shape: {}, Training Labels Shape: {}".format(train_x.shape, train_y.shape))
    print ("Validation Inputs Shape: {}, Validation Labels Shape: {}".format(validate_x.shape, validate_y.shape))
    print("Creating classifier with layer dimensions: {}, to classify classes: {} and {}".format(LAYERS_DIMS,
                                                                                                 np.squeeze(classes)[0].decode("utf-8"),
                                                                                                 np.squeeze(classes)[1].decode("utf-8")))
    classifier=CreateClassifier(train_x, LAYERS_DIMS, classes.shape[0])
    print("Training Classifier...")
    classifier.fit(train_x, train_y,
                   epochs=EPOCHS,
                   batch_size=BATCHSIZE,
                   validation_data=(validate_x, validate_y))
    print("Making Predictions and Saving Submission and Review files...")
    predictions=PredictAndSave(modelidentifier, classifier, test_x, classes, test_ids, threshold=THRESHOLD)
#!wc -l dogsvscatsredux-kanwalindersingh-inception_v3-base-review-8-21-2018-18-5-1.csv
#!cat dogsvscatsredux-kanwalindersingh-inception_v3-base-review-8-21-2018-18-5-1.csv