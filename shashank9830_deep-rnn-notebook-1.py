# when training the smaller model



!wget "http://shashankdeeplearning.000webhostapp.com/train_small.json" -O "train_small.json"

!wget "http://shashankdeeplearning.000webhostapp.com/model_trained_last.h5" -O "model_trained_last.h5"

!wget "http://shashankdeeplearning.000webhostapp.com/last_session_data.json" -O "last_session_data.json"

!wget "http://shashankdeeplearning.000webhostapp.com/sample_weights.json" -O "sample_weights.json"
import os
working_dir = '/kaggle/working/'

dataset_dir = '/kaggle/input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/'



assert os.path.exists(working_dir + 'train_small.json'), "Upload train_small.json to /kaggle/working/"

assert os.path.exists(working_dir + 'model_trained_last.h5'), "Upload model_trained_last.h5 file to /kaggle/working/"

assert os.path.exists(working_dir + 'last_session_data.json'), "Upload last_session_data.json file to /kaggle/working/"

assert os.path.exists(working_dir + 'sample_weights.json'), "Upload sample_weights.json file to /kaggle/working/"
from tensorflow.keras.models import load_model

import numpy as np

import pydicom

import json

import time
model = load_model('model_trained_last.h5')
with open('last_session_data.json') as file:

    last_session_data = json.load(file)
class DicomCuboidBatchGenerator:



    def __init__(self, partitionLabel, batchSize=32, shuffle=True, resuming=False, last_session_data=None):

        """ Initialize DicomCubeBatchGenerator """



        self.partitionLabel = partitionLabel

        self.batchSize = batchSize

        self.shuffle = shuffle

        with open(f'sample_weights.json') as file:

            self.sampleWeightData = json.load(file)

        with open(f'{partitionLabel}.json') as file:

            self.labelData = json.load(file)

        self.datasetSize = len(self.labelData)

        if resuming:

            self.randomIndexList = np.array(last_session_data['randomIndexList'])

            self.preventChangeAtResumedEpoch = True

        else:

            self.randomIndexList = np.arange(0, self.datasetSize)

            self.preventChangeAtResumedEpoch = False

        



    def on_epoch_begin(self):

        """ After each epoch shuffle the indexes """

        

        if self.preventChangeAtResumedEpoch:

            self.preventChangeAtResumedEpoch = False

        elif self.shuffle:

            np.random.shuffle(self.randomIndexList)



    def __len__(self):

        """ Number of batches per epoch """



        return np.int(np.ceil(self.datasetSize / self.batchSize))



    def getBatch(self, index):

        """ Return batch[index] """



        batchStartIndex = index * self.batchSize

        batchEndIndex = (index + 1) * self.batchSize # actually end index + 1



        if batchEndIndex > self.datasetSize:

            batchEndIndex = self.datasetSize



        return self.generateBatch(batchStartIndex, batchEndIndex)



    def generateBatch(self, batchStartIndex, batchEndIndex):

        """ Generate the batch for the given range """



        curBatchSize = batchEndIndex - batchStartIndex



        X = []

        Y = []

        SW = []



        for i in range(batchStartIndex, batchEndIndex):



            actIdx = self.randomIndexList[i]



            dicomFilename = 'ID_' + self.labelData[actIdx][0] + '.dcm'

            

            imageCube = self.getImageCubeFromDicom(dicomFilename)

            X.append(imageCube)

            labelRowVector = np.array(self.labelData[actIdx][1])

            Y.append(labelRowVector)

            SW.append(self.sampleWeightData[str(self.labelData[actIdx][1])])



        X = np.array(X)

        X = X.reshape(*X.shape, 1)  # add a single channel dimension

        assert X.shape == (curBatchSize, 10, 768, 768, 1), f"Incorrect input batch shape {X.shape}"



        Y = np.array(Y)

        assert Y.shape == (curBatchSize, 6), f"Incorrect output batch shape {Y.shape}"

        

        SW = np.array(SW)

        assert SW.shape == (curBatchSize,), f"Incorrect sample weight list shape {SW.shape}"

        

        return X, Y, SW



    def getImageCubeFromDicom(self, filename):

        """ Read DICOM data and convert it to cube with multiple windows """



        filedir = '/kaggle/input/rsna-intracranial-hemorrhage-detection/'

        filedir += 'rsna-intracranial-hemorrhage-detection/'

        filedir += 'stage_2_train/'

        

        filepath = filedir + filename



        dicomData = pydicom.dcmread(filepath)



        windows = [['default', 0, 0],

                    ['brain', 40, 80],

                    ['subdural-min', 50, 130],

                    ['subdural-mid', 75, 215],

                    ['subdural-max', 100, 300],

                    ['tissue-min', 20, 350],

                    ['tissue-mid', 40, 375],

                    ['tissue-max', 60, 400],

                    ['bone', 600, 2800],

                    ['grey-white', 32, 8]]



        if type(dicomData.WindowCenter) == pydicom.multival.MultiValue:

            windows[0][1] = dicomData.WindowCenter[0]

        else:

            windows[0][1] = dicomData.WindowCenter



        if type(dicomData.WindowWidth) == pydicom.multival.MultiValue:

            windows[0][2] = dicomData.WindowWidth[0]

        else:

            windows[0][2] = dicomData.WindowWidth



        imageCube = []

        for window in windows:

            image = self.getImageForWindow(dicomData, window[1], window[2])

            imageCube.append(image)



        imageCube = np.array(imageCube)



        assert imageCube.shape == (10, 768, 768), "Cube shape incorrect"



        return imageCube



    def getImageForWindow(self, ds, windowCenter, windowWidth):

        """ Get the image for given window settings """

        

        # perform linear transformation on the original pixel_array

        img = ds.pixel_array * ds.RescaleSlope + ds.RescaleIntercept

        

        # pad the image to make sure it's 768 x 768

        l = (768 - ds.Columns) // 2

        r = 768 - ds.Columns - l

        t = (768 - ds.Rows) // 2

        b = 768 - ds.Rows - t

        img = np.pad(img, ((t, b), (l, r)), mode='constant', constant_values=img[0,0])

        

        # perform windowing

        img_min = windowCenter - windowWidth // 2

        img_max = windowCenter + windowWidth // 2

        img[img < img_min] = img_min

        img[img > img_max] = img_max

        

        # rescale the image

        img = (img - img_min) / (img_max - img_min)



        assert img.shape == (768, 768), "Image shape incorrect"



        return img
batchSize = last_session_data['batchSize']



if last_session_data['last_completed'] == True:

    training_gen = DicomCuboidBatchGenerator('train_small', batchSize)

else:

    training_gen = DicomCuboidBatchGenerator('train_small', batchSize, True, True, last_session_data)
steps_training = len(training_gen)



number_of_epochs = last_session_data['number_of_epochs']
# get last session variables

last_executing_epoch_index = last_session_data['last_executing_epoch_index']

last_executed_batch_index = last_session_data['last_executed_batch_index']



#start the training, set it to train for 5 hours (5*60*60 seconds)

start = time.time()

timeToStop = False



print(f'Resuming from : EPOCH {last_executing_epoch_index+1}/{number_of_epochs} and BATCH {last_executed_batch_index+2}/{steps_training}')



for epoch in range(last_executing_epoch_index, number_of_epochs):

    

    training_gen.on_epoch_begin()

    

    for batch_index in range(last_executed_batch_index+1, steps_training):

        

        print(f'Running EPOCH {epoch+1}/{number_of_epochs} and BATCH {batch_index+1}/{steps_training}', end='\r')

        X, Y, SW = training_gen.getBatch(batch_index)

        

        RM = True if batch_index == 0 else False # reset metrics only at the beginning of each epoch



        model.train_on_batch(X, Y, sample_weight=SW, reset_metrics=RM) # train the current batch

        

        last_executed_batch_index = batch_index # update the last executed batch index

                

        # stop if time is more than 5 hours

        if (time.time()-start) > (5.5*60*60):

            timeToStop = True

            break

    

    # breaks the outer loop if inner loop was terminated via break

    if timeToStop:

        break



print(f'Last Run : EPOCH {epoch+1}/{number_of_epochs} and BATCH {batch_index+1}/{steps_training}')

if not timeToStop:

    print('All epochs completed')

else:

    print('Resume the process next time')
model.save('model_trained_last.h5')
last_session_data = dict() # reset the previous last_session_data dictionary



last_session_data['batchSize'] = batchSize

last_session_data['number_of_epochs'] = number_of_epochs



# save the batchSize, last_executing_epoch_index, last_executed_batch_index and number_of_epochs

last_session_data['last_executing_epoch_index'] = 0 if not timeToStop else epoch

last_session_data['last_executed_batch_index'] = -1 if not timeToStop else last_executed_batch_index

last_session_data['last_completed'] = True if not timeToStop else False



# save the randomIndexList

last_session_data['randomIndexList'] = list(map(int, list(training_gen.randomIndexList)))
with open('last_session_data.json', 'w') as file:

    json.dump(last_session_data, file, indent=4)