import torchvision.transforms as transforms

import torch

import torch.utils.data as data

import numpy as np

from sklearn import svm

import librosa

import csv

import time



from customdatasets import TrainDataset, TestDataset



toFloat = transforms.Lambda(lambda x: x / np.iinfo(np.int16).max)



trainDataset = TrainDataset("../input/oeawai/train/kaggle-train", transform=toFloat)

print(len(trainDataset))



testDataset = TestDataset("../input/oeawai/kaggle-test/kaggle-test", transform=toFloat)

print(len(testDataset))
familyClassifier = svm.SVC()



trainLoader = data.DataLoader(trainDataset, batch_size=640, shuffle=True)

start = time.time()

for samples, instrumentsFamily in trainLoader:

    familyClassifier.fit(np.array(samples.data)[:, :16000], np.array(instrumentsFamily.data))

    break # SVM is only fitted to a fixed size of data

print("Fitting the SVM took " + str((time.time()-start)/60) + "mins")
batch_size = 32

testloader = data.DataLoader(testDataset, batch_size=batch_size, shuffle=False) #!!! Shuffle should be false



familyPredictions = np.zeros(len(testDataset), dtype=np.int)

start = time.time()

for index, samples in enumerate(testloader):

    familyPredictions[index*batch_size:(index+1)*batch_size] = familyClassifier.predict(np.array(samples.data)[:, :16000])

print("Classifying took " + str((time.time()-start)/60) + "mins")
import csv

familyPredictionStrings = trainDataset.transformInstrumentsFamilyToString(familyPredictions.astype(int))



with open('SVM-time-submission.csv', 'w', newline='') as writeFile:

    fieldnames = ['Id', 'Predicted']

    writer = csv.DictWriter(writeFile, fieldnames=fieldnames, delimiter=',',

                            quotechar='|', quoting=csv.QUOTE_MINIMAL)

    writer.writeheader()

    for index in range(len(testDataset)):

        writer.writerow({'Id': index, 'Predicted': familyPredictionStrings[index]})

def computeMelspectrogram(numpyArray, sample_rate):

    S = librosa.feature.melspectrogram(y=numpyArray, sr=sample_rate, n_mels=128, fmax=8000)

    return np.log(S+1e-4)

sample_rate = 16000



batch_size = 16

trainLoader = data.DataLoader(trainDataset, batch_size=batch_size, shuffle=True)

mfccs = np.zeros((batch_size, 128))

for samples, instrumentsFamily in trainLoader:

    for index, sample in enumerate(samples):

        mfccs[index] = np.mean(computeMelspectrogram(sample.numpy(), sample_rate), axis=1)

    family = trainDataset.transformInstrumentsFamilyToString(instrumentsFamily.numpy().astype(int))

    break # SVM is only fitted to a fixed size of data



import matplotlib.pyplot as plt

    

for i in range(batch_size):

    plt.plot(mfccs[i])

    print(family[i])

    plt.show()
informedFamilyClassifier = svm.SVC()



trainLoader = data.DataLoader(trainDataset, batch_size=6400, shuffle=True)

start = time.time()

for samples, instrumentsFamily in trainLoader:

    mfccs = np.zeros((len(samples), 128))

    for index, sample in enumerate(samples.data):

        mfccs[index] = np.mean(computeMelspectrogram(sample.numpy(), sample_rate), axis=1)

    informedFamilyClassifier.fit(mfccs, instrumentsFamily.numpy())

    break # SVM is only fitted to a fixed size of data

    

print("Fitting the SVM took " + str((time.time()-start)/60) + "mins")
batch_size = 32

testloader = data.DataLoader(testDataset, batch_size=batch_size, shuffle=False) #!!! Shuffle should be false



informedFamilyPredictions = np.zeros(len(testDataset), dtype=np.int)

start = time.time()

for index, samples in enumerate(testloader):

    mfccs = np.zeros((len(samples), 128))

    for inner_index, sample in enumerate(samples.data):

        mfccs[inner_index] = np.mean(computeMelspectrogram(sample.numpy(), sample_rate), axis=1)

    informedFamilyPredictions[index*batch_size:(index+1)*batch_size] = informedFamilyClassifier.predict(mfccs)



print("Classifying took " + str((time.time()-start)/60) + "mins")
informedFamilyPredictionStrings = trainDataset.transformInstrumentsFamilyToString(informedFamilyPredictions.astype(int))



with open('SVM-informed-submission.csv', 'w', newline='') as writeFile:

    fieldnames = ['Id', 'Predicted']

    writer = csv.DictWriter(writeFile, fieldnames=fieldnames, delimiter=',',

                            quotechar='|', quoting=csv.QUOTE_MINIMAL)

    writer.writeheader()

    for index in range(len(testDataset)):

        writer.writerow({'Id': index, 'Predicted': informedFamilyPredictionStrings[index]})
