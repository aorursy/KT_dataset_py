# Import necessary libraries.

import time

zNtim = time.time()

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn import datasets, metrics

from sklearn import linear_model, svm

from sklearn.neural_network import BernoulliRBM

from sklearn.pipeline import Pipeline

zAtim = [[0, time.time()-zNtim, "", "", "", "Import necessary libraries."]]
# Enable on-screen display of plots.

%matplotlib inline
zNtim = time.time()

def fDrvs(zFtrn, zNrhv, zNrtk, zNpto, zNpco):

    '''Define: Make datasets for training and validation.'''

    zDtrn = pd.read_csv(zFtrn)

    zAtrn = zDtrn.as_matrix()

    zAtrX = zAtrn[0:zNrtk, 1:zNpto]/zNpco

    zAtrY = zAtrn[0:zNrtk, 0]

    zAvaX = zAtrn[(zNrhv-int(zNrtk/2)):zNrhv, 1:zNpto]/zNpco

    zAvaY = zAtrn[(zNrhv-int(zNrtk/2)):zNrhv, 0]

    return zDtrn, zAtrX, zAtrY, zAvaX, zAvaY

zAtim = zAtim + [[1, time.time()-zNtim, "", "", "", "Define: Make datasets for training and validation."]]
zNtim = time.time()

def fPtrs(zDtrn, zNpro, zNsho, zNsha):

    '''Define: Plot image zNsho and zNsha images from training dataset.'''

    zAtrm = zDtrn.drop(zDtrn.columns[[0]], axis=1).as_matrix()

    zAimg = zAtrm[zNsho].reshape(zNpro,zNpro)

    plt.figure(figsize=(2, 2))

    plt.title('mnist image %d' % (zNsho), fontsize=12)

    plt.imshow(zAimg, cmap=plt.get_cmap('gray_r'))

    plt.show()

    plt.close()

    #####

    plt.figure(figsize=(10, zNsha/10))

    for zNshu in range(zNsha):

        zAimg = zAtrm[zNshu].reshape(zNpro,zNpro)

        plt.subplot(zNsha/10, 10, zNshu + 1)

        plt.imshow(zAimg, cmap=plt.cm.gray_r, interpolation='nearest')

        plt.xticks(())

        plt.yticks(())

    plt.suptitle('mnist %d images of digits' % (zNsha), fontsize=12)

    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

    plt.show()

    plt.close()

zAtim = zAtim + [[2, time.time()-zNtim, "", "", "", "Define: Plot an image and several images from training dataset."]]
zNtim = time.time()

def fAhis(zArvX, zNpro):

    '''Define: Create arrays of histograms of colors and color gradients of images.'''

    def fAoff(zArsV, zNpro):

        zNlen = zArsV.shape[0]

        zAoff = np.concatenate((zArsV[0:zNlen-1] - zArsV[1:zNlen], zArsV[0:zNlen-zNpro] - zArsV[zNpro:zNlen],

                                zArsV[0:zNlen-zNpro-1] - zArsV[zNpro+1:zNlen]), axis = 0)

        return zAoff

    def fAhit(zArvW):

        zAhit = np.histogram(zArvW, bins=10, range=None, normed=False, weights=None, density=True)

        zAhit = zAhit[0]

        zAhit = np.round(zAhit / np.sum(zAhit), 3)

        return zAhit

    zAhiC = fAhit(zArvX[0])

    zAhiG = fAhit(fAoff(zArvX[0], zNpro))

    for zNsho in range(0, zArvX.shape[0]-1):

        zAhiC = np.vstack((zAhiC, fAhit(zArvX[zNsho])))

        zAhiG = np.vstack((zAhiG, fAhit(fAoff(zArvX[zNsho], zNpro))))

    return zAhiC, zAhiG

zAtim = zAtim + [[3, time.time()-zNtim, "", "", "", "Define: Create arrays of histograms of colors and color gradients."]]
zNtim = time.time()

def fClog():

    '''Define: Make classifiers of logistic regression, support vector machines, and Bernoulli restricted Boltzmann machines.

restricted Boltzmann machines.'''

    zClog = linear_model.LogisticRegression(C = 6000.0)

    zCsvm = svm.SVC(gamma=0.001, C=100.)

    yCrbm = BernoulliRBM(random_state=0, verbose=True, learning_rate = 0.06, n_iter = 10, n_components = 1000)

    yClog = linear_model.LogisticRegression()

    zCrbm = Pipeline(steps=[('yCrbm', yCrbm), ('yClog', yClog)])

    return zClog, zCsvm, zCrbm

zAtim = zAtim + [[4, time.time()-zNtim, "", "", "", "Define: Make classifiers of logistic regression, SVM, and RBM."]]
zNtim = time.time()

def fCfit(zCabc, zAtrX, zAtrY):

    '''Define: Fit the classifier zCabc to a training dataset of images.'''

    zCfit = zCabc.fit(zAtrX, zAtrY)

    return zCfit

zAtim = zAtim + [[5, time.time()-zNtim, "", "", "", "Define: Fit the classifier to a training dataset."]]
zNtim = time.time()

def fRprd(zCfit, zAvaX, zAvaY):

    '''Define: Make predictions on validation dataset zAvaX and get a score.'''

    zAprd = zCfit.predict(zAvaX)

    zRrpt = metrics.classification_report(zAvaY, zAprd)

    zRpcn, zRrcl, zRf1s, zRspt = metrics.precision_recall_fscore_support(zAvaY, zAprd)

    zRpcn, zRrcl, zRf1s, zRspt = np.round(np.average(zRpcn),2), np.round(np.average(zRrcl),2

                                ), np.round(np.average(zRf1s),2), np.round(np.average(zRspt),2)

    return zRrpt, zRpcn, zRrcl, zRf1s

zAtim = zAtim + [[6, time.time()-zNtim, "", "", "", "Define: Make predictions on validation dataset and get a score."]]
# Use MNIST training dataset of images of digits as downloaded from Kaggle's Digit Recognizer cometition.

zNtim = time.time()

zFtrn = "../input/train.csv" # Path of training csv file

zNrhv = 41999 # Number of training records available - 1

zNrtk = 2499 # int(input("Number of records to take for training a model: min 2000 max 28000:   ")) - 1

zNpto = 784 # Number of pixels in each image

zNpro = 28 # Number of pixels in each row of each image

zNpco = 255 # Number of color values possible in a pixel

zAtim = zAtim + [[7, time.time()-zNtim, "", "", "", "Import MNIST training dataset of images of digits."]]
# Import training and validation records from MNIST dataset.

zNtim = time.time()

zDtrn, zAtrX, zAtrY, zAvaX, zAvaY = fDrvs(zFtrn, zNrhv, zNrtk, zNpto, zNpco)

print (zAtrX)

print (zAtrY)

zAtim = zAtim + [[8, time.time()-zNtim, "", "", "", "Import training and validation records from MNIST dataset."]]
# Plot from dataset of images zArvX, image 2 and some images.

zNtim = time.time()

fPtrs(zDtrn, zNpro, zNsho=0, zNsha=20)

zAtim = zAtim + [[9, time.time()-zNtim, "", "", "", "Plot image 2 and some images from test dataset."]]
# Create training arrays of histograms of colors and color gradients of images.

zNtim = time.time()

zAhrC, zAhrG = fAhis(zAtrX, zNpro)

print (zAhrC.shape)

print (zAhrC)

print ()

print (zAhrG.shape)

print (zAhrG)

zAtim = zAtim + [[10, time.time()-zNtim, "", "", "", "Create training arrays of histograms of colors and gradients."]]
# Create validation arrays of histograms of colors and color gradients of images.

zNtim = time.time()

zAhvC, zAhvG = fAhis(zAvaX, zNpro)

print (zAhvC.shape)

print (zAhvC)

print ()

print (zAhvG.shape)

print (zAhvG)

zAtim = zAtim + [[11, time.time()-zNtim, "", "", "", "Create validation arrays of histograms of colors and gradients."]]
# Make classifiers based on logistic regression, support vector machines, and restricted Boltzmann machines.

zNtim = time.time()

zClog, zCsvm, zCrbm = fClog()

zAtim = zAtim + [[12, time.time()-zNtim, "", "", "", "Make classifiers of logistic regression, SBM, and RBM."]]
# Predict using logistic regression.

zNtim = time.time()

print (zAvaX.shape)

zCfit = fCfit(zClog, zAtrX, zAtrY)

zRrpt, zRpcn, zRrcl, zRf1s = fRprd(zCfit, zAvaX, zAvaY)

print("Predict using logistic regression.\n%s\n" % (zRrpt))

zAtim = zAtim + [[13, time.time()-zNtim, zRpcn, zRrcl, zRf1s, "Predict using logistic regression."]]
# Predict using support vector machines.

zNtim = time.time()

zCfit = fCfit(zCsvm, zAtrX, zAtrY)

zRrpt, zRpcn, zRrcl, zRf1s = fRprd(zCfit, zAvaX, zAvaY)

print("Predict using support vector machines.\n%s\n" % (zRrpt))

zAtim = zAtim + [[14, time.time()-zNtim, zRpcn, zRrcl, zRf1s, "Predict using SVM."]]
# Predict using Bernoulli restricted Boltzmann machines.

zNtim = time.time()

zCfit = fCfit(zCrbm, zAtrX, zAtrY)

zRrpt, zRpcn, zRrcl, zRf1s = fRprd(zCfit, zAvaX, zAvaY)

print("Predict using Bernoulli restricted Boltzmann machines.\n%s\n" % (zRrpt))

zAtim = zAtim + [[15, time.time()-zNtim, zRpcn, zRrcl, zRf1s, "Predict using RBM."]]
# Predict using logistic regression with color statistics.

zNtim = time.time()

zCfit = fCfit(zClog, zAhrC, zAtrY)

zRrpt, zRpcn, zRrcl, zRf1s = fRprd(zCfit, zAhvC, zAvaY)

print("Predict using logistic regression with color statistics.\n%s\n" % (zRrpt))

zAtim = zAtim + [[16, time.time()-zNtim, zRpcn, zRrcl, zRf1s, "Predict using logistic regression with color statistics."]]
# Predict using support vector machines with color statistics.

zNtim = time.time()

zCfit = fCfit(zCsvm, zAhrC, zAtrY)

zRrpt, zRpcn, zRrcl, zRf1s = fRprd(zCfit, zAhvC, zAvaY)

print("Predict using support vector machines with color statistics.\n%s\n" % (zRrpt))

zAtim = zAtim + [[17, time.time()-zNtim, zRpcn, zRrcl, zRf1s, "Predict using SVM with color statistics."]]
# Predict using Bernoulli restricted Boltzmann machines with color statistics.

zNtim = time.time()

zCfit = fCfit(zCrbm, zAhrC, zAtrY)

zRrpt, zRpcn, zRrcl, zRf1s = fRprd(zCfit, zAhvC, zAvaY)

print("Predict using Bernoulli restricted Boltzmann machines with color statistics.\n%s\n" % (zRrpt))

zAtim = zAtim + [[18, time.time()-zNtim, zRpcn, zRrcl, zRf1s, "Predict using RBM with color statistics."]]
# Predict using logistic regression with color gradient statistics.

zNtim = time.time()

zCfit = fCfit(zClog, zAhrG, zAtrY)

zRrpt, zRpcn, zRrcl, zRf1s = fRprd(zCfit, zAhvG, zAvaY)

print("Predict using logistic regression with color gradient statistics.\n%s\n" % (zRrpt))

zAtim = zAtim + [[19, time.time()-zNtim, zRpcn, zRrcl, zRf1s, "Predict using logistic regression with gradient statistics."]]
# Predict using support vector machines with color gradient statistics.

zNtim = time.time()

zCfit = fCfit(zCsvm, zAhrG, zAtrY)

zRrpt, zRpcn, zRrcl, zRf1s = fRprd(zCfit, zAhvG, zAvaY)

print("Predict using support vector machines with color gradient statistics.\n%s\n" % (zRrpt))

zAtim = zAtim + [[20, time.time()-zNtim, zRpcn, zRrcl, zRf1s, "Predict using SVM with gradient statistics."]]
# Predict using Bernoulli restricted Boltzmann machines with color gradient statistics.

zNtim = time.time()

zCfit = fCfit(zCrbm, zAhrG, zAtrY)

zRrpt, zRpcn, zRrcl, zRf1s = fRprd(zCfit, zAhvG, zAvaY)

print("Predict using Bernoulli restricted Boltzmann machines with color gradients.\n%s\n" % (zRrpt))

zAtim = zAtim + [[21, time.time()-zNtim, zRpcn, zRrcl, zRf1s, "Predict using RBM with gradient statistics."]]
# Print the timings taken for various activities.

print (" SerN     Time  Preci  Recal  F1Sco  Activity")

for i in range(22):

    print ("%5d %8.3f %6s %6s %6s  %10s" % (zAtim[i][0], zAtim[i][1], zAtim[i][2], zAtim[i][3], zAtim[i][4], zAtim[i][5]))