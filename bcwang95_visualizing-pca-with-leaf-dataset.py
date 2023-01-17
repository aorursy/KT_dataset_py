import pandas as pd

import numpy as np

import matplotlib

import matplotlib.image as mpimg

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec



from sklearn import model_selection

from sklearn import decomposition

from sklearn import linear_model

from sklearn import ensemble

from sklearn import neighbors

from sklearn.preprocessing import LabelEncoder

from sklearn.neighbors import KernelDensity

from sklearn.manifold import TSNE

from sklearn.metrics import accuracy_score



from skimage.transform import rescale

from scipy import ndimage as ndi



matplotlib.style.use('fivethirtyeight')
#%% load the data

dataDir = '../input/'

trainData = pd.read_csv(dataDir + 'train.csv')

classEncoder = LabelEncoder()

trainLabels = classEncoder.fit_transform(trainData.ix[:,'species'])

trainIDs = np.array(trainData.ix[:,'id'])



#plt.figure()

#for k in range(28):

#    randTrainInd = np.random.randint(len(trainIDs))

#    randomID = trainIDs[randTrainInd]

#    imageFilename = dataDir + 'images/' + str(randomID) + '.jpg'

#    plt.subplot(4,7,k+1); plt.imshow(mpimg.imread(imageFilename), cmap='gray')

#    plt.title(classEncoder.classes_[trainLabels[randTrainInd]]); plt.axis('off')



##%% go over training images and store them in a list

numImages = 1584



shapesMatrix = np.zeros((2,numImages))

listOfImages = []

for k in range(numImages):

    imageFilename = dataDir + 'images/' + str(k+1) + '.jpg'

    currImage = mpimg.imread(imageFilename)

    shapesMatrix[:,k] = np.shape(currImage)

    listOfImages.append(currImage)

    

# create a large 3d array with all images

maxShapeSize = shapesMatrix.max(axis=1)

for k in range(len(maxShapeSize)):

    if maxShapeSize[k] % 2 == 0:

        maxShapeSize[k] += 311

    else:

        maxShapeSize[k] += 310

    

fullImageMatrix3D = np.zeros(np.hstack((maxShapeSize,np.shape(shapesMatrix[1]))).astype(int),dtype=np.dtype('u1'))

destXc = (maxShapeSize[1]+1)/2; destYc = (maxShapeSize[0]+1)/2

for k, currImage in enumerate(listOfImages):

    Yc, Xc = ndi.center_of_mass(currImage)

    Xd = destXc - Xc; Yd = destYc - Yc

    fullImageMatrix3D[round(Yd):round(Yd)+np.shape(currImage)[0],round(Xd):round(Xd)+np.shape(currImage)[1],k] = currImage



#plt.figure()

#for k in range(28):

#    randInd = np.random.randint(np.shape(fullImageMatrix3D)[2])

#    plt.subplot(4,7,k+1); plt.imshow(fullImageMatrix3D[:,:,randInd], cmap='gray'); plt.axis('off')



##%% remove redundent rows and columns

#plt.figure(); 

#plt.subplot(1,2,1); plt.imshow(fullImageMatrix3D.mean(axis=2),cmap='gray'); plt.axis('off')

#plt.subplot(1,2,2); plt.imshow(fullImageMatrix3D.mean(axis=2) > 0,cmap='gray'); plt.axis('off')



xValid = fullImageMatrix3D.mean(axis=2).sum(axis=0) > 0

yValid = fullImageMatrix3D.mean(axis=2).sum(axis=1) > 0

xLims = (np.nonzero(xValid)[0][0],np.nonzero(xValid)[0][-1])

yLims = (np.nonzero(yValid)[0][0],np.nonzero(yValid)[0][-1])

fullImageMatrix3D = fullImageMatrix3D[yLims[0]:yLims[1],xLims[0]:xLims[1],:]



#plt.figure()

#for k in range(28):

#    randInd = np.random.randint(np.shape(fullImageMatrix3D)[2])

#    plt.subplot(4,7,k+1); plt.imshow(fullImageMatrix3D[:,:,randInd], cmap='gray'); plt.axis('off')



##%% scale down all images

rescaleFactor = 0.15



scaledDownImage = rescale(fullImageMatrix3D[:,:,0],rescaleFactor)

scaledDownImages = np.zeros(np.hstack((np.shape(scaledDownImage),np.shape(fullImageMatrix3D)[2])),dtype=np.dtype('f4'))

for imInd in range(np.shape(fullImageMatrix3D)[2]):

    scaledDownImages[:,:,imInd] = rescale(fullImageMatrix3D[:,:,imInd],rescaleFactor)

    

del fullImageMatrix3D
np.random.seed(1) # use a nice looking random seed



matplotlib.rcParams['font.size'] = 4

matplotlib.rcParams['figure.figsize'] = (9,7)    

plt.figure();

for k in range(25):

    randInd = np.random.randint(np.shape(scaledDownImages)[2])

    plt.subplot(5,5,k+1); plt.imshow(scaledDownImages[:,:,randInd], cmap='gray'); plt.axis('off')

    plt.title('imageID = ' + str(randInd))

plt.tight_layout()
#%% define GaussianModel class



class GaussianModel:

    def __init__(self, X, numBasisFunctions=10, objectPixels=None):

        '''

        inputs: 

            X                    - numSamples x numDimentions matrix

            numBasisFunctions       - number of basis function to use

            objectPixels (optional) - an binnary mask image used for presentation

                                      will be used as Im[objectPixels] = dataSample

                                      must satisfy objectPixels.ravel().sum() = X.shape[1]

        '''

        

        self.numBasisFunctions = numBasisFunctions        

        if objectPixels == None:

            self.objectPixels = np.ones((1,X.shape[1]),dtype=np.bool)

        else:

            self.objectPixels = objectPixels

        assert(self.objectPixels.ravel().sum() == X.shape[1])



        PCAModel = decomposition.PCA(n_components=numBasisFunctions, whiten=True)

        self.dataRepresentation = PCAModel.fit_transform(X)

        self.PCAModel = PCAModel



    def RepresentUsingModel(self, X):

        return self.PCAModel.transform(X)



    def ReconstructUsingModel(self, X_transformed):

        return self.PCAModel.inverse_transform(X_transformed)



    def InterpretUsingModel(self, X):

        return self.PCAModel.inverse_transform(self.PCAModel.transform(X))



    # shows the eigenvectors of the gaussian covariance matrix

    def ShowVarianceDirections(self, numDirectionsToShow=16):

        numDirectionsToShow = min(numDirectionsToShow, self.numBasisFunctions)

        

        numFigRows = 4; numFigCols = 4;

        numDirectionsPerFigure = numFigRows*numFigCols

        numFigures = int(np.ceil(float(numDirectionsToShow)/numDirectionsPerFigure))

        

        for figureInd in range(numFigures):

            plt.figure()

            for plotInd in range(numDirectionsPerFigure):

                eigVecInd = numDirectionsPerFigure*figureInd + plotInd

                if eigVecInd >= self.numBasisFunctions:

                    break

                deltaImage = np.zeros(np.shape(self.objectPixels))

                deltaImage[self.objectPixels] = self.PCAModel.components_[eigVecInd,:].ravel()



                plt.subplot(numFigRows,numFigCols,plotInd+1)

                if np.shape(self.objectPixels)[0] == 1:

                    plt.plot(deltaImage)

                else:

                    plt.imshow(deltaImage); plt.axis('off')

                plt.title(str(100*self.PCAModel.explained_variance_ratio_[eigVecInd]) + '% explained');

            plt.tight_layout()

            

    # shows several random model reconstructions

    def ShowReconstructions(self, X, numReconstructions=5):

        assert(np.shape(X)[1] == self.objectPixels.ravel().sum())

        numSamples = np.shape(X)[0]

        numReconstructions = min(numReconstructions, numSamples)

        

        originalImage      = np.zeros(np.shape(self.objectPixels))

        reconstructedImage = np.zeros(np.shape(self.objectPixels))

        

        numReconstructionsPerFigure = min(5, numReconstructions)

        numFigures = int(np.ceil(float(numReconstructions)/numReconstructionsPerFigure))

        

        for figureInd in range(numFigures):

            plt.figure()

            for plotCol in range(numReconstructionsPerFigure):

                dataSampleInd = np.random.randint(numSamples)

                originalImage[self.objectPixels] = X[dataSampleInd,:].ravel()

                reconstructedImage[self.objectPixels] = self.InterpretUsingModel(np.reshape(X[dataSampleInd,:],[1,-1])).ravel()

                diffImage = abs(originalImage - reconstructedImage)

                

                # original image

                plt.subplot(3,numReconstructionsPerFigure,0*numReconstructionsPerFigure+plotCol+1)

                if np.shape(self.objectPixels)[0] == 1:

                    plt.plot(originalImage); plt.title('original signal')

                else:

                    plt.imshow(originalImage, cmap='gray'); plt.title('original image'); plt.axis('off')

                    

                # reconstred image

                plt.subplot(3,numReconstructionsPerFigure,1*numReconstructionsPerFigure+plotCol+1)

                if np.shape(self.objectPixels)[0] == 1:

                    plt.plot(reconstructedImage); plt.title('reconstructed signal')

                else:

                    plt.imshow(reconstructedImage, cmap='gray'); plt.title('reconstructed image'); plt.axis('off')



                # diff image

                plt.subplot(3,numReconstructionsPerFigure,2*numReconstructionsPerFigure+plotCol+1)

                if np.shape(self.objectPixels)[0] == 1:

                    plt.plot(diffImage); plt.title('abs difference signal')

                else:

                    plt.imshow(diffImage, cmap='gray'); plt.title('abs difference image'); plt.axis('off')

            plt.tight_layout()



    # shows distrbution along the variance directions and several images along that variance direction

    def ShowModelVariations(self, numVariations=5):

        #matplotlib.rcParams['font.size'] = 14



        showAsTraces = (np.shape(self.objectPixels)[0] == 1)

        numVariations = min(numVariations, self.numBasisFunctions)

                

        numVarsPerFigure = min(5,numVariations)

        numFigures = int(np.ceil(float(numVariations)/numVarsPerFigure))

        

        lowRepVec     = np.percentile(self.dataRepresentation, 2, axis=0)

        medianRepVec  = np.percentile(self.dataRepresentation, 50, axis=0)

        highRepVec    = np.percentile(self.dataRepresentation, 98, axis=0)



        for figureInd in range(numFigures):

            plt.figure()

            for plotCol in range(numVarsPerFigure):

                eigVecInd = numVarsPerFigure*figureInd+plotCol

                if eigVecInd >= self.numBasisFunctions:

                    break



                # create the low and high precentile representation activation vectors

                currLowPrecentileRepVec             = medianRepVec.copy()

                currLowPrecentileRepVec[eigVecInd]  = lowRepVec[eigVecInd]

                currHighPrecentileRepVec            = medianRepVec.copy()

                currHighPrecentileRepVec[eigVecInd] = highRepVec[eigVecInd]



                # create blank images

                deltaImage          = np.zeros(np.shape(self.objectPixels))

                medianImage         = np.zeros(np.shape(self.objectPixels))

                lowPrecentileImage  = np.zeros(np.shape(self.objectPixels))

                highPrecentileImage = np.zeros(np.shape(self.objectPixels))



                # fill the object pixels with the relevant data

                deltaImage[self.objectPixels]          = self.PCAModel.components_[eigVecInd,:].ravel()

                lowPrecentileImage[self.objectPixels]  = self.ReconstructUsingModel(currLowPrecentileRepVec).ravel()

                medianImage[self.objectPixels]         = self.ReconstructUsingModel(medianRepVec).ravel()

                highPrecentileImage[self.objectPixels] = self.ReconstructUsingModel(currHighPrecentileRepVec).ravel()



                # calculate the Gaussian smoothed distribution of values along the eignevector direction

                sigmaOfKDE = 0.12

                pdfStart   = min(self.dataRepresentation[:,eigVecInd]) - 3*sigmaOfKDE

                pdfStop    = max(self.dataRepresentation[:,eigVecInd]) + 3*sigmaOfKDE

                xAxis = np.linspace(pdfStart,pdfStop,200)

                PDF_Model = KernelDensity(kernel='gaussian', bandwidth=sigmaOfKDE).fit(self.dataRepresentation[:,eigVecInd].reshape(-1,1))

                logPDF = PDF_Model.score_samples(xAxis.reshape(-1,1))



                # show distribution of current component 

                plt.subplot(5,numVarsPerFigure,0*numVarsPerFigure+plotCol+1)

                plt.fill(xAxis, np.exp(logPDF), fc='b');

                plt.title(str(100*self.PCAModel.explained_variance_ratio_[eigVecInd]) + '% explained'); 

                

                # show variance direction (eigenvector)

                plt.subplot(5,numVarsPerFigure,1*numVarsPerFigure+plotCol+1);

                if showAsTraces:

                    plt.plot(deltaImage); plt.title('eigenvector ' + str(eigVecInd))

                else:

                    plt.imshow(deltaImage); plt.title('eigenvector ' + str(eigVecInd)); plt.axis('off')



                # show 2nd precentile image

                plt.subplot(5,numVarsPerFigure,2*numVarsPerFigure+plotCol+1)

                if showAsTraces:

                    plt.plot(lowPrecentileImage); plt.title('2nd precentile')

                else:

                    plt.imshow(lowPrecentileImage, cmap='gray'); plt.title('2nd precentile image'); plt.axis('off')



                # show median image

                plt.subplot(5,numVarsPerFigure,3*numVarsPerFigure+plotCol+1)

                if showAsTraces:

                    plt.plot(medianImage); plt.title('median signal')

                else:

                    plt.imshow(medianImage, cmap='gray'); plt.title('median Image'); plt.axis('off')



                # show 98th precentile image

                plt.subplot(5,numVarsPerFigure,4*numVarsPerFigure+plotCol+1)

                if showAsTraces:

                    plt.plot(highPrecentileImage); plt.title('98th precentile')

                else:

                    plt.imshow(highPrecentileImage, cmap='gray'); plt.title('98th precentile image'); plt.axis('off')

            plt.tight_layout()

        

    # shows distrbution along the variance directions and several images along that variance direction

    def ShowSingleComponentVariation(self, X, listOfComponents=[0,1]):

        #matplotlib.rcParams['font.size'] = 14



        showAsTraces = (np.shape(self.objectPixels)[0] == 1)

        assert(all([(x in range(self.numBasisFunctions)) for x in listOfComponents]))

                

        X_rep = self.RepresentUsingModel(X)

        

        percentilesToShow = [1,20,40,60,80,99]

        numReadDataSamplePerPercentile = 4

        representationPercentiles = []

        for percentile in percentilesToShow:

            representationPercentiles.append(np.percentile(self.dataRepresentation, percentile, axis=0))

        medianRepVec =  np.percentile(self.dataRepresentation, 50, axis=0)



        for eigVecInd in listOfComponents:

            plt.figure(); gs = gridspec.GridSpec(numReadDataSamplePerPercentile+2,len(percentilesToShow))



            # calculate the Gaussian smoothed distribution of values along the eignevector direction

            sigmaOfKDE = 0.12

            pdfStart   = min(self.dataRepresentation[:,eigVecInd]) - 3*sigmaOfKDE

            pdfStop    = max(self.dataRepresentation[:,eigVecInd]) + 3*sigmaOfKDE

            xAxis = np.linspace(pdfStart,pdfStop,200)

            PDF_Model = KernelDensity(kernel='gaussian', bandwidth=sigmaOfKDE).fit(self.dataRepresentation[:,eigVecInd].reshape(-1,1))

            logPDF = PDF_Model.score_samples(xAxis.reshape(-1,1))

            percentileValuesToShow = [representationPercentiles[x][eigVecInd] for x in range(len(representationPercentiles))]

            percentilesToShowLogPDF = PDF_Model.score_samples(np.array(percentileValuesToShow).reshape(-1,1))



            # show distribution of current component and red dots at the list of precentiles to show 

            plt.subplot(gs[0,:])

            plt.fill(xAxis, np.exp(logPDF), fc='b');

            plt.scatter(percentileValuesToShow, np.exp(percentilesToShowLogPDF), c='r',s=40);

            plt.title(str(100*self.PCAModel.explained_variance_ratio_[eigVecInd]) + '% explained');

            

            for plotCol, currPrecentile in enumerate(percentilesToShow):                

                currPrecentileRepVec             = medianRepVec.copy()

                currPrecentileRepVec[eigVecInd]  = representationPercentiles[plotCol][eigVecInd]

                

                currPrecentileImage = np.zeros(np.shape(self.objectPixels))

                currPrecentileImage[self.objectPixels]  = self.ReconstructUsingModel(currPrecentileRepVec).ravel()

                

                # show the median image with current precentile as activation of the curr image

                plt.subplot(gs[1,plotCol]);

                if showAsTraces:

                    plt.plot(currPrecentileImage); plt.title('precentile: ' + str(percentilesToShow[plotCol]) + '%')

                else:

                    plt.imshow(currPrecentileImage, cmap='gray'); plt.title('precentile: ' + str(percentilesToShow[plotCol]) + '%'); plt.axis('off')



                # find the most suitible candidates in X for current precentile

                distFromPercentile = abs(X_rep[:,eigVecInd] - representationPercentiles[plotCol][eigVecInd])

                X_inds = np.argpartition(distFromPercentile, numReadDataSamplePerPercentile)[:numReadDataSamplePerPercentile]

                for k, X_ind in enumerate(X_inds):

                    currNearestPrecentileImage = np.zeros(np.shape(self.objectPixels))

                    currNearestPrecentileImage[self.objectPixels]  = X[X_ind,:].ravel()

                    

                    plt.subplot(gs[2+k,plotCol]);

                    if showAsTraces:

                        plt.plot(currNearestPrecentileImage); plt.title('NN with closest percentile');

                    else:

                        plt.imshow(currNearestPrecentileImage, cmap='gray'); plt.title('NN with closest percentile'); plt.axis('off')

            plt.tight_layout()
matplotlib.rcParams['font.size'] = 4

matplotlib.rcParams['figure.figsize'] = (8,6)

leaf_PCAModel.ShowVarianceDirections(numDirectionsToShow=16)