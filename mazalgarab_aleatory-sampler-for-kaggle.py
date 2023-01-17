"""
Load Input image files of the competition

Please add data manually coresponding to the following competition:

ALASKA2 Image Steganalysis

"""

# Input directories

input_Cover = '/kaggle/input/alaska2-image-steganalysis/Cover/'
input_JMiPOD = '/kaggle/input/alaska2-image-steganalysis/JMiPOD/'
input_JUNIWARD = '/kaggle/input/alaska2-image-steganalysis/JUNIWARD/'
input_UERD = '/kaggle/input/alaska2-image-steganalysis/UERD/'
input_Test = '/kaggle/input/alaska2-image-steganalysis/Test/'
directoryListInput = [input_Cover]+[input_JMiPOD]+[input_JUNIWARD]+[input_UERD]+[input_Test]

# Number of Input Images and set of sample number:

def SampleFilesToCompete(directoryListInput, trainSample, testSample):

    import os
    
    print('\n*******************************************************************************************************************')
    print('Sampling Method: Aleatory extraction')
    print('*******************************************************************************************************************')
    
    input_Test = '/kaggle/input/alaska2-image-steganalysis/Test/'
    KaggleInputList = []
    
    for i in directoryListInput:
        
        listy = os.listdir(i)
        
        if i == input_Test:
            listySample = listy[0:testSample]
        else:
            listySample = listy[0:trainSample]
        
        print('\nSourced Directory of Kaggle: ',i)
        print('\nExpected Number of Files to be extracted: \n',len(listySample), ' from ', len(listy), ' available instances')
        
        KaggleInputList = KaggleInputList + [listySample]
        
    
    print('\n*******************************************************************************************************************')
    print('Sample Features:')
    print('*******************************************************************************************************************')
    print('\nTest-sample instances number: ', testSample)
    print('\nTrain-sample instances number: ', trainSample)
    print('\nDirectory Input List to compete sourced from Kaggle',directoryListInput)
    print('\nSample Files per directory',KaggleInputList)
    
    #pairSample = [directoryListInput, KaggleInputList]
    
    return directoryListInput, KaggleInputList

# Input training and testing instances number for sampling.

trainSample = input('\nPlease insert a sample-number of instances for training (0-7500) non-test-images:')
trainSample = int(trainSample)
testSample = input('\nPlease insert a sample-number of instances for testing (0-5000) test-images:')
testSample = int(testSample)

#trainSample = 10
#testSample = 10
directoryListInput, KaggleInputList = SampleFilesToCompete(directoryListInput,trainSample,testSample)
              

"""
Create directories to prototype with input image files of the competition data
"""

# Required directories

path_generic1 = '/kaggle/working/input/'
path_generic2 = '/kaggle/working/input/inputtoprototype-alaska2imagesteganalysis/'

path_Cover = '/kaggle/working/input/inputtoprototype-alaska2imagesteganalysis/Cover/'
path_JMiPOD = '/kaggle/working/input/inputtoprototype-alaska2imagesteganalysis/JMiPOD/'
path_JUNIWARD = '/kaggle/working/input/inputtoprototype-alaska2imagesteganalysis/JUNIWARD/'
path_UERD = '/kaggle/working/input/inputtoprototype-alaska2imagesteganalysis/UERD/'
path_Test = '/kaggle/working/input/inputtoprototype-alaska2imagesteganalysis/Test/'

directoryList = [path_generic1]+[path_generic2]+[path_Cover]+[path_JMiPOD]+[path_JUNIWARD]+[path_UERD]+[path_Test]
directoryListG = [path_Cover]+[path_JMiPOD]+[path_JUNIWARD]+[path_UERD]+[path_Test]

print('\n*******************************************************************************************************************')
print('Required directories to create')
print('*******************************************************************************************************************')
print('\nDirectories to create:','\n', directoryList)

# Function directoryCreation(directoryList)

def directoryCreation(directoryList):
    
    import os

    print('\n*******************************************************************************************************************')
    print('Created or existing directories')
    print('*******************************************************************************************************************')
    
    for i in directoryList:
    
        try:
            os.mkdir(i)
        except FileExistsError: # directory already exists
            pass
        
        print('\n Created or existing directory: ',i)
    
    print('\nDirectories are created and available to upload data...')
    
    return

# Run

directoryCreation(directoryList)
directoryList.remove(path_generic1)
directoryList.remove(path_generic2)

"""
Load Input image files to prototype
"""

# Function

def SampleFilesExtraction(directoryListInput, directoryListOutput, fileList):

    # Set working directory
    
    # Copy task
    
    # Based partially in Copy all JPG file in a directory to another directory in Python? (n.d.).
    
    print('\n*******************************************************************************************************************')
    print('Kaggle images extraction for sampling')
    print('*******************************************************************************************************************')
    print('\n')
    
    import shutil, os
    
    for i in range (0,len(directoryListInput)):
    
        print('Origin Directory: ', directoryListInput[i])
        print('Destination Directory: ', directoryListOutput[i])
    
        for f in fileList[i]:

            try: 
                # Set origin directory
                origin = directoryListInput[i]
                os.chdir(origin)

                # Copy task
                # Based partially in Copy all JPG file in a directory to another directory in Python? (n.d.).
                destination = directoryListOutput[i]
                shutil.copy(f, destination)
            
            except FileExistsError: # file already exists
                os.removedirs(f)
                
                # Set origin directory
                origin = directoryListInput[i]
                os.chdir(origin)

                # Copy task
                # Based partially in Copy all JPG file in a directory to another directory in Python? (n.d.).
                destination = directoryListOutput[i]
                shutil.copy(f, destination)
                
                pass
            
            print('  Extracted File: ', f)
        print('\n')
    
    return

# Run
directoryListOutput = directoryList
directoryList, fileList = directoryListInput, KaggleInputList
SampleFilesExtraction(directoryListInput, directoryListOutput, fileList)


"""
Reference:

Copy all JPG file in a directory to another directory in Python? (n.d.). Retrieved from
    https://stackoverflow.com/questions/11903037/copy-all-jpg-file-in-a-directory-to-another-directory-in-python

"""
