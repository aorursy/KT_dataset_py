# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import torch 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
DF = pd.read_csv('/kaggle/input/train.csv')

DF[~DF.EncodedPixels.isna()].head()
DF.head(20)
DF.dtypes
DF['Class'] = DF.ImageId_ClassId.apply(lambda x: x.split('_')[1])

DF['FileName'] = DF.ImageId_ClassId.apply(lambda x: x.split('_')[0])
DF.head()
for dirname ,_ , filenames in os.walk('/kaggle/input/train_images/'):

    Data = []

    for i,filename in enumerate(filenames):

        print("Completion : {}%".format(round((i+1)/len(filenames)*100)),end='\r')

        I = plt.imread(os.path.join(dirname,filename))

        Data.append(list(I.shape)) 
D = np.array(Data)
Sizes = pd.DataFrame(D,columns=['Height','Width','Channels'])

Sizes.hist()

plt.show()

WIDTH = int(Sizes.Width.mode())

HEIGHT = int(Sizes.Height.mode())



# Thank you @robertkag

# https://www.kaggle.com/robertkag/rle-to-mask-converter

def rleToMask(rleString,height,width):

    rows,cols = height,width

    rleNumbers = [int(numstring) for numstring in rleString.split(' ')]

    rlePairs = np.array(rleNumbers).reshape(-1,2)

    img = np.zeros(rows*cols,dtype=np.uint8)

    for index,length in rlePairs:

        index -= 1

        img[index:index+length] = 1

    img = img.reshape(cols,rows)

    img = img.T

    return img

  

I = rleToMask(DF[~DF.EncodedPixels.isna()].sample().EncodedPixels.values[0],HEIGHT,WIDTH)

plt.imshow(I)

plt.show()



del I,D,Sizes,Data
! mkdir /kaggle/train_masks

! echo "These are masks as numpy arrayas" > /kaggle/train_masks/README.MD
Sample = DF[~DF.EncodedPixels.isna()].sample()

fileName = Sample.FileName.values[0]

# print(fileName)

SampleDF = DF[DF.FileName==fileName][['EncodedPixels','Class']]

SampleDF.head()



Tensor = np.zeros((HEIGHT,WIDTH,4),dtype=np.uint8)

for i,j in SampleDF.values:

    if str(i) == 'nan':

        pass

    else:

        Tensor[:,:,int(j)-1] = rleToMask(i,HEIGHT,WIDTH)
I = plt.imread(os.path.join('/kaggle/input/train_images/',fileName))
Sample = DF[~DF.EncodedPixels.isna()].sample()

fileName = Sample.FileName.values[0]

# print(fileName)

SampleDF = DF[DF.FileName==fileName][['EncodedPixels','Class']]

SampleDF.head()



Tensor = np.zeros((HEIGHT,WIDTH,4),dtype=np.uint8)

for i,j in SampleDF.values:

    if str(i) == 'nan':

        pass

    else:

        Tensor[:,:,int(j)-1] = rleToMask(i,HEIGHT,WIDTH)



Tensor = np.expand_dims(Tensor.argmax(axis=2),axis=2)

# print(Tensor.shape)

I = plt.imread(os.path.join('/kaggle/input/train_images/',fileName))

# print(np.squeeze(Tensor).shape)

for i in np.unique(Tensor):

    tempI=I.copy()

    tempI[:,:,0] = 255*np.squeeze(Tensor==i)+(1-np.squeeze(Tensor==i))*I[:,:,0]

    plt.imshow(tempI)

    plt.title('Class {}'.format(i+1))

plt.show()
def checkSegmentation(I,Tensor):

    

    for i in np.unique(Tensor):

        tempI=I.copy()

        tempI[:,:,0] = 255*np.squeeze(Tensor==i)+(1-np.squeeze(Tensor==i))*I[:,:,0]

        plt.imshow(tempI)

        plt.title('Segmentation for Class {}'.format(i+1))

        plt.show()
NoClass = ~np.prod(Tensor,axis=2,keepdims=True)

NoClass = (Tensor.sum(axis=2,keepdims=1)<1).astype(np.uint8)

plt.imshow(np.squeeze(NoClass))
TotalTensor = np.dstack((Tensor,NoClass))
TotalTensor.shape
def getMaskTensor(name):

    SampleDF = DF[DF.FileName==name][['EncodedPixels','Class']]

    SampleDF.head()



    Tensor = np.zeros((HEIGHT,WIDTH,4),dtype=np.uint8)

    for i,j in SampleDF.values:

        if str(i) == 'nan':

            pass

        else:

            Tensor[:,:,int(j)-1] = rleToMask(i,HEIGHT,WIDTH)

    NoClass = (Tensor.sum(axis=2,keepdims=1)<1).astype(np.uint8)

    TotalTensor = np.dstack((Tensor,NoClass))

    return np.expand_dims(TotalTensor.argmax(axis=2),axis=2)



maskTensor = getMaskTensor(fileName)
for i in np.unique(maskTensor):

    plt.imshow(maskTensor[:,:,0]==i)

    plt.title('Class {}'.format(i))

    plt.show()
MaskTensor = getMaskTensor(fileName)

np.save('/kaggle/working/train_masks/'+fileName,MaskTensor)

TheMask = np.load('/kaggle/working/train_masks/'+fileName+'.npy')

TheMask.shape
for idx,filename in enumerate([_,filenames,_ in os.walk('/kaggle/input/train_images/')][1]):

    print("{} Files done".format(idx),end='\r')

    MaskTensor = getMaskTensor(filename)

    np.save('/kaggle/train_masks/'+filename,MaskTensor)
def DataLoader(DirectoryInputs='/kaggle/input/train_images/',

               DirectoryOutputs='/kaggle/train_masks/',sampleSize=3):

    filenames = np.array(list(os.walk(DirectoryInputs))[0][2])

    sampleIDX = np.random.randint(0,len(filenames),size=sampleSize,dtype=np.uint32)

#     print(filenames)

#     print(type(sampleIDX))

    InputTensors = []

    OutputTensors = []

    for i in filenames[[int(x) for x in sampleIDX]]:

        InputTensors.append(plt.imread(os.path.join(DirectoryInputs,i)))

        OutputTensors.append(np.load(os.path.join(DirectoryOutputs,i+'.npy')))

        

    Input = np.stack(InputTensors,axis=0)

    Output = np.stack(OutputTensors,axis=0)



    return Input,Output



Input,Output = DataLoader(sampleSize=5)



for im in range(Input.shape[0]):

    checkSegmentation(Input[im],Output[im])
def torchDataLoader(DirectoryInputs='/kaggle/input/train_images/',

                    DirectoryOutputs='/kaggle/train_masks/',sampleSize=5):

    Input,Output = DataLoader(DirectoryInputs,DirectoryOutputs,sampleSize)

    

    InputTorchTensor = torch.cuda.FloatTensor(Input)

    OutputTorchTensor = torch.cuda.LongTensor(Output)

    InputTorchTensor = InputTorchTensor.transpose(1,3).transpose(2,3)

    OutputTorchTensor = OutputTorchTensor.transpose(1,3).transpose(2,3)

    return InputTorchTensor,OutputTorchTensor
X,Y = torchDataLoader()
X.shape
Y.shape
del X, Y

torch.cuda.empty_cache()
class tinyUNet(torch.nn.Module):

    def __init__(self,in_channels=3,

                 filters1=64,filters2 = 128,filters3=256,

                 out_classes=5,

                 filter_size=3,Pools=4):

        super(tinyUNet,self).__init__()

        self.Conv1 = torch.nn.Sequential(

            torch.nn.BatchNorm2d(in_channels),

            torch.nn.Conv2d(

                in_channels=in_channels,

                out_channels=filters1,

                kernel_size = filter_size,

                padding=(filter_size-1)//2,

                stride=1),

            torch.nn.ReLU(),

            torch.nn.Conv2d(

                in_channels=filters1,

                out_channels=filters1,

                kernel_size = filter_size,

                padding=(filter_size-1)//2,

                stride=1),

            torch.nn.ReLU())

        self.Down1 = torch.nn.MaxPool2d(Pools)

        self.Conv2 = torch.nn.Sequential(

            torch.nn.BatchNorm2d(filters1),

            torch.nn.Conv2d(

                in_channels=filters1,

                out_channels=filters2,

                kernel_size = filter_size,

                padding=(filter_size-1)//2,

                stride=1),

            torch.nn.ReLU(),

            torch.nn.Conv2d(

                in_channels=filters2,

                out_channels=filters2,

                kernel_size = filter_size,

                padding=(filter_size-1)//2,

                stride=1),

            torch.nn.ReLU())

        self.Down2 = torch.nn.MaxPool2d(Pools)

        self.Conv3 = torch.nn.Sequential(

            torch.nn.BatchNorm2d(filters2),

            torch.nn.Conv2d(

                in_channels=filters2,

                out_channels=filters3,

                kernel_size = filter_size,

                padding=(filter_size-1)//2,

                stride=1),

            torch.nn.ReLU(),

            torch.nn.Conv2d(

                in_channels=filters3,

                out_channels=filters3,

                kernel_size = filter_size,

                padding=(filter_size-1)//2,

                stride=1),

            torch.nn.ReLU())

        self.Up1 = torch.nn.ConvTranspose2d(

            in_channels=filters3,

            out_channels=filters2,

            kernel_size=Pools,

            stride=Pools,

            padding=0)

        self.Conv4 = torch.nn.Sequential(

            torch.nn.BatchNorm2d(filters2+filters2),

            torch.nn.Conv2d(

                in_channels=filters2+filters2,

                out_channels=filters2,

                kernel_size = filter_size,

                padding=(filter_size-1)//2,

                stride=1),

            torch.nn.ReLU(),

            torch.nn.Conv2d(

                in_channels=filters2,

                out_channels=filters2,

                kernel_size = filter_size,

                padding=(filter_size-1)//2,

                stride=1),

            torch.nn.ReLU())

        self.Up2 = torch.nn.ConvTranspose2d(

            in_channels=filters2,

            out_channels=filters1,

            kernel_size=Pools,

            stride=Pools,

            padding=0)

        self.Conv5 = torch.nn.Sequential(

            torch.nn.BatchNorm2d(filters1+filters1),

            torch.nn.Conv2d(

                in_channels=filters1+filters1,

                out_channels=filters1,

                kernel_size = filter_size,

                padding=(filter_size-1)//2,

                stride=1),

            torch.nn.ReLU(),

            torch.nn.Conv2d(

                in_channels=filters1,

                out_channels=filters1,

                kernel_size = filter_size,

                padding=(filter_size-1)//2,

                stride=1),

            torch.nn.ReLU())

        self.Out = torch.nn.Conv2d(in_channels=filters1,

                                   out_channels=out_classes,

                                   kernel_size=1,

                                   padding=0,

                                   stride=1)

        

        self.Final = torch.nn.Softmax(dim=1)

        

    def imScaler(self,X):

        self.ScaledInput = 2 * (X/255) - 1

        return self.ScaledInput

    

    def forward(self,Input):

        self.Conv1Out = self.Conv1(self.imScaler(Input))

        self.Down1Out = self.Down1(self.Conv1Out)

        self.Conv2Out = self.Conv2(self.Down1Out)

        self.Down2Out = self.Down2(self.Conv2Out)

        self.Conv3Out = self.Conv3(self.Down2Out)

        self.Up1Out = self.Up1(self.Conv3Out)

        self.Conv4Out = self.Conv4(torch.cat((self.Conv2Out,self.Up1Out),dim=1))

        self.Up2Out = self.Up2(self.Conv4Out)

        self.Conv5Out = self.Conv5(torch.cat((self.Conv1Out,self.Up2Out),dim=1))

        self.Logit = self.Out(self.Conv5Out)

        return self.Logit

  

    def predict(self,Input):

        self.Logit = self.forward(Input)

        self.Output = self.Final(self.Logit)

        return self.Output
theNet = tinyUNet().to('cuda')

torch.cuda.empty_cache()

LR = 1e-3

lossFunc = torch.nn.CrossEntropyLoss()

optim = torch.optim.Adam(theNet.parameters(),lr=LR)

batchSize=7

epochs = int(2e4)

Losses = torch.empty(epochs)

for i in range(epochs):

    X,Y = torchDataLoader(sampleSize=batchSize)

    

    logit = theNet.forward(X)

    print('Training Completion: {}% Epoch: {}'.

          format(round((i+1)/epochs*100),i+1),end='\r')

    loss = lossFunc(logit,torch.squeeze(Y))

    Losses[i] = loss.detach().cpu()

    loss.backward()

    optim.step()

    optim.zero_grad()

    del loss ,logit

numpyLosses = Losses.detach().cpu().numpy()

plt.plot(numpyLosses)



del Losses, optim, lossFunc

torch.cuda.empty_cache()

print("\nCUDA memory allocated: ")

print(torch.cuda.memory_allocated()/(1024**3))
X,Y = torchDataLoader(sampleSize=2)



predY = torch.argmax(theNet.forward(X),dim=1,keepdim=True)



X = X.transpose(3,1).transpose(1,2).detach().cpu().numpy().astype(np.uint8)

predY = predY.transpose(3,1).transpose(1,2).detach().cpu().numpy().astype(np.uint8)

Y = Y.transpose(3,1).transpose(1,2).detach().cpu().numpy().astype(np.uint8)



for im in range(X.shape[0]):

    print("Image {} True Segmentation".format(im+1))

    checkSegmentation(X[im],Y[im])

    print("Image {} Predicted Segmentation".format(im+1))

    checkSegmentation(X[im],predY[im])

    

del X,Y,predY
torch.save(theNet.state_dict(),'miniUNet')
torch.save(theNet.state_dict(),'/miniUNet')
# Thanks @rakhlin for sharing!

# https://www.kaggle.com/rakhlin/fast-run-length-encoding-python



def rle_encoding(x):

    '''

    x: numpy array of shape (height, width), 1 - mask, 0 - background

    Returns run length as list

    '''

    dots = np.where(x.T.flatten()==1)[0] # .T sets Fortran order down-then-right

    run_lengths = []

    prev = -2

    for b in dots:

        if (b>prev+1): run_lengths.extend((b+1, 0))

        run_lengths[-1] += 1

        prev = b

    return run_lengths


