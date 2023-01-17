%reload_ext autoreload

%autoreload 2

%matplotlib inline

import numpy as np 

import pandas as pd

from fastai.metrics import accuracy

from fastai.vision import *

import os

print(os.listdir("../input/recursion-cellular-image-classification"))
path = '../input/recursion-cellular-image-classification'

dftrain = pd.read_csv(path+'/train.csv')

dftrain = dftrain[['id_code','sirna']]



i = 0

dic = {}

dictest = {}

for fold1 in os.listdir(path+'/train'):

    for fold2 in os.listdir(path+'/train/'+fold1):

        for image in os.listdir(path+'/train/'+fold1+'/'+fold2):

            dic[str(fold1)+'_'+fold2[5:]+'_'+image[0:3]] = str(fold1)+'/'+fold2+'/'+image

df = pd.DataFrame(list(dic.items()), columns=['id_code','Item'])

dftraindf = pd.merge(df, dftrain)

trainData = dftraindf[['Item','sirna']]

#df = df.astype({"a": int, "b": complex})

#trainData = trainData.astype({'sirna': str})

trainData.to_csv(r'../working/trainData.csv', index = None, header=True);

#print(pd. read_csv ('../working/trainData.csv'))

path = '../input/recursion-cellular-image-classification'

dftest = pd.read_csv(path+'/test.csv')

dftest = dftest['id_code']

i = 0

dictest = {}

for fold1 in os.listdir(path+'/test'):

    for fold2 in os.listdir(path+'/test/'+fold1):

        for image in os.listdir(path+'/test/'+fold1+'/'+fold2):

            dictest[str(fold1)+'_'+fold2[5:]+'_'+image[0:3]] = str(fold1)+'/'+fold2+'/'+image

df = pd.DataFrame(list(dictest.items()), columns=['id_code','foldPath'])

"""dftraindf = pd.merge(df, dftest)

testData = dftraindf[['Item']]"""

df.to_csv(r'../working/testData.csv', index = None, header=True);

#print(pd. read_csv ('../working/testData.csv'))

#print(df.head())
tfms = get_transforms()

#df = pd.read_csv(path/'labels.csv', header='infer')

#path = Path(path)

# Set the parameters and create the data for the model

np.random.seed(42) #makes sure you get same results each time you run the code

src = (ImageList.from_csv('../', 'working/trainData.csv', folder='input/recursion-cellular-image-classification/train')

       .split_by_rand_pct(0.2)

       .label_from_df(label_delim=' '))

tfms = get_transforms()

data = (src.transform(tfms, size=128)

        .databunch().normalize(imagenet_stats))







#data = ImageDataBunch.from_df('../','working/trainData.csv',folder=path+'/train', ds_tfms=tfms, size=128)
# Since kaggle doesn't allow write on the iput directory, we create a new directory outside it where

# we can freely work and make it the path

#path = Path("../working")

#path
#img = open_image(path+'/train/HEPG2-04/Plate1/O23_s2_w4.png')

#img
#trainData

#trainData.loc[trainData['sirna'] == 810]

#len(data.classes)
#print(data.classes)

print((len(data.train_ds),len(data.valid_ds)))

data.show_batch(rows=3, figsize=(7,8)) #View portion of dataset
def accuracy(input:Tensor, targs:Tensor)->Rank0Tensor:

    "Computes accuracy with `targs` when `input` is bs * n_classes."

    n = targs.shape[0]

    input = input.argmax(dim=-1).view(n,-1)

    targs = targs.view(n,-1)

    return (input==targs.long()).float().mean()



"""From https://www.kaggle.com/leighplt/densenet121-pytorch"""

def accuracy1(output, target, topk=(1,)):

    """Computes the accuracy over the k top predictions for the specified values of k"""

    with torch.no_grad():

        maxk = max(topk)

        batch_size = target.size(0)



        _, pred = output.topk(maxk, 1, True, True)

        pred = pred.t()

        correct = pred.eq(target.view(1, -1).expand_as(pred))



        res = []

        for k in topk:

            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)

            res.append(correct_k.mul_(100.0 / batch_size).item())

        return torch.Tensor(np.array(res))
#Set the metrics. Use F-score

#acc_02 = partial(accuracy_thresh, thresh=0.2)

#f_score = partial(fbeta, thresh=0.2)

#Use CNN (Convolutional Neural Network) and pretrained model (resnet50)  to train

#learn = cnn_learner(data, models.resnet50, metrics=[acc_02,f_score])

learn = cnn_learner(data, models.resnet50, metrics=[accuracy])
#Find and plot learning rate

learn.lr_find()

learn.recorder.plot()
#set learning rate

lr = 0.05
#Fit the model

learn.fit_one_cycle(5,slice(lr))#5
# Save it

learn.save('stage-1-rn50')
####learn.load('stage-1-rn50');


# Unfreeze the model, that is, traing afresh without the pretrained model

learn.unfreeze()

# Find and plot the learning rate

learn.lr_find()

learn.recorder.plot()
# Fit the model

learn.fit_one_cycle(5, slice(1e-5, 1e-4))
# Save this latest trained model

learn.save('stage-2-rn50')
# Create a new dataset with batch size = 256

data = (src.transform(tfms, size=256)

        .databunch().normalize(imagenet_stats))

# Set the learners data as data

learn.data = data

data.train_ds[0][0].shape
# Freeze and find learning rate

learn.freeze()

learn.lr_find()

learn.recorder.plot()
# Fit and save the model

lr=1e-2/2

learn.fit_one_cycle(5, slice(lr))

learn.save('stage-1-256-rn50')
# Freeze and find learning rate

learn.unfreeze()

learn.lr_find()

learn.recorder.plot()
# Fit and save the model

learn.fit_one_cycle(10, slice(2e-5, 4e-5))

learn.recorder.plot_losses()

learn.save('stage-2-256-rn50')
learn.unfreeze()

learn.fit_one_cycle(10, slice(1e-5, lr/5))

learn.save('stage-3-256-rn50')
learn.export()
test = ImageList.from_csv('../', 'working/testData.csv', cols='foldPath', folder='input/recursion-cellular-image-classification/test')

learn = load_learner('../', test=test)
# Find the prediction

preds,_ = learn.get_preds(ds_type=DatasetType.Test)

labelled_preds = [learn.data.classes[(pred).tolist().index(max((pred).tolist()))] for pred in preds]

#Althernatively, you can replace line two with these two lines of code below

#labels = np.argmax(preds, 1)

#labelled_preds = [data.classes[int(x)] for x in labels]

#print(labelled_preds)
lsttest = []

for item in learn.data.test_ds.items:

    lst = item.split('/')[-3:]

    lsttest.append(str(lst[0])+'_'+lst[1][5:]+'_'+lst[-1].split('_')[0])

df = pd.DataFrame(lsttest, columns=['id_code'])

#print(df.head())
path = '../input/recursion-cellular-image-classification'

dftestcsv = pd.read_csv(path+'/test.csv')



tes = OrderedDict([('id_code',lsttest), ('sirna', labelled_preds)] )

df = pd.DataFrame.from_dict(tes)



dftestcsv = pd.DataFrame(list(dftestcsv['id_code']), columns=['id_code'])

dftestdfcsv = pd.merge(dftestcsv, df)

dftestdfcsv.to_csv('../working/submission.csv', index=False)
dftestdfcsv.tail()