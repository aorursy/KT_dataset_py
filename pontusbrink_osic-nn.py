# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



# Loss function and lots of inspiration from https://www.kaggle.com/ulrich07/osic-multiple-quantile-regression-starter#kln-163



import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch as pt

from torch.utils.data import Dataset, DataLoader, RandomSampler

import torch.optim as optim

import matplotlib.pyplot as plt

import pydicom

from sklearn.model_selection import KFold

import seaborn as sns

import pydicom

from glob import glob

import scipy.ndimage

from skimage import morphology

from skimage import measure

from skimage.filters import threshold_otsu, median

from scipy.ndimage import binary_fill_holes

from skimage.segmentation import clear_border

from scipy.stats import describe

    

trainImagesPath = "/kaggle/input/osic-pulmonary-fibrosis-progression/train/"

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

dtype = pt.float

use_cuda = pt.cuda.is_available()

device = pt.device("cuda:0" if use_cuda else "cpu")







# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

#

#inputCols = ['PercentIn','AgeIn',  'Smoke', 'Gender', 'WeekIn']

inputCols = ['PercentIn', 'AgeIn', 'WeekIn', 'base_FVC', 'min_FVC', 'SmokingStatus_Currently smokes', 'SmokingStatus_Ex-smoker', 'SmokingStatus_Never smoked', 'Sex_Male', 'Sex_Female']

#inputCols=['PercentIn']

test = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv")

train = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv")

subms = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/sample_submission.csv")

subms['Patient'] = subms.Patient_Week.apply(lambda x: x.split("_")[0])

subms['Weeks'] = subms.Patient_Week.apply(lambda x: int(x.split("_")[-1]))

train['Split'] = "train"

test["Split"] = "test"

test['PatientDir'] = test.Patient.apply(lambda x: "../input/osic-pulmonary-fibrosis-progression/test/" + x)

train['PatientDir'] = train.Patient.apply(lambda x: "../input/osic-pulmonary-fibrosis-progression/train/" + x)

subms['PatientDir'] = subms.Patient.apply(lambda x: "../input/osic-pulmonary-fibrosis-progression/train/" + x)



#Add submissions

subms = subms[['Patient', 'Weeks', 'Patient_Week']]

subms = subms.merge(test.drop("Weeks", axis=1), on="Patient")

subms["Split"] = "subm"

data = test.append([subms, train])



# Scale week input relative to first week

data['first_week'] = data.Weeks

data['first_week'] = data.groupby('Patient')['first_week'].transform('min') # This assumes all patient ids in submission set exist in test set later. Set missing to train set mean?

data['first_week'] = data.Weeks - data.first_week



# Minimum FVC and FVC on first test

data['min_FVC'] = data.groupby('Patient')['FVC'].transform('min')

data = data.sort_values(['Patient', 'Weeks'], ascending=True)

b = data[data['first_week'] == 0]

b['base_FVC'] = b.FVC

b = b[['Patient', 'base_FVC']]

b = b.drop_duplicates('Patient')

data = data.merge(b, on='Patient', how='left')

data['WeekIn'] = (data.first_week - data.first_week.min()) / (data.first_week.max() - data.first_week.min())

data = pd.concat([data, pd.get_dummies(data.SmokingStatus, prefix="SmokingStatus")], axis=1)

data = pd.concat([data, pd.get_dummies(data.Sex, prefix="Sex")], axis=1)



# Normalize

data['Smoke'] = data.SmokingStatus.replace({'Ex-smoker' : 0.5, 'Never smoked' : 0, 'Currently smokes' : 1})

data['Gender'] = data.Sex.replace({'Male' : 1, 'Female' : 0})

data['min_FVC'] = (data.min_FVC - data.min_FVC.min()) / (data.min_FVC.max() - data.min_FVC.min())

data['base_FVC'] = (data.base_FVC - data.base_FVC.min()) / (data.base_FVC.max() - data.base_FVC.min())

data['WeekIn'] = (data.first_week - data.first_week.min()) / (data.first_week.max() - data.first_week.min())

data['AgeIn'] = (data.Age - data.Age.min()) / (data.Age.max() - data.Age.min())

data['PercentIn'] = (data.Percent - data.Percent.min()) / (data.Percent.max() - data.Percent.min())



train = data.loc[data.Split == 'train'].copy()

subms = data.loc[data.Split == "subm"].copy()

test = data.loc[data.Split == "test"].copy()

data.corr()



# I/O for nn

inputs = [data.columns.get_loc(c) for c in inputCols if c in data.columns]

outputColI = data.columns.get_loc('FVC')
print(data['PatientDir'])

print((data.nunique()))
class OSICDataSet(Dataset):

    def __init__(self, data, mode = "train"):

        self.data = data

        self.mode = mode

    

    def __len__(self):

        return len(self.data)

    

    def __getitem__(self, idx):

        #image = self.data.loc[idx, ['Patient']].map(lambda filename: pydicom.dcmread(trainImagesPath + filename + "/1.dcm")).item().pixel_array.astype(np.float64)

        otherData = pt.from_numpy(self.data.iloc[idx, inputs].values.astype(np.float32))

        if self.mode == "train":

            targets = pt.from_numpy(self.data.iloc[idx, outputColI].values.astype(np.float32))

            return otherData, targets

            #return otherData, targets

        return otherData #TODO: Add image data

    

class Model(pt.nn.Module):

    def __init__(self):

        super(Model, self).__init__()

        self.start = pt.nn.Sequential(

            pt.nn.Linear(len(inputCols), 128),

            pt.nn.ReLU(),

            pt.nn.Linear(128, 256),

            pt.nn.ReLU(),

            pt.nn.Linear(256, 512),

            pt.nn.ReLU(),

            pt.nn.Linear(512, 1024),

            pt.nn.ReLU(),

            pt.nn.Linear(1024, 512),

            pt.nn.ReLU(),

            pt.nn.Linear(512, 256),

            pt.nn.ReLU(),

        )

        self.left = pt.nn.Sequential(

            pt.nn.Linear(256, 128)

        )

        self.sigmoid = pt.nn.Sigmoid()

        self.right = pt.nn.Sequential(

            pt.nn.Linear(256,128)

        )

        self.last = pt.nn.Sequential(

            pt.nn.Linear(128, 3),

        )

        self.lastRelu = pt.nn.Sequential(

            pt.nn.Linear(128, 3),

            pt.nn.ReLU()

        )

        

        

    def forward(self, x):

        h = self.start(x)

        l = self.left(h)

        r = self.right(h)

        h = l * self.sigmoid(r)

        p1 = self.last(h)

        p2 = self.lastRelu(h)

        out = p1 + pt.cumsum(p2, 1)

        return out



model = Model()



C1, C2 = pt.FloatTensor([70]), pt.FloatTensor([1000])

#=============================#

def score(y_pred, y_true):

    sigma = y_pred[:, 2] - y_pred[:, 0]

    fvc_pred = y_pred[:, 1]



    #sigma_clip = sigma + C1

    sigma_clip = pt.max(sigma, C1.expand_as(sigma))

    delta = pt.abs(y_true - fvc_pred)

    delta = pt.min(delta, C2)

    sq2 = pt.sqrt(pt.FloatTensor([2]))

    metric = (delta / sigma_clip)*sq2 + pt.log(sigma_clip* sq2)

    return pt.mean(metric)



def pinballLoss(pred, label, quant):

    err = label.unsqueeze(1) - pred

    m = pt.mean(pt.max(quant * err, (quant-1)*err))

    return m



def loss(pred, label):

    quantiles = pt.FloatTensor([0.2, 0.5, 0.8])

    return 0.8 * pinballLoss(pred, label, quantiles) + 0.2 * score(pred, label)



def earlyStop(useEarlyStopping, earlyStopping):

    if useEarlyStopping:

        return not earlyStopping

    return True
trainDataSet = OSICDataSet(train) 

optimizer = optim.Adam(model.parameters(), lr=0.075, weight_decay = 0.1, eps=0.001)

losses = []

valLosses = []

nSplits = 5

kf = KFold(n_splits=5)

c = 0

useEarlyStopping = True

earlyStopping = False

submDataSet = OSICDataSet(subms, "submission")

stopping = 0

i = 0

while i < 800 and earlyStop(useEarlyStopping, earlyStopping):

    for train_index, val_index in kf.split(trainDataSet):

        train_x, train_y = trainDataSet[train_index]

        val_x, val_y = trainDataSet[val_index]

        optimizer.zero_grad()

        pred = model(train_x)

        los = loss(pred, train_y)

        los.backward()

        if c % 10 == 0:

            pred = model(val_x)

            valLos = loss(pred, val_y)

            losses.append(los)

            valLosses.append(valLos)

            if useEarlyStopping and (len(valLosses) > 1 and valLosses[-1] < valLosses[-2]):

                stopping += 1

                if stopping > nSplits:

                    earlyStopping = True

                break

        c += 1

        i += 1

        stopping = 0

        optimizer.step()

        c+=1

    
sns.set(style="white", palette="muted", color_codes=True)



plt.plot(losses, label="Train loss")

plt.plot(valLosses, label="Val loss")

plt.legend()

plt.show()



asd = pt.from_numpy(trainDataSet.data[inputCols].values.astype(np.float32))

pred = model(asd).detach()

idxs = np.random.randint(0, len(trainDataSet), 100)

plt.plot(trainDataSet.data.iloc[idxs, outputColI].values.astype(np.float32), label="ground truth")

plt.plot(pred[idxs, 0], label="q25")

plt.plot(pred[idxs, 1], label="q50")

plt.plot(pred[idxs, 2], label="q75")

plt.legend(loc="best")

plt.show()

c = pred[:,2] - pred[:,0]

f = pred[:,1]





fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))

sns.distplot(c, color="g", kde=False, kde_kws={"shade": True}, ax=axes[0]).set_title("Predicted Confidence on train set")

sns.distplot(f, color="g", kde=False, kde_kws={"shade": True}, ax=axes[1]).set_title("Predicted FVC on train set")

plt.show()



submDataSet = OSICDataSet(subms, "submission")

inp = pt.from_numpy(submDataSet.data[inputCols].values.astype(np.float32))

out = model(inp).detach()

confidence = out[:,2] - out[:,0]

fvc = out[:,1]

fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))

sns.distplot(confidence, color="g", kde=False, kde_kws={"shade": True}, ax=axes[0]).set_title("Confidence on submission set")

sns.distplot(fvc, color="g", kde=False, kde_kws={"shade": True}, ax=axes[1]).set_title("FVC on submision set")

plt.show()



submDataSet.data['pFVC'] = fvc

submDataSet.data['cConf'] = confidence



submission = pd.DataFrame({'Patient_Week' : submDataSet.data.Patient_Week, 'FVC' : fvc, 'Confidence' : confidence})

otest = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')

for i in range(len(otest)):

    submission.loc[submission['Patient_Week']==otest.Patient[i]+'_'+str(otest.Weeks[i]), 'FVC'] = otest.FVC[i]

    submission.loc[submission['Patient_Week']==otest.Patient[i]+'_'+str(otest.Weeks[i]), 'Confidence'] = 0.1



submission.to_csv("submission.csv", index=False)

null_columns=submission.columns[submission.isnull().any()]

#print(submission[submission.isnull().any(axis=1)][null_columns].head())

print(submission.head())

print(len(submission))