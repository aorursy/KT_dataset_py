from fastai.vision import DataBunch, Learner, accuracy, top_k_accuracy, callbacks, ClassificationInterpretation, DatasetType

import pandas as pd

import os

from sklearn.model_selection import train_test_split

from torchvision import transforms

import torch

import numpy as np



from torch import nn



import matplotlib.pyplot as plt



from functools import partial



plt.style.use('seaborn')
torch.manual_seed(0)

torch.backends.cudnn.deterministic = True

torch.backends.cudnn.benchmark = False

np.random.seed(0)
train_data = pd.read_csv('../input/digit-recognizer/train.csv')

test_data = pd.read_csv('../input/digit-recognizer/test.csv')

print(train_data.shape, test_data.shape)

train_data.head()
y = train_data.values[:, 0]

X = train_data.values[:, 1:]
X_test = test_data.values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.shape, X_val.shape
class MNIST(torch.utils.data.Dataset):



    def __init__(self, X, y=None, transform=None):

        self.X = X.astype(np.uint8).reshape(-1, 28, 28)

        

        if y is None:

            y = torch.zeros(len(X_test))

        

        class dummy:

            def __getitem__(self, idx):

                return y[idx]            

            classes = np.unique(y).tolist()

            

        self.y = dummy()

        

        self.transform = transform

        

        self.c = len(self.y.classes)



    def __len__(self):

        return len(self.X)

        

    def __getitem__(self, idx):        

        X = self.X[idx]

        if self.transform:

            X = self.transform(X)



        return X, self.y[idx]

    
train_dataset = MNIST(X_train, y_train, transform=transforms.Compose([

                                               transforms.ToPILImage(mode='L'),

                                               transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),

                                               transforms.RandomResizedCrop(28, scale=(0.98, 1.02), ratio=(1, 1)),

                                               transforms.ToTensor(),

                                               transforms.Normalize(X_train.mean()[None], X_train.std()[None]),

                                           ]))





val_dataset = MNIST(X_val, y_val, transform=transforms.Compose([

                                               transforms.ToPILImage(mode='L'),                                           

                                               transforms.ToTensor(),

                                               transforms.Normalize(X_val.mean()[None], X_val.std()[None]),  

                                           ]))

test_dataset = MNIST(X_test, transform=transforms.Compose([

                                               transforms.ToPILImage(mode='L'), 

                                               transforms.ToTensor(),

                                               transforms.Normalize(X_test.mean()[None], X_test.std()[None]),

                                           ]))
def get_loader(dataset, batch_size, shuffle=True,):

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)



def get_Databunch(bs=128):

    return DataBunch(

            train_dl=get_loader(train_dataset, batch_size=bs),  

            valid_dl=get_loader(val_dataset, batch_size=bs),  

            test_dl=get_loader(test_dataset, batch_size=bs, shuffle=False)

        )
fig, axs = plt.subplots(10, 10, figsize=(25, 25))

db = get_Databunch(bs=128)

b = db.one_batch()

for i, ax in enumerate(axs.flatten()):

    ax.set_title(b[1][i].item())

    ax.imshow(b[0][i][0].numpy())

    ax.axis('off')
number_of_classes = 10

input_features = 784

class MNISClassifier(nn.Module):

    def __init__(self, do = 0.25):

        super(MNISClassifier, self).__init__()

        self.layer1 = nn.Sequential(

            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),

            nn.BatchNorm2d(64),

            nn.PReLU(),

            

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), 

            nn.BatchNorm2d(64),

            nn.PReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),            

            nn.Dropout(do)    

        )

        

        self.layer2 = nn.Sequential(

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),

            nn.BatchNorm2d(128),

            nn.PReLU(),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), 

            nn.BatchNorm2d(128),

            nn.PReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),            

            nn.Dropout(do)    

        )

        

        self.layer3 = nn.Sequential(

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),

            nn.BatchNorm2d(256),

            nn.PReLU(),

            nn.BatchNorm2d(256),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), 

            nn.BatchNorm2d(256),

            nn.PReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),            

            nn.Dropout(do),

            

            nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=1),

            nn.PReLU(),

            nn.BatchNorm2d(512),

            nn.MaxPool2d(kernel_size=2, stride=2), 

            nn.BatchNorm2d(512),

            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=1), 

            nn.PReLU(),

            nn.BatchNorm2d(512),

            nn.MaxPool2d(kernel_size=2, stride=2),            

            nn.Dropout(do)    

        )        

        

        self.layer4 = nn.Sequential(

            nn.Linear(512, 512),

            nn.BatchNorm1d(512),

            nn.PReLU(),

            nn.Linear(512, 256),

            nn.BatchNorm1d(256),

            nn.PReLU(),

            nn.Linear(256, number_of_classes)

        )

        

        

    def forward(self, x):

        batch_size = x.size(0)

        x = self.layer1(x.contiguous())        

        x = self.layer2(x)

        x = self.layer3(x)        

        x = x.view(batch_size, -1)

        x = self.layer4(x)        

        

        return x

label_value_counts = train_data.label.value_counts().reset_index()

label_value_counts.columns = ['Class', 'Count']

label_value_counts.set_index('Class').plot.barh()
weights = (train_data.label.value_counts().mean()/train_data.label.value_counts())



w = weights.reset_index()

w.columns = ['Class', 'Weight']

w.set_index('Class').plot.barh()
weights_tensor = torch.from_numpy(weights.values).float().cuda()

learn = Learner(    

    get_Databunch(bs=512), 

    MNISClassifier(0.2), 

    loss_func=nn.CrossEntropyLoss(weight=weights_tensor),

    metrics=accuracy

)

learn.fit(

    epochs=5, 

    lr=0.0005, 

    callbacks=callbacks.SaveModelCallback(learn, every='improvement', monitor='accuracy')

)
def top_2_acc(*args):

    return top_k_accuracy(*args, k=2)





learn = Learner(    

    get_Databunch(bs=1024),

    learn.model, 

    loss_func=nn.CrossEntropyLoss(weight=weights_tensor),

    metrics=[accuracy, top_2_acc]

)

def current_lr(out, true, learn):

    return torch.tensor(learn.recorder.lrs[-1])



def current_mom(out, true, learn):

    return torch.tensor(learn.recorder.moms[-1])



learn.metrics.append(partial(current_lr, learn=learn))

learn.metrics.append(partial(current_mom, learn=learn))

epochs=30

learn.fit_one_cycle(

    cyc_len=epochs, 

    max_lr=0.0005,

    div_factor=40,

    callbacks=callbacks.SaveModelCallback(learn, every='improvement', monitor='accuracy')

)

learn.recorder.plot_losses()
ci = ClassificationInterpretation.from_learner(learn)

ci.plot_confusion_matrix(figsize=(6,6), normalize=False)
preds, _ = learn.get_preds(DatasetType.Test)

labels = preds.cpu().numpy().argmax(1)



sub =  pd.DataFrame(data=dict(ImageId=range(1, len(labels)+1), Label=labels))

sub.head()
sub.Label.value_counts().plot.bar()
sub.to_csv('sub.csv', index=False)