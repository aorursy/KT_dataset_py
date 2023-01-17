# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    print(dirname)



# Any results you write to the current directory are saved as output.
import copy

import time

import torch

import pandas as pd

import torch.nn as nn

import matplotlib.pyplot as plt

import torch.nn.functional as F



from PIL import Image

from tqdm.notebook import tqdm

from sklearn.metrics import f1_score

from torchvision import transforms, models

from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split





train_csv_path = '/kaggle/input/plant-pathology-2020-fgvc7/train.csv'

test_csv_path = '/kaggle/input/plant-pathology-2020-fgvc7/test.csv'

images_dir = '/kaggle/input/plant-pathology-2020-fgvc7/images/'

submission_df_path = '/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv'
model_name = 'resnext'

num_classes = 4

batch_size = 16

num_epochs = 2 #4

# set val_split = 0 for no validation

val_split = 0.0 #0.0

# if dev_mode = True loads only a few samples

dev_mode = False

num_dev_samples = 0

# feature_extract = False   ==> fine-tune the whole model 

# feature_extract = True    ==> only update the reshaped layer parameters

feature_extract = False

pre_trained = True

num_cv_folds = 5
'''

Define Transforms

Define Dataset Class

'''



class ppDataset(Dataset):

    def __init__(self, df, image_dir, return_labels=False, transforms=None):

        self.df = df

        self.image_dir = image_dir

        self.transforms = transforms

        self.return_labels = return_labels

        self.label_map = {'healthy':0, 'multiple_diseases':1, 'rust':2, 'scab':3}

        self.label_map_reverse = {v:k for k,v in self.label_map.items()}

        

    def __len__(self):

        return self.df.__len__()

    

    def __getitem__(self, idx):

        image_path = self.image_dir + self.df.loc[idx, 'image_id'] + '.jpg'

        image = Image.open(image_path).convert('RGB')

        

        if self.transforms:

            image = self.transforms(image)



        if self.return_labels:

            # label = torch.tensor(self.df.loc[idx, ['healthy', 'multiple_diseases', 'rust', 'scab']]).unsqueeze(1)

            label = torch.tensor(self.df.loc[idx, ['healthy', 'multiple_diseases', 'rust', 'scab']]).unsqueeze(-1)

            return image, label, self.label_map_reverse[label.squeeze(1).numpy().argmax()]

        else:

            return image





data_transforms = {

    'train': transforms.Compose([

        transforms.Resize((224, 224)),

        transforms.Pad(50, padding_mode='reflect'),        

        transforms.RandomHorizontalFlip(p=0.5),

        transforms.RandomVerticalFlip(p=0.5),

        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.05, hue=0.05),

        transforms.RandomAffine(degrees=[0,45]),

        transforms.CenterCrop(224),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        ]),

    'val': transforms.Compose([

        transforms.Resize((224, 244)),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),

        ]),

    'test': transforms.Compose([

        transforms.Resize((224, 244)),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),

        ])

}



def train_model(model, datasets_dict, criterion, optimizer, batch_size, num_epochs = 25, lr_scheduler=None):

    

    since = time.time()

    

    device = get_device()

    if device.type != 'cpu':

        model.cuda()

    model.to(device)

    

    do_validation = (datasets_dict.get('val') != None)

    

    train_acc_hist = []

    train_loss_hist = []



    if do_validation:

        val_acc_history = []

        val_loss_history = []

        val_f1_history = []

        val_dataloader = DataLoader(datasets_dict['val'], batch_size=batch_size, shuffle=True)

        print('Validating on {} samples.'.format(datasets_dict['val'].__len__()))

        best_model_wts = copy.deepcopy(model.state_dict())

        best_acc = 0.0





    for epoch in range(num_epochs):



        tr_dataloader = DataLoader(datasets_dict['train'], batch_size=batch_size, shuffle=True, num_workers=4)



        if do_validation:

            val_preds = []

            val_labels = []

            phases = ['train', 'val']

            dataloaders = {'train' : tr_dataloader, 'val' : val_dataloader}

        else:

            phases = ['train']

            dataloaders = {'train' : tr_dataloader}





        print('Epoch {}/{}'.format(epoch+1, num_epochs))

        print('-'*10)



        for phase in phases:

            if phase == 'train':    model.train()

            elif phase == 'val':    model.eval()



            running_loss = 0.0

            running_corrects = 0



            for inputs, labels, _ in dataloaders[phase]:



                inputs = inputs.to(device)

                labels = labels.to(device)

                labels = labels.squeeze(-1)

                

                optimizer.zero_grad()

                model.zero_grad()



                with torch.set_grad_enabled(phase == 'train'):



                    outputs = model(inputs)

                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)



                    if phase == 'train':

                        loss.backward()

                        optimizer.step()

                        if lr_scheduler:

                            lr_scheduler.step()

                

                running_loss += loss.item() * inputs.size(0)

                running_corrects += torch.sum(labels.argmax(dim=1) == outputs.argmax(dim=1))

                if do_validation:

                    val_preds.append(preds.cpu())

                    val_labels.append(labels.argmax(dim=1).cpu())



            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)



            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            

            if phase == 'val' and epoch_acc > best_acc:

                best_acc = epoch_acc

                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'val':

                val_acc_history.append(epoch_acc)

                val_loss_history.append(epoch_loss)

            if phase == 'train':

                train_acc_hist.append(epoch_acc)

                train_loss_hist.append(epoch_loss)



        if do_validation:

            val_f1 = get_f1(val_preds, val_labels)

            print("val F1 : ", val_f1)

            val_f1_history.append(val_f1)



        print()

    

    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    if do_validation:

        print('Best val Acc: {:4f}'.format(best_acc))

        model.load_state_dict(best_model_wts)

        metrics = {'val_acc_hist':val_acc_history, 'val_f1_hist':val_f1_history, 'val_loss_hist':val_loss_history, 'train_acc_hist':train_acc_hist, 'train_loss_hist':train_loss_hist}

    else:

        metrics = {'train_acc_hist':train_acc_hist, 'train_loss_hist':train_loss_hist}

    

    return model, metrics





class DenseCrossEntropy(nn.Module):

    def __init__(self):

        super(DenseCrossEntropy, self).__init__()        

    def forward(self, logits, labels):

        logits = logits.float()

        labels = labels.float()

        logprobs = F.log_softmax(logits, dim=-1)

        loss = -labels * logprobs

        loss = loss.sum(-1)

        return loss.mean()





def set_parameters_requires_grad(model, feature_extracting):

    if feature_extracting:

        for param in model.parameters():

            param.requires_grad = False



def get_device():

    if torch.cuda.is_available():

        device = torch.device("cuda")

        print('There are %d GPU(s) available.' % torch.cuda.device_count())

        print('We will use the GPU:', torch.cuda.get_device_name(0))

    else:

        print('No GPU available, using the CPU instead.')

        device = torch.device("cpu")

    return device

            

def initialize_model(model_name, num_classes, features_extract, use_pretrained):

    model_ft = None

    input_size = 0

    valid_model_names = ['resnet101', 'resnet152', 'resnet50', 'resnext']

    

    if model_name not in valid_model_names:

        print('Invalid model name, exiting. . .') 

        exit();

    elif model_name == 'resnext':

        model_ft = models.resnext101_32x8d(pretrained=use_pretrained)

    elif model_name == 'resnet152':

        model_ft = models.resnet152(pretrained=use_pretrained)

    elif model_name == 'resnet50':

        model_ft = models.resnet50(pretrained=use_pretrained)

    elif model_name == 'resnet101':

        model_ft = models.resnet101(pretrained=use_pretrained)



    set_parameters_requires_grad(model_ft, feature_extract)

    num_ftrs = model_ft.fc.in_features

    model_ft.fc = nn.Linear(num_ftrs, num_classes)

    input_size = 224



    return model_ft, input_size





def get_f1(val_preds, val_labels):

    VP = []

    VL = []

    for l in [list(x.numpy()) for x in val_preds]: VP.extend(l)

    for l in [list(x.numpy()) for x in val_labels]: VL.extend(l)

    return f1_score(VL, VP, average='weighted')





def get_params_to_update(model, feature_extract):

    if feature_extract:

        params_to_update = []

        for name,param in model.named_parameters():

            if param.requires_grad == True:

                params_to_update.append(param)

    else:

        params_to_update = model.parameters()    

    return params_to_update





from sklearn.model_selection import KFold





def train_k_fold(num_cv_folds, dataframe_path, batch_size_kf = batch_size, num_epochs_kf = num_epochs):

    

    # read train dataframe

    criterion_kf = DenseCrossEntropy()

    dataframe = pd.read_csv(dataframe_path)

    

    if dev_mode:

        dataframe = dataframe.sample(num_dev_samples)

    

    dataframe.reset_index(drop=True, inplace=True)

    

    kf = KFold(n_splits = num_cv_folds, shuffle=True)

    hist_folds = []

    for fold_idx,(tr_idx, val_idx) in enumerate(kf.split(dataframe)):

        print('Fold {} of {}. . .'.format(fold_idx+1, num_cv_folds ))

        

        tr_df_kf = dataframe.loc[tr_idx]

        val_df_kf = dataframe.loc[val_idx]        

        

        tr_df_kf.reset_index(drop=True, inplace=True)

        val_df_kf.reset_index(drop=True, inplace=True)   

        

        tr_dataset_kf = ppDataset(tr_df_kf, images_dir, return_labels = True, transforms = data_transforms['train'])

        val_dataset_kf = ppDataset(val_df_kf, images_dir, return_labels = True, transforms = data_transforms['val'])

        datasets_dict_kf = {'train' : tr_dataset_kf, 'val' : val_dataset_kf }

        

        model_kf, _ = initialize_model(model_name, num_classes, feature_extract, use_pretrained=pre_trained)

        optimizer_kf = torch.optim.AdamW(get_params_to_update(model_kf, feature_extract), lr = 2e-5, eps = 1e-8 )

    

        model_kf, hist_kf = train_model(model_kf, datasets_dict_kf, criterion_kf,\

                                       optimizer_kf, batch_size_kf, num_epochs=num_epochs_kf, lr_scheduler=None)

        hist_folds.append(hist_kf)

        

    return hist_folds

        

    



if num_cv_folds > 0:

    hist_folds = train_k_fold(num_cv_folds, train_csv_path)

    print('{} fold validation accuracy: {}'.format(num_cv_folds, float(sum([x['val_acc_hist'][-1] for x in hist_folds])) / len(hist_folds)))

model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=pre_trained)





params_to_update = get_params_to_update(model_ft, feature_extract)



# optimizer_ft = torch.optim.SGD(params_to_update, lr=0.001, momentum=0.9)

optimizer_ft = torch.optim.AdamW(params_to_update, lr = 2e-5, eps = 1e-8 )



# lr_scheduler    =   torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size = 1, gamma = 0.6)

lr_scheduler = None





criterion = DenseCrossEntropy()
'''

RUN VALIDATION ON val_split amount of data

If val_split = 0, return a model trained on all data

'''



# read dataFrames

if dev_mode:

    # tr_df = pd.read_csv(train_csv_path, nrows=num_dev_samples)

    tr_df = pd.read_csv(train_csv_path).sample(num_dev_samples)

    tr_df.reset_index(drop=True, inplace=True)

else:

    tr_df = pd.read_csv(train_csv_path)





if val_split > 0:

    tr_df, val_df = train_test_split(tr_df, test_size = val_split)

    val_df = val_df.reset_index(drop=True)

    tr_df = tr_df.reset_index(drop=True)

    



te_df = pd.read_csv(test_csv_path)







# create Dataset objects

tr_dataset = ppDataset(tr_df, images_dir, return_labels = True, transforms = data_transforms['train'])



te_dataset = ppDataset(te_df, images_dir, return_labels = False, transforms = data_transforms['test'])





if val_split > 0:

    val_dataset = ppDataset(val_df, images_dir, return_labels = True, transforms = data_transforms['val'])

    datasets_dict = {'train' : tr_dataset, 'test' : te_dataset, 'val' : val_dataset }

else:

    datasets_dict = {'train' : tr_dataset, 'test' : te_dataset,}





# run the training loop

model_ft, hist = train_model(model_ft, datasets_dict, criterion, optimizer_ft, batch_size, num_epochs=num_epochs)

'''

PLOT train/val accuracy/loss

'''

if val_split > 0:

    plt.grid()

    plt.xlabel('epochs')

    num_epochs = hist['train_acc_hist'].__len__()

    plt.plot(range(1,num_epochs+1), hist['train_acc_hist'])

    # plt.plot(range(1,num_epochs+1), hist['train_loss_hist'])

    if hist.get('val_acc_hist'):

        plt.plot(range(1,num_epochs+1), hist['val_acc_hist'])

        plt.plot(range(1,num_epochs+1), hist['val_f1_hist'])

        plt.legend(['train acc', 'val acc', 'val F1'])

    else:

        plt.legend(['train acc'])

    plt.show()



    plt.grid()

    plt.xlabel('epochs')

    num_epochs = hist['train_loss_hist'].__len__()

    plt.plot(range(1,num_epochs+1), hist['train_loss_hist'])

    # plt.plot(range(1,num_epochs+1), hist['train_loss_hist'])

    if hist.get('val_loss_hist'):

        plt.plot(range(1,num_epochs+1), hist['val_loss_hist'])

        plt.legend(['train loss', 'val loss'])

    else:

        plt.legend(['train loss'])

    plt.show()
'''

GENERATE PREDICTIONS

'''



print('Generating predictions for {} samples'.format(te_dataset.__len__()))



te_dataloader = DataLoader(te_dataset, batch_size=batch_size, shuffle=False)

submission_df = pd.read_csv(submission_df_path)





device = get_device()



model_ft.eval()

model_ft.to(device)

test_preds = None



for inputs in tqdm(te_dataloader):



    inputs = inputs.to(device)



    with torch.no_grad():

        outputs = model_ft(inputs)



        if test_preds is None:

            test_preds = outputs.data.cpu()

        else:

            test_preds = torch.cat((test_preds, outputs.data.cpu()), dim=0)

    

test_preds = torch.softmax(test_preds, dim=1, dtype=float)



submission_df[['healthy', 'multiple_diseases', 'rust', 'scab']] = test_preds



submission_df.to_csv('submission_3.csv', index=False)
submission_df