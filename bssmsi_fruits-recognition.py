import os

import time

import copy

import json



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split



import torch

from torch import nn, optim

import torchvision

from torchvision import datasets, models, transforms
# Use GPU

if torch.cuda.is_available():

    print(torch.cuda.get_device_name())

    device = torch.device('cuda')

else:

    device = torch.device('cpu')

print(device)
os.environ["TORCH_HOME"] = "../working"

train_path = "../input/fruits/fruits-360/Training"

test_path = "../input/fruits/fruits-360/Test"

num_classes = 131
def set_parameter_requires_grad(model):

    for param in model.parameters():

        param.requires_grad = False
def init_model(model_name, num_classes, feature_extract=True, use_pretrained=True):

    '''

    batch_size:int - Optimal batch_size for each model identified for this fruits dataset 

                     to ensure maximum GPU utilization (for 16GB RAM)

                     Note consider both validation and training batches

    '''

    is_inception = False

    if "alexnet" in model_name:

        model_ft = models.alexnet(pretrained=use_pretrained)

        if feature_extract:

            set_parameter_requires_grad(model_ft)

        num_ftrs = model_ft.classifier[6].in_features

        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)

        input_size = None

        batch_size = 10000



    elif "resnet" in model_name:

        if 'wide' in model_name:

            if '50' in model_name:

                model_ft = models.wide_resnet50_2(pretrained=use_pretrained)

            elif '101' in model_name:

                model_ft = models.wide_resnet101_2(pretrained=use_pretrained)

            batch_size = 5000    # Yet to determine

        elif '18' in model_name:

            model_ft = models.resnet18(pretrained=use_pretrained)

            batch_size = 5000    # 

        elif '34' in model_name:

            model_ft = models.resnet34(pretrained=use_pretrained)

            batch_size = 5000    # Yet to determine

        elif '50' in model_name:

            model_ft = models.resnet50(pretrained=use_pretrained)

            batch_size = 5000    # Yet to determine

        elif '101' in model_name:

            model_ft = models.resnet101(pretrained=use_pretrained)

            batch_size = 5000    # Yet to determine

        else:

            model_ft = models.resnet152(pretrained=use_pretrained)

            batch_size = 5000    # Yet to determine

        if feature_extract:

            set_parameter_requires_grad(model_ft)

        num_ftrs = model_ft.fc.in_features

        model_ft.fc = nn.Linear(num_ftrs, num_classes)

        input_size = None



    elif "vgg" in model_name:

        if '11' in model_name:

            if 'bn' in model_name:

                model_ft = models.vgg11_bn(pretrained=use_pretrained)

            else:

                model_ft = models.vgg11(pretrained=use_pretrained)

            batch_size = 2000

        elif '13' in model_name:

            if 'bn' in model_name:

                model_ft = models.vgg13_bn(pretrained=use_pretrained)

                batch_size = 1000

            else:

                model_ft = models.vgg13(pretrained=use_pretrained)

                batch_size = 2000

        elif '16' in model_name:

            if 'bn' in model_name:

                model_ft = models.vgg16_bn(pretrained=use_pretrained)

                batch_size = 2000

            else:

                model_ft = models.vgg16(pretrained=use_pretrained)

                batch_size = 1000

        elif '19' in model_name:

            if 'bn' not in model_name:

                model_ft = models.vgg19(pretrained=use_pretrained)

            else:

                model_ft = models.vgg19_bn(pretrained=use_pretrained)

            batch_size = 2000

        else:

            model_ft = models.vgg16_bn(pretrained=use_pretrained)

            batch_size = 2000



        if feature_extract:

            set_parameter_requires_grad(model_ft)

        num_ftrs = model_ft.classifier[6].in_features

        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)

        input_size = None



    elif "squeezenet" in model_name:

        if '1' in model_name and '0' in model_name:

            model_ft = models.squeezenet1_0(pretrained=use_pretrained)

            batch_size = 5000    # Yet to determine

        else:

            model_ft = models.squeezenet1_1(pretrained=use_pretrained)

            batch_size = 10000

        

        if feature_extract:

            set_parameter_requires_grad(model_ft)

        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))

        model_ft.num_classes = num_classes

        input_size = None



    elif 'shufflenet' in model_name:

        if '0' in model_name and '5' in model_name:

            model_ft = models.shufflenet_v2_x0_5(pretrained=use_pretrained)

            batch_size = 20000    # 16.1GB

        elif '1' in model_name and '5' in model_name:

            #model_ft = models.shufflenet_v2_x1_5(pretrained=use_pretrained)

            print("ShuffleNet 2.0 not implemented yet, using 1.0 instead")

            model_ft = models.shufflenet_v2_x1_0(pretrained=use_pretrained)

            batch_size = 20000

        elif '2' in model_name and '0' in model_name:

            #model_ft = models.shufflenet_v2_x2_0(pretrained=use_pretrained)

            print("ShuffleNet 2.0 not implemented yet, using 1.0 instead")

            model_ft = models.shufflenet_v2_x1_0(pretrained=use_pretrained)

            batch_size = 20000

        else:

            model_ft = models.shufflenet_v2_x1_0(pretrained=use_pretrained)

            batch_size = 20000    # Yet to determine



        if feature_extract:

            set_parameter_requires_grad(model_ft)

        num_ftrs = model_ft.fc.in_features

        model_ft.fc = nn.Linear(num_ftrs, num_classes)

        input_size = None



    elif "densenet" in model_name:

        if '121' in model_name:

            model_ft = models.densenet121(pretrained=use_pretrained)

            batch_size = 5000

        elif '169' in model_name:

            model_ft = models.densenet169(pretrained=use_pretrained)

            batch_size = 5000

        elif '201' in model_name:

            model_ft = models.densenet201(pretrained=use_pretrained)

            batch_size = 5000

        else:

            model_ft = models.densenet161(pretrained=use_pretrained)

            batch_size = 3000    # 15.963GB



        if feature_extract:

            set_parameter_requires_grad(model_ft)

        num_ftrs = model_ft.classifier.in_features

        model_ft.classifier = nn.Linear(num_ftrs, num_classes)

        input_size = None



    elif "inception" in model_name or 'googlenet' in model_name:

        # Be careful, expects (299,299) sized images and has auxiliary output

        if '1' in model_name or 'googlenet' in model_name:

            model_ft = models.googlenet(pretrained=use_pretrained)

            if feature_extract:

                set_parameter_requires_grad(model_ft)

            num_ftrs = model_ft.fc.in_features

            model_ft.fc = nn.Linear(num_ftrs, num_classes)

            input_size = None

            batch_size = 6000    # Usage 12GB

        else:

            is_inception = True

            model_ft = models.inception_v3(pretrained=use_pretrained)

            if feature_extract:

                set_parameter_requires_grad(model_ft)

            # Handle the auxilary net

            num_ftrs1 = model_ft.AuxLogits.fc.in_features

            model_ft.AuxLogits.fc = nn.Linear(num_ftrs1, num_classes)

            # Last Layer

            num_ftrs = model_ft.fc.in_features

            model_ft.fc = nn.Linear(num_ftrs,num_classes)

            input_size = 299

            batch_size = 500    # 13GB



    elif 'mnasnet' in model_name:

        if '0' in model_name and '75' in model_name:

            #model_ft = models.mnasnet0_75(pretrained=use_pretrained)

            print("Mnasnet 0.75 not implemented yet, using 1.0 instead")

            model_ft = models.mnasnet1_0(pretrained=use_pretrained)

            batch_size = 8000

        elif '0' in model_name and '5' in model_name:

            model_ft = models.mnasnet0_5(pretrained=use_pretrained)

            batch_size = 16000

        elif '1' in model_name and '3' in model_name:

            #model_ft = models.mnasnet1_3(pretrained=use_pretrained)

            print("Mnasnet 1.3 not implemented yet, using 1.0 instead")

            model_ft = models.mnasnet1_0(pretrained=use_pretrained)

            batch_size = 8000

        else:

            model_ft = models.mnasnet1_0(pretrained=use_pretrained)

            batch_size = 8000

        if feature_extract:

            set_parameter_requires_grad(model_ft)

        num_ftrs = model_ft.classifier[1].in_features

        model_ft.classifier[1] = nn.Linear(num_ftrs, num_classes)

        input_size = None



    elif 'mobilenet' in model_name:

        model_ft = models.mobilenet_v2(pretrained=use_pretrained)

        if feature_extract:

            set_parameter_requires_grad(model_ft)

        num_ftrs = model_ft.classifier[1].in_features

        model_ft.classifier[1] = nn.Linear(num_ftrs, num_classes)

        input_size = None

        batch_size = 5000    # Usage 14.7 GB

    

    elif 'resnext' in model_name:

        if '50' in model_name:

            model_ft = models.resnext50_32x4d(pretrained=use_pretrained)

            batch_size = 5000

        else:

            model_ft = models.resnext101_32x8d(pretrained=use_pretrained)

            batch_size = 3000    # 15.983GB

        if feature_extract:

            set_parameter_requires_grad(model_ft)

        num_ftrs = model_ft.fc.in_features

        model_ft.fc = nn.Linear(num_ftrs, num_classes)

        input_size = None



    else:

        print("Invalid model name, exiting...")

        exit()



    return model_ft, batch_size, input_size, is_inception
class CustomSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, indices):

        self.indices = indices



    def __iter__(self):

        return iter(self.indices)



    def __len__(self):

        return len(self.indices)
model_names = ['mobilenet', 'densenet161', 'resnet101']

feature_extract = False

use_pretrained = False

val_split = 0.2

random_seed = 42

num_epochs = 50

diff = 1000
fig, ((train_acc, val_acc), (train_loss, val_loss)) = plt.subplots(2, 2, figsize=(15, 15))

for model_name in model_names:

    print("-"*20,model_name,"-"*20)

    # # INITIALIZE MODEL

    model, BATCH_SIZE, input_size, is_inception = init_model(model_name, num_classes=num_classes, 

                                                             feature_extract=feature_extract, use_pretrained=use_pretrained)

    optimize = {

        'name': 'adam',

        'lr': 0.01

    }

    # # Parameters to update

    params_to_update = list(model.parameters())

    n_p = 0

    for p in model.parameters():

        n_p += p.numel()



    # # Transforming image data

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if input_size is not None:

        transformer = transforms.Compose([transforms.Resize(input_size), transforms.ToTensor(), normalize])

    else:

        transformer = transforms.Compose([transforms.ToTensor(), normalize])



    # # TRAINING MODEL    

    patience = 5    # Early stopping patience

    tol = 0    # Early stopping tolerance

    

    model = model.to(device)



    metrics = {

        'n_epochs': num_epochs,

        'optimizer': optimize,

        'train_acc': [],

        'train_loss': [],

        'val_acc': [],

        'val_loss': [],

        'test_acc': 0,

        'test_loss': 0,

        'train_time': 0,

        'n_params': n_p

    }

    best_model_wts = copy.deepcopy(model.state_dict())

    

    start_time = time.time()

    while True:

        try:

            #  # LOSS AND OPTIMIZER

            criterion = nn.CrossEntropyLoss()

            if optimize['name'] == 'adam':

                optimizer = optim.Adam(params_to_update, lr=optimize['lr'])

            elif optimize['name'] == 'sgd':

                optimizer = optim.SGD(params_to_update, lr=optimize['lr'], momentum=optimize['m'])

            # # DATA LOADER

            dataset = datasets.ImageFolder(root=train_path, transform=transformer)

            #labels_mapping = pd.Series(dataset.classes)

            targets = dataset.targets

            train_idx, val_idx = train_test_split(np.arange(len(targets)), test_size=val_split, shuffle=True, 

                                                  random_state=random_seed, stratify=targets)

            n_train, n_val = len(train_idx), len(val_idx)

            # Don't shuffle again since above shuflling is already done

            train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, sampler=CustomSampler(train_idx))

            val_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, sampler=CustomSampler(val_idx))



            test_dataset = datasets.ImageFolder(root=test_path, transform=transformer)

            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

            wait = 0

            for epoch in range(num_epochs):

                # ## Training

                model.train()  # Set model to training mode

                running_loss = 0.0

                running_acc = 0

                for inputs, labels in iter(train_loader):

                    inputs = inputs.to(device)

                    labels = labels.to(device)

                    # zero the parameter gradients

                    optimizer.zero_grad()

                    if is_inception:

                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958

                        outputs, aux_outputs = model(inputs)

                        loss1 = criterion(outputs, labels)

                        loss2 = criterion(aux_outputs, labels)

                        loss = loss1 + 0.4*loss2

                    else:

                        outputs = model.forward(inputs)

                        loss = criterion(outputs, labels)



                    _, preds = torch.max(outputs, 1)



                    loss.backward()

                    optimizer.step()

                    # loss.item() gives averaged loss for this batch, hence get back the original sum of losses by multiplying the batch size

                    running_loss += loss.item() * inputs.size(0)

                    running_acc += torch.sum(preds == labels.data).item()

                # Mean accuracy of epoch

                metrics['train_acc'].append(100*running_acc / n_train)

                # Mean loss of epoch

                metrics['train_loss'].append(running_loss / n_train)

                del inputs, labels, loss

                if 'cuda' in str(device):

                    torch.cuda.empty_cache()

                # ## Validation

                model.eval()

                running_loss = 0.0

                running_acc = 0

                for inputs, labels in iter(val_loader):

                    inputs = inputs.to(device)

                    labels = labels.to(device)

                    with torch.set_grad_enabled(False):

                        outputs = model.forward(inputs)

                        loss = criterion(outputs, labels)



                    _, preds = torch.max(outputs, 1)

                    # item() to return only value, otherwise cuda tensor is returned

                    # loss.item() gives averaged loss for this batch, hence get back the original sum of losses by multiplying the batch size

                    running_loss += loss.item() * inputs.size(0)

                    running_acc += torch.sum(preds == labels.data).item()

                # Mean validation accuracy of epoch

                metrics['val_acc'].append(100*running_acc / n_val)

                # Mean validation loss of epoch

                metrics['val_loss'].append(running_loss / n_val)



                # ## Early Stopping

                if metrics['val_acc'][-1] >= max(metrics['val_acc'])-tol:

                    wait = 0

                    # Store weights for restoring best weights in case early stopped

                    best_model_wts = copy.deepcopy(model.state_dict())

                else:

                    wait += 1

                    # Stop if validation accuracy is not increasing for the past 'wait' epochs

                    if wait >= patience:

                        # Load best model weights

                        model.load_state_dict(best_model_wts)

                        print("Early stopping at epoch %d out of total %d" % (epoch+1, num_epochs))

                        break



            metrics['train_time'] = time.time() - start_time



            # Plot train and validation metrics

            train_acc.plot(metrics['train_acc'], label=model_name)

            train_loss.plot(metrics['train_loss'])

            val_acc.plot(metrics['val_acc'])

            val_loss.plot(metrics['val_loss'])



            del inputs, labels, loss, optimizer

            if 'cuda' in str(device):

                torch.cuda.empty_cache()



            # # TEST DATA

            model.eval()

            running_loss = 0.0

            running_acc = 0

            for inputs, labels in iter(test_loader):

                inputs, labels = inputs.to(device), labels.to(device)



                with torch.set_grad_enabled(False):

                    outputs = model.forward(inputs)

                    loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)



                running_loss += loss.item() * inputs.size(0)

                running_acc += torch.sum(preds == labels.data).item()

            metrics['test_acc'] = 100*running_acc / len(test_loader.dataset)

            metrics['test_loss'] = running_loss / len(test_loader.dataset)



            print("Best train accuracy = %.2f, Best validation accuracy = %.2f, Test accuracy = %.2f" % 

                  (max(metrics['train_acc']), max(metrics['val_acc']), metrics['test_acc']))



            # # Save model weights and related metrics

            torch.save({'metrics': metrics,

                        'model_state_dict': model.state_dict()}, 

                        f'{os.environ["TORCH_HOME"]}/{model_name}')



            # # Clear GPU Memory

            del inputs, labels, loss, model

            if 'cuda' in str(device):

                torch.cuda.empty_cache()



            # # Delete downloaded model

            pretrained_model_dir = os.path.join(os.environ['TORCH_HOME'], 'checkpoints')

            l = os.listdir(pretrained_model_dir)

            for filename in l:

                os.remove(os.path.join(pretrained_model_dir, filename))

            print("Best Batch Size = ", BATCH_SIZE)

            break

        except RuntimeError:

            print("Error with Batch Size ", BATCH_SIZE)

            if BATCH_SIZE - diff > 0:

                BATCH_SIZE = int(BATCH_SIZE - diff)

            else:

                diff = int(diff / 2)

                BATCH_SIZE = int(BATCH_SIZE - diff)

            

        except ValueError as ve:

            print(f"Error in {model_name}: {ve}")

        finally:

            # # Clear GPU Memory

            try:

                del inputs, labels, loss, optimizer

                torch.cuda.empty_cache()

            except:

                torch.cuda.empty_cache()



# Prettifying plot

yticks = range(50, 110, 10)

train_acc.set_title('Training Accuracy')

train_acc.set_yticks(yticks)

train_loss.set_title('Training Loss')

val_acc.set_title('Validation Accuracy')

val_acc.set_yticks(yticks)

val_loss.set_title('Validation Loss')



fig.legend()

# add a big axis, hide frame

fig.add_subplot(111, frameon=False)

# hide tick and tick label of the big axis

plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

plt.xlabel("Epochs ->")

plt.ylabel("Metric")

plt.tight_layout()
os.rmdir(os.path.join(os.environ['TORCH_HOME'], 'checkpoints'))