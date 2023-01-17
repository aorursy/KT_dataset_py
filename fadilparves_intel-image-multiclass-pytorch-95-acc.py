from pathlib import Path

import numpy as np

import cv2

import pandas as pd

from tqdm import tqdm

import PIL.Image as Image

import seaborn as sns

from pylab import rcParams

import matplotlib.pyplot as plt

from matplotlib import rc

from matplotlib.ticker import MaxNLocator

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report

from glob import glob

import shutil

from collections import defaultdict



import torch, torchvision

from torch import nn, optim

from torch.optim import lr_scheduler

import torch.nn.functional as F

import torchvision.transforms as T

from torchvision.datasets import ImageFolder

from torch.utils.data import DataLoader

from torchvision import models



%matplotlib inline



sns.set(style='whitegrid', palette='muted', font_scale=1.2)



HAPPY_COLORS_PALETTE=["#01BEFE","#FFDD00","#FF7D00","#FF006D","#ADFF02","#8F00FF"]



sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))



rcParams['figure.figsize'] = 15, 10



RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

torch.manual_seed(RANDOM_SEED)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device
train_folders = sorted(glob('../input/intel-image-classification/seg_train/seg_train/*'))

len(train_folders)
def load_image(img_path, resize=True):

    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    

    if resize:

        img = cv2.resize(img, (64,64), interpolation = cv2.INTER_AREA)

    

    return img



def show_image(img_path):

    img = load_image(img_path)

    plt.imshow(img)

    plt.axis('off')

    

def show_sign_grid(image_paths):

    images = [load_image(img) for img in image_paths]

    images = torch.as_tensor(images)

    images = images.permute(0,3,1,2)

    grid_img = torchvision.utils.make_grid(images, nrow=11)

    plt.figure(figsize=(24,12))

    plt.imshow(grid_img.permute(1,2,0))

    plt.axis('off')
sample_images = [np.random.choice(glob(f'{tf}/*jpg')) for tf in train_folders]

show_sign_grid(sample_images)
img_path = glob(f'{train_folders[0]}/*jpg')[1]



show_image(img_path)
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']



class_indices = [0,1,2,3,4,5]
!rm -rf data



DATA_DIR = Path('data')



DATASETS = ['train', 'val']



for ds in DATASETS:

    for cls in class_names:

        (DATA_DIR / ds / cls).mkdir(parents=True, exist_ok=True)
for i, cls_index in enumerate(class_indices):

    image_paths = np.array(glob(f'{train_folders[cls_index]}/*jpg'))

    class_name = class_names[i]

    print(f'{class_name}: {len(image_paths)}')

    np.random.shuffle(image_paths)

    

    ds_split = np.split(

        image_paths,

        indices_or_sections=[int(.8*len(image_paths)), int(.9*len(image_paths))]

    )

    

    dataset_data = zip(DATASETS, ds_split)

    for ds, images in dataset_data:

        for img_path in images:

            shutil.copy(img_path, f'{DATA_DIR}/{ds}/{class_name}/')
mean_nums = [0.485, 0.456, 0.406]

std_nums = [0.229, 0.224, 0.225]



transforms = {'train': T.Compose([

    T.RandomResizedCrop(size=256),

    T.RandomRotation(degrees=15),

    T.RandomHorizontalFlip(),

    T.ToTensor(),

    T.Normalize(mean_nums, std_nums)

]), 'val': T.Compose([

    T.Resize(size=256),

    T.CenterCrop(size=224),

    T.ToTensor(),

    T.Normalize(mean_nums, std_nums)

]),}
image_datasets = {

    d: ImageFolder(f'{DATA_DIR}/{d}', transforms[d]) for d in DATASETS

}



data_loaders = {

    d: DataLoader(image_datasets[d], batch_size=4, shuffle=True, num_workers=4) for d in DATASETS

}
dataset_sizes = {d: len(image_datasets[d]) for d in DATASETS}

class_names = image_datasets['train'].classes



dataset_sizes
def imshow(inp, title=None):

    inp = inp.numpy().transpose((1,2,0))

    mean = np.array([mean_nums])

    std = np.array([std_nums])

    inp = std * inp + mean

    inp = np.clip(inp,0,1)

    plt.imshow(inp)

    if title is not None:

        plt.title(title)

    plt.axis('off')

    

inputs, classes = next(iter(data_loaders['train']))

out = torchvision.utils.make_grid(inputs)



imshow(out, title=[class_names[x] for x in classes])
def create_model(n_classes):

    model = models.resnet34(pretrained=True)

    

    n_features = model.fc.in_features

    model.fc = nn.Linear(n_features, n_classes)

    

    return model.to(device)
base_model = create_model(len(class_names))
def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):

    model = model.train() #Convert to train mode

    losses = []

    correct_predictions = 0

    

    for inputs, labels in data_loader:

        inputs = inputs.to(device) #Push array to gpu

        labels = labels.to(device)

        

        outputs = model(inputs) #get prob of output per class

        

        _, preds = torch.max(outputs, dim=1) # get max of pred

        loss = loss_fn(outputs, labels) # get loss

        

        correct_predictions += torch.sum(preds==labels)

        losses.append(loss.item())

        

        loss.backward()

        optimizer.step()

        optimizer.zero_grad()

    

    scheduler.step()

    

    return correct_predictions.double() / n_examples, np.mean(losses)



def eval_model(model, data_loader, loss_fn, device, n_examples):

    model = model.eval() #Evaluation mode

    

    losses = []

    correct_predictions = 0

    

    with torch.no_grad():

        for inputs, labels in data_loader:

            inputs = inputs.to(device)

            labels = labels.to(device)

            

            outputs = model(inputs)

            

            _, preds = torch.max(outputs, dim=1)

            

            loss = loss_fn(outputs, labels)

            

            correct_predictions += torch.sum(preds==labels)

            losses.append(loss.item())

    

    return correct_predictions.double() / n_examples, np.mean(losses) 
def train_model(model, data_loaders, dataset_sizes, device, n_epochs=5):

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    loss_fn = nn.CrossEntropyLoss().to(device)

    

    history = defaultdict(list)

    best_accuracy = 0

    

    for epoch in range(n_epochs):

        print(f'Epoch {epoch + 1}/{n_epochs}')

        print('-' * 10)

        

        train_acc, train_loss = train_epoch(model, data_loaders['train'], loss_fn, 

                                            optimizer, device, scheduler, dataset_sizes['train'])

        

        print(f'Train loss {train_loss} accuracy {train_acc}')

        

        val_acc, val_loss = eval_model(model, data_loaders['val'], loss_fn, device, dataset_sizes['val'])

        

        print(f'Val loss {val_loss} accuracy {val_acc}')

        print()

        

        history['train_acc'].append(train_acc)

        history['train_loss'].append(train_loss)

        history['val_acc'].append(val_acc)

        history['val_loss'].append(val_loss)

        

        if val_acc > best_accuracy:

            torch.save(model.state_dict(), 'best_model_state.bin')

            best_accuracy = val_acc

            

    print(f'Best val accuracy: {best_accuracy}')

    

    model.load_state_dict(torch.load('best_model_state.bin'))

    

    return model, history
%%time



base_model, history = train_model(base_model, data_loaders, dataset_sizes, device, n_epochs=20)
def plot_training_history(history):

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(18,6))

    ax1.plot(history['train_loss'], label='train loss')

    ax1.plot(history['val_loss'], label='validation loss')

    

    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax1.set_ylim([-0.05,1.05])

    ax1.legend()

    ax1.set_ylabel('Loss')

    ax1.set_xlabel('Epoch')

    

    ax2.plot(history['train_acc'], label='train accuracy')

    ax2.plot(history['val_acc'], label='validation accuracy')

    

    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax2.set_ylim([-0.05,1.05])

    ax2.legend()

    ax2.set_ylabel('Accuracy')

    ax2.set_xlabel('Epoch')

    

    fig.suptitle('Training History')

    

plot_training_history(history)
test_folders = sorted(glob('../input/intel-image-classification/seg_test/seg_test/*'))

len(test_folders)
sample_images = [np.random.choice(glob(f'{tf}/*jpg')) for tf in test_folders]

show_sign_grid(sample_images)
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']



class_indices = [0,1,2,3,4,5]



DATASETS = ['test']



for ds in DATASETS:

    for cls in class_names:

        (DATA_DIR / ds / cls).mkdir(parents=True, exist_ok=True)

        

for i, cls_index in enumerate(class_indices):

    image_paths = np.array(glob(f'{test_folders[cls_index]}/*jpg'))

    class_name = class_names[i]

    print(f'{class_name}: {len(image_paths)}')

    np.random.shuffle(image_paths)

    

    ds_split = np.split(

        image_paths,

        indices_or_sections=[int(.8*len(image_paths)), int(.9*len(image_paths))]

    )

    

    dataset_data = zip(DATASETS, ds_split)

    for ds, images in dataset_data:

        for img_path in images:

            shutil.copy(img_path, f'{DATA_DIR}/{ds}/{class_name}/')

            

mean_nums = [0.485, 0.456, 0.406]

std_nums = [0.229, 0.224, 0.225]



transforms = {'test': T.Compose([

    T.Resize(size=256),

    T.CenterCrop(size=224),

    T.ToTensor(),

    T.Normalize(mean_nums, std_nums)

]),}



image_datasets = {

    d: ImageFolder(f'{DATA_DIR}/{d}', transforms[d]) for d in DATASETS

}



data_loaders = {

    d: DataLoader(image_datasets[d], batch_size=4, shuffle=True, num_workers=4) for d in DATASETS

}



dataset_sizes = {d: len(image_datasets[d]) for d in DATASETS}

class_names = image_datasets['test'].classes



dataset_sizes
def show_predictions(model, class_names, n_images=6):

    model = model.eval()

    images_handeled = 0

    plt.figure()

    

    with torch.no_grad():

        for i, (inputs, labels) in enumerate(data_loaders['test']):

            inputs = inputs.to(device)

            labels = labels.to(device)

            

            outputs = model(inputs)

            

            _, preds = torch.max(outputs, 1)

            

            for j in range(inputs.shape[0]):

                images_handeled += 1

                ax = plt.subplot(2, n_images//2, images_handeled)

                ax.set_title(f'predicted: {class_names[preds[j]]}')

                imshow(inputs.cpu().data[j])

                ax.axis('off')

                

                if images_handeled == n_images:

                    return

                

show_predictions(base_model, class_names, n_images=8)
def get_predictions(model, data_loader):

    model = model.eval()

    predictions = []

    real_values = []

    with torch.no_grad():

        for inputs, labels in data_loader:

            inputs = inputs.to(device)

            labels = labels.to(device)

            

            outputs = model(inputs)

            

            _, preds = torch.max(outputs, 1)

            predictions.extend(preds)

            real_values.extend(labels)

    predictions = torch.as_tensor(predictions).cpu()

    real_values = torch.as_tensor(real_values).cpu()

    

    return predictions, real_values



y_pred, y_test = get_predictions(base_model, data_loaders['test'])



print(classification_report(y_test, y_pred, target_names=class_names))
!rm -rf data/pred_test
show_image('../input/intel-image-classification/seg_pred/seg_pred/10004.jpg')
def predict_proba(model, img_path):

    img = Image.open(img_path)

    img = img.convert('RGB')

    img = transforms['test'](img).unsqueeze(0)

    

    pred = model(img.to(device))

    pred = F.softmax(pred,1)

    return pred.detach().cpu().numpy().flatten()



def show_prediction_confidence(prediction, class_names):

    pred_df = pd.DataFrame({'class_names': class_names, 'values': prediction})

    sns.barplot(x='values', y='class_names', data=pred_df, orient='h')

    plt.xlim([0, 1]);



pred = predict_proba(base_model, '../input/intel-image-classification/seg_pred/seg_pred/10004.jpg')

show_prediction_confidence(pred, class_names)
show_image('../input/intel-image-classification/seg_pred/seg_pred/10079.jpg')
pred = predict_proba(base_model, '../input/intel-image-classification/seg_pred/seg_pred/10079.jpg')

show_prediction_confidence(pred, class_names)
show_image('../input/intel-image-classification/seg_pred/seg_pred/101.jpg')
pred = predict_proba(base_model, '../input/intel-image-classification/seg_pred/seg_pred/101.jpg')

show_prediction_confidence(pred, class_names)