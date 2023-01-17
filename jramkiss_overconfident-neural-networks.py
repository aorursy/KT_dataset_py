import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy

import glob
from PIL import Image

from scipy.special import softmax

import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.image as mpimg
mpl.rcParams['figure.dpi'] = 200
img_size = (300, 300)

# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor()
    ]),
    'val': transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor()
    ]),
}

data_dir = '/kaggle/input/animal-faces/afhq/'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# plot data:
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 4))
#fig.suptitle("Training Data", fontsize=20)

img1 = data_dir + 'train/wild/flickr_wild_000769.jpg'
ax[0].axis('off')
ax[0].imshow(mpimg.imread(img1))

img2 = data_dir + 'train/cat/pixabay_cat_000250.jpg'
ax[1].axis('off')
ax[1].imshow( mpimg.imread(img2))
ax[1].set(title = "Training data")

img3 = data_dir + 'train/dog/pixabay_dog_000368.jpg'
ax[2].axis('off')
ax[2].imshow(mpimg.imread(img3));
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])
                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
        
# function that accepts an image path and returns a format to be consumed by the model 
def process_img (img_path) : 
    transform_pipeline = transforms.Compose([transforms.Resize(img_size),
                                             transforms.ToTensor()])
    img = transform_pipeline(Image.open(img_path).convert("RGB")).to(device)
    return (img.unsqueeze(0))
torch.manual_seed(42)
np.random.seed(42)

PATH = "/kaggle/resnet18_state_dict.pkl"
model_conv = torchvision.models.resnet18(pretrained=True)
#model_conv = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True) # mobilenet

for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, len(class_names), bias = False)
for param in model_conv.fc.parameters():
    param.requires_grad = True
    
# training:
model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
wd = 5e-4
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9, weight_decay = wd)

# Decay LR by a factor of 0.1 every 4 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=4, gamma=0.1)
model_conv = train_model(model = model_conv, 
                         criterion = criterion, 
                         optimizer = optimizer_conv,
                         scheduler = exp_lr_scheduler, 
                         num_epochs = 10)
# find a dog that is predicted as "dog" with less than 94% probability 
low_prob_dogs = []
for pth in glob.glob(data_dir + 'val/dog/*') :
    _ = softmax(model_conv(process_img(pth))[0].cpu().detach().numpy())
    _ind = np.argmax(_) # index of prediction 
    if class_names[_ind] == "dog" :
        if _[_ind] < 0.94 :
            low_prob_dogs.append(pth)
            
low_prob_dogs
#!pip install backpack-for-pytorch
#!pip install backpack-for-pytorch

from math import *
from backpack import extend, backpack, extensions
from torch.distributions.multivariate_normal import MultivariateNormal
# hack to remove the last layer of ResNet and extract features
feature_extr = nn.Sequential(*list(model_conv.children())[:-1])
print("Number of features: ", list(feature_extr.parameters())[-1].shape[0])

# training script
W = list(model_conv.fc.parameters())[0] #list(model_conv.parameters())[-2]
shape_W = W.shape

# Use BackPACK to get the Kronecker-factored last-layer covariance
_ = extend(model_conv.fc)
loss_func = extend(nn.CrossEntropyLoss(reduction='sum'))

for x_train, y_train in dataloaders["train"] :
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    loss = loss_func(model_conv(x_train), y_train)
    with backpack(extensions.KFAC()): # calculate the hessian 
        loss.backward()

# The Kronecker-factored Hessian of the negative log-posterior
A, B = W.kfac

print("LL Laplace Approx complete")
# predict method
@torch.no_grad()
def predict(x):
    x = x.to(device)
    phi = feature_extr(x).reshape(1, list(feature_extr.parameters())[-1].shape[0])
    # MAP prediction
    m = phi @ W.T
    v = torch.diag(phi @ V @ phi.T).reshape(-1, 1, 1) * U
    output_dist = MultivariateNormal(m, v)
    # MC-integral
    n_sample = 2500
    py = 0
    for _ in range(n_sample):
        out_s = output_dist.rsample()
        py += torch.softmax(out_s, 1)
    py /= n_sample
    smx = np.round(softmax(m.cpu().numpy())[0], 2) # rounded softmax prediction
    laplace = np.round(py.cpu().numpy()[0], 2) # rounded prediction from llla
    smx_top = (class_names[np.argmax(smx)], smx[np.argmax(smx)])
    laplace_top = (class_names[np.argmax(laplace)], laplace[np.argmax(laplace)])
    return (smx, laplace, smx_top, laplace_top)
# The weight decay used for training is the Gaussian prior's precision
prec0 = 2.5 # increasing causes the predictions to approach the softmax preds, decreasing causes it to tend to uniform

# The posterior covariance's Kronecker factors
U = torch.inverse(A + sqrt(prec0)*torch.eye(shape_W[0]).to(device))
V = torch.inverse(B + sqrt(prec0)*torch.eye(shape_W[1]).to(device))
## Softmax only model: 
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

img1 = '../input/unseendata/me.jpeg'
ax[0].axis('off')
ax[0].imshow(mpimg.imread(img1))
_p1 = predict(process_img(img1))
ax[0].set(title = "Prediction: " + str(_p1[2][0]) + '\n' + 'Probability: ' + str(_p1[2][1]));

img2 = data_dir + 'val/dog/pixabay_dog_001356.jpg' #flickr_dog_000849.jpg' # 101
ax[1].axis('off')
ax[1].imshow( mpimg.imread(img2))
_p2 = predict(process_img(img2))
ax[1].set(title = "Prediction: " + str(_p2[2][0]) + '\n' + 'Probability: ' + str(_p2[2][1]));
# Softmax model + LLLA model

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

img1 = '../input/unseendata/me.jpeg'
ax[0].axis('off')
ax[0].imshow(mpimg.imread(img1))
ax[0].set(title = "LLLA: " + str(predict(process_img(img1))[3]) + '\n' +
                  "Softmax: " + str(predict(process_img(img1))[2]))

img2 = data_dir + 'val/dog/pixabay_dog_001356.jpg'
ax[1].axis('off')
ax[1].imshow( mpimg.imread(img2))
ax[1].set(title = "LLLA: " + str(predict(process_img(img2))[3]) + '\n' +
                  "Softmax: " + str(predict(process_img(img2))[2]));

# img3 = data_dir + 'val/cat/flickr_cat_000709.jpg'
# ax[2].axis('off')
# ax[2].imshow(mpimg.imread(img3))
# ax[2].set(title = "LLLA: " + str(predict(process_img(img3))[3]) + '\n' +
#                   "Softmax: " + str(predict(process_img(img3))[2]));
# run the model over the validation set and plot histograms of the top predicted class
softmax_preds = []
bayesian_preds = []
true_labels = []
img_paths = []
for imgs, lbls in dataloaders["val"] :
    for i in range(len(lbls)) :
        img_paths.append(imgs[i])
        p_ = predict(imgs[i].unsqueeze(0))
        softmax_preds.append(p_[2])
        bayesian_preds.append(p_[3])
        true_labels.append(class_names[lbls[i].numpy()])

softmax_preds = np.array(softmax_preds)
bayesian_preds = np.array(bayesian_preds)
true_labels = np.array(true_labels)

print("Softmax validation set accuracy: ", np.mean(true_labels == list(map(lambda x: x[0], softmax_preds))))
print("Laplace Approximation validation set accuracy: ", np.mean(true_labels == list(map(lambda x: x[0], bayesian_preds))))
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

sns.distplot(list(map(lambda x: x[1], softmax_preds)), kde = False, ax = ax[0]).set_title("Softmax Predictions - Animal Validation Data")
ax[0].set(xlabel = "Top class probability")

sns.distplot(list(map(lambda x: x[1], bayesian_preds)), kde = False, ax = ax[1]).set_title("LLLA Predictions - Animal Validation Data")
ax[1].set(xlabel = "Top class probability");
low_ind = np.arange(1500)[np.array(list(map(lambda x: x[1], bayesian_preds)), dtype = float) < 0.45]
hi_ind = np.arange(1500)[np.array(list(map(lambda x: x[1], bayesian_preds)), dtype = float) > 0.8]
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12, 7))
fig.suptitle("LLLA Low Confidence Predictions", fontsize=12)

for i, ax in enumerate(ax.flatten()) :
    if i == sum(low_ind) : 
        ax.axis('off')
        break
    inp = img_paths[low_ind[i]]
    inp = inp.numpy().transpose((1, 2, 0))
    ax.axis('off')
    ax.set(title = "LLLA: " + str(bayesian_preds[low_ind[i]]) + '\n' + "Softmax: " + str(softmax_preds[low_ind[i]]))
    ax.imshow(inp);
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12, 7))
fig.suptitle("LLLA High Confidence Predictions", fontsize=12)

for i, ax in enumerate(ax.flatten()) :
    if i == sum(hi_ind) : 
        ax.axis('off')
        break
    inp = img_paths[hi_ind[i]]
    inp = inp.numpy().transpose((1, 2, 0))
    ax.axis('off')
    ax.set(title = "LLLA: " + str(bayesian_preds[hi_ind[i]]) + '\n' + "Softmax: " + str(softmax_preds[hi_ind[i]]))
    ax.imshow(inp);
# set a threshold on the prediction, anything below the threshold will be predicted as "other"
thresholds = [0.4, 0.45, 0.5, 0.55, 0.6, 0.7]
softmax_accuracies = []
bayesian_accuracies = []

for threshold in thresholds :
    thresh_softmax_preds = []
    thresh_bayesian_preds = []
    thresh_true_labels = []
    thresh_img_paths = []
    for imgs, lbls in dataloaders["val"] :
        for i in range(len(lbls)) :
            thresh_img_paths.append(imgs[i])
            p_ = predict(imgs[i].unsqueeze(0))
            thresh_softmax_preds.append(p_[2])
            if p_[3][1] > threshold :
                thresh_bayesian_preds.append(p_[3])
            else : 
                thresh_bayesian_preds.append(("other", 1.0))
            thresh_true_labels.append(class_names[lbls[i].numpy()])

    thresh_softmax_preds = np.array(thresh_softmax_preds)
    thresh_bayesian_preds = np.array(thresh_bayesian_preds)
    thresh_true_labels = np.array(thresh_true_labels)
    softmax_accuracies.append(np.mean(thresh_true_labels == list(map(lambda x: x[0], thresh_softmax_preds))))
    bayesian_accuracies.append(np.mean(thresh_true_labels == list(map(lambda x: x[0], thresh_bayesian_preds))))
    print("Softmax validation set accuracy: ", 
          np.mean(thresh_true_labels == list(map(lambda x: x[0], thresh_softmax_preds))))
    print("Laplace Approximation validation set accuracy: ", 
          np.mean(thresh_true_labels == list(map(lambda x: x[0], thresh_bayesian_preds))))
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 2))
sns.lineplot(thresholds, bayesian_accuracies, ax = ax, 
             label = "LLLA Accuracies",
             color = sns.xkcd_rgb["denim blue"])
sns.lineplot(thresholds, softmax_accuracies, ax = ax,
             label = "Softmax Accuracies",
             color = sns.xkcd_rgb["pale red"])
ax.legend(loc = "lower left", fontsize=8);
#Image.open(glob.glob('/kaggle/input/simpsons-faces/cropped/*')).resize(img_size)
simpsons_pths = glob.glob('/kaggle/input/simpsons-faces/cropped/*')[:300]

# plot data:
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 4))
img1 = simpsons_pths[1]
ax[0].axis('off')
ax[0].imshow(mpimg.imread(img1))

img2 = simpsons_pths[20]
ax[1].axis('off')
ax[1].imshow( mpimg.imread(img2))
ax[1].set(title = "Out-of-distribution Data")

img3 = simpsons_pths[11]
ax[2].axis('off')
ax[2].imshow(mpimg.imread(img3));
simpsons_bayesian = []
simpsons_softmax = []

for img_pth in simpsons_pths: 
    _p = predict(process_img(img_pth))
    simpsons_bayesian.append(_p[3])
    simpsons_softmax.append(_p[2])
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

sns.distplot(list(map(lambda x: x[1], simpsons_softmax)), ax = ax[0]).set_title("Softmax Top Predictions - Simpsons Data")
ax[0].set(xlabel = "Top class probability")

sns.distplot(list(map(lambda x: x[1], simpsons_bayesian)), ax = ax[1]).set_title("LLLA Top Predictions - Simpsons Data")
ax[1].set(xlabel = "Top class probability");
