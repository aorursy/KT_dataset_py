import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import random



from PIL import Image

from imageio import imread



import seaborn as sns

sns.set_style("dark")

sns.set()



import torch

import torch.nn as nn

import torch.optim as optim

from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CyclicLR

import torchvision

from torchvision import datasets, models, transforms

from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.utils.class_weight import compute_class_weight

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.metrics import accuracy_score



from skimage.morphology import closing, disk, opening



import cv2

import time

import copy

from tqdm import tqdm_notebook as tqdm



from os import listdir

from skimage.segmentation import mark_boundaries



torch.manual_seed(42)

torch.cuda.manual_seed(42)

np.random.seed(42)

random.seed(42)

torch.backends.cudnn.deterministic=True
run_training=False
listdir("../input")
base_path = "../input/v2-plant-seedlings-dataset/nonsegmentedv2/"

OUTPUT_PATH = "segmented_seedlings"

MODEL_PATH = "../input/seedlingsmodel/segmented_seedlings"

LOSSES_PATH = "../input/seedlingsmodel/"

subfolders = listdir(base_path)

subfolders
total_images = 0

for folder in subfolders:

    total_images += len(listdir(base_path + folder))



plantstate = pd.DataFrame(index=np.arange(0, total_images), columns=["width", "height", "species"])



k = 0

all_images = []

for m in range(len(subfolders)):

    folder = subfolders[m]

    

    images = listdir(base_path + folder)

    all_images.extend(images)

    n_images = len(images)

    

    for n in range(0, n_images):

        image = imread(base_path + folder + "/" + images[n])

        plantstate.loc[k, "width"] = image.shape[1]

        plantstate.loc[k, "height"] = image.shape[0]

        plantstate.loc[k, "species"] = folder

        plantstate.loc[k, "image_name"] = images[n]

        k+=1



plantstate.width = plantstate.width.astype(np.int)

plantstate.height = plantstate.height.astype(np.int)

plantstate.head()
encoder = LabelEncoder()

labels = encoder.fit_transform(plantstate.species.values)

plantstate["target"] = labels

NUM_CLASSES = plantstate.target.nunique()

plantstate.head()
fig, ax = plt.subplots(4,3,figsize=(20,25))



for m in range(4):

    for n in range(3):

        folder = subfolders[m+n*4]

        files = listdir(base_path + folder + "/")

        image = imread(base_path + folder + "/" + files[0])

        ax[m,n].imshow(image)

        ax[m,n].grid(False)

        ax[m,n].set_title(folder + "/" + str(m+n*4+1))
example_path = base_path + "Sugar beet/27.png" 
fig,ax = plt.subplots(3,3,figsize=(20,17))



titles = [["Red", "Green", "Blue"],

         ["Hue", "Saturation", "Value"],

         ["\n lightness from black to white", "\n A - from green to red", "\n B - from blue to yellow"]]

pil_image = Image.open(example_path)

np_image = np.array(pil_image)

image_hvs = cv2.cvtColor(np_image, cv2.COLOR_BGR2HSV)

image_lab = cv2.cvtColor(np_image, cv2.COLOR_BGR2LAB)



for n in range(3):

    ax[0,n].imshow(np_image[:,:,n], cmap="RdYlGn")

    ax[0,n].grid(False)

    ax[0,n].set_title("Sugar beet/27" + " - " + titles[0][n]);

    ax[1,n].imshow(image_hvs[:,:,n], cmap="RdYlGn")

    ax[1,n].set_title("Sugar beet/27" + " - " + titles[1][n]);

    ax[2,n].imshow(image_lab[:,:,n], cmap="RdYlGn")

    ax[2,n].set_title("Sugar beet/27" + " - " + titles[2][n]);

plt.savefig("Colorspace", dpi=500)
my_threshold = 121

my_radius = 2
def get_mask(image, threshold, radius):

    mask = np.where(image < threshold, 1, 0)

    selem = disk(radius)

    mask = closing(mask, selem)

    return mask
def segment_plant(np_image, threshold, radius):

    image_lab = cv2.cvtColor(np_image, cv2.COLOR_BGR2LAB)

    mask = get_mask(image_lab[:,:,1], threshold, radius)

    masked_image = np_image.copy()

    for n in range(3):

        masked_image[:,:,n] = np_image[:,:,n] * mask

    return masked_image
fig, ax = plt.subplots(1,3,figsize=(20,5))

sns.distplot(image_lab[:,:,1].flatten(), ax=ax[0], kde=False)

mask = get_mask(image_lab[:,:,1], my_threshold, my_radius)

ax[1].imshow(mask);

ax[2].imshow(segment_plant(np_image, my_threshold, my_radius))

ax[0].grid(False)

ax[1].grid(False)

ax[2].grid(False)
fig, ax = plt.subplots(4,6,figsize=(20,14))



for m in range(6):

    folder = subfolders[m]

    files = listdir(base_path + folder + "/")

    image = np.array(Image.open(base_path + folder + "/" + files[0]))

    ax[0,m].imshow(image)

    ax[1,m].imshow(segment_plant(image, my_threshold, my_radius))

    ax[0,m].grid(False)

    ax[1,m].grid(False)

    ax[0,m].set_title(folder + "/" + files[0])

    

    folder = subfolders[m+6]

    files = listdir(base_path + folder + "/")

    image = np.array(Image.open(base_path + folder + "/" + files[0]))

    ax[2,m].imshow(image)

    ax[3,m].imshow(segment_plant(image, my_threshold, my_radius))

    ax[2,m].grid(False)

    ax[3,m].grid(False)

    ax[2,m].set_title(folder + "/" + files[0])
class SegmentPlant(object):

    

    def __call__(self, image):

        np_image = np.array(image)

        image = segment_plant(np_image, my_threshold, my_radius)

        pil_image = Image.fromarray(image)

        return pil_image
class RandomZoom(object):

    

    def __call__(self, image):

        zoom_factor = np.random.uniform(0.7, 1.2)

        height = image.size[0]

        width = image.size[1]

        new_size = (np.int(zoom_factor*height), np.int(zoom_factor*width))

        return transforms.Resize(new_size)(image)
def my_transform(key="train", plot=False):

    train_sequence = [RandomZoom(),

        transforms.Resize(size=256),

            transforms.CenterCrop(224),

            SegmentPlant(),

            transforms.RandomAffine(30),

            transforms.RandomHorizontalFlip(),

            transforms.RandomVerticalFlip()]

    val_sequence = [transforms.Resize(size=256),

            transforms.CenterCrop(224),

            SegmentPlant()]

    if plot==False:

        train_sequence.extend([

            transforms.ToTensor(),

            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        val_sequence.extend([

            transforms.ToTensor(),

            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        

    data_transforms = {'train': transforms.Compose(train_sequence),'val': transforms.Compose(val_sequence)}

    return data_transforms[key]
fig, ax = plt.subplots(3,6,figsize=(20,11))



train_transform = my_transform(key="train", plot=True)

val_transform = my_transform(key="val", plot=True)



for m in range(6):

    folder = subfolders[m]

    files = listdir(base_path + folder + "/")

    image = Image.open(base_path + folder + "/" + files[0])

    ax[0,m].imshow(image)

    transformed_img = train_transform(image)

    ax[1,m].imshow(transformed_img)

    ax[2,m].imshow(val_transform(image))

    ax[0,m].grid(False)

    ax[1,m].grid(False)

    ax[2,m].grid(False)

    ax[0,m].set_title(folder + "/" + files[0])

    ax[1,m].set_title("Preprocessing for train")

    ax[2,m].set_title("Preprocessing for val")
class SeedlingsDataset(Dataset):

    

    def __init__(self, root_dir, df, transform=None):

        self.root_dir = root_dir

        self.states = df

        self.transform=transform

      

    def __len__(self):

        return len(self.states)

        

    def __getitem__(self, idx):

        image_path = self.root_dir + self.states.species.values[idx] + "/" 

        image_path += self.states.image_name.values[idx]

        image = Image.open(image_path)

        image = image.convert('RGB')

        

        if self.transform:

            image = self.transform(image)

         

        target = self.states.target.values[idx]

        return {"image": image, "label": target}
# Obtain the training data by splitting into 60% train and 40% for the next split

train_idx, sub_test_idx = train_test_split(plantstate.index.values,

                                           test_size=0.4,

                                           random_state=2019,

                                           stratify=plantstate.target.values)



# Split the residual 40% into two parts (each 20% of the original data): 

dev_idx, test_idx = train_test_split(sub_test_idx,

                                     test_size=0.5,

                                     random_state=2019,

                                     stratify=plantstate.loc[sub_test_idx, "target"].values)
BATCH_SIZE = 32
train_df = plantstate.loc[train_idx].copy()

dev_df = plantstate.loc[dev_idx].copy()

test_df = plantstate.loc[test_idx].copy()



train_dataset = SeedlingsDataset(base_path, train_df, transform=my_transform(key="train"))

dev_dataset = SeedlingsDataset(base_path, dev_df, transform=my_transform(key="val"))

test_dataset = SeedlingsDataset(base_path, test_df, transform=my_transform(key="val"))



image_datasets = {"train": train_dataset, "dev": dev_dataset, "test": test_dataset}

dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "dev", "test"]}



print(len(train_dataset), len(dev_dataset), len(test_dataset))



train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

dev_dataloader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)



dataloaders = {"train": train_dataloader, "dev": dev_dataloader, "test": test_dataloader}
def train_loop(model, criterion, optimizer, scheduler=None, num_epochs = 10, lam=0.0):

    since = time.time()



    best_model_wts = copy.deepcopy(model.state_dict())

    best_acc = 0.0

    

    loss_dict = {"train": [], "dev": [], "test": []}

    lam_tensor = torch.tensor(lam, device=device)

    

    running_loss_dict = {"train": [], "dev": [], "test": []}

    

    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        print('-' * 10)



        for phase in ["train", "dev", "test"]:

            if phase == "train":

                model.train()

            else:

                model.eval()



            running_loss = 0.0

            running_corrects = 0

            

            tk0 = tqdm(dataloaders[phase], total=int(len(dataloaders[phase])))



            counter = 0

            for bi, d in enumerate(tk0):

                inputs = d["image"]

                labels = d["label"]

                inputs = inputs.to(device, dtype=torch.float)

                labels = labels.to(device, dtype=torch.long)

                

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

                        

                        #l2_reg = torch.tensor(0., device=device)

                        #for param in model.parameters():

                            #l2_reg = lam_tensor * torch.norm(param)

                        

                        #loss += l2_reg

            

                        optimizer.step()

                        

                        

                        

                # statistics

                running_loss += loss.item() * inputs.size(0)

                running_corrects += torch.sum(preds == labels.data)

                

                if phase == 'train':

                    if scheduler is not None:

                        scheduler.step()

                

                counter += 1

                tk0.set_postfix({'loss': running_loss / (counter * dataloaders[phase].batch_size),

                                 'accuracy': running_corrects.double() / (counter*dataloaders[phase].batch_size)})

                running_loss_dict[phase].append(running_loss / (counter * dataloaders[phase].batch_size))

                

            epoch_loss = running_loss / dataset_sizes[phase]

            loss_dict[phase].append(epoch_loss)

            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(

                phase, epoch_loss, epoch_acc))

            

            # deep copy the model

            if phase == 'dev' and epoch_acc > best_acc:

                best_acc = epoch_acc

                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    

    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(

        time_elapsed // 60, time_elapsed % 60))

    print('Best val Acc: {:4f}'.format(best_acc))              

    

    # load best model weights

    model.load_state_dict(best_model_wts)

    return model, loss_dict, running_loss_dict
listdir("../input/pretrained-pytorch-models/")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

device
model = torchvision.models.resnet18(pretrained=False)

if run_training:

    model.load_state_dict(torch.load("../input/pretrained-pytorch-models/resnet18-5c106cde.pth"))

num_features = model.fc.in_features

print(num_features)



model.fc = nn.Sequential(

    nn.Linear(num_features, 512),

    nn.ReLU(),

    nn.BatchNorm1d(512),

    nn.Dropout(0.5),

    

    nn.Linear(512, 256),

    nn.ReLU(),

    nn.BatchNorm1d(256),

    nn.Dropout(0.5),

    

    nn.Linear(256, NUM_CLASSES))



def init_weights(m):

    if type(m) == nn.Linear:

        torch.nn.init.xavier_uniform_(m.weight)

        m.bias.data.fill_(0.01)



model.apply(init_weights)

model = model.to(device)
weights = compute_class_weight(y=train_df.target.values, class_weight="balanced", classes=train_df.target.unique())    

class_weights = torch.FloatTensor(weights)

if device.type=="cuda":

    class_weights = class_weights.cuda()

print(class_weights)
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.SGD(model.fc.parameters(), lr=0.09, momentum=0.9)

scheduler = CyclicLR(optimizer, base_lr=0.01, max_lr=0.09)
if run_training:

    model, loss_dict, running_loss_dict = train_loop(model, criterion, optimizer, scheduler=scheduler, num_epochs = 10)

    

    if device == "cpu":

        OUTPUT_PATH += ".pth"

    else:

        OUTPUT_PATH += "_cuda.pth"

        

    torch.save(model.state_dict(), OUTPUT_PATH)

    

    losses_df = pd.DataFrame(loss_dict["train"],columns=["train"])

    losses_df.loc[:, "dev"] = loss_dict["dev"]

    losses_df.loc[:, "test"] = loss_dict["test"]

    losses_df.to_csv("losses_segmented_seedlings.csv", index=False)

    

    running_losses_df = pd.DataFrame(running_loss_dict["train"], columns=["train"])

    running_losses_df.loc[0:len(running_loss_dict["dev"])-1, "dev"] = running_loss_dict["dev"]

    running_losses_df.loc[0:len(running_loss_dict["test"])-1, "test"] = running_loss_dict["test"]

    running_losses_df.to_csv("running_losses_segmented_seedlings.csv", index=False)

else:

    if device == "cpu":

        MODEL_PATH += ".pth"

    else:

        MODEL_PATH += "_cuda.pth"

    model.load_state_dict(torch.load(MODEL_PATH))

    model.eval()

    

    losses_df = pd.read_csv(LOSSES_PATH + "losses_segmented_seedlings.csv")

    running_losses_df = pd.read_csv(LOSSES_PATH + "running_losses_segmented_seedlings.csv")
fig, ax = plt.subplots(3,1,figsize=(20,15))



ax[0].plot(running_losses_df["train"], '-o', label="train")

ax[0].set_xlabel("Step")

ax[0].set_ylabel("Weighted x-entropy")

ax[0].set_title("Loss change over steps")

ax[0].legend();



ax[1].plot(running_losses_df["dev"], '-o', label="dev", color="orange")

ax[1].set_xlabel("Step")

ax[1].set_ylabel("Weighted x-entropy")

ax[1].set_title("Loss change over steps")

ax[1].legend();



ax[2].plot(running_losses_df["test"], '-o', label="test", color="mediumseagreen")

ax[2].set_xlabel("Step")

ax[2].set_ylabel("Weighted x-entropy")

ax[2].set_title("Loss change over steps")

ax[2].legend();
plt.figure(figsize=(20,5))

plt.plot(losses_df["train"], '-o', label="train")

plt.plot(losses_df["dev"], '-o', label="dev")

plt.plot(losses_df["test"], '-o', label="test")

plt.xlabel("Epoch")

plt.ylabel("Weighted x-entropy")

plt.title("Loss change over epochs");

plt.legend();
dev_predictions = pd.DataFrame(index = np.arange(0, dataset_sizes["dev"]), columns = ["true", "predicted"])

test_predictions = pd.DataFrame(index = np.arange(0, dataset_sizes["test"]), columns = ["true", "predicted"])



plt.ion()



def imshow(inp, title=None):

    """Imshow for Tensor."""

    inp = inp.numpy().transpose((1, 2, 0))

    mean = np.array([0.485, 0.456, 0.406])

    std = np.array([0.229, 0.224, 0.225])

    inp = std * inp + mean

    inp = np.clip(inp, 0, 1)

    plt.imshow(inp)

    if title is not None:

        plt.title(title)

    plt.pause(0.001)  # pause a bit so that plots are updated



def evaluate_model(model, predictions_df, key):

    was_training = model.training

    model.eval()



    with torch.no_grad():

        for i, data in enumerate(dataloaders[key]):

            inputs = data["image"].to(device)

            labels = data["label"].to(device)

            

            outputs = model(inputs)

            _, preds = torch.max(outputs, 1)

            

            predictions_df.loc[i*BATCH_SIZE:(i+1)*BATCH_SIZE-1, "true"] = data["label"].numpy().astype(np.int)

            predictions_df.loc[i*BATCH_SIZE:(i+1)*BATCH_SIZE-1, "predicted"] = preds.cpu().numpy().astype(np.int)

    predictions_df = predictions_df.dropna()

    return predictions_df
import warnings

warnings.filterwarnings("ignore")



species_map = {0: "Black-grass",

               1: "Charlock",

               2: "Cleavers",

               3: "Common Chickweed",

               4: "Common wheat",

               5: "Fat Hen",

               6: "Loose Silky-bent",

               7: "Maize",

               8: "Scentless Mayweed",

               9: "Shepherd's Purse",

               10: "Small-flowered Cranesbill",

               11: "Sugar beet"}



dev_predictions = evaluate_model(model, dev_predictions, "dev")

dev_predictions.loc[:,"true"] = dev_predictions.loc[:, "true"].astype(np.int)

dev_predictions.loc[:, "predicted"] = dev_predictions.loc[:, "predicted"].astype(np.int)

dev_predictions.loc[:, "true species"] = dev_predictions.loc[:, "true"].map(species_map)

dev_predictions.loc[:, "predicted species"] = dev_predictions.loc[:, "predicted"].map(species_map)

dev_predictions.head()
accuracy_score(dev_predictions.true.values, dev_predictions.predicted.values)
test_predictions = evaluate_model(model, test_predictions, "test")

test_predictions.loc[:,"true"] = test_predictions.loc[:, "true"].astype(np.int)

test_predictions.loc[:, "predicted"] = test_predictions.loc[:, "predicted"].astype(np.int)

test_predictions.loc[:, "true species"] = test_predictions.loc[:, "true"].map(species_map)

test_predictions.loc[:, "predicted species"] = test_predictions.loc[:, "predicted"].map(species_map)

test_predictions.head()
accuracy_score(test_predictions.true.values, test_predictions.predicted.values)
single_accuracies = []

names = []



for key, val in species_map.items():

    y_pred = dev_predictions[dev_predictions.true==key].predicted.values

    y_true = dev_predictions[dev_predictions.true==key].true.values

    

    single_accuracies.append(np.int(accuracy_score(y_true,y_pred)*100))

    names.append(val)



plt.figure(figsize=(20,5))

sns.barplot(x=names, y=single_accuracies)

plt.xticks(rotation=90);

plt.ylabel("%")

plt.title("Single accuracies of plant species classes \n belonging to the dev data");
dev_confusion = confusion_matrix(dev_predictions["true species"].values, dev_predictions["predicted species"].values)

test_confusion = confusion_matrix(test_predictions["true species"].values, test_predictions["predicted species"].values)



fig, ax = plt.subplots(1,2,figsize=(25,10))

sns.heatmap(dev_confusion, annot=True, cmap="Oranges", square=True, cbar=False, linewidths=1, ax=ax[0]);

sns.heatmap(test_confusion, annot=True, cmap="Greens", square=True, cbar=False, linewidths=1, ax=ax[1]);

ax[0].set_title("Confusion matrix of Dev-data");

ax[0].set_xticklabels([name for val, name in species_map.items()], rotation=90)

ax[0].set_yticklabels([name for val, name in species_map.items()], rotation=45)

ax[0].set_xlabel("predicted")

ax[0].set_ylabel("true")

ax[1].set_title("Confusion matrix of Test-data");

ax[1].set_xticklabels([name for val, name in species_map.items()], rotation=90)

ax[1].set_yticklabels([name for val, name in species_map.items()], rotation=45);

ax[1].set_xlabel("predicted")

ax[1].set_ylabel("true");

plt.savefig("Confusion", dpi=200)
accuracy_score(dev_predictions[dev_predictions.true.isin([6,0])==False].true.values,

               dev_predictions[dev_predictions.true.isin([6,0])==False].predicted.values)
fig, ax = plt.subplots(4,6,figsize=(20,14))



for m in range(6):

    folder = subfolders[m]

    files = listdir(base_path + folder + "/")

    image = np.array(Image.open(base_path + folder + "/" + files[0]))

    ax[0,m].imshow(image)

    ax[1,m].imshow(segment_plant(image, my_threshold, my_radius))

    ax[0,m].grid(False)

    ax[1,m].grid(False)

    ax[0,m].set_title(folder + "/" + files[0])

    

    folder = subfolders[m+6]

    files = listdir(base_path + folder + "/")

    image = np.array(Image.open(base_path + folder + "/" + files[0]))

    ax[2,m].imshow(image)

    ax[3,m].imshow(segment_plant(image, my_threshold, my_radius))

    ax[2,m].grid(False)

    ax[3,m].grid(False)

    ax[2,m].set_title(folder + "/" + files[0])