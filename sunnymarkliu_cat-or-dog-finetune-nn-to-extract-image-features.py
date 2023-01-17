import os

from PIL import Image

import pandas as pd

import torch

import torch.nn as nn

import torch.optim as optim

from torch.optim import lr_scheduler

import numpy as np

import torchvision

from torch.autograd import Variable

from torchvision import datasets, models, transforms

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

import time

import copy

plt.ion()   # interactive mode



import time

from tqdm import tqdm, trange

tqdm.pandas()



import matplotlib.pyplot as plt

%matplotlib inline



import os

from shutil import copyfile

print(os.listdir("../input"))
train_df = pd.read_csv('../input/train/train.csv')

test_df = pd.read_csv('../input/test/test.csv')

test_df['AdoptionSpeed'] = [-1] * len(test_df)

data_df = pd.concat([train_df, test_df], axis=0).reset_index()

print(train_df.shape[0], test_df.shape[0], data_df.shape[0])
data_df.head(2)
def pil_loader(path):

    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)

    with open(path, 'rb') as f:

        img = Image.open(f)

        return img.convert('RGB')





def accimage_loader(path):

    import accimage

    try:

        return accimage.Image(path)

    except IOError:

        # Potentially a decoding problem, fall back to PIL.Image

        return pil_loader(path)





def default_loader(path):

    from torchvision import get_image_backend

    if get_image_backend() == 'accimage':

        return accimage_loader(path)

    else:

        return pil_loader(path)



class DogCatDataset(Dataset):

    """Dog Cat classify dataset."""

    

    def __init__(self, data_df, root_dir='../input/', train_or_valid='train', transform=None):

        super(DogCatDataset, self).__init__()

        self.classes = ['dog', 'cat']

        self.class_to_idx = {'dog':0, 'cat':1}

        

        self.transform = transform

        self.img_list = [] # read train/valid image path

        petids = data_df['PetID'].values

        for petid in tqdm(petids):

            row = data_df.loc[data_df['PetID'] == petid, :]

            anim_type = 'cat' if row['Type'].values[0] == 2 else 'dog'

            photo_amt = row['PhotoAmt'].values[0]

            img_type = 'train' if row['AdoptionSpeed'].values[0] >= 0 else 'test'

            

            if train_or_valid == 'train':

                for i in range(2, int(photo_amt) + 1):

                    img_path = f'{root_dir}{img_type}_images/{petid}-{i}.jpg'

                    if not os.path.exists(img_path): continue

                    self.img_list.append((img_path, self.class_to_idx[anim_type]))

            else:  # valid

                img_path = f'{root_dir}{img_type}_images/{petid}-1.jpg'

                if not os.path.exists(img_path): continue

                self.img_list.append((img_path, self.class_to_idx[anim_type]))

    

    def __len__(self):

        return len(self.img_list)

    

    def __getitem__(self, index):

        path, target = self.img_list[index]

        image = default_loader(path)

        if self.transform is not None:

            image = self.transform(image)

        return image, target

        
batch_size = 64



image_transforms = {

    'train': transforms.Compose([

        transforms.Resize(256),

        transforms.CenterCrop(224),

        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ]),

    'valid': transforms.Compose([

        transforms.Resize(256),

        transforms.CenterCrop(224),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ]),

}



image_datasets = {x: DogCatDataset(data_df, train_or_valid=x, transform=image_transforms[x])

                  for x in ['train', 'valid']}



dataloaders = {x: torch.utils.data.dataloader.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True)#, num_workers=4)

               for x in ['train', 'valid']}
print('Train:', len(image_datasets['train']), ', Valid:', len(image_datasets['valid']))

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}

class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('class:', class_names)

print('device:', device)
def imshow(inp, title=None):

    """Imshow for Tensor."""

    inp = inp.numpy().transpose((1, 2, 0))

    mean = np.array([0.485, 0.456, 0.406])

    std = np.array([0.229, 0.224, 0.225])

    inp = std * inp + mean

    inp = np.clip(inp, 0, 1)

    plt.figure(figsize=(16, 6))

    plt.imshow(inp)

    if title is not None:

        plt.title(title)

    plt.show()
# Get a batch of training data

inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch

out = torchvision.utils.make_grid(inputs[:4])

imshow(out, title=[class_names[x] for x in classes[:4]])
model = models.resnet18(pretrained=True)

fc_in_features = model.fc.in_features

model.fc = nn.Linear(fc_in_features, 2)

model = model.to(device)



loss_fn = nn.CrossEntropyLoss()



# Observe that all parameters are being optimized

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
epochs = 6



best_valid_loss = np.inf

best_valid_acc = 0.

best_model_wts = copy.deepcopy(model.state_dict())

didnt_improve_count = 0



for epoch in range(epochs):

    start_time = time.time()

    # set train mode

    model.train()

    avg_train_loss = 0.

    train_corrects = 0.

    

    for x_batch, y_batch in dataloaders['train']:

        x_batch = x_batch.to(device)

        y_batch = y_batch.to(device)

        

        y_pred = model(x_batch)

        loss = loss_fn(y_pred, y_batch)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        avg_train_loss += loss.item() / len(dataloaders['train'])

        

        _, y_pred = torch.max(y_pred, 1)

        train_corrects += torch.sum(y_pred == y_batch.data).double()

    

    train_acc = train_corrects / dataset_sizes['train']

    

    torch.cuda.empty_cache()

    

    model.eval()

    avg_val_loss = 0.

    valid_corrects = 0.

    for x_batch, y_batch in dataloaders['valid']:

        with torch.no_grad():

            x_batch = x_batch.to(device)

            y_batch = y_batch.to(device)

        

            y_pred = model(x_batch)

            loss = loss_fn(y_pred, y_batch)

            avg_val_loss += loss.item() / len(dataloaders['valid'])

        

            _, y_pred = torch.max(y_pred, 1)

            valid_corrects += torch.sum(y_pred == y_batch.data).double()

    

    valid_acc = valid_corrects / dataset_sizes['valid']

    

    elapsed_time = time.time() - start_time 

    print('Epoch {}/{}  train-loss={:.4f}  train-acc={:.4f}  val_loss={:.4f}  valid-acc={:.4f}  time={:.2f}s'.format(

        epoch + 1, epochs, avg_train_loss, train_acc, avg_val_loss, valid_acc, elapsed_time))

    

    # deep copy the model

    if avg_val_loss < best_valid_loss:

        best_valid_loss = avg_val_loss

        best_valid_acc = valid_acc

        didnt_improve_count = 0

        best_model_wts = copy.deepcopy(model.state_dict())

    else:

        didnt_improve_count += 1

        if didnt_improve_count > 2:

            break

    

print('Best valid-loss={:.4f} \t valid-acc={:.4f}'.format(best_valid_loss, best_valid_acc))

print('save and load best model weights')

model.load_state_dict(best_model_wts)

torch.save(model.state_dict(), 'best_resnet18_weights.model')
# Get a batch of training data

inputs, classes = next(iter(dataloaders['train']))

inputs, classes = inputs[:4], classes[:4]

ground_truth = [class_names[i] for i in classes]

# Make a grid from batch

out = torchvision.utils.make_grid(inputs)



inputs = inputs.cuda()

preds = model(inputs)

_, preds = torch.max(preds, 1)

predict_class = [class_names[i] for i in preds]

imshow(out, title=f"Truth : {ground_truth}\nPredict: {predict_class}")
image_features = []

def hook_feature(module, input, output):

    # hook the feature extractor

    image_features.append(np.squeeze(output.data.cpu().numpy()))



model._modules.get('avgpool').register_forward_hook(hook_feature)
extract_transform = image_transforms['valid']
train_pids = train_df.PetID.values

input_tensor = torch.zeros(1, 3, 224, 224)



train_image_features = {}

for petid in tqdm(train_pids):

    train_img = f"../input/train_images/{petid}-1.jpg"

    if not os.path.exists(train_img): continue

    

    train_img = Image.open(train_img)

    train_img = extract_transform(train_img)

    input_tensor[0, :, :, :] = train_img

    input_tensor = input_tensor.cuda()

    model(input_tensor)

    train_image_features[petid] = image_features[0]

    image_features.clear()

train_image_features = pd.DataFrame.from_dict(train_image_features, orient='index')

train_image_features.columns = [f'img_nn_feat{idx}' for idx in train_image_features.columns.values]

train_image_features = train_image_features.reset_index().rename(columns={'index':'PetID'})
train_image_features.head()
train_image_features.to_csv('train_image_features.csv', index=False)
test_pids = test_df.PetID.values

input_tensor = torch.zeros(1, 3, 224, 224)



test_image_features = {}

for petid in tqdm(test_pids):

    test_img = f"../input/test_images/{petid}-1.jpg"

    if not os.path.exists(test_img): continue

    

    test_img = Image.open(test_img)

    test_img = extract_transform(test_img)

    input_tensor[0, :, :, :] = test_img

    input_tensor = input_tensor.cuda()

    model(input_tensor)

    test_image_features[petid] = image_features[0]

    image_features.clear()

test_image_features = pd.DataFrame.from_dict(test_image_features, orient='index')

test_image_features.columns = [f'img_nn_feat{idx}' for idx in test_image_features.columns.values]

test_image_features = test_image_features.reset_index().rename(columns={'index':'PetID'})
test_image_features.head()
test_image_features.to_csv('test_image_features.csv', index=False)