# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

print(os.path.isdir("../input/train"))

print(os.listdir("../input/train/train")[:10])



# Any results you write to the current directory are saved as output.
from IPython.display import Image

cat_img = Image(filename='../input/train/train/cat.0.jpg')

cat_img
dog_img = Image(filename='../input/train/train/dog.0.jpg')

dog_img
from skimage import io

import matplotlib.pyplot as plt
plt.imshow(io.imread('../input/train/train/dog.1.jpg'))
dog_img1 = io.imread('../input/train/train/dog.1.jpg')
dog_img1.shape
from torch.utils.data import Dataset, DataLoader
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

device
class ImageDataset(Dataset):

    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir

        self.images = os.listdir(root_dir)

        self.transform = transform



    def __len__(self):

        return len(self.images)



    def __getitem__(self, idx):

        filename = self.images[idx]

        image = io.imread(os.path.join(self.root_dir, filename))

        label = 1 if "dog" in filename else 0

        sample = {'image': image, 'label': label, 'filename': filename }

        if self.transform:

            sample = self.transform(sample)

        return sample
def label_to_str(label):

    if label == 1:

        return "dog"

    return "cat"
ds = ImageDataset('../input/train/train')
print(len(ds))
fig = plt.figure()

for i in range(4):

    sample = ds[i]

    species = label_to_str(sample['label'])

    print(i, sample['image'].shape, species, sample['filename'])

    ax = plt.subplot(1, 4, i + 1)

    plt.tight_layout()

    ax.set_title(f"Sample {i} - {species}")

    ax.axis("off")

    plt.imshow(sample['image'])

plt.show()
# These classes have been stolen and lightly rejigged from

# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

from skimage import transform

import torch

from torchvision import transforms



class Rescale(object):

    """Rescale the image in a sample to a given size.



    Args:

        output_size (tuple or int): Desired output size. If tuple, output is

            matched to output_size. If int, smaller of image edges is matched

            to output_size keeping aspect ratio the same.

    """



    def __init__(self, output_size):

        assert isinstance(output_size, (int, tuple))

        self.output_size = output_size



    def __call__(self, sample):

        image = sample['image']



        h, w = image.shape[:2]

        if isinstance(self.output_size, int):

            if h > w:

                new_h, new_w = self.output_size * h / w, self.output_size

            else:

                new_h, new_w = self.output_size, self.output_size * w / h

        else:

            new_h, new_w = self.output_size



        new_h, new_w = int(new_h), int(new_w)



        img = transform.resize(image, (new_h, new_w))



        return {'image': img, 'label': sample['label'], 'filename': sample['filename']}





class RandomCrop(object):

    """Crop randomly the image in a sample.



    Args:

        output_size (tuple or int): Desired output size. If int, square crop

            is made.

    """



    def __init__(self, output_size):

        assert isinstance(output_size, (int, tuple))

        if isinstance(output_size, int):

            self.output_size = (output_size, output_size)

        else:

            assert len(output_size) == 2

            self.output_size = output_size



    def __call__(self, sample):

        image = sample['image']



        h, w = image.shape[:2]

        new_h, new_w = self.output_size



        top = np.random.randint(0, h - new_h)

        left = np.random.randint(0, w - new_w)



        image = image[top: top + new_h,

                      left: left + new_w]



        return {'image': image, 'label': sample['label'], 'filename': sample['filename'] }



class Normalize(object):

    def __init__(self, *args, **kwargs):

        self.inner = transforms.Normalize(*args, **kwargs)

    

    def __call__(self, sample):

        image = sample['image']

        image = self.inner(image)

        return {'image': image, 'label': sample['label'], 'filename': sample['filename']}



class ToTensor(object):

    """Convert ndarrays in sample to Tensors."""



    def __call__(self, sample):

        image = sample['image']



        # swap color axis because

        # numpy image: H x W x C

        # torch image: C X H X W

        image = image.transpose((2, 0, 1))

        return {'image': torch.from_numpy(image).to(torch.float32),

                'label': sample['label'],

                'filename': sample['filename']}

scale = Rescale(256)

crop = RandomCrop(224)

# as expected by pretrained models, from https://pytorch.org/docs/stable/torchvision/models.html

normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

composed = transforms.Compose([scale, crop, ToTensor(), normalize])
from torch.utils.data.dataset import random_split

transformed_dataset = ImageDataset('../input/train/train', transform=composed)

num_images = len(transformed_dataset)

num_test = num_images // 10

num_train = num_images - 2 * num_test

train, test, validate = random_split(transformed_dataset, [num_train, num_test, num_test])



def make_dl(dataset):

    return DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)



dataloaders = {'train': make_dl(train), 'test': make_dl(test), 'validate': make_dl(validate)}
from torch import nn



model = nn.Sequential(

          nn.Conv2d(3, 20, 5),

          nn.MaxPool2d(2, 2),

          nn.Conv2d(20, 64, 5),

          nn.Linear(64, 2)

        )
params_to_update = model.parameters()

print("Params to learn:")

for name,param in model.named_parameters():

    if param.requires_grad == True:

        print("\t",name, param.size())
model.train()

optimizer = torch.optim.SGD(params_to_update, lr=0.001, momentum=0.9)
criterion = torch.nn.CrossEntropyLoss()
%%time

num_epochs = 5

model = model.to(device)

model.train()



for i in range(num_epochs):

    for phase in ['train', 'test']:

        phase_loss = 0

        for j, batch in enumerate(dataloaders[phase]):

            optimizer.zero_grad()

            inputs = batch['image'].to(device)

            outputs = model(inputs)

            labels = batch['label'].to(device)

            loss = criterion(outputs, labels)

            phase_loss += loss.item()

            if phase == 'train':

                loss.backward()

                optimizer.step()

        print(f"Epoch {i}, {phase} loss = {phase_loss / j}")
torch.save(model.state_dict(), 'model.pt')

# to load, use

#   model = torch.load(PATH)

#   model.eval()
model.eval()

num_correct = 0

total = len(validate)

for j, batch in enumerate(dataloaders['validate']):

    inputs = batch['image'].to(device)

    labels = batch['label'].to(device)

    outputs = model(inputs)

    best_guesses = outputs.argmax(1)

    num_correct += (labels == best_guesses).sum()

print(f"{num_correct} correct out of {total}: success rate {100 * num_correct / total}%")
test_ds = ImageDataset('../input/test1/test1', transform=composed)

test_dl = DataLoader(test_ds, batch_size=1)
import pathlib



predictions = []

for j, image in enumerate(test_dl):

    inputs = image['image'].to(device)

    outputs = model(inputs)

    best_guesses = outputs.argmax(1)

    file_id = int(pathlib.Path(image['filename'][0]).stem)

    predictions.append({'id': file_id, 'label': best_guesses.item()})
df = pd.DataFrame(predictions)

df.to_csv('submission.csv')

df.head()