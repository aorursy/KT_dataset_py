import numpy as np

import torch

import pandas as pd

from torchvision import datasets, transforms

from sklearn.model_selection import train_test_split

from torch import nn, optim

import torch.nn.functional as F
# Load data

df_train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

df_test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
# View top of training data

df_train.head()
df_test.head()
# Create the training, validation and test sets

all_train_images = df_train.drop('label', axis=1)

all_train_labels = df_train['label']

test_images = df_test



# Split training data so we have a small validation set and training set

train_images, val_images, train_labels, val_labels = train_test_split(all_train_images, all_train_labels, test_size=0.2)



# Reindex so we can refer to first element using index 0

train_images.reset_index(drop=True, inplace=True)

val_images.reset_index(drop=True, inplace=True)

train_labels.reset_index(drop=True, inplace=True)

val_labels.reset_index(drop=True, inplace=True)

# Size of input images - 28 x 28 pixels

IMG_SIZE = 28

# Create a Dataset to hold the input data and make it available to dataloader

class MyDataset(torch.utils.data.Dataset):

    def __init__(self, images, labels, transforms=None):

        self.X = images

        self.y = labels

        self.transforms = transforms

        

    def __len__(self):

        return (len(self.X))

    

    def __getitem__(self, i):

        data = self.X.iloc[i, :]

        data = np.array(data).astype(np.uint8).reshape(IMG_SIZE, IMG_SIZE, 1)

        

        if self.transforms:

            data = self.transforms(data)

            

            # Also return label if we have it

            if self.y is not None:

                return (data, self.y[i])

            else:

                return data    
# Training Image transformations

train_trans = transforms.Compose([

    transforms.ToPILImage(),

    transforms.ToTensor()

]

)

train_trans
# Validation Image transformations

val_trans = transforms.Compose([

    transforms.ToPILImage(),

    transforms.ToTensor()

]

)

val_trans
# Create data loaders

batch_size = 64



train_dataset = MyDataset(train_images, train_labels, train_trans)

val_dataset = MyDataset(val_images, val_labels, val_trans)

test_dataset = MyDataset(test_images, None, val_trans)



train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
# Create classifier module

class Classifier(nn.Module):

    def __init__(self):

        super().__init__()

        

        self.fc1 = nn.Linear(IMG_SIZE * IMG_SIZE, 256)

        self.fc2 = nn.Linear(256, 128)

        self.fc3 = nn.Linear(128, 64)

        self.fc4 = nn.Linear(64, 10)

        

    def forward(self, x):

        x = x.view(x.shape[0], -1)

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = F.relu(self.fc3(x))

        x = F.log_softmax(self.fc4(x), dim=1)

        return x
# Check that the model is setup correctly

model = Classifier()

model
# Check that the output has the expected size [64, 10] and that the model does not error out when sending input through it

images, labels = next(iter(train_loader))

log_ps = model(images)

assert (log_ps.shape[0] == 64) & (log_ps.shape[1] == 10) # check the size is as expected
# Configure criterion and optimizer

learning_rate = 0.003

criterion = nn.NLLLoss()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# Train the model



epochs = 30



train_losses, val_losses = [], []

for e in range(epochs):

    running_loss = 0

    for images, labels in train_loader:

        # Clear last image's results

        optimizer.zero_grad()

        log_ps = model(images)

        loss = criterion(log_ps, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    else:

        val_loss = 0

        accuracy = 0

        with torch.no_grad():

            for images, labels in val_loader:

                log_ps = model(images)

                val_loss += criterion(log_ps, labels)

                ps = torch.exp(log_ps)

                top_p, top_class = ps.topk(1, dim=1)

                equals = top_class == labels.view(*top_class.shape)

                accuracy += torch.mean(equals.type(torch.FloatTensor))

        

        train_losses.append(running_loss/len(train_loader))

        val_losses.append(val_loss/len(val_loader))

        

        print("Epoch: {}/{}.. ".format(e+1, epochs),

             "Training Loss: {:.3f}.. ".format(running_loss/len(train_loader)),

             "Validation Loss: {:.3f}.. ".format(val_loss/len(val_loader)),

             "Accuracy: {:.3f}".format(accuracy/len(val_loader)))

        
# Plot the losses

%matplotlib inline

%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt



plt.plot(train_losses, label='Training Loss')

plt.plot(val_losses, label='Validation Loss')

plt.legend(frameon=False);
# Function to view image and prediction

def imshow(image, ax=None, title=None, normalize=True):

    """Imshow for Tensor."""

    if ax is None:

        fig, ax = plt.subplots()

    image = image.numpy().transpose((1, 2, 0))



    if normalize:

        mean = np.array([0.485, 0.456, 0.406])

        std = np.array([0.229, 0.224, 0.225])

        image = std * image + mean

        image = np.clip(image, 0, 1)



    ax.set_title(title)

    ax.imshow(image)

    ax.spines['top'].set_visible(False)

    ax.spines['right'].set_visible(False)

    ax.spines['left'].set_visible(False)

    ax.spines['bottom'].set_visible(False)

    ax.tick_params(axis='both', length=0)

    ax.set_xticklabels('')

    ax.set_yticklabels('')

    

    return ax
# Do a manual comparison of images and the predicted labels

images = next(iter(test_loader))

for i in range(20):

    log_ps = model(images)

    ps = torch.exp(log_ps)

    top_p, top_class = ps.topk(1, dim=1)

    imshow(images[i], title=top_class[i].item());
 # Create submission

test_labels = []

with torch.no_grad():

    for images in test_loader:

        log_ps = model(images)

        ps = torch.exp(log_ps)

        top_p, top_class = ps.topk(1, dim=1)

        for p in top_class:

            test_labels.append(p.item())



#Look at top 10 labels

test_labels[:10]
submission_df = pd.DataFrame()

submission_df['ImageId'] = range(1, len(test_labels) + 1)

submission_df['Label'] = test_labels

submission_df.head()
submission_df.to_csv('/kaggle/submission2.csv', index=False)