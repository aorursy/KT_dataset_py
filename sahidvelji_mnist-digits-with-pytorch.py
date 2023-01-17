import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import time

import random



from sklearn.model_selection import train_test_split



import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torch.optim.lr_scheduler import ExponentialLR

from torch.utils.data import DataLoader, Dataset



# For data augmentation

from albumentations import Compose, ShiftScaleRotate, ToFloat

from albumentations.pytorch import ToTensorV2

import cv2



import plotly.express as px

px.defaults.template = 'plotly_white'

px.defaults.color_discrete_sequence = ['steelblue']



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
def set_seed(seed=0):

    np.random.seed(seed)

    random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)

    

set_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

sample_sub = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')
train.head()
test.head()
sample_sub.head()
label_counts = (train.label

                .value_counts()

                .to_frame()

                .reset_index()

                .rename(columns={'index': 'Label', 'label': 'Count'})

                .sort_values('Label')

               )

fig = px.bar(label_counts, x='Label', y='Count')

fig.update_traces(textposition='outside',

                  texttemplate='%{y}',

                  cliponaxis=False,

                  hovertemplate=

                  'Label: <b>%{x}</b><br>'+

                  'Count: <b>%{y}</b>'

                 )

fig.update_layout(title='Distribution of labels',

                  yaxis_title='Count',

                  xaxis_title='Label',

                  xaxis_type='category',

                  yaxis_tickformat=',',

                  hoverlabel_bgcolor="white",

                  hoverlabel_font_size=14,

                  hovermode="x"

                 )

fig.show()
X = train.drop(columns='label').values.reshape(-1, 28, 28, 1)

y = train.label.values



train_X, valid_X, train_y, valid_y = train_test_split(X,

                                                      y,

                                                      test_size=0.2

                                                     )
class MNISTDataset(Dataset):

    def __init__(self, X, y=None, is_test=False, transforms=None):

        self.X = X

        self.y = y

        self.is_test = is_test

        self.transforms = transforms



    def __len__(self):

        return len(self.X)



    def __getitem__(self, index):

        image = self.X[index]

        

        if self.transforms:

            image = self.transforms(image=image)['image']

            

        if self.is_test:

            return image

        else:

            return image, self.y[index]
train_transforms = Compose([ToFloat(max_value=255),

                            ShiftScaleRotate(shift_limit=0.1,

                                             scale_limit=0.1,

                                             rotate_limit=10,

                                             border_mode=cv2.BORDER_CONSTANT,

                                             value=0,

                                             p=1),

                            ToTensorV2()

                           ])

valid_transforms = Compose([ToFloat(max_value=255),

                            ToTensorV2()

                           ])
fig = plt.figure(figsize=(12, 2))

for k in range(10):

    idx = random.choice(train.label[train.label.eq(k)].index)

    image, label = train.drop(columns='label').iloc[idx].values.reshape(28, 28), k

    

    ax = plt.subplot(2, 10, k+1)

    ax.title.set_text(label)

    plt.axis('off')

    plt.imshow(image, cmap='gray')

    

    image = np.expand_dims(image, 2)

    image = train_transforms(image=image)['image']

    image = image.squeeze(0)

    ax = plt.subplot(2, 10, 10+k+1)

    plt.axis('off')

    plt.imshow(image, cmap='gray')
example_transforms = Compose([ToFloat(max_value=255),

                            ShiftScaleRotate(shift_limit=0.3,

                                             scale_limit=0.3,

                                             rotate_limit=30,

#                                              border_mode=cv2.BORDER_CONSTANT,

#                                              value=0,

                                             p=1),

                            ToTensorV2()

                           ])
fig = plt.figure(figsize=(12, 2))

for k in range(10):

    idx = random.choice(train.label[train.label.eq(k)].index)

    image, label = train.drop(columns='label').iloc[idx].values.reshape(28, 28), k

    

    ax = plt.subplot(2, 10, k+1)

    ax.title.set_text(label)

    plt.axis('off')

    plt.imshow(image, cmap='gray')

    

    image = np.expand_dims(image, 2)

    image = example_transforms(image=image)['image']

    image = image.squeeze(0)

    ax = plt.subplot(2, 10, 10+k+1)

    plt.axis('off')

    plt.imshow(image, cmap='gray')
class CNN(nn.Module):

    def __init__(self):

        super(CNN, self).__init__()

        self.dropout = nn.Dropout2d(0.5)

        

        self.bn1_1 = nn.BatchNorm2d(48)

        self.bn1_2 = nn.BatchNorm2d(48)



        self.bn2 = nn.BatchNorm1d(256)

        

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=48, kernel_size=3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, stride=2, padding=1)

        

        self.fc1 = nn.Linear(in_features=48 * 14 * 14, out_features=256)

        self.fc2 = nn.Linear(in_features=256, out_features=10)

        

    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = self.bn1_1(x)

        x = F.relu(self.conv2(x))

        x = self.bn1_2(x)

        x = self.dropout(x)

        

        x = torch.flatten(x, 1)

        

        x = F.relu(self.fc1(x))

        x = self.bn2(x)

        x = self.dropout(x)

        

        x = self.fc2(x)

        return x
class EarlyStopping:

    def __init__(self, mode, path, patience=3, delta=0):

        if mode not in {'min', 'max'}:

            raise ValueError("Argument mode must be one of 'min' or 'max'.")

        if patience <= 0:

            raise ValueError("Argument patience must be a postive integer.")

        if delta < 0:

            raise ValueError("Argument delta must not be a negative number.")

            

        self.mode = mode

        self.patience = patience

        self.delta = delta

        self.path = path

        self.best_score = np.inf if mode == 'min' else -np.inf

        self.counter = 0

        

    def _is_improvement(self, val_score):

        """Return True iff val_score is better than self.best_score."""

        if self.mode == 'max' and val_score > self.best_score + self.delta:

            return True

        elif self.mode == 'min' and val_score < self.best_score - self.delta:

            return True

        return False

        

    def __call__(self, val_score, model):

        """Return True iff self.counter >= self.patience.

        """

        

        if self._is_improvement(val_score):

            self.best_score = val_score

            self.counter = 0

            torch.save(model.state_dict(), self.path)

            print('Val loss improved. Saved model.')

            return False

        else:

            self.counter += 1

            print(f'Early stopping counter: {self.counter}/{self.patience}')

            if self.counter >= self.patience:

                print(f'Stopped early. Best val loss: {self.best_score:.4f}')

                return True
def train_one_epoch(model, train_loader, optimizer, device, criterion):

    """Train model for one epoch and return the mean train_loss."""

    model.train()

    running_loss_train = 0

    for inputs, labels in train_loader:

        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss_train += loss.item()

    train_loss = running_loss_train / len(train_loader.dataset)

    return train_loss
def validate(model, valid_loader, device, criterion):

    """Validate model and return the accuracy and mean loss."""

    model.eval()

    correct = 0

    running_loss_val = 0

    with torch.no_grad():

        for inputs, labels in valid_loader:

            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            pred = outputs.argmax(dim=1)

            correct += pred.eq(labels).sum().item()

            running_loss_val += loss.item()

    val_acc = correct / len(valid_loader.dataset)

    val_loss = running_loss_val / len(valid_loader.dataset)

    return val_acc, val_loss
def fit(model, train_loader, valid_loader, learning_rate, num_epochs):

    criterion = nn.CrossEntropyLoss(reduction='sum')

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    es = EarlyStopping(mode='min', path='model.pth', patience=5)

    model = model.to(device)

    scheduler = ExponentialLR(optimizer, gamma=0.9)



    for epoch in range(1, num_epochs + 1):

        train_loss = train_one_epoch(model, train_loader, optimizer, device, criterion)

        val_acc, val_loss = validate(model, valid_loader, device, criterion)

        scheduler.step()

        print(f'Epoch {epoch:2}/{num_epochs}',

              f'train loss: {train_loss:.4f}',

              f'val loss: {val_loss:.4f}',

              f'val acc: {val_acc:.2%}',

              sep=' | '

             )

        if es(val_loss, model):

            break
TRAIN_BATCH_SIZE = 64

VALID_BATCH_SIZE = 512

NUM_EPOCHS = 50

LEARNING_RATE = 1e-3

NUM_WORKERS = 0



mnist_train = MNISTDataset(train_X, train_y, is_test=False, transforms=train_transforms)

mnist_valid = MNISTDataset(valid_X, valid_y, is_test=False, transforms=valid_transforms)



train_loader = DataLoader(mnist_train, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)

valid_loader = DataLoader(mnist_valid, batch_size=VALID_BATCH_SIZE, shuffle=False)



model = CNN()

start = time.time()

fit(model, train_loader, valid_loader, learning_rate=LEARNING_RATE, num_epochs=NUM_EPOCHS)

print(f'Total training time: {time.time() - start}')

model.load_state_dict(torch.load('model.pth'))
TEST_BATCH_SIZE = 512



test_transforms = Compose([ToFloat(max_value=255),

                           ToTensorV2()

                          ])



test_X = test.values.reshape(-1, 28, 28, 1)

mnist_test = MNISTDataset(test_X, is_test=True, transforms=test_transforms)

test_loader = DataLoader(mnist_test, batch_size=TEST_BATCH_SIZE, shuffle=False)
def predict(model, test_loader, device):

    """Make predictions on the test data and return 

    the submission data frame.

    """

    

    model.eval()

    predictions = sample_sub['Label'].values

    with torch.no_grad():

        for i, inputs in enumerate(test_loader):

            inputs = inputs.to(device)

            outputs = model(inputs)

            pred = outputs.argmax(dim=1).to('cpu').numpy()

            predictions[i*TEST_BATCH_SIZE:i*TEST_BATCH_SIZE+len(inputs)] = pred

    

    output = sample_sub.copy()

    output['Label'] = predictions

    output.to_csv('submission.csv', index=False)

    return output



output = predict(model, test_loader, device)

output
fig = plt.figure(figsize=(12, 10))

for k in range(10):

    indices = output[output.Label.eq(k)].sample(10).index

    for j, idx in enumerate(indices):

        image, label = test_X[idx], output.loc[idx, 'Label'].item()

        image = image.squeeze(2)

        ax = plt.subplot(10, 10, 10*k+j+1)

        ax.title.set_text(label)

        plt.axis('off')

        plt.imshow(image, cmap='gray')



plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.6)
prediction_counts = (output.Label

                     .value_counts()

                     .to_frame()

                     .reset_index()

                     .rename(columns={'index': 'Prediction', 'Label': 'Count'})

                     .sort_values('Prediction')

                    )

fig = px.bar(prediction_counts, x='Prediction', y='Count')

fig.update_traces(textposition='outside',

                  texttemplate='%{y}',

                  cliponaxis=False,

                  hovertemplate=

                  'Prediction: <b>%{x}</b><br>'+

                  'Count: <b>%{y}</b>'

                 )

fig.update_layout(title='Distribution of predictions',

                  yaxis_title='Count',

                  xaxis_title='Prediction',

                  xaxis_type='category',

                  yaxis_tickformat=',',

                  hoverlabel_bgcolor="white",

                  hoverlabel_font_size=14,

                  hovermode="x"

                 )

fig.show()