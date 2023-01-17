import numpy as np

import pandas as pd

import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder



import torch

from torch import nn, optim

import torch.nn.functional as F

import torchvision.transforms as transforms

from torch.autograd import Variable
def plot_images(df, n, label):

    for i in range(n**2):

        random_value = np.random.randint(len(df[label]))

        plt.subplot(n, n, i+1)

        X = df.iloc[random_value, :-1].values.astype("int").reshape(32, 32)

        plt.imshow(X, cmap="binary")

        plt.title(str(df[label][random_value]), color="green", fontsize=28)

        plt.axis("off")

        plt.subplots_adjust(left=1, right=4.5, bottom=.5, top=2.5)

    plt.figure(figsize=(10, 10))

    plt.show()
input_dir = "../input/devanagari-character-set/"

df = pd.read_csv(input_dir+"data.csv")
df.head()
plot_images(df, 4, "character")
targets_numpy = df.character.values

features_numpy = df.iloc[:, :-1].values/255.0
le = LabelEncoder()

targets_encoded = le.fit_transform(targets_numpy)

features_train, features_test, targets_train, targets_test = train_test_split(features_numpy, targets_encoded, test_size=0.2, random_state=123)
def data_loader(features,labels, batch_size):

  feature = torch.from_numpy(features)

  target = torch.from_numpy(labels)



  data = torch.utils.data.TensorDataset(feature, target)



  dataloaded = torch.utils.data.DataLoader(data, batch_size = batch_size, shuffle = False)



  return dataloaded
train_loader = data_loader(features_train, targets_train, 128)

test_loader = data_loader(features_test, targets_test, 128)
class ANNModel(nn.Module):

    def __init__(self):

        super(ANNModel, self).__init__()

        self.fc1 = nn.Linear(32*32, 512)         

        self.fc2 = nn.Linear(512, 256)        

        self.fc3 = nn.Linear(256, 128)        

        self.fc4 = nn.Linear(128, 64)

        self.fc5 = nn.Linear(64, 46)



        self.dropout = nn.Dropout(p=0.2)

        self.log_softmax = F.log_softmax

    

    def forward(self, x):

      x = self.dropout(F.relu(self.fc1(x)))

      x = self.dropout(F.relu(self.fc2(x)))

      x = self.dropout(F.relu(self.fc3(x)))

      x = self.dropout(F.relu(self.fc4(x)))



      x = self.log_softmax(self.fc5(x), dim=1)

      return x
model = ANNModel()

# Define our loss function

criterion = nn.CrossEntropyLoss()

# Define the optimier

optimizer = optim.Adam(model.parameters(), lr=0.0015)



epochs = 25

steps = 0

print_every = 50

train_losses, test_losses = [], []
for e in range(epochs):

  running_loss = 0

  for images, labels in train_loader:

    steps += 1

    optimizer.zero_grad()

    prediction = model(images.float())

    loss = criterion(prediction, labels)

    loss.backward()

    optimizer.step()



    running_loss += loss.item()

    if steps % print_every == 0:

      test_loss = 0

      accuracy = 0



      with torch.no_grad():

        model.eval()

        for images, labels in test_loader:

          prediction_test = model(images.float())

          test_loss += criterion(prediction_test, labels)



          prediction_test = torch.exp(prediction_test)



          top_k, top_class = prediction_test.topk(1, dim=1)

          equals = (top_class == labels.view(*top_class.shape))

          accuracy += torch.mean(equals.type(torch.FloatTensor))



      model.train()

      train_losses.append(running_loss/len(train_loader))

      test_losses.append(test_loss/len(test_loader))



      print("Epoch: {}/{}.. ".format(e+1, epochs),

            "Training Loss: {:.3f}.. ".format(train_losses[-1]),

            "Test Loss: {:.3f}.. ".format(test_losses[-1]),

            "Test Accuracy: {:.3f}.. ".format(accuracy/len(test_loader)))

plt.plot(train_losses, label='Training loss')

plt.plot(test_losses, label='Validation loss')

plt.legend(frameon=False)
classes = list(le.classes_)



def make_predictions(data, number):

  images, labels = next(iter(data))

  img = images[number].view(1, 32*32)

  with torch.no_grad():

    logpreds = model(img.float())

  pred = torch.exp(logpreds)



  pred = pred.data.numpy().squeeze()

  fig, (ax1, ax2) = plt.subplots(figsize=(8, 8), ncols=2)

  ax1.imshow(img.reshape(32, 32), cmap="binary")

  ax1.axis("off")

  ax2.barh(np.arange(46), pred)

  ax2.set_aspect(0.1)

  ax2.set_yticks(np.arange(46))

  ax2.set_yticklabels(classes)

  ax2.set_title('Highest Probability Of a class')

  ax2.set_xlim(0, 1.1)



  plt.tight_layout()
make_predictions(test_loader, 44)