from IPython.display import Image

Image(filename="../input/sign-language-mnist/amer_sign2.png")
import pandas as pd

train_df = pd.read_csv("../input/sign-language-mnist/sign_mnist_train/sign_mnist_train.csv")

test_df = pd.read_csv("../input/sign-language-mnist/sign_mnist_test/sign_mnist_test.csv")



train_df.head()
train_labels = train_df["label"].values

test_labels = test_df["label"].values



train_pixels = train_df.drop("label", axis=1).values.astype("float32")

test_pixels = test_df.drop("label", axis=1).values.astype("float32")
print("training images shape", train_pixels.shape)

print("training targets shape ",train_labels.shape)

print("test images shape", test_pixels.shape)

print("test targets shape", test_labels.shape)
train_images = train_pixels.reshape(train_pixels.shape[0], 1, 28, 28)

test_images = test_pixels.reshape(test_pixels.shape[0], 1, 28, 28)



print("training images shape", train_images.shape)

print("test images shape", test_images.shape)
import matplotlib.pyplot as plt



train_images = train_images

test_images = test_images



image = train_images[0].squeeze()

label = train_labels[0]



plt.title(f"label {label}")

plt.imshow(image, cmap="gray")

plt.show()
import torch



train_image_tensor = torch.tensor(train_images) / 255.0

test_image_tensor  = torch.tensor(test_images) / 255.0

train_label_tensor = torch.tensor(train_labels)

test_label_tensor  = torch.tensor(test_labels)
from torch.utils.data import TensorDataset

train_set = TensorDataset(train_image_tensor, train_label_tensor)

test_set = TensorDataset(test_image_tensor, test_label_tensor)
from torch.utils.data import DataLoader



batch_size = 16

num_workers = 2



train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
from torchvision.utils import make_grid

import numpy as np



def show(img):

    npimg = img.numpy()

    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')

    

images, labels = next(iter(train_loader))

grid = make_grid(images, nrow=4)

show(grid)
import torch.nn as nn



modal = nn.Sequential(

    nn.Conv2d(1, 32, 5),

    nn.ReLU(),

    nn.Dropout2d(0.4),



    nn.MaxPool2d(2, 2),

    nn.BatchNorm2d(32),



    nn.Conv2d(32, 64, 5),

    nn.ReLU(),

    nn.Dropout2d(0.4),



    nn.MaxPool2d(2, 2),

    nn.BatchNorm2d(64),



    nn.Flatten(start_dim=1),

    

    nn.Linear(1024, 128),

    nn.ReLU(),

    nn.Dropout(0.4),

    

    nn.BatchNorm1d(128),



    nn.Linear(128, 26)

)



modal
import torch.optim as optim



learning_rate = 0.001



criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(modal.parameters(), lr = learning_rate)
%%time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

modal.to(device)



total_train_loss = []

total_val_loss = []



epochs = 10

print(f"training on {device}")

for epoch in range(epochs):

    train_loss = 0

    val_loss = 0

    

    modal.train()

    for images, labels in train_loader:

        images, labels = images.to(device=device), labels.to(device=device)

        optimizer.zero_grad()

        preds = modal(images)

        loss = criterion(preds, labels)

        loss.backward()

        optimizer.step()

        

        train_loss += loss.item() * train_loader.batch_size

        

    modal.eval()

    with torch.no_grad():

        for images, labels in test_loader:

            images, labels = images.to(device=device), labels.to(device=device)

            preds = modal(images)

            loss = criterion(preds, labels)

            

            val_loss += loss.item() * test_loader.batch_size

    

    total_train_loss.append(train_loss / len(train_loader))

    total_val_loss.append(val_loss / len(test_loader))

    

    print(

        f"epoch: {epoch+1}/{epochs} train_loss: {total_train_loss[-1]} val_loss: {total_val_loss[-1]}"

    )
plt.plot(total_train_loss, label="train loss")

plt.plot(total_val_loss, label="val loss")

plt.xlabel("epoch")

plt.ylabel("loss")

plt.legend()

plt.show()
def no_of_correct(preds, targets):

    return targets.eq(preds.argmax(dim=1)).sum().item()



total_correct = 0

modal.eval()

with torch.no_grad():

    for images, labels in test_loader:

        images, labels = images.to(device=device), labels.to(device=device)

        preds = modal(images)

        total_correct += no_of_correct(preds, labels)



print(f"{total_correct}/{len(test_set)} correct Accuracy: {(total_correct/len(test_set))*100:.3f}")