!pip install efficientnet_pytorch

!pip install torchsummary
import os

import random

import numpy as np

import torch

from efficientnet_pytorch import EfficientNet

from torchsummary import summary



from torchvision.datasets import MNIST

from torch.utils.data import DataLoader

import torchvision.transforms as transforms

import torch.nn.functional as F
def fix_randomness(seed):

    np.random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)

    torch.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False
BATCH_SIZE = 64

SEED = 42

LOG_INTERVAL = 100

EPOCHS = 4



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

fix_randomness(SEED)
transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize((0.5, ), (0.5, )),

])



train_dataset = MNIST(".", train=True, transform=transform, download=True)

val_dataset = MNIST(".", train=False, transform=transform, download=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
model = EfficientNet.from_pretrained("efficientnet-b0")

model._fc = torch.nn.Linear(1280, 10)

model.to(device)

summary(model, (3, 28, 28))



optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda= lambda epoch: 0.95 ** epoch)
for epoch in range(EPOCHS):

    

    # Train.

    model.train()

    for step, (data, label) in enumerate(train_loader):

        data = data.repeat(1, 3, 1, 1)

        data = data.to(device)

        label = label.to(device)



        optimizer.zero_grad()



        output = model(data)

        prob = F.log_softmax(output, dim=1)

        loss = F.nll_loss(prob, label)



        loss.backward()

        optimizer.step()



        if step % LOG_INTERVAL == 0:

            print(f"(Step {step}) Loss: {loss.item()}")

    

    # Validate.

    correct = 0

    total = 0

    model.eval()

    with torch.no_grad():

        for step, (data, label) in enumerate(val_loader):

            data = data.repeat(1, 3, 1, 1)

            data = data.to(device)

            label = label.to(device)



            output = model(data)

            prob = F.log_softmax(output, dim=1)

            loss = F.nll_loss(prob, label, reduction="sum").item()

            preds = output.argmax(1, True)

            correct += preds.eq(label.view_as(preds)).sum().item()

            total += BATCH_SIZE

    

    scheduler.step()

    print("[Epoch {}] Correct: {} / {} (Accuracy: {})".format(epoch, correct, total, correct / total))
import cv2

import matplotlib.pyplot as plt



#img = cv2.imread("../input/mnist-png/mnist_png/training/3/10000.png", cv2.IMREAD_GRAYSCALE)

img = cv2.imread("../input/mydigit/digit.png", cv2.IMREAD_GRAYSCALE)

plt.imshow(img)

img = cv2.resize(img, (28, 28))
input_tensor = transform(img).unsqueeze(0).repeat(1, 3, 1, 1)

input_tensor = input_tensor.to(device)

model.eval()

output = model(input_tensor)

pred = output.argmax(1, keepdim=True)

print("Predict: {}".format(pred.cpu()[0][0]))