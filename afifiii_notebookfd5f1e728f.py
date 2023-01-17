import torch

import torch.nn as nn

import torch.optim as optim

import numpy as np

import torchvision

from torchvision import datasets, models, transforms

import matplotlib.pyplot as plt

from PIL import Image



from torch.nn import Flatten, Linear, ReLU, Conv2d, MaxPool2d
!ls /kaggle/input/roadlane-detection-evaluation-2013/data_road/training/gt_image_2/
img = Image.open('/kaggle/input/roadlane-detection-evaluation-2013/data_road/training/image_2/umm_000000.png')
img
seg = Image.open('/kaggle/input/roadlane-detection-evaluation-2013/data_road/training/gt_image_2/umm_road_000000.png')
seg
seg_np = np.array(seg)

print(seg_np.shape)
seg_np[0,0]   # Red
seg_np[350, 0]
def colors_to_labels(seg):

    h,w,c = seg.shape

    labels = np.zeros((h,w), dtype=np.uint8)

    

    mask = (seg[:,:,2] == 255)    # (255, 0, 255)

    

    labels[mask] = 1 

    return labels
labels = colors_to_labels(seg_np)

plt.imshow(labels, cmap='gray')
model = torch.nn.modules.Sequential(Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1),

                                    ReLU(),

                                    Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),

                                    ReLU(),

                                    Conv2d(in_channels=64, out_channels=2, kernel_size=3, padding=1, stride=1),

                                    )
transform = transforms.ToTensor()

x = transform(img)

print(x.shape)          # CHW

print(x.min())

print(x.max())
scores = model(x.reshape(1, 3, 375, 1242))

print(scores.shape)
model = torch.nn.modules.Sequential(Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1),

                                    ReLU(),

                                    Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),

                                    ReLU(),

                                    Conv2d(in_channels=64, out_channels=2, kernel_size=3, padding=1, stride=1),

                                    )





model = model.cuda()

x = x.cuda()

labels = torch.tensor(labels).long().cuda()



optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)



# Optimization loop

for epoch in range(500):

    

    # Forward

    scores = model(x[None])



    # Loss

    loss = torch.nn.functional.cross_entropy(scores, labels[None])



    # Accuracy

    preds = torch.argmax(scores, dim=1)

    num_correct = torch.sum(preds == labels)

    total = 375 * 1242

    accuracy = num_correct / float(total)

    

    print(epoch, loss.item(), accuracy.item())

    

    # Backward

    loss.backward()



    # Update

    with torch.no_grad():

        optimizer.step()

        optimizer.zero_grad()
# Visualize

out = preds.cpu().numpy()
print(out.shape)

out = out.squeeze()

print(out.shape)
plt.imshow(out, cmap='gray')