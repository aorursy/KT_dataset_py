import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
transform_train = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Normalize the test set same as training set without augmentation
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

inv_transform = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.2023, 1/0.1994, 1/0.2010 ]),
                                transforms.Normalize(mean = [ -0.4914, -0.4822, -0.4465 ],
                                                     std = [ 1., 1., 1. ]),
                               ])
dataroot = "/kaggle/working/"
trainset = torchvision.datasets.CIFAR10(
    root=dataroot, train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(
    root=dataroot, train=False, download=True, transform=transform_test)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=5, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=5, shuffle=True, num_workers=2)

num_images_to_show = 10
images = {}
for img in (trainset):
    label = img[1]
    img = inv_transform(img[0]).permute(1,2,0)
    if(label not in images.keys()):
        images[label] = []
    if(len(images[label]) < num_images_to_show):
        images[label].append(img)

f, axarr = plt.subplots(len(images.keys()),num_images_to_show)
for i in range(len(images.keys())):
    for j in range(num_images_to_show):
        axarr[i][j].imshow(images[i][j])
f.set_figheight(8)
f.set_figwidth(8)
plt.show()
del images , axarr , f
class TreeCNN(nn.Module):
    """TreeCNN."""

    def __init__(self):
        super(TreeCNN, self).__init__()


    def forward(self, x):
        """Perform forward."""
        return x

