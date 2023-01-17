import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
from PIL import Image
import time
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
torch.__version__
import torchvision
torchvision.__version__
# Kaggle Kernel-dependent
input_path = "../input/alien_vs_predator_thumbnails/data/"
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# Use resnet pre-processing
data_transforms = {
    'train':
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]),
    'validation':
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ]),
}

image_datasets = {
    'train': 
    datasets.ImageFolder(input_path + 'train', data_transforms['train']),
    'validation': 
    datasets.ImageFolder(input_path + 'validation', data_transforms['validation'])
}

dataloaders = {
    'train':
    torch.utils.data.DataLoader(image_datasets['train'],
                                batch_size=32,
                                shuffle=True,
                                num_workers=0),  # for Kaggle
    'validation':
    torch.utils.data.DataLoader(image_datasets['validation'],
                                batch_size=32,
                                shuffle=False,
                                num_workers=0)  # for Kaggle
}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device
model = models.resnet50(pretrained=True).to(device)
# model = models.densenet121(pretrained=True).to(device)
# print(model)
for param in model.parameters():
    param.requires_grad = False   

# num_ftrs = model.classifier.in_features
model.fc = nn.Sequential(
               nn.Linear(2048, 128),
               nn.ReLU(inplace=True),
               nn.Linear(128, 2)).to(device)
# model.classifier = nn.Sequential(
#                nn.Linear(num_ftrs, 128),
#                nn.ReLU(inplace=True),
#                nn.Linear(128, 2)).to(device)
criterion = nn.CrossEntropyLoss()
# Optimzer
optimizer = optim.Adam(model.fc.parameters(), lr=0.0001)
# LR scheduler
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                    cooldown=4,
                                                    verbose=True)
def train_model(model, criterion, optimizer, num_epochs=3):
    last_end = time.time()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])
            epoch_time = time.time() - last_end
            last_end = time.time()
            print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,
                                                        epoch_loss,
                                                        epoch_acc))
            print('Epoch Time {epoch_time:.3f}'.format(epoch_time=epoch_time))
            if phase == 'validation':
                lr_scheduler.step(epoch_loss)
                print('-' * 10)
    return model
model_trained = train_model(model, criterion, optimizer, num_epochs=20)
!mkdir models
!mkdir models/pytorch
torch.save(model_trained.state_dict(), 'models/pytorch/weights.h5')
model = models.resnet50(pretrained=False).to(device)
model.fc = nn.Sequential(
               nn.Linear(2048, 128),
               nn.ReLU(inplace=True),
               nn.Linear(128, 2)).to(device)
model.load_state_dict(torch.load('models/pytorch/weights.h5'))
validation_img_paths = ["validation/alien/11.jpg",
                        "validation/alien/22.jpg",
                        "validation/predator/33.jpg"]
img_list = [Image.open(input_path + img_path) for img_path in validation_img_paths]
validation_batch = torch.stack([data_transforms['validation'](img).to(device)
                                for img in img_list])
pred_logits_tensor = model(validation_batch)
pred_logits_tensor
pred_probs = F.softmax(pred_logits_tensor, dim=1).cpu().data.numpy()
pred_probs
fig, axs = plt.subplots(1, len(img_list), figsize=(20, 5))
for i, img in enumerate(img_list):
    ax = axs[i]
    ax.axis('off')
    ax.set_title("{:.0f}% Alien, {:.0f}% Predator".format(100*pred_probs[i,0],
                                                            100*pred_probs[i,1]))
    ax.imshow(img)