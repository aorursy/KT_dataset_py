# https://www.kaggle.com/hkthirano/resnet18-pretrained-w-92-test-accuracy
import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
import numpy as np
np.random.seed(0)
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import PIL
DC = torchvision.datasets.ImageFolder(
    root      = "/kaggle/input/training_set/training_set", 
    transform = transforms.Compose([
        transforms.Resize(200),
        transforms.Pad(100, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10, resample=PIL.Image.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ]))

dataloader = torch.utils.data.DataLoader(DC, batch_size=16, shuffle=True, num_workers=8)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)

model.fc = nn.Linear(512, 2)
model.to(device)
optimizer_ft = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()
iteration = 0
num_epoch = 3
model.train()
for epoch in range(num_epoch):
    for x, y in dataloader:
        pred = model(x.to(device))
        loss = loss_fn(pred, y.to(device))
        optimizer_ft.zero_grad()
        loss.backward()
        optimizer_ft.step()
        iteration = iteration + 1
        print(f"\r[{iteration * 100.0 / num_epoch/len(dataloader):7.3f} % ] loss: {loss.item():5.3f}", end="")
torch.save(model.state_dict(), "model_dogvscat_model.pt")
%rm -r antialiased-cnns
DC_test = torchvision.datasets.ImageFolder(
    root      = "/kaggle/input/test_set/test_set", 
    transform = transforms.Compose([
        transforms.Resize((224,224)),
#         transforms.Pad(100, padding_mode="reflect"),
#         transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ]))
    
def acc(model, dataset):
    model.eval()
    hit = 0
    total = 0
    dataloader_test = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    for x, y in dataloader_test:
        pred = model(x.to(device))
        hit = hit + sum( pred.argmax(dim=-1).cpu() == y ).item()
        total = total + x.size(0)

    model.train()
    return hit*100.0 / total


print("Test accuracy: ", acc(model, DC_test))
print("Train acc: ", acc(model, DC))