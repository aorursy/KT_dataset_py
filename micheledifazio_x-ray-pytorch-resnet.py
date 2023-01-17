import os
import torch
import torchvision
import torch.nn as nn
from tqdm.notebook import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import DataLoader
TRAIN_DIR = "../input/chest-xray-pneumonia/chest_xray/train"
VAL_DIR = "../input/chest-xray-pneumonia/chest_xray/val"
TEST_DIR = "../input/chest-xray-pneumonia/chest_xray/test"
transform_train = T.Compose([
    T.Resize((64, 64)),
    T.RandomCrop(64, padding_mode="reflect"),
    T.RandomHorizontalFlip(),
    T.ToTensor()
])

transform_val = T.Compose([
    T.Resize((64, 64)),
    T.ToTensor()
])
train_ds = torchvision.datasets.ImageFolder(root=TRAIN_DIR, transform=transform_train)
val_ds = torchvision.datasets.ImageFolder(root=VAL_DIR, transform=transform_val)
batch_size=128
train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size, num_workers=4, pin_memory=True)
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
    
def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        for x in self.dl:
            yield to_device(x, self.device)
            
    def __len__(self):
        return len(self.dl)
    
device = get_device()
device
train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        return loss
    
    def validation_step(self, batch):
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        acc = accuracy(outputs, labels)
        return {"val_loss": loss.detach(), "val_acc": acc}
    
    def validation_epoch_end(self, outputs):
        batch_loss = [x["val_loss"] for x in outputs]
        epoch_loss = torch.stack(batch_loss).mean()
        batch_acc = [x["val_acc"] for x in outputs]
        epoch_acc = torch.stack(batch_acc).mean()
        return {"val_loss": epoch_loss.item(), "val_acc": epoch_acc.item()}
    
    def epoch_end(self, epoch, epochs, result):
        print("Epoch [{}/{}], last_lr: {:.6f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
        epoch+1, epochs, result["lrs"][-1], result["train_loss"], result["val_loss"], result["val_acc"]))
class ResNet(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = models.resnet18(pretrained=True)
        number_of_features = self.network.fc.in_features
        self.network.fc = nn.Linear(number_of_features, 2)
        
    def forward(self, xb):
        return self.network(xb)
    
    def freeze(self):
        for param in self.network.parameters():
            param.requires_grad=False
        for param in self.network.fc.parameters():
            param.requires_grad=True
            
    def unfreeze(self):
        for param in self.network.parameters():
            param.requires_grad=True
model = to_device(ResNet(), device)
@torch.no_grad()
def evaluate(model, val_dl):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_dl]
    return model.validation_epoch_end(outputs)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]
    
def fit_one_cycle(epochs, max_lr, model, train_dl, val_dl,
                 weight_decay=0, grad_clip=None, opt_func=torch.optim.Adam):
    history = []
    
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
                                        steps_per_epoch=len(train_dl))
    
    for epoch in range(epochs):
        model.train()
        train_loss = []
        lrs = []
        for batch in tqdm(train_dl):
            loss = model.training_step(batch)
            train_loss.append(loss)
            loss.backward()
            
            
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
                
            optimizer.step()
            optimizer.zero_grad()
            
            lrs.append(get_lr(optimizer))
            scheduler.step()
        result = evaluate(model, val_dl)
        result["train_loss"] = torch.stack(train_loss).mean().item()
        result["lrs"] = lrs
        model.epoch_end(epoch, epochs, result)
        history.append(result)
    return history
result = evaluate(model, val_dl)
result
model.freeze()
epochs = 10
max_lr = 10e-4
grad_clip = 0.01
weight_decay = 10e-4
opt_func= torch.optim.Adam
%%time

history= fit_one_cycle(epochs, max_lr, model, train_dl, val_dl,
                       grad_clip=grad_clip, weight_decay=weight_decay,
                       opt_func=opt_func)
model.unfreeze()
%%time
history += fit_one_cycle(epochs, max_lr, model, train_dl, val_dl,
                       grad_clip=grad_clip, weight_decay=weight_decay,
                       opt_func=opt_func)
accuracy = [x["val_acc"] for x in history]
plt.plot(accuracy, "-rx")
plt.title("Accuracy")
plt.xlabel("number of epochs")
plt.ylabel("Accuracy")
val_loss = [x["val_loss"] for x in history]
train_loss = [x.get("train_loss") for x in history]
plt.plot(val_loss, "-bx")
plt.plot(train_loss, "-gx")
plt.title("Loss")
plt.legend(["val_loss", "train_loss"])
plt.xlabel("number of epochs")
TEST_DIR ="../input/chest-xray-pneumonia/chest_xray/test"
transform_test = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor()
])

test_ds = torchvision.datasets.ImageFolder(root=TEST_DIR, transform=transform_test)
test_dl = DataLoader(test_ds, batch_size, num_workers=4, pin_memory=True)
test_dl = DeviceDataLoader(test_dl, device)
def predict_image(images, model):
    xb = to_device(images.unsqueeze(0), device)
    out = model(xb)
    _, preds = torch.max(out, dim=1)
    prediction = preds[0].item()
    return prediction
images, labels = test_ds[11]
plt.imshow(images.permute(1,2,0))
print("Label", labels, "Prediction:", predict_image(images, model))