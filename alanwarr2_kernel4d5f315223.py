# Uncomment and run the commands below if imports fail

# !conda install numpy pandas pytorch torchvision cpuonly -c pytorch -y

# !pip install matplotlib --upgrade --quiet
import os

import torch

import torchvision

import tarfile

from torchvision.datasets.utils import download_url

from torch.utils.data import random_split
# Dowload the dataset

# dataset_url = "http://files.fast.ai/data/cifar10.tgz"

# download_url(dataset_url, '.')
# Extract from archive

# with tarfile.open('./cifar10.tgz', 'r:gz') as tar:

#     tar.extractall(path='./data')
data_dir = '../input/10-monkey-species'



print(os.listdir(data_dir))

classes = os.listdir(data_dir + "/validation/validation")

print(classes)
from torchvision.datasets import ImageFolder

from torchvision.transforms import ToTensor

import torchvision.transforms as tt
train_tfms = tt.Compose([tt.CenterCrop(512),

                         tt.Resize(32),

                         tt.ToTensor()])

valid_tfms = tt.Compose([tt.CenterCrop(512), tt.Resize(32),tt.ToTensor()])



train_ds = ImageFolder(data_dir+'/training/training', train_tfms)

val_ds = ImageFolder(data_dir+'/validation/validation', valid_tfms)
img, label = train_ds[0]

print(img.shape, label)

img
print(train_ds.classes)
import matplotlib.pyplot as plt



def show_example(img, label):

    if label == 0: display_name = "mantled_howler"

    if label == 1: display_name = "patas_monkey"

    if label == 2: display_name = "bald_uakari"

    if label == 3: display_name = "japanese_macaque"

    if label == 4: display_name = "pygmy_marmoset"

    if label == 5: display_name = "white_headed_capuchin"

    if label == 6: display_name = "silvery_marmoset"

    if label == 7: display_name = "common_squirrel_monkey"

    if label == 8: display_name = "black_headed_night_monkey"

    if label == 9: display_name = "nilgiri_langur"

    print('Label: ', train_ds.classes[label],"("+str(label)+")")

    print(display_name)

    plt.imshow(img.permute(1, 2,  0))
show_example(*train_ds[0])
show_example(*train_ds[400])
random_seed = 42

torch.manual_seed(random_seed);
from torch.utils.data.dataloader import DataLoader



batch_size=128
train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)

val_dl = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)
from torchvision.utils import make_grid



def show_batch(dl):

    for images, labels in dl:

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.set_xticks([]); ax.set_yticks([])

        ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))

        break
show_batch(train_dl)
def apply_kernel(image, kernel):

    ri, ci = image.shape       # image dimensions

    rk, ck = kernel.shape      # kernel dimensions

    ro, co = ri-rk+1, ci-ck+1  # output dimensions

    output = torch.zeros([ro, co])

    for i in range(ro): 

        for j in range(co):

            output[i,j] = torch.sum(image[i:i+rk,j:j+ck] * kernel)

    return output
sample_image = torch.tensor([

    [3, 3, 2, 1, 0], 

    [0, 0, 1, 3, 1], 

    [3, 1, 2, 2, 3], 

    [2, 0, 0, 2, 2], 

    [2, 0, 0, 0, 1]

], dtype=torch.float32)



sample_kernel = torch.tensor([

    [0, 1, 2], 

    [2, 2, 0], 

    [0, 1, 2]

], dtype=torch.float32)



apply_kernel(sample_image, sample_kernel)
import torch.nn as nn

import torch.nn.functional as F
simple_model = nn.Sequential(

    nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),

    nn.MaxPool2d(2, 2)

)
for images, labels in train_dl:

    print('images.shape:', images.shape)

    out = simple_model(images)

    print('out.shape:', out.shape)

    break
def accuracy(outputs, labels):

    _, preds = torch.max(outputs, dim=1)

    return torch.tensor(torch.sum(preds == labels).item() / len(preds))



class ImageClassificationBase(nn.Module):

    def training_step(self, batch):

        images, labels = batch 

        out = self(images)                  # Generate predictions

        loss = F.cross_entropy(out, labels) # Calculate loss

        return loss

    

    def validation_step(self, batch):

        images, labels = batch 

        out = self(images)                    # Generate predictions

        loss = F.cross_entropy(out, labels)   # Calculate loss

        acc = accuracy(out, labels)           # Calculate accuracy

        return {'val_loss': loss.detach(), 'val_acc': acc}

        

    def validation_epoch_end(self, outputs):

        batch_losses = [x['val_loss'] for x in outputs]

        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses

        batch_accs = [x['val_acc'] for x in outputs]

        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies

        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    

    def epoch_end(self, epoch, result):

        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(

            epoch, result['train_loss'], result['val_loss'], result['val_acc']))
class Cifar10CnnModel(ImageClassificationBase):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=3, padding=1), #input size

            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), #input size (first number)

            nn.ReLU(),

            nn.MaxPool2d(2, 2), # output: 64 x 16 x 16



            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),

            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),

            nn.ReLU(),

            nn.MaxPool2d(2, 2), # output: 128 x 8 x 8



            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),

            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),

            nn.ReLU(),

            nn.MaxPool2d(2, 2), # output: 256 x 4 x 4



            nn.Flatten(), 

            nn.Linear(256*4*4, 1024),

            nn.ReLU(),

            nn.Linear(1024, 512),

            nn.ReLU(),

            nn.Linear(512, 10))

        

    def forward(self, xb):

        return self.network(xb)
model = Cifar10CnnModel()

model
for images, labels in train_dl:

    print('images.shape:', images.shape)

    out = model(images)

    print('out.shape:', out.shape)

    print('out[0]:', out[0])

    break
def get_default_device():

    """Pick GPU if available, else CPU"""

    if torch.cuda.is_available():

        return torch.device('cuda')

    else:

        return torch.device('cpu')

    

def to_device(data, device):

    """Move tensor(s) to chosen device"""

    if isinstance(data, (list,tuple)):

        return [to_device(x, device) for x in data]

    return data.to(device, non_blocking=True)



class DeviceDataLoader():

    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):

        self.dl = dl

        self.device = device

        

    def __iter__(self):

        """Yield a batch of data after moving it to device"""

        for b in self.dl: 

            yield to_device(b, self.device)



    def __len__(self):

        """Number of batches"""

        return len(self.dl)
device = get_default_device()

device
train_dl = DeviceDataLoader(train_dl, device)

val_dl = DeviceDataLoader(val_dl, device)

to_device(model, device);
@torch.no_grad()

def evaluate(model, val_loader):

    model.eval()

    outputs = [model.validation_step(batch) for batch in val_loader]

    return model.validation_epoch_end(outputs)



def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):

    history = []

    optimizer = opt_func(model.parameters(), lr)

    for epoch in range(epochs):

        # Training Phase 

        model.train()

        train_losses = []

        for batch in train_loader:

            loss = model.training_step(batch)

            train_losses.append(loss)

            loss.backward()

            optimizer.step()

            optimizer.zero_grad()

        # Validation phase

        result = evaluate(model, val_loader)

        result['train_loss'] = torch.stack(train_losses).mean().item()

        model.epoch_end(epoch, result)

        history.append(result)

    return history
model = to_device(Cifar10CnnModel(), device)
evaluate(model, val_dl)
num_epochs = 25

opt_func = torch.optim.Adam

lr = 0.001
history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)
def plot_accuracies(history):

    accuracies = [x['val_acc'] for x in history]

    plt.plot(accuracies, '-x')

    plt.xlabel('epoch')

    plt.ylabel('accuracy')

    plt.title('Accuracy vs. No. of epochs');
plot_accuracies(history)
def plot_losses(history):

    train_losses = [x.get('train_loss') for x in history]

    val_losses = [x['val_loss'] for x in history]

    plt.plot(train_losses, '-bx')

    plt.plot(val_losses, '-rx')

    plt.xlabel('epoch')

    plt.ylabel('loss')

    plt.legend(['Training', 'Validation'])

    plt.title('Loss vs. No. of epochs');
plot_losses(history)
test_dataset = ImageFolder(data_dir+'/validation/validation', valid_tfms)
def predict_image(img, model):

    # Convert to a batch of 1

    xb = to_device(img.unsqueeze(0), device)

    # Get predictions from model

    yb = model(xb)

    # Pick index with highest probability

    _, preds  = torch.max(yb, dim=1)

    # Retrieve the class label

    return train_ds.classes[preds[0].item()]
img, label = test_dataset[0]

if label == 0: display_name = "mantled_howler"

if label == 1: display_name = "patas_monkey"

if label == 2: display_name = "bald_uakari"

if label == 3: display_name = "japanese_macaque"

if label == 4: display_name = "pygmy_marmoset"

if label == 5: display_name = "white_headed_capuchin"

if label == 6: display_name = "silvery_marmoset"

if label == 7: display_name = "common_squirrel_monkey"

if label == 8: display_name = "black_headed_night_monkey"

if label == 9: display_name = "nilgiri_langur"

if predict_image(img, model) == "n0": predicted_name = "mantled_howler"

if predict_image(img, model) == "n1": predicted_name = "patas_monkey"

if predict_image(img, model) == "n2": predicted_name = "bald_uakari"

if predict_image(img, model) == "n3": predicted_name = "japanese_macaque"

if predict_image(img, model) == "n4": predicted_name = "pygmy_marmoset"

if predict_image(img, model) == "n5": predicted_name = "white_headed_capuchin"

if predict_image(img, model) == "n6": predicted_name = "silvery_marmoset"

if predict_image(img, model) == "n7": predicted_name = "common_squirrel_monkey"

if predict_image(img, model) == "n8": predicted_name = "black_headed_night_monkey"

if predict_image(img, model) == "n9": predicted_name = "nilgiri_langur"   

plt.imshow(img.permute(1, 2, 0))

print('Label:', train_ds.classes[label], ', Predicted:', predict_image(img, model))

print('Actual: ', display_name, 'Predicted: ', predicted_name)
img, label = test_dataset[100]

if label == 0: display_name = "mantled_howler"

if label == 1: display_name = "patas_monkey"

if label == 2: display_name = "bald_uakari"

if label == 3: display_name = "japanese_macaque"

if label == 4: display_name = "pygmy_marmoset"

if label == 5: display_name = "white_headed_capuchin"

if label == 6: display_name = "silvery_marmoset"

if label == 7: display_name = "common_squirrel_monkey"

if label == 8: display_name = "black_headed_night_monkey"

if label == 9: display_name = "nilgiri_langur"

if predict_image(img, model) == "n0": predicted_name = "mantled_howler"

if predict_image(img, model) == "n1": predicted_name = "patas_monkey"

if predict_image(img, model) == "n2": predicted_name = "bald_uakari"

if predict_image(img, model) == "n3": predicted_name = "japanese_macaque"

if predict_image(img, model) == "n4": predicted_name = "pygmy_marmoset"

if predict_image(img, model) == "n5": predicted_name = "white_headed_capuchin"

if predict_image(img, model) == "n6": predicted_name = "silvery_marmoset"

if predict_image(img, model) == "n7": predicted_name = "common_squirrel_monkey"

if predict_image(img, model) == "n8": predicted_name = "black_headed_night_monkey"

if predict_image(img, model) == "n9": predicted_name = "nilgiri_langur"   

plt.imshow(img.permute(1, 2, 0))

print('Label:', train_ds.classes[label], ', Predicted:', predict_image(img, model))

print('Actual: ', display_name, 'Predicted: ', predicted_name)
img, label = test_dataset[200]

if label == 0: display_name = "mantled_howler"

if label == 1: display_name = "patas_monkey"

if label == 2: display_name = "bald_uakari"

if label == 3: display_name = "japanese_macaque"

if label == 4: display_name = "pygmy_marmoset"

if label == 5: display_name = "white_headed_capuchin"

if label == 6: display_name = "silvery_marmoset"

if label == 7: display_name = "common_squirrel_monkey"

if label == 8: display_name = "black_headed_night_monkey"

if label == 9: display_name = "nilgiri_langur"

if predict_image(img, model) == "n0": predicted_name = "mantled_howler"

if predict_image(img, model) == "n1": predicted_name = "patas_monkey"

if predict_image(img, model) == "n2": predicted_name = "bald_uakari"

if predict_image(img, model) == "n3": predicted_name = "japanese_macaque"

if predict_image(img, model) == "n4": predicted_name = "pygmy_marmoset"

if predict_image(img, model) == "n5": predicted_name = "white_headed_capuchin"

if predict_image(img, model) == "n6": predicted_name = "silvery_marmoset"

if predict_image(img, model) == "n7": predicted_name = "common_squirrel_monkey"

if predict_image(img, model) == "n8": predicted_name = "black_headed_night_monkey"

if predict_image(img, model) == "n9": predicted_name = "nilgiri_langur"   

plt.imshow(img.permute(1, 2, 0))

print('Label:', train_ds.classes[label], ', Predicted:', predict_image(img, model))

print('Actual: ', display_name, 'Predicted: ', predicted_name)
test_loader = DeviceDataLoader(DataLoader(test_dataset, batch_size*2), device)

result = evaluate(model, test_loader)

result
torch.save(model.state_dict(), 'cifar10-cnn.pth')
model2 = to_device(Cifar10CnnModel(), device)
model2.load_state_dict(torch.load('cifar10-cnn.pth'))
evaluate(model2, test_loader)