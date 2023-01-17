import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, random_split, DataLoader
from PIL import Image,ImageStat
import torchvision.models as models
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from sklearn.metrics import f1_score
import torch.nn.functional as F
import torch.nn as nn
from torchvision.utils import make_grid
from torch.utils.data import Subset
%matplotlib inline
torch.manual_seed(10)
batch_size = 128
DATA_DIR = '../input/jovian-pytorch-z2g/Human protein atlas'

TRAIN_DIR = DATA_DIR + '/train'                           # Contains training images
TEST_DIR = DATA_DIR + '/test'                             # Contains test images

TRAIN_CSV = DATA_DIR + '/train.csv'                       # Contains real labels for training images
TEST_CSV = '../input/jovian-pytorch-z2g/submission.csv'   # Contains dummy labels for test image
train_df = pd.read_csv(TRAIN_CSV)
# train_df = train_df[train_df['Label'].str.contains('')]
test_file = pd.read_csv(TEST_CSV)
train_df[train_df['Label']=='']
labels = {
    0: 'Mitochondria',
    1: 'Nuclear bodies',
    2: 'Nucleoli',
    3: 'Golgi apparatus',
    4: 'Nucleoplasm',
    5: 'Nucleoli fibrillar center',
    6: 'Cytosol',
    7: 'Plasma membrane',
    8: 'Centrosome',
    9: 'Nuclear speckles'
}
def encode_label(label):
    target = torch.zeros(10)
    for l in str(label).split(' '):
        target[int(l)] = 1.
    return target

def decode_target(target, text_labels=False, threshold=0.5):
    result = []
    for i, x in enumerate(target):
        if (x >= threshold):
            if text_labels:
                result.append(labels[i] + "(" + str(i) + ")")
            else:
                result.append(str(i))
    return ' '.join(result)
    
row = test_file.loc[47]
img_id, img_label = row['Image'], row['Label']
img_fname = TEST_DIR + "/" + str(img_id) + ".png"
img = Image.open(img_fname)
img
# arr = np.array(img)
# Image.fromarray(arr[0:265,0:265,:])
def brightness(im_file):
    im = im_file.convert('LA')
    stat = ImageStat.Stat(im)
    return stat.mean[0]
    
row = test_file.loc[46]
img_id, img_label = row['Image'], row['Label']
img_fname = TEST_DIR + "/" + str(img_id) + ".png"
img = Image.open(img_fname)
img
arr = np.array(img)

crop_options = {0:arr[0:256,0:256,:],
                1: arr[0:256,256:512,:],
                2: arr[256:512,0:256,:],
                3: arr[256:512,256:512,:],
                4: arr[128:384,128:384,:]}
l = [brightness(Image.fromarray(crop_options[x])) for x in crop_options.keys()]
min_pos = l.index(min(l))
crops = [c for c in crop_options.items()]
crops.pop(min_pos)
arr = crops[crop_id]
img = Image.fromarray(arr)
        
        
l = [a,b,c,d,e]
p = l.index(max(l))
print(p)
print(a)
print(b)
print(c)
print(d)
print(e)
class HumanProteinDataset(Dataset):#(TRAIN_CSV, TRAIN_DIR, transform=transform)
    def __init__(self, csv_file, root_dir, transform=None,limit_class=''):
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df['Label'].str.contains('|'.join(limit_class.split(' ')))]
        self.df = self.df.sample(frac=1).reset_index(drop=False)
        self.transform = transform
        self.root_dir = root_dir
        #shuffle
        
    def __len__(self):
        return len(self.df)*4
    
    def __getitem__(self, idx):
        image_id = int(idx/4)
        crop_id = idx % 4
        row = self.df.loc[image_id]
        
        img_id, img_label = row['Image'], row['Label']
        img_fname = self.root_dir + "/" + str(img_id) + ".png"
        img = Image.open(img_fname)
        arr = np.array(img)

        crop_options = {0:arr[0:256,0:256,:],
                        1: arr[0:256,256:512,:],
                        2: arr[256:512,0:256,:],
                        3: arr[256:512,256:512,:],
                        4: arr[128:384,128:384,:]}
        l = [brightness(Image.fromarray(x)) for x in crop_options.values()]
        min_pos = l.index(min(l))
        crops = [c for c in crop_options.values()]
        crops.pop(min_pos)
        arr = crops[crop_id]
        img = Image.fromarray(arr)
        
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        return img, encode_label(img_label)
transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomVerticalFlip(p=0.5),
#                                 transforms.RandomAffine(degrees = 0,shear = 25),
                                transforms.RandomPerspective(),
                                transforms.ToTensor()
#                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                               ])
dataset = HumanProteinDataset(TRAIN_CSV, TRAIN_DIR, transform=transform, limit_class = '')

train_set = HumanProteinDataset(TRAIN_CSV, TRAIN_DIR, transform=transform, limit_class = '')
val_set = HumanProteinDataset(TRAIN_CSV, TRAIN_DIR, transform=None, limit_class = '')

train_idx = range(int(len(train_set)*.9))
val_idx = range(int(len(val_set)*.9),len(val_set))

train_set = Subset(train_set,train_idx)
val_set = Subset(val_set,val_idx)
print(len(train_idx))
print(len(val_set))

def show_sample(img, target, invert=False):
    if invert:
        plt.imshow(1 - img.permute((1, 2, 0)))
    else:
        plt.imshow(img.permute(1, 2, 0))
    print('Labels:', decode_target(target, text_labels=True))
    
show_sample(*dataset[2], invert=False)
train_set 
val_set 
train_dl = DataLoader(train_set, batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_dl = DataLoader(val_set, batch_size, num_workers=2,shuffle=False, pin_memory=True)
def show_batch(dl, invert=False):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.set_xticks([]); ax.set_yticks([])
        data = 1-images if invert else images
        ax.imshow(make_grid(data, nrow=16).permute(1, 2, 0))
        break
show_batch(val_dl)
def F_score(output, label, threshold=0.5, beta=1):
    prob = output > threshold
    label = label > threshold

    TP = (prob & label).sum(1).float()
    TN = ((~prob) & (~label)).sum(1).float()
    FP = (prob & (~label)).sum(1).float()
    FN = ((~prob) & label).sum(1).float()

    precision = torch.mean(TP / (TP + FP + 1e-12))
    recall = torch.mean(TP / (TP + FN + 1e-12))
    F2 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-12)
    return F2.mean(0)
class MultilabelImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, targets = batch 
        out = self(images)                      
        loss = F.binary_cross_entropy(out, targets)      
        return loss
    
    def validation_step(self, batch):
        images, targets = batch 
        out = self(images)                           # Generate predictions shapel(bs,10)
        loss = F.binary_cross_entropy(out, targets)  # Calculate loss
        score = F_score(out, targets)
        return {'val_loss': loss.detach(), 'val_score': score.detach() , 'out_shape':out.size()}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_scores = [x['val_score'] for x in outputs]
        epoch_score = torch.stack(batch_scores).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_score': epoch_score.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_score: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_score']))
class ProteinCnnModel2(MultilabelImageClassificationBase):
    def __init__(self):
        super().__init__()
        # Use a pretrained model
        self.network = models.resnet50(pretrained=True)
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, 10)
    
    def forward(self, xb):
        return torch.sigmoid(self.network(xb))
model = ProteinCnnModel2()
torch.save(model,'drop_darkest_fifth_blank.pth')
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
def try_batch(dl):
    for images, labels in dl:
        print('images.shape:', images.shape)
        out = model(images)
        print('out.shape:', out.shape)
        print('out[0]:', out[0])
        break

try_batch(train_dl)
for batch in val_dl:
    out = model.validation_step(batch)
    break # return {'val_loss': loss.detach(), 'val_score': score.detach() , 'out_shape':out.size()}

out['val_score']
from tqdm.notebook import tqdm
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []
    optimizer = opt_func#(model.parameters(), lr)
    for epoch in range(epochs):
        print(scheduler.get_last_lr())
        # Training Phase 
        model.train()
        train_losses = []
        for batch in tqdm(train_loader):
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        # Validation phase
        result = evaluate(model, val_loader)
        scheduler.step()
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
        torch.save(model,'drop_darkest_fifth_' + str(epoch) + '.pth')
    return history
model = to_device(ProteinCnnModel2(), device)
num_epochs = 15
lr = 5e-4
opt_func = torch.optim.Adam(model.parameters(), lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=opt_func, step_size=1, gamma=0.8)

history = fit(num_epochs, model, train_dl, val_dl, opt_func)
#
# Epoch [9], train_loss: 0.1356, val_loss: 0.1250, val_score: 0.8398
# [5.3687091200000036e-05]
# torch.save(model,'quart_im_res50.pth')
model = torch.load('drop_darkest_fifth_9.pth')
def predict_single(image):
    xb = image.unsqueeze(0)
    xb = to_device(xb, device)
    preds = model(xb)
    prediction = preds[0]
    print("Prediction: ", prediction)
    show_sample(image, prediction)
class SubmissionDataset(Dataset):#(TRAIN_CSV, TRAIN_DIR, transform=transform)
    def __init__(self, csv_file, root_dir, transform=None,limit_class=''):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.root_dir = root_dir
        #shuffle
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.loc[idx]
        
        img_id, img_label = row['Image'], row['Label']
        img_fname = self.root_dir + "/" + str(img_id) + ".png"
        img = Image.open(img_fname)
        arr = np.array(img)
        crop_options = {0:arr[0:256,0:256,:],
                        1: arr[0:256,256:512,:],
                        2: arr[256:512,0:256,:],
                        3: arr[256:512,256:512,:],
                        4: arr[128:384,128:384,:]}
        a = brightness(Image.fromarray(crop_options[0]))
        b = brightness(Image.fromarray(crop_options[1]))
        c = brightness(Image.fromarray(crop_options[2]))
        d = brightness(Image.fromarray(crop_options[3]))
        e = brightness(Image.fromarray(crop_options[4]))

        l = [a,b,c,d,e]
        p = l.index(max(l))
        
        img = Image.fromarray(crop_options[p])
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        return img, encode_label(img_label)
test_dataset = SubmissionDataset(TEST_CSV, TEST_DIR, transform=None)
img, target = test_dataset[0]
img.shape
predict_single(test_dataset[100][0])
predict_single(test_dataset[74][0])
test_dl = DeviceDataLoader(DataLoader(test_dataset, batch_size,shuffle=False, num_workers=2, pin_memory=True), device)
@torch.no_grad()
def predict_dl(dl, model):
    torch.cuda.empty_cache()
    batch_probs = []
    for xb, _ in tqdm(dl):
        probs = model(xb)
        batch_probs.append(probs.cpu().detach())
    batch_probs = torch.cat(batch_probs)
    return [decode_target(x) for x in batch_probs]
test_preds = predict_dl(test_dl, model)
len(test_preds)
submission_df = pd.read_csv(TEST_CSV)
len(submission_df)

submission_df.Label = test_preds
submission_df.head()
sub_fname = 'resnet50_quartert_inputs_submission_2.csv'
submission_df.to_csv(sub_fname, index=False)
!pip install jovian --upgrade
import jovian
jovian.commit(project='zerogans-protein-competition')