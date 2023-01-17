import torch as torch
import torch.optim as optim
import torch.nn as nn
import os
import torchvision
import torch.nn as nn
from torch.autograd import Variable as var
import logging as log
import gc
import numpy as np
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset
from PIL import Image
from glob import glob
!nvidia-smi
# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
## Constants
N = 32 # Batch Size
T = 0.07 # Temperature
C = 3 # Number of Channels
m = 0.9 # momemntum contrast
K = 4096 # dictionary size
DATA = '/kaggle/input/imagenetmini-1000/imagenet-mini/'
class ImageNet(Dataset):
    def __init__(self, root_dir, train=False, transform=None):

            self.root_dir = root_dir
            
            self.transform = transform

            self.sub_directory = 'train' if train else 'val'
            
            path = os.path.join(
            root_dir, self.sub_directory, "*","*")
            
            self.imgs = glob(path)
            
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self,idx):
        img = Image.open(self.imgs[idx],).convert('RGB')
        if self.transform is not None:
            img = self.transform(img);

        return img;
## Load Data
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])

train_data = ImageNet(
    root_dir=DATA, train=True,  transform=transform)
test_data = ImageNet(
    root_dir=DATA, train=False,  transform=transform)

train_set = torch.utils.data.DataLoader(
    train_data, batch_size=N,shuffle=True,num_workers = 4, pin_memory=True, drop_last=True)
test_set = torch.utils.data.DataLoader(
    test_data, batch_size=N,shuffle=False,num_workers = 4,pin_memory=True, drop_last=True)
print(len(train_set))
## Augmentations

def get_random_augmentation():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(size=224),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(3),
        transforms.ToTensor()
    ])

def apply_augmentation_transform(x):
    batch = [];
    aug = get_random_augmentation();
    for i in range(x.size(0)):
        augmented = aug(x[1:][i-1].cpu()).reshape([1,C,224,224])
        batch.append(augmented)
    result= torch.cat(batch,dim=0)
    return result.cuda();
class Resnet50Model(nn.Module):
    def __init__(self):
        super(Resnet50Model, self).__init__()

        model = models.resnet50(pretrained=False)
        modules = list(model.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.fc = nn.Sequential(nn.Linear(2048, N), nn.ReLU())
    
    def forward(self,x):
        x = self.resnet(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        x = torch.nn.functional.normalize(x,dim=0)
        
        return x
    
encoder_q = Resnet50Model().cuda()
encoder_k = Resnet50Model().cuda()
s  = sum(np.prod(list(p.size())) for p in encoder_q.parameters())
print(s)
optimizer = torch.optim.SGD(encoder_q.parameters(), lr=0.03, momentum=0.9, weight_decay=0.0001)
cec = nn.CrossEntropyLoss().cuda()
for param_q, param_k in zip(encoder_q.parameters(), encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
@torch.no_grad()
def enqueue(queue,k):
    return torch.cat([queue[:,1:], k],dim=1)

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
@torch.no_grad()
def _dequeue_and_enqueue(self, keys):
    # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % K  # move pointer

        self.queue_ptr[0] = ptr
# register_buffer("queue", torch.randn(N, K))
# queue = nn.functional.normalize(queue, dim=0)
# register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
queue = torch.randn(N,K).cuda()
queue = nn.functional.normalize(queue, dim=0)

for e in range(100):
    epoch_loss = 0.0
    running_loss = 0.0
    for i,(images) in enumerate(train_set):
    
        images = var(images.cuda())
        optimizer.zero_grad()

        images_q = apply_augmentation_transform(images);
        images_k = apply_augmentation_transform(images);

        q = encoder_q.forward(images_q);
        k = encoder_k.forward(images_k);


        k = k.detach();

         # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, queue.clone().detach()])


        logits = torch.cat([l_pos, l_neg], dim=1)

        labels = torch.zeros(N).type(torch.LongTensor).cuda()

        logits = logits/T;

        loss = cec(logits, labels)

        #updating query encoder
        loss.backward()
        optimizer.step()

        epoch_loss += q.shape[0] * loss.item()

        running_loss += loss.item()

        with torch.no_grad():
            for p_k,p_q in zip(encoder_k.parameters(),encoder_q.parameters()):
                val = (1-m)*p_q.data + m*p_k.data
                p_k.data = p_k.data.copy_(val)

        queue = enqueue(queue, k)
        if((i+1) % 50 == 0):
            print('Epoch :',e+1,'Batch :',(i+1),'Loss :',float(running_loss/50))
            running_loss = 0.0
        
    if((e+1) % 10 == 0):
        torch.save(encoder_q,"./kaggle/output/saved_models/encoder_q_"+str(e+1)+".pth")
        print('Epoch :',e+1, 'Loss :',epoch_loss/len(train_set))