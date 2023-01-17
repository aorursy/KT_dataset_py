import torch
import glob
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss, Linear
transformslist = transforms.Compose([
    transforms.Resize(233),
    transforms.RandomCrop(224),
    transforms.RandomRotation(360),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class BrainTumorDataset(Dataset):
    def __init__(self, transform=transformslist):
        self.filelist = glob.glob("/kaggle/input/brain-mri-images-for-brain-tumor-detection/brain_tumor_dataset/*/*")
        self.transform = transform
        
    def __len__(self):
        return len(self.filelist)
    
    def __getitem__(self, index):
        file = self.filelist[index]
        label = 1.0 if file.split("/")[-2] == "yes" else 0.0
        image = Image.open(file).convert('RGB')
        return self.transform(image), torch.tensor([label])
bt_dataset = BrainTumorDataset()
idx = torch.randperm(len(bt_dataset))
train_split = int(len(idx)*0.7)
valid_split = int(len(idx)*0.9)
train_idx = idx[:train_split]
valid_idx = idx[train_split:valid_split]
test_idx = idx[valid_split:]
train_loader = DataLoader(bt_dataset, sampler=SubsetRandomSampler(train_idx), batch_size=32)
valid_loader = DataLoader(bt_dataset, sampler=SubsetRandomSampler(valid_idx), batch_size=32)
device = "cuda"
model = resnet18(pretrained=True)
model.fc = Linear(512, 1)
model = model.to(device)
opt = Adam(model.parameters(), 1e-3)
loss = BCEWithLogitsLoss()
EPOCH = 100
best_loss = 1e10
best_model_param = model.state_dict()
for i in range(EPOCH):
    l = 0
    cnt = 0
    for ims, labels in train_loader:
        ims = ims.to(device)
        labels = labels.to(device)
        opt.zero_grad()
        ll = loss(model(ims), labels)
        ll.backward()
        l += ll
        opt.step()
    print(f"epoch {i}, validation loss: {l/cnt}")
    l = 0
    cnt = 0
    with torch.no_grad():
        for ims, labels in valid_loader:
            ims = ims.to(device)
            labels = labels.to(device)
            l += loss(model(ims), labels)
            cnt += ims.shape[0]
        print(f"epoch {i}, validation loss: {l/cnt}")
        if l < best_loss:
            best_loss = l
            best_model_param = model.state_dict()
model = resnet18()
model.fc = Linear(512, 1)
model.load_state_dict(best_model_param)
model.cuda()
def test_im(index, n=3):
    label = bt_dataset[index][1].to(device)
    r = 0
    for i in range(n):
        r += model(bt_dataset[index][0][None, ...].to(device))
    return (r/n > 0).float() == label
acc = 0
for idx in test_idx:
    acc += test_im(idx).float()
print(acc/len(test_idx))
from matplotlib import pyplot as plt

for i in test_idx:
    im, label = bt_dataset[i]
    plt.imshow(im[0])
    plt.show()
    print(label, model(im[None, ...].to(device)))
