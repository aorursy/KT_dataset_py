import torch
import torch.nn as nn 
import pandas as pd
import numpy as np 
import glob 
import os
from sklearn import preprocessing
from sklearn import model_selection
import albumentations
from PIL import Image
from PIL import ImageFile
from torch.nn import functional as F
from  tqdm import tqdm
from sklearn import metrics
ImageFile.LOAD_TRUNCATED_IMAGES=True
DATA_DIR = '../input/captcha-version-2-images/samples'
BATCH_SIZE = 8
IMAGE_WIDTH = 200
IMAGE_HEIGHT = 50
NUM_WORKERS = 8
EPOCHS = 200
DEVICE = "cpu"
image_files = glob.glob(os.path.join(DATA_DIR,"*.png"))
# ../input/captcha-version-2-images/samples/2s22s.png
targets_o = [x.split('/')[-1].split('.')[0] for x in image_files]
targets = [[c for c in x] for x in targets_o]
targets_flatten = [c for clist in targets for c in clist]
lbl_enc = preprocessing.LabelEncoder()
lbl_enc.fit(targets_flatten)
targets_enc = [lbl_enc.transform(x) for x in targets]
targets_enc =np.array(targets_enc)

targets_enc = targets_enc +1
(train_img, test_img, train_targets,test_targets,_, test_targets_orig) = model_selection.train_test_split(image_files, targets_enc,targets_o, test_size = 0.1, random_state=42 )
class CaptchaDataset():
    def __init__(self,image_paths ,targets, resize=None):
        self.image_paths = image_paths
        self.targets = targets
        self.resize = resize 
        mean = (0.485,0.456,0.406)
        std = (0.229,0.224, 0.225)
        
        self.aug = albumentations.Compose( [albumentations.Normalize(mean,std, max_pixel_value=255.0,always_apply=True)])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self,index):
        image = Image.open(self.image_paths[index]).convert("RGB")
        targets = self.targets[index]
        
        if self.resize is not None:
            image = image.resize((self.resize[1],self.resize[0]) ,resample=Image.BILINEAR)
        
        image =np.array(image)
        augmented = self.aug(image=image)
        image = augmented["image"]
        image= np.transpose(image,(2,0,1)).astype(np.float32)
        
        return { "images": torch.tensor(image,dtype =torch.float),
               "targets": torch.tensor(targets,dtype=torch.long)}
class CaptchaModel(nn.Module):
    def __init__(self, num_chars):
        super(CaptchaModel, self).__init__()
        self.conv_1 = nn.Conv2d(3, 128, kernel_size=(3, 3), padding=(1, 1))
        self.pool_1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv_2 = nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(1, 1))
        self.pool_2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.linear_1 = nn.Linear(768, 64)
        self.drop_1 = nn.Dropout(0.2)
        self.lstm = nn.GRU(64, 32, bidirectional=True, num_layers=2, dropout=0.25, batch_first=True)
        self.output = nn.Linear(64, num_chars + 1)

    def forward(self, images, targets=None):
#         print(images.size())
        bs, c, h, w = images.size() # 16,3,50,200
        x = F.relu(self.conv_1(images)) # 16,128,50,200
        print(x.size())
        x = self.pool_1(x) # 16,64,25,100
        print(x.size())
        x = F.relu(self.conv_2(x)) # 16,64,25,100
        print(x.size())
        x = self.pool_2(x) # 16,64,12,50
        print(x.size())
        x = x.permute(0, 3, 1, 2) # 16,50,64,12
        print(x.size()) 
        x = x.view(bs, x.size(1), -1) # 16,50,64*12
        print(x.size())
        x = F.relu(self.linear_1(x))
        print(x.size())
        x = self.drop_1(x)
        x, _ = self.lstm(x)
        print(x.size())
        x = self.output(x)
        print(x.size())
        x = x.permute(1, 0, 2)

        if targets is not None:
            log_probs = F.log_softmax(x, 2)
            input_lengths = torch.full(
                size=(bs,), fill_value=log_probs.size(0), dtype=torch.int32
            )
            target_lengths = torch.full(
                size=(bs,), fill_value=targets.size(1), dtype=torch.int32
            )
            loss = nn.CTCLoss(blank=0)(
                log_probs, targets, input_lengths, target_lengths
            )
            return x, loss

        return x, None
if __name__ == "__main__":
    cm = CaptchaModel(19)
    img = torch.rand((1, 3, 50, 200))
    x, _ = cm(img, torch.rand((1, 5)))

def train_fn(model, data_loader, optimizer):
    model.train()
    fin_loss = 0
    tk0 = tqdm(data_loader, total=len(data_loader))
    for data in tk0:
        for key, value in data.items():
            data[key] = value.to(DEVICE)
        optimizer.zero_grad()
        _, loss = model(**data)
        loss.backward()
        optimizer.step()
        fin_loss += loss.item()
    return fin_loss / len(data_loader)



def eval_fn(model, data_loader):
    model.eval()
    fin_loss = 0
    fin_preds = []
    tk0 = tqdm(data_loader, total=len(data_loader))
    for data in tk0:
        for key, value in data.items():
            data[key] = value.to(DEVICE)
        batch_preds, loss = model(**data)
        fin_loss += loss.item()
        fin_preds.append(batch_preds)
    return fin_preds, fin_loss / len(data_loader)

def remove_duplicates(x):
    if len(x) < 2:
        return x
    fin = ""
    for j in x:
        if fin == "":
            fin = j
        else:
            if j == fin[-1]:
                continue
            else:
                fin = fin + j
    return fin


def decode_predictions(preds, encoder):
    preds = preds.permute(1, 0, 2)
    preds = torch.softmax(preds, 2)
    preds = torch.argmax(preds, 2)
    preds = preds.detach().cpu().numpy()
    cap_preds = []
    for j in range(preds.shape[0]):
        temp = []
        for k in preds[j, :]:
            k = k - 1
            if k == -1:
                temp.append("§")
            else:
                p = encoder.inverse_transform([k])[0]
                temp.append(p)
        tp = "".join(temp).replace("§", "")
        cap_preds.append(remove_duplicates(tp))
    return cap_preds




image_files = glob.glob(os.path.join(DATA_DIR, "*.png"))
targets_orig = [x.split("/")[-1][:-4] for x in image_files]
targets = [[c for c in x] for x in targets_orig]
targets_flat = [c for clist in targets for c in clist]

lbl_enc = preprocessing.LabelEncoder()
lbl_enc.fit(targets_flat)
targets_enc = [lbl_enc.transform(x) for x in targets]
targets_enc = np.array(targets_enc)
targets_enc = targets_enc + 1

(
    train_imgs,
    test_imgs,
    train_targets,
    test_targets,
    _,
    test_targets_orig,
) = model_selection.train_test_split(
    image_files, targets_enc, targets_orig, test_size=0.1, random_state=42
)

train_dataset = CaptchaDataset(
    image_paths=train_imgs,
    targets=train_targets,
    resize=(IMAGE_HEIGHT, IMAGE_WIDTH),
)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=True,
)

test_dataset = CaptchaDataset(
    image_paths=test_imgs,
    targets=test_targets,
    resize=(IMAGE_HEIGHT, IMAGE_WIDTH),
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=False,
)

model = CaptchaModel(num_chars=len(lbl_enc.classes_))
model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.8, patience=5, verbose=True
)
for epoch in range(EPOCHS):
    train_loss = train_fn(model, train_loader, optimizer)
    valid_preds, test_loss = eval_fn(model, test_loader)
    valid_captcha_preds = []
    for vp in valid_preds:
        current_preds = decode_predictions(vp, lbl_enc)
        valid_captcha_preds.extend(current_preds)
    combined = list(zip(test_targets_orig, valid_captcha_preds))
    print(combined[:10])
    test_dup_rem = [remove_duplicates(c) for c in test_targets_orig]
    accuracy = metrics.accuracy_score(test_dup_rem, valid_captcha_preds)
    print(
        f"Epoch={epoch}, Train Loss={train_loss}, Test Loss={test_loss} Accuracy={accuracy}"
    )
    scheduler.step(test_loss)

