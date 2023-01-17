import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data

import pandas as pd
import numpy as np

import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from PIL import Image

from matplotlib import pyplot as plt

import tqdm

import pickle
import os
import gc

#python-levenshtein
import Levenshtein
REC_IMG_SIZE = (98, 34)
MARGIN = (24,8)
IMG_SIZE = tuple(x1 + x2 for x1, x2 in zip(REC_IMG_SIZE, MARGIN))

NUM_CHARS = 22
NUM_LENGTH = 9

SQUEEZE_FACTOR = 2
REC_SCORE_CUTOFF = 7.
#get rid of cyrillic doubles
translation_table = str.maketrans("АВЕКМНОРСТУХ", "ABEKMHOPCTYX")
#we will extract images slightly larger than we need so that we can apply xy translations augumentation
def expand_crop_box(crop_box):
    w, h = crop_box[2] - crop_box[0], crop_box[3] - crop_box[1]
    w = w*IMG_SIZE[0]/REC_IMG_SIZE[0]
    h = h*IMG_SIZE[1]/REC_IMG_SIZE[1]
    
    x, y = (crop_box[0] + crop_box[2])/2, (crop_box[1] + crop_box[3])/2
    
    crop_box = [x - w/2, y - h/2, x + w/2, y + h/2]
    return crop_box
ann = pd.read_json(os.path.join('data','train.json'))[:-1]

images = []
targets = []
for samples, im_path in tqdm.tqdm(zip(ann['nums'], ann['file']), total=len(ann)):
    im_path = os.path.join('data',im_path)
    image = Image.open(im_path).convert('RGB')
    for sample in samples:
        box = np.array(sample['box'])
        x_min, y_min = box.min(axis=0)
        x_max, y_max = box.max(axis=0)
        targets.append(sample['text'].upper().translate(translation_table))
        crop_box = [x_min, y_min, x_max, y_max]
        crop_box = expand_crop_box(crop_box)
        images.append(image.crop(crop_box).resize(IMG_SIZE).tobytes())
        
plates_corpus = {'img_mode': 'RGB', 'img_size': IMG_SIZE, 'img_bytes': images, 'nums': targets}

with open('plates.pickle', 'wb') as fp:
    pickle.dump(plates_corpus, fp)
with open('plates.pickle', 'rb') as fp:
    plates_data = pickle.load(fp)
    
char_set = set([c for c in ''.join(plates_data['nums'])])
char_list = list(char_set)
char_list.sort()

char_dict = {char: idx for idx, char in enumerate(char_list)}

inv_char_dict = {idx: char for idx, char in enumerate(char_list)}
inv_char_dict[NUM_CHARS] = ''
class PlatesDataset(data.Dataset):
    def __init__(self, path, char_dict, train=True, pad_to = NUM_LENGTH):
        super().__init__()
        with open(path, 'rb') as fp:
            self.data = pickle.load(fp)
            if train:
                begin, end = 0, int(0.8*len(self.data['img_bytes']))
            else:
                begin, end = int(0.8*len(self.data['img_bytes'])), len(self.data['img_bytes'])
            for key in ['img_bytes', 'nums']:
                self.data[key] = self.data[key][begin:end]
            
                
        self.char_dict = char_dict
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.train = train
        self.pad_to = pad_to
        
    def __len__(self):
        return len(self.data['img_bytes'])
    
    def __getitem__(self, idx):
        img = Image.frombytes(self.data['img_mode'], self.data['img_size'], self.data['img_bytes'][idx])
        target = self.data['nums'][idx]
        target = [self.char_dict[char] for char in target]
        target = target + [len(self.char_dict) for _ in range(self.pad_to - len(target))]
        
        if self.train:
            x_offset = np.random.randint(0, 1 + self.data['img_size'][0] - REC_IMG_SIZE[0])
            y_offset = np.random.randint(0, 1 + self.data['img_size'][1] - REC_IMG_SIZE[1])
        else:
            x_offset = (self.data['img_size'][0] - REC_IMG_SIZE[0])/2
            y_offset = (self.data['img_size'][1] - REC_IMG_SIZE[1])/2
        
        crop_box = np.array((0,0) + REC_IMG_SIZE) + np.array((x_offset, y_offset)*2)
        img = img.crop(crop_box)
        
        img = self.img_transform(img)
        target = torch.LongTensor(target)
        return img, target
train_dataset = PlatesDataset('plates.pickle', char_dict = char_dict)
train_loader = data.DataLoader(train_dataset, batch_size=128, num_workers=8, shuffle=True, drop_last=True)

val_dataset = PlatesDataset('plates.pickle', char_dict = char_dict, train=False)
val_loader = data.DataLoader(val_dataset, batch_size=128, num_workers=8)
# the extra layer made a huge difference in my experiments

rec = torchvision.models.resnet34(pretrained=True)
rec.fc = nn.Sequential(
    nn.Linear(512, 2048, bias=False),
    nn.BatchNorm1d(2048),
    nn.LeakyReLU(),
    nn.Dropout(),
    nn.Linear(2048, (NUM_CHARS + 1)*NUM_LENGTH)
)
rec.avgpool = nn.AdaptiveMaxPool2d((1,1))
rec = rec.cuda()
opt = torch.optim.Adam(rec.parameters(), lr=1e-3)
# around 10-15 epochs should be enough

n_epochs = 50

best_val_loss = np.inf

train_loss_hist = []
val_loss_hist = []
for ep in range(n_epochs):
    rec = rec.train()
    train_loss = []
    for batch in train_loader:
        inputs, targets = batch
        inputs = inputs.cuda()
        targets = targets.cuda().reshape(-1)

        outputs = rec(inputs).reshape(-1, (NUM_CHARS + 1))

        loss = F.cross_entropy(outputs, targets)
        train_loss.append(loss.item())

        opt.zero_grad()
        loss.backward()
        opt.step()
    train_loss = np.array(train_loss).mean()
    train_loss_hist.append(train_loss)
    
    rec = rec.eval()
    val_loss = []
    for batch in val_loader:
        inputs, targets = batch
        inputs = inputs.cuda()
        targets = targets.cuda().reshape(-1)

        with torch.no_grad():
            outputs = rec(inputs).reshape(-1, (NUM_CHARS + 1))

        loss = F.cross_entropy(outputs, targets)
        val_loss.append(loss.item())
    val_loss = np.array(val_loss).mean()
    val_loss_hist.append(val_loss)
    
    if val_loss < best_val_loss:
        torch.save(rec.state_dict(), 'recog_checkpoint.pth')
        
    print(f"Train: {train_loss_hist[-1]:.3f}, val: {val_loss_hist[-1]:.3f}")
def decode(idx_list):
    return ''.join([inv_char_dict[int(idx)] for idx in idx_list])
rec = rec.eval()

lev_distances = []
mean_scores = []
for batch in val_loader:
    inputs, targets = batch
    inputs = inputs.cuda()
    
    true_nums = [decode(idx_list) for idx_list in targets]
    
    with torch.no_grad():
        scores, chars = rec(inputs).reshape(-1, NUM_LENGTH, (NUM_CHARS+1)).max(dim=2)
        
    mean_scores.append(scores.mean(dim=1))
    
    pred_nums = [decode(idx_list) for idx_list in chars]

    for num_1, num_2 in zip(true_nums, pred_nums):
        lev_distances.append(Levenshtein.distance(num_1, num_2))
np.array(lev_distances).mean()
mean_scores = torch.cat(mean_scores).cpu().numpy()
plt.scatter(mean_scores, lev_distances)
plt.show()
def box_to_rect(box):
    box = np.array(box)
    x_min, y_min = box.min(axis=0)
    x_max, y_max = box.max(axis=0)
    return [x_min, y_min, x_max, y_max]

def crop_on_plate(image, boxes, crop_size):
    box = boxes[np.random.randint(0, len(boxes))]
    x_c, y_c = (box[0] + box[2])/2, (box[1] + box[3])/2
    x_c = x_c - min(0, x_c - crop_size[0]/2) - min(0, image.size[0] - x_c - crop_size[0]/2)
    y_c = y_c - min(0, y_c - crop_size[1]/2) - min(0, image.size[1] - y_c - crop_size[1]/2)
    
    crop_region = [x_c - crop_size[0]/2, y_c - crop_size[1]/2, x_c + crop_size[0]/2, y_c + crop_size[1]/2]
    
    image = image.crop(crop_region)
    boxes = boxes - np.array(crop_region[:2]*2)

    return image, boxes

def collate_fn(batch):
    inputs = []
    targets = []
    for el in batch:
        inputs.append(el[0])
        targets.append(el[1])
    inputs = torch.stack(inputs, dim=0)
    
    return inputs, targets

class DetectionDataset(data.Dataset):
    def __init__(self, root, crop_size, train=True):
        self.root = root
        self.annot = pd.read_json(os.path.join(root,'train.json'))[:-1]
        self.crop_size = crop_size
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        if train:
            self.annot = self.annot[:int(0.8*len(self.annot))]
        else:
            self.annot = self.annot[int(0.8*len(self.annot)):]
        
    def __len__(self):
        return len(self.annot)
    
    def __getitem__(self, idx):
        boxes, im_path = self.annot.iloc[idx]
        im_path = os.path.join(self.root, im_path)
        image = Image.open(im_path).convert('RGB')
        boxes = [box_to_rect(b['box']) for b in boxes]
        
        image, boxes = crop_on_plate(image, boxes, self.crop_size)
        
        #we squeeze the image and the bounding boxes in the x dimension
        #here is the reasoning: the majority of licence plates' bounding boxes are very thin, and
        #they match poorly to the anchors used in the model. 
        #Hence squeezing the image brings their aspect ratio closer to 1, making them more anchor-friendly.
        boxes[:, 0] = boxes[:, 0]/SQUEEZE_FACTOR
        boxes[:, 2] = boxes[:, 2]/SQUEEZE_FACTOR
        image = image.resize((int(image.size[0]/SQUEEZE_FACTOR), image.size[1]))
        
        targets = {
            'boxes': torch.FloatTensor(boxes),
            'labels': torch.ones(len(boxes)).long(),
        }
        
        image = self.img_transform(image)
        
        return image, targets
train_dataset = DetectionDataset('data', crop_size=(1000,400))
train_loader = data.DataLoader(train_dataset, batch_size=32, num_workers=8, shuffle=True, drop_last=True,
                              collate_fn=collate_fn)

val_dataset = DetectionDataset('data', crop_size=(1000,400), train=False)
val_loader = data.DataLoader(val_dataset, batch_size=16, num_workers=8, collate_fn=collate_fn)
#the model does some internal image resizing controlled by min_size and max_size parameters.
#Numbers used here make sure this resizing has no effect.

det = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, min_size=400)
det.roi_heads.box_predictor = FastRCNNPredictor(det.roi_heads.box_predictor.cls_score.in_features, 2)
# for p in model.parameters():
#     p.requires_grad = False
# for p in model.backbone.fpn.parameters():
#     p.requires_grad = True
# for p in model.rpn.parameters():
#     p.requires_grad = True
# for p in model.roi_heads.parameters():
#     p.requires_grad = True
    
det = det.cuda()
opt = torch.optim.Adam(det.parameters(), lr=1e-3)
#after a couple of epochs the loss more or less stabilizes
#TODO: a proper validation

n_epochs = 1

train_loss_hist = []
for ep in range(n_epochs):
    det = det.train()
    train_loss = []
    for b_num, batch in enumerate(train_loader):
        inputs, targets = batch
        inputs = inputs.cuda()
        
        for target in targets:
            for key, val in target.items():
                target[key] = val.cuda()

        loss_dict = det(inputs, targets)

        loss = sum([l for l in loss_dict.values()])
        print(f"it {b_num + 1}/{len(train_loader)}, loss: {loss:.3f}")
        train_loss.append(loss.item())

        opt.zero_grad()
        loss.backward()
        opt.step()
    train_loss = np.array(train_loss).mean()
    train_loss_hist.append(train_loss)
    
    torch.save(det.state_dict(), 'detect_checkpoint.pth')
#rec.load_state_dict(torch.load('recog_checkpoint.pth'))
#det.load_state_dict(torch.load('detect_checkpoint.pth'))

rec = rec.eval().cuda()
det = det.eval().cuda()

rec_transform = transforms.Compose([
    transforms.Resize((34,98)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

files = os.listdir(os.path.join('data','test'))
files = {int(file.split('.')[0]): file for file in files}

f_name_list = []
nums_list = []
for ind in tqdm.tqdm(range(len(files))):
    f_name = 'test/' + files[ind]
    img = Image.open(os.path.join('.', 'data', f_name)).convert('RGB')
    
    #squeezing the input image for detection
    img_for_detection = img.resize((int(img.size[0]/SQUEEZE_FACTOR), img.size[1]))
    with torch.no_grad():
        det_results = det(transforms.functional.to_tensor(img_for_detection)[None,...].cuda())[0]
    boxes = det_results['boxes']
    scores = det_results['scores']
    boxes = boxes[scores > .95].cpu().numpy()
    lower_left = boxes[:,0].copy()
    boxes = boxes[lower_left.argsort()]
    
    #restoring the aspect ratio of the bounding boxes
    boxes[:,0] = boxes[:,0]*SQUEEZE_FACTOR
    boxes[:,2] = boxes[:,2]*SQUEEZE_FACTOR
    
    nums = []
    for box in boxes:
        plate = img.crop(box)
        plate = rec_transform(plate)[None,...].cuda()
        with torch.no_grad():
            rec_scores, rec_chars = rec(plate).reshape(-1, NUM_LENGTH, (NUM_CHARS+1)).max(dim=2)
        rec_chars = decode(rec_chars[0])
        
        #dropping low recognition score plates
        if rec_scores.mean() > REC_SCORE_CUTOFF:
            nums.append(rec_chars)
    
    f_name_list.append(f_name)
    nums_list.append(' '.join(nums))
df_submit = pd.DataFrame({'file_name': f_name_list, 'plates_string': nums_list})
df_submit.to_csv('submit.csv', index=False)