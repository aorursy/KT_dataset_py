from copy import deepcopy
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import tqdm
from torch.utils import data
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.nn import functional as fnn

import matplotlib.pyplot as plt
import matplotlib.patches as patches

np.random.seed(2205)
torch.manual_seed(2205)
class CarPlatesDatasetWithRectangularBoxes(data.Dataset):
    def __init__(self, root, transforms, split='train', train_size=0.9):
        super(CarPlatesDatasetWithRectangularBoxes, self).__init__()
        self.root = Path(root)
        self.train_size = train_size
        
        self.image_names = []
        self.image_ids = []
        self.image_boxes = []
        self.image_texts = []
        self.box_areas = []
        
        self.transforms = transforms
        
        if split in ['train', 'val']:
            plates_filename = self.root / 'train.json'
            with open(plates_filename) as f:
                json_data = json.load(f)
            train_valid_border = int(len(json_data) * train_size) + 1 # граница между train и valid
            data_range = (0, train_valid_border) if split == 'train' \
                else (train_valid_border, len(json_data))
            self.load_data(json_data[data_range[0]:data_range[1]]) # загружаем названия файлов и разметку
            return

        if split == 'test':
            plates_filename = self.root / 'submission.csv'
            self.load_test_data(plates_filename, split, train_size)
            return

        raise NotImplemented(f'Unknown split: {split}')
        
    def load_data(self, json_data):
        for i, sample in enumerate(json_data):
            if sample['file'] == 'train/25632.bmp':
                continue
            self.image_names.append(self.root / sample['file'])
            self.image_ids.append(torch.Tensor([i]))
            boxes = []
            texts = []
            areas = []
            for box in sample['nums']:
                points = np.array(box['box'])
                x_0 = np.min([points[0][0], points[3][0]])
                y_0 = np.min([points[0][1], points[1][1]])
                x_1 = np.max([points[1][0], points[2][0]])
                y_1 = np.max([points[2][1], points[3][1]])
                boxes.append([x_0, y_0, x_1, y_1])
                texts.append(box['text'])
                areas.append(np.abs(x_0 - x_1) * np.abs(y_0 - y_1))
            boxes = torch.FloatTensor(boxes)
            areas = torch.FloatTensor(areas)
            self.image_boxes.append(boxes)
            self.image_texts.append(texts)
            self.box_areas.append(areas)
        
    
    def load_test_data(self, plates_filename, split, train_size):
        df = pd.read_csv(plates_filename, usecols=['file_name'])
        for row in df.iterrows():
            self.image_names.append(self.root / row[1][0])
        self.image_boxes = None
        self.image_texts = None
        self.box_areas = None
         
    
    def __getitem__(self, idx):
        target = {}
        if self.image_boxes is not None:
            boxes = self.image_boxes[idx].clone()
            areas = self.box_areas[idx].clone()
            num_boxes = boxes.shape[0]
            target['boxes'] = boxes
            target['area'] = areas
            target['labels'] = torch.LongTensor([1] * num_boxes)
            target['image_id'] = self.image_ids[idx].clone()
            target['iscrowd'] = torch.Tensor([False] * num_boxes)
#             target['texts'] = self.image_texts[idx]

        image = cv2.imread(str(self.image_names[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            image = self.transforms(image)
        return image, target

    def __len__(self):
        return len(self.image_names)
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def create_model(device):
    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    num_classes = 2  # 1 class (person) + background
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model.to(device)

# Вспомогательная функция для создания dataloader'а
def collate_fn(batch):
    return tuple(zip(*batch))
transformations= transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                    ])
# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = create_model(device)

# use our dataset and defined transformations
train_dataset = CarPlatesDatasetWithRectangularBoxes('data', transformations, 'train')
val_dataset = CarPlatesDatasetWithRectangularBoxes('data', transformations, 'val')
test_dataset = CarPlatesDatasetWithRectangularBoxes('data', transformations, 'test')

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=2, shuffle=True, num_workers=4,
    collate_fn=collate_fn)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=2, shuffle=False, num_workers=4,
    collate_fn=collate_fn)
# Часть кода взята из  pytorch utils
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)

num_epochs = 1

for epoch in range(num_epochs):
    model.train()

    for images, targets in tqdm.tqdm(train_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
    
    batch_losses = []
    for images, targets in tqdm.tqdm(val_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        batch_losses.append(losses.item())
        optimizer.zero_grad()
    
    batch_losses = np.array(batch_losses)
    batch_losses = batch_losses[np.isfinite(batch_losses)]
    print(f'Valid_loss: {np.mean(batch_losses)}')
    lr_scheduler.step()

print("That's it!")
print(f'Средний лосс на валидации: {np.mean(batch_losses)}')
# Сохранили
# with open('fasterrcnn_resnet50_fpn_1_epoch', 'wb') as fp:
#     torch.save(model.state_dict(), fp)
# Загрузили
with open('fasterrcnn_resnet50_fpn_1_epoch', 'rb') as fp:
    state_dict = torch.load(fp, map_location="cpu")
model.load_state_dict(state_dict)
model.to(device)
unnormalize_1 = transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                         std=[1, 1, 1])
unnormalize_2 = transforms.Normalize(mean=[0, 0, 0],
                                         std=[1/0.229, 1/0.224, 1/0.225])
unnormalize = transforms.Compose([unnormalize_2, unnormalize_1])

start = 2

images = []
for i in range(start, start + 2):
    images.append(val_dataset[i][0].to(device))
def detach_dict(pred):
    return{k:v.detach().cpu() for (k,v) in pred.items()}

model.eval()
preds = model(images)
preds = [detach_dict(pred) for pred in preds]
preds
fig,ax = plt.subplots(1, 2, figsize = (20, 8))

for i in range(2):
    image = unnormalize(images[i].clone().cpu())
    ax[i].imshow(image.numpy().transpose([1,2,0]))
    for box in preds[i]['boxes']:
        box = box.detach().cpu().numpy()
        rect = patches.Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],linewidth=1,edgecolor='r',facecolor='none')
        ax[i].add_patch(rect)

plt.show()
class CarPlatesFragmentsDataset(data.Dataset):
    def __init__(self, root, transforms, split='train', train_size=0.9):
        super(CarPlatesFragmentsDataset, self).__init__()
        self.root = Path(root)
        self.train_size = train_size
        
        self.image_names = []
        self.image_ids = []
        self.image_boxes = []
        self.image_texts = []
        self.box_areas = []
        
        self.transforms = transforms
        
        if split in ['train', 'val']:
            plates_filename = self.root / 'train.json'
            with open(plates_filename) as f:
                json_data = json.load(f)
            train_valid_border = int(len(json_data) * train_size) + 1 # граница между train и valid
            data_range = (0, train_valid_border) if split == 'train' \
                else (train_valid_border, len(json_data))
            self.load_data(json_data[data_range[0]:data_range[1]]) # загружаем названия файлов и разметку
            return

        if split == 'test':
            plates_filename = self.root / 'test_boxes.json'
            with open(plates_filename) as f:
                json_data = json.load(f)
            self.load_test_data(json_data)
            return
            
        raise NotImplemented(f'Unknown split: {split}')
        
    def load_data(self, json_data):
        for i, sample in enumerate(json_data):
            if sample['file'] == 'train/25632.bmp':
                continue
            for box in sample['nums']:
                points = np.array(box['box'])
                x_0 = np.min([points[0][0], points[3][0]])
                y_0 = np.min([points[0][1], points[1][1]])
                x_1 = np.max([points[1][0], points[2][0]])
                y_1 = np.max([points[2][1], points[3][1]])
                if x_0 > x_1 or y_0 > y_1:
                    # Есть несколько примеров, когда точки пронумерованы в другом порядке - пока не выясняем
                    continue
                self.image_boxes.append(np.clip([x_0, y_0, x_1, y_1], a_min=0, a_max=None))
                self.image_texts.append(box['text'])
                self.image_names.append(self.root / sample['file'])
                
    def load_test_data(self, json_data):
        for i, sample in enumerate(json_data):
            for box in sample['boxes']:
                if box[0] >= box[2] or box[1] >= box[3]:
                    continue
                points = np.array(box)
                self.image_boxes.append(np.clip(points, a_min=0, a_max=None))
                self.image_names.append(sample['file'])
        self.image_texts = None
    
    def __getitem__(self, idx):
        target = {}
        image = cv2.imread(str(self.image_names[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        target = {}
        if self.image_boxes is not None:
            box = self.image_boxes[idx]
            image = image.copy()[box[1]:box[3], box[0]:box[2]]
            
        if self.image_texts is not None:
            target['text'] = self.image_texts[idx]

        if self.transforms is not None:
            image = self.transforms(image)
        return image, target

    def __len__(self):
        return len(self.image_names)
transformations= transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                    ])

train_dataset = CarPlatesFragmentsDataset('data', transformations, 'train')
# Фунции для поиска отдельных символов в номере

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
 
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
 
    # return the edged image
    return edged

def find_number_boxes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh_inv = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,39,1)
    edges = auto_canny(thresh_inv)
    ctrs, hier = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
    img_area = image.shape[0]*image.shape[1]
    boxes = []
    for i, ctr in enumerate(sorted_ctrs):
        x, y, w, h = cv2.boundingRect(ctr)
        roi_area = w*h
        roi_ratio = roi_area/img_area
        if((roi_ratio >= 0.015) and (roi_ratio < 0.09)):
            if ((h>1.2*w) and (3*w>=h)):
                boxes.append([x, y, w, h])
    # Оставим только уникальные
    unique_boxes = []
    for box in boxes:
        if box not in unique_boxes:
            unique_boxes.append(box)
    # Уберем вложенные и сильнопересекающиеся
    box_num = len(unique_boxes)
    valid_boxes = [True] * box_num
    i = 0
    while i < box_num:
        if not valid_boxes[i]:
            i += 1
            continue
        area_i = unique_boxes[i][2] * unique_boxes[i][3]
        j = i + 1
        while j < box_num:
            if not valid_boxes[j]:
                j += 1
                continue
            # Находим пересечения
            left = max(unique_boxes[i][0], unique_boxes[j][0])
            right = min(unique_boxes[i][0] + unique_boxes[i][2], unique_boxes[j][0] + unique_boxes[j][2])
            top = max(unique_boxes[i][1], unique_boxes[j][1])
            bottom = min(unique_boxes[i][1] + unique_boxes[i][3], unique_boxes[j][1] + unique_boxes[j][3])
            if left >= right or top >= bottom:
                j += 1
                continue
            intersection_area = (right - left) * (bottom - top)
            area_j = unique_boxes[j][2] * unique_boxes[j][3]
            share_i = intersection_area / area_i
            share_j = intersection_area / area_j
            if share_i >= share_j:
                if share_i > 0.75:
                    valid_boxes[i] = False
            else:
                if share_j > 0.75:
                    valid_boxes[j] = False
            j += 1
        i += 1
    boxes = []
    for i, box in enumerate(unique_boxes):
        if valid_boxes[i]:
            boxes.append(box)
        
    return boxes
train_dataset = CarPlatesFragmentsDataset('data', transformations, 'train', train_size=1)
idx = 26793
img_for_search = train_dataset[idx][0].clone().cpu()
img_for_search = unnormalize(img_for_search).numpy().transpose([1,2,0])
img_for_search = (img_for_search * 255).astype(np.uint8)
boxes = find_number_boxes(img_for_search)
print(train_dataset[idx][1]['text'])
image = unnormalize(train_dataset[idx][0].clone().cpu())
fig, ax = plt.subplots(1)
ax.imshow(image.numpy().transpose([1,2,0]))
for box in boxes:
    rect = patches.Rectangle((box[0],box[1]),box[2],box[3],linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
train_dataset = CarPlatesFragmentsDataset('data', transformations, 'train', train_size=1)
valid_number_of_letters = 0
for sample in tqdm.tqdm(train_dataset):
    text = sample[1]['text']
    img_for_search = unnormalize(sample[0].clone().cpu()).numpy().transpose([1,2,0])
    img_for_search = (img_for_search * 255).astype(np.uint8)
    boxes = find_number_boxes(img_for_search)
    if len(text) == len(boxes):
        valid_number_of_letters += 1
print(f'Всего изображений, где нашли правильное количество символов {valid_number_of_letters} из {len(train_dataset)}')
path = Path('data')
symbols_path = path / 'symbols'
symbols_path.mkdir(exist_ok=True)
num = 0
labels = {}
for sample in tqdm.tqdm(train_dataset):
    text = sample[1]['text']
    img_for_search = unnormalize(sample[0].clone().cpu()).numpy().transpose([1,2,0])
    img_for_search = (img_for_search * 255).astype(np.uint8)
    boxes = find_number_boxes(img_for_search)
    if len(text) == len(boxes):
        for i in range(len(boxes)):
            file_name = f'{num}.jpg'
            labels[file_name] = text[i]
            box = boxes[i]
            symbol = img_for_search.copy()[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
            cv2.imwrite(str(symbols_path / file_name), symbol )
            num += 1
labels_df = pd.DataFrame(list(labels.items()), columns=['file_name', 'label'])

labels_df.loc[labels_df.label == 'А', 'label'] = 'A'
labels_df.loc[labels_df.label == 'О', 'label'] = 'O'
labels_df.loc[labels_df.label == 'Н', 'label'] = 'H'
labels_df.loc[labels_df.label == 'К', 'label'] = 'K'
labels_df.loc[labels_df.label == 'С', 'label'] = 'C'
labels_df.loc[labels_df.label == 'Р', 'label'] = 'P'
labels_df.loc[labels_df.label == 'В', 'label'] = 'B'
labels_df.loc[labels_df.label == 'Х', 'label'] = 'X'
labels_df.loc[labels_df.label == 'Е', 'label'] = 'E'
labels_df.loc[labels_df.label == 'Т', 'label'] = 'T'
labels_df.loc[labels_df.label == 'М', 'label'] = 'M'
labels_df.loc[labels_df.label == 'е', 'label'] = 'E'
labels_df.loc[labels_df.label == 'o', 'label'] = 'O'
labels_df.loc[labels_df.label == 'м', 'label'] = 'M'
labels_df.loc[labels_df.label == 'e', 'label'] = 'E'
labels_df.loc[labels_df.label == 'к', 'label'] = 'K'
labels_df.loc[labels_df.label == 'У', 'label'] = 'Y'
labels_df.loc[labels_df.label == 'о', 'label'] = 'O'
labels_df.loc[labels_df.label == 'в', 'label'] = 'B'
labels_df.loc[labels_df.label == 'y', 'label'] = 'Y'

labels_df.to_csv('data/symbols.csv', index=False)
print(f'Всего разных символов: {labels_df.label.nunique()}')
class SymbolsDataset(data.Dataset):
    def __init__(self, root, transforms, split='train', train_size=0.9):
        super(SymbolsDataset, self).__init__()
        self.root = Path(root)
        self.image_path = self.root / 'symbols'
        self.train_size = train_size
        self.label_to_class = {}
        self.class_to_label = {}
        self.data = None
        
        self.transforms = transforms
        
        symbols_filename = self.root / 'symbols.csv'
        labels_df = pd.read_csv(symbols_filename)
        for i, ch in enumerate(labels_df.label.unique()):
            self.label_to_class[ch] = i
            self.class_to_label[i] = ch
        
        if split in ['train', 'val']:
            train_valid_border = int(len(labels_df) * train_size) + 1 # граница между train и valid
            if split == 'train':
                self.data = labels_df.iloc[:train_valid_border]
            else:
                self.data = labels_df.iloc[train_valid_border:]
            return

        raise NotImplemented(f'Unknown split: {split}')
    
    def __getitem__(self, idx):
        target = {}
        file_name = self.image_path / self.data.iloc[idx].file_name
        label = self.data.iloc[idx].label
        image = cv2.imread(str(file_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #COLOR_BGR2GRAY
        image = cv2.resize(image, (75, 100))
        
        if self.transforms is not None:
            image = self.transforms(image)
        return image, torch.LongTensor([self.label_to_class[label]])

    def __len__(self):
        return len(self.data)
symbol_transforms= transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
                    ])
symbol_train_dataset = SymbolsDataset('data', symbol_transforms, 'train')
symbol_train_dataloader = torch.utils.data.DataLoader(
        symbol_train_dataset, batch_size=64, shuffle=True, num_workers=4)

symbol_val_dataset = SymbolsDataset('data', symbol_transforms, 'val')
symbol_val_dataloader = torch.utils.data.DataLoader(
        symbol_val_dataset, batch_size=64, shuffle=False, num_workers=4)
batch = next(iter(symbol_train_dataloader))
fig, axes = plt.subplots(4, 4, figsize=(20, 20))
for i, sample in enumerate(list(zip(*batch))[:16]):
    axes[i // 4, i % 4].imshow(sample[0].numpy().transpose([1,2,0]))
    axes[i // 4, i % 4].set_title(symbol_train_dataset.class_to_label[sample[1].item()])
def create_symbol_classifier(num_symbols, devide):
    model = models.alexnet(pretrained=False)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_symbols, bias=True)
    return model.to(device)
num_symbols = 22
classifier = create_symbol_classifier(num_symbols, device)
optimizer = optim.Adam(classifier.parameters(), lr=1e-4, amsgrad=True)
loss_fn = fnn.cross_entropy
num_epochs = 5
for epoch in range(num_epochs):
    print(f'Epoch #{epoch + 1}')
    time.sleep(0.5)
    train_loss = []
    model.train()
    for batch in tqdm.tqdm(symbol_train_dataloader, total=len(symbol_train_dataloader), desc="training..."):
        images = batch[0].to(device)
        classes = batch[1].squeeze()

        pred_classes = classifier(images).cpu()
        loss = loss_fn(pred_classes, classes, reduction="mean")
        train_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Train loss: {np.mean(train_loss):0.4f}')
    model.eval()
    valid_loss = []
    correct_predictions = 0
    for batch in tqdm.tqdm(symbol_val_dataloader, total=len(symbol_val_dataloader), desc="validation..."):
        images = batch[0].to(device)
        classes = batch[1].squeeze()
        pred_classes = classifier(images).cpu()
        loss = loss_fn(pred_classes, classes, reduction="mean")
        valid_loss.append(loss.item())
        
        pred_class = torch.argmax(pred_classes, axis=1)
        correct_predictions += torch.sum(classes == pred_class).item()
    print(f'Valid loss: {np.mean(valid_loss):0.4f}')
    print(f'Validation accuracy :{correct_predictions / len(symbol_val_dataloader):0.2f}, correct: {correct_predictions}')
batch = next(iter(symbol_train_dataloader))
output = classifier(batch[0].to(device))
predictions = torch.argmax(output, axis=1).cpu().numpy()
ground_true = batch[1].squeeze().numpy()
predictions = [symbol_val_dataset.class_to_label[c] for c in predictions]
ground_true = [symbol_val_dataset.class_to_label[c] for c in ground_true]
fig, axes = plt.subplots(4, 4, figsize=(20, 20))
for i, sample in enumerate(list(zip(*batch))[:16]):
    axes[i // 4, i % 4].imshow(sample[0].numpy().transpose([1,2,0]))
    axes[i // 4, i % 4].set_title(f'Pred: {predictions[i]} True: {ground_true[i]}')
# Сохранили
# with open('alexnet_symbol_classifier', 'wb') as fp:
#     torch.save(classifier.state_dict(), fp)
# Загрузили
with open('alexnet_symbol_classifier', 'rb') as fp:
    state_dict = torch.load(fp, map_location="cpu")
classifier.load_state_dict(state_dict)
classifier.to(device)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Модель для нахождения номерных знаков на фотографиях
model = create_model(device)

# Загрузили
with open('fasterrcnn_resnet50_fpn_1_epoch', 'rb') as fp:
    state_dict = torch.load(fp, map_location="cpu")
model.load_state_dict(state_dict)
model.to(device)

# Test dataset
test_dataset = CarPlatesDatasetWithRectangularBoxes('data', transformations, 'test')

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=2, shuffle=False, num_workers=4,
    collate_fn=collate_fn)
predicted_boxes = []
for batch in tqdm.tqdm(test_loader):
    images = list(image.to(device) for image in batch[0])
    model.eval()
    preds = model(images)
    preds = [{k: v.detach().cpu().numpy() for k, v in prediction.items()} for prediction in preds]
    predicted_boxes.extend(preds)
boxes = [box.astype(int).tolist() for box in (boxes_in_image['boxes'] for boxes_in_image in predicted_boxes)]
assert len(boxes) == len(test_dataset)
json_data = []
for file_name, box in zip(test_dataset.image_names, boxes):
    json_data.append({'boxes': box, 'file': str(file_name)})

with open('data/test_boxes.json', 'w') as fp:
    json.dump(json_data, fp)
transformations = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                    ])

test_plates_dataset = CarPlatesFragmentsDataset('data', transformations, 'test')
idx = 0
img_for_search = test_plates_dataset[idx][0].clone().cpu()
img_for_search = unnormalize(img_for_search).numpy().transpose([1,2,0])
img_for_search = (img_for_search * 255).astype(np.uint8)
boxes = find_number_boxes(img_for_search)
image = unnormalize(test_plates_dataset[idx][0].clone().cpu())
fig, ax = plt.subplots(1)
ax.imshow(image.numpy().transpose([1,2,0]))
for box in boxes:
    rect = patches.Rectangle((box[0],box[1]),box[2],box[3],linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
num_symbols = 22
classifier = create_symbol_classifier(num_symbols, device)

with open('alexnet_symbol_classifier', 'rb') as fp:
    state_dict = torch.load(fp, map_location="cpu")
classifier.load_state_dict(state_dict)
classifier.to(device)
transformations = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                    ])
symbol_val_dataset = SymbolsDataset('data', None, 'val')
class_to_label = symbol_val_dataset.class_to_label
submit = {}
for idx in tqdm.tqdm(range(len(test_plates_dataset))):
    img_for_search = test_plates_dataset[idx][0].clone().cpu()
    file_name = test_plates_dataset.image_names[idx][5:]
    img_for_search = unnormalize(img_for_search).numpy().transpose([1,2,0])
    img_for_search = (img_for_search * 255).astype(np.uint8)
    symbols = find_number_boxes(img_for_search)
    symbols_array = []
    if file_name not in submit:
            submit[file_name] = []
    if symbols:
        for symbol in symbols:
            symbol_img = img_for_search.copy()[symbol[1]:symbol[1]+symbol[3], symbol[0]:symbol[0]+symbol[2]]
            symbol_img = cv2.resize(symbol_img, (75, 100))
            symbols_array.append(transformations(symbol_img))
        batch = torch.stack(symbols_array).to(device)
        output = classifier(batch)
        predictions = torch.argmax(output, axis=1).cpu().numpy()
        plate = ''.join([class_to_label[p] for p in predictions])
        submit[file_name].append(plate)
submit = [(k, ' '.join(v)) for k,v in submit.items()]
submission = pd.DataFrame(submit, columns=['file_name', 'plates_string'])
random_submission = pd.read_csv('submission.csv')
submission = pd.merge(random_submission, submission, how='left', on='file_name')
submission.drop('plates_string_x', axis=1, inplace=True)
submission.columns = ['file_name', 'plates_string']
submission
submission.to_csv('baseline.csv', index=False)