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
print(f'Validation average loss: {np.mean(batch_losses)}')
# Save
# with open('fasterrcnn_resnet50_fpn_1_epoch', 'wb') as fp:
#     torch.save(model.state_dict(), fp)
# Load
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
    def __init__(self, root, transforms, split='train', train_size=0.9, alphabet=abc):
        super(CarPlatesFragmentsDataset, self).__init__()
        self.root = Path(root)
        self.alphabet = alphabet
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
                if x_0 >= x_1 or y_0 >= y_1:
                    # Есть несколько примеров, когда точки пронумерованы в другом порядке - пока не выясняем
                    continue
                if (y_1 - y_0) * 20 < (x_1 - x_0):
                    continue
                self.image_boxes.append(np.clip([x_0, y_0, x_1, y_1], a_min=0, a_max=None))
                self.image_texts.append(box['text'])
                self.image_names.append(sample['file'])
        self.revise_texts()
                
    def revise_texts(self):
        wrong = 'АОНКСРВХЕТМУ'
        correct = 'AOHKCPBXETMY'
        for i in range(len(self.image_texts)):
            self.image_texts[i] = self.image_texts[i].upper()
            for (a, b) in zip(wrong, correct):
                self.image_texts[i] = self.image_texts[i].replace(a, b)
            
                
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
        file_name = self.root / self.image_names[idx]
        image = cv2.imread(str(file_name))
        if image is None:
            file_name = self.image_names[idx]
            image = cv2.imread(str(file_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        text = ''
        
        if self.image_boxes is not None:
            box = self.image_boxes[idx]
            image = image.copy()[box[1]:box[3], box[0]:box[2]]
            
        if self.image_texts is not None:
            text = self.image_texts[idx]
            
        seq = self.text_to_seq(text)
        seq_len = len(seq)

        output = dict(image=image, seq=seq, seq_len=seq_len, text=text, file_name=file_name)
        
        if self.transforms is not None:
            output = self.transforms(output)
        
        return output
    
    def text_to_seq(self, text):
        """Encode text to sequence of integers.
        Accepts string of text.
        Returns list of integers where each number is index of corresponding characted in alphabet + 1.
        """
        # YOUR CODE HERE
        seq = [self.alphabet.find(c) + 1 for c in text]
        return seq

    def __len__(self):
        return len(self.image_names)
class FeatureExtractor(nn.Module):
    
    def __init__(self, input_size=(64, 320), output_len=20):
        super(FeatureExtractor, self).__init__()
        
        h, w = input_size
        resnet = getattr(models, 'resnet18')(pretrained=True)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])
        
        self.pool = nn.AvgPool2d(kernel_size=(h // 32, 1))        
        self.proj = nn.Conv2d(w // 32, output_len, kernel_size=1)
  
        self.num_output_features = self.cnn[-1][-1].bn2.num_features    
    
    def apply_projection(self, x):
        """Use convolution to increase width of a features.
        Accepts tensor of features (shaped B x C x H x W).
        Returns new tensor of features (shaped B x C x H x W').
        """
        # YOUR CODE HERE
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        return x
   
    def forward(self, x):
        # Apply conv layers
        features = self.cnn(x)
        
        # Pool to make height == 1
        features = self.pool(features)
        
        # Apply projection to increase width
        features = self.apply_projection(features)
        
        return features
class SequencePredictor(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3, bidirectional=False):
        super(SequencePredictor, self).__init__()
        
        self.num_classes = num_classes        
        self.rnn = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          dropout=dropout,
                          bidirectional=bidirectional)
        
        fc_in = hidden_size if not bidirectional else 2 * hidden_size
        self.fc = nn.Linear(in_features=fc_in,
                            out_features=num_classes)
    
    def _init_hidden_(self, batch_size):
        """Initialize new tensor of zeroes for RNN hidden state.
        Accepts batch size.
        Returns tensor of zeros shaped (num_layers * num_directions, batch, hidden_size).
        """
        # YOUR CODE HERE
        num_directions = 2 if self.rnn.bidirectional else 1
        return torch.zeros(self.rnn.num_layers * num_directions, batch_size, self.rnn.hidden_size)
        
    def _prepare_features_(self, x):
        """Change dimensions of x to fit RNN expected input.
        Accepts tensor x shaped (B x (C=1) x H x W).
        Returns new tensor shaped (W x B x H).
        """
        # YOUR CODE HERE
        x = x.squeeze(1)
        x = x.permute(2, 0, 1)
        return x
    
    def forward(self, x):
        x = self._prepare_features_(x)
        
        batch_size = x.size(1)
        h_0 = self._init_hidden_(batch_size)
        h_0 = h_0.to(x.device)
        x, h = self.rnn(x, h_0)
        
        x = self.fc(x)
        return x
abc = "0123456789ABEKMHOPCTYX"  # this is our alphabet for predictions.

class CRNN(nn.Module):
    
    def __init__(self, alphabet=abc,
                 cnn_input_size=(64, 320), cnn_output_len=20,
                 rnn_hidden_size=128, rnn_num_layers=2, rnn_dropout=0.3, rnn_bidirectional=False):
        super(CRNN, self).__init__()
        self.alphabet = alphabet
        self.features_extractor = FeatureExtractor(input_size=cnn_input_size, output_len=cnn_output_len)
        self.sequence_predictor = SequencePredictor(input_size=self.features_extractor.num_output_features,
                                                    hidden_size=rnn_hidden_size, num_layers=rnn_num_layers,
                                                    num_classes=len(alphabet)+1, dropout=rnn_dropout,
                                                    bidirectional=rnn_bidirectional)
    
    def forward(self, x):
        features = self.features_extractor(x)
        sequence = self.sequence_predictor(features)
        return sequence
def pred_to_string(pred, abc):
    seq = []
    for i in range(len(pred)):
        label = np.argmax(pred[i])
        seq.append(label - 1)
    out = []
    for i in range(len(seq)):
        if len(out) == 0:
            if seq[i] != -1:
                out.append(seq[i])
        else:
            if seq[i] != -1 and seq[i] != seq[i - 1]:
                out.append(seq[i])
    out = ''.join([abc[c] for c in out])
    return out

def decode(pred, abc):
    pred = pred.permute(1, 0, 2).cpu().data.numpy()
    outputs = []
    for i in range(len(pred)):
        outputs.append(pred_to_string(pred[i], abc))
    return outputs
class Resize(object):

    def __init__(self, size=(320, 64)):
        self.size = size

    def __call__(self, item):
        """Accepts item with keys "image", "seq", "seq_len", "text".
        Returns item with image resized to self.size.
        """
        # YOUR CODE HERE
        item['image'] = cv2.resize(item['image'], self.size, interpolation=cv2.INTER_AREA)
        return item
crnn = CRNN()
crnn.to(device)
num_epochs = 10
batch_size = 128
num_workers = 4
optimizer = torch.optim.Adam(crnn.parameters(), lr=3e-4, amsgrad=True, weight_decay=1e-4)
transformations = transforms.Compose([
    Resize(),
                    ])

train_plates_dataset = CarPlatesFragmentsDataset('data', transformations, 'train')
val_plates_dataset = CarPlatesFragmentsDataset('data', transformations, 'val')
def collate_fn(batch):
    """Function for torch.utils.data.Dataloader for batch collecting.
    Accepts list of dataset __get_item__ return values (dicts).
    Returns dict with same keys but values are either torch.Tensors of batched images, sequences, and so.
    """
    images, seqs, seq_lens, texts, file_names = [], [], [], [], []
    for sample in batch:
        images.append(torch.from_numpy(sample["image"]).permute(2, 0, 1).float())
        seqs.extend(sample["seq"])
        seq_lens.append(sample["seq_len"])
        texts.append(sample["text"])
        file_names.append(sample["file_name"])
    images = torch.stack(images)
    seqs = torch.Tensor(seqs).int()
    seq_lens = torch.Tensor(seq_lens).int()
    
    batch = {"image": images, "seq": seqs, "seq_len": seq_lens, "text": texts, "file_name": file_names}
    return batch
train_dataloader = torch.utils.data.DataLoader(train_plates_dataset, 
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers, pin_memory=True, 
                                               drop_last=True, collate_fn=collate_fn)
val_dataloader = torch.utils.data.DataLoader(val_plates_dataset, 
                                             batch_size=batch_size, shuffle=False,
                                             num_workers=num_workers, pin_memory=True, 
                                             drop_last=True, collate_fn=collate_fn)
crnn.train()
for i, epoch in enumerate(range(num_epochs)):
        epoch_losses = []

        for j, b in enumerate(tqdm.tqdm(train_dataloader, total=len(train_dataloader))):
            images = b["image"].to(device)
            seqs_gt = b["seq"]
            seq_lens_gt = b["seq_len"]

            seqs_pred = crnn(images).cpu()
            log_probs = fnn.log_softmax(seqs_pred, dim=2)
            seq_lens_pred = torch.Tensor([seqs_pred.size(0)] * seqs_pred.size(1)).int()

            loss = fnn.ctc_loss(log_probs=log_probs,  # (T, N, C)
                                targets=seqs_gt,  # N, S or sum(target_lengths)
                                input_lengths=seq_lens_pred,  # N
                                target_lengths=seq_lens_gt)  # N

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        print(i, np.mean(epoch_losses))
val_losses = []
crnn.eval()
for i, b in enumerate(tqdm.tqdm(val_dataloader, total=len(val_dataloader))):
    images = b["image"].to(device)
    seqs_gt = b["seq"]
    seq_lens_gt = b["seq_len"]

    with torch.no_grad():
        seqs_pred = crnn(images).cpu()
    log_probs = fnn.log_softmax(seqs_pred, dim=2)
    seq_lens_pred = torch.Tensor([seqs_pred.size(0)] * seqs_pred.size(1)).int()

    loss = fnn.ctc_loss(log_probs=log_probs,  # (T, N, C)
                        targets=seqs_gt,  # N, S or sum(target_lengths)
                        input_lengths=seq_lens_pred,  # N
                        target_lengths=seq_lens_gt)  # N

    val_losses.append(loss.item())

print(np.mean(val_losses))
y_ticks = ["-"] + [x for x in abc]

images = b["image"]
seqs_gt = b["seq"]
seq_lens_gt = b["seq_len"]
texts = b["text"]

preds = crnn(images.to(device)).cpu().detach()
texts_pred = decode(preds, crnn.alphabet)

for i in range(10):
    plt.figure(figsize=(15, 5))
    pred_i = preds[:, i, :].T

    plt.subplot(1, 2, 1)
    image = images[i].permute(1, 2, 0).numpy()
    plt.imshow(image)
    plt.axis("off")
    plt.title(texts[i])

    plt.subplot(1, 2, 2)
    plt.yticks(range(pred_i.size(0)), y_ticks)
    plt.imshow(pred_i)
    plt.title(texts_pred[i])

    plt.show()
# Save
# with open('crnn_10_epochs.pth', 'wb') as fp:
#     torch.save(crnn.state_dict(), fp)
# Load
with open('crnn_10_epochs.pth', 'rb') as fp:
    state_dict = torch.load(fp, map_location="cpu")
crnn.load_state_dict(state_dict)
crnn.to(device)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Модель для нахождения номерных знаков на фотографиях
# A model which find bboxes for license plates
model = create_model(device)

# Load
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
    Resize()])

test_plates_dataset = CarPlatesFragmentsDataset('data', transformations, 'test')
test_dataloader = torch.utils.data.DataLoader(test_plates_dataset, 
                                              batch_size=1, shuffle=False,
                                              num_workers=num_workers, pin_memory=True, 
                                              drop_last=True, collate_fn=collate_fn)
crnn.eval()
submit = {}
for b in tqdm.tqdm(test_dataloader):
    file_name = b['file_name'][0][5:]
    if file_name not in submit:
            submit[file_name] = []
    images = b["image"]
    preds = crnn(images.to(device)).cpu().detach()
    texts_pred = decode(preds, crnn.alphabet)
    submit[file_name].append(texts_pred[0])
submit = [(k, ' '.join(v)) for k,v in submit.items()]
submission = pd.DataFrame(submit, columns=['file_name', 'plates_string'])
random_submission = pd.read_csv('submission.csv')
submission = pd.merge(random_submission, submission, how='left', on='file_name')
submission.drop('plates_string_x', axis=1, inplace=True)
submission.columns = ['file_name', 'plates_string']
submission
submission.to_csv('baseline_crnn.csv', index=False)
