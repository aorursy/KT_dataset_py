import torch.utils.data
import csv
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torch.utils.data import Dataset
import cv2
import os
import json
from torchvision import models
from torch.nn import Module, Sequential, Conv2d, AvgPool2d, GRU, Linear

import math
from torchvision import transforms
import torch
import torchvision
import tqdm
import numpy as np
from torch.nn.functional import ctc_loss, log_softmax
class DetectionDataset(torch.utils.data.Dataset):
    def __init__(self, root, transformations=None, split="train", splitSize=0.9):
        super(DetectionDataset, self).__init__()
        self.transformations = transformations
        self.root = root

        with open(os.path.join(root, "train.json"), 'r') as f:
            data = json.load(f)

        data = [x for x in data if x['file'] != "train/25632.bmp"]

        num_lines = len(data)

        sz = round(splitSize * num_lines)
        if split == "train":
            self.data_dict = data[:sz]
        elif split == "val":
            self.data_dict = data[sz:]

    def __getitem__(self, idx):
        data = self.data_dict[idx]
        img_path = os.path.join(self.root, data['file'])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        original_boxes = data['nums']
        num_objs = len(original_boxes)
        boxes = []
        for bbox in original_boxes:
            bbox = bbox['box']
            xmin = min(bbox[0][0], bbox[3][0])
            xmax = max(bbox[1][0], bbox[2][0])
            ymin = min(bbox[0][1], bbox[1][1])
            ymax = max(bbox[2][1], bbox[3][1])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transformations is not None:
            image = self.transformations(image)

        return image, target

    def __len__(self):
        return len(self.data_dict)
class DetectionDatasetTest(torch.utils.data.Dataset):
    def __init__(self, root, transformations=None):
        super(DetectionDatasetTest, self).__init__()
        self.transformations = transformations
        self.root = root

        self.images = []
        line_count = 0
        with open(os.path.join(root, "submission.csv"), 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                else:
                    self.images.append(row[0])

    def __getitem__(self, idx):
        im_p = self.images[idx]
        img_path = os.path.join(self.root, im_p)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        target = {"file": im_p}
        if self.transformations is not None:
            image = self.transformations(image)

        return image, target

    def __len__(self):
        return len(self.images)
def collate_fn(batch):
    return tuple(zip(*batch))
train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
batch_size = 4

train_dataset = DetectionDataset('data', train_transforms, split="train")
val_dataset = DetectionDataset('data', train_transforms, split="val")
data_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
    pin_memory=True,
    collate_fn=collate_fn, drop_last=True)
data_loader_val = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
    pin_memory=True,
    collate_fn=collate_fn, drop_last=False)
print("Creating model...")
device = torch.device("cuda: 0") if torch.cuda.is_available() else torch.device("cpu")

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, progress=True,
                                                             num_classes=91, pretrained_backbone=True)
num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
for param in model.parameters():
    param.requires_grad = False;
for param in model.rpn.parameters():
    param.requires_grad = True;
for param in model.roi_heads.parameters():
    param.requires_grad = True;
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.01,momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
def train_one_epoch(model, optimizer, data_loader, device):
    model.train()

    total_losses = []
    for images, targets in tqdm.tqdm(data_loader, position=0, leave=True):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        if math.isfinite(loss_value):
            total_losses.append(loss_value)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()


    print(f"Train: {np.mean(np.array(total_losses))}")
def evaluate(model, optimizer, data_loader, device):
    total_losses = []
    for images, targets in tqdm.tqdm(data_loader, position=0, leave=True):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        optimizer.zero_grad()
        if math.isfinite(loss_value):
            total_losses.append(loss_value)

    print(f"Validation: {np.mean(np.array(total_losses))}")
model.to(device)
for epoch in range(3):
    train_one_epoch(model, optimizer, data_loader, device)
    evaluate(model, optimizer, data_loader_val, device=device)

torch.save(model.state_dict(), f'first_model.pth')
test_dataset = DetectionDatasetTest('data', train_transforms)
data_loader_test = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
    pin_memory=True,
    collate_fn=collate_fn, drop_last=False)

model.eval()
thresh = 0.8

result = {}

for images, targets in tqdm.tqdm(data_loader_test, position=0, leave=True):
    images = list(image.to(device) for image in images)
    preds = model(images)

    for j in range(len(preds)):
        file = targets[j]['file']
        boxes = []

        prediction = preds[j];
        for i in range(len(prediction['boxes'])):
            x_min, y_min, x_max, y_max = map(int, prediction['boxes'][i].tolist())
            label = int(prediction['labels'][i].cpu())
            score = float(prediction['scores'][i].cpu())
            if score > thresh:
                boxes.append([x_min, y_min, x_max, y_max])
        result[file] = boxes

js = json.dumps(result)
with open(f"first_answer.json", "w") as f:
    f.write(js)
def collate_fn_difsize(batch):
    images, seqs, seq_lens, texts = [], [], [], []
    for sample in batch:
        images.append(torch.from_numpy(sample["image"]).permute(2, 0, 1).float())
        seqs.extend(sample["seq"])
        seq_lens.append(sample["seq_len"])
        texts.append(sample["text"])
    images = torch.stack(images)
    seqs = torch.Tensor(seqs).int()
    seq_lens = torch.Tensor(seq_lens).int()
    batch = {"image": images, "seq": seqs, "seq_len": seq_lens, "text": texts}
    return batch

class Resize(object):
    def __init__(self, size=(320, 64)):
        self.size = size

    def __call__(self, item):
        item['image'] = cv2.resize(item['image'], self.size, interpolation=cv2.INTER_AREA)
        return item

class RecognitionDataset(Dataset):
    def __init__(self, root, alphabet="0123456789ABEKMHOPCTYX", transforms=None, split="train", splitSize=0.9):
        super(RecognitionDataset, self).__init__()

        self.alphabet = alphabet
        self.root = root
        with open(os.path.join(root, "train.json"), 'r') as f:
            data = json.load(f)

        data = [x for x in data if x['file'] != "train/25632.bmp"]

        self.data_dict = []
        for item in data:
            for number in item['nums']:
                r = {
                    "box": number["box"],
                    "text": number["text"],
                    "file": item["file"]
                }
                self.data_dict.append(r)

        sz = round(splitSize * len(self.data_dict))
        if split == "train":
            self.data_dict = self.data_dict[:sz]
        elif split == "val":
            self.data_dict = self.data_dict[sz:]

        self.transforms = transforms

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        val = self.data_dict[item]

        box = np.array(val["box"])
        xmin = box[:,0].min()
        xmax = box[:,0].max()
        ymin = box[:,1].min()
        ymax = box[:,1].max()

        xmin = max(xmin, 0)
        ymin = max(ymin, 0)

        img_path = os.path.join(self.root, val["file"])
        # print(img_path, xmin, xmax, ymin, ymax)
        image = cv2.imread(img_path).astype(np.float32) / 255.
        image = image[ymin:ymax+1, xmin:xmax+1]

        text = val["text"]
        seq = self.text_to_seq(text)
        seq_len = len(seq)
        output = dict(image=image, seq=seq, seq_len=seq_len, text=text)
        if self.transforms is not None:
            output = self.transforms(output)
        return output

    def text_to_seq(self, text):
        seq = [self.alphabet.find(c) + 1 for c in text]
        return seq

class FeatureExtractor(Module):

    def __init__(self, input_size=(64, 320), output_len=20):
        super(FeatureExtractor, self).__init__()

        h, w = input_size
        resnet = getattr(models, 'resnet18')(pretrained=True)
        self.cnn = Sequential(*list(resnet.children())[:-2])

        self.pool = AvgPool2d(kernel_size=(h // 32, 1))
        self.proj = Conv2d(w // 32, output_len, kernel_size=1)

        self.num_output_features = self.cnn[-1][-1].bn2.num_features

    def apply_projection(self, x):
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        return x

    def forward(self, x):
        features = self.cnn(x)
        features = self.pool(features)
        features = self.apply_projection(features)

        return features

class SequencePredictor(Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.1, bidirectional=False):
        super(SequencePredictor, self).__init__()

        self.num_classes = num_classes
        self.rnn = GRU(input_size=input_size,
                       hidden_size=hidden_size,
                       num_layers=num_layers,
                       dropout=dropout,
                       bidirectional=bidirectional)

        fc_in = hidden_size if not bidirectional else 2 * hidden_size
        self.fc = Linear(in_features=fc_in,
                         out_features=num_classes)

    def _init_hidden_(self, batch_size):
        num_directions = 2 if self.rnn.bidirectional else 1
        return torch.zeros(self.rnn.num_layers * num_directions, batch_size, self.rnn.hidden_size)

    def _prepare_features_(self, x):
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


class CRNN(Module):

    def __init__(self, alphabet="0123456789ABEKMHOPCTYX",
                 cnn_input_size=(64, 320), cnn_output_len=20,
                 rnn_hidden_size=128, rnn_num_layers=2, rnn_dropout=0.1, rnn_bidirectional=False):
        super(CRNN, self).__init__()
        self.alphabet = alphabet
        self.features_extractor = FeatureExtractor(input_size=cnn_input_size, output_len=cnn_output_len)
        print(self.features_extractor.num_output_features)
        self.sequence_predictor = SequencePredictor(input_size=self.features_extractor.num_output_features,
                                                    hidden_size=rnn_hidden_size, num_layers=rnn_num_layers,
                                                    num_classes=len(alphabet) + 1, dropout=rnn_dropout,
                                                    bidirectional=rnn_bidirectional)

    def forward(self, x):
        features = self.features_extractor(x)
        sequence = self.sequence_predictor(features)
        return sequence

def pred_to_string(pred, abc = "0123456789ABEKMHOPCTYX"):
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

def decode(pred, abc = "0123456789ABEKMHOPCTYX"):
    pred = pred.permute(1, 0, 2).cpu().data.numpy()
    outputs = []
    for i in range(len(pred)):
        outputs.append(pred_to_string(pred[i], abc))
    return outputs
class RecognitionDatasetTest(Dataset):
    def __init__(self, root, alphabet="0123456789ABEKMHOPCTYX", transforms=None):
        super(RecognitionDatasetTest, self).__init__()

        self.alphabet = alphabet
        self.root = root
        with open("first_answer.json", 'r') as f:
            data = json.load(f)

        self.data_dict = []
        for file, boxes in data.items():
            for box in boxes:
                r = {
                    "box": box,
                    "file": file
                }
                self.data_dict.append(r)

        self.transforms = transforms

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        val = self.data_dict[item]

        box = np.array(val["box"])
        xmin = box[0]
        xmax = box[2]
        ymin = box[1]
        ymax = box[3]

        xmin = max(xmin, 0)
        ymin = max(ymin, 0)

        img_path = os.path.join(self.root, val["file"])
        image = cv2.imread(img_path).astype(np.float32) / 255.
        image = image[ymin:ymax+1, xmin:xmax+1]

        seq = []
        seq_len = 0
        output = dict(image=image, seq=seq, seq_len=seq_len, text=val["file"])
        if self.transforms is not None:
            output = self.transforms(output)
        return output
crnn = CRNN()
num_epochs = 10
batch_size = 512
num_workers = 4

optimizer = torch.optim.Adam(crnn.parameters(), lr=3e-4, amsgrad=True, weight_decay=1e-4)

train_dataset = RecognitionDataset("data", transforms=Resize())
val_dataset = RecognitionDataset("data", split="val", transforms=Resize())

train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                               pin_memory=True,
                                               drop_last=True, collate_fn=collate_fn_difsize)
val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                             pin_memory=True,
                                             drop_last=False, collate_fn=collate_fn_difsize)
crnn.to(device);

for i, epoch in enumerate(range(num_epochs)):
    epoch_losses = []
    crnn.train()
    for j, b in enumerate(tqdm.tqdm(train_dataloader, total=len(train_dataloader))):
        images = b["image"].to(device)
        seqs_gt = b["seq"]
        seq_lens_gt = b["seq_len"]

        seqs_pred = crnn(images).cpu()
        log_probs = log_softmax(seqs_pred, dim=2)
        seq_lens_pred = torch.Tensor([seqs_pred.size(0)] * seqs_pred.size(1)).int()

        loss = ctc_loss(log_probs=log_probs,  # (T, N, C)
                        targets=seqs_gt,  # N, S or sum(target_lengths)
                        input_lengths=seq_lens_pred,  # N
                        target_lengths=seq_lens_gt)  # N

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())
    print("Train ", np.mean(epoch_losses))
    crnn.eval()
    val_losses = []
    for i, b in enumerate(tqdm.tqdm(val_dataloader, total=len(val_dataloader))):
        images = b["image"].to(device)
        seqs_gt = b["seq"]
        seq_lens_gt = b["seq_len"]

        with torch.no_grad():
            seqs_pred = crnn(images).cpu()
        log_probs = log_softmax(seqs_pred, dim=2)
        seq_lens_pred = torch.Tensor([seqs_pred.size(0)] * seqs_pred.size(1)).int()

        loss = ctc_loss(log_probs=log_probs,  # (T, N, C)
                        targets=seqs_gt,  # N, S or sum(target_lengths)
                        input_lengths=seq_lens_pred,  # N
                        target_lengths=seq_lens_gt)  # N

        val_losses.append(loss.item())

    print("Eval", np.mean(val_losses))

torch.save(crnn.state_dict(), f'second_model.pth')
crnn.eval()
test_dataset = RecognitionDatasetTest("data", transforms=Resize())

test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                              pin_memory=True,
                                              drop_last=False, collate_fn=collate_fn_difsize)

result = {}
for i, b in enumerate(tqdm.tqdm(test_dataloader, total=len(test_dataloader))):
    images = b["image"].to(device)
    preds = crnn(images.to(device)).cpu().detach()
    texts_pred = decode(preds, crnn.alphabet)

    for i in range(len(texts_pred)):
        file = b["text"][i]
        pred = texts_pred[i]

        if file not in result:
            result[file] = []
        result[file].append(pred)
line_count = 0;
submit = {file : ' '.join(text) for file, text in result.items()}

with open(os.path.join("data", "submission.csv"), 'r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            if row[0] not in result:
                submit[row[0]] = ''

with open('my_submit.csv', 'w') as f:
    f.write(f"file_name,plates_string\n")
    for key in submit.keys():
        f.write(f"{key},{submit[key]}\n")
