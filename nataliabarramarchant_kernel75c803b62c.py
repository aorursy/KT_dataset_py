import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from PIL import Image
from torchvision import transforms

def random_changes(image):
    augmentation = transforms.Compose([
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
    ])
    return augmentation(image)

def preprocess_image(path, augmentation=False):
    input_image = Image.open(path)
    input_image = input_image.convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    if augmentation:
        input_image = random_changes(input_image)
    return preprocess(input_image)
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):

    def __init__(self, images, augmentation=False):
        self.images = images
        self.augmentation = augmentation

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path, label = self.images[idx]
        input_image = preprocess_image(path, self.augmentation)
        return {'image': input_image.to(device), 'label': torch.tensor(label).to(device)}
def accuracy(scores, y, target):
    scores = scores.max(1)[1]
    correct = (scores == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    
    total_positives = (scores == target).nonzero().flatten().tolist()
    total_negatives = (scores != target).nonzero().flatten().tolist()

    cond_positives = (y == target)
    cond_negatives = (y != target)

    p = cond_positives.sum().item()
    n = len(correct) - p
    
    hit = [1 for element in total_positives if element in cond_positives.nonzero().flatten().tolist()]
    sel = [1 for element in total_negatives if element in cond_negatives.nonzero().flatten().tolist()]

    tp = sum(hit)
    tn = sum(sel)    
    return acc, p, tp, n, tn    
def train_model(model, iterator, optimizer, criterion, target):
    
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    
    for batch in iterator:
        
        optimizer.zero_grad()
        output = model(batch['image'])
        loss = criterion(output, batch['label'])
        acc, _, _, _, _ = accuracy(output, batch['label'], target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    train_loss = epoch_loss / len(iterator)
    train_acc = epoch_acc / len(iterator)
    return train_loss, train_acc

def evaluate(model, iterator, criterion, target):
    
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    
    true_p = 0.0
    true_n = 0.0
    p = 0.0
    n = 0.0
    
    with torch.no_grad():
    
        for batch in iterator:
            output = model(batch['image'])
            loss = criterion(output, batch['label'])
            acc, bp, btp, bn, btn = accuracy(output, batch['label'], target)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            p += bp
            true_p += btp
            n += bn
            true_n += btn

    eval_loss = epoch_loss / len(iterator)
    eval_acc = epoch_acc / len(iterator)
    sensitivity = true_p/p
    specificity = true_n/n
    return eval_loss, eval_acc, sensitivity, specificity
def train_and_validate_model(epochs, model, optimizer, target):
    for epoch in range(epochs):
        print("Epoch: {}".format(epoch+1))    
        train_loss, train_acc = train_model(model, train_set, optimizer, criterion, target)
        valid_loss, valid_acc, valid_sens, valid_spec = evaluate(model, valid_set, criterion, target)
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.3f}%')
        print(f'\tVal. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.3f}%')
        print(f'\tVal. Sens: {valid_sens*100:.3f}% | Val. Spec: {valid_spec*100:.3f}%')

def test_model(model, target):
    test_loss, test_acc, sensitivity, specificity = evaluate(model, test_set, criterion, target)
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.3f}%')
    print(f'Test Sens: {sensitivity*100:.3f}% | Test Spec: {specificity*100:.3f}%')
from torchvision import models
from torch import optim
from torch import nn

criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)

def train_and_save_model(model, namefile, target, epochs):
    model.to(device)
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    train_and_validate_model(epochs, model, opt, target)
    torch.save(model.state_dict(), namefile)
import os
import random

PATH ='/kaggle/input/covid19-radiography-database/COVID-19 Radiography Database/'

images = {
    'covid': [],
    'normal': [],
    'pneumonia': []
}

for filename in os.listdir(PATH + 'COVID-19'):
    images['covid'].append((PATH + 'COVID-19/' + filename, 2))
covid_examples = len(images['covid'])

for filename in os.listdir(PATH + 'NORMAL'):
    images['normal'].append((PATH + 'NORMAL/' + filename, 0))
    
for filename in os.listdir(PATH + 'Viral Pneumonia'):
    images['pneumonia'].append((PATH + 'Viral Pneumonia/' + filename, 1))
train = []
valid = []
test = []

example_paths = {}

test_index = int(0.9 * covid_examples)
valid_index = int(0.7 * covid_examples)

for k in images.keys():
    group = images[k]
    random.shuffle(group)
    train += group[:valid_index]
    valid += group[valid_index:test_index]
    test += group[test_index:covid_examples]
    if k != 'covid':
        example_paths[k] = group[covid_examples:]
import json

with open('test_examples.json', 'w') as exfile:
    json.dump(example_paths, exfile)
print(len(train))
print(len(valid))
print(len(test))
train_set = DataLoader(CustomDataset(train, True), batch_size=8, shuffle=True, num_workers=0)
valid_set = DataLoader(CustomDataset(valid), batch_size=8, shuffle=True, num_workers=0)
test_set = DataLoader(CustomDataset(test), batch_size=8, shuffle=True, num_workers=0)
densenet201_diag = models.densenet201(pretrained=True)
densenet201_diag.classifier = nn.Linear(densenet201_diag.classifier.in_features,3)
train_and_save_model(densenet201_diag, 'densenet201_diag.pth', 2, 20)
mobilenet_v2_diag = models.mobilenet_v2(pretrained=True)
mobilenet_v2_diag.classifier[1] = nn.Linear(mobilenet_v2_diag.last_channel,3)
train_and_save_model(mobilenet_v2_diag, 'mobilenet_v2_diag.pth', 2, 20)
resnet18_diag = models.resnet18(pretrained=True)
resnet18_diag.fc = nn.Linear(resnet18_diag.fc.in_features,3)
train_and_save_model(resnet18_diag, 'resnet18_diag.pth', 2, 20)
print("Densenet")
test_model(densenet201_diag, 2)
print("Mobilenet")
test_model(mobilenet_v2_diag, 2)
print("Resnet")
test_model(resnet18_diag, 2)
import os
import random

PATH ='/kaggle/input/chexray/CheXray/'
JSON_PATH = 'test_examples.json'

images = {
    'early': [],
    'mid': [],
    'late': []
}

for filename in os.listdir(PATH + 'early'):
    images['early'].append((PATH + 'early/' + filename, 0))

for filename in os.listdir(PATH + 'mid'):
    images['mid'].append((PATH + 'mid/' + filename, 1))
    
for filename in os.listdir(PATH + 'late'):
    images['late'].append((PATH + 'late/' + filename, 2))
    
min_size = 999
for ex in images.values():
    size = len(ex)
    if size < min_size:
        min_size = size
import json

with open(JSON_PATH) as exfile:
    example_paths = json.load(exfile)
train = []
valid = []
test = []

test_index = int(0.8 * min_size)
valid_index = int(0.6 * min_size)
examples_index = int(0.9 * min_size)

for k in images.keys():
    group = images[k]
    random.shuffle(group)
    train += group[:valid_index]
    valid += group[valid_index:test_index]
    test += group[test_index:examples_index]
    example_paths[k] = group[examples_index:]
with open('test_examples.json', 'w') as exfile:
    json.dump(example_paths, exfile)
train_set = DataLoader(CustomDataset(train, True), batch_size=8, shuffle=True, num_workers=0)
valid_set = DataLoader(CustomDataset(valid), batch_size=8, shuffle=True, num_workers=0)
test_set = DataLoader(CustomDataset(test), batch_size=8, shuffle=True, num_workers=0)
print(len(train))
print(len(valid))
print(len(test))
densenet201_phase = models.densenet201(pretrained=True)
densenet201_phase.classifier = nn.Linear(densenet201_phase.classifier.in_features,3)
train_and_save_model(densenet201_phase, 'densenet201_phase.pth', 0, 40)
mobilenet_v2_phase = models.mobilenet_v2(pretrained=True)
mobilenet_v2_phase.classifier[1] = nn.Linear(mobilenet_v2_phase.last_channel,3)
train_and_save_model(mobilenet_v2_phase, 'mobilenet_v2_phase.pth', 0, 40)
resnet18_phase = models.resnet18(pretrained=True)
resnet18_phase.fc = nn.Linear(resnet18_phase.fc.in_features,3)
train_and_save_model(resnet18_phase, 'resnet18_phase.pth', 0, 40)
from torchvision import models

MODELS_PATH = '/kaggle/input/results/'

diag201_phase = models.densenet201()
diag201_phase.classifier = nn.Linear(diag201_phase.classifier.in_features,3)
diag201_phase.load_state_dict(torch.load(MODELS_PATH + 'densenet201_diag.pth'))
train_and_save_model(diag201_phase, 'diag201_phase.pth', 0, 40)
diagmobile_phase = models.mobilenet_v2()
diagmobile_phase.classifier[1] = nn.Linear(diagmobile_phase.last_channel,3)
diagmobile_phase.load_state_dict(torch.load(MODELS_PATH + 'mobilenet_v2_diag.pth'))
train_and_save_model(diagmobile_phase, 'diagmobile_phase.pth', 0, 40)
diag18_phase = models.resnet18()
diag18_phase.fc = nn.Linear(diag18_phase.fc.in_features,3)
diag18_phase.load_state_dict(torch.load(MODELS_PATH + 'resnet18_diag.pth'))
train_and_save_model(diag18_phase, 'diag18_phase.pth', 0, 40)
print("Densenet")
test_model(densenet201_phase, 0)
print("Mobilenet")
test_model(mobilenet_v2_phase, 0)
print("Resnet")
test_model(resnet18_phase, 0)
print("Densenet")
test_model(diag201_phase, 0)
print("Mobilenet")
test_model(diagmobile_phase, 0)
print("Resnet")
test_model(diag18_phase, 0)
from torchvision import models
from torch import nn

MODELS_PATH = '/kaggle/input/results/'

densenet201_diag_comp = models.densenet201()
densenet201_diag_comp.classifier = nn.Linear(densenet201_diag_comp.classifier.in_features,3)
densenet201_diag_comp.load_state_dict(torch.load(MODELS_PATH + 'densenet201_diag.pth'))
densenet201_diag_comp.to(device)
densenet201_diag_comp.eval()

densenet201_phase_comp = models.densenet201()
densenet201_phase_comp.classifier = nn.Linear(densenet201_phase_comp.classifier.in_features,3)
densenet201_phase_comp.load_state_dict(torch.load(MODELS_PATH + 'densenet201_phase.pth'))
densenet201_phase_comp.to(device)
densenet201_phase_comp.eval()
def classify_xray(path):
    image = preprocess_image(path)
    image = image.to(device)
    with torch.no_grad():
        out_diag = resnet18_diag(image.unsqueeze(0))
        diagnostic = out_diag.max(1)[1].item()
        phase = 99

        if diagnostic == 2: # covid-19
            out_phase = diag18_phase(image.unsqueeze(0))
            phase = out_phase.max(1)[1].item()

        return diagnostic, phase
import random
import json

JSON_PATH = 'test_examples.json'

with open(JSON_PATH) as exfile:
    example_paths = json.load(exfile)

min_size = 999
for ex in example_paths.values():
    size = len(ex)
    if size < min_size:
        min_size = size
print(min_size)
test = []

test += [(example[0], 2, 0) for example in random.sample(example_paths['early'], min_size)]
test += [(example[0], 2, 1) for example in random.sample(example_paths['mid'], min_size)]
test += [(example[0], 2, 2) for example in random.sample(example_paths['late'], min_size)]
test += [(example[0], 0, 99) for example in random.sample(example_paths['normal'], min_size*3)]
test += [(example[0], 1, 99) for example in random.sample(example_paths['pneumonia'], min_size*3)]
pred_diag = []
pred_phases = []
real_diag = []
real_phases = []

for element in test:
    diag, phase = classify_xray(element[0])
    pred_diag.append(diag)
    pred_phases.append(phase)
    real_diag.append(element[1])
    real_phases.append(element[2])
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_cm(pred, real, labels, title):
    ax= plt.subplot()
    cm = confusion_matrix(real, pred)
    sns.heatmap(cm, annot=True, ax = ax)

    ax.set_xlabel('Predicción')
    ax.set_ylabel('Real')
    ax.set_title(title)
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels) 
plot_cm(pred_diag, real_diag, ['Sano', 'Neumonía', 'Covid-19'], 'Clasificador Diagnóstico')
plot_cm(pred_phases, real_phases, ['0-3 días', '4-7 días', '>7 días', 'No Covid-19'], 'Clasificador Fase Covid-19')
pred_phases_filter = [num for num in pred_phases if num < 99]
real_phases_filter = [num for num in real_phases if num < 99]

plot_cm(pred_phases_filter, real_phases_filter, ['0-3 días', '4-7 días', '>7 días'], 'Clasificador Fase Covid-19')
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

print("Reporte Clasificador Diagnóstico\n")
print(classification_report(real_diag, pred_diag, target_names=['Covid-19', 'Sano', 'Neumonía']))
print("Reporte Clasificador Fase Covid-19\n")
print(classification_report(real_phases_filter, pred_phases_filter, target_names=['0-3 días', '4-7 días', '>7 días']))
print("Reporte Clasificador Compuesto\n")
print(classification_report(real_phases, pred_phases, target_names=['0-3 días', '4-7 días', '>7 días', 'No Covid']))