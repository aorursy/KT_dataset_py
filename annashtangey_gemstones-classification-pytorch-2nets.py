# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pickle
from skimage import io
from tqdm import tqdm, tqdm_notebook
from PIL import Image
import PIL
from pathlib import Path
import torch
from torchvision import transforms
from multiprocessing.pool import ThreadPool
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
from matplotlib import colors, pyplot as plt
torch.cuda.is_available()

import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
DATA_MODES = ['train', 'val', 'test']
RESCALE_SIZE = 299
DEVICE = torch.device("cuda")
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
def pretramsform(x):
    transform = transforms.Compose([
                                  transforms.ColorJitter(brightness=np.random.uniform(0,0.2),contrast=np.random.uniform(0,0.2),hue=0,saturation=0),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.RandomRotation(np.random.uniform(0,360))
                                  ])
    return transform(x)
class GemstonesDataset(Dataset):
  
    
    def __init__(self, files, mode):
        super().__init__()
        
        self.files = sorted(files)
        
        self.mode = mode

        if self.mode not in DATA_MODES:
            print(f"{self.mode} is not correct; correct modes: {DATA_MODES}")
            raise NameError

        self.len_ = len(self.files)
     
        self.label_encoder = LabelEncoder()

        if self.mode != 'test':
            self.labels = [path.parent.name for path in self.files]
            self.label_encoder.fit(self.labels)

            with open('label_encoder.pkl', 'wb') as le_dump_file:
                  pickle.dump(self.label_encoder, le_dump_file)
                      
    def __len__(self):
        return self.len_
      
    def load_sample(self, file):
        image = Image.open(file)
        image.load()
        return image.convert('RGB')
  
    def __getitem__(self, index):
        
        
        if self.mode == 'test':
            transform = transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                 
                       
            x = self.load_sample(self.files[index])
            x = self._prepare_sample(x)
            x = np.array(x / 255, dtype='float32')
            x = transform(x)
            return x
        else:
            transform = transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.ToPILImage(),
                                          transforms.RandomChoice([transforms.ColorJitter(brightness=np.random.uniform(0,0.2),contrast=np.random.uniform(0,0.2),hue=0,saturation=0),
                                                                   transforms.ColorJitter(brightness=0,contrast=0,hue=np.random.uniform(0,0.1),saturation=np.random.uniform(0,0.1)),
                                                                   transforms.RandomHorizontalFlip(),
                                                                   transforms.RandomRotation(np.random.uniform(0,5))]),
                                          
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                                          
            x = self.load_sample(self.files[index])
            x = self._prepare_sample(x)
            x = np.array(x / 255, dtype='float32')
            x = transform(x)
       
            label = self.labels[index]
            label_id = self.label_encoder.transform([label])
            y = label_id.item()
            return x, y
        
    def _prepare_sample(self, image):
        image = image.resize((RESCALE_SIZE, RESCALE_SIZE))
        return np.array(image)
def imshow(inp, title=None, plt_ax=plt, default=False):
    
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt_ax.imshow(inp)
    if title is not None:
        plt_ax.set_title(title)
    plt_ax.grid(False)
TRAIN_DIR = Path('/kaggle/input/gemstones-images/train')
TEST_DIR = Path('/kaggle/input/gemstones-images/test')

train_val_files = sorted(list(TRAIN_DIR.rglob('*.jpg')))
test_files = sorted(list(TEST_DIR.rglob('*.jpg')))
from sklearn.model_selection import train_test_split

train_val_labels = [path.parent.name for path in train_val_files]
train_files, val_files = train_test_split(train_val_files, test_size=0.25, \
                                          stratify=train_val_labels, shuffle=True)
val_dataset = GemstonesDataset(val_files, mode='val')
test_labels = [path.parent.name for path in test_files]
test_dataset = GemstonesDataset(test_files, mode='test')
fig, ax = plt.subplots(nrows=3, ncols=3,figsize=(8, 8), \
                        sharey=True, sharex=True)
for fig_x in ax.flatten():
    random_characters = int(np.random.uniform(0,len(val_dataset)))
    im_val, label = val_dataset[random_characters]
    img_label = " ".join(map(lambda x: x.capitalize(),\
                val_dataset.label_encoder.inverse_transform([label])[0].split('_')))
    imshow(im_val.data.cpu(), \
          title=img_label,plt_ax=fig_x)
from torchvision.models.inception import inception_v3
inception_model=inception_v3(pretrained=True, aux_logits=False)
from torchvision.models.resnet import resnet50
resnet_model=resnet50(pretrained=True)

def fit_epoch(model, train_loader, criterion, optimizer):
    
    running_loss = 0.0
    running_corrects = 0
    processed_data = 0
    
  
    for inputs, labels in train_loader:
        
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        
        optimizer.zero_grad()

        
        outputs = model(inputs)
        
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        preds = torch.argmax(outputs, 1)
        
        running_loss += loss.item() * inputs.size(0)
        
        running_corrects += torch.sum(preds == labels.data)
        
        processed_data += inputs.size(0)
          
    train_loss = running_loss / processed_data
    train_acc = running_corrects.cpu().numpy() / processed_data
    
    return train_loss, train_acc
def eval_epoch(model, val_loader, criterion):
    
    model.eval()
    
    running_loss = 0.0
    running_corrects = 0
    processed_size = 0

    for inputs, labels in val_loader:
        
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.set_grad_enabled(False):
            
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            
            preds = torch.argmax(outputs, 1)
        
        running_loss += loss.item() * inputs.size(0)
        
        running_corrects += torch.sum(preds == labels.data)
        
        processed_size += inputs.size(0)
    
    val_loss = running_loss / processed_size
    val_acc = running_corrects.double() / processed_size
    
    return val_loss, val_acc
def train(train_files, val_files, model, epochs, batch_size, mode):
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    history = []
    
    
    log_template = "\nEpoch {ep:03d} train_loss: {t_loss:0.4f} \
    val_loss {v_loss:0.4f} train_acc {t_acc:0.4f} val_acc {v_acc:0.4f}"

    with tqdm(desc="epoch", total=epochs) as pbar_outer:
        params_to_update = []
      
        for param in model.parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
      
        opt = torch.optim.Adam(params_to_update, lr=0.0001)
       
        scheduler=torch.optim.lr_scheduler.StepLR(opt,7,gamma=0.3)
      
        criterion = nn.CrossEntropyLoss()

      
        for epoch in range(epochs):
        
            train_loss, train_acc = fit_epoch(model, train_loader, criterion, opt)
        
            print("loss", train_loss)
        
            scheduler.step(train_loss)
        
            val_loss, val_acc = eval_epoch(model, val_loader, criterion)
        
            history.append((train_loss, train_acc, val_loss, val_acc))

            pbar_outer.update(1)
            tqdm.write(log_template.format(ep=epoch+1, t_loss=train_loss,\
                                       v_loss=val_loss, t_acc=train_acc, v_acc=val_acc))
         
    return history
def predict(model, test_loader):
    with torch.no_grad():
        logits = []
    
        for inputs in test_loader:
            
            inputs = inputs.to(DEVICE)
            
            model.eval()
            
            outputs = model(inputs).cpu()
            
            logits.append(outputs)
           
    probs = nn.functional.softmax(torch.cat(logits), dim=-1).numpy()
    return probs
n_classes = len(np.unique(train_val_labels))
print("we will classify :{}".format(n_classes))
ct=0
for name, child in inception_model.named_children():
    ct += 1
    if ct < 7:
        for name2, par in child.named_parameters():
            par.requires_grad = False



inception_model.fc = nn.Sequential(
    nn.Dropout2d(),
    nn.Linear(2048, out_features=n_classes))   
inception_model.to(DEVICE)
ct=0
for name, child in resnet_model.named_children():
    ct += 1
    if ct < 7:
        for name2, par in child.named_parameters():
            par.requires_grad = False




for param in resnet_model.fc.parameters() or resnet_model.layer1.parameters() or resnet_model.layer2.parameters() or resnet_model.layer3.parameters():
    param.requires_grad = True

resnet_model.fc = nn.Sequential(
    nn.Dropout2d(),
    nn.Linear(resnet_model.fc.in_features, out_features=n_classes)
    )
resnet_model.to(DEVICE)
val_dataset = GemstonesDataset(val_files, mode='val')
    
train_dataset = GemstonesDataset(train_files, mode='train')
history_inception = train(train_dataset, val_dataset, model=inception_model, epochs=25, batch_size=128, mode='i')
loss, acc, val_loss, val_acc = zip(*history_inception)
plt.figure(figsize=(15, 9))
plt.title('Inception')
plt.plot(loss, label="train_loss")
plt.plot(val_loss, label="val_loss")
plt.legend(loc='best')
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()
history_resnet = train(train_dataset, val_dataset, model=resnet_model, epochs=25, batch_size=128,  mode='r')
loss, acc, val_loss, val_acc = zip(*history_resnet)
plt.figure(figsize=(15, 9))
plt.title('Resnet')
plt.plot(loss, label="train_loss")
plt.plot(val_loss, label="val_loss")
plt.legend(loc='best')
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()
def predict_one_sample(model, inputs, device=DEVICE):
    
    with torch.no_grad():
        inputs = inputs.to(device)
        model.eval()
        logit = model(inputs).cpu()
        probs = torch.nn.functional.softmax(logit, dim=-1).numpy()
    return probs
random_characters = int(np.random.uniform(0,360))
ex_img = test_dataset[random_characters]
probs_im_inception = predict_one_sample(inception_model, ex_img.unsqueeze(0))
probs_im_resnet = predict_one_sample(resnet_model, ex_img.unsqueeze(0))
CONF_THRESH=0.5
label_encoder = pickle.load(open("label_encoder.pkl", 'rb'))

random_characters = int(np.random.uniform(0,360))
ex_img = test_dataset[random_characters]
probs_im_inception = predict_one_sample(inception_model, ex_img.unsqueeze(0))
probs_im_resnet = predict_one_sample(resnet_model, ex_img.unsqueeze(0))



predicted_proba_i = np.max(probs_im_inception)*100
predicted_proba_li = np.argmax((probs_im_inception))
predicted_proba_r = np.max(probs_im_resnet)*100
predicted_proba_lr = np.argmax((probs_im_resnet))
pthi=label_encoder.classes_[predicted_proba_li]
pthr=label_encoder.classes_[predicted_proba_lr]

print('Inseption')
print(pthi)
print(predicted_proba_i)
print()
print('Resnet')
print(pthr)
print(predicted_proba_r)


imshow(ex_img.cpu())
idxs = list(map(int, np.random.uniform(0,len(val_dataset), 20)))
imgs = [val_dataset[id][0].unsqueeze(0) for id in idxs]


probs_ims_inception = predict(inception_model, imgs)
probs_ims_resnet = predict(resnet_model, imgs)
label_encoder = pickle.load(open("label_encoder.pkl", 'rb'))
y_pred_resnet = np.argmax(probs_ims_resnet,-1)
y_pred_inception = np.argmax(probs_ims_inception,-1)
print ('Resnet:    ', y_pred_resnet)
print('Inception: ',y_pred_inception)

actual_labels = [val_dataset[id][1] for id in idxs]
y_pred=[]
for i in range(len(actual_labels)):
    if y_pred_resnet[i] == y_pred_inception[i]:
        y_pred.append(y_pred_inception[i])
    elif probs_ims_resnet[i,y_pred_resnet[i]]>probs_ims_inception[i,y_pred_inception[i]]:
        y_pred.append(y_pred_resnet[i])
    else:
        y_pred.append(y_pred_inception[i]) 
print('Ensemble:  ',y_pred)
print('Right answer:       ',actual_labels)
from sklearn.metrics import f1_score
sc_resnet=f1_score(actual_labels, y_pred_resnet, average='macro')
print ('Resnet: F1_score=', sc_resnet)
sc_inception=f1_score(actual_labels, y_pred_inception, average='macro')
print ('Inception: F1_score=', sc_inception)
sc=f1_score(actual_labels, y_pred, average='macro')
print ('Ensemble: F1_score=',sc)
import matplotlib.patches as patches
from matplotlib.font_manager import FontProperties
print("Resnet predictions")
fig, ax = plt.subplots(nrows=3, ncols=3,figsize=(12, 12), \
                        sharey=True, sharex=True)

for fig_x in ax.flatten():
    random_characters = int(np.random.uniform(0,len(val_dataset)))
    im_val, label = val_dataset[random_characters]
    img_label = " ".join(map(lambda x: x.capitalize(),\
                val_dataset.label_encoder.inverse_transform([label])[0].split('_')))
    
    

    imshow(im_val.data.cpu(), \
          title=img_label,plt_ax=fig_x)
    
    actual_text = "Actual : {}".format(img_label)
            
    fig_x.add_patch(patches.Rectangle((0, 53),86,35,color='white'))
    font0 = FontProperties()
    font = font0.copy()
    font.set_family("fantasy")
    prob_pred = predict_one_sample(resnet_model, im_val.unsqueeze(0))
    predicted_proba = np.max(prob_pred)*100
    y_pred = np.argmax(prob_pred)
    
    predicted_label = label_encoder.classes_[y_pred]
    predicted_label = predicted_label[:len(predicted_label)//2] + '\n' + predicted_label[len(predicted_label)//2:]
    predicted_text = "{} : {:.0f}%".format(predicted_label,predicted_proba)
            
    fig_x.text(1, 59, predicted_text , horizontalalignment='left', fontproperties=font,
                    verticalalignment='top',fontsize=8, color='black',fontweight='bold')
import matplotlib.patches as patches
from matplotlib.font_manager import FontProperties
print("Inception predictions")
fig, ax = plt.subplots(nrows=3, ncols=3,figsize=(12, 12), \
                        sharey=True, sharex=True)

for fig_x in ax.flatten():
    random_characters = int(np.random.uniform(0,len(val_dataset)))
    im_val, label = val_dataset[random_characters]
    img_label = " ".join(map(lambda x: x.capitalize(),\
                val_dataset.label_encoder.inverse_transform([label])[0].split('_')))
    
    

    imshow(im_val.data.cpu(), \
          title=img_label,plt_ax=fig_x)
    
    actual_text = "Actual : {}".format(img_label)
            
    fig_x.add_patch(patches.Rectangle((0, 53),86,35,color='white'))
    font0 = FontProperties()
    font = font0.copy()
    font.set_family("fantasy")
    prob_pred = predict_one_sample(inception_model, im_val.unsqueeze(0))
    predicted_proba = np.max(prob_pred)*100
    y_pred = np.argmax(prob_pred)
    
    predicted_label = label_encoder.classes_[y_pred]
    predicted_label = predicted_label[:len(predicted_label)//2] + '\n' + predicted_label[len(predicted_label)//2:]
    predicted_text = "{} : {:.0f}%".format(predicted_label,predicted_proba)
            
    fig_x.text(1, 59, predicted_text , horizontalalignment='left', fontproperties=font,
                    verticalalignment='top',fontsize=8, color='black',fontweight='bold')
import matplotlib.patches as patches
from matplotlib.font_manager import FontProperties
print("Ensemble predictions")
fig, ax = plt.subplots(nrows=3, ncols=3,figsize=(12, 12), \
                        sharey=True, sharex=True)
for fig_x in ax.flatten():
    random_characters = int(np.random.uniform(0,len(val_dataset)))
    im_val, label = val_dataset[random_characters]
    img_label = " ".join(map(lambda x: x.capitalize(),\
                val_dataset.label_encoder.inverse_transform([label])[0].split('_')))
    
    

    imshow(im_val.data.cpu(), \
          title=img_label,plt_ax=fig_x)
    
    actual_text = "Actual : {}".format(img_label)
            
    fig_x.add_patch(patches.Rectangle((0, 53),86,35,color='white'))
    font0 = FontProperties()
    font = font0.copy()
    font.set_family("fantasy")
    prob_pred_inception = predict_one_sample(inception_model, im_val.unsqueeze(0))
    prob_pred_resnet = predict_one_sample(resnet_model, im_val.unsqueeze(0))

    y_pred_resnet = np.argmax(prob_pred_resnet,-1)
    y_pred_inception= np.argmax(prob_pred_inception,-1)
    
    y_pred=[]
    for i in range(1):

        if y_pred_resnet[i] == y_pred_inception[i]:
            y_pred.append(y_pred_resnet[i])
            predicted_proba = np.max(prob_pred_resnet)*100
        elif prob_pred_resnet[i,y_pred_resnet[i]]>prob_pred_inception[i,y_pred_inception[i]]:
            y_pred.append(y_pred_resnet[i])
            predicted_proba = np.max(prob_pred_resnet)*100
        else:
            y_pred.append(y_pred_inception[i]) 
            predicted_proba = np.max(prob_pred_inception)*100
    


    predicted_proba = np.max(prob_pred)*100
    
    
    predicted_label = label_encoder.classes_[y_pred]
   # predicted_label = predicted_label[:len(predicted_label)//2] + '\n' + predicted_label[len(predicted_label)//2:]
    predicted_text = "{} : {:.0f}%".format(predicted_label,predicted_proba)
            
    fig_x.text(1, 59, predicted_text , horizontalalignment='left', fontproperties=font,
                    verticalalignment='top',fontsize=8, color='black',fontweight='bold')
import itertools
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
   
    cm = cm.T
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    
    plt.figure(figsize=(16,11))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.tight_layout()
    
def show_confusion_matrix_fucn(model):
    
    y_test_all = torch.Tensor().long()
    predictions_all = torch.Tensor().long()

    
    with torch.no_grad():
        for inputs, labels in dataloaders['val']:
            predictions = model(inputs.to(DEVICE))
            y_test = labels
            _, predictions = torch.max(predictions.cpu(), 1)

            
            y_test_all = torch.cat((y_test_all, y_test), 0)
            predictions_all = torch.cat((predictions_all, predictions), 0)

    feature_names = sorted(set(dataloaders['val'].dataset.labels))

    y_test_all = y_test_all.numpy()
    predictions_all = predictions_all.numpy()

    
    cm = confusion_matrix(y_test_all, predictions_all, np.arange(n_classes))
    
    plot_confusion_matrix(cm, feature_names, normalize=True)
    
    return y_test_all, predictions_all
  
def accurancy_for_each_class(y_test_all, predictions_all):
    class_correct = [0 for i in range(n_classes)]
    class_total = [0 for i in range(n_classes)]
    feature_names = sorted(set(dataloaders['val'].dataset.labels))

    c = (predictions_all == y_test_all).squeeze()
    for i in range(len(predictions_all)):
        label = predictions_all[i]            
        class_correct[label] += c[i].item()
        class_total[label] += 1

    print(class_total)
    print(len(class_total))

    for i in range(n_classes):
        print('Accuracy of %5s : %2d %%' % (
            (feature_names[i], (100 * class_correct[i] / class_total[i]) if class_total[i] != 0 else -1)))
BATCH_SIZE = 128

dataloaders = {'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True),
               'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)}
dataset_sizes = {'train': len(train_dataset), 'val':len(val_dataset) }
print ('Inception:')
y_test_all, predictions_all = show_confusion_matrix_fucn(inception_model)
accurancy_for_each_class(y_test_all, predictions_all)
print ('Resnet:')
y_test_all, predictions_all = show_confusion_matrix_fucn(resnet_model)
accurancy_for_each_class(y_test_all, predictions_all)