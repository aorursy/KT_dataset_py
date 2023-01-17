import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib.pyplot as plt
from glob import glob
import cv2 as cv
import random
import torchvision.models as models
from tqdm.notebook import tqdm
import torchvision.transforms as T
from PIL import Image
import PIL
from torch import nn, optim
import torch
from sklearn.preprocessing import LabelEncoder
import numpy as np
import multiprocessing
import seaborn as sns
import sklearn.metrics as metrics
import time
# Get efficientnet for pytorch: https://www.kaggle.com/haqishen/train-efficientnet-b0-w-36-tiles-256-lb0-87
import os
import sys
sys.path = [
    '../input/efficientnet-pytorch/EfficientNet-PyTorch/EfficientNet-PyTorch-master',
] + sys.path
from efficientnet_pytorch import model as enet
torch.manual_seed(14)
np.random.seed(14)
train_image_path = '../input/landmark-recognition-2020/train/'
data_csv = pd.read_csv('../input/landmark-recognition-2020/train.csv')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
BATCH_SIZE = 32
EPOCHS = 3
class LandmarkDataset(Dataset):
    def __init__(self, csv, image_dir, transform, partition):
        # Variables
        self.csv = csv
        self.image_dir = image_dir
        self.transform = transform
        self.partition = partition
        self.picked_data = None
        self.label_encoder = None
        
        # All data
        #self.data = self.load_all_data()
        
        # Only do for training/validation
        if self.partition == 'Training/Validation':
            self.csv.landmark_id = self.map_labels()
    
            # Picked Data
            self.update_pick_data()
            self.len_data = len(self.picked_data)
        else:
            self.data = self.load_all_data()
    
    def __getitem__(self, idx):
        if self.partition == 'Training/Validation':
            
            #image_name = self.picked_data.iloc[idx].id 
            #label = self.picked_data.iloc[idx].landmark_id
            image_name = self.picked_data.id[idx]
            label = self.picked_data.landmark_id[idx]
        elif self.partition == 'Test':
            #image_name = self.csv.iloc[idx].id
            image_name = self.picked_data.id[idx]
            label = -1
        
        image_path = f'{self.image_dir}{image_name[0]}/{image_name[1]}/{image_name[2]}/{image_name}.jpg'
        image = Image.open(image_path)
        image = self.transform(image)
        
        #print(f'Path: {image_path}, Label: {label}')
        return image, label
    
    
    def __len__(self):
        if self.partition == 'Test':
            return(len(self.csv))
        if self.partition == 'Training/Validation':
            return self.len_data
    
    def update_pick_data(self):
        #del self.picked_data
        #torch.cuda.empty_cache()
        
        ids = []
        landmark_ids = []
        for i in range(len(self.csv.landmark_id.unique())):
            samples = self.csv.loc[self.csv['landmark_id'] == i]
            if len(samples) > 10:
                samples = self.csv.loc[self.csv['landmark_id'] == i].sample(10)
            ids.extend(samples['id'].tolist())
            landmark_ids.extend(samples['landmark_id'].tolist())
        
        # Picked Data
        self.picked_data = pd.DataFrame({'id': ids, 'landmark_id': landmark_ids})
        self.len_data = len(self.picked_data)
        #return pd.DataFrame({'id': ids, 'landmark_id': landmark_ids}) 
        
        
    
    def map_labels(self):
        le = LabelEncoder()
        le.fit(self.csv.landmark_id.values)
        self.label_encoder = le
        return le.transform(self.csv.landmark_id)
    
    
    def load_all_data(self):
        image_paths = glob(f'{self.image_dir}/*/*/*/*')
        return image_paths
    
    
    def get_random_examples(self, rows = 2, columns = 5, transform = False):
        img_paths = random.sample(self.data, rows*columns)
        
        f, axarr = plt.subplots(rows, columns, figsize=(30,30))
        image_counter = 0
        for r in range(rows):
            for c in range(columns):
                #img = cv.cvtColor(cv.imread(img_paths[image_counter]), cv.COLOR_BGR2RGB)
                img = Image.open(img_paths[image_counter])
                if transform == True:
                    img = self.transform(img)
                    img = img.permute(1, 2, 0)
                axarr[r,c].imshow(img)
                image_counter += 1
    
    def get_examples_by_label2(self, label_id, num_samples = 3, transform = True):
        names = self.csv.loc[self.csv['landmark_id'] == label_id]
        names = names.sample(num_samples, replace=False)

        f, axarr = plt.subplots(1, len(names), figsize=(30,30))
        for idx, name in enumerate(names.id):
            image_path = f'{self.image_dir}{name[0]}/{name[1]}/{name[2]}/{name}.jpg'
            img = Image.open(image_path)
            if transform == True:
                img = self.transform(img)
                img = img.permute(1, 2, 0)
            axarr[idx].imshow(img)
            
    def get_examples_by_index(self, begin_idx, end_idx, columns = 5, transform = False):
        img_paths = self.data[begin_idx:end_idx]
        
        rows = int((end_idx-begin_idx)/columns)

        f, axarr = plt.subplots(rows, columns, figsize=(30,30))
        image_counter = 0
        for r in range(rows):
            for c in range(columns):
                img = Image.open(img_paths[image_counter])
                if transform == True:
                    img = self.transform(img)
                    img = img.permute(1, 2, 0)
                axarr[r,c].imshow(img)
                image_counter += 1
        
train_transforms = T.Compose([
                   T.Resize((224,224)),
                   T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1, hue=0.1),
                   T.RandomRotation((-90, 90), resample=PIL.Image.BICUBIC),
                   T.ToTensor(),
                   T.RandomErasing(p=0.5, scale=(0.05, 0.5), ratio=(0.3, 3.3))
                   #T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ])
def data_splitting(data, try_split_rate):
    #all_indicies = data['landmark_id'][]
    #print(all_indicies)
    train_list = []
    valid_list = []
    
    for i in tqdm(range(len(data['landmark_id'].unique()))):
        samples = data.loc[data['landmark_id'] == i]
        if len(samples) > 30:
            samples = samples.sample(30)
        
        train_samples = samples[0 : int(try_split_rate * len(samples))]
        valid_samples = samples[len(train_samples) : len(samples)]
        
        train_list.append(train_samples)
        valid_list.append(valid_samples)
    
    train_df = pd.concat(train_list, ignore_index=True) #, ignore_index=True
    valid_df = pd.concat(valid_list, ignore_index=True) #, ignore_index=True
    
    print(f'Total images: {len(data)}. Train: {len(train_df)} (ratio = {round(len(train_df)/len(data)*100, 2)}%). Valid: {len(valid_df)} (ratio = {round(len(valid_df)/len(data)*100, 2)}%)')
    return train_df, valid_df
    
train_df, valid_df = data_splitting(data_csv, 0.9)

train_dataset = LandmarkDataset(train_df, train_image_path, train_transforms, partition='Training/Validation')
valid_dataset = LandmarkDataset(valid_df, train_image_path, train_transforms, partition='Training/Validation')

train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, num_workers=multiprocessing.cpu_count(), shuffle=True, pin_memory=True)
valid_dataloader = DataLoader(valid_dataset, batch_size = BATCH_SIZE, num_workers=multiprocessing.cpu_count(), shuffle=True, pin_memory=True)
num_classes =  len(data_csv['landmark_id'].unique())
print(f'Originally the dataset consists of {num_classes} classes and {len(data_csv)} images.')
sns.set(rc={'figure.figsize':(25,10)})
ax = sns.countplot(x='landmark_id', data=data_csv[0:2000], order = data_csv['landmark_id'][0:2000].value_counts().index)
#print(f'The class with the most images ({data_csv.landmark_id.value_counts()[55807]}), presents {round((data_csv.landmark_id.value_counts()[55807] / len(data_csv)) * 100, 6)}% of the dataset.')
#print(f'A class with the least number of images ({data_csv.landmark_id.value_counts()[79]}), presents {round((data_csv.landmark_id.value_counts()[79] / len(data_csv)) * 100, 6)}% of the dataset.')
print(f'Around {round((data_csv.landmark_id.value_counts() <= 9).sum() / len(data_csv.landmark_id.unique()) * 100, 3)}% of the classes in the dataset has less than 10 images.')
print(f'After deleting data, the dataset consists of {len(data_csv.landmark_id.unique())} classes and {len(data_csv)} images.')
print(f'Which means that there have been a reducment of {round((len(train_df) + len(valid_df)) / len(data_csv) * 100, 4)}% images.')
#data_csv.get_examples_by_label2(label_id=55807, num_samples=5)
train_dataset.get_examples_by_label2(label_id=5, num_samples=2, transform=False)
train_dataset.get_examples_by_label2(label_id=5, num_samples=2, transform=True)
#train_dataset.get_examples_by_label2(label_id=9, num_samples=10, transform=False)
#landmark_dataset.get_random_examples(rows=3, columns=5, transform=True)
class AverageMeter:
    ''' Computes and stores the average and current value '''
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def GAP(predicts: torch.Tensor, confs: torch.Tensor, targets: torch.Tensor) -> float:
    ''' Simplified GAP@1 metric: only one prediction per sample is supported '''
    assert len(predicts.shape) == 1
    assert len(confs.shape) == 1
    assert len(targets.shape) == 1
    assert predicts.shape == confs.shape and confs.shape == targets.shape

    _, indices = torch.sort(confs, descending=True)

    confs = confs.cpu().numpy()
    predicts = predicts[indices].cpu().numpy()
    targets = targets[indices].cpu().numpy()

    res, true_pos = 0.0, 0

    for i, (c, p, t) in enumerate(zip(confs, predicts, targets)):
        rel = int(p == t)
        true_pos += rel

        res += true_pos / (i + 1) * rel

    res /= targets.shape[0] # FIXME: incorrect, not all test images depict landmarks
    return res
class VGG16(nn.Module):
    def __init__(self, num_classes = 1000):
        super(VGG16, self).__init__()
        self.vgg16_layers = models.vgg16_bn(pretrained=False)
        self.vgg16_layers.classifier[6] = nn.Linear(4096, num_classes)
        
    
    def forward(self, x):
        x = self.vgg16_layers(x)
        return x
class efficientnet(nn.Module):
    def __init__(self, num_classes = 1000):
        super(efficientnet, self).__init__()
        self.enet = enet.EfficientNet.from_name("efficientnet-b0")
        
        self.fc = nn.Linear(self.enet._fc.in_features, num_classes)
        self.enet._fc = nn.Identity()
    
    def forward(self, x):
        x = self.enet(x)
        x = self.fc(x)
        return x
#model = VGG16(num_classes = len(data_csv.landmark_id.unique()))
model = efficientnet(num_classes = len(data_csv.landmark_id.unique()))
model.to(device)
model
def training(model, dataloader, scheduler, optimizer):
    model.train()
    
    predict_label_list = []
    correct_label_list = []
    avg_score = AverageMeter()
    
    for data, labels in tqdm(dataloader):
        data = data.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        output = model(data)
        
        loss = criterion(output, labels)

        loss.backward()
        optimizer.step()
        scheduler.step()
        
        confedence, predictions = torch.max(output.detach(), dim=1)
        
        avg_score.update(GAP(predictions, confedence, labels))

        #predictions = output.max(dim=1)[1]
        #predict_label_list.append(predictions.cpu().detach()) #Item later
        #correct_label_list.append(labels.cpu().detach())

    # Calculate accuracy
    #predict_label_list = np.concatenate(predict_label_list)
    #correct_label_list = np.concatenate(correct_label_list)
    #accuracy = metrics.accuracy_score(predict_label_list, correct_label_list)
    return predict_label_list, correct_label_list, avg_score.avg
    #print(accuracy)
def validation(model, dataloader, scheduler):
    model.eval()
    
    predict_label_list = []
    correct_label_list = []
    avg_score = AverageMeter()
    
    with torch.no_grad():
        for data, labels in tqdm(dataloader):
            data = data.to(device)
            labels = labels.to(device)
            
            output = model(data)
            loss = criterion(output, labels)
            
            confedence, predictions = torch.max(output.detach(), dim=1)
            
            avg_score.update(GAP(predictions, confedence, labels))
            #predict_label_list.append(predictions.cpu().detach()) #Item later
            #correct_label_list.append(labels.cpu().detach())
    
    # Calculate accuracy
    #predict_label_list = np.concatenate(predict_label_list)
    #correct_label_list = np.concatenate(correct_label_list)
    #accuracy = metrics.accuracy_score(predict_label_list, correct_label_list)
    return predict_label_list, correct_label_list, avg_score.avg
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
criterion = nn.CrossEntropyLoss()
lr = 0.001
adam_optimizer = optim.Adam(model.parameters(), lr=lr)
sgd_optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.8, weight_decay=5e-4)
scheduler = lr_scheduler.CosineAnnealingLR(adam_optimizer, BATCH_SIZE, eta_min=lr/100)
train_acc = []
val_acc = []
train_gap = []
val_gap = []

start_time = time.time()
for epoch in range(EPOCHS):
    print(f'Epoch: {epoch}')
    if epoch > 0:
        train_dataset.update_pick_data()
        #valid_dataset.update_pick_data()
    #    landmark_dataset.update_pick_data()
    #    train_dataset, valid_dataset = random_split(landmark_dataset, (train_count, valid_count))

        train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, num_workers=multiprocessing.cpu_count(), shuffle=True, pin_memory=True)
    #    valid_dataloader = DataLoader(valid_dataset, batch_size = BATCH_SIZE, num_workers=multiprocessing.cpu_count(), shuffle=True, pin_memory=True)
    
    # Training part
    training_predicts, training_corrects, train_gap_score = training(model, train_dataloader, scheduler, optimizer=adam_optimizer)
    #training_acc = metrics.accuracy_score(training_predicts, training_corrects)
    #train_acc.append(training_acc)
    train_gap.append(train_gap_score)
    
    validation_predicts, validation_corrects, val_gap_score = validation(model, valid_dataloader, scheduler)
    #validation_acc = metrics.accuracy_score(validation_predicts, validation_corrects)
    #val_acc.append(validation_acc)
    val_gap.append(val_gap_score)
end_time = time.time()
print(f'Time taken for training: {(end_time - start_time)/60} minutes.')
from sklearn.metrics import confusion_matrix
confus = confusion_matrix(validation_corrects, validation_predicts)
confus_df = pd.DataFrame(confus)
plt.figure(figsize = (20,20))
sns.heatmap(confus_df, annot=True)
valid_dataset.get_examples_by_label2(label_id=35, num_samples=2, transform=False)
train_dataset.get_examples_by_label2(label_id=86, num_samples=2, transform=False)
fig = plt.figure(figsize=(20,10))
plt.title("GAP")
plt.plot(train_gap, label='Train')
plt.plot(val_gap, label='Validation')
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('GAP', fontsize=12)
plt.legend(loc='best')
fig = plt.figure(figsize=(20,10))
plt.title("Accuracy")
plt.plot(train_acc, label='Train')
plt.plot(val_acc, label='Validation')
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(loc='best')
submission_csv = pd.read_csv('../input/landmark-recognition-2020/sample_submission.csv')
test_image_path = '../input/landmark-recognition-2020/test/'
label_encoder = train_dataset.label_encoder

test_transforms = T.Compose([
                  T.Resize((224,224)),
                  T.ToTensor()
                   ])

test_dataset = LandmarkDataset(submission_csv, test_image_path, test_transforms,partition='Test')
test_dataloader = DataLoader(test_dataset, batch_size = BATCH_SIZE, num_workers=multiprocessing.cpu_count(), shuffle=False, pin_memory=True)
def testing(model, dataloader):
    model.eval()
    
    activation = nn.Softmax(dim=1)
    predict_label_list = []
    confedent_label_list = []
    
    with torch.no_grad():
        for data, labels in tqdm(dataloader):
            data = data.to(device)
        
            output = model(data)
            output = activation(output)
            
            confedence, predictions = torch.topk(output, 1) #20
            confedent_label_list.extend(confedence.cpu().detach())
            predict_label_list.extend(predictions.cpu().detach())
    
    return predict_label_list, confedent_label_list
def create_submission_csv(predictions, confidence):
    predictions = label_encoder.inverse_transform(predictions)
    predictions_after_thresh = []
    for i in range(len(submission_csv['landmarks'])):
        predicted_label = predictions[i].item()
        confedence_of_prediction = round(confidence[i].item(),4)
        if confedence_of_prediction > 0.9:
            submission_csv['landmarks'][i] = f'{predicted_label} {confedence_of_prediction}'
            predictions_after_thresh.append(predicted_label)
        else:
            submission_csv['landmarks'][i] = ''
    return submission_csv, predictions_after_thresh
predictions, confidence = testing(model, test_dataloader)
submission_csv, prediction_threshed = create_submission_csv(predictions,confidence)
submission_csv.to_csv('submission.csv', index=False)
prediction_threshed = label_encoder.transform(prediction_threshed)
p_df = pd.DataFrame(prediction_threshed, columns = ['landmark_id'])
sns.set(rc={'figure.figsize':(25,10)})
ax = sns.countplot(x='landmark_id', data=p_df, order = p_df['landmark_id'].value_counts().index)
def show_predictions(names, num_samples = 3, transform = False):
    names = names.sample(num_samples)

    f, axarr = plt.subplots(1, len(names), figsize=(30,30))
    for idx, name in enumerate(names):
        image_path = f'../input/landmark-recognition-2020/test/{name[0]}/{name[1]}/{name[2]}/{name}.jpg'
        img = Image.open(image_path)
        if transform == True:
            img = self.transform(img)
            img = img.permute(1, 2, 0)
        axarr[idx].imshow(img)
ids = p_df.loc[p_df['landmark_id'] == 63].index
predicted_names = submission_csv['id'].loc[ids]
show_predictions(predicted_names, num_samples = 2)
train_dataset.get_examples_by_label2(label_id=85, num_samples=2, transform=False)
