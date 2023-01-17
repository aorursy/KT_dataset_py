! pip install pytorch-lightning==0.8.1
import os
import random as rn
from glob import glob

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import torchvision
from torchvision import transforms

import torch
import torch.nn as nn

from torch.nn import functional as F
from torch.autograd import Variable

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer

import matplotlib.pyplot as plt
%matplotlib inline

#fix random seed
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
torch.manual_seed(2019)
torch.cuda.manual_seed(2019)
torch.cuda.manual_seed_all(2019)
torch.backends.cudnn.deterministic = True
print(os.listdir("../input/data"))
all_xray_df = pd.read_csv('../input/data/Data_Entry_2017.csv')

#all_xray_df = all_xray_df.iloc[:1000]

all_image_paths = {os.path.basename(x): x for x in glob(os.path.join('..', 'input/data', 'images*', '*', '*.png'))}

print('Scans found:', len(all_image_paths), ', Total Headers', all_xray_df.shape[0])
all_xray_df['path'] = all_xray_df['Image Index'].map(all_image_paths.get)
#all_xray_df['Patient Age'] = all_xray_df['Patient Age'].map(lambda x: int(x[:-1]))
all_xray_df.sample(3)
label_counts = all_xray_df['Finding Labels'].value_counts()[:15]
fig, ax1 = plt.subplots(1,1,figsize = (12, 8))
ax1.bar(np.arange(len(label_counts))+0.5, label_counts)
ax1.set_xticks(np.arange(len(label_counts))+0.5)
_ = ax1.set_xticklabels(label_counts.index, rotation = 90)
labels = ['Cardiomegaly', 
          'Emphysema', 
          'Effusion', 
          'Hernia', 
          'Infiltration', 
          'Mass', 
          'Nodule', 
          'Atelectasis',
          'Pneumothorax',
          'Pleural_Thickening', 
          'Pneumonia', 
          'Fibrosis', 
          'Edema', 
          'Consolidation']
drop_column = ['Patient Age','Patient Gender','View Position','Follow-up #','OriginalImagePixelSpacing[x','y]','OriginalImage[Width','Height]','Unnamed: 11']
all_xray_df['Finding Labels'] = all_xray_df['Finding Labels'].map(lambda x: x.replace('No Finding', ''))
from itertools import chain
all_labels = np.unique(list(chain(*all_xray_df['Finding Labels'].map(lambda x: x.split('|')).tolist())))
all_labels = [x for x in all_labels if len(x)>0]
print('All Labels ({}): {}'.format(len(all_labels), all_labels))
for c_label in all_labels:
    if len(c_label)>1: # leave out empty labels
        all_xray_df[c_label] = all_xray_df['Finding Labels'].map(lambda finding: 1 if c_label in finding else 0)


all_xray_df.sample(3)
all_xray_df = all_xray_df.drop(drop_column,axis=1)
all_xray_df.sample(3)
label_counts = 100*np.mean(all_xray_df[all_labels].values,0)
fig, ax1 = plt.subplots(1,1,figsize = (12, 8))
ax1.bar(np.arange(len(label_counts))+0.5, label_counts)
ax1.set_xticks(np.arange(len(label_counts))+0.5)
ax1.set_xticklabels(all_labels, rotation = 90)
ax1.set_title('Adjusted Frequency of Diseases in Patient Group')
_ = ax1.set_ylabel('Frequency (%)')
all_xray_df['disease_vec'] = all_xray_df.apply(lambda x: [x[all_labels].values], 1).map(lambda x: x[0])
train_df, valid_df, test_df = np.split(all_xray_df.sample(frac=1), [int(.6*len(all_xray_df)), int(.8*len(all_xray_df))])
print('train', train_df.shape[0], 'validation', valid_df.shape[0], 'test', test_df.shape[0])
train_df.sample(3)
def compute_class_freqs(labels):
    """
    Compute positive and negative frequences for each class.

    Args:
        labels (np.array): matrix of labels, size (num_examples, num_classes)
    Returns:
        positive_frequencies (np.array): array of positive frequences for each
                                         class, size (num_classes)
        negative_frequencies (np.array): array of negative frequences for each
                                         class, size (num_classes)
    """
    
    # total number of patients (rows)
    N = len(labels)
    
    positive_frequencies = np.sum(labels, axis=0) / N
    negative_frequencies = 1 - positive_frequencies

    return positive_frequencies, negative_frequencies


train_labels = []
ds_len = train_df.shape[0]

for inx in range(ds_len):
    row = train_df.iloc[inx]
    vec = np.array(row['disease_vec'], dtype=np.int)
    train_labels.append(vec)


freq_pos, freq_neg = compute_class_freqs(train_labels)
freq_pos


data = pd.DataFrame({"Class": labels, "Label": "Positive", "Value": freq_pos})
data = data.append([{"Class": labels[l], "Label": "Negative", "Value": v} for l,v in enumerate(freq_neg)], ignore_index=True)
plt.xticks(rotation=90)
f = sns.barplot(x="Class", y="Value", hue="Label" ,data=data)

def weighted_loss(y_true, y_pred, pos_weights, neg_weights, epsilon=1e-7):
        """
        Return weighted loss value. 

        Args:
            y_true (Tensor): Tensor of true labels, size is (num_examples, num_classes)
            y_pred (Tensor): Tensor of predicted labels, size is (num_examples, num_classes)
        Returns:
            loss (Tensor): overall scalar loss summed across all classes
        """
        # initialize loss to zero
        loss = 0.0

        for i in range(len(pos_weights)):
            # for each class, add average weighted loss for that class 
            loss += -(torch.mean( pos_weights[i] * y_true[:,i] * torch.log(y_pred[:,i] + epsilon) + \
                                neg_weights[i] * (1 - y_true[:,i]) * torch.log(1 - y_pred[:,i] + epsilon), axis = 0))
            
        return loss
pos_weights = freq_neg
neg_weights = freq_pos
pos_contribution = freq_pos * pos_weights 
neg_contribution = freq_neg * neg_weights
data = pd.DataFrame({"Class": labels, "Label": "Positive", "Value": pos_contribution})
data = data.append([{"Class": labels[l], "Label": "Negative", "Value": v} 
                        for l,v in enumerate(neg_contribution)], ignore_index=True)
plt.xticks(rotation=90)
sns.barplot(x="Class", y="Value", hue="Label" ,data=data);
ids_train = train_df['Patient ID'].values
ids_valid = valid_df['Patient ID'].values

# Create a "set" datastructure of the training set id's to identify unique id's
ids_train_set = set(ids_train)
print(f'There are {len(ids_train_set)} unique Patient IDs in the training set')
# Create a "set" datastructure of the validation set id's to identify unique id's
ids_valid_set = set(ids_valid)
print(f'There are {len(ids_valid_set)} unique Patient IDs in the training set')
# Identify patient overlap by looking at the intersection between the sets
patient_overlap = list(ids_train_set.intersection(ids_valid_set))
n_overlap = len(patient_overlap)
# print(f'There are {n_overlap} Patient IDs in both the training and validation sets')
# print('')
# print(f'These patients are in both the training and validation datasets:')
# print(f'{patient_overlap}')
train_overlap_idxs = []
valid_overlap_idxs = []
for idx in range(n_overlap):
    train_overlap_idxs.extend(train_df.index[train_df['Patient ID'] == patient_overlap[idx]].tolist())
    valid_overlap_idxs.extend(valid_df.index[valid_df['Patient ID'] == patient_overlap[idx]].tolist())
    
# print(f'These are the indices of overlapping patients in the training set: ')
# print(f'{train_overlap_idxs}')
# print(f'These are the indices of overlapping patients in the validation set: ')
# print(f'{valid_overlap_idxs}')
# Drop the overlapping rows from the validation set
valid_df.drop(valid_overlap_idxs, inplace=True)
# Extract patient id's for the validation set
ids_valid = valid_df['Patient ID'].values
# Create a "set" datastructure of the validation set id's to identify unique id's
ids_valid_set = set(ids_valid)
print(f'There are {len(ids_valid_set)} unique Patient IDs in the training set')
# Identify patient overlap by looking at the intersection between the sets
patient_overlap = list(ids_train_set.intersection(ids_valid_set))
n_overlap = len(patient_overlap)
print(f'There are {n_overlap} Patient IDs in both the training and validation sets')
def check_for_leakage(df1, df2, patient_col):
    """
    Return True if there any patients are in both df1 and df2.

    Args:
        df1 (dataframe): dataframe describing first dataset
        df2 (dataframe): dataframe describing second dataset
        patient_col (str): string name of column with patient IDs
    
    Returns:
        leakage (bool): True if there is leakage, otherwise False
    """
 
    df1_patients_unique = set(df1[patient_col].unique().tolist())
    df2_patients_unique = set(df2[patient_col].unique().tolist())
    
    patients_in_both_groups = df1_patients_unique.intersection(df2_patients_unique)

    # leakage contains true if there is patient overlap, otherwise false.
    leakage = len(patients_in_both_groups) >= 1 # boolean (true if there is at least 1 patient in both groups)
    
    return leakage
print("leakage between train and test: {}".format(check_for_leakage(train_df, valid_df, 'Patient ID')))
from PIL import Image
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_frame, transforms=None):
        self.data_frame = data_frame
        self.transforms = transforms
        self.len = data_frame.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        row = self.data_frame.iloc[index]
        address = row['path']
        x = Image.open(address).convert('RGB')
        
        vec = np.array(row['disease_vec'], dtype=np.float)
        y = torch.FloatTensor(vec)
        
        if self.transforms:
            x = self.transforms(x)
        return x, y
    

train_transform = transforms.Compose([ 
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop(224, scale=(0.63, 1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ])

test_transform = transforms.Compose([ 
    transforms.Resize(230),
    transforms.CenterCrop(224),
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ])

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)
dsetTrain = CustomDataset(train_df, train_transform) 
dsetVal = CustomDataset(valid_df, test_transform) 
dsetTest = CustomDataset(test_df, test_transform)

trainloader = torch.utils.data.DataLoader( dataset = dsetTrain, batch_size = 12, shuffle = True, num_workers = 8 )
valloader = torch.utils.data.DataLoader( dataset = dsetVal, batch_size = 12, shuffle = False, num_workers = 8 )
testloader = torch.utils.data.DataLoader( dataset = dsetTest, batch_size = 100, shuffle = False, num_workers = 8 )

class DensModel(LightningModule):

    def __init__(self):
        super().__init__()
               
        #self.metric = Accuracy()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        
        feature_extracting = True
        self.set_parameter_requires_grad(self.densenet121, feature_extracting)

        num_ftrs = self.densenet121.classifier.in_features

        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 14),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam (model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience = 5, mode = 'min')

        return [optimizer], [scheduler]

    def set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
#         loss = F.binary_cross_entropy(y_hat, y, size_average = True)
        loss = weighted_loss(y, y_hat, pos_weights, neg_weights, epsilon=1e-7)
        
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}


    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        #loss = F.binary_cross_entropy(y_hat, y, size_average = True)
        loss = weighted_loss(y, y_hat, pos_weights, neg_weights, epsilon=1e-7)
        #acc = self.metric(y_hat, y)
    
        return {'val_loss': loss}


    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}


    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

#         loss = F.binary_cross_entropy(y_hat, y, size_average = True)
        
        loss = weighted_loss(y, y_hat, pos_weights, neg_weights, epsilon=1e-7)

        return {'test_loss': loss}


    def test_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss}
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}


    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}



model = DensModel()

early_stopping = EarlyStopping('val_loss')
checkpoint_callback = ModelCheckpoint(verbose=False, monitor='avg_val_loss', mode='min')

trainer = Trainer(gpus=1, max_epochs=5, checkpoint_callback=checkpoint_callback, early_stop_callback=early_stopping)
trainer.fit(model, trainloader, valloader)

trainer.test(model, test_dataloaders=testloader)




def get_roc_curve(labels, predicted_vals, gt_labels):
    auc_roc_vals = []
    for i in range(len(labels)):
        try:
            gt = gt_labels[:, i]
            pred = predicted_vals[:, i]
            auc_roc = roc_auc_score(gt, pred)
            auc_roc_vals.append(auc_roc)
            fpr_rf, tpr_rf, _ = roc_curve(gt, pred)
            plt.figure(1, figsize=(10, 10))
            plt.plot([0, 1], [0, 1], 'k--')
            plt.plot(fpr_rf, tpr_rf,
                     label=labels[i] + " (" + str(round(auc_roc, 3)) + ")")
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            plt.title('ROC curve')
            plt.legend(loc='best')
        except:
            print(
                f"Error in generating ROC curve for {labels[i]}. "
                f"Dataset lacks enough examples."
            )
    plt.show()
    return auc_roc_vals
# for x, y in testloader:
    
#     model.eval()
#     x = torch.autograd.Variable(x).cuda()
    
    
#     with torch.no_grad():
#         outputs = model(x)
#         print(outputs.shape)
        
#         gt_labels = y.cpu().numpy()
#         predicted_vals = outputs.cpu().numpy()
        
#         for c_label, s_count in zip(all_labels, 100*np.mean(y.numpy(),0)):
#             print('%s: %2.2f%%' % (c_label, s_count))
#         break
# from sklearn.metrics import roc_curve, auc
# test_Y = y.numpy()
# pred_Y = predicted_vals
# fig, c_ax = plt.subplots(1,1, figsize = (9, 9))

# for (idx, c_label) in enumerate(all_labels):
#     fpr, tpr, thresholds = roc_curve(test_Y[:,idx].astype(int), pred_Y[:,idx])
#     c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, auc(fpr, tpr)))

# c_ax.legend()
# c_ax.set_xlabel('False Positive Rate')
# c_ax.set_ylabel('True Positive Rate')
# fig.savefig('barely_trained_net.png')
