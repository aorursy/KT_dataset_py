!pip --quiet install pytorch-lightning
import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set(style="darkgrid")



from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix, roc_curve, auc



import torch

from torch.utils.data import Dataset, DataLoader

import torch.optim as torch_optim

import torch.nn as nn

import torch.nn.functional as F



from pytorch_lightning.core.lightning import LightningModule

from pytorch_lightning import Trainer

from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from pytorch_lightning.metrics.functional import auroc, accuracy



from pytorch_lightning.trainer import seed_everything
seed_everything(seed=123)
col_types = {"image_name": str,

            "patient_id": str,

            "sex": str,

            "age_approx": np.float16,

            "anatom_site_general_challenge": str,

            "diagnosis": str,

            "benign_malignant": str,

            "target": np.uint8}

df = pd.read_csv("../input/siim-isic-melanoma-classification/train.csv", dtype=col_types)

df.reset_index(drop=True, inplace=True)

df.head(3)
class MelanomaImageDataset(Dataset):

    def __init__(self, path, df):

        self.df = df

        if "target" in self.df.columns:

            self.target = torch.from_numpy(df.target.values).type(torch.FloatTensor)

        self.path = path



    def __getitem__(self, idx):

        image_name = self.df.image_name[idx]

        image = torch.load(os.path.join(self.path, image_name + '.pt'))

        if "target" in self.df.columns:

            label = self.target[idx]

            return image, label

        else:

            return image

                        

    def __len__(self):

        return self.df.shape[0]
PATH_PT_FILE = '../input/melanoma-grayscale-64x64/images64x64.pt/train'

BATCH_SIZE = 128



# split train/val/test

train_df, val_df = train_test_split(df, stratify=df.target, test_size=0.20)

train_df.reset_index(drop=True, inplace=True)

val_df.reset_index(drop=True, inplace=True)



val_df, test_df = train_test_split(val_df, stratify=val_df.target, test_size=0.5)

val_df.reset_index(drop=True, inplace=True)

test_df.reset_index(drop=True, inplace=True)



#creating train, valid and test datasets

train_ds = MelanomaImageDataset(PATH_PT_FILE, train_df)

valid_ds = MelanomaImageDataset(PATH_PT_FILE, val_df)

test_ds = MelanomaImageDataset(PATH_PT_FILE, test_df)

#creating train, valid and test dataloaders

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=32, shuffle=True)

valid_dl = DataLoader(valid_ds, batch_size=BATCH_SIZE, num_workers=32, shuffle=False)

test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=32, shuffle=False)
print(train_df.shape, val_df.shape, test_df.shape)

pd.DataFrame({'train': train_df.target.value_counts(), 'val': val_df.target.value_counts(), 'test': test_df.target.value_counts()})
class MelanomaModel(LightningModule):



    def __init__(self):

        super().__init__()

        # conv kernels

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3)

        self.conv2 = nn.Conv2d(6, 16, 3)

        self.conv3 = nn.Conv2d(16, 32, 3)

        # dense layers: an affine operation: y = Wx + b

        self.fc1 = nn.Linear(32 * 6 * 6, 120)  # 6*6 from image dimension

        self.fc2 = nn.Linear(120, 84)

        self.fc3 = nn.Linear(84, 1)

        self.sigmoid = nn.Sigmoid() 

        

        # callback metrics

        self.metrics = {'train_loss': [], 'train_acc': [], 'train_aucroc': [],

                        'val_loss': [], 'val_acc': [], 'val_aucroc': []}



    def forward(self, x):

        # Max pooling over a (2, 2) window

        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))

        # If the size is a square we can only specify a single number

        x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        x = F.max_pool2d(F.relu(self.conv3(x)), 2)

        x = torch.flatten(x, start_dim=1) # except the batch size dim

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = self.fc3(x)

        return self.sigmoid(x).flatten()

        

    def training_step(self, batch, batch_idx):

        x, y = batch

        y_hat = self(x)

        loss = F.binary_cross_entropy(y_hat, y)

        return {'loss': loss, "y_hat": y_hat, "y": y}

    

    def training_epoch_end(self, outputs):

        # concat or stack batchs outputs

        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        train_y_hat = torch.cat([x['y_hat'] for x in outputs], dim=-1)

        train_y = torch.cat([x['y'] for x in outputs], dim=-1)

        # compute accuracy and roc auc

        train_bin_y_hat = (train_y_hat > 0.5).float() * 1

        acc = accuracy(train_bin_y_hat, train_y)

        aucroc = auroc(train_y_hat, train_y)

#         print("train_loss {0:.6f} || train_acc {1:.6f} || train_aucroc {2:.6f}".format(avg_loss, acc, aucroc))

        # Record metrics

        self.metrics['train_loss'].append(avg_loss)

        self.metrics['train_acc'].append(acc)

        self.metrics['train_aucroc'].append(aucroc)

        return {'train_loss': avg_loss}

    

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        return optimizer

    

    def validation_step(self, batch, batch_idx):

        x, y = batch

        y_hat = self(x)

        return {'val_loss': F.binary_cross_entropy(y_hat, y), "y_hat": y_hat, "y": y}



    def validation_epoch_end(self, outputs):

        # concat or stack batchs outputs

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        val_y_hat = torch.cat([x['y_hat'] for x in outputs], dim=-1)

        val_y = torch.cat([x['y'] for x in outputs], dim=-1)

        # compute accuracy and roc auc

        val_bin_y_hat = (val_y_hat > 0.5).float() * 1

        acc = accuracy(val_bin_y_hat, val_y)

        aucroc = auroc(val_y_hat, val_y)

#         print("val_loss {0:.6f} || val_acc {1:.6f} || val_aucroc {2:.6f}".format(avg_loss, acc, aucroc))

        # Record metrics

        self.metrics['val_loss'].append(avg_loss)

        self.metrics['val_acc'].append(acc)

        self.metrics['val_aucroc'].append(aucroc)

        return {'val_loss': avg_loss, 'val_auc': aucroc} 
model = MelanomaModel()

model
tensor_images = torch.randn(5, 1, 64, 64)

outputs = model(tensor_images)

print(outputs)
%%time

early_stop_callback = EarlyStopping(monitor='val_auc',

                                    min_delta=0.00,

                                    patience=5,

                                    verbose=True,

                                    mode='max')

trainer = Trainer(max_epochs=100,

                  gpus=1,

                  check_val_every_n_epoch=1,

                  num_sanity_val_steps=0,

                  early_stop_callback=early_stop_callback)

trainer.fit(model, train_dl, valid_dl)
for key in model.metrics:

    model.metrics[key] = np.array([val.detach().cpu().numpy() for val in model.metrics[key]])



df_metrics = pd.DataFrame.from_dict(model.metrics)

df_metrics["epochs"] = np.arange(0, len(df_metrics))

df_metrics.head(3)
sns.set(font_scale=1.)

plt.figure(figsize=(25,7))

plt.subplot(131)

plt.title("Loss")

sns.lineplot(x="epochs", y="train_loss", data=df_metrics, label="train_loss")

sns.lineplot(x="epochs", y="val_loss", data=df_metrics, label="val_loss")

plt.subplot(132)

plt.title("Accuracy")

sns.lineplot(x="epochs", y="train_acc", data=df_metrics, label="train_acc")

sns.lineplot(x="epochs", y="val_acc", data=df_metrics, label="val_acc")

plt.subplot(133)

plt.title("AUC ROC")

sns.lineplot(x="epochs", y="train_aucroc", data=df_metrics, label="train_aucroc")

sns.lineplot(x="epochs", y="val_aucroc", data=df_metrics, label="val_aucroc")

plt.show()
preds = []

for image, label in test_dl:

    preds.append(model(image).detach())
y_pred_prob = torch.cat(preds, dim=-1).cpu().numpy()

y_pred = np.where(y_pred_prob >= .5, 1, 0)

y_true = test_df.target.values
plt.figure(figsize=(15,6))



## CONFUSION MATRIX

plt.subplot(121)

# Set up the labels for in the confusion matrix

cm = confusion_matrix(y_true, y_pred)

names = ['True Negatives', 'False Positives', 'False Negatives', 'True Positives']

counts = ['{0:0.0f}'.format(value) for value in cm.flatten()]

percentages = ['{0:.2%}'.format(value) for value in cm.flatten()/np.sum(cm)]

labels = [f'{v1}\n{v2}' for v1, v2 in zip(names, percentages)]

labels = np.asarray(labels).reshape(2,2)

ticklabels = ['Normal', 'Pneumonia']



# Create confusion matrix as heatmap

sns.set(font_scale = 1.4)

ax = sns.heatmap(cm, annot=labels, fmt='', cmap='Oranges', xticklabels=ticklabels, yticklabels=ticklabels )

plt.xticks(size=12)

plt.yticks(size=12)

plt.title("Confusion Matrix") #plt.title("Confusion Matrix\n", fontsize=10)

plt.xlabel("Predicted", size=14)

plt.ylabel("Actual", size=14) 

#plt.savefig('cm.png', transparent=True) 



## ROC CURVE

plt.subplot(122)

fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)

auc = roc_auc_score(y_true, y_pred_prob)

plt.title('ROC Curve')

plt.plot([0, 1], [0, 1], 'k--', label = "Random (AUC = 50%)")

plt.plot(fpr, tpr, label='CNN (AUC = {:.2f}%)'.format(auc*100))

plt.xlabel('False Positive Rate', size=14)

plt.ylabel('True Positive Rate', size=14)

plt.legend(loc='best')

#plt.savefig('roc.png', bbox_inches='tight', pad_inches=1)



## END PLOTS

plt.tight_layout()



## Summary Statistics

TN, FP, FN, TP = cm.ravel() # cm[0,0], cm[0, 1], cm[1, 0], cm[1, 1]

acc = (TP + TN) / np.sum(cm) # % positive out of all predicted positives

precision = TP / (TP+FP) # % positive out of all predicted positives

recall =  TP / (TP+FN) # % positive out of all supposed to be positives

specificity = TN / (TN+FP) # % negative out of all supposed to be negatives

f1 = 2*precision*recall / (precision + recall)

stats_summary = '[Summary Statistics]\nAccuracy = {:.2%} | Precision = {:.2%} | Recall = {:.2%} | Specificity = {:.2%} | F1 Score = {:.2%}'.format(acc, precision, recall, specificity, f1)

print(stats_summary)
df_test = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')

df_test.head(3)
PATH_TEST_PT_FILES = '../input/melanoma-grayscale-64x64/images64x64.pt/test'

BATCH_SIZE = 128



test_ds = MelanomaImageDataset(PATH_TEST_PT_FILES, df_test)

test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=32, shuffle=False)
preds = []

for image in test_dl:

    preds.append(model(image).detach())

y_pred_prob = torch.cat(preds, dim=-1).cpu().numpy()
sub = pd.read_csv('../input/siim-isic-melanoma-classification/sample_submission.csv')

sub['target'] = y_pred_prob

sub.to_csv('submission.csv', index=False)