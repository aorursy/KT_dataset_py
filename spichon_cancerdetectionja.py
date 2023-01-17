import numpy as np 

import pandas as pd 

import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import random



#Importation de la librairie FastAI

from fastai import *

from fastai.vision import *

from torchvision.models import *



#Utilisation de la séparation train/test de sklearn

from sklearn.model_selection import train_test_split  



print("Affichage de notre environnement de travail")

print(os.listdir("../input"))



data = pd.read_csv('/kaggle/input/train_labels.csv')

#Configuration du chemin d'entrainement

train_path = '/kaggle/input/train/'

#Configuration du chemin de test

test_path = '/kaggle/input/test/'
train_df = data.set_index('id')

train_names = train_df.index.values

train_labels = np.asarray(train_df['label'].values)

tr_n, tr_idx, val_n, val_idx = train_test_split(train_names, range(len(train_names)), test_size=0.1, stratify=train_labels, random_state=123)
# Création du dataframe d'entrainement

train_dict = {'name': train_path + train_names, 'label': train_labels}

df = pd.DataFrame(data=train_dict)

# Création du dataset de test

test_names = []

for f in os.listdir(test_path):

    test_names.append(test_path + f)

df_test = pd.DataFrame(np.asarray(test_names), columns=['name'])
#On créé ici notre dataBunch, équivalent d'un flow de données alimentant notre modèle de deep learning

tfms = get_transforms()

BATCH_SIZE=128

SIZE=90

data = ImageList.from_df(path='/', df=df, suffix='.tif').split_by_idx(val_idx).label_from_df(cols='label').add_test(ImageList.from_df(path='/', df=df_test)).databunch(bs=BATCH_SIZE).normalize(imagenet_stats)
#Affichage de données sans et avec cancer

data.show_batch(rows=3, figsize=(7, 8))
print(data)
# L'architecture de notre modèle, ici densenet169 

arch = densenet169    

# Exraction du nom du modèle

MODEL_PATH = str(arch).split()[1] 

#Création de notre learner, réseau convolutif d'architecture densenet169

learn = create_cnn(data, arch, metrics=error_rate)
# Utilisation de la GPU

defaults.device = torch.device('cuda') 

# Premier entrainement du modèle

learn.fit_one_cycle(4)

#Enregistrement du modèle

learn.save('cancer-detection-1')
# On dégèle les dernière couches du modèles pour l'entrainer une seconde fois par la suite

learn.unfreeze() 

#Avant de réentrainer le modèle, on recherche du meilleur taux d'apprentissage en suivant la technique "différential learning late".

learn.lr_find()

learn.recorder.plot()
#Via l'analyse graphique, le meilleur taux d'apprentissage se trouver être 4e-4

learn.fit_one_cycle(4, max_lr=1e-4)
#Enregistrement du modèle

learn.save('cancer-detection-2')
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix()
learn.recorder.plot_lr()

learn.recorder.plot_losses()
preds,y, loss = learn.get_preds(with_loss=True)

acc = accuracy(preds, y)

print('The accuracy is {0} %.'.format(acc))
from sklearn.metrics import roc_curve, auc

probs = np.exp(preds[:,1])

fpr, tpr, thresholds = roc_curve(y, probs, pos_label=1)



roc_auc = auc(fpr, tpr)

print('ROC area is {0}'.format(roc_auc))

plt.figure()

plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

plt.xlim([-0.01, 1.0])

plt.ylim([0.0, 1.01])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")
#Inférence sur notre test_set

preds,y = learn.get_preds(ds_type=DatasetType.Test, with_loss=False)

tumor_preds = preds[:, 1]
#Exporter sous le bon format pour la soumission

SAMPLE_SUB = '/kaggle/input/sample_submission.csv'

sample_df = pd.read_csv(SAMPLE_SUB)

sample_list = list(sample_df.id)

pred_list = [float(p) for p in tumor_preds]

pred_dic = dict((key, value) for (key, value) in zip(learn.data.test_ds.items, pred_list))

pred_list_cor = [pred_dic['///kaggle/input/test/' + id + '.tif'] for id in sample_list]

df_sub = pd.DataFrame({'id':sample_list,'label':pred_list_cor})

df_sub.to_csv('{0}_submission.csv'.format(MODEL_PATH), header=True, index=False)

df_sub