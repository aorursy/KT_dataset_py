import numpy as np 

import pandas as pd

import torch 



from fastai import * 

from fastai.vision import *
# Lendo os arquivos e adicionando a variaveis

train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/sample_submission.csv")
# Armazenando as imagens para teste

test_img = ImageList.from_df(test_df, path='../input/test', folder='test')
trfm2 = get_transforms(do_flip=True, flip_vert=True, max_rotate=0, max_zoom=0, max_lighting=0.2, max_warp=0.2, p_affine=0.5, p_lighting=0.5)
train_img2 = (ImageList.from_df(train_df, path='../input/train', folder='train')

        .split_by_rand_pct(0.01)

        .label_from_df()

        .add_test(test_img)

        .transform(trfm2, size=128)

        .databunch(path='.', bs=64, device= torch.device('cuda:0'))

        .normalize(imagenet_stats)

       )
learn2 = cnn_learner(train_img2, models.densenet161, metrics=[error_rate, accuracy])
learn2.fit_one_cycle(5, slice(3e-02))
#Pegando os dados em forma de DataSet

preds,_ = learn2.get_preds(ds_type=DatasetType.Test)
#Formatando o dataset

test_df.has_cactus = preds.numpy()[:, 0]
#gerando o arquivo para submissão 

test_df.to_csv('submission.csv', index=False)
