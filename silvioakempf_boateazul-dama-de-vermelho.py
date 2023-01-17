# Trabalho de Reconhecimento de Imagens Competição Kaggle

# Componentes:

# Alexandre 

# Eduardo

# Diogo

# Igor

# Silvio

# O Kernel do nosso trabalho partiu da base do Kernel: https://www.kaggle.com/kenseitrg/simple-fastai-exercise

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#Biblioteca de aprendizado profundo baseada na linguagemde programação LUA

#https://en.wikipedia.org/wiki/Torch_(machine_learning)

import torch 



# Biblioteca para facilitar a analise e processamento de imagens

# Guia https://docs.fast.ai/

from fastai import * 

from fastai.vision import *
# Lendo os arquivos e adicionando a variaveis

train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/sample_submission.csv")
# Armazenando as imagens para teste

test_img = ImageList.from_df(test_df, path='../input/test', folder='test')
# Criando variável com padrões para formatação das imagens

# do_flip : se True, um flip aleatório é aplicado com probabilidade de 0.5

# flip_vert : requer do_flip = True. Se for Verdadeiro, a imagem pode ser invertida verticalmente ou girada em 90 graus, caso contrário, apenas uma inversão horizontal é aplicada

# max_rotate : se não for None, uma rotação aleatória entre -max_rotate e max_rotate degrees é aplicada com probabilidade p_affine

# max_zoom : se não 1. ou menos, um zoom aleatório entre 1. e max_zoom é aplicado com probabilidade p_affine

# max_lighting : se não for None, uma alteração aleatória de raio e contraste controlada por max_lighting é aplicada com probabilidade p_lighting

# max_warp : se não for nenhum, uma deformação simétrica aleatória de magnitude entre -max_warp e maw_warp é aplicada com probabilidade p_affine

# p_affine : a probabilidade de cada transformada de afim e dobra simétrica ser aplicada

# p_lighting : a probabilidade de que cada transformação de iluminação seja aplicada



trfm2 = get_transforms(do_flip=True, flip_vert=True, max_rotate=0, max_zoom=0, max_lighting=0.2, max_warp=0.2, p_affine=0.5, p_lighting=0.5)
# Criando o treinamento baseado nas configurações criadas anteriormente, sendo processadas em cima da placa grafica 

# split_by_rand_pct : Como dividir em treino / válido? -> aleatoriamente com o padrão de 20% em válido

train_img2 = (ImageList.from_df(train_df, path='../input/train', folder='train')

        .split_by_rand_pct(0.01)

        .label_from_df()

        .add_test(test_img)

        .transform(trfm2, size=128)

        .databunch(path='.', bs=64, device= torch.device('cuda:0'))

        .normalize(imagenet_stats)

       )
# Aplicando o treinamento no modelo densenet161, que obteve a maior acurácia, tendo sido testados os modelos resnet18, inception_v3 e o densenet161

# Modelos https://pytorch.org/docs/stable/torchvision/models.html

learn2 = cnn_learner(train_img2, models.densenet161, metrics=[error_rate, accuracy])
# Gerando 5 ciclos de aprendizagem com o slice de 3e-02 que obteve a maior acurácia

# Slice :

#Em vez de definir manualmente um LR para cada grupo, geralmente é mais fácil de usar Learner.lr_range. 

#Este é um método de conveniência que retorna uma taxa de aprendizado para cada grupo de camadas. Se você passar, 

#slice(start,end)então a taxa de aprendizado do primeiro grupo é starta última end, e as restantes são uniformemente geometricamente espaçadas.

#Se você passar só slice(end)então a taxa de aprendizado do último grupo é end, e todos os outros grupos são end/10.

learn2.fit_one_cycle(5, slice(3e-02))
#Pegando os dados em forma de DataSet

preds,_ = learn2.get_preds(ds_type=DatasetType.Test)
#Formatando o dataset

test_df.has_cactus = preds.numpy()[:, 0]
#gerando o arquivo para submissão 

test_df.to_csv('submission.csv', index=False)
