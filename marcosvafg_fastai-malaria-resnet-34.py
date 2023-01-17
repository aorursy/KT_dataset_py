# importando bibliotecas

import numpy as np

from fastai import *

from fastai.vision import *



# random seed

np.random.seed(42)
# Definindo o caminho das imagens

img_dir = '/kaggle/input/cell-images-for-detecting-malaria/cell_images/cell_images'



# transformando em path

path = Path(img_dir)

path
# carregando as imagens

# definindo os conjuntos de treino e validação - 20%

# normalizando os dados (ImageNet)

data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,

                                  ds_tfms=get_transforms(flip_vert=True, max_warp=0),

                                  size=224, bs=64, num_workers=0)

data.normalize(imagenet_stats)
# Visulizando algumas imagens

data.show_batch(rows=3, figsize=(7,6))
# Criando umas CNN usando um modelo pré-treinado (ResNet 34)

learn = create_cnn(data, models.resnet34, metrics=[error_rate, accuracy],

                   model_dir="/tmp/model/")
# Treinando a rede por 10 epochs

learn.fit_one_cycle(10)
# Salvando o modelo

learn.save('modelo-1')
# Encontrando o melhor Learning Rate para retrinar o modelo

learn.lr_find()

learn.recorder.plot()
# Descongelamos o modelo

learn.unfreeze()



# e retreinamos

learn.fit_one_cycle(2, max_lr=slice(1e-3,1e-2))
# Salvando o modelo

learn.save('modelo-2')
# Interpretação

interp = ClassificationInterpretation.from_learner(learn)
# Exibindo os maiores erros

interp.plot_top_losses(9, figsize=(15,11))
# Confusion Matrix

interp.plot_confusion_matrix(figsize=(8,8), dpi=60)