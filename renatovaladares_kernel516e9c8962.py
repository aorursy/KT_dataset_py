#Conceitos de Redes Convolucionais



#Arquitetura



#Trata se da organização da estrutura da rede, ou seja:

#Quantidade de camadas (e quantidade de nós em cada camada);

#Conexões;

#Parâmetros;

#Unidades de aprendizado.

#As redes são organizadas em camadas, que ganham um formato de cadeia. Em cada camada é aplicada a função de ativação.

#Os valores de saída das camadas servem de entrada para a camada seguinte.





#Camadas de Convolução



#Existem Arquiteturas específicas, isto é, desenvolvidas para realizar tarefas de alta complexidade: aplicação de filtros para gerar imagens novas ou reconhecimento de imagens.

#As redes convolucionais são um exemplo deste tipo de arquitetura, são redes especializadas em processamento de dados com topologias que podem ser semelhante a grades – dados de imagem #com representação em uma grade 2D de pixels por exemplo.

#O nome convolucional faz referência a compilação matemática realizada neste tipo de rede. Faz-se uso da convolução no lugar da multiplicação de matrizes em pelo menos uma de suas #camadas.

#Convolução – Operador linear aplicado a duas funções cujo objetivo é medir uma terceira área subentendida pela sobreposição das mesmas em função do deslocamento existente entre elas.



    

#Camadas de Pooling



#As camadas de agrupamento possuem a função de reduzir progressivamente o tamanho espacial da representação para reduzir a quantidade de parâmetros e o esforço computacional.

#Simplifica a informação da camada anterior.



    

#Funções de ativação





#São funções que Trazem a não-linearidade ao sistema -  para que a rede consiga aprender qualquer tipo de funcionalidade



#Filtros



#É um componente da camada de convolução, Trata - se da Matriz utilizada para multiplicar um conjunto de pixels para realizar uma transformação – comportamento, imagens. Cada filtro #produz um ângulo novo da imagem.





#Camada densa



#Última camada, normalmente representada por uma rede Perceptron de múltiplas camadas – MLP.



#Transfer Learning



#Aproveitamento de uma rede (Camadas, Filtros e pesos da rede neural) Que serão expostos a um novo grupo de treinamento com novas classes de saída.

#Visa Aproveitar um Conhecimento Prévio e otimizar o tempo Face a  possibilidade de convergência mais rápida para os pesos reais.





#Data Augmentation



#Aumentar a quantidade de amostras de forma artificial, significa a Agregação de valor aos dados da base através da adição de informações derivadas de transformações.





#Importando Bibliotecas 



#Bibliotecas Módulos Utilização



#Numpy            Usada com frequência para realizar cálculos em arrays multidimensionais

#Pandas           Fornece ferramentas de análise de dados e estruturas de dados.

#Pathlib          Object oriented filesystem paths  Trata se de um módulo com várias classes.

#Fastai           Biblioteca opensource para Deep Learning

#Fastai.vision    Módulo da biblioteca fastai com funções para definir um Dataset e treinar modelos para

#                        tarefas de visão computacional

#Torch            Biblioteca de machine learning, framework de computação científica e script language

#                         baseado em uma linguagem de programação (Lua).

#OS               Módulo de Iteração com o sistema operacional  vem na instalação do Python

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import torch 



# Biblioteca para facilitar a analise e processamento de imagens

# Guia https://docs.fast.ai/

from fastai import * 

from fastai.vision import *
#Leitura dos arquivos e declaração de variáveis
train_df = pd.read_csv("../input/aerial-cactus-identification/train.csv")

test_df = pd.read_csv("../input/aerial-cactus-identification/sample_submission.csv")
test_img = ImageList.from_df(test_df, path='../input/aerial-cactus-identification/test', folder='test')
#Criar variável com padrões para formatação da imagem



#do_flip : seTrue, um flip aleatório é aplicado com probabilidade de 0,5 

#flip_vert : requer do_flip=True . Se for Verdadeiro, a imagem pode ser invertida verticalmente ou girada em 90 graus, caso contrário, apenas uma inversão horizontal é aplicada	

#max_rotate : se não for None, uma rotação aleatória entre -max_rotate e max_rotate degrees é aplicada com probabilidade p_affine

#max_zoom : se não 1. ou menos, um zoom aleatório entre 1. e max_zoom é aplicado com probabilidade p_affine

#max_lighting : se não for None, uma alteração aleatória de raio e contraste controlada por max_lighting é aplicada com probabilidade p_lighting 

#max_warp : se não for nenhum, uma deformação simétrica aleatória de magnitude entre -max_warp e maw_warp é aplicada com probabilidade p_affine

#p_affine : a probabilidade de cada transformada de afim e dobra simétrica ser aplicada 

#p_lighting : a probabilidade de que cada transformação de iluminação seja aplicada
trfm2 = get_transforms(do_flip=True, flip_vert=True, max_rotate=0, max_zoom=0, max_lighting=0.2, max_warp=0.2, p_affine=0.5, p_lighting=0.5)
#Criar treinamento baseado nas configurações criadas anteriormente!



#Processadas em cima da placa grafica 	# split_by_rand_pct : Como dividir em treino / válido? -> aleatoriamente com o padrão de 20% em válido
train_img2 = (ImageList.from_df(train_df, path='../input/aerial-cactus-identification/train', folder='train')

        .split_by_rand_pct(0.01)

        .label_from_df()

        .add_test(test_img)

        .transform(trfm2, size=128)

        .databunch(path='.', bs=64, device= torch.device('cuda:0'))

        .normalize(imagenet_stats)

       )
#Aplicar o treinamento no modelo densenet161

# Modelo que obteve a maior acurácia, tendo sido testados os modelos resnet18, inception_v3 e o densenet161

# Modelos https://pytorch.org/docs/stable/torchvision/models.html
learn2 = cnn_learner(train_img2, models.densenet161, metrics=[error_rate, accuracy])
# Gerar 5 ciclos de aprendizagem com o slice de 3e-02 

# Obteve a maior acurácia

# Slice #Em vez de definir manualmente um LR para cada grupo, geralmente é mais fácil de usar Learner.lr_range.

#Este é um método de conveniência que retorna uma taxa de aprendizado para cada grupo de camadas.

#Se você passar, #slice(start,end) então a taxa de aprendizado do primeiro grupo é start a última end, e as restantes são uniformemente geometricamente espaçadas.

#Se você passar só slice(end) então a taxa de aprendizado do último grupo é end, e todos os outros grupos são end/10.
learn2.fit_one_cycle(5, slice(3e-02))
# Buscar os dados em forma de DataSet

# Formatando o dataset
preds,_ = learn2.get_preds(ds_type=DatasetType.Test)
test_df.to_csv('submission.csv', index=False)
#Gerar arquivo para submissão:
test_df.has_cactus = preds.numpy()[:, 0]