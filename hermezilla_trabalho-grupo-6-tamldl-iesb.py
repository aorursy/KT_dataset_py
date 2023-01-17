# Importando as bibliotecas
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import cv2

from keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Activation, BatchNormalization
from tensorflow.keras.models import Model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import seaborn as sns

print(tf.__version__)
#Criando variáveis para referenciar o caminho dos arquivos
data_dir = "../input/coin-images/coins/data"

data_train_path =  data_dir + '/train'
data_valid_path = data_dir + '/validation'
data_test_path =  data_dir + '/test'

print(os.listdir("../input/coin-images/coins/data"))
#Criando o tradutor das classes de moedas
with open('../input/coin-images/cat_to_name.json', 'r') as json_file:
    cat_2_name = json.load(json_file)


# Listando 10 classes aleatórias
for i in np.random.choice(a = range(1,211), size = 10, replace=False):
    print('Classe '+str(i)+': '+cat_2_name[str(i)])
#Criando um dataframe com as informações relevantes
fileray = []
classray = []
partitionray = []
classnameray = []


for dirname, _, filenames in os.walk(data_train_path):
    for filename in filenames:
        fileray.append(os.path.join(dirname, filename))
        classray.append(int(dirname.replace(data_train_path+'/','')))
        partitionray.append(0)
        classnameray.append(cat_2_name[dirname.replace(data_train_path+'/','')])

for dirname, _, filenames in os.walk(data_valid_path):
    for filename in filenames:
        fileray.append(os.path.join(dirname, filename))
        classray.append(int(dirname.replace(data_valid_path+'/','')))
        partitionray.append(1)
        classnameray.append(cat_2_name[dirname.replace(data_valid_path+'/','')])
        
for dirname, _, filenames in os.walk(data_test_path):
    for filename in filenames:
        fileray.append(os.path.join(dirname, filename))
        classray.append(int(dirname.replace(data_test_path+'/','')))
        partitionray.append(2)
        classnameray.append(cat_2_name[dirname.replace(data_test_path+'/','')])
        
df = pd.DataFrame(list(zip(fileray, partitionray, classray, classnameray)), 
               columns =['File', 'Partition', 'Class', 'Class Name']) 

df.set_index('File', inplace=True)
# Tamanho das partições
pd.value_counts(df['Partition'])
# Função para carregar e mudar o formato das imagens
def load_reshape_img(fname):
    x = cv2.imread(fname)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = cv2.resize(x, (224, 224))
    x = img_to_array(x)/255.

    return x

# Função para gerar as partições de dados
def generate_df(partition, num_samples):
    df_ = df[df['Partition'] == partition].sample(int(num_samples))
    
    x_ = np.array([load_reshape_img(fname) for fname in df_.index])
    y_ = df_['Class'].values

    return x_, y_
#Tamanho das amostras
train_sample = 6413 # 6413 max
valid_sample = 844 # 844 max
test_sample = 844 # 844 max

#Dados de Treino
x_train, y_train = generate_df(0, train_sample)

#Dados de Validação
x_valid, y_valid = generate_df(1, valid_sample)

#Dados de Teste
x_test, y_test = generate_df(2, test_sample)
# Gera quantidade X de imagens de exemplo
qtd_imagens = 8

fig = plt.figure(figsize=(20,10))
fig.subplots_adjust(wspace=0.2, hspace=0.4)

count = 0
for i in np.random.choice(range(0,len(x_train)-1), size=qtd_imagens):
    ax = fig.add_subplot(2, 4, count + 1, xticks=[], yticks=[], title=cat_2_name[str(y_train[i])])
    ax.imshow(x_train[i])
    count += 1
# Quantidade de classes
K = 212
print("Última camada:", K)
# Quantidade de Épocas
epocas = 8
print("Quantidade de epochs:", epocas)

i = Input(shape=x_train[0].shape)
x = Conv2D(32, (3, 3), activation='relu')(i)
x = MaxPooling2D(data_format="channels_last", pool_size=(2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D(data_format="channels_last", pool_size=(2, 2))(x)
#x = Conv2D(128, (3, 3), activation='relu')(x)
#x = MaxPooling2D(data_format="channels_last",pool_size=(2, 2))(x)
x = Flatten()(x)
#x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dense(K, activation='softmax')(x)

model = Model(i, x)

model.compile(optimizer=Adam(lr=0.0015),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

r = model.fit(x_train,
              y_train, 
              validation_data=(x_valid, y_valid),
              epochs=epocas)
plt.figure(figsize=(20,12))

print("\n\t\t\t\tEvolução Loss e Accuracy ao longo das Epochs (Qtd: " + str(epocas) + ")")

plt.subplot(2,2,1)
plt.plot(r.history['loss'], label='treino')
plt.plot(r.history['val_loss'], label='validacao')
plt.title("Loss")
plt.legend()

plt.subplot(2,2,2)
plt.plot(r.history['accuracy'], label='treino')
plt.plot(r.history['val_accuracy'], label='validacao')
plt.title("Accuracy")
plt.legend()

plt.show()
p_test = model.predict(x_test).argmax(axis=1)
total = len(list(p_test))
acertos = len(list(p_test)) - len(list(np.where(p_test != y_test))[0].tolist())
erros = len(list(np.where(p_test != y_test))[0].tolist())

print("Resultado da base de Teste", "\n\tTotal:\t", total, "\n\tAcertos:", acertos, "\n\tErros:\t", erros)
print("\nPorcentagem de acerto: " + str(round(acertos * 100 / total, 2)) + "%")
df_resultado = pd.DataFrame([p_test, y_test]).T
df_resultado.columns = ["Previsto", "Real"]
for key in cat_2_name.keys():
    df_resultado.replace(int(key), cat_2_name[key], inplace=True)
df_resultado["Acertos"] = df_resultado["Previsto"] == df_resultado["Real"]
df_resultado["Acertos"] = df_resultado["Acertos"].astype(str).replace("True", "1").replace("False", "0").astype(int)

df_resultado["Erros"] = df_resultado["Previsto"] != df_resultado["Real"]
df_resultado["Erros"] = df_resultado["Erros"].astype(str).replace("True", "1").replace("False", "0").astype(int)

qtd_acertos_erros = df_resultado.groupby(['Real'])\
    .agg({'Acertos':'sum', 'Erros':'sum', 'Previsto':'count'})\
    .rename(columns={'Acertos':'Qtd Acertos', 'Erros':'Qtd Erros', 'Previsto':'Qtd imagens'})\
    .reset_index()

qtd_acertos_erros.head()
qtd_acertos_erros["% Acertos"] = round((qtd_acertos_erros["Qtd Acertos"] / qtd_acertos_erros["Qtd imagens"]) * 100, 2)
qtd_acertos_erros["% Erros"] = round((qtd_acertos_erros["Qtd Erros"] / qtd_acertos_erros["Qtd imagens"]) * 100, 2)
qtd_acertos_erros.head()
distribuicao_acertos = qtd_acertos_erros.groupby(['% Acertos'])\
    .agg({'% Erros':'count'})\
    .rename(columns={'% Erros':'Quantidade'})\
    .reset_index()

plt.figure(figsize=(12,6))
ax = sns.barplot(
    data=distribuicao_acertos,
    x='% Acertos', 
    y='Quantidade',
    palette='Blues_d')

for index, row in distribuicao_acertos.iterrows():    
    ax.text(index, row['Quantidade'], row['Quantidade'].astype(int), color='black', va='bottom', rotation=0)

plt.title("Distribuição da porcentagem de acertos")
plt.xlabel("")
plt.tight_layout()
plt.show()
top10_acertos = qtd_acertos_erros.sort_values('% Acertos')\
    .nlargest(15, '% Acertos')\
    .reset_index()

plt.figure(figsize=(12,6))
ax = sns.barplot(
    data=top10_acertos,
    y='Real', 
    x='% Acertos',
    palette='Blues_d')

plt.title("As 15 moedas com maior porcentagem de acertos")
plt.xlabel("")
plt.tight_layout()
plt.show()
# Lista 6 casos em que houve acerto na previsão
qtd_imagens = 6

fig = plt.figure(figsize=(20,10))
fig.subplots_adjust(wspace=0.2, hspace=0.4)

_classerr_idx = np.where(p_test == y_test)[0]

acertos = np.random.choice(a = _classerr_idx, size = qtd_imagens, replace=False)
count = 0

print("\t\t\t\t\tCasos em que houve acerto na previsão")
    
for x in acertos:    
    ax = fig.add_subplot(2, 3, count + 1, xticks=[], yticks=[], 
                         title="Real: " + cat_2_name[str(y_test[x])] +
                                "\nPrevisto: " + cat_2_name[str(p_test[x])]
    )
    ax.imshow(x_test[x])
    count += 1
top10_erros = qtd_acertos_erros.sort_values('% Erros')\
    .nlargest(15, '% Erros')\
    .reset_index()

plt.figure(figsize=(12,6))
ax = sns.barplot(
    data=top10_erros,
    y='Real', 
    x='% Erros',
    palette='Blues_d')

plt.title("As 15 moedas com maior porcentagem de erros")
plt.xlabel("")
plt.tight_layout()
plt.show()
# Lista 6 casos em que houve falha na previsão
qtd_imagens = 6

fig = plt.figure(figsize=(20,10))
fig.subplots_adjust(wspace=0.2, hspace=0.4)

_classerr_idx = np.where(p_test != y_test)[0]

erros = np.random.choice(a = _classerr_idx, size = qtd_imagens, replace=False)
count = 0

print("\t\t\t\t\tCasos em que houve falha na previsão")
    
for x in erros:    
    ax = fig.add_subplot(2, 3, count + 1, xticks=[], yticks=[], 
                         title="Real: " + cat_2_name[str(y_test[x])] +
                                "\nPrevisto: " + cat_2_name[str(p_test[x])]
    )
    ax.imshow(x_test[x])
    count += 1