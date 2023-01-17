import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import IPython.display as ipd
import math
from pathlib import Path
import urllib
import scipy, matplotlib.pyplot as plt, sklearn, urllib, IPython.display as ipd
import librosa, librosa.display
from sklearn.metrics import accuracy_score

import os
csv = pd.read_csv('/kaggle/input/urbansound8k/UrbanSound8K.csv')
csv
#retorna o caminho completo de cada arquivo do CSV. 
def path_class(filename):
    #cria um filtro com a linha onde o arquivo está. 
    excerpt = csv[csv['slice_file_name'] == filename]
    
    #cria o path completo
    path_name = os.path.join('/kaggle/input/urbansound8k/', 'fold'+str(excerpt.fold.values[0]), filename)
    return path_name, excerpt['class'].values[0]
#retorna o audio e informações do mesmo. 
def to_dataset (df, fold=[]):
    # df = dataframe a ser percorrido.
    # fold = qual/quais folds ler. 
    audio = []
    audio_signals = []
    label= []
    labels=[]
    paths=[]   
    sampling_rate=[]
    librosa_sampling_rate = []
    
    # quando fold nulo, significa para ler tudo.         
    if fold != []:
        #filtra somentes os folds que foram enviados.  
        filter_fold = df.fold.isin(fold)
        df = df[filter_fold]

    ###Descomentar para compilar mais rapido.     
    #df = df.head(100)   

    #para cada fold, pega todos os arquivos de dentro.            
    for i in (df.fold.unique()):
        #filtra o cada fold em cada iteração
        filter_slice =  df['fold']==i
        dt_fold = df[filter_slice]
        
        #Iteração para ler os arquivos da pasta fold da vez.
        for p in dt_fold['slice_file_name']:
            # Librosa já converte os dois canais para um canal e normaliza os dados entre 1 e -1. 
            audio,librosa_sampling_rate  = librosa.load('/kaggle/input/urbansound8k/fold'+str(i)+'/' + p)
            audio_signals.append(audio)
            sampling_rate.append(librosa_sampling_rate)
            
            #busca as classes e paths.
            path, label = path_class(p)
            paths.append(path)
            labels.append(label)

    
    print('Reading...')    
    
    #audio = contém os arquivos de audio
    #paths = contém o caminho completo do arquivo.
    #labels = contem as classificações, 
    #librosa_sampling_rate = contém o sampling rate, que pode ser visto adiante. 
    return audio_signals,paths,labels,sampling_rate
#Cria os dados de treino.
#Descomentar para rodar mais rápidp
#Train, Train_Path, Train_Label,Train_S_Rate = to_dataset(csv,[1,2])
Train, Train_Path, Train_Label,Train_S_Rate = to_dataset(csv,[1,2,3,4,5,6,7,8,9])

print("Train set size: " + str(len(Train)))

#Cria os dados de teste
test, test_path, test_label,test_S_Rate  = to_dataset(csv,[10])
print("Test set size: " + str(len(test)))
# Para anlisar cada classe, primeiro é necessário extrair cada uma e criar um dataframe. 
dt_class = pd.DataFrame()
dt_fold = (csv[csv['fold']==1])
for i in (dt_fold.classID.unique()):
    dt_class = dt_class.append(dt_fold[dt_fold['classID']==i].head(1))    
dt_class
#Envia o dataframe com as 10 classes para recuperar os audios e demais informações. 
sample, sample_path, sample_label, sample_S_Rate = to_dataset(dt_class)
sample_label
#Leitura de um wav de exemplo
import struct
ipd.Audio(sample_path[0])
# encontrei arquivos de audio que nao tem dois canais. Embora a biblioteca deva resolver, 
# seria interessante fazer esse tratamento com wav.
# nesses mesmo samples, alguns arquivos nao podem ser ouvidos pelo ipd.audio. 
print('simple rate:')
print(sample_S_Rate)

# Librosa converte o sampling rate para 22050 no processo. 
#Plotando com Librosa (WAVEPLOT)
plt.figure(figsize=(15, 6))
for i, x in enumerate(sample):
    plt.subplot(4, 3, i+1)
    plt.title(label = sample_label[i])
    librosa.display.waveplot(x[:10000])
    #plt.ylim(-1, 1)

# É possivel notar algumas diferenças na onda. 
# Se 
#Plotando com Librosa (Fourier Transform)
#converte tempo em frequencia. 

plt.figure(figsize=(10, 5))
for i, x in enumerate(sample):
    plt.subplot(4, 3, i+1)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(sample[i])), ref=np.max)
    librosa.display.specshow(D, y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title(sample_label[i])
    
# Fica mais claro as diferenças. 
#exemplo de plot de cada classe. Usando Mel-Frequency Cepstral Coefficients (MFCC).
#Define a função vista em aula para captura de feature. Realiza a media. 
def extract_features(signal):
    return  librosa.feature.mfcc(y=signal, n_mfcc = 40)

# Cria um array com o MFCC de cada classe. 
dt_features = ([extract_features(x) for x in sample])
print(len(dt_features) )

# Cria um plot para cada classe.  
plt.figure(figsize=(8,8))
for i, x in enumerate(sample):
    plt.subplot(4, 3, i+1)
    librosa.display.specshow(dt_features[i], sr=sample_S_Rate[i], x_axis='time')
    plt.colorbar(format='%+2.0f dB');
    plt.title(sample_label[i])
#exemplo de plot de cada classe. Usando Mel Spectrogram
def extract_features(signal):
    S = librosa.feature.melspectrogram(signal, n_fft=2048,hop_length=512, n_mels=128)
    return librosa.power_to_db(S, ref=np.max)

# Cria um array com o MFCC de cada classe. 
dt_features = ([extract_features(x) for x in sample])
#print(len(dt_features) )

# Cria um plot para cada classe.  
plt.figure(figsize=(8,8))
for i, x in enumerate(sample):
    plt.subplot(4, 3, i+1)
    librosa.display.specshow(dt_features[i], hop_length=512, 
                         x_axis='time', y_axis='mel');
    plt.colorbar(format='%+2.0f dB');
    plt.title(sample_label[i])

#### !!!!! vou comentar esse trecho, pois na função a seguir usarei o vetor sem fazer a media. 

#Extraindo as features para efetivamente Treinar (redefinindo a função para treino)
#def extract_features(signal):                
#    return  (
#        #librosa.feature.zero_crossing_rate(signal).mean(),
#        librosa.feature.mfcc(signal)
#    )
#converte para array para poder criar um dataframe mais a frente. 
#Train_label_np = np.array(Train_Label)

#le a primeira feature
#data = ([extract_features(x) for x in Train])
#data


#alguns autores nao fazem a media, deixam a coluna feature com o vetor inteiro... qual a diferença?



#Train2 = Train+test#
#Train_Label2 = Train_Label + test_label
#remover ao implementar o cross validation
Train = Train+test
Train_Label = Train_Label+test_label
def extract_features(signal):
    mfccs = librosa.feature.mfcc(y=signal,  n_mfcc=40)
    mfccs_processed = np.mean(mfccs.T,axis=0)
     
    return mfccs_processed     

features = []
# Iterate through each sound file and extract the features 
for x in Train:
    features.append(extract_features(x))    

#normalizar features
#scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
#training_features = scaler.fit_transform(data)
#print(training_features.min(axis=0))
#print(training_features.max(axis=0))
#training_features

#Embora eu tenha feira o scalar, nao vou usar para nada. Vou treinar com a feature no valor integral. 
#converte em um pandas df
df = pd.DataFrame({'feature':features, 'label':Train_Label})
df
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics 
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
# Como os datasets estão como list, vou converter pra array para poder fazer o train_test_split. 
X = np.array(df.feature.tolist())
y = np.array(df.label.tolist())

# Como eu trouxe os labels da colunas de texto, vou fazer um encoder para mudar para numerico. 
le = LabelEncoder()
yy = to_categorical(le.fit_transform(y)) 



import pickle
d = dict(zip(le.classes_, le.transform(le.classes_)))

filename = 'dict'
pickle.dump(d, open(filename, 'wb'))
# separação do dataset
from sklearn.model_selection import train_test_split 

x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state = 42)

yy.shape[1]
# 
num_labels = yy.shape[1]

# Usar um sSequential do Keras simples. Os parametros são os mais comuns. 3 camadas. 

# O input será 40, devido ao n_mfcc=40 que usei antes. 
# Referência https://keras.io/getting-started/sequential-model-guide/

def build_model(input_shape=(40,)):
    model = Sequential()
    # Uma primeira camada
    model.add(Dense(256,  activation='relu'))
    
    #Dropout para reduziro o overfitting
    model.add(Dropout(0.5))
    
    #Camada intermediária para completar o modelo
    # Relu por ser a mais usada e com boa performance. 
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))    
    
    # como deve ter uma saida para cada classe, eu faço de acordo com o num_labels
    model.add(Dense(num_labels))
    
    #Softmax por ser mais de duas classes
    #https://missinglink.ai/guides/neural-network-concepts/7-types-neural-network-activation-functions-right/    
    model.add(Activation('softmax'))
    # Compila o modelo
    # Da lista de metricas, a unica que me parece fazer sentido é a acurácia 
    # Fonte: https://keras.io/metrics/
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    return model

#Cria o modelo
model = build_model()
#Compila o modelo. 
# Descomentar se precisar recompilar
#model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# Pré Avalia o modelo.  
score = model.evaluate(x_test, y_test, verbose=1)
accuracy = 100*score[1]
print("Pre-training accuracy: %.4f%%" % accuracy) 

# Mesmo baixa, o valor do treino final fica melhor. 
model.summary()
# Usando o early stopping para evitar perder tempo. 
# Usando checkpoint para nao perder o melhor resultado. 
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint 
mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=20)


from datetime import datetime 
num_epochs = 300
num_batch_size = 32
model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_split=0.33, verbose=1,callbacks=[es, mc])
# Usando a metrica de acurácia que é a mais comum. 
score = model.evaluate(x_train, y_train, verbose=0)
print("Training Accuracy: {0:.2%}".format(score[1]))
score = model.evaluate(x_test, y_test, verbose=0)
print("Testing Accuracy: {0:.2%}".format(score[1]))
score
#Export do modelo para usar no proximo algoritmo. (Além do best_model, claro) 
import pickle
# save the model to disk
filename = 'keras_audio_sequential.sav'
pickle.dump(model, open(filename, 'wb'))


x_train.shape
