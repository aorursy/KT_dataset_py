import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
import librosa
import os
from os import listdir
from scipy import signal
from scipy.io import wavfile
import random
from sklearn.metrics import confusion_matrix

np.random.seed(42)
!p7zip -d --keep 'train.7z'
def importar_arquivos(df):
  # df: dataframe. Exemplo de linha: 'silence/youtube_3491.wav'
  # X: lista, cada linha e um array de audio
  # y: nome da classe. Ex: "one"
  X = []
  y = []
  for k in range(len(df)):
    path = 'train/audio/' + df.iloc[k].to_string(index=False).replace(' ','')
    y.append(df.iloc[k].to_string(index=False).split('/')[0].replace(' ',''))
    X.append(carregar_audio(path)[0])
    
  return X,y
# Funcao para calculo do espectrograma utilizando LIBROSA
def convert_imagem_librosa(audio_array, output_shape, n_fft, window_size, hop_length, window, fs=1):
  # calculo do espectrograma

  spec = librosa.core.spectrum.stft(audio_array, 
                                  n_fft = n_fft,
                                  win_length = window_size,
                                  hop_length = hop_length,
                                  window=window)
  
  '''spec = librosa.feature.melspectrogram(audio_array, 
                                  n_fft = n_fft,
                                  hop_length = hop_length)
  '''
  spec_db = librosa.amplitude_to_db(abs(spec), ref=np.max)  # conversao para dB
  spec_db_norm = (spec_db - spec_db.min())/(spec_db.max()-spec_db.min())  # normalizando: valores entre 0 e 1
  spec_db_norm = (spec_db_norm - 0.5)*2
  
  im = Image.fromarray(np.uint8(cm.jet(spec_db_norm)*255))  # aplicando colormap
  rgb_image = im.convert('RGB') # convertento de RGBA para RGB
  #rgb_image = rgb_image[..., tf.newaxis]
  # retorna a imagem
  return (rgb_image.resize(output_shape))
# Funcao para calculo do espectrograma utilizando SCIPY
def convert_imagem(audio, fs, output_shape, n_fft, window_size, hop_length, window):
  f,t,spec = signal.spectrogram(x=audio,
                     fs=fs,
                     window=window,
                     noverlap=window_size - hop_length,
                     nfft=n_fft,
                     mode='complex')
  
  spec_db = 20*np.log10(0.000001 + abs(spec)/abs(spec).max()) # convertendo para dB
  spec_db_norm = (spec_db-spec_db.min())/(spec_db.max() - spec_db.min()) # normalizando entre 0 e 1
  spec_db_norm = (spec_db_norm - 0.5)*2

  im = Image.fromarray(np.uint8(cm.jet(spec_db_norm)*255)) # calculando o colormap
    
  rgb_image = im.convert('RGB')
  return (rgb_image.resize(output_shape))
# Funcao para importar audio utilizando SCIPY
def carregar_audio(path):
  fs, som = wavfile.read(path)
  return(np.array(som),fs)
# Funcao para garantir que os audios tenham 1s
def audio_reshape(audio):
  tamanho = len(audio[0])
 #tamanho_final = audio[1]
  tamanho_final = 16000
  if tamanho == tamanho_final:
    return audio
  elif tamanho < tamanho_final: # completar com zeros
    return (np.concatenate((audio[0], np.zeros(tamanho_final-tamanho))),tamanho_final)
  elif tamanho > tamanho_final: # remove os primeiros elementos
    return (audio[0][tamanho-tamanho_final:], tamanho_final)
# Funcao para dividir os audios '_background_noise_' em trechos de 1s e salvar na pasta especificada
def background_noise_split(audio, final_shape, nome, path):
  tamanho = len(audio[0])
  fs = audio[1] # frequencia de amostragem (amostras/s)
  for cont in range(0,tamanho//final_shape,1):
    audio_split = audio[0][cont*final_shape : (cont+1)*final_shape]
    #librosa.output.write_wav('gdrive/My Drive/MDC2020/speech/train/audio/silence/'+nome+'_'+str(cont)+'.wav',audio_split, fs)
    librosa.output.write_wav(path+nome+'_'+str(cont)+'.wav',audio_split, fs)
# Funcao para plotar os histogramas contendo a quantidade de classes dos dados
def plot_histogram(my_df, dsrd_column, x_size, y_size, class_dict):
  import seaborn as sns
  sns.set(font_scale=1.5)
  
  # Criando data frames que serão utilizados para plotar o número de classes
  class_dict_reduced = {0: 'down', 1: 'go', 2: 'left', 3: 'no', 4: 'off', 5: 'on', 6: 'right', 7: 'silence', 8: 'stop', 9: 'up', 10: 'yes', 11: 'unknown'}
  #class_dict_reduced = {'down':0, 'go':1, 'left':2, 'no':3, 'off':4, 'on':5, 'right':6, 'silence':7, 'stop':8, 'up':9, 'yes':10, 'unknown':11}

  plot_df = pd.DataFrame(my_df.str[0].to_numpy(), columns=['Class'])
  plot_df['Class_number'] = plot_df['Class'].apply(map_values, args = (class_dict,))
  plot_df['New Class'] = plot_df['Class_number'].apply(map_values, args = (class_dict_reduced,))
    
  f, ax = plt.subplots(figsize=(x_size, y_size))
  ax = sns.countplot(x=dsrd_column, data=plot_df, palette="GnBu_d")
  plt.xlabel('Classe')
  plt.ylabel('Quantidade de Amostras')
  sns.despine()
  plt.show()
# Funcao para mapear o dicionario
def map_values(row, values_dict):
    return values_dict[row]
# função para avaliar o modelo treinado a partir de metricas e da matriz de confusao
def avaliar_modelo(modelo, gerador, normalizado=True, num_classes=12):
  from sklearn import metrics
  import itertools

  y_true = np.array(gerador.get_labels())
  y_pred = np.argmax(modelo.predict(gerador), axis=1)

  # calculo da matriz de confusao
  cm = confusion_matrix(y_true, y_pred)
  cmn = cm / cm.astype(np.float).sum(axis=1, keepdims=True) # normalizada

  # calculo das metricas
  balanced_acc = np.trace(cmn) / num_classes # acuracia balanceada
  acc = metrics.accuracy_score(y_true, y_pred) # acuracia
  print('Balanced Accuracy: '+str(round(balanced_acc,4)))
  print('Accuracy: '+str(round(acc,4)))

  #plotando
  plt.style.use('default') 
  if normalizado == True:
    plt.figure(figsize=(12,6))
    plt.imshow(cmn, cmap=plt.cm.Blues)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    thresh = cmn.max() / 2.
    for i, j in itertools.product(range(cmn.shape[0]), range(cmn.shape[1])):
      plt.text(j, i, round(cmn[i, j],2),
              horizontalalignment="center",
              color="white" if cmn[i, j] > thresh else "black")
    plt.tight_layout
  else:
    plt.figure(figsize=(12,6))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, round(cm[i, j],2),
              horizontalalignment="center",
              color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout
# funcao para converter audio de stereo para mono
def stereo2mono(som):
  length = som.shape[0]
  mono = (som[0:length:1, 0]+som[0:length:1, 1])/2
  return(mono)
# Funcao para gerar um data frame, a ser convertido para .csv e submetido no Kaggle
def gerar_csv_kaggle(modelo, test_path, class_dict, threshold=0.4):
  import pandas as pd
  test_files = listdir(test_path)
  fname=[]
  label=[]
  i=0
  for amostra_teste in test_files:
    if i%1000 == 0:
      print(i)
    
    audio = carregar_audio(test_path + amostra_teste)
    img = convert_imagem(audio=audio[0], fs=audio[1], output_shape=output_shape, n_fft=n_fft, window_size=window_size, hop_length=hop_length, window=window)
    img = np.array(img)*(1./255)
    img_arr = img[np.newaxis,:]
    y_softmax = modelo.predict(img_arr) 
    y_pred = np.argmax(y_softmax, axis=1)
    if y_softmax[0][y_pred] < threshold:
      y_pred = 7 # prediz como silence
    for classe,num in class_dict.items():
      if y_pred == num:
        label.append(classe)
        break
    fname.append(amostra_teste)
    i+=1

  data = {'fname':fname, 'label':label}
  return pd.DataFrame(data)
# função para avaliar o modelo treinado a partir de metricas e da matriz de confusao
def avaliar_modelo_threshold(modelo, gerador, normalizado=True, num_classes=12, threshold=0.5):
  from sklearn import metrics
  import itertools

  y_true = np.array(gerador.get_labels())
  y_softmax = modelo.predict(gerador)
  y_pred = np.argmax(y_softmax, axis=1)

  for i in range(0,len(y_pred)):
   if y_softmax[i,][y_pred[i]] < threshold:
      y_pred[i] = 7 # prediz como silence

  # calculo da matriz de confusao
  cm = confusion_matrix(y_true, y_pred)
  cmn = cm / cm.astype(np.float).sum(axis=1, keepdims=True) # normalizada

  # calculo das metricas
  balanced_acc = np.trace(cmn) / num_classes # acuracia balanceada
  acc = metrics.accuracy_score(y_true, y_pred) # acuracia
  print('Balanced Accuracy: '+str(round(balanced_acc,4)))
  print('Accuracy: '+str(round(acc,4)))

  #plotando
  plt.style.use('default') 
  if normalizado == True:
    plt.figure(figsize=(12,6))
    plt.imshow(cmn, cmap=plt.cm.Blues)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    thresh = cmn.max() / 2.
    for i, j in itertools.product(range(cmn.shape[0]), range(cmn.shape[1])):
      plt.text(j, i, round(cmn[i, j],2),
              horizontalalignment="center",
              color="white" if cmn[i, j] > thresh else "black")
    plt.tight_layout
  else:
    plt.figure(figsize=(12,6))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, round(cm[i, j],2),
              horizontalalignment="center",
              color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout
def aumentar_audio(audio, time_stretch = (False, 1), pitch_shift = (False, 1), noise = (False, 0.01)):
  audio_aug = audio
  if time_stretch[0] == True: # aplica stretch no sinal
    audio_aug = librosa.effects.time_stretch(audio_aug.astype('float64'), time_stretch[1])

  if pitch_shift[0] == True: # aplica translacao no sinal
    audio_aug = librosa.effects.pitch_shift(audio_aug.astype('float64'), sr=16000, n_steps=pitch_shift[1])

  if noise[0] == True: # insere ruido aleatorio (segundo distribuicao normal) no sinal
    ampl = noise[1]*np.max(audio_aug)
    audio_aug = audio_aug.astype('float64') + ampl*np.random.normal(size = audio_aug.shape[0])

  return audio_aug
# background noise original
background_noise = ['white_noise.wav', 'running_tap.wav', 'pink_noise.wav', 'exercise_bike.wav', 'dude_miaowing.wav', 'doing_the_dishes.wav']
os.mkdir('train/audio/silence_youtube')
#o arquivo do youtube deve ser convertido para formato Mono antes do split

#youtube_audio, youtube_fs = carregar_audio('gdrive/My Drive/MDC2020/speech/train/audio/_background_noise_/youtube_sample.wav')
youtube_audio, youtube_fs = librosa.load('gdrive/My Drive/MDC2020/speech/train/audio/_background_noise_/youtube_sample.wav', sr=16000) # aqui a funcao do Librosa e utilizada pois, com ela, e possivel re-amostrar na taxa de 16000 samples/s, igual aos outros audios

background_noise_split(#audio = [stereo2mono(youtube_audio),youtube_fs],
                       audio=[youtube_audio,youtube_fs],
                       final_shape = youtube_fs,
                       nome = 'youtube',
                       path='train/audio/silence_youtube/')   

youtube_silence_files= listdir('train/audio/silence_youtube/')        
os.mkdir('train/audio/silence_original')
for noise in background_noise:
  background_noise_split(audio = librosa.load('train/audio/_background_noise_/'+noise, sr = 16000, mono = False),
                         final_shape = 16000,
                         nome = noise.split('.')[0],
                         path='train/audio/silence_original/')    
os.mkdir('train/audio/silence')
youtube_audio, youtube_fs = carregar_audio('gdrive/My Drive/MDC2020/speech/train/audio/_background_noise_/youtube_sample.wav')

background_noise_split(audio = [stereo2mono(youtube_audio),youtube_fs],
                       final_shape = youtube_fs,
                       nome = 'youtube',
                       path='train/audio/silence/')   

for noise in background_noise:
  background_noise_split(audio = librosa.load('train/audio/_background_noise_/'+noise, sr = 16000, mono = False),
                         final_shape = 16000,
                         nome = noise.split('.')[0],
                         path='train/audio/silence/') 
validation_list = pd.read_csv('gdrive/My Drive/MDC2020/speech/train/' + 'validation_list.txt', sep=' ', header=None)
train_list = pd.read_csv('gdrive/My Drive/MDC2020/speech/train/' + 'train_list.txt', sep=' ', header=None)
test_list = pd.read_csv('gdrive/My Drive/MDC2020/speech/train/' + 'testing_list.txt', sep=' ', header=None)
print('Validation: '+str(len(validation_list)))
print('Train: '+ str(len(train_list)))
print('Test: '+str(len(test_list)))
from sklearn.model_selection import train_test_split

youtube_silence_files = listdir('train/audio/silence_youtube/')
original_silence_files = listdir('train/audio/silence_original/')

youtube_silence_files = ['silence/' + s for s in youtube_silence_files]
original_silence_files = ['silence/' + s for s in original_silence_files]

# arquivos de silence sao divididos aleatoriamente

silence_train, silence_val_test = train_test_split(original_silence_files, train_size = 0.7, test_size = 0.3)
silence_test, silence_val = train_test_split(silence_val_test, train_size = 0.5, test_size = 0.5)

# desta forma, teremos, para o Silence, 70% no treino, 15% na validacao e 15% no teste

train_list_silence_original = pd.concat([train_list, pd.DataFrame(silence_train)], ignore_index=True)
validation_list_silence_original = pd.concat([validation_list, pd.DataFrame(silence_val)], ignore_index=True)
test_list_silence_original = pd.concat([test_list, pd.DataFrame(silence_test)], ignore_index=True)
from sklearn.model_selection import train_test_split

youtube_silence_files = listdir('train/audio/silence_youtube/')
original_silence_files = listdir('train/audio/silence_original/')

youtube_silence_files = ['silence/' + s for s in youtube_silence_files]
original_silence_files = ['silence/' + s for s in original_silence_files]

# arquivos do YouTube -> conjunto de treino
# demais arquivos -> split em validação e teste

silence_train = youtube_silence_files
silence_val, silence_test = train_test_split(original_silence_files, train_size = 0.5, test_size = 0.5)

train_list = pd.concat([train_list, pd.DataFrame(silence_train)], ignore_index=True)
validation_list = pd.concat([validation_list, pd.DataFrame(silence_val)], ignore_index=True)
test_list = pd.concat([test_list, pd.DataFrame(silence_test)], ignore_index=True)
print('Validation (com dados do YouTube): ' + str(len(validation_list)))
print('Train (com dados do YouTube): ' + str(len(train_list)))
print('Test (com dados do YouTube): ' + str(len(test_list)))
print('Validation (silence original): ' + str(len(validation_list_silence_original)))
print('Train (silence original): ' + str(len(train_list_silence_original)))
print('Test (silence original): ' + str(len(test_list_silence_original)))
X_train, y_train = importar_arquivos(train_list)
X_val, y_val = importar_arquivos(validation_list)
X_test, y_test = importar_arquivos(test_list)

X_train_original, y_train_original = importar_arquivos(train_list_silence_original)
X_val_original, y_val_original = importar_arquivos(validation_list_silence_original)
X_test_original, y_test_original = importar_arquivos(test_list_silence_original)
validation_label_nome = validation_list[0].str.split('/')
train_label_nome = train_list[0].str.split('/')
test_label_nome = test_list[0].str.split('/')
validation_label_nome_original = validation_list_silence_original[0].str.split('/')
train_label_nome_original = train_list_silence_original[0].str.split('/')
test_label_nome_original = test_list_silence_original[0].str.split('/')
# Diminuindo para 12 classes (enunciado Kaggle)
kaggle_list = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "silence"]

# Dicionario
class_list = ['bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'four', 'go', 'happy', 'house', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'silence', 'six', 'stop', 'three',
              'tree', 'two', 'up', 'wow', 'yes', 'zero']
num = 0
class_dict = {}
for classe in class_list:
  if classe != '_background_noise_':
    if classe in kaggle_list:
      class_dict[classe] = num
      num+=1
    else:
      class_dict[classe] = 11 # others outside of the list are unknown
      
class_dict
plot_histogram(my_df=train_label_nome, dsrd_column='Class', x_size=30, y_size=8, class_dict=class_dict)
plot_histogram(my_df=train_label_nome, dsrd_column='New Class', x_size=20, y_size=10, class_dict=class_dict)
from sklearn.utils import class_weight

class_dict_reduced = {0: 'down', 1: 'go', 2: 'left', 3: 'no', 4: 'off', 5: 'on', 6: 'right', 7: 'stop', 8: 'up', 9: 'yes', 10: 'silence', 11: 'unknown'}

train_df = pd.DataFrame(train_label_nome.str[0].to_numpy(), columns=['Class'])
train_df['Class_number'] = train_df['Class'].apply(map_values, args = (class_dict,))
train_df['New Class'] = train_df['Class_number'].apply(map_values, args = (class_dict_reduced,))

arr_class_weights = class_weight.compute_class_weight(
                    'balanced',
                    np.unique(train_df['Class_number']),
                    np.array(train_df['Class_number']))

class_weights = dict(enumerate(arr_class_weights))
print(class_weights)
fs = 16000
window_size = 256
n_fft = 512
hop_length = 32
window = np.hanning(window_size)
output_shape=(128,128) # dimensao da imagem gerada
n_channel = 3
(L,W) = output_shape
num_classes = 12
path_example = 'gdrive/My Drive/MDC2020/speech/train/audio/'+validation_list[0].str.split('/')[0][0]+'/'+validation_list[0].str.split('/')[0][1]
audio_exemplo = carregar_audio(path_example)
img_exemplo = convert_imagem(audio=audio_exemplo[0], fs=audio_exemplo[1], output_shape=output_shape, n_fft=n_fft, window_size=window_size, hop_length=hop_length, window=window)
plt.style.use('default') 
plt.imshow(img_exemplo)
plt.plot(audio_exemplo[0])
audio_stretch = aumentar_audio(audio_exemplo[0],
                             time_stretch = (True, 3))

img_stretch1 = convert_imagem(audio = aumentar_audio(audio_exemplo[0], time_stretch = (True, 2)), 
                             fs=16000, output_shape=output_shape, n_fft=n_fft, window_size=window_size, hop_length=hop_length, window=window)

img_stretch2 = convert_imagem(audio = aumentar_audio(audio_exemplo[0], time_stretch = (True, 0.01)), 
                             fs=16000, output_shape=output_shape, n_fft=n_fft, window_size=window_size, hop_length=hop_length, window=window)

plt.figure(figsize=(20,4))
plt.subplot(1,3,1)
plt.imshow(img_exemplo)
plt.subplot(1,3,2)
plt.imshow(img_stretch1)
plt.subplot(1,3,3)
plt.imshow(img_stretch2)
plt.tight_layout
audio_noise = aumentar_audio(audio_exemplo[0], noise = (True, 0.01))

img_noise1 = convert_imagem(audio=aumentar_audio(audio_exemplo[0], noise = (True, 0.01)),
                           fs=16000, output_shape=output_shape, n_fft=n_fft, window_size=window_size, hop_length=hop_length, window=window)

img_noise2 = convert_imagem(audio=aumentar_audio(audio_exemplo[0], noise = (True, 0.05)),
                           fs=16000, output_shape=output_shape, n_fft=n_fft, window_size=window_size, hop_length=hop_length, window=window)

plt.figure(figsize=(20,4))
plt.subplot(1,3,1)
plt.imshow(img_exemplo)
plt.subplot(1,3,2)
plt.imshow(img_noise1)
plt.subplot(1,3,3)
plt.imshow(img_noise2)
plt.tight_layout
img_pitch1 = convert_imagem(audio=aumentar_audio(audio_exemplo[0], pitch_shift = (True, 3)),
                           fs=16000, output_shape=output_shape, n_fft=n_fft, window_size=window_size, hop_length=hop_length, window=window)

img_pitch2 = convert_imagem(audio=aumentar_audio(audio_exemplo[0], pitch_shift = (True, -3)),
                           fs=16000, output_shape=output_shape, n_fft=n_fft, window_size=window_size, hop_length=hop_length, window=window)

plt.figure(figsize=(20,4))
plt.subplot(1,3,1)
plt.imshow(img_exemplo)
plt.subplot(1,3,2)
plt.imshow(img_pitch1)
plt.subplot(1,3,3)
plt.imshow(img_pitch2)
plt.tight_layout
ex_sil_orig1 = convert_imagem(carregar_audio('train/audio/silence_original/dude_miaowing_1.wav')[0], fs=fs, output_shape=output_shape, n_fft=n_fft, window_size=window_size, hop_length=hop_length, window=window)
ex_sil_orig2 = convert_imagem(carregar_audio('train/audio/silence_original/doing_the_dishes_1.wav')[0], fs=fs, output_shape=output_shape, n_fft=n_fft, window_size=window_size, hop_length=hop_length, window=window)
ex_sil_orig3 = convert_imagem(carregar_audio('train/audio/silence_original/running_tap_1.wav')[0], fs=fs, output_shape=output_shape, n_fft=n_fft, window_size=window_size, hop_length=hop_length, window=window)

ex_sil_yt = convert_imagem(carregar_audio('train/audio/silence_youtube/youtube_100.wav')[0], fs=fs, output_shape=output_shape, n_fft=n_fft, window_size=window_size, hop_length=hop_length, window=window)
plt.figure(figsize=(20,4))
plt.subplot(1,4,1)
plt.imshow(ex_sil_orig1)
plt.subplot(1,4,2)
plt.imshow(ex_sil_orig2)
plt.subplot(1,4,3)
plt.imshow(ex_sil_orig3)
plt.subplot(1,4,4)
plt.imshow(ex_sil_yt)
plt.tight_layout
class Gerador(tf.keras.utils.Sequence): 
  def __init__(self, output_shape, X, y, class_dict, path, fs, batch_size, n_channels, n_classes, shuffle, n_fft, window_size, hop_length, window, augment):
      # inicializacao
      self.fs = fs
      self.X = X
      self.y = y
      self.output_shape = output_shape
      self.path = path
      self.batch_size = batch_size
      self.n_channels = n_channels
      self.n_classes = n_classes
      self.shuffle = shuffle
      self.n_fft = n_fft
      self.window_size = window_size
      self.hop_length = hop_length
      self.window = window
      self.class_dict = class_dict

      # Variaveis de aumentacao:
      self.augment = augment # booleano, True: aplica aumentacao

      self.on_epoch_end()
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  def get_labels(self):
    return np.vectorize(class_dict.get)(self.y[0 : self.batch_size*(len(self.y)//self.batch_size)])
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  def on_epoch_end(self):
    # criacao dos índices da proxima epoca (com ou sem schuffle)
    self.indexes = np.arange(len(self.y))
    
    if self.shuffle == True:
      np.random.shuffle(self.indexes)
    
    if self.augment == True:
      import random
      self.pitch_augm = random.randint(-300,300)/100 # numero aleatorio entre -3 e 3
      self.time_augm = random.randint(1,200)/100 # numero aleatorio entre 0 e 1
      self.noise_augm = random.randint(1,2)/100 # numero aleatorio entre 0 e 0.02
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  def __data_generation(self, X_batch, y_batch):
    # metodo para geracao de batches
    X = np.empty((self.batch_size, *self.output_shape, self.n_channels))
    y = np.empty((self.batch_size), dtype=int)   # labels

    # gerando os dados
    for i in range(len(y_batch)):
      audio_array = X_batch[i]

      if self.augment == True:
        audio_array = aumentar_audio(audio = audio_array, time_stretch = (True, self.time_augm), pitch_shift = (False, self.pitch_augm), noise = (True, self.noise_augm)) # aplicar aumentacao no array
      
      X[i,] = convert_imagem(audio=audio_array, fs=self.fs,output_shape=self.output_shape, n_fft=self.n_fft, window_size=self.window_size, hop_length=self.hop_length, window=self.window)
      y[i] = self.class_dict[y_batch[i]]

    # utilizando to_categorical para fazer one-hot-encoding
    return X*(1./255), tf.keras.utils.to_categorical(y, num_classes=self.n_classes)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  def __len__(self):
    # numero de batches por epoca
    return int(np.floor(len(self.y)/self.batch_size))
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  def __getitem__(self, index):
    # metodo para gerar um batch
    # gerando indices do batch. Exemplo: [1,2,3,4,5], batch = 2 -> [1,2] em index=0, [3,4] em index=1, etc
    indexes = self.indexes[index*self.batch_size : (index+1)*self.batch_size] # gerando os indices
    # gerando lista de label-nome do batch
    X_batch = [self.X[k] for k in indexes]
    y_batch = [self.y[k] for k in indexes]

    # gerando os  dados
    X,y = self.__data_generation(X_batch, y_batch)
    
    return X,y
train_generator_1 = Gerador(X=X_train_original,
                            y=y_train_original,
                            batch_size=32,
                            output_shape=output_shape,
                            n_fft=n_fft,
                            window=window,
                            window_size=window_size,
                            hop_length=hop_length,
                            fs=fs,
                            path = 'train/audio/',
                            class_dict=class_dict,
                            n_classes=num_classes,
                            n_channels=n_channel,
                            shuffle = True,
                            augment = False)
validation_generator_1 = Gerador(X=X_val_original,
                                 y=y_val_original,
                                 batch_size = 32,
                                 output_shape = output_shape,
                                 n_fft = n_fft,
                                 window = window,
                                 window_size = window_size,
                                 hop_length = hop_length,
                                 fs = fs,
                                 path= 'train/audio/',
                                 class_dict = class_dict,
                                 n_channels = n_channel,
                                 n_classes = num_classes, 
                                 shuffle = False,
                                 augment = False)
test_generator_1 = Gerador(X=X_test_original,
                           y=y_test_original,
                           batch_size = 32,
                           output_shape = output_shape,
                           n_fft = n_fft,
                           window = window,
                           window_size = window_size,
                           hop_length = hop_length,
                           fs = fs,
                           path= 'train/audio/',
                           class_dict = class_dict,
                           n_channels = n_channel,
                           n_classes = num_classes, 
                           shuffle = False,
                           augment = False)
train_generator_2 = Gerador(X=X_train,
                            y=y_train,
                            batch_size=32,
                            output_shape=output_shape,
                            n_fft=n_fft,
                            window=window,
                            window_size=window_size,
                            hop_length=hop_length,
                            fs=fs,
                            path = 'train/audio/',
                            class_dict=class_dict,
                            n_classes=num_classes,
                            n_channels=n_channel,
                            shuffle = True,
                            augment = False)
validation_generator_2 = Gerador(X=X_val,
                                 y=y_val,
                                 batch_size = 32,
                                 output_shape = output_shape,
                                 n_fft = n_fft,
                                 window = window,
                                 window_size = window_size,
                                 hop_length = hop_length,
                                 fs = fs,
                                 path= 'train/audio/',
                                 class_dict = class_dict,
                                 n_channels = n_channel,
                                 n_classes = num_classes, 
                                 shuffle = False,
                                 augment = False)
test_generator_2 = Gerador(X=X_test,
                           y=y_test,
                           batch_size = 32,
                           output_shape = output_shape,
                           n_fft = n_fft,
                           window = window,
                           window_size = window_size,
                           hop_length = hop_length,
                           fs = fs,
                           path= 'train/audio/',
                           class_dict = class_dict,
                           n_channels = n_channel,
                           n_classes = num_classes, 
                           shuffle = False,
                           augment = False)
train_generator_3 = Gerador(X=X_train,
                            y=y_train,
                            batch_size=32,
                            output_shape=output_shape,
                            n_fft=n_fft,
                            window=window,
                            window_size=window_size,
                            hop_length=hop_length,
                            fs=fs,
                            path = 'train/audio/',
                            class_dict=class_dict,
                            n_classes=num_classes,
                            n_channels=n_channel,
                            shuffle = True,
                            augment = True)
validation_generator_3 = Gerador(X=X_val,
                                 y=y_val,
                                 batch_size = 32,
                                 output_shape = output_shape,
                                 n_fft = n_fft,
                                 window = window,
                                 window_size = window_size,
                                 hop_length = hop_length,
                                 fs = fs,
                                 path= 'train/audio/',
                                 class_dict = class_dict,
                                 n_channels = n_channel,
                                 n_classes = num_classes, 
                                 shuffle = False,
                                 augment = False)
test_generator_3 = Gerador(X=X_test,
                           y=y_test,
                           batch_size = 32,
                           output_shape = output_shape,
                           n_fft = n_fft,
                           window = window,
                           window_size = window_size,
                           hop_length = hop_length,
                           fs = fs,
                           path= 'train/audio/',
                           class_dict = class_dict,
                           n_channels = n_channel,
                           n_classes = num_classes, 
                           shuffle = False,
                           augment = False)
from tensorflow.keras import layers

baseline1 = tf.keras.Sequential()

# Conv 2D e Max Pooling
baseline1.add(layers.Conv2D(10, 3, padding='valid', input_shape=(L,W,n_channel)))
baseline1.add(layers.BatchNormalization(momentum=0.8))
baseline1.add(layers.ReLU())
baseline1.add(layers.MaxPooling2D(pool_size=(2,2)))

# Conv 2D e Max Pooling
baseline1.add(layers.Conv2D(10, 3, padding='valid'))
baseline1.add(layers.BatchNormalization(momentum=0.8))
baseline1.add(layers.ReLU())
baseline1.add(layers.MaxPooling2D(pool_size=(2,2)))

# Conv 2D e Max Pooling
baseline1.add(layers.Conv2D(10, 3, padding='valid'))
baseline1.add(layers.BatchNormalization(momentum=0.8))
baseline1.add(layers.ReLU())
baseline1.add(layers.MaxPooling2D(pool_size=(2,2)))

# Flatten
baseline1.add(layers.Flatten())

# Dense Final
baseline1.add(layers.Dense(num_classes, activation='softmax'))

baseline1.summary()
'''checkpoints = tf.keras.callbacks.ModelCheckpoint(filepath = 'gdrive/My Drive/MDC2020/speech/train/callback_baseline1/test_baseline1-{epoch:02d}-{val_loss:.2f}.hdf5',
                                                 monitor = 'val_loss',
                                                 save_best_only=False, 
                                                 save_weights_only=False,
                                                 mode = 'min')
training_logs = tf.keras.callbacks.CSVLogger('gdrive/My Drive/MDC2020/speech/train/callback_baseline1/test_baseline1_log.csv', separator=",", append=False)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 5)

# Definindo o compilador
baseline1.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics='accuracy')

# Treinamento
baseline1_history = baseline1.fit(train_generator_1,             
                                epochs=20,
                                validation_data=validation_generator_1,
                                class_weight=class_weights,
                                workers=8,
                                callbacks = [checkpoints, training_logs, early_stopping])'''
# Continuando o treinamento
'''baseline1_cont = tf.keras.models.load_model('gdrive/My Drive/MDC2020/speech/train/callback_baseline1/baseline1-13-0.64.hdf5')

checkpoints = tf.keras.callbacks.ModelCheckpoint(filepath = 'gdrive/My Drive/MDC2020/speech/train/callback_baseline1/baseline1-{epoch:02d}-{val_loss:.2f}.hdf5',
                                                 monitor = 'val_loss',
                                                 save_best_only=True, 
                                                 save_weights_only=False,
                                                 mode = 'min')
training_logs_cont = tf.keras.callbacks.CSVLogger('gdrive/My Drive/MDC2020/speech/train/callback_baseline1/callback_baseline1_cont_log.csv', separator=",", append=False)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 5)

baseline1_cont_hist = baseline1_cont.fit(train_generator_1,             
                                epochs=20,
                                validation_data=validation_generator_1,
                                initial_epoch=13,
                                workers=8,
                                class_weight=class_weights,
                                callbacks = [checkpoints, training_logs_cont, early_stopping])'''
'''from tensorflow.keras import layers

baseline2 = tf.keras.Sequential()

# Conv 2D e Max Pooling
baseline2.add(layers.Conv2D(10, 3, padding='valid', input_shape=(L,W,n_channel)))
baseline2.add(layers.BatchNormalization(momentum=0.8))
baseline2.add(layers.ReLU())
baseline2.add(layers.MaxPooling2D(pool_size=(2,2)))

# Conv 2D e Max Pooling
baseline2.add(layers.Conv2D(10, 3, padding='valid'))
baseline2.add(layers.BatchNormalization(momentum=0.8))
baseline2.add(layers.ReLU())
baseline2.add(layers.MaxPooling2D(pool_size=(2,2)))

# Conv 2D e Max Pooling
baseline2.add(layers.Conv2D(10, 3, padding='valid'))
baseline2.add(layers.BatchNormalization(momentum=0.8))
baseline2.add(layers.ReLU())
baseline2.add(layers.MaxPooling2D(pool_size=(2,2)))

# Flatten
baseline2.add(layers.Flatten())

# Dense Final
baseline2.add(layers.Dense(num_classes, activation='softmax'))

baseline2.summary()'''
'''checkpoints = tf.keras.callbacks.ModelCheckpoint(filepath = 'gdrive/My Drive/MDC2020/speech/train/callback_baseline2_mais_silencio/ger_baseline_mais_silencio-{epoch:02d}-{val_loss:.2f}.hdf5',
                                                 monitor = 'val_loss',
                                                 save_best_only=False, 
                                                 save_weights_only=False,
                                                 mode = 'min')
training_logs = tf.keras.callbacks.CSVLogger('gdrive/My Drive/MDC2020/speech/train/callback_baseline2_mais_silencio/ger_baseline_mais_silencio_log.csv', separator=",", append=False)

# Definindo o compilador
baseline2.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics='accuracy')

# Treinamento
baseline2_history = baseline2.fit(train_generator_2,             
                                epochs=20,
                                validation_data=validation_generator_2,
                                workers=8,
                                class_weight=class_weights,
                                callbacks = [checkpoints, training_logs])'''
# Continuando o treinamento
baseline2_cont = tf.keras.models.load_model('gdrive/My Drive/MDC2020/speech/train/callback_baseline2_mais_silencio/ger_baseline_mais_silencio-07-1.26.hdf5')

checkpoints = tf.keras.callbacks.ModelCheckpoint(filepath = 'gdrive/My Drive/MDC2020/speech/train/callback_baseline2_mais_silencio/ger_baseline2_mais_silencio-{epoch:02d}-{val_loss:.2f}.hdf5',
                                                 monitor = 'val_loss',
                                                 save_best_only=True, 
                                                 save_weights_only=False,
                                                 mode = 'min')
training_logs_cont = tf.keras.callbacks.CSVLogger('gdrive/My Drive/MDC2020/speech/train/callback_baseline2_mais_silencio/ger_baseline2_mais_silencio_cont_log.csv', separator=",", append=False)
baseline2_cont_cont = baseline2_cont.fit(train_generator_2,             
                                epochs=20,
                                validation_data=validation_generator_2,
                                initial_epoch=7,
                                workers=8,
                                class_weight=class_weights,
                                callbacks = [checkpoints, training_logs_cont])
from tensorflow.keras import layers

baseline3 = tf.keras.Sequential()

# Conv 2D e Max Pooling
baseline3.add(layers.Conv2D(10, 3, padding='valid', input_shape=(L,W,n_channel)))
baseline3.add(layers.BatchNormalization(momentum=0.8))
baseline3.add(layers.ReLU())
baseline3.add(layers.MaxPooling2D(pool_size=(2,2)))

# Conv 2D e Max Pooling
baseline3.add(layers.Conv2D(10, 3, padding='valid'))
baseline3.add(layers.BatchNormalization(momentum=0.8))
baseline3.add(layers.ReLU())
baseline3.add(layers.MaxPooling2D(pool_size=(2,2)))

# Conv 2D e Max Pooling
baseline3.add(layers.Conv2D(10, 3, padding='valid'))
baseline3.add(layers.BatchNormalization(momentum=0.8))
baseline3.add(layers.ReLU())
baseline3.add(layers.MaxPooling2D(pool_size=(2,2)))

# Flatten
baseline3.add(layers.Flatten())

# Dense Final
baseline3.add(layers.Dense(num_classes, activation='softmax'))

baseline3.summary()
checkpoints = tf.keras.callbacks.ModelCheckpoint(filepath = 'gdrive/My Drive/MDC2020/speech/train/callback_baseline3_mais_silencio/gen_baseline3_mais_silencio-{epoch:02d}-{val_loss:.2f}.hdf5',
                                                 monitor = 'val_loss',
                                                 save_best_only=False, 
                                                 save_weights_only=False,
                                                 mode = 'min')
training_logs = tf.keras.callbacks.CSVLogger('gdrive/My Drive/MDC2020/speech/train/callback_baseline3_mais_silencio/gen_baseline3_mais_silencio_log.csv', separator=",", append=False)

# Definindo o compilador
baseline3.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics='accuracy')

# Treinamento
baseline3_history = baseline3.fit(train_generator_3,             
                                epochs=40,
                                validation_data=validation_generator_3,
                                class_weight=class_weights,
                                workers=8,
                                callbacks = [checkpoints, training_logs])
# Continuando o treinamento
baseline3_cont = tf.keras.models.load_model('gdrive/My Drive/MDC2020/speech/train/callback_baseline3_mais_silencio/gen_baseline3_mais_silencio-09-1.55.hdf5')

checkpoints = tf.keras.callbacks.ModelCheckpoint(filepath = 'gdrive/My Drive/MDC2020/speech/train/callback_baseline3_mais_silencio/gen_baseline_mais_silencio-{epoch:02d}-{val_loss:.2f}.hdf5',
                                                 monitor = 'val_loss',
                                                 save_best_only=False, 
                                                 save_weights_only=False,
                                                 mode = 'min')
training_logs_cont = tf.keras.callbacks.CSVLogger('gdrive/My Drive/MDC2020/speech/train/callback_baseline3_mais_silencio/gen_baseline3_mais_silencio_cont_log.csv', separator=",", append=False)
baseline3_history_cont = baseline3_cont.fit(train_generator_3,             
                                epochs=20,
                                validation_data=validation_generator_3,
                                initial_epoch=9,
                                workers=8,
                                class_weight=class_weights,
                                callbacks = [checkpoints, training_logs_cont])
# Continuando o treinamento
baseline3_cont = tf.keras.models.load_model('gdrive/My Drive/MDC2020/speech/train/callback_baseline3_mais_silencio/gen_baseline_mais_silencio-10-1.36.hdf5')

checkpoints = tf.keras.callbacks.ModelCheckpoint(filepath = 'gdrive/My Drive/MDC2020/speech/train/callback_baseline3_mais_silencio/gen_baseline_mais_silencio-{epoch:02d}-{val_loss:.2f}.hdf5',
                                                 monitor = 'val_loss',
                                                 save_best_only=False, 
                                                 save_weights_only=False,
                                                 mode = 'min')
training_logs_cont = tf.keras.callbacks.CSVLogger('gdrive/My Drive/MDC2020/speech/train/callback_baseline3_mais_silencio/gen_baseline3_mais_silencio_cont2_log.csv', separator=",", append=False)
baseline3_history_cont = baseline3_cont.fit(train_generator_3,             
                                epochs=20,
                                validation_data=validation_generator_3,
                                initial_epoch=10,
                                workers=8,
                                class_weight=class_weights,
                                callbacks = [checkpoints, training_logs_cont])
baseline_1_log = pd.read_csv('gdrive/My Drive/MDC2020/speech/train/callback_baseline1/baseline1_log.csv', sep=',')    
plt.style.use('default') 
plt.figure(figsize=(6,4))
plt.plot(baseline_1_log['epoch'], baseline_1_log['val_loss'])
plt.plot(baseline_1_log['epoch'], baseline_1_log['loss'])
plt.legend(['Validation', 'Train'])
plt.ylabel('Loss')
plt.xlabel('Epoch')   
baseline1_best_model = tf.keras.models.load_model('gdrive/My Drive/MDC2020/speech/train/callback_baseline1/baseline1-08-0.61.hdf5')
avaliar_modelo(baseline1_best_model, test_generator_1, normalizado=True, num_classes=12)
baseline_2_log = pd.concat([pd.read_csv('gdrive/My Drive/MDC2020/speech/train/callback_baseline2_mais_silencio/ger_baseline_mais_silencio_log.csv', sep=','),
                            pd.read_csv('gdrive/My Drive/MDC2020/speech/train/callback_baseline2_mais_silencio/ger_baseline2_mais_silencio_cont_log.csv', sep=',')])


plt.style.use('default') 
plt.figure(figsize=(6,4))
plt.plot(baseline_2_log['epoch'], baseline_2_log['val_loss'])
plt.plot(baseline_2_log['epoch'], baseline_2_log['loss'])
plt.legend(['Validation', 'Train'])
plt.ylabel('Loss')
plt.xlabel('Epoch')   
baseline2_best_model = tf.keras.models.load_model('gdrive/My Drive/MDC2020/speech/train/callback_baseline2_mais_silencio/baseline_mais_silencio-02-1.01.hdf5')
avaliar_modelo(baseline2_best_model, test_generator_2, normalizado=True, num_classes=12)
baseline_3_log = pd.concat([pd.read_csv('gdrive/My Drive/MDC2020/speech/train/callback_baseline3_mais_silencio/gen_baseline3_mais_silencio_log.csv', sep=','),
                            pd.read_csv('gdrive/My Drive/MDC2020/speech/train/callback_baseline3_mais_silencio/gen_baseline3_mais_silencio_cont_log.csv', sep=','),
                            pd.read_csv('gdrive/My Drive/MDC2020/speech/train/callback_baseline3_mais_silencio/gen_baseline3_mais_silencio_cont2_log.csv', sep=',')])
                            
plt.figure(figsize=(6,4))
plt.plot(baseline_3_log['epoch'], baseline_3_log['val_loss'])
plt.plot(baseline_3_log['epoch'], baseline_3_log['loss'])
plt.legend(['Validation', 'Train'])
plt.ylabel('Loss')
plt.xlabel('Epoch')   
baseline3_best_model = tf.keras.models.load_model('gdrive/My Drive/MDC2020/speech/train/callback_baseline3_mais_silencio/gen_baseline_mais_silencio-13-1.34.hdf5')
avaliar_modelo(baseline3_best_model, test_generator_3, normalizado=True, num_classes=12)
y_true = np.array(validation_generator_3.get_labels())
y_softmax = baseline3_best_model.predict(validation_generator_3)
y_pred = np.argmax(y_softmax, axis=1)
index = np.full((12, len(y_true)), False)
for i in range(0,12):
  index[i,:] = y_true == i

class_prob = np.zeros((num_classes, num_classes))
for i in range(0,12):
  mtrx = np.asmatrix(y_softmax[index[i],])
  class_prob[i,] = mtrx.mean(axis=0)

plt.figure(figsize=(16,10))
for i in range(0,12):
  plt.subplot(3,4,i+1)
  plt.plot(class_prob[i,])
  plt.gca().set_title('True: '+str(i) + ', Max: '+str(round(max(class_prob[i,]),2)))
  plt.ylabel('Softmax Probability')
  plt.xlabel('Class')
  plt.tight_layout()
avaliar_modelo_threshold(baseline3_best_model, test_generator_3, normalizado=True, num_classes=12, threshold = 0.4)
baseline2_best_model = tf.keras.models.load_model('gdrive/My Drive/MDC2020/speech/train/callback_baseline2_mais_silencio/baseline_mais_silencio-02-1.01.hdf5')
avaliar_modelo_threshold(baseline2_best_model, test_generator_2, normalizado=True, num_classes=12, threshold = 0.4)
'''transfer_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(L, W, 3))'''
'''# Congelando
for layer in transfer_model.layers:
    layer.trainable = False

# Acoplando o modelo VGG a uma camada convolucional (2 Camadas Dense)
full_model = tf.keras.Sequential([
  transfer_model,

  tf.keras.layers.GlobalAveragePooling2D(),

  tf.keras.layers.Dense(100, activation='relu') 
  
  tf.keras.layers.Dropout(0.2)

  tf.keras.layers.Dense(50, activation='relu') 

  tf.keras.layers.Dropout(0.2)

  tf.keras.layers.Dense(num_classes, activation='softmax')
])

full_model.summary()'''
# Definindo o Call Back
checkpoints = tf.keras.callbacks.ModelCheckpoint(filepath = 'gdrive/My Drive/MDC2020/speech/train/callback_vgg/ger_vgg2_dense-{epoch:02d}-{val_loss:.2f}.hdf5',
                                                 monitor = 'val_loss',
                                                 save_best_only=True, 
                                                 save_weights_only=False,
                                                 mode = 'min')
training_logs = tf.keras.callbacks.CSVLogger('gdrive/My Drive/MDC2020/speech/train/callback_vgg/ger_vgg2_dense_log.csv', separator=",", append=False)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 9)

# Definindo o compilador
full_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics='accuracy')

# Treinamento
full_model_history = full_model.fit(train_generator_3,             
                                epochs=40,
                                validation_data=validation_generator_3,
                                workers=8,
                                class_weight=class_weights,
                                callbacks = [checkpoints, training_logs, early_stopping])
# plotando curvas de Loss
vgg_results = pd.read_csv('gdrive/My Drive/MDC2020/speech/train/callback_vgg/vgg2_dense_log.csv', sep=',')

plt.plot(vgg_results['epoch'],vgg_results['val_loss'])
plt.plot(vgg_results['epoch'],vgg_results['loss'])
plt.legend(['Validation', 'Train'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
vgg_dense_best_model = tf.keras.models.load_model('gdrive/My Drive/MDC2020/speech/train/callback_vgg/vgg2_dense-07-1.69.hdf5')
avaliar_modelo(vgg_dense_best_model, test_generator_3, normalizado=True, num_classes=12)
for layer in vgg_dense_best_model.layers:
    layer.trainable = True

vgg_dense_best_model.summary()
# Definindo o Call Back
checkpoints = tf.keras.callbacks.ModelCheckpoint(filepath = 'gdrive/My Drive/MDC2020/speech/train/callback_vgg/ft_vgg-{epoch:02d}-{val_loss:.2f}.hdf5',
                                                 monitor = 'val_loss',
                                                 save_best_only=True, 
                                                 save_weights_only=False,
                                                 mode = 'min')
training_logs = tf.keras.callbacks.CSVLogger('gdrive/My Drive/MDC2020/speech/train/callback_vgg/ft_vgg_log.csv', separator=",", append=False)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 9)

# Definindo o compilador
vgg_dense_best_model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics='accuracy')

# Treinamento
vgg_dense_best_model_history = vgg_dense_best_model.fit(train_generator_3,             
                                epochs=40,
                                validation_data=validation_generator_3,
                                workers=8,
                                class_weight=class_weights,
                                callbacks = [checkpoints, training_logs, early_stopping])
# Continuando o treinamento
vgg_ft_cont = tf.keras.models.load_model('gdrive/My Drive/MDC2020/speech/train/callback_vgg/ft_vgg-02-0.55.hdf5')

checkpoints = tf.keras.callbacks.ModelCheckpoint(filepath = 'gdrive/My Drive/MDC2020/speech/train/callback_vgg/ft_vgg_cont.hdf5',
                                                 monitor = 'val_loss',
                                                 save_best_only=False, 
                                                 save_weights_only=False,
                                                 mode = 'min')
training_logs_cont = tf.keras.callbacks.CSVLogger('gdrive/My Drive/MDC2020/speech/train/callback_vgg/ft_vgg_log.csv', separator=",", append=False)
vgg_ft_cont_hist = vgg_ft_cont.fit(train_generator_3,             
                                epochs=10,
                                validation_data=validation_generator_3,
                                initial_epoch=2,
                                workers=8,
                                class_weight=class_weights,
                                callbacks = [checkpoints, training_logs_cont])
# plotando curvas de Loss
vgg_ft_results = pd.read_csv('gdrive/My Drive/MDC2020/speech/train/callback_vgg/ft_vgg_log.csv', sep=',')

plt.plot(vgg_ft_results['epoch'],vgg_ft_results['val_loss'])
plt.plot(vgg_ft_results['epoch'],vgg_ft_results['loss'])
plt.legend(['Validation', 'Train'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
from tensorflow.keras import layers

complexo1 = tf.keras.Sequential()

# Conv 2D e Max Pooling
complexo1.add(layers.Conv2D(10, 3, padding='valid', input_shape=(L,W,n_channel)))
complexo1.add(layers.BatchNormalization(momentum=0.8))
complexo1.add(layers.ReLU())
complexo1.add(layers.MaxPooling2D(pool_size=(2,2)))

# Conv 2D e Max Pooling
complexo1.add(layers.Conv2D(50, 3, padding='valid'))
complexo1.add(layers.BatchNormalization(momentum=0.8))
complexo1.add(layers.ReLU())
complexo1.add(layers.MaxPooling2D(pool_size=(2,2)))

# Conv 2D e Max Pooling
complexo1.add(layers.Conv2D(100, 3, padding='valid'))
complexo1.add(layers.BatchNormalization(momentum=0.8))
complexo1.add(layers.ReLU())
complexo1.add(layers.MaxPooling2D(pool_size=(2,2)))

# Conv 2D e Max Pooling
complexo1.add(layers.Conv2D(150, 3, padding='valid'))
complexo1.add(layers.BatchNormalization(momentum=0.8))
complexo1.add(layers.ReLU())
complexo1.add(layers.MaxPooling2D(pool_size=(2,2)))

# Flatten
complexo1.add(layers.Flatten())

# Densa 1
complexo1.add(layers.Dense(100, activation='relu'))

# Dropout

complexo1.add(tf.keras.layers.Dropout(0.2))

# Densa 2
complexo1.add(layers.Dense(50, activation='relu'))

# Dense Final
complexo1.add(layers.Dense(num_classes, activation='softmax'))

complexo1.summary()
'''# Definindo o Call Back
checkpoints = tf.keras.callbacks.ModelCheckpoint(filepath = 'gdrive/My Drive/MDC2020/speech/train/callback_complexo1/complexo1-{epoch:02d}-{val_loss:.2f}.hdf5',
                                                 monitor = 'val_loss',
                                                 save_best_only=True, 
                                                 save_weights_only=False,
                                                 mode = 'min')
training_logs = tf.keras.callbacks.CSVLogger('gdrive/My Drive/MDC2020/speech/train/callback_complexo1/complexo1_log.csv', separator=",", append=False)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 9)

# Definindo o compilador
complexo1.compile(optimizer=tf.keras.optimizers.Adam(1e-2),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics='accuracy')

# Treinamento
complexo1_history = complexo1.fit(train_generator_3,             
                                epochs=40,
                                validation_data=validation_generator_3,
                                workers=8,
                                class_weight=class_weights,
                                callbacks = [checkpoints, training_logs, early_stopping])'''
'''# Continuando o treinamento
complexo1_cont = tf.keras.models.load_model('gdrive/My Drive/MDC2020/speech/train/callback_complexo1/complexo1-08-1.08.hdf5')

checkpoints = tf.keras.callbacks.ModelCheckpoint(filepath = 'gdrive/My Drive/MDC2020/speech/train/callback_complexo1/cont_complexo1-{epoch:02d}-{val_loss:.2f}.hdf5',
                                                 monitor = 'val_loss',
                                                 save_best_only=True, 
                                                 save_weights_only=False,
                                                 mode = 'min')
training_logs_cont = tf.keras.callbacks.CSVLogger('gdrive/My Drive/MDC2020/speech/train/callback_complexo1/complexo1_cont_log.csv', separator=",", append=False)
complexo1_cont_h = complexo1_cont.fit(train_generator_3,             
                                epochs=40,
                                validation_data=validation_generator_3,
                                initial_epoch=8,
                                workers=8,
                                class_weight=class_weights,
                                callbacks = [checkpoints, training_logs_cont])'''
rede_conv = pd.concat([pd.read_csv('gdrive/My Drive/MDC2020/speech/train/callback_complexo1/complexo1_log.csv', sep=',')[1:8],
                            pd.read_csv('gdrive/My Drive/MDC2020/speech/train/callback_complexo1/complexo1_cont_log.csv', sep=',')])
                            
plt.figure(figsize=(6,4))
plt.plot(rede_conv['epoch'], rede_conv['val_loss'])
plt.plot(rede_conv['epoch'], rede_conv['loss'])
plt.legend(['Validation', 'Train'])
plt.ylabel('Loss')
plt.xlabel('Epoch')   
rede_conv_best_model = tf.keras.models.load_model('gdrive/My Drive/MDC2020/speech/train/callback_complexo1/cont_complexo1-14-1.03.hdf5')
avaliar_modelo_threshold(rede_conv_best_model, test_generator_3, normalizado=True, num_classes=12, threshold=0.4)
rede_conv = pd.read_csv('gdrive/My Drive/MDC2020/speech/train/callback_vgg/ft_vgg_log.csv', sep=',')
                     #       pd.read_csv('gdrive/My Drive/MDC2020/speech/train/callback_vgg/ft_vgg_log.csv', sep=',')])
                            
plt.figure(figsize=(6,4))
plt.plot(rede_conv['epoch'], rede_conv['val_loss'])
plt.plot(rede_conv['epoch'], rede_conv['loss'])
plt.legend(['Validation', 'Train'])
plt.ylabel('Loss')
plt.xlabel('Epoch')   
vgg_tf_best_model = tf.keras.models.load_model('gdrive/My Drive/MDC2020/speech/train/callback_vgg/ft_vgg_cont.hdf5')
avaliar_modelo_threshold(vgg_tf_best_model, test_generator_3, normalizado=True, num_classes=12, threshold=0.4)
!p7zip -d --keep 'gdrive/My Drive/MDC2020/speech/test.7z'
# melhor modelo
vgg_tf_best_model = tf.keras.models.load_model('gdrive/My Drive/MDC2020/speech/train/callback_vgg/ft_vgg_cont.hdf5')
baseline1_best_model = tf.keras.models.load_model('gdrive/My Drive/MDC2020/speech/train/callback_vgg/baseline1-08-0.61.hdf5')

# gerando o .csv
vgg_tf_csv = gerar_csv_kaggle(vgg_tf_best_model, 'test/audio/',class_dict, threshold=0.4)
baseline1_best_model_csv = gerar_csv_kaggle(baseline1_best_model, 'test/audio/',class_dict, threshold=0.4)

# salvando o .csv
vgg_tf_csv.to_csv('gdrive/My Drive/MDC2020/speech/vgg.csv', index=False)
baseline1_best_model_csv.to_csv('gdrive/My Drive/MDC2020/speech/baseline.csv', index=False)