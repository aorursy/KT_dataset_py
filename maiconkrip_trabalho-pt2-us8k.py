# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
!pip install pydub
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy, scipy, matplotlib.pyplot as plt
import os
# Input import numpy, scipy, matplotlib.pyplot as plt files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# Any results you write to the current directory are saved as output.
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
    # quando fold nulo, significa para ler tudo.     
    audio = []
    audio_signals = []
    label= []
    labels=[]
    paths=[]   
    sampling_rate=[]
    librosa_sampling_rate = []
    
    
    if fold != []:
        #filtra somentes os folds que foram enviados.  
        filter_fold = df.fold.isin(fold)
        df = df[filter_fold]

        
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
# Para anlisar cada classe, primeiro é necessário extrair cada uma e criar um dataframe. 
dt_class = pd.DataFrame()
dt_fold = (csv[csv['fold']==1])
for i in (dt_fold.classID.unique()):
    dt_class = dt_class.append(dt_fold[dt_fold['classID']==i].head(1))    
dt_class
import librosa
sample, sample_path, sample_label, sample_S_Rate = to_dataset(dt_class)
from pydub import AudioSegment

combined_sounds = AudioSegment.silent(duration=1000)

for x in sample_path:
    sound = AudioSegment.from_wav(x)
    combined_sounds =  combined_sounds +  AudioSegment.silent(duration=2000)  + sound

combined_sounds.export("joinedFile.wav", format="wav")
import struct
import IPython.display as ipd
ipd.Audio('../working/joinedFile.wav')
%matplotlib inline
import numpy, scipy, matplotlib.pyplot as plt, IPython.display as ipd
import librosa, librosa.display
plt.rcParams['figure.figsize'] = (14, 5)
plt.style.use('seaborn-muted')
plt.rcParams['figure.figsize'] = (14, 5)
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.spines.left'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.bottom'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['image.interpolation'] = None
# Usando dois exemplos. O que eu criei e o dado para o trabalho.
#../working/joinedFile.wav
#../input/exemplo2/exemplo2.wav
signal1, sr1 = librosa.load('../input/exemplos-recebidos/exemplo.wav')
signal2, sr2 = librosa.load('../input/exemplos-recebidos/exemplo2.wav')
signal3, sr3 = librosa.load('../input/exemplos-recebidos/exemplo3.wav')
signaljoined, srjoined = librosa.load('../working/joinedFile.wav')

#Plotando o audio
librosa.display.waveplot(signal1, sr=sr1)
# Audio lido
ipd.Audio(signal1, rate=sr1)
#Plotando o audio
librosa.display.waveplot(signal2, sr=sr2)
ipd.Audio(signal2, rate=sr2)
#Plotando o audio
librosa.display.waveplot(signal3, sr=sr3)
ipd.Audio(signal3, rate=sr3)
#Plotando o audio
librosa.display.waveplot(signaljoined, sr=srjoined)
ipd.Audio(signaljoined, rate=srjoined)
# funcao plotar a energia e o RMS
def plot_RMSE(signal,sr):
    # Determina o hop e o frame para começar a calcular a energia e o RMS
    hop_length = 256
    frame_length = 512
    energy = numpy.array([
        sum(abs(signal[i:i+frame_length]**2))
        for i in range(0, len(signal), hop_length)
    ])
    rmse = librosa.feature.rms(signal, frame_length=frame_length, hop_length=hop_length, center=True)
    rmse = rmse[0]
    frames = range(len(energy))
    t = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
    librosa.display.waveplot(signal, sr=sr, alpha=0.4)
    plt.plot(t, energy/energy.max(), 'r--')             # normalized for visualization
    plt.plot(t[:len(rmse)], rmse/rmse.max(), color='g') # normalized for visualization
    plt.legend(('Energy', 'RMSE'))
print('Exemplo.wav')
plot_RMSE(signal1,sr1)
print('Exemplo2.wav')
plot_RMSE(signal2,sr2)
print('Exemplo3.wav')
plot_RMSE(signal3,sr3)
print('Arquivo com 10 Classes.wav')
plot_RMSE(signaljoined,srjoined)
#funcao para pegar o inicio de cada aumento de energia e seu final. 
def strip(signal, frame_length, hop_length, index, thresh):

    # Compute RMSE.
    rms = librosa.feature.rms(signal, frame_length=frame_length, hop_length=hop_length, center=True)  

    # inicializa um ponteiro na primeira posicao.     
    frame_index = index

    # Anda ate achar aumento no rms
    while rms[0][frame_index] < thresh and frame_index < len(rms[0])-1:
        frame_index += 1
     
    # Converte os frames em samples
    start_sample_index = librosa.frames_to_samples(frame_index, hop_length=hop_length)
    
    #remove a parte do vetor que ja foi lida.         
    frame_index_2 = frame_index
    
    #anda ate encontrar queda na energia abaixo do threeshold.
    while  rms[0][frame_index_2] > thresh and frame_index_2 < len(rms[0])-1:
        frame_index_2 += 1       
    
    #Converte os frames    
    end_sample_index = librosa.frames_to_samples(frame_index_2, hop_length=hop_length)
       
    #signal é o pedaço de audio.
    # frame_index_2 é o ponteiro onde a leitura parou
    # len(rms...) é para que o loop externo pare caso o arquivo tenha acabado. 
    return signal[start_sample_index:end_sample_index], frame_index_2, len(rms[0])-1
# Pegando o primeiro arquivo e colocando em uma variável unica que vai ser usada pra todos. 
signal = signaljoined
sr = srjoined
ipd.Audio(signal, rate=sr)

#loop para pegar cada som do arquivo. 
extracted_signals = []
index = 0
stop=0
thresh = 0.002
hop_length = 256
frame_length = 512
# Enquanto nao identificar o fim do arquivo...
while stop !=1:
    y,index,quit = strip(signal, frame_length, hop_length,index,thresh)
    extracted_signals.append(y)
    if index >= quit:
        stop=1

#deleta a ultima posicao que não armazena nada.
extracted_signals= extracted_signals[:-1]

print ('Quantidade De Audios Encontrados')
print(len(extracted_signals))
#exemplo de audio extraido
ipd.Audio(extracted_signals[1], rate=sr)
# O top_db varia de acordo com o arquivo. O melhor resultado foi 62 que conseguiu separar um audio com 10 classes.
# Nos demais arquivos, fica entre 40 e 50. 
def cut_signal(signal, top_db):
    y = librosa.effects.split(signal,top_db=top_db)
    extracted_signals = []
    for i in y:
        extracted_signals.append( signal[i[0]:i[1]] )
        #emphasized_signal = np.concatenate(l,axis=0)

    return extracted_signals
#Lembrando de cada arquivo:
#signal1, sr1 = librosa.load('../input/exemplos-recebidos/exemplo.wav')
#signal2, sr2 = librosa.load('../input/exemplos-recebidos/exemplo2.wav')
#signal3, sr3 = librosa.load('../input/exemplos-recebidos/exemplo3.wav')

#Realiza o corte para cada arquivo e armazena em diferentes variáveis. 
extracted_signals1 = cut_signal(signal1, 40)
extracted_signals2 = cut_signal(signal2, 50)
extracted_signals3 = cut_signal(signal3, 50)
extracted_signalsjoined = cut_signal(signaljoined, 62)

#exemplo de audio extraido
ipd.Audio(extracted_signals1[0], rate=sr)
#Extrai as features.
def extract_features(signal):
    mfccs = librosa.feature.mfcc(y=signal,  n_mfcc=40)
    mfccs_processed = np.mean(mfccs.T,axis=0)
     
    return mfccs_processed    
#cria um array para predição para cada audio
predict1 = []
for x in extracted_signals1:
    predict1.append(extract_features(x))    

predict2 = []
for x in extracted_signals2:
    predict2.append(extract_features(x))    

predict3 = []
for x in extracted_signals3:
    predict3.append(extract_features(x))    
    
predictjoined = []
for x in extracted_signalsjoined:
    predictjoined.append(extract_features(x))        
# A extração foi bem melhor. 
print('Classes encontradas no Exemplo.wav')
print(len(predict1))

print('Classes encontradas no Exemplo2.wav')
print(len(predict2))

print('Classes encontradas no Exemplo3.wav')
print(len(predict3))

print('Classes encontradas no Exemplojoined.wav')
print(len(predictjoined))

#importa o modelo 
from keras.models import Sequential, load_model
model = load_model('../input/model-keras/best_model.h5')

# realiza a predição
results1 = model.predict_classes(np.array(predict1))
results2 = model.predict_classes(np.array(predict2))
results3 = model.predict_classes(np.array(predict3))
resultsjoined = model.predict_classes(np.array(predictjoined))

import pickle

def translate(results):
    # Para mostrar as classes como texto, usamos o dicionario criado com base no encoder do modelo.
    d = pickle.load( open( "../input/dictionary/dict", "rb" ) )

    # Converte o dicionario
    d = {v: k for k, v in d.items()}

    # Cria a coluna de labels com base do dicionario. 
    label = [d[x] for x in results]
    df = pd.DataFrame({'classID':results, 'label':label})
    return df
pd_results_1 = translate(results1)
pd_results_2 = translate(results2)
pd_results_3 = translate(results3)
pd_results_joined = translate(resultsjoined)

# Compare o dataframe com a predição e o audio. 
print('Arquivo.wav')
print(pd_results_1)
ipd.Audio(signal1, rate=sr)

# Compare o dataframe com a predição e o audio. 
print('Arquivo2.wav')
print(pd_results_2)
ipd.Audio(signal2, rate=sr2)

# Compare o dataframe com a predição e o audio. 
print('Arquivo3.wav')
print(pd_results_3)
ipd.Audio(signal3, rate=sr3)
# Compare o dataframe com a predição e o audio. 
print('JoinedFile.wav')
print(pd_results_joined)
ipd.Audio(signaljoined, rate=srjoined)
