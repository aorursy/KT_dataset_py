# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import soundfile as sf

import seaborn as sns

import matplotlib.pyplot as plt

import IPython.display as ipd

import librosa

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split #para separação de dados de treinamento e teste 

from sklearn.decomposition import PCA #para reduzir as dimensões

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        break

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/trabalho-01-classificao-musical/file_dataset.csv")

df

def get_soundfile(file, genre):

    return sf.read(f"/kaggle/input/gtzan-genre-collection/genres/{genre}/{file}")

## Exemplo

sound, samplerate = get_soundfile("metal.00091.au","metal")

sound, samplerate


#Transformar o dataframe dado em numpy array para estudar as suas informações

array_df=df.values #transforma df em numpy array
#analise pro rms

metal=[]

reggae=[]

for elem in array_df: 

    if elem[1]=="metal":

        sound,samplerate=get_soundfile(elem[0],elem[1])#salva a musica e o samplerate de cada linha do array

        rms_sound=librosa.feature.rms(y=sound)[0].min()

        metal.append(rms_sound)

    elif elem[1]=="reggae":

        sound,samplerate=get_soundfile(elem[0],elem[1])#salva a musica e o samplerate de cada linha do array

        rms_sound=librosa.feature.rms(y=sound)[0].min()

        reggae.append(rms_sound)

sns.lineplot(x=range(1,len(metal)+1),y=metal,label="metal")

sns.lineplot(x=range(1,len(reggae)+1),y=reggae,label="reggae")

plt.xlabel("n-esima musica")

plt.ylabel("RMS mínimo")

plt.show



rms_reggae=reggae

rms_metal=metal

#analise pro centroid

metal=[]

reggae=[]

for elem in array_df: 

    if elem[1]=="metal":

        sound,samplerate=get_soundfile(elem[0],elem[1])#salva a musica e o samplerate de cada linha do array

        rms_sound=librosa.feature.spectral_centroid(y=sound)[0].min()

        metal.append(rms_sound)

    elif elem[1]=="reggae":

        sound,samplerate=get_soundfile(elem[0],elem[1])#salva a musica e o samplerate de cada linha do array

        rms_sound=librosa.feature.spectral_centroid(y=sound)[0].min()

        reggae.append(rms_sound)

sns.lineplot(x=range(1,len(metal)+1),y=metal,label="metal")

sns.lineplot(x=range(1,len(reggae)+1),y=reggae,label="reggae")

plt.xlabel("n-esima musica")

plt.ylabel("centróide mínimo")

plt.show



centroid_reggae=reggae

centroid_metal=metal
sns.scatterplot(x=centroid_metal,y=rms_metal,label="metal")

sns.scatterplot(x=centroid_reggae,y=rms_metal,label="reggae")

plt.show()
#de df crio o dataframe df_reggae_metal com as features das musicas de reggae e metal e os labels ;da parte 1 do trabalho; 

#para esses generos utilizaremos somente rms e o centroide

array_df_reggae_metal=np.zeros((1,3))#array que vo usar pra criar o dataframe

for elem in array_df:# se o genero for reggae ou metal vai tacar no meu novo array

    if elem[1]=="metal" or elem[1]=='reggae':

        sound,samplerate=get_soundfile(elem[0],elem[1])#salva a musica e o samplerate de cada linha do array

        rms_sound=librosa.feature.rms(y=sound).min()#vamos pegar o minimo pq a o som mais baixo do metal normalmente eh mais alto que o som mais baixo do reggae

        centroid_sound=librosa.feature.spectral_centroid(sound)[0].min()#de forma análoga ao rms tb vamos pegar o min

        linha_df=np.array([rms_sound, centroid_sound, elem[1]])#cada linha do dataframe terah o rms o centroid e o genero

        array_df_reggae_metal=np.vstack([array_df_reggae_metal,linha_df])#um vstack pra ir adicionando as linhas

array_df_reggae_metal=array_df_reggae_metal[1:,:]#tira a primeira linha de zeros

df_reggae_metal=pd.DataFrame(data=array_df_reggae_metal, columns=['RMS min', 'CENTROID min', 'genre'])



df_reggae_metal
#separando samples em X e labels das samples em y

X, y = df_reggae_metal.iloc[:,:-1], df_reggae_metal.iloc[:,-1]



#salvando X e y em variáveis de garantia

X_garantia=X

y_garantia=y



#fatorizando as labels em y. Isto eh, passando de strings pra numeros

y, labels=pd.factorize(y)



#separacao de dados de treinamento e de teste

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=99)



#normalizando

scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)



#normalizando Xtrain e Xtest

scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test) 





#aplicando randomforests e verificando a taxa de acertos nos dados de teste nos dados de treino

#conforme a quantidade de arvores:

train_scores = np.zeros(0)

test_scores = np.zeros(0)

random_forests = []

for i in range(1,10):

    # Ao escolhermos o parametro n_estimators, estamos dizendo quantas arvores queremos que nossa random forest tenha

    random_forest = RandomForestClassifier(n_estimators=i,max_depth=10,random_state=99)

    random_forest.fit(X_train,y_train)

    train_scores = np.hstack([train_scores, random_forest.score(X_train,y_train)])

    test_scores = np.hstack([test_scores, random_forest.score(X_test,y_test)])

    random_forests.append(random_forest)

sns.lineplot(x=range(1,10),y=train_scores,label="train score")

sns.lineplot(x=range(1,10),y=test_scores,label="test score")

plt.xlabel('Número de Árvores')

plt.ylabel('Taxa de Acertos')

plt.legend()

plt.show()





       





#analise pro rms

popular=[]

disco=[]

for elem in array_df: 

    if elem[1]=="pop":

        sound,samplerate=get_soundfile(elem[0],elem[1])#salva a musica e o samplerate de cada linha do array

        rms_sound=librosa.feature.rms(y=sound)[0].max()

        popular.append(rms_sound)

    elif elem[1]=="disco":

        sound,samplerate=get_soundfile(elem[0],elem[1])#salva a musica e o samplerate de cada linha do array

        rms_sound=librosa.feature.rms(y=sound)[0].max()

        disco.append(rms_sound)

sns.lineplot(x=range(1,len(popular)+1),y=popular,label="popular")

sns.lineplot(x=range(1,len(disco)+1),y=disco,label="disco")

plt.xlabel("n-esima musica")

plt.ylabel("RMS máximo")

plt.show()



rms_disco=disco

rms_popular=popular
#analise pro centroid

popular=[]

disco=[]

for elem in array_df: 

    if elem[1]=="pop":

        sound,samplerate=get_soundfile(elem[0],elem[1])#salva a musica e o samplerate de cada linha do array

        centroid = librosa.feature.spectral_centroid(sound)[0].max()

        popular.append(centroid)

    elif elem[1]=="disco":

        sound,samplerate=get_soundfile(elem[0],elem[1])#salva a musica e o samplerate de cada linha do array

        centroid = librosa.feature.spectral_centroid(sound)[0].max()

        disco.append(centroid)

sns.lineplot(x=range(1,len(popular)+1),y=popular,label="popular")

sns.lineplot(x=range(1,len(disco)+1),y=disco,label="disco")

plt.xlabel("n-esima musica")

plt.ylabel("Centróide máximo")

plt.show()



centroid_disco=disco

centroid_popular=popular
#analise pro flatness

popular=[]

disco=[]

for elem in array_df: 

    if elem[1]=="pop":

        sound,samplerate=get_soundfile(elem[0],elem[1])#salva a musica e o samplerate de cada linha do array

        flatness = librosa.feature.spectral_flatness(sound)[0].max()

        popular.append(flatness)

    elif elem[1]=="disco":

        sound,samplerate=get_soundfile(elem[0],elem[1])#salva a musica e o samplerate de cada linha do array

        flatness = librosa.feature.spectral_flatness(sound)[0].max()

        disco.append(flatness)

sns.lineplot(x=range(1,len(popular)+1),y=popular,label="popular")

sns.lineplot(x=range(1,len(disco)+1),y=disco,label="disco")

plt.xlabel("n-esima musica")

plt.ylabel("Flatness máximo")

plt.show()



flatness_disco=disco

flatness_popular=popular
#analise pro mfcc

popular=[]

disco=[]

for elem in array_df: 

    if elem[1]=="pop":

        sound,samplerate=get_soundfile(elem[0],elem[1])#salva a musica e o samplerate de cada linha do array

        mfcc = librosa.feature.mfcc(sound)[0].max()

        popular.append(mfcc)

    elif elem[1]=="disco":

        sound,samplerate=get_soundfile(elem[0],elem[1])#salva a musica e o samplerate de cada linha do array

        mfcc = librosa.feature.mfcc(y=sound, sr=samplerate)[0].max()

        disco.append(mfcc)

sns.lineplot(x=range(1,len(popular)+1),y=popular,label="popular")

sns.lineplot(x=range(1,len(disco)+1),y=disco,label="disco")

plt.xlabel("n-esima musica")

plt.ylabel("MFCC máximo")

plt.show()



mfcc_disco=disco

mfcc_popular=popular
#de df crio o dataframe df_pop_disco com as features das musicas de pop e disco e os labels ;da parte 2 do trabalho; 

#para esses generos utilizaremos somente rms, o centroide e flatness



array_df_pop_disco=np.zeros((1,5))#array que vo usar pra criar o dataframe

for elem in array_df:# se o genero for pop ou disco vai tacar no meu novo array

    if elem[1]=="pop" or elem[1]=='disco':

        sound,samplerate=get_soundfile(elem[0],elem[1])#salva a musica e o samplerate de cada linha do array

        rms_sound=librosa.feature.rms(y=sound).max()#vamos pegar o max pq o som mais alto do disco normalmente eh mais alto que o som mais alto do pop

        centroid_sound=librosa.feature.spectral_centroid(sound)[0].max()#de forma análoga ao rms tb vamos pegar o max

        flatness_sound=librosa.feature.spectral_flatness(sound)[0].max()#flatness aqui

        mfcc_sound=librosa.feature.mfcc(y=sound, sr=samplerate)[0].max()#mfcc aqui

        linha_df=np.array([rms_sound, centroid_sound, flatness_sound, mfcc_sound, elem[1]])#cada linha do dataframe terah o rms, o centroid, o flatness, o mfcc e o genero

        array_df_pop_disco=np.vstack([array_df_pop_disco,linha_df])#um vstack pra ir adicionando as linhas

array_df_pop_disco=array_df_pop_disco[1:,:]#tira a primeira linha de zeros

df_pop_disco=pd.DataFrame(data=array_df_pop_disco, columns=['RMS max', 'CENTROID max', 'FLATNESS max', 'MFCC max', 'genre'])



df_pop_disco




#separando samples em X e labels das samples em y

X, y = df_pop_disco.iloc[:,:-1], df_pop_disco.iloc[:,-1]



#salvando X e y em variáveis de garantia

X_garantia=X

y_garantia=y



#fatorizando as labels em y. Isto eh, passando de strings pra numeros

y, labels=pd.factorize(y)



#separacao de dados de treinamento e de teste

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=99)



#normalizando

scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)



#normalizando Xtrain e Xtest

scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test) 





#aplicando randomforests e verificando a taxa de acertos nos dados de teste nos dados de treino conforme a quantidade de arvores:

train_scores = np.zeros(0)

test_scores = np.zeros(0)

random_forests = []

for i in range(1,10):

    # Ao escolhermos o parametro n_estimators, estamos dizendo quantas arvores queremos que nossa random forest tenha

    random_forest = RandomForestClassifier(n_estimators=i,max_depth=10,random_state=99)

    random_forest.fit(X_train,y_train)

    train_scores = np.hstack([train_scores, random_forest.score(X_train,y_train)])

    test_scores = np.hstack([test_scores, random_forest.score(X_test,y_test)])

    random_forests.append(random_forest)

sns.lineplot(x=range(1,10),y=train_scores,label="train score")

sns.lineplot(x=range(1,10),y=test_scores,label="test score")

plt.xlabel('Número de Árvores')

plt.ylabel('Taxa de Acertos')

plt.legend()

plt.show()





       





pca = PCA(2)



pca.fit(X_garantia.values)



pca_X = pca.transform(X_garantia.values)



sns.scatterplot(x=pca_X[:,0],y=pca_X[:,1],hue=y_garantia)
#analise pro rms

rock=[]

country=[]

for elem in array_df: 

    if elem[1]=="rock":

        sound,samplerate=get_soundfile(elem[0],elem[1])#salva a musica e o samplerate de cada linha do array

        rms_sound=librosa.feature.rms(y=sound)[0].min()

        rock.append(rms_sound)

    elif elem[1]=="country":

        sound,samplerate=get_soundfile(elem[0],elem[1])#salva a musica e o samplerate de cada linha do array

        rms_sound=librosa.feature.rms(y=sound)[0].min()

        country.append(rms_sound)

sns.lineplot(x=range(1,len(rock)+1),y=rock,label="rock")

sns.lineplot(x=range(1,len(country)+1),y=country,label="country")

plt.xlabel("n-esima musica")

plt.ylabel("RMS min")

plt.show()



rock=[]

country=[]

for elem in array_df: 

    if elem[1]=="rock":

        sound,samplerate=get_soundfile(elem[0],elem[1])#salva a musica e o samplerate de cada linha do array

        rms_sound=librosa.feature.rms(y=sound)[0].mean()

        rock.append(rms_sound)

    elif elem[1]=="country":

        sound,samplerate=get_soundfile(elem[0],elem[1])#salva a musica e o samplerate de cada linha do array

        rms_sound=librosa.feature.rms(y=sound)[0].mean()

        country.append(rms_sound)

sns.lineplot(x=range(1,len(rock)+1),y=rock,label="rock")

sns.lineplot(x=range(1,len(country)+1),y=country,label="country")

plt.xlabel("n-esima musica")

plt.ylabel("RMS medio")

plt.show()



rock=[]

country=[]

for elem in array_df: 

    if elem[1]=="rock":

        sound,samplerate=get_soundfile(elem[0],elem[1])#salva a musica e o samplerate de cada linha do array

        rms_sound=librosa.feature.rms(y=sound)[0].max()

        rock.append(rms_sound)

    elif elem[1]=="country":

        sound,samplerate=get_soundfile(elem[0],elem[1])#salva a musica e o samplerate de cada linha do array

        rms_sound=librosa.feature.rms(y=sound)[0].max()

        country.append(rms_sound)

sns.lineplot(x=range(1,len(rock)+1),y=rock,label="rock")

sns.lineplot(x=range(1,len(country)+1),y=country,label="country")

plt.xlabel("n-esima musica")

plt.ylabel("RMS max")

plt.show()
#analise pro centroid

rock=[]

country=[]

for elem in array_df: 

    if elem[1]=="rock":

        sound,samplerate=get_soundfile(elem[0],elem[1])#salva a musica e o samplerate de cada linha do array

        centroid=librosa.feature.spectral_centroid(y=sound)[0].min()

        rock.append(centroid)

    elif elem[1]=="country":

        sound,samplerate=get_soundfile(elem[0],elem[1])#salva a musica e o samplerate de cada linha do array

        centroid=librosa.feature.spectral_centroid(y=sound)[0].min()

        country.append(centroid)

sns.lineplot(x=range(1,len(rock)+1),y=rock,label="rock")

sns.lineplot(x=range(1,len(country)+1),y=country,label="country")

plt.xlabel("n-esima musica")

plt.ylabel("CENTROID min")

plt.show()



rock=[]

country=[]

for elem in array_df: 

    if elem[1]=="rock":

        sound,samplerate=get_soundfile(elem[0],elem[1])#salva a musica e o samplerate de cada linha do array

        centroid=librosa.feature.spectral_centroid(y=sound)[0].mean()

        rock.append(centroid)

    elif elem[1]=="country":

        sound,samplerate=get_soundfile(elem[0],elem[1])#salva a musica e o samplerate de cada linha do array

        centroid=librosa.feature.spectral_centroid(y=sound)[0].mean()

        country.append(centroid)

sns.lineplot(x=range(1,len(rock)+1),y=rock,label="rock")

sns.lineplot(x=range(1,len(country)+1),y=country,label="country")

plt.xlabel("n-esima musica")

plt.ylabel("CENTROID medio")

plt.show()



rock=[]

country=[]

for elem in array_df: 

    if elem[1]=="rock":

        sound,samplerate=get_soundfile(elem[0],elem[1])#salva a musica e o samplerate de cada linha do array

        centroid=librosa.feature.spectral_centroid(y=sound)[0].max()

        rock.append(centroid)

    elif elem[1]=="country":

        sound,samplerate=get_soundfile(elem[0],elem[1])#salva a musica e o samplerate de cada linha do array

        centroid=librosa.feature.spectral_centroid(y=sound)[0].max()

        country.append(centroid)

sns.lineplot(x=range(1,len(rock)+1),y=rock,label="rock")

sns.lineplot(x=range(1,len(country)+1),y=country,label="country")

plt.xlabel("n-esima musica")

plt.ylabel("CENTROID max")

plt.show()
#analise pro flatness

rock=[]

country=[]

for elem in array_df: 

    if elem[1]=="rock":

        sound,samplerate=get_soundfile(elem[0],elem[1])#salva a musica e o samplerate de cada linha do array

        flatness = librosa.feature.spectral_flatness(sound)[0].max()

        rock.append(flatness)

    elif elem[1]=="country":

        sound,samplerate=get_soundfile(elem[0],elem[1])#salva a musica e o samplerate de cada linha do array

        flatness = librosa.feature.spectral_flatness(sound)[0].max()

        country.append(flatness)

sns.lineplot(x=range(1,len(rock)+1),y=rock,label="rock")

sns.lineplot(x=range(1,len(country)+1),y=country,label="country")

plt.xlabel("n-esima musica")

plt.ylabel("Flatness max")

plt.show()



rock=[]

country=[]

for elem in array_df: 

    if elem[1]=="rock":

        sound,samplerate=get_soundfile(elem[0],elem[1])#salva a musica e o samplerate de cada linha do array

        flatness = librosa.feature.spectral_flatness(sound)[0].mean()

        rock.append(flatness)

    elif elem[1]=="country":

        sound,samplerate=get_soundfile(elem[0],elem[1])#salva a musica e o samplerate de cada linha do array

        flatness = librosa.feature.spectral_flatness(sound)[0].mean()

        country.append(flatness)

sns.lineplot(x=range(1,len(rock)+1),y=rock,label="rock")

sns.lineplot(x=range(1,len(country)+1),y=country,label="country")

plt.xlabel("n-esima musica")

plt.ylabel("Flatness medio")

plt.show()



rock=[]

country=[]

for elem in array_df: 

    if elem[1]=="rock":

        sound,samplerate=get_soundfile(elem[0],elem[1])#salva a musica e o samplerate de cada linha do array

        flatness = librosa.feature.spectral_flatness(sound)[0].min()

        rock.append(flatness)

    elif elem[1]=="country":

        sound,samplerate=get_soundfile(elem[0],elem[1])#salva a musica e o samplerate de cada linha do array

        flatness = librosa.feature.spectral_flatness(sound)[0].min()

        country.append(flatness)

sns.lineplot(x=range(1,len(rock)+1),y=rock,label="rock")

sns.lineplot(x=range(1,len(country)+1),y=country,label="country")

plt.xlabel("n-esima musica")

plt.ylabel("Flatness min")

plt.show()
#analise pro mfcc

rock=[]

country=[]

for elem in array_df: 

    if elem[1]=="rock":

        sound,samplerate=get_soundfile(elem[0],elem[1])#salva a musica e o samplerate de cada linha do array

        mfcc = librosa.feature.mfcc(sound)[0].max()

        rock.append(mfcc)

    elif elem[1]=="country":

        sound,samplerate=get_soundfile(elem[0],elem[1])#salva a musica e o samplerate de cada linha do array

        mfcc = librosa.feature.mfcc(y=sound, sr=samplerate)[0].max()

        country.append(mfcc)

sns.lineplot(x=range(1,len(rock)+1),y=rock,label="rock")

sns.lineplot(x=range(1,len(country)+1),y=country,label="country")

plt.xlabel("n-esima musica")

plt.ylabel("MFCC máximo")

plt.show()



rock=[]

country=[]

for elem in array_df: 

    if elem[1]=="rock":

        sound,samplerate=get_soundfile(elem[0],elem[1])#salva a musica e o samplerate de cada linha do array

        mfcc = librosa.feature.mfcc(sound)[0].mean()

        rock.append(mfcc)

    elif elem[1]=="country":

        sound,samplerate=get_soundfile(elem[0],elem[1])#salva a musica e o samplerate de cada linha do array

        mfcc = librosa.feature.mfcc(y=sound, sr=samplerate)[0].mean()

        country.append(mfcc)

sns.lineplot(x=range(1,len(rock)+1),y=rock,label="rock")

sns.lineplot(x=range(1,len(country)+1),y=country,label="country")

plt.xlabel("n-esima musica")

plt.ylabel("MFCC médio")

plt.show()



rock=[]

country=[]

for elem in array_df: 

    if elem[1]=="rock":

        sound,samplerate=get_soundfile(elem[0],elem[1])#salva a musica e o samplerate de cada linha do array

        mfcc = librosa.feature.mfcc(sound)[0].min()

        rock.append(mfcc)

    elif elem[1]=="country":

        sound,samplerate=get_soundfile(elem[0],elem[1])#salva a musica e o samplerate de cada linha do array

        mfcc = librosa.feature.mfcc(y=sound, sr=samplerate)[0].min()

        country.append(mfcc)

sns.lineplot(x=range(1,len(rock)+1),y=rock,label="rock")

sns.lineplot(x=range(1,len(country)+1),y=country,label="country")

plt.xlabel("n-esima musica")

plt.ylabel("MFCC min")

plt.show()



#vizualização de dados chroma_stft



lis=np.zeros(0)

country=[]

rock=[]

for elem in array_df: 

    lis=np.zeros(0)

    if elem[1]=="country":

        sound,samplerate=get_soundfile(elem[0],elem[1])#salva a musica e o samplerate de cada linha do array

        chromagram=librosa.feature.chroma_stft(sound, samplerate)

        for a in chromagram:

            lis=np.hstack([lis,a.mean()])

        country.append(lis.mean())



for elem in array_df: 

    lis=np.zeros(0)

    if elem[1]=="rock":

        sound,samplerate=get_soundfile(elem[0],elem[1])#salva a musica e o samplerate de cada linha do array

        chromagram=librosa.feature.chroma_stft(sound, samplerate)

        for a in chromagram:

            lis=np.hstack([lis, a.mean()])

        rock.append(lis.mean())

sns.lineplot(x=range(1,len(rock)+1),y=rock,label="rock")

sns.lineplot(x=range(1,len(country)+1),y=country,label="country")

plt.xlabel("n-esima musica")

plt.ylabel("chroma stft")

plt.show()



#vizualização de dados chroma_cqt



lis=np.zeros(0)

country=[]

rock=[]

for elem in array_df: 

    lis=np.zeros(0)

    if elem[1]=="country":

        sound,samplerate=get_soundfile(elem[0],elem[1])#salva a musica e o samplerate de cada linha do array

        chromagram=librosa.feature.chroma_cqt(sound, samplerate)

        for a in chromagram:

            lis=np.hstack([lis,a.mean()])

        country.append(lis.mean())



for elem in array_df: 

    lis=np.zeros(0)

    if elem[1]=="rock":

        sound,samplerate=get_soundfile(elem[0],elem[1])#salva a musica e o samplerate de cada linha do array

        chromagram=librosa.feature.chroma_cqt(sound, samplerate)

        for a in chromagram:

            lis=np.hstack([lis, a.mean()])

        rock.append(lis.mean())

sns.lineplot(x=range(1,len(rock)+1),y=rock,label="rock")

sns.lineplot(x=range(1,len(country)+1),y=country,label="country")

plt.xlabel("n-esima musica")

plt.ylabel("chroma cqt")

plt.show()



#vizualização de dados chroma_cens



lis=np.zeros(0)

country=[]

rock=[]

for elem in array_df: 

    lis=[]

    if elem[1]=="country":

        sound,samplerate=get_soundfile(elem[0],elem[1])#salva a musica e o samplerate de cada linha do array

        chromagram=librosa.feature.chroma_cens(sound, samplerate)

        for a in chromagram:

            lis=np.hstack([lis,a.mean()])

        country.append(lis.mean())



for elem in array_df: 

    lis=np.zeros(0)

    if elem[1]=="rock":

        sound,samplerate=get_soundfile(elem[0],elem[1])#salva a musica e o samplerate de cada linha do array

        chromagram=librosa.feature.chroma_cens(sound, samplerate)

        for a in chromagram:

            lis=np.hstack([lis, a.mean()])

        rock.append(lis.mean())

sns.lineplot(x=range(1,len(rock)+1),y=rock,label="rock")

sns.lineplot(x=range(1,len(country)+1),y=country,label="country")

plt.xlabel("n-esima musica")

plt.ylabel("chroma cens")

plt.show()



#de df crio o dataframe df_rock_country com as features das musicas de country e rock e os labels ;da parte 3 do trabalho; 

#num primeiro momento utilizaremos todos as features só pra ver



array_df_rock_country=np.zeros((1,8))#array que vo usar pra criar o dataframe

for elem in array_df:# se o genero for country ou rock vai tacar no meu novo array

    if elem[1]=="country" or elem[1]=='rock':

        sound,samplerate=get_soundfile(elem[0],elem[1])#salva a musica e o samplerate de cada linha do array

        chromagram_stft=librosa.feature.chroma_stft(sound, samplerate)# chromagra_stft é uma matriz, ou seja vetor de vetores

        chromagram_cqt=librosa.feature.chroma_cqt(sound, samplerate)#chromagra_cqt é uma matriz, ou seja vetor de vetores

        chromagram_cens=librosa.feature.chroma_cens(sound, samplerate)#chromagra_cens é uma matriz, ou seja vetor de vetores

        #vamos pegar o valor médio de cada uma dessas matrizes

        matrizes=[chromagram_stft, chromagram_cqt, chromagram_cens]

        lista_de_features=[librosa.feature.mfcc(y=sound, sr=samplerate)[0].mean(),librosa.feature.spectral_flatness(sound)[0].mean(),librosa.feature.spectral_centroid(y=sound)[0].mean(),librosa.feature.rms(y=sound)[0].mean()]#aqui vamos salvando o valor medio de cada matriz. essa lista vamos botar no dataframe

        for matriz in matrizes:

            vetor_de_medias=np.zeros(0)

            for vetor in matriz:

                vetor_de_medias=np.hstack([vetor_de_medias,vetor.mean()])

            lista_de_features.append(vetor_de_medias.mean()) #salvar a media das medias no vetor de features

        linha_df=np.array(lista_de_features+[elem[1]])#cada linha do dataframe terah o stft, cqt, cens e o genero

        array_df_rock_country=np.vstack([array_df_rock_country,linha_df])#um vstack pra ir adicionando as linhas

array_df_rock_country=array_df_rock_country[1:,:]#tira a primeira linha de zeros

df_rock_country=pd.DataFrame(data=array_df_rock_country,columns=["MFCC medio", 'flatness medio', 'centroid medio', 'RMS medio', 'chroma_stft mean', 'chroma_cqt mean', 'chroma_cens mean', 'genre'])



df_rock_country




#separando samples em X e labels das samples em y

X, y = df_rock_country.iloc[:,:-1], df_rock_country.iloc[:,-1]



#salvando X e y em variáveis de garantia

X_garantia=X

y_garantia=y



#fatorizando as labels em y. Isto eh, passando de strings pra numeros

y, labels=pd.factorize(y)



#separacao de dados de treinamento e de teste

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=99)



#normalizando

scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)



#normalizando Xtrain e Xtest

scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test) 





#aplicando randomforests e verificando a taxa de acertos nos dados de teste nos dados de treino conforme a quantidade de arvores:

train_scores = np.zeros(0)

test_scores = np.zeros(0)

random_forests = []

for i in range(1,10):

    # Ao escolhermos o parametro n_estimators, estamos dizendo quantas arvores queremos que nossa random forest tenha

    random_forest = RandomForestClassifier(n_estimators=i,max_depth=10,random_state=99)

    random_forest.fit(X_train,y_train)

    train_scores = np.hstack([train_scores, random_forest.score(X_train,y_train)])

    test_scores = np.hstack([test_scores, random_forest.score(X_test,y_test)])

    random_forests.append(random_forest)

sns.lineplot(x=range(1,10),y=train_scores,label="train score")

sns.lineplot(x=range(1,10),y=test_scores,label="test score")

plt.xlabel('Número de Árvores')

plt.ylabel('Taxa de Acertos')

plt.legend()

plt.show()





       





pca = PCA(2)



pca.fit(X_garantia.values)



pca_X = pca.transform(X_garantia.values)



sns.scatterplot(x=pca_X[:,0],y=pca_X[:,1],hue=y_garantia)
#de df crio o dataframe df_rock_country com as features das musicas de country e rock e os labels ;da parte 3 do trabalho; 

#Agora só com os features que julgamos ser bons



array_df_rock_country=np.zeros((1,4))#array que vo usar pra criar o dataframe

for elem in array_df:# se o genero for country ou rock vai tacar no meu novo array

    if elem[1]=="country" or elem[1]=='rock':

        sound,samplerate=get_soundfile(elem[0],elem[1])#salva a musica e o samplerate de cada linha do array

        chromagram_stft=librosa.feature.chroma_stft(sound, samplerate)# chromagra_stft é uma matriz, ou seja vetor de vetores

        chromagram_cqt=librosa.feature.chroma_cqt(sound, samplerate)#chromagra_cqt é uma matriz, ou seja vetor de vetores

        chromagram_cens=librosa.feature.chroma_cens(sound, samplerate)#chromagra_cens é uma matriz, ou seja vetor de vetores

        #vamos pegar o valor médio de cada uma dessas matrizes

        matrizes=[chromagram_stft, chromagram_cqt, chromagram_cens]

        lista_de_features=[]#aqui vamos salvando o valor medio de cada matriz. essa lista vamos botar no dataframe

        for matriz in matrizes:

            vetor_de_medias=np.zeros(0)

            for vetor in matriz:

                vetor_de_medias=np.hstack([vetor_de_medias,vetor.mean()])

            lista_de_features.append(vetor_de_medias.mean()) #salvar a media das medias no vetor de features

        linha_df=np.array(lista_de_features+[elem[1]])#cada linha do dataframe terah o stft, cqt, cens e o genero

        array_df_rock_country=np.vstack([array_df_rock_country,linha_df])#um vstack pra ir adicionando as linhas

array_df_rock_country=array_df_rock_country[1:,:]#tira a primeira linha de zeros

df_rock_country=pd.DataFrame(data=array_df_rock_country,columns=['chroma_stft mean', 'chroma_cqt mean', 'chroma_cens mean', 'genre'])



df_rock_country




#separando samples em X e labels das samples em y

X, y = df_rock_country.iloc[:,:-1], df_rock_country.iloc[:,-1]



#salvando X e y em variáveis de garantia

X_garantia=X

y_garantia=y



#fatorizando as labels em y. Isto eh, passando de strings pra numeros

y, labels=pd.factorize(y)



#separacao de dados de treinamento e de teste

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=99)



#normalizando

scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)



#normalizando Xtrain e Xtest

scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test) 





#aplicando randomforests e verificando a taxa de acertos nos dados de teste nos dados de treino conforme a quantidade de arvores:

train_scores = np.zeros(0)

test_scores = np.zeros(0)

random_forests = []

for i in range(1,10):

    # Ao escolhermos o parametro n_estimators, estamos dizendo quantas arvores queremos que nossa random forest tenha

    random_forest = RandomForestClassifier(n_estimators=i,max_depth=10,random_state=99)

    random_forest.fit(X_train,y_train)

    train_scores = np.hstack([train_scores, random_forest.score(X_train,y_train)])

    test_scores = np.hstack([test_scores, random_forest.score(X_test,y_test)])

    random_forests.append(random_forest)

sns.lineplot(x=range(1,10),y=train_scores,label="train score")

sns.lineplot(x=range(1,10),y=test_scores,label="test score")

plt.xlabel('Número de Árvores')

plt.ylabel('Taxa de Acertos')

plt.legend()

plt.show()

pca = PCA(2)



pca.fit(X_garantia.values)



pca_X = pca.transform(X_garantia.values)



sns.scatterplot(x=pca_X[:,0],y=pca_X[:,1],hue=y_garantia)