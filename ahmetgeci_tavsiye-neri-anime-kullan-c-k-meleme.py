import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D



%matplotlib inline





plt.rcParams['figure.figsize'] = (6, 4)

plt.style.use('ggplot')

%config InlineBackend.figure_formats = {'png', 'retina'}

data = pd.read_csv('../input/anime.csv')
data.head()  # head paremetresiz ilk 5 teki verileri listeler
data.shape   # elimizde dataset 12294 satır 7 sütündan oluşuyo
#sıra geldi kullanıcılarımızı eklemeye

kullanici = pd.read_csv('../input/rating.csv')
kullanici.head(20)  # içine paremetre koyduğunuz ilk 10 veri listeler
kullanici.shape 
kullanici[kullanici['user_id']==1].rating.mean() 

 #Gördüğünüz gibi kullanıcı id' si 1 olan kullanızımız sıfırcı hoca gibi :))
kullanici[kullanici['user_id']==2].rating.mean()

#bu kullanıcımız ise pozitif bir kullanıcımız 
#En sondaki kullanımızı inceleyelim bir de 

kullanici[kullanici['user_id']==73516].rating.mean() 

#son kullanıcımız oldukça bonkör :)) 
K_basina_ortalama_listesi = kullanici.groupby(['user_id']).mean().reset_index()

K_basina_ortalama_listesi['mean_rating'] = K_basina_ortalama_listesi['rating']

K_basina_ortalama_listesi.drop(['anime_id' , 'rating'] , axis = 1 , inplace = True)

K_basina_ortalama_listesi.head(10)
databas = K_basina_ortalama_listesi.head()

datason = K_basina_ortalama_listesi.tail()

conc_data_row = pd.concat([databas,datason],axis = 0 , ignore_index=True)

conc_data_row
# Bu tarz listelemin yanında şu şekilde de listenizi geliştirebilirsiniz 

kullanici=pd.merge(kullanici,K_basina_ortalama_listesi,on=['user_id','user_id'])

#burada merge(birleştirme)metodu  ile iki listeyi user_id lerine göre birleştiriyoruz

#Bu şekilde kullanmanızı tavsiye ederim !!!

kullanici.head()
kullanici = kullanici.drop(kullanici[kullanici.rating < kullanici.mean_rating].index)



#ortalamanın üstündeki değerlendirme  verilerimize çıkartalım 
kullanici[kullanici['user_id'] ==1].head()
kullanici[kullanici['user_id']== 2].head(10)

kullanici[kullanici['user_id'] == 5].head(10)
kullanici.shape  

#data setimizin son halindeki satır ve sütünları 


kullanici['user_id'].value_counts(dropna=False)

# burada id numaralarının listemizde ne kadar yer aldığını öğreniriz

#48766 user_id sine sahip kullanıcı 10227 farklı şekilde değerlendirme yaptığını çıkartabiliriz 

kullanici['user_id'].unique()
kullanici = kullanici.rename({'rating':'userRating'}, axis='columns')

Birlesmis_data = pd.merge(data,kullanici,on=['anime_id', 'anime_id'])

Birlesmis_data = Birlesmis_data[Birlesmis_data.user_id <=20000]

Birlesmis_data.head(10)



# Not : Buradaki data =  anime.csv 'dir