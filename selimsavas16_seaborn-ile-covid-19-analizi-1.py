import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

covid_19 = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")
# ObservationDate : Gözlem Tarihi

# Province/State : İl / Eyalet

# Country/Region : Ülke / Bölge

# Last Update : Son Güncelleme tarihi

# Confirmed : Onaylanmış kişi sayısı

# Deaths : Ölümler

# Recovered : Kurtarılan



covid_19.head()
covid_19_columns_v2 = ['sNo','ObservationDate','Province_State','Country_Region','Last Update',"Confirmed","Deaths","Recovered"]

covid_19.columns = covid_19_columns_v2

covid_19.head()
covid_19.info() # Değiştirilmiş mi?
seaborn_data_1 = pd.DataFrame({"Province_State": covid_19.Province_State,"Confirmed" : covid_19.Confirmed})

# seaborn_data_1 # Değerleri Görebilirsiniz?
Province_State_u = list(covid_19["Province_State"].unique())

# Province_State_u # Değerleri görebilirsiniz.


seaborn_data = pd.DataFrame({"Province_State": Province_State_u,"Confirmed" : ""})



seaborn_data
j = 0



for i in Province_State_u: # Benzersiz eyalet/şehir verimizde dolaşalım.

    

    x = seaborn_data_1[seaborn_data_1['Province_State']==i] # Benzersiz değerimiz , içinde çok sayıda aynı eyalet/şehir'i barındıran liste içinde eşleştiğinde, x değişkenine değeri ata.

    

    # İçi boş Confirmed değerlerine sahip, seaborn_data veri setimizin, j indeksindekin Confirmed sütununa,

    # for döngüsüyle gezdiğimiz sıradaki eyalet/şehir'e ait onaylanmış tanı sayısını basalım.

    seaborn_data.iloc[j,seaborn_data.columns.get_loc("Confirmed")] = sum(x.Confirmed)

    

    j+=1 # indeks değerimiz döngü boyunca 1 artsın. (Confirmed sütununda gezinmek için)

     
seaborn_data # Confirmed değerleri dolu
plt.figure(figsize=(10,10)) # Matplotlib sayesinde figür oluşturuyoruz.





# x: Eyalet/Şehirler

# y: Onaylanmış hasta sayısı

sns.barplot(x=seaborn_data['Province_State'], y=seaborn_data['Confirmed']) # sns kütüphanesi ile barplot oluşturuyoruz

plt.show()



# Farkettiğimiz gibi görselleştirme pek verimli olmadı. Gelin tüm değerleri değil sadece ilk 15 değerimizi görselleştirerlim:
new_index = (seaborn_data["Confirmed"].sort_values(ascending=False)).index.values



# Sort edilmiş yeni datamızı oluşturalım

sorted_data = seaborn_data.reindex(new_index)

sorted_data
plt.figure(figsize=(20,10)) 



# sns kütüphanesinden bir barplot oluştur 

sns.barplot(x=sorted_data['Province_State'][:20], y=sorted_data['Confirmed'][:20])



# Görsellik için yazılarımız

plt.xticks(rotation= 45)

plt.xlabel('Eyaletler / Şehirler')

plt.ylabel('Onaylanmış Covid-19 Hastası')

plt.title("Onaylanmış hasta sayısına göre eyalet/şehir 'lerin sıralanması")

plt.show()