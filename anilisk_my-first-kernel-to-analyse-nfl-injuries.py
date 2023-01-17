# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
Player_Trackdf = pd.read_csv("/kaggle/input/nfl-playing-surface-analytics/PlayerTrackData.csv")

Player_Trackdf.info()



# Oyuncularin maclardaki fiziksel aktivite dosyasini Player_Trackdf adina cevirerek indirdik ve inceledik

Gamedf = pd.read_csv("/kaggle/input/nfl-playing-surface-analytics/PlayList.csv")

Gamedf.info()





# Oynanan maclarin detaylarini (hava durumu, cim tipi vb) barindiran dosyayi Gamedf adina cevirerek indirdik ve inceledik.
import matplotlib.pyplot as plt



Injurydf = pd.read_csv("/kaggle/input/nfl-playing-surface-analytics/InjuryRecord.csv")



Injurydf.info()



# Sakatlik verilerini barindiran dosyayı Injurydf adina cevirerek indirdik ve inceledik





Knee_Injury = Injurydf[Injurydf.BodyPart == "Knee"]

Ankle_Injury = Injurydf[Injurydf.BodyPart == "Ankle"]

Foot_Injury = Injurydf[Injurydf.BodyPart == "Foot"]

Toes_Injury = Injurydf[Injurydf.BodyPart == "Toes"]

Heel_Injury = Injurydf[Injurydf.BodyPart == "Heel"]





print(Knee_Injury.describe())

print(Ankle_Injury.describe())

print(Foot_Injury.describe())

print(Toes_Injury.describe())

print(Heel_Injury.describe())



Injurydf.columns



# sakatliklarin olustugu bölgeleri ayri tanımladik ve sakatlik dosyasi altındaki parametrelerle basit ilişkisine baktik

# Diz sakatliklarinin genel olarak 7 gun ve uzeri gun kaybına;

# Ayak bilegi sakatliklarinin da diz sakatliklarina gore daha az olmakla birlikte genel olarak 7 gun ve uzeri gun kaybına;

# Ayak bas parmak sakatliklarinin da diz sakatliklari gibi genel olarak 7 gun ve uzeri gun kaybına;

# Ayak sakatiklarinin ise en fazla gun kaybına (28 gun ve uzeri + 42 gun ve uzeri) neden oldugunu goruyoruz.

# Bu anlamda en ciddi sakatliklarin Ayak sakatliklari oldugunu soyleyebiliriz.

# Topuk sakatliklarinin 28 gunden fazla gun kaybına neden olmadıgı (eksik data yuzunden de olabilir) gorulmekte.

Natural = Injurydf[Injurydf.Surface == "Natural"]

Synthetic = Injurydf[Injurydf.Surface == "Synthetic"]



print(Natural.describe())

print(Synthetic.describe())



# Injurydf dataframe'i icindeki cim tipininin df deki diger parametrelerle basit ilişkisine baktik

# Burada iliskisi incelenebilen datalar float olan datalar (ozellikle sakatliklarin neden oldugu gun kaybı)

# Cim tipinin BodyPart ile iliskisine bakamadik cunku BodyPart bir string (buna nasil bakilmasi gerektigini inceliyorum)
print(Natural.info())

print(Synthetic.info())



# Dogal cim tipi ile Sentetik cim tipi ile degiskenlik gösteren datalari inceledik

# Dogal cim tipi ile ilgili 48 adet data girisi var, dogal cimde oynanan mac sayisi = 48

# Sentetik cim tipi ile ilgili 57 data girisi var, sentetik cimde oynanan mac sayisi = 57
Injurydf.groupby('BodyPart').count()['PlayerKey'].plot(kind='bar', figsize=(15, 5), title='Injury Num by Body Part')

plt.show()



Injurydf.groupby('Surface').count()['PlayerKey'].plot(kind='barh', figsize=(15, 5), title='Injury Num by Body Part',color = "g")

plt.show()



Injurydf.groupby(['BodyPart','Surface']).count().unstack('BodyPart')['PlayerKey'].T.plot(kind='bar', figsize=(15, 5), title='Injury Body Part by Turf Type')

plt.show()





# Sakatliklari olustuklari anatomik bolgelere gore gruplandirdik ve indirdigimiz Matplotlib ile Bar grafiginde gorsellestirdik

# Sakatliklari olustuklari zemin tipine göre gruplandirdik ve yatay (horizontal) Bar grafiginde gorsellestirdik

# Sakatliklari hem olustuklari anatomik bolgelere gore hem de olustuklari zemin tipine gore gruplandirdik ve stack Bar grafiginde gorsellestirdik

# itiraf ediyorum barh ve unstack kodlarini kaggle discussionlarda buldum :)

# Grafiklerden ozellikle sentetik cimin ayak bas parmagi ve ayak bilegi sakatliklarinda on plana ciktigini goruyoruz

# diz sakatliklarinda dogal ve sentetik cim farki goze carpmiyor

# ayak sakatliklari dogal cimde daha cok, topuk sakatliklari ise sadece dogal cimde olusmus gorunuyor
Injurydf_1 = Injurydf.copy()

Injurydf_1.DM_M1 = Injurydf_1.DM_M1 - Injurydf_1.DM_M7

Injurydf_1.DM_M7 = Injurydf_1.DM_M7 - Injurydf_1.DM_M28

Injurydf_1.DM_M28 = Injurydf_1.DM_M28 - Injurydf_1.DM_M42



M1 = Injurydf_1.DM_M1.sum()

M7 = Injurydf_1.DM_M7.sum()

M28 = Injurydf_1.DM_M28.sum()

M42 = Injurydf_1.DM_M42.sum()



print(M1,M7,M28,M42)





data = [29, 39, 8, 29]

plt.bar(["0-7","7-28","28-42","over 42"], data)

plt.title('Days Missing')

plt.show()



# Sakatlik yuzunden kaybedilen gunleri gurupladık ve bar chartta goruntuledik.

# En fazla 7-28 gun kaybı yaratan sakatliklarin oldugunu goruyoruz.