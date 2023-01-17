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
#titanic datamizi okuyoruz

data=pd.read_csv("../input/_titanic_.csv")

#datamizin ilk 5 satirini gozlemliyoruz

data.head()
#datamizin son 5 satirini gozlemliyoruz

data.tail()
#Veri setine ait genel sayıları görüntülenmiştir.

#Buna göre veri seti 12 sütundan ve 891 satırdan oluşmaktadir

#yaş(age), kabin no(cabin) ve gemiye biniş limanı(embarked) sütunları dışında eksik satıra sahip sütun bulunmamaktadır.

data.info()
#Veri setine ait tanımlayıcı istatistik tablosu elde edilmiştir

#Tabloda yer alan anlamlı kısımları yorumlanırsa; 

#yolcuların %38’i hayatta kalmıştır,

#Yolcuların yaş ortalaması ondalıklı değerle ifade edilirse 29,6’dır

#En yaşlı yolcunun yaşı 80 iken, en genç yolcu 0.42 ile ifade edildiğine göre 5 aylık bir bebektir

#Yolcuların ödedikleri bilet fiyatı ortalama 32,2 birimdir(dolar)

#Bir bilet için ödenen en yüksek değer 512’dir

#Yolcular arasında en fazla kardeş/eş sayısına sahip yolcu için rakam 8 iken, en fazla ebeveyn/çocuğa sahip için rakam 6’dır.

data.describe()
#Veri setinde analize tabi tutulmayacak / analiz için gereksiz sütunlar çıkarılmıştır.

#Buna göre yolcu ID no, isim, kamara no ve bilet kodu değişkenleri çıkarılmıştır.

data = data.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1)
data.head()
#Yolcuların cinsiyete göre dağılımlarına bakıldığında:

#erkek yolcu sayısı, 577

#kadın yolcu sayısı, 314’tür

print ('Number of male passengers: ', len(data.groupby('Sex').groups['male']))

print ('Number of female passengers: ', len(data.groupby('Sex').groups['female']))
#Daha sonra kullanılmak üzere, erkek ve kadın yolcuları ayrı ayrı analize tabi tutabilmek için cinsiyete göre,

#‘male_passenger’ ve ‘female_passenger’ adında iki ayrı veri çerçevesi (data frame) oluşturulmuştur.

male_passenger = data[data['Sex']== 'male']

female_passenger = data[data['Sex']== 'female']
#Erkek ve kadınların haricinde, 

#yaşlarını 16 altı olarak kabul ettiğim çocuk yolcu(kid_passenger) data-frame oluşturulmuştur.

kid_passenger = data[data['Age'] < 16]
#Yetişkin erkek ve yetişkin kadın yolcuları belirlemek amacıyla,

#oluşturulan çocuk yolcu (kid_passenger) data-frame ayrı ayrı erkek (male_passenger) ve kadın (female_passenger)

#yolcu veri çerçevelerinden ayırarak öncelikle erkek (male_kid_passenger) ve kız çocuklarını (female_kid_passenger) ayrı ayrı belirlenmiştir.

male_kid_passenger = kid_passenger[kid_passenger['Sex'] == 'male']

female_kid_passenger = kid_passenger[kid_passenger['Sex'] == 'female']
#Pandas .drop() fonksiyonu kullanımıyla erkek (male_kid_passenger) ve

#kız çocuklarının (female_kid_passenger) erkek ve kadın yolcu data-frame ayrılarak 

#yetişkin erkek (adult_male_passenger) ve yetişkin kadın (adult_female_passenger) data-frame oluşturulmuştur.

adult_male_passenger = male_passenger.drop(male_kid_passenger.index[:])

adult_female_passenger = female_passenger.drop(female_kid_passenger.index[:])
#Cinsiyet göre ve yetişkin/çocuk yolcu sayıları elde edilmiştir.

print ('Number of all passengers:', len(data))

print ('Number of male passengers:', len(male_passenger))

print ('Number of female passengers:', len(female_passenger))

print ('Number of adult male passengers:', len(adult_male_passenger))

print ('Number of adult female passengers:', len(adult_female_passenger))

print ('Number of kid passengers:', len(kid_passenger)) 
#.groupby() fonksiyonu kullanılarak istenen yolcu sayılarını elde etmede alternatif kod kullanımı gösterilmiştir.

print( 'Number of male passengers:', len(data.groupby('Sex').groups['male']))

print( 'Number of female passengers:', len(data.groupby('Sex').groups['female']))

print( 'Number of kid passengers:', data['Age'].apply(lambda x: x < 16).sum())
#Alternatif bir yaş dağılımı bulgusunun elde edilmesi gösterilmiştir.

#Yeni bir fonksiyon tanımlanarak .apply() uygulamasıyla elde edilen farklı yaş gruplarına göre dağılım elde edilmiştir. 

#Tanımlanan fonksiyona göre 0-15 yaş aralığı ‘Child'(çocuk), 16-24 yaş aralığı ‘Young'(genç),

#24 yaş üstü ‘Adult'(Yetişkin) olarak adlandırılmıştır. 

#.value_counts() fonksiyonu ile bu yaş aralıklarındaki yolcu sayıları elde edilmiştir.

def age_distribution(x):

    if x>=0 and x <16:

        return 'Child'

    elif x>=16 and x<=24:

        return 'Young'

    else:

        return 'Adult'

    

data['Age'].apply(age_distribution).value_counts()
#.mean() fonksiyonu ile yetişkin erkek (adult_male_passenger),

#yetişkin kadın (adult_female_passenger) ve çocuk yolcuların(kid_passenger) yaş ortalamaları elde edilmiştir.

print( 'Average age of adult male passengers:', adult_male_passenger['Age'].mean())

print( 'Average age of adult female passengers:', adult_female_passenger['Age'].mean())

print( 'Average age of kid passengers:', kid_passenger['Age'].mean())
#Sosyo-ekonomik bir gösterge olarak yolcuların yolcu sınıflarına(Pclass) göre sayıları 

#.value_counts() fonksiyonuyla elde edilmiştir.

data['Pclass'].value_counts()
#Bu çalışmanın içeriğinde herhangi bir analize tabi tutulmasa da her ayrı yolcu sınıfı için veri çerçeveleri oluşturulmuştur.

first_class_passenger = data[data['Pclass'] == 1]

second_class_passenger = data[data['Pclass'] == 2]

third_class_passenger = data[data['Pclass'] == 3]
#Yolcuların gemiye biniş limanı bilgisini içeren ‘Embarked’ değişkeninin değerleri .describe() fonksiyonu ile tanımlanmıştır.

#Sonuca göre; yolcular 3 ayrı limandan gemiye binmişler,

#en fazla yolcunun-(644 yolcunun) biniş yaptığı liman ‘S’ ile tanımlanan Southampton limanı olmuştur.

#891 yolcudan ikisinin biniş bilgisi yoktur.

data['Embarked'].describe()
#‘Embarked’ sütununda eksik iki hücre, sütunda en fazla değer olan ‘S’ ile doldurulmuştur.

data['Embarked'] = data['Embarked'].fillna('S')
#Cinsiyete göre hayatını kaybeden(0) ve kurtulan yolcuların(1) sayıları gösterilmiştir.

data.groupby('Sex')['Survived'].value_counts()
#Hayatta kalan yetişkin kadın ve yetişkin erkek yolcuların oranı incelenmiştir.

#Buna göre yetişkin kadın yolcuların %75’i hayatta kalırken, yetişken erkek yolcuların %84’ü hayatını kaybetmiştir.

print( 'Mean of survived adult female passengers:', adult_female_passenger['Survived'].mean())

print( 'Mean of survived adult male passengers:', adult_male_passenger['Survived'].mean())
#Yolcu sınıflarına göre hayatta kalan/hayatını kaybeden yolcu sayılarını bulmada size() ve unstack() fonksiyonlarının kullanılmıştır.

#Öncelikle .groupby() ile ‘Pclass’ ve ‘Survived’ sütunlarını gruplanmıştır.

class_survived = data.groupby(['Pclass', 'Survived'])
#size() ile yolcu sınıflarına göre hayatını kaybeden(0) ve hayatta kalanların(1) sayıları elde edilmiştir.

class_survived.size()
#unstack() ile elde edilen sonuç daha okunabilir bir formata dönüştürülmüştür.

class_survived.size().unstack()
#Yolcu sınıflarına göre hayatını kaybeden ve hayatta kalan erkek yolcu sayıları tablosu gösterilmiştir.

print('Surviving numbers of male passengers by passenger class:\n ',male_passenger.groupby(['Pclass', 'Survived']).size().unstack())
#Yolcu sınıflarına göre hayatını kaybeden ve hayatta kalan kadın yolcu sayıları tablosu gösterilmiştir.

print('Surviving numbers of female passengers by passenger class:\n',female_passenger.groupby(['Pclass', 'Survived']).size().unstack())
#Yalnız yolcuları (alone_passenger) gösteren data-frame oluşturulmuştur.

#‘Yalnız yolcu’dan kasıt, gemide kardeş/eş ve ebeveyn/çocuğu olmayan yolculardır.

#Buna göre ‘SibSp’ ve ‘Parch’ değerlerinin her ikisi de sıfır olan yolcular, oluşturulan “alone_passenger” veri çerçevesine dahildir.

#.head() ile, oluşturulan veri çerçevesinin ilk 7 satırının görüntülenmiştir. 

without_sibsp_passenger = data[data['SibSp']==0]

alone_passenger = without_sibsp_passenger[without_sibsp_passenger['Parch']==0]

alone_passenger.head(7)
#Ailesiyle birlikte yolculuk eden yolcuları (family_passenger) gösteren data-frame oluşturulmuştur.

#Bu data-frame, veri setinden (data) yalnız yolcuların (alone_passenger) çıkarılmasıyla elde edilmiştir.

#Buna göre ‘SibSp’ veya ‘Parch’ değerlerinden herhangi biri sıfırdan büyük bir değere sahip yolcular bu data-frame (family_passenger) dahildir. 

#.tail() ile, oluşturulan data-frame son 6 satırı görüntülenmiştir.

family_passenger = data.drop(alone_passenger.index[:])

family_passenger.tail(6)
#Cinsiyete göre; yalnız seyahat eden, ailesiyle birlikte seyahat eden yolcuların hayatta kalma oranları ve

#yolcu sınıfına göre; yalnız seyahat eden, ailesiyle birlikte seyahat eden yolcuların hayatta kalma oranları elde edilmiştir.

print('Mean of survived alone passengers:', alone_passenger.groupby('Sex')['Survived'].mean())

print('Mean of survived passengers with family:', family_passenger.groupby('Sex')['Survived'].mean())

print('Mean of survived alone passengers:', alone_passenger.groupby('Pclass')['Survived'].mean())

print('Mean of survived passengers with family:', family_passenger.groupby('Pclass')['Survived'].mean())
#Ödenen bilet ücretlerine göre hayatta kalma oranlarının bulunmuştur.

#Bunun için bilet ücretleri (0’dan 512’ye) büyük farklarla çeşitlilik gösterdiğinden 

#bilet ücreti değerleri eşit aralıklara bölünerek hayatta kalma oranları elde edilmiştir.

#Öncelikle ücret sütununda yer alan boş hücreler ücret (‘Fare’) ortalama değeri ile doldurulmuştur.

data['Fare'].fillna(data['Fare'].dropna().median(), inplace=True)
#.qcut() fonksiyonu ile bilet ücret (‘Fare’) değerleri yolcu sayılarına göre dört eşit gruba ayrılmıştır.

#Buna göre, oluşan ücret bandları (‘FareBand’) “0-7.91”, “7.91-14.454”, “14.454-31”, “31-512.329” olmuştur.

data['FareBand'] = pd.qcut(data['Fare'], 4)

data['FareBand'].value_counts().sort_values(ascending= False)
#Oluşturulan ücret bandlarına göre hayatta kalma oranları gösterilmiştir.

#Bu sonuca göre yolcular arasında ödenen bilet ücretleri arttıkça hayatta kalma oranları da artmıştır.

data[['FareBand', 'Survived']].groupby(['FareBand'],as_index=False).mean().sort_values(by='FareBand',ascending=True)