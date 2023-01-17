import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



import missingno as msno 



import re



import matplotlib.pyplot as plt
data = pd.read_csv('../input/dokdokanonim/dokdokanonim.csv')

data
data.axes
data.isnull().any()
data.info()
data.isnull().sum()
msno.matrix(data)
datanull = pd.read_csv('../input/dokdokanonim/dokdokanonim.csv',na_values='yok')

msno.matrix(datanull)
msno.heatmap(datanull)
msno.dendrogram(datanull)
msno.heatmap(data)
data[data["ayaktakalmasure"].isnull()]
data[data["boyu"].isnull()]
data[data["cay"].isnull()]
data[data["dogumtarihi"].isnull()]
data[data["kilo"].isnull()]
data[data["su"].isnull()]
data[data["uykusuresi"].isnull()]
data.boyu.replace(15.0 , 155.0 , inplace= True)

data.kilo.replace(7.0 , 76.0 , inplace= True)

data.drop([81], axis=0, inplace=True)
data[(data['dogumtarihi']== 22021944.0)]

data.dogumtarihi.replace(22021944.0 , 76.0 , inplace= True)
data[data['dogumtarihi'] == 6.0]
data[data['dogumtarihi'] == 6.0].describe().max()

data.dogumtarihi.replace(6.0 , 60.0 , inplace= True)
data[data['kilo'] <= 50.0].kilo

data.kilo.replace(5.0 , 56.0 , inplace= True)
data=data.rename(columns = {'dogumtarihi':'yas'})
data=data.dropna()

data
data.menopoz.value_counts()

erkekdata=data[(data['cinsiyet']== 'erkek')]
erkekdata[(erkekdata['menopoz']== 'evet')]
erkekdata[(erkekdata['adetduzen']== 'evet')]
data.adetgun.value_counts()

erkekdata[(erkekdata['adetgun']!= 'yok')]
data.adetsuan.value_counts()

erkekdata[(erkekdata['adetsuan']== 'evet')]
#hepsi bayan

data['cinsiyet'][92]='kadin'

data['cinsiyet'][129]='kadin'

data['cinsiyet'][230]='kadin'

erkekdata=data[(data['cinsiyet']== 'erkek')]

erkekdata[(erkekdata['adetsuan']== 'evet')]
def ameliyatsayibul(ameliyat1):

    sayı=0

    if (ameliyat1=='yok'):

        sayı =0

    else:

        sayı=1

        a=re.findall('[0-9]',ameliyat1)

        if (a):

            b=a[0]

            sayı=int(b)

    return sayı

ameliyatsayi1 = data['ameliyat1'].apply(ameliyatsayibul)

def ameliyatsayibul2(ameliyat2):

    sayı=0

    if (ameliyat2=='yok'):

        sayı =0

    else:

        sayı=1

    return sayı

ameliyatsayi2 = data['ameliyat2'].apply(ameliyatsayibul2)

def ameliyatsayibul3(ameliyat3):

    sayı=0

    if (ameliyat3=='yok'):

        sayı =0

    else:

        sayı=1

    return sayı

ameliyatsayi3 = data['ameliyat3'].apply(ameliyatsayibul3)

def ameliyatsayibul4(ameliyat4):

    sayı=0

    if (ameliyat4=='yok'):

        sayı =0

    else:

        sayı=1

        a=re.findall('-',ameliyat4)

        if (a):

            sayı=len(a)+1

    return sayı

ameliyatsayi4 = data['ameliyat4'].apply(ameliyatsayibul4)

ameliyatsayi=ameliyatsayi1+ameliyatsayi2+ameliyatsayi3+ameliyatsayi4

data['ameliyatsayisi']=ameliyatsayi
data.to_csv ('dokdoktemiz.csv', index = False)
data.describe().T
kadindata=data[data.cinsiyet=='kadin']

kadindata.adetduzen.value_counts()
sns.countplot(kadindata.adetduzen)
kadindata.adetgun.value_counts()
sns.countplot(kadindata.adetgun)
kadindata.adetsuan.value_counts()
sns.countplot(kadindata.adetsuan)
kadindata.menopoz.value_counts()
sns.countplot(kadindata.menopoz)
data.aksam.value_counts()
sns.countplot(data.aksam)
data[data.aksam=='hayir'].cinsiyet.value_counts()
data.ajerjik.value_counts()
sns.countplot(data.ajerjik)
data[data.ajerjik=='var'].cinsiyet.value_counts()
sns.countplot(data[data.ajerjik=='var'].ameliyatsayisi)
sns.countplot(data[data.ajerjik=='var'].araogun)
sns.countplot(data[data.ajerjik=='var'].ayakkabi)
sns.countplot(data[data.ajerjik=='var'].ayaktakalmasure)
sns.countplot(data[data.ajerjik=='var'].beslenmesekli)
sns.distplot(data[data.ajerjik=='var'].boyu)
sns.countplot(data[data.ajerjik=='var'].cay)
sns.countplot(data[data.ajerjik=='var'].cinsiyet)
sns.distplot(data[data.ajerjik=='var'].yas)
sns.countplot(data[data.ajerjik=='var'].gozrengi)
sns.countplot(data[data.ajerjik=='var'].ilacalerjisi)
sns.countplot(data[data.ajerjik=='var'].kahve)
sns.countplot(data[data.ajerjik=='var'].kalp)
sns.countplot(data[data.ajerjik=='var'].kangrubu)
sns.distplot(data[data.ajerjik=='var'].kilo)
sns.countplot(data[data.ajerjik=='var'].menopoz)
data[data.ajerjik=='var'].meslek.value_counts()
sns.countplot(data[data.ajerjik=='var'].ogle)
sns.countplot(data[data.ajerjik=='var'].psikolojikdurum)
sns.countplot(data[data.ajerjik=='var'].romatizma)
sns.countplot(data[data.ajerjik=='var'].sacrengi)
sns.countplot(data[data.ajerjik=='var'].seker)
sns.countplot(data[data.ajerjik=='var'].sigara)
sns.countplot(data[data.ajerjik=='var'].spor)
sns.countplot(data[data.ajerjik=='var'].su)
sns.countplot(data[data.ajerjik=='var'].tansiyon)
sns.countplot(data[data.ajerjik=='var'].tenrengi)
sns.countplot(data[data.ajerjik=='var'].uykusuresi)
sns.countplot(data[data.ajerjik=='var'].yuruyus)
data.alerjikhastalikilac.value_counts()
data.alkol.value_counts()
sns.countplot(data.alkol)
data.alkolmiktar.value_counts()
data.ameliyat1.value_counts()
data.ameliyat1.unique()
data.ameliyat2.value_counts()
data.ameliyat3.value_counts()
data.ameliyat4.value_counts()
data.ameliyatsayisi.value_counts()
sns.countplot(data.ameliyatsayisi)
data.araogun.value_counts()
sns.countplot(data.araogun)
data[(data['araogun']== 'yok')]
data.ayakkabi.value_counts()
sns.countplot(data.ayakkabi)
data[(data['ayakkabi']== 'yok')]
data.beslenmesekli.value_counts()
sns.countplot(data.beslenmesekli)
data[(data['beslenmesekli']== 'yok')]
data.boyu.value_counts()
sns.boxplot(x= data.boyu)
data[data['boyu'] >= 185.0]
data[data['boyu'] >= 185.0].describe().T
data.cay.value_counts()
sns.boxplot(x= data.cay)
sns.countplot(data.cay)
data.cinsiyet.value_counts()
sns.countplot(data.cinsiyet)
data.yas.value_counts()
sns.boxplot(x= data.yas)
data.gozrengi.value_counts()
sns.countplot(data.gozrengi)
data[(data['gozrengi']== 'diger')]
data[(data['gozrengi']== 'yok')]
data.ilacalerjisi.value_counts()
sns.countplot(data.ilacalerjisi)
data.ilacalerjisiilaci.value_counts()
ilacalerjidata=data[data.ilacalerjisi=='var']
ilacalerjidata.ilacalerjisiilaci.value_counts()
sns.countplot(ilacalerjidata.ilacalerjisiilaci)
data.kahvalti.value_counts()
sns.countplot(data.kahvalti)
data[(data['kahvalti']== 'yok')]
data.kahve.value_counts()
sns.countplot(data.kahve)
data.kalp.value_counts()
sns.countplot(data.kalp)
data.kalpilac.value_counts()
kalpdata=data[data.kalp=='var']

data[data.kalp=='var'].kalpilac.value_counts()
data.kangrubu.value_counts()
sns.countplot(data.kangrubu)
data[(data['kangrubu']== 'yok')]
data.kayittarihi.value_counts()
data.kilo.value_counts()
sns.boxplot(x= data.kilo)
data.meslek.value_counts()
# meslekler gruplandırılabilir  data.meslek.replace('159' , 'yok' , inplace= True)
data.ogle.value_counts()
sns.countplot(data.ogle)
data[data.ogle=='yok']
data.psikolojikdurum.value_counts()
sns.countplot(data.psikolojikdurum)
data[data.psikolojikdurum=='yok']
data.romatizma.value_counts()
sns.countplot(data.romatizma)
data.romatizmailac.value_counts()
data[data.romatizma=='var'].romatizmailac.value_counts()
data.sacrengi.value_counts()
sns.countplot(data.sacrengi)
data[data.sacrengi=='yok']
data.searchKey.value_counts()
data.seker.value_counts()
sns.countplot(data.seker)
data.sekerilac.value_counts()
sekerdata=data[data.seker=='var']

sekerdata.sekerilac.value_counts()
sns.countplot(sekerdata.sekerilac)
data.sigara.value_counts()
sns.countplot(data.sigara)
data.spor.value_counts()
sns.countplot(data.spor)
data.sporsure.value_counts()
data.sporsıklık.value_counts()
data.sporturu.value_counts()
data.su.value_counts()
sns.boxplot(x=data.su)
sns.countplot(data.su)
data.tansiyon.value_counts()
sns.countplot(data.tansiyon)
data.tansiyonilac.value_counts()
tansiyondata=data[data.tansiyon=='var']

tansiyondata.tansiyonilac.value_counts()
sns.countplot(tansiyondata.tansiyonilac)
data.tenrengi.value_counts()
sns.countplot(data.tenrengi)
data[data.tenrengi=='yok']
data.uykusuresi.value_counts()
sns.boxplot(x= data.uykusuresi)
sns.countplot(data.uykusuresi)
data.yuruyus.value_counts()
sns.countplot(data.yuruyus)
data.yuruyuskm.value_counts()
sns.countplot(data.yuruyuskm)
data.describe().T
data.groupby(data.yas).mean()
sns.jointplot(x='ayaktakalmasure', y='cay', data=data, kind='kde')
sns.lmplot(x='cay', y='ayaktakalmasure', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')

sns.lmplot(x='ayaktakalmasure', y='cay', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')

sns.lmplot(x='ayaktakalmasure', y='cay', hue='ajerjik', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='cay', y='ayaktakalmasure', hue='araogun', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')

sns.lmplot(x='cay', y='ayaktakalmasure', col='ayakkabi', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')

sns.lmplot(x='ayaktakalmasure', y='cay', hue='ayakkabi', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')

sns.lmplot(x='cay', y='ayaktakalmasure', col='beslenmesekli', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')

sns.lmplot(x='ayaktakalmasure', y='cay', hue='cinsiyet', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')

sns.lmplot(x='ayaktakalmasure', y='cay', col='ilacalerjisi', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')

sns.lmplot(x='cay', y='ayaktakalmasure', col='kalp', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')

sns.lmplot(x='cay', y='ayaktakalmasure', col='kangrubu', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')

sns.lmplot(x='ayaktakalmasure', y='cay', hue='ogle', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')

sns.lmplot(x='cay', y='ayaktakalmasure', col='psikolojikdurum', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')

sns.lmplot(x='ayaktakalmasure', y='cay', col='romatizma', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')

sns.lmplot(x='cay', y='ayaktakalmasure', hue='seker', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')

sns.lmplot(x='ayaktakalmasure', y='cay', col='spor', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')

sns.lmplot(x='ayaktakalmasure', y='cay', hue='tansiyon', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')

sns.lmplot(x='ayaktakalmasure', y='cay', col='yuruyus', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.jointplot(x='boyu', y='yas', data=data, kind='kde')
sns.lmplot(x='yas', y='boyu', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='yas', y='boyu', hue='ajerjik', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='boyu', y='yas', col='araogun', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='boyu', y='yas', col='ayakkabi', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='boyu', y='yas', col='beslenmesekli', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='yas', y='boyu', hue='cinsiyet', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='boyu', y='yas', col='ilacalerjisi', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='boyu', y='yas', hue='kalp', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='boyu', y='yas', col='kangrubu', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='boyu', y='yas', col='ogle', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='boyu', y='yas', col='psikolojikdurum', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='yas', y='boyu', col='romatizma', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='boyu', y='yas', hue='seker', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='boyu', y='yas', col='spor', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='boyu', y='yas', hue='tansiyon', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='yas', y='boyu', hue='yuruyus', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.jointplot(x='boyu', y='kilo', data=data, kind='kde')
sns.lmplot(x='boyu', y='kilo', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='kilo', y='boyu', col='ajerjik', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='kilo', y='boyu', hue='araogun', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='boyu', y='kilo', col='ayakkabi', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='boyu', y='kilo', col='beslenmesekli', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='kilo', y='boyu', hue='cinsiyet', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='boyu', y='kilo', col='ilacalerjisi', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='boyu', y='kilo', col='kalp', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='boyu', y='kilo', col='kangrubu', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='kilo', y='boyu', hue='ogle', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='boyu', y='kilo', hue='ogle', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='kilo', y='boyu', col='psikolojikdurum', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='boyu', y='kilo', hue='romatizma', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='kilo', y='boyu', hue='romatizma', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='boyu', y='kilo', hue='seker', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='kilo', y='boyu', hue='spor', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='boyu', y='kilo', hue='tansiyon', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='kilo', y='boyu', hue='yuruyus', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.jointplot(x='yas', y='ameliyatsayisi', data=data, kind='kde')
sns.lmplot(x='yas', y='ameliyatsayisi', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='yas', y='ameliyatsayisi', col='ajerjik', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='yas', y='ameliyatsayisi', col='araogun', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='yas', y='ameliyatsayisi', col='ayakkabi', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='yas', y='ameliyatsayisi', col='beslenmesekli', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='yas', y='ameliyatsayisi', col='cinsiyet', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='yas', y='ameliyatsayisi', col='gozrengi', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='yas', y='ameliyatsayisi', col='ilacalerjisi', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='ameliyatsayisi', y='yas', hue='kalp', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='ameliyatsayisi', y='yas', col='kangrubu', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='ameliyatsayisi', y='yas', hue='ogle', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='yas', y='ameliyatsayisi', col='psikolojikdurum', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='yas', y='ameliyatsayisi', col='romatizma', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='yas', y='ameliyatsayisi', col='seker', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='yas', y='ameliyatsayisi', col='spor', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='yas', y='ameliyatsayisi', hue='tansiyon', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='yas', y='ameliyatsayisi', hue='yuruyus', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.jointplot(x='kilo', y='su', data=data, kind='kde')
sns.lmplot(x='su', y='kilo', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='su', y='kilo', hue='ajerjik', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='kilo', y='su', hue='araogun', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='su', y='kilo', col='ayakkabi', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='kilo', y='su', col='beslenmesekli', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='su', y='kilo', hue='cinsiyet', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='kilo', y='su', col='gozrengi', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='su', y='kilo', hue='ilacalerjisi', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='kilo', y='su', hue='kalp', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='kilo', y='su', hue='ogle', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='kilo', y='su', col='psikolojikdurum', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='kilo', y='su', col='romatizma', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='kilo', y='su', col='sacrengi', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='kilo', y='su', hue='seker', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='kilo', y='su', col='spor', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='kilo', y='su', hue='tansiyon', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='kilo', y='su', hue='yuruyus', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.jointplot(x='yas', y='uykusuresi', data=data, kind='kde')
sns.lmplot(x='yas', y='uykusuresi', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.jointplot(x='uykusuresi', y='ameliyatsayisi', data=data, kind='kde')
sns.lmplot(x='uykusuresi', y='ameliyatsayisi', hue='ajerjik', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='uykusuresi', y='ameliyatsayisi', hue='araogun', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='uykusuresi', y='ameliyatsayisi', col='ayakkabi', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='uykusuresi', y='ameliyatsayisi', col='beslenmesekli', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='uykusuresi', y='ameliyatsayisi', hue='cinsiyet', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='uykusuresi', y='ameliyatsayisi', col='gozrengi', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='uykusuresi', y='ameliyatsayisi', hue='ilacalerjisi', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='uykusuresi', y='ameliyatsayisi', col='kalp', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='uykusuresi', y='ameliyatsayisi', col='kangrubu', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='ameliyatsayisi', y='uykusuresi', hue='ogle', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='ameliyatsayisi', y='uykusuresi', col='psikolojikdurum', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='uykusuresi', y='ameliyatsayisi', hue='romatizma', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='uykusuresi', y='ameliyatsayisi', col='sacrengi', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='ameliyatsayisi', y='uykusuresi', hue='seker', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='uykusuresi', y='ameliyatsayisi', col='spor', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='uykusuresi', y='ameliyatsayisi', hue='tansiyon', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
sns.lmplot(x='uykusuresi', y='ameliyatsayisi', hue='yuruyus', data=data,

           fit_reg=True, aspect=1.25, palette='Accent')
corr = data.corr()

corr
sns.heatmap(corr,xticklabels=corr.columns.values,yticklabels=corr.columns.values)