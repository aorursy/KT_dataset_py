import seaborn as sns
import pandas as pd
import numpy as np
# week5-lecture4
#Random State- random urettigimiz sayilar sabit kalir
state=np.random.RandomState(10)
veri1=state.randint(1,10,size=5) # yukaridaki ile ayni blokta olmali!!
veri1
# her run ettigimizde farkli sayilar random edilir
veri2=np.random.randint(1,10,size=5)
veri2
# Katagorik Degisken ve Frekans Grafikleri

diamonds=sns.load_dataset('diamonds')
df=diamonds.copy()
df.head()
df.info()
df.cut.value_counts() #string degerler-katagorig degisken-kendi icinde siralama oldugundan ordinaldir
df.color.value_counts() #string degerler-katagorig degisken-kendi icinde siralama oldugundan ordinaldir
df_num = df.select_dtypes(include=['float64','int64'])
df_num
from pandas.api.types import CategoricalDtype
# cut degiskeninin typeni object ten category cevirmek icin import ettik CategoricalDtype ve astype metodu kullandik
df.cut=df.cut.astype(CategoricalDtype(ordered=True))
df.dtypes
df.cut.head() # siralamasini gorduk bizim istedimiz gibi deil
cut_catog=['Fairy','Good','Very Good','Premium','Ideal']
df.cut=df.cut.astype(CategoricalDtype(categories=cut_catog ,ordered=True))
df.cut.head() # simdi istedigimiz gibi catogoriledik
color_catog=['D','E','F','G','H','I','J']
df.color=df.color.astype(CategoricalDtype(categories=color_catog ,ordered=True))
df.color.head() # simdi color u daistedigimiz gibi catogoriledik
# pandas ile gorsellestirme 

df['cut'].value_counts().plot.barh() 
# ; yaparsak sonuna <matplotlib.axes._subplots.AxesSubplot at 0x122fa8940> ortadan kalkar
df['cut'].value_counts().plot.barh();
# set_title metodu ile grfige isim verebilirz. alt alta kisaltmis version icin () kullan
(df['cut']
.value_counts()
.plot.barh()
.set_title('CUT GRAFIK'));
# seaborn ile gorsellestrime
sns.barplot(x='cut',y=df.cut.index, data=df);
sns.catplot(x='cut',y='price',data=df); 
# 3.boyut ekleyelim color olarak
sns.barplot(x='cut',y='price',hue='color',data=df);
df.groupby(['cut','color'])['price'].mean()
# Merkezi Dagilim Olculeri
# Mean, Median ve Mod


veri3=np.random.randint(1,100,20)
veri3
np.mean(veri3) # meandan yuksek olmasi aykiri degerlerin oldugunu gosterir 96 gibi
np.median(veri3)
from scipy import stats # np de mod cok kullanilmadigi icin yok.stats i ondan import ettik
stats.mode(veri3) # en cok tetrar eden 22 ondan da 2 tane var
# Uygulama
# A ve B şirketinin 20 aylık gelirleri bilinmektedir. Merkezi eğilim ölçülerine göre, bu şirketlerin 20 aylık gelireri baz alındığında geliri yüksek olan şirket belirlenmek istenmektedir.

A = np.array([39667, 49651, 34014, 33109, 38746, 33037, 38886, 36205, 47191, 44696, 42030, 38918, 45726, 34036, 41673, 42319, 65034,95000,98000,87900])

B = np.array([46278, 55157, 51654, 40634, 46878, 42816, 55754, 49209, 43619, 58025, 51963, 47610, 50448, 56156, 57669, 54364, 41339, 52887, 40470, 42079])

np.mean(A),np.mean(B)
np.median(A),np.median(B)

# week5-lecture5

# MERKEZI YAYILIM OLCUTLERI
 # Range(Aciklik)
  #Inter Quartile Range-ceyrek acikligi
    # Standard Deviation -satndart sapma
    # Variance
np.std(veri3)    # STANDAD SAPMA
np.var(veri3) # VARIANCE
veri4=pd.DataFrame(veri3) # np den pd ya gectik df cevirerek
veri4.describe().T
desc=veri4[0].describe()
Q1=desc[4]
Q3=desc[6]
Q1,Q3
# ceyrekler acikligi
IQR=Q3-Q1
IQR
alt_sinir=Q1-1.5*IQR
ust_sinir=Q3+1.5*IQR
alt_sinir,ust_sinir # bu sinirlara gore eger disinda kalan veri varsa istersek cikarabiliriz istatistik anlamli hale gelsin diye
 # Kutu Grafikleri
import seaborn as sns
tips = sns.load_dataset("tips")
df = tips.copy()
df.head()
df.describe().T
df['sex'].value_counts() # erkek agirlikli kitle
df['smoker'].value_counts() # cogu sigara icmeyen kitle
df['day'].value_counts() # haftasonu musterisi daha cok
df['time'].value_counts() # aksam yemegidende msuteisi cok
#boxplot-gorsellestirme grafik
sns.boxplot(x=df['total_bill']); # 40 dan  sonrasi noktalar out-layer(ust sinir)
sns.boxplot(x='day',y='total_bill',data=df); # en karli gun sun
sns.boxplot(x='size',y='total_bill',data=df); # kisi sayisi arttikca para artmis
# violin grafik
sns.catplot(x='total_bill',kind='violin',data=df); # beyaz nokta median
sns.catplot(x='day', y='total_bill',kind='violin',hue='sex',data=df);
# scatter plot-serpme grafigi
sns.scatterplot(x='total_bill',y='tip',data=df); # fatura arttikca bahsiste genelde artmis
sns.catplot(x='total_bill',y='tip',hue='time',data=df);
# dogrusal 
import matplotlib.pyplot as plt
sns.lmplot(x='total_bill',y='tip',data=df);
# corralation- iki veri arasindaki iliski -1 1 arasi deger
plt.figure(figsize=(5,5));
sns.heatmap(df.corr(),annot=True,cmap='coolwarm')
df.corr()    
