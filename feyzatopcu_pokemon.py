# Bu python 3 ortamı bir çok yardımcı analiz kütüphaneleri ile birlikte kurulu olarak gelmektedir.
# Docker image'i olarak kaggle/python'dadır.(https://github.com/kaggle/docker-python)
# Örneğin, yüklenecek birkaç yardımcı paket var


import numpy as np # cebir
import pandas as pd # veri işleme, CSV dosyaları I/O (örn. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # görselleştirme aracı
import matplotlib.pyplot as plt #visualization

# Giriş veri dosyaları "../input/" dizinindedir.
# Örneğin bu hücreyi çalıştırmak için Shift+Enter'a aynı anda basarsanız hücre çalışır ve o dizindeki dosyaları sıralar.
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Geçerli dizine yazdığınız herhangi bir sonuç çıktı olarak kaydedilir.
veri = pd.read_csv('../input/pokemon.csv')# verimizi import ediyoruz.

pokemon_df = pd.read_csv('../input/pokemon.csv')
combats_df = pd.read_csv('../input/combats.csv')
test_df = pd.read_csv('../input/tests.csv')
prediction_df = test_df.copy()
#bellek kullanımı ve veri türleri
veri.head(5)
veri.info()
veri.corr()
# Korelasyon haritası
f,ax = plt.subplots(figsize=(20,20))
sns.heatmap(veri.corr(), annot=True, linewidths=.8, fmt= '.2f',ax=ax)
plt.show()

# Histogram
# bins = şekildeki çubuk sayısı
veri.Speed.plot(kind = 'hist',bins = 40,figsize = (10,10))
plt.show()
veri.tail() # tail son 5 satırı gösterir.
#columns sütın isimlerini verir.
veri.columns
#shape satıe ve sütunların sayısını verir.
veri.shape
# Çizgi Grafiği
veri.Speed.plot(kind = 'line', color = 'g',label = 'Generation',linewidth=2,alpha = 0.8,grid = True,linestyle = ':')
veri.Defense.plot(color = 'r',label = 'Legendary',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = etiketi grafiğe koyar
plt.xlabel('x ekseni')              # label = etiket adı
plt.ylabel('y ekseni')
plt.title('Grafiğin Başlığı')            # title = grafiğin başlığı
plt.show()
veri.isnull().sum()
veri.describe() # null girdileri görmezden gelir
veri ['Name'] = veri['Name'].fillna('Boş')
sns.boxplot(x=veri['HP']) # uç değer kontrolleri
veri ['Name']= veri ['Name'].fillna('Boş')
#Mevcut özniteliklerden yeni bir öznitelik oluşturma
def guc_durumu (Attack):
    return (Attack >=80)

veri [' POWER '] = veri ['Attack'].apply(guc_durumu)
#Veri Normalleştirme
from sklearn import preprocessing
#Puan özniteliğini normalleştirmek istiyoruz.
x= veri [['Attack']].values.astype(float)
#Normalleştirme için MinMax normalleştirme yöntemi kullanıyoruz.
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
veri['Attack2']=pd.DataFrame(x_scaled)
veri
pokemon_df12 = pd.read_csv('../input/pokemon.csv')
pokemon_df12.drop( ['Attack','Name'], axis=1 , inplace=True )
print(pokemon_df12)
#test_data = test_data.drop(["Cabin_type_Unknown"], axis=1)

#MODELLEŞTİRME
veri.dropna(axis=0, how='any')
# Veri kümesini Eğitim seti ve Test kümesine ayırdık
X = veri.iloc[:, 5:11].values
y = veri.iloc[:, 11].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#Kullanacağımız modeller için kullanacağımız kütüphaneler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
#Modellerin eğitilmesi
models = []
models.append(('Naive Bayes', GaussianNB()))
models.append(('Logistic Regression', LogisticRegression()))
combats = pd.read_csv('../input/combats.csv')
combats.head(3)
pokemon = pd.read_csv('../input/pokemon.csv')
pokemon_266_298 = pokemon[pokemon['#'].isin([266, 298])]
pokemon_266_298

names_dict = dict(zip(pokemon['#'], pokemon["Name"]))

cols = ["First_pokemon","Second_pokemon","Winner"]
combats_name = combats[cols].replace(names_dict)
combats_name.head(3)
import numpy as np
from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image
combats = pd.read_csv('../input/combats.csv')
combats.head(3)

combats_names = combats[cols].replace(names_dict)
print (combats_names["Winner"].value_counts()[:10])
winners = list(combats_names["Winner"])
winners_str = [str(i) for i in winners]
winners_text = (",").join(winners_str)
wc = WordCloud(background_color= "black", random_state=1, margin=3).generate(winners_text)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#Machine Learning tools
#Önişleme 
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.preprocessing import StandardScaler

#Model Seçimi
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

#Makine Öğrenmesi Modelleri
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

#Metrikler
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report,roc_curve, auc

#Sistem Kütüphaneleri
import os
print(os.listdir("../input"))
import warnings
#Sonuçların okunmasını zorlaştırdığı için uyarıları kapatıyoruz
warnings.filterwarnings("ignore")
print("Uyarılar Kapatıldı")
#'Veri' sütünun adı 'target' olarak değitiriliyor
veri.rename(columns={'veri':'target'},inplace=True)
veri.head()
#creating battles dataframe
name_dict = dict(zip(pokemon_df['#'], pokemon_df['Name']))
combats_name_df = combats_df[['First_pokemon', 'Second_pokemon', 'Winner']].replace(name_dict)
print(combats_name_df.head())
first_battle = combats_name_df['First_pokemon'].value_counts()
second_battle = combats_name_df['Second_pokemon'].value_counts()
win_counts = combats_name_df['Winner'].value_counts()
total_battle = first_battle + second_battle
win_percentage = win_counts / total_battle

win_percentage = win_percentage.sort_values()
print(win_percentage.head(10))
print(pokemon_df['Generation'].value_counts())

sns.countplot(x='Generation', data=pokemon_df, order=pokemon_df['Generation'].value_counts().index)
plt.show()
print(pokemon_df['Legendary'].value_counts())

sns.countplot(x='Legendary', data=pokemon_df, order=pokemon_df['Legendary'].value_counts().index)
plt.show()
