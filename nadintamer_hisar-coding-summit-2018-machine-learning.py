# Kaggle'a önceden yüklü olarak gelen python kütüphanelerini aşağıdaki linkten inceleyebilirsiniz:
# https://github.com/kaggle/docker-python

#data analiz kütüphaneleri 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

#data görselleştirme kütüphaneleri
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

# "../input/" directory'mizdeki data kaynaklarına bakalım. 
# Atölye boyunca birçok farklı dataset'ten yararlanacağız.
#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))
# mısır gevrekleriyle ilgili dataset'imizi okuyarak cereal adlı bir değişkende depoluyoruz
cereal = pd.read_csv('../input/80-cereals/cereal.csv')
# head() fonksiyonunu kullanarak dataset'in ilk dört sırasına bakıyoruz
cereal.head(4)
#describe() fonksiyonuyla dataset'in özetine bakıyoruz
cereal.describe(include="all")
# "../input/starbucks-menu/starbucks_drinkMenu_expanded.csv" uzantılı dataset'i okuyup
# starbucks adlı bir değişkende depolayın. sonra head fonksiyonunu kullanarak ilk altı sırasına bakın.


# describe() metotunu kullanarak starbucks dataset'inin bir özetine bakın.
# tüm sütunları görmek için include="all" demeyi unutmayın!

cereal.head()
cereal.dtypes
# starbucks dataset'ine dtypes metotunu uygulayarak data tiplerinin tahminlerinize uygun olup olmadığına bakın!

cereal.shape
# starbucks dataset'ine shape metotunu uygulayın

# columns metotunu kullanarak cereal dataset'inin sütun adlarını inceliyoruz
cereal.columns
# columns metotuyla cereal dataset'indeki sütunların adlarını değiştiriyoruz (potass ve mfr sütunlarının)
# listedeki isimlerin sırasına dikkat edin!
cereal.columns = ['name', 'manufacturer', 'type', 'calories', 'protein', 'fat', 'sodium', 'fiber',
       'carbo', 'sugars', 'potassium', 'vitamins', 'shelf', 'weight', 'cups',
       'rating']
cereal.head()
# columns metotuyla starbucks dataset'inin sütun isimlerine bakın

# columns metotuyla starbucks dataset'inin sütun isimlerini yukarıdaki listeyle değiştirin
# yeni sütun isimlerine bakmak için head() fonksiyonunu kullanın

# dot notation kullanarak cereal dataset'inin calories sütununu alıyoruz
# çok uzun olmaması için head fonksiyonuyla ilk 10 değeri alıyoruz
cereal.calories.head(10)
# bracket notation kullanarak cereal dataset'inin fiber sütununu alıyoruz
# çok uzun olmaması için head fonksiyonuyla ilk 10 değeri alıyoruz
cereal['fiber'].head(10)
# dot notation kullanarak starbucks dataset'inin calories sütununu alın
# çok uzun olmaması için head fonksiyonuyla ilk 10 değeri alın

# bracket notation kullanarak starbucks dataset'inin sodium sütununu alın
# çok uzun olmaması için head fonksiyonuyla ilk 10 değeri alın

# cereal'ın sadece fiber'ın 8'den büyük olduğu sıralarına bakıyoruz
cereal[cereal['fiber'] >= 8]
# conditional selection ile trans_fat'in >= 1 olduğu sıralara bakın

sns.set_style("darkgrid") # bu satır, seaborn'un kullanacağı stili belirliyor, grafiklerimize çok bir etkisi yok.
# describe() metotundaki unique sırasını hatırlayın! kaç farklı kategorik özellik olduğunu sayıyordu.
# örneğin, cereal dataset'imizde yedi farklı mısır gevreği üreticisi var. 
# countplot, her farklı özelliğin dataset'te kaç kere bulunduğunu göstere bir grafik çizer.
sns.countplot(x='manufacturer',data=cereal)
plt.xlabel('Mısır Gevreği Üreticisi', fontsize=12) # x ekseninin etiketini ve etiketin font büyüklüğünü ayarlıyoruz
plt.ylabel('Mısır Gevreği Sayısı', fontsize=12) # y ekseninin etiketini ve etiketin font büyüklüğünü ayarlıyoruz
plt.title('Hangi Üretici Ne Kadar Mısır Gevreği Üretiyor?', fontsize=14) # grafiğimizin başlığını koyuyoruz
plt.show() # grafiği gösteriyoruz
plt.xticks(rotation=90, fontsize=10) # bu satır, x eksenindeki tüm etiketleri 90 derece döndürüyor ki hepsi okunabilsin.

# seaborn'un countplot fonksiyonunu kullanarak beverage_category özelliği için bir countplot çizin!

# x eksenini etiketleyin

# y eksenini etiketleyin

# grafiğin başlığını koyun

# grafiği gösterin!
plt.show()
# barplot, x eksenindeki her kategori için y eksenindeki özelliğin (calories) ortalamasını hesaplar.
sns.barplot(x='manufacturer',y='rating',data=cereal)
plt.xlabel('Mısır Gevreği Üreticisi', fontsize=12) # x ekseninin etiketini ve etiketin font büyüklüğünü ayarlıyoruz
plt.ylabel('Ortalama Rating', fontsize=12) # y ekseninin etiketini ve etiketin font büyüklüğünü ayarlıyoruz
plt.title('Hangi Üreticinin Rating\'leri En Yüksek?', fontsize=14) # grafiğimizin başlığını koyuyoruz 
plt.show() # grafiğimizi gösteriyoruz
plt.xticks(rotation=90, fontsize=10) # bu satır, x eksenindeki tüm etiketleri 90 derece döndürüyor ki hepsi okunabilsin.

# seaborn'un barplot fonksiyonunu kullanarak beverage_category ve calories özellikleri için bir barplot çizin!

# x eksenini etiketleyin

# y eksenini etiketleyin

# grafiğin başlığını koyun

# grafiği gösterin!
plt.show()
sodium = cereal['sodium'] # sodium sütununu alıyoruz
plt.hist(sodium, bins = 9, edgecolor = "black") # histogramı çiziyoruz
# bins kaç tane aralık olacağını belirliyor. edgecolor'la aralıkların kolay ayırt edilmesi için kenar çizgisi ekliyoruz.
plt.xlabel("Sodyum") # x eksenine etiket ekliyoruz
plt.title("Mısır Gevreklerinde Sodyum") # başlık ekliyoruz
plt.show() #grafiği gösteriyoruz
# total_carbs sütununu carbs adlı bir değişkende depolayın

# matplotlib'in hist fonksiyonunu kullanarak total_carbs sütunu için bir histogram çizin!
# parametre olarak bins = 9 ve edgecolor = 'black' kullanın

# x eksenini etiketleyin

# grafiğin başlığını koyun

# grafiği gösterin!
plt.show()
sns.distplot(sodium) # sodium için distplot çiziyoruz
plt.xlabel("Sodyum") # x eksenine etiket ekliyoruz
plt.title("Mısır Gevreklerinde Sodyum") # başlık ekliyoruz
plt.show() # grafiği gösteriyoruz
# seaborn'un distplot fonksiyonunu kullanarak total_carbs sütunu için bir distribution plot çizin!

# x eksenini etiketleyin

# grafiğin başlığını koyun

# grafiği gösterin!
plt.show()
sns.regplot(x='potassium',y='rating',data=cereal, line_kws={'color':'purple'}) # scatter plot'u çiziyoruz
plt.xlabel('Potasyum') # x eksenine etiket ekliyoruz
plt.ylabel('Rating') # y eksenine etiket ekliyoruz
plt.title('Mısır Gevreklerinde Potasyum ve Rating') # başlik ekliyoruz
plt.show() # grafiği gösteriyoruz
# seaborn'un regplot fonksiyonunu kullanarak cholesterol ve sugars sütunları için bir scatter plot çizin!

# x eksenini etiketleyin

# y eksenini etiketleyin

# grafiğin başlığını koyun

# grafiği gösterin!
plt.show()
# bir correlation table görmek için corr() metotunu kullanabiliriz, ancak sonuç kolayca yorumlanamaz
cereal.corr()
# bu nedenle seaborn'un heatmap fonksiyonunu kullanarak bir correlation heatmap çizeriz
sns.heatmap(cereal.corr())
plt.title('Cereal Dataset\'i için Correlation Heatmap', fontsize=14)
plt.show()
# heatmap ve corr fonksiyonlarını kullanarak starbucks dataset'i için bir correlation heatmap çizin

# grafiğin başlığını ekleyin

# grafiği gösterin!
plt.show()
# meme kanseriyle ilgili verileri yüklüyoruz ve bir pandas dataset'ine çeviriyoruz
from sklearn import datasets
cancer = datasets.load_breast_cancer()
cancer.feature_names
cancerdf = pd.DataFrame(cancer.data, columns=cancer.feature_names)
cancerdf['type'] = cancer.target
cancerdf.head()
# seaborn'un heatmap fonksiyonunu kullanarak bir correlation heatmap çiziyoruz
# annot = True parametresiyle correlation değerlerinin heatmap üstünde gösterilmesini sağlıyoruz
sns.heatmap(cancerdf.corr(), annot=True)
# grafiğe başlık ekliyoruz
plt.title('Correlation Heatmap for Breast Cancer Dataset')
# grafiği gösteriyoruz
plt.show()
# DecisionTreeClassifier algoritmasını ve accuracy_score fonksiyonunu import ediyoruz
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# decisiontree adlı bir değişkende yeni bir DecisionTreeClassifier objesi tanımlıyoruz
decisiontree = DecisionTreeClassifier()

# cancerdf dataframe'inin ilk 50 sırasını test olarak, geriye kalan sıraları ise train olarak tanımlıyoruz
train = cancerdf[50:]
test = cancerdf[:50]

# x_train'i train'in type (yani tahmin etmeye çalıştığımız değerin) drop edilmiş hali olarak tanımlıyoruz
x_train = train.drop('type', axis=1)
# y_train'i train'in type sütunu olarak tanımlıyoruz
y_train = train['type']

# aynısını test için de yapıyoruz
x_test = test.drop('type', axis=1)
y_test = test['type']

# decisiontree adlı modelimizi x_train, y_train verilerine göre fit ediyoruz
decisiontree.fit(x_train, y_train)

# decisiontree modeline x_test hakkında tahmin yaptırıp pred değişkeninde saklıyoruz
pred = decisiontree.predict(x_test)

# accuracy_score fonksiyonuyla modelimizin doğruluğunu kontrol ediyoruz
print("accuracy:", accuracy_score(y_test, pred))
# "../input/the-ultimate-halloween-candy-power-ranking/candy-data.csv" uzantılı dataset'i okuyup 
# candy adlı bir değişkende depolayın

# head fonksiyonuyla ilk beş sırasına bakın
# dt adlı bir değişkende yeni bir DecisionTreeClassifier objesi tanımaylın

# test'i candy'nin ilk 25 sırası, train'iyse geri kalan tüm sıralar olacak şekilde tanımlayın



# x_train'i train'in ['chocolate', 'competitorname'] sütunlarının drop edilmiş hali yapın
# (competitorname sütununu sayısal bir özellikl olmadığından drop ediyoruz)

# y_train'i train'in chocolate sütunu olarak tanımlayın

# aynısını x_test ve y_test için de yapın


# dt modelini x_train, y_train verilerine göre fit edin

# dt'ye x_test göre tahmin yaptırıp preds adlı bir değişkende depolayın

# accuracy_score fonksiyonuyla modelinizin doğruluğunu kontrol edin

# starbucks menüsünü yükleyip starbucks adlı bir değişkende depoluyoruz
starbucks = pd.read_csv('../input/starbucks-menu/starbucks_drinkMenu_expanded.csv')

# mean_squared_error ve linear_model'ı import ediyoruz
from sklearn.metrics import mean_squared_error
from sklearn import linear_model

# train ve test setlerimizi tanımlıyoruz
sugar = starbucks[' Sugars (g)'].to_frame()
calories = starbucks['Calories'].to_frame()

sugar_train = sugar[20:]
calories_train = calories[20:]

sugar_test = sugar[:20]
calories_test = calories[:20]

# model adlı bir değişkende yeni bir LinearRegression objesi tanımlıyoruz
model = linear_model.LinearRegression()

# modelimizi sugar_train, calories_train verilerine göre fit ediyoruz
model.fit(sugar_train, calories_train)

# modele tahmin yaptırıp predictions adlı bir değişkende depoluyoruz
predictions = model.predict(sugar_test)

# çizdiğimiz doğrunun katsayısını ve mean squared error'ına bakıyoruz
print('Coefficients: \n', model.coef_)
print('Mean Squared Error:', mean_squared_error(calories_test, predictions))

# linear regresyon modelimizi görsel hale getiriyoruz
sns.regplot(x=' Sugars (g)', y='Calories', data=starbucks, fit_reg=False)
plt.plot(sugar, model.predict(sugar), color='red')
plt.title('Sugars vs.Calories')
plt.show()
# veriyi yükleyip ilk beş sırasına bakıyoruz
movies = pd.read_csv('../input/tmdb-movie-metadata/tmdb_5000_movies.csv')
genres = movies['genres']
movies.head()
# film türlerinin dağılımını ve aldıkları ortalama puanı incelemek için veriyi hazırlıyoruz
import json

def transformGenres(g):
    copy = genres.copy()
    x = 0
    for genreList in genres:
        genreList = json.loads(genreList)
        genreListNew = []
        for y in genreList:
            genreListNew.append(y['name'])
        copy[x] = genreListNew
        x += 1
    return copy

newGenres = transformGenres(genres)
movies['genres'] = newGenres

def uniqueGenres(genreList):
    unique = []
    for movie in genreList:
        for genre in movie:
            if [genre, 0, 0] not in unique:
                unique.append([genre, 0, 0])
    return unique

unique = uniqueGenres(movies['genres'])

for x in unique:
    for y in range(movies.shape[0]):
        if x[0] in movies.iloc[y]['genres']:
            x[1] += 1
            x[2] += movies.iloc[y]['vote_average']
    x[2] /= x[1]
            
uniquedf = pd.DataFrame(unique, columns=['genre','count', 'average_rating'])
uniquedf.head(10)
# seaborn'un barplot fonksiyonunu kullanarak uniquedf'in genre ve count sütunları için bir grafik çizin

# grafiği x etiketlerini çevirip grafiği gösteriyoruz
plt.xticks(rotation=90)
plt.show()
# seaborn'un barplot fonksiyonunu kullanarak uniquedf'in genre ve average_rating sütunları için bir grafik çizin

# grafiği x etiketlerini çevirip grafiği gösteriyoruz
plt.xticks(rotation=90)
plt.show()
# seaborn'un regplot fonksiyonunu kullanarak movies'in budget ve revenue sütunları için bir grafik çizin

# grafiği gösterin
plt.show()
# "../input/top-tracks-of-2017/featuresdf.csv" uzantılı dataset'i okuyup spotify adlı bir değişkende depolayalım

# head fonksiyonuyla ilk beş sırasına bakalım

# describe fonksiyonuyla dataset'in bir özetine bakalım (include='all')

# figure büyüklüğüne karar verelim
sns.set(rc={'figure.figsize':(20,13)})

# Danceability
plt.subplot(421)
# SİZİN KODUNUZ AŞAĞIYA

plt.xlabel('DANCEABILITY', fontsize=12)
plt.legend(fontsize=12)

# Energy
plt.subplot(422)
# SİZİN KODUNUZ AŞAĞIYA

plt.xlabel('ENERGY', fontsize=13)
plt.legend(fontsize=13)

# Mode
plt.subplot(423)
# SİZİN KODUNUZ AŞAĞIYA

plt.xlabel('MODE', fontsize=13)
plt.legend(fontsize=13)

# Speechiness
plt.subplot(424)
# SİZİN KODUNUZ AŞAĞIYA

plt.xlabel('SPEECHINESS', fontsize=13)
plt.legend(fontsize=13)

# Acousticness
plt.subplot(425)
# SİZİN KODUNUZ AŞAĞIYA

plt.xlabel('ACOUSTICNESS', fontsize=13)
plt.legend(fontsize=13)

# Instrumentalness
plt.subplot(426)
# SİZİN KODUNUZ AŞAĞIYA

plt.xlabel('INSTRUMENTALNESS', fontsize=13)
plt.legend(fontsize=13)

# Liveness
plt.subplot(427)
# SİZİN KODUNUZ AŞAĞIYA

plt.xlabel('LIVENESS', fontsize=13)
plt.legend(fontsize=13)

# Valence
plt.subplot(428)
# SİZİN KODUNUZ AŞAĞIYA

plt.xlabel('VALENCE', fontsize=13)
plt.legend(fontsize=13)

plt.tight_layout()
plt.show()
# seaborn'un regplot fonksiyonunu kullanarak spotify'in loudness ve energy sütunları için bir grafik çizin

# grafiği gösterin
plt.show()
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

# loudness ve energy sütunlarını alarak birer dataframe'e çevirin



# ikisinin de ilk 10 sırasını loudness_test ve energy_test olarak depolayın



# ikisinin de kalan sıralarını loudness_train ve energy_train olarak depolayın



# spotifyModel adlı bir değişkende yeni bir LinearRegression objesi tanımlayın


# spotifyModel'i loudness_train, energy_train verileriyle fit edin


# spotifyModel'a loudness_test'e göre tahmin yaptırın ve tahminleri spotifyPred adlı bir değişkende saklayın


# spotifyModel'in katsayılarına ve mean squared error'ına bakın
print('Coefficients: \n', spotifyModel.coef_)
print('Mean Squared Error:', mean_squared_error(energy_test, spotifyPred))

# modelinizi görsel hale getirin
sns.regplot(x='loudness', y='energy', data=spotify, fit_reg=False, color='red')
plt.plot(loudness, spotifyModel.predict(loudness), color='green')
plt.title('Loudness vs. Energy')
plt.show()