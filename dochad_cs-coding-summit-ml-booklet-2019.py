# Kaggle'a önceden yüklü olarak gelen python kütüphanelerini aşağıdaki linkten inceleyebilirsiniz:

# https://github.com/kaggle/docker-python



#data analiz kütüphaneleri 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import os



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

print(os.listdir("../input/"))
pokemonDS = pd.read_csv("../input/pokemon/Pokemon.csv")

pokemonDS.head()
pokemonDS.describe(include="all")
# "../input/starbucks-menu/starbucks_drinkMenu_expanded.csv" uzantılı dataset'i okuyup

# starbucks adlı bir değişkende depolayın. sonra head fonksiyonunu kullanarak ilk altı sırasına bakın.
# describe() metotunu kullanarak starbucks dataset'inin bir özetine bakın.

# tüm sütunları görmek için include"all" demeyi unutmayın!
pokemonDS.head()
pokemonDS.dtypes
# starbucks dataset'ine dtypes metotunu uygulayarak data tiplerinin tahminlerinize uygun olup olmadığına bakın!
pokemonDS.shape
# starbucks dataset'ine shape metotunu uygulayın
pokemonDS.columns
pokemonDS.columns = ['No', 'Name', 'Type1', 'Type 2', 'TotalStats',

       'HP ', 'Atk', 'Def', 'Sp. Atk','Sp. Def', 'Spd', 'Gen', 'Legendary']

pokemonDS.head()
# columns metotuyla starbucks dataset'inin sütun isimlerine bakın
# dot notation kullanarak cereal dataset'inin calories sütununu alıyoruz

# çok uzun olmaması için head fonksiyonuyla ilk 10 değeri alıyoruz

pokemonDS.GRE.head(10)
# bracket notation kullanarak cereal dataset'inin isim sütununu alıyoruz

# çok uzun olmaması için head fonksiyonuyla ilk 10 değeri alıyoruz

pokemonDS['Name'].head(10)
# dot notation kullanarak starbucks dataset'inin calories sütununu alın

# çok uzun olmaması için head fonksiyonuyla ilk 10 değeri alın.
# bracket notation kullanarak starbucks dataset'inin CGPA sütununu alın

# çok uzun olmaması için head fonksiyonuyla ilk 10 değeri alın
# Öğrencilerin sadece CGPA'ın 8'den büyük olduğu sıralarına bakıyoruz

pokemonDS[pokemonDS['Atk'] >= 160]
# conditional selection ile trans_fat'in >= 1 olduğu sıralara bakın
sns.set_style("darkgrid") # bu satır, seaborn'un kullanacağı stili belirliyor, grafiklerimize çok bir etkisi yok.
# describe() metotundaki unique sırasını hatırlayın! kaç farklı kategorik özellik olduğunu sayıyordu.

# örneğin, Pokemon dataset'imizde on sekiz farklı pokemon tipi var. 

# countplot, her farklı özelliğin dataset'te kaç kere bulunduğunu göstere bir grafik çizer.

plt.figure(figsize=(16,6))

sns.countplot(x='Type1', data=pokemonDS)

plt.xlabel('Tipler', fontsize=12) # x ekseninin etiketini ve etiketin font büyüklüğünü ayarlıyoruz

plt.ylabel("Pokemon Sayısı", fontsize=12) # y ekseninin etiketini ve etiketin font büyüklüğünü ayarlıyoruz

plt.title('Hangi tipte daha çok Pokemon var?', fontsize=16) # grafiğimizin başlığını koyuyoruz

plt.show() # grafiği gösteriyoruz
plt.xticks(rotation=90, fontsize=10) # bu satır, x eksenindeki tüm etiketleri 90 derece döndürüyor ki hepsi okunabilsin.



# seaborn'un barplot fonksiyonunu kullanarak beverage_category ve calories özellikleri için bir barplot çizin!



# x eksenini etiketleyin



# y eksenini etiketleyin



# grafiğin başlığını koyun



# grafiği gösterin!

plt.show()
AtkVal = pokemonDS['Atk'] # Atk sütununu alıyoruz

plt.hist(AtkVal, bins = 9, edgecolor = "black") # histogramı çiziyoruz

# bins kaç tane aralık olacağını belirliyor. edgecolor'la aralıkların kolay ayırt edilmesi için kenar çizgisi ekliyoruz.

plt.xlabel("Saldırı gücü",fontsize=16) # x eksenine etiket ekliyoruz

plt.title("Saldırı Gücü Dağılımları") # başlık ekliyoruz

plt.show() #grafiği gösteriyoruz
# total_carbs sütununu carbs adlı bir değişkende depolayın



# matplotlib'in hist fonksiyonunu kullanarak total_carbs sütunu için bir histogram çizin!

# parametre olarak bins = 9 ve edgecolor = 'black' kullanın



# x eksenini etiketleyin



# grafiğin başlığını koyun



# grafiği gösterin!

plt.show()
sns.distplot(AtkVal) # Saldırı güçleri için distplot çiziyoruz

plt.xlabel("Saldırı Gücü") # x eksenine etiket ekliyoruz

plt.title("Saldırı Gücü Dağılımları")# başlık ekliyoruz

plt.show() # grafiği gösteriyoruz
# seaborn'un distplot fonksiyonunu kullanarak total_carbs sütunu için bir distribution plot çizin!



# x eksenini etiketleyin



# grafiğin başlığını koyun



# grafiği gösterin!

plt.show()
# seaborn'un regplot fonksiyonunu kullanarak cholesterol ve sugars sütunları için bir scatter plot çizin!



# x eksenini etiketleyin



# y eksenini etiketleyin



# grafiğin başlığını koyun



# grafiği gösterin!

plt.show()
Type_mapping = {"Grass": 1, "Fire": 2, "Water": 3, "Bug": 4, "Poison": 5, "Electric": 6,"Ground": 7, "Fairy": 8, "Fighting": 9, "Psychic": 10, "Rock": 11, "Ghost": 12, "Ice": 13, "Dragon": 14, "Dark": 15, "Steel": 16, "Flying": 17}

NewTypes = pokemonDS

NewTypes["Type1"] = NewTypes["Type1"].map(Type_mapping)

plt.figure(figsize=(16,6))

sns.regplot(x='Type1' ,y='Atk',data=NewTypes, line_kws={'color':'purple'}) # scatter plot'u çiziyoruz

plt.xlabel('Tipler') # x eksenine etiket ekliyoruz

plt.ylabel('Saldırı') # y eksenine etiket ekliyoruz

plt.title('Tip/Saldırı ilişkisi') # başlik ekliyoruz

plt.show() # grafiği gösteriyoruz
# bir correlation table görmek için corr() metotunu kullanabiliriz, ancak sonuç kolayca yorumlanamaz

pokemonDS.corr()
sns.heatmap(pokemonDS.corr())

plt.title('UniversityDataset\'i için Correlation Heatmap', fontsize=14)

plt.show()
# heatmap ve corr fonksiyonlarını kullanarak starbucks dataset'i için bir correlation heatmap çizin



# grafiğin başlığını ekleyin



# grafiği gösterin!

plt.show()
from sklearn import datasets

wine = datasets.wine()

wine.feature_names

wineDF = pd.DataFrame(wine.data, columns=wine.feature_names)

wineDF['type'] = wine.target

wineDF.head()
# seaborn'un heatmap fonksiyonunu kullanarak bir correlation heatmap çiziyoruz

# annot = True parametresiyle correlation değerlerinin heatmap üstünde gösterilmesini sağlıyoruz

sns.heatmap(wineDF.corr(), annot=True)

# grafiğe başlık ekliyoruz

plt.title('Correlation Heatmap for Wine Dataset')

# grafiği gösteriyoruz

plt.show()
# DecisionTreeClassifier algoritmasını ve accuracy_score fonksiyonunu import ediyoruz

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score



# decisiontree adlı bir değişkende yeni bir DecisionTreeClassifier objesi tanımlıyoruz

decisiontree = DecisionTreeClassifier()



# cancerdf dataframe'inin ilk 50 sırasını test olarak, geriye kalan sıraları ise train olarak tanımlıyoruz

wineTrainX = wineDF.drop("type",axis=1)

wineTrainY = wineDF["type"]

train_X,test_X,train_Y,test_Y = train_test_split(wineTrainX,wineTrainY,test_size = 0.3, random_state=3)



# decisiontree adlı modelimizi x_train, y_train verilerine göre fit ediyoruz

decisiontree.fit(train_X, train_Y)



# decisiontree modeline x_test hakkında tahmin yaptırıp pred değişkeninde saklıyoruz

pred = decisiontree.predict(test_X)



# accuracy_score fonksiyonuyla modelimizin doğruluğunu kontrol ediyoruz

print("accuracy:", accuracy_score(test_Y, pred))