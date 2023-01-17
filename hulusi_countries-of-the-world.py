import numpy as np # numerik değerler kütüphanesi
import pandas as pd #string değerler kütüphanesi, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #data görselleştirme kütüphanesi
import seaborn as sns  # visualization tool
#Verilerimizi csv dosyasından alıyoruz 
data = pd.read_csv('../input/countries of the world.csv')
#İnfo kısmı bize verimiz hakkında genel(eksik verilerden data boyutuna kadar) bilgileri verir.
data.info()
data = pd.read_csv('../input/countries of the world.csv',decimal=',')
data.info()  #Eskiden object olan sayısal değerlerimiz şimdi float64 tipine dönüştürüldü
data.shape #Datada toplam ne kadar satır ve sütun sayısı olduğunu gösterir
data.columns #datamızın sütun isimlerinde uygunsuz yazılar,boşluklar,karakter tipinde değerler vs. 
                #var ise belirlenmesi için bilgi sahibi oluruz.
data.isnull().sum() #Datamız içerisinde tanımlanmamış değerlerimizin sayısını görebiliyoruz
data.isnull().sum().sum() #Datamızda toplamda kaç tane eksik değer olduğunu görürüz
data.head() #Default olarak datanın ilk 5 satırını verir
data.tail() # Datanın son 5 satırını verir
#Describe fonksiyonu bize tablomuzdaki sayısal değerlerin mod,medyan,standart sapma,max min gibi değerlerine ulaşmamızı sağlar
data.describe()
data['Region'].value_counts() #toplam 227 tane verimizden aynı isimli kaç tane olduğu bilgisini verir
#Yani bölgelerdeki toplam ülke sayılarına ve/veya aykırı isimde bir feature var olup olmadığı bilgisine ulaştık
'''Pie Chart'''
explode = (0, 0, 0, 0,0,0.1,0) 
sizes=[12.33,5.28,2.64,9.25,12.33,22.46,2.20]
labels="ASIA","EASTERN EUROPE","NORTHERN AFRICA","OCEANIA","WESTERN EUROPE","SUB-SAHARAN AFRICA","NORTHERN AMERICA"
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels,explode=explode, autopct='%1.1f%%', shadow=True, startangle=90)
ax1.axis('equal')  # Eşit en boy oranı daire şeklinde çizilmesini sağlar.
plt.show()
region = data["Region"].value_counts()
plt.figure(figsize=(10,7))
sns.barplot(x=region.index,y=region.values)
plt.xticks(rotation=90)
plt.ylabel('Ülke Sayısı')
plt.xlabel('Bölge')
plt.title('Bölgelere Ait Ülkeler Grafiği',color = 'red',fontsize=20)
plt.plot()
countries_grouped= data.groupby('Region')[['Country','Population','Area (sq. mi.)']].agg({'Country':'count', 'Population':'sum','Area (sq. mi.)':'sum'})
countries_grouped.sort_values('Country', axis = 0, ascending = False).head()
#Datamızı bölgelerdeki ülke sayılarına ve popülasyona göre gruplandırdık
yogunluk=data["Pop. Density (per sq. mi.)"].mean()
data["Yogunluk"]=["Kalabalık" if i>yogunluk else "Sakin" for i in data["Pop. Density (per sq. mi.)"]]
yogunlukData = data.loc[:,["Country","Population","Area (sq. mi.)","Pop. Density (per sq. mi.)","Yogunluk"]]
yogunlukData.head(20)
data['Birthrate'] = data['Birthrate'].transform(lambda x: x.fillna(x.mean()))
data['Deathrate'] = data['Deathrate'].transform(lambda x: x.fillna(x.mean()))
data['Industry'] = data['Industry'].transform(lambda x: x.fillna(x.mean()))
data['Literacy (%)'] = data['Literacy (%)'].transform(lambda x: x.fillna(x.mean()))
data['Phones (per 1000)'] = data['Phones (per 1000)'].transform(lambda x: x.fillna(x.mean()))
data['GDP ($ per capita)'] = data['GDP ($ per capita)'].transform(lambda x: x.fillna(x.mean()))
data['Infant mortality (per 1000 births)'] = data['Infant mortality (per 1000 births)'].transform(lambda x: x.fillna(x.mean()))
#Datamızın içindeki eksik değerleri yok etmek yerine onlara ortalama değerler vererek korelasyonu bozmamaya çalışıyoruz
data.isnull().sum()
data.corr()
#İki veya daha fazla değişken arasındaki ilişkinin varlığı,bu ilişkinin yönü ve şiddeti korelasyon analizi ile belirlenir.
#datamızda yanlış type değerli verilerimiz var string değerler korelasyonda görünmez
#correlation map
f,ax = plt.subplots(figsize=(12, 10))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
#korelasyon tablosu,annot=içindeki sayıların görünürlüğü,linewidths=çerçeve
plt.show()
arinmisData = data.loc[:,["Country","Region","Population","GDP ($ per capita)","Literacy (%)","Phones (per 1000)","Birthrate","Deathrate"]]
arinmisData.head()
#Datadaki veriler çok karmaşık olduğu için işe yarayanları bir araya topladık
sns.jointplot(arinmisData.loc[:,'GDP ($ per capita)'], arinmisData.loc[:,'Phones (per 1000)'], kind="regg", color="#ce1414")
#Seaborn grafiğimizde en iyi sonucu para ve telefon kullanımı verdi,ikisinin arasındaki uyumu grafiğe dökelim
sns.jointplot(arinmisData["Literacy (%)"], arinmisData["Birthrate"], kind="kde", size=8)
arinmisData.sort_values('GDP ($ per capita)', axis = 0, ascending = False).head()
# Kişilerin kazançlarına göre ilk beş ülke
#Veri Normalleştirme
from sklearn import preprocessing

#Maaş özniteliğini normalleştirmek istiyoruz
x = arinmisData[['GDP ($ per capita)']].values.astype(float)

#Normalleştirme için MinMax normalleştirme yöntemini kullanıyoruz.
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
arinmisData['NormalizeMaas'] = pd.DataFrame(x_scaled)
arinmisData.head()

#BoxPlot ile aykırı değerleri kolaylıkla görebiliyoruz
#Kişi başına düşen gelir bazen 50000$ ların üzerine çıktığını görüyoruz bu datamızın genel rakamlarından oldukça uç bir değer olduğunu bize göserir
sns.boxplot(y=arinmisData['NormalizeMaas'])
plt.show()
sns.distplot(arinmisData["NormalizeMaas"],hist=False,bins=20,kde=True,color="g",kde_kws={"shade":True})
plt.tight_layout()#Increases the alignment of the drawn graph.
plt.plot()
#Histogram
arinmisData["Deathrate"].plot(kind="hist",color="green",bins=30,grid=True,alpha=0.4,label="Deathrate",figsize=(18,8))
plt.legend()
plt.xlabel("Deathrate")
plt.ylabel("Rate")
plt.title("Mortality rates")
plt.show()
X=arinmisData.iloc[:,3].values.reshape(-1,1)#NormalizeMaas sütunu
Y=arinmisData["Phones (per 1000)"].values.reshape(-1,1) #telefon sütunu
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=7) 

plt.scatter(X,Y)
plt.show()
from sklearn.linear_model import LinearRegression  
model = LinearRegression()  
model.fit(X_train,y_train)

from sklearn.preprocessing import PolynomialFeatures
polyReg = PolynomialFeatures(degree=2,interaction_only = False,include_bias=True)
x_polyTrain = polyReg.fit_transform(X_train)
x_polyTest = polyReg.fit_transform(X_test)
polyReg.fit(x_polyTrain,y_train)
PolyLineer = LinearRegression()
PolyLineer.fit(x_polyTrain,y_train)
print("Kesim noktası:", model.intercept_) 
print("Eğim:", model.coef_)
y_pred = model.predict(X_test) 
y_predPoly = PolyLineer.predict(x_polyTest)
df = pd.DataFrame({'Gerçek': y_test[:,0], 'Tahmin Edilen': y_pred[:,0]})  
#print(df)


from sklearn.metrics import r2_score
lbasarim=r2_score(y_test,y_pred)
pbasarim=r2_score(y_test,y_predPoly)
print(lbasarim)
print(pbasarim)



plt.scatter(X_train, y_train, color = 'red')
modelin_tahmin_ettigi_y = model.predict(X_train)
plt.plot(X_train, modelin_tahmin_ettigi_y, color = 'blue')
plt.title('Phones Per 1000 - Gdp Per Capita')
plt.xlabel("Gdp Per Capita")
plt.ylabel("Phones Per 1000")
plt.show()
from sklearn import metrics   
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))