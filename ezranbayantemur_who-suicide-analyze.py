# Öncelikle kütüphanelerimizi çağırıyoruz.
import pandas as pd # Verimizi okumak ve gruplandırmak için: pandas
import numpy as np  # Verimiz üzerinde sayısal işlemler yapmak için: numpy
# Verimizi görselleştirmek için: matplot ve seaborn
import matplotlib.pyplot as plt 
import seaborn as sns
# Kullanacağımız verimizi okuyoruz.
df = pd.read_csv("../input/who_suicide_statistics.csv")
# Veri setimizin ilk 10 değerini inceliyoruz
df.head(10)
# Veri setimizin son 5 değerini inceliyoruz
df.tail()
# Veri setimizin satır sayısına ve öz nitelik adedine bakıyoruz.
df.shape
# Kolonlarımızı ve veri tiplerimizi gözlemliyoruz.
df.info()
# Sayısal değerlerimize bakıyoruz.
df.describe() 
# Verimizin korelasyon durumunu gözlemliyoruz. 
sns.heatmap(df.corr())
# Verimizdeki toplam boş değer sayısını kontrol ediyoruz.
df.isnull().sum()
# Popülasyon kolonumuzu kaldırıyoruz. Gerektiği görselleştirme kısımlarında geri çağıracağız.
df.drop(["population"], axis=1, inplace=True)
# Herhangi bir özniteliği boş olan verileri temizliyoruz
df = df.dropna(axis=0, how="any")
# Boş değerlerimizin kalıp kalmadığını gözlemleyelim
df.isnull().sum()
# Kullanacağımız kolonları ayırıyoruz.
years = df.iloc[:, 1].values.reshape(-1,1)
suicides = df.iloc[:, 4].values.reshape(-1,1)
# Histogram (frekans grafiği) ile intihar olayının en çok tekrar ettiği yılları görebiliriz.
plt.figure(figsize=(20,10))
plt.hist(years, bins=50)
plt.xlabel("Yıllar")
plt.ylabel("İntihar Sıklığı")
# Bütün ülkeleri intihar sayılarına göre gruplandırıyoruz ve dataframe'imizi düzenliyoruz
country_suicides = pd.DataFrame(df.groupby("country")["suicides_no"].sum())
country_suicides['country'] = country_suicides.index
country_suicides = country_suicides.reset_index(drop = True)

country_suicides.head()
# Ülkeleri intihar sayılarına göre düzenleyip en çok ilk 50 ülkemizi ayırıyoruz
sorted_data = country_suicides.sort_values("suicides_no", ascending=False)
sorted_data = sorted_data.iloc[:50,:]
# Görselimizi oluşturuyoruz
plt.figure(figsize=(20,10))
sns.barplot(x=sorted_data.country, y=sorted_data.suicides_no)
plt.xticks(rotation= 90)
plt.xlabel('Ülkeler')
plt.ylabel('Toplam İntihar Sayısı')
plt.title('İNTİHARLAR')
# Ülkeleri isim olarak ayırıyoruz
unique = pd.DataFrame(df.country.unique(), columns = ["countries"])
unique.sample(10)
# Toplayacağımız verileri saklayacağımız veri tablosunu oluşturuyoruz.
suicides_by_gender = pd.DataFrame(columns=['country', # ülke
                                           'total s', # total suicide:  toplam intihar sayısı
                                           'per f',   # percent female: kadın intiharları yüzdesi
                                           'per m',]) # percent male:   erkek intiharları yüzdesi
for index, rows in unique.iterrows():
   test_df = df[df["country"] == rows["countries"]]
   suicides_by_gender = suicides_by_gender.append({'country': rows["countries"],
        'total s' : df[df["country"] == rows["countries"]].suicides_no.sum(),
        'per f': test_df[test_df["sex"] == "female"].suicides_no.sum()/df[df["country"] == rows["countries"]].suicides_no.sum()*100,
        'per m': test_df[test_df["sex"] == "male"].suicides_no.sum()/df[df["country"] == rows["countries"]].suicides_no.sum()*100}, ignore_index=True)
# Verimizdeki intihar sayısı baz alınarak aykırı verileri temizliyoruz.
P = np.percentile(suicides_by_gender['total s'], [10, 100])
suicides_by_gender = suicides_by_gender[(suicides_by_gender['total s'] > P[0]) & (suicides_by_gender['total s'] <= P[1])]
# Oluşan veri tablomuzu inceleyelim
suicides_by_gender.head()
# Verimizi kadın intiharları yüzdesine göre sıralayalım.
suicides_by_gender = suicides_by_gender.sort_values("per f", ascending=False)
suicides_by_gender.head(10)
# Verimizi erkek intihar yüzdesine göre sıralayalım.
suicides_by_gender = suicides_by_gender.sort_values("per m", ascending=False)
suicides_by_gender.head(10)
# Ülke isimlerini "unique" içerisinde daha önceden kaydetmiştik.
# Toplayacağımız verileri saklayacağımız veri tablosunu oluşturuyoruz.
suicides_by_age = pd.DataFrame(columns=[
    'country',   # ülke
    'total s',   # total suicide: toplam intihar sayısı
    'per 5-14',  # 5 ve 14 yaş arası kadın ve erkeklerin toplam yüzdesi
    'per 15-24', # 15 ve 24 yaş arası kadın ve erkeklerin toplam yüzdesi
    'per 25-34', # 25 ve 34 yaş arası kadın ve erkeklerin toplam yüzdesi
    'per 35-54', # 35 ve 54 yaş arası kadın ve erkeklerin toplam yüzdesi
    'per 55-74', # 55 ve 74 yaş arası kadın ve erkeklerin toplam yüzdesi
    'per 75+'])  # 75 yaş ve üzeri kadın ve erkeklerin toplam yüzdesi
# Döngümüzle ana veri tablomuzdan eşleşen ülkelerin yaş aralıklarındaki intiharlar düzenleniyor.
for index, rows in unique.iterrows():
   test_df = df[df["country"] == rows["countries"]]
   s = df[df["country"] == rows["countries"]].suicides_no.sum()
   suicides_by_age = suicides_by_age.append({'country': rows["countries"],
        'total s': df[df["country"] == rows["countries"]].suicides_no.sum(),
        'per 5-14': test_df[test_df["age"] == "5-14 years"].suicides_no.sum()/s *100, 
        'per 15-24': test_df[test_df["age"] == "15-24 years"].suicides_no.sum()/s*100, 
        'per 25-34': test_df[test_df["age"] == "25-34 years"].suicides_no.sum()/s*100, 
        'per 35-54': test_df[test_df["age"] == "35-54 years"].suicides_no.sum()/s*100, 
        'per 55-74': test_df[test_df["age"] == "55-74 years"].suicides_no.sum()/s*100,
        'per 75+': test_df[test_df["age"] == "75+ years"].suicides_no.sum()/s*100}, ignore_index=True)
# Verimizdeki intihar sayısını göz önünde bulundurarak aykırı verileri temizliyoruz.
P = np.percentile(suicides_by_age['total s'], [10, 100])
suicides_by_age = suicides_by_age[(suicides_by_age['total s'] > P[0]) & (suicides_by_age['total s'] <= P[1])]
# Oluşan veri tablomuzu inceleyelim
suicides_by_age.head()
# Verimizi 5-14 yüzdesine göre sıralayalım.
suicides_by_age = suicides_by_age.sort_values("per 5-14", ascending=False)
suicides_by_age.head(10)
# Verimizi 15-24 yüzdesine göre sıralayalım.
suicides_by_age = suicides_by_age.sort_values("per 15-24", ascending=False)
suicides_by_age.head(10)

# Verimizi 25-34 yüzdesine göre sıralayalım.
suicides_by_age = suicides_by_age.sort_values("per 25-34", ascending=False)
suicides_by_age.head(10)
# Verimizi 35-54 yüzdesine göre sıralayalım.
suicides_by_age = suicides_by_age.sort_values("per 35-54", ascending=False)
suicides_by_age.head(10)
# Verimizi 55-74 yüzdesine göre sıralayalım.
suicides_by_age = suicides_by_age.sort_values("per 55-74", ascending=False)
suicides_by_age.head(10)
# Verimizi 75+ yüzdesine göre sıralayalım.
suicides_by_age = suicides_by_age.sort_values("per 75+", ascending=False)
suicides_by_age.head(10)
df2 = pd.read_csv("../input/who_suicide_statistics.csv")
df2 = df2.dropna(axis=0, how="any")
df2.isnull().sum()
unique = pd.DataFrame(df2.country.unique(), columns = ["countries"])
unique.sample(10)
# Toplayacağımız verileri saklayacağımız veri tablosunu oluşturuyoruz.
suicides_by_pop = pd.DataFrame(columns=['country', 
                                           'total s', 
                                           'mean pop',
                                           'per s'])
# Her bir ülkeyi gezip intiharların toplamını popülasyonun ortalamasını alıyor, oluşturduğumuz dataframe'e ekliyoruz.
for index, rows in unique.iterrows():
   test_df = df2[df2["country"] == rows["countries"]]
   suicides_by_pop = suicides_by_pop.append({
        'country': rows["countries"],
        'total s' : test_df.suicides_no.sum(),
        'mean pop': test_df.population.mean(),
        'per s': test_df.suicides_no.sum()/test_df.population.mean()*100}, 
            ignore_index=True)
# Aykırı verileri temizliyoruz.
P = np.percentile(suicides_by_pop['per s'], [10, 100])
suicides_by_pop = suicides_by_pop[(suicides_by_pop['per s'] > P[0]) & (suicides_by_pop['per s'] <= P[1])]
# Oluşan veri tablomuzu inceleyelim
suicides_by_pop.head()
# Verimizi intihar/popülasyon oran yüzdesine göre sıralayalım.
suicides_by_pop = suicides_by_pop.sort_values("per s", ascending=False)
suicides_by_pop.head(10)
# Regresyon modellerimizin başarısını ölçmek için r2 score fonksiyonunu kullanacağız.
# Değer ne kadar 1'e yakınsa o kadar iyi sonuç alınmıştır demektir.
from sklearn.metrics import r2_score
# Yıllara göre intihar sayılarının toplamını grupluyoruz.
analyze_df = pd.DataFrame(df.groupby("year")["suicides_no"].sum())
analyze_df.head()
analyze_df.tail()
# Girdi değerlerine yılları, çıktı değerlerine intiharları atıyoruz. 
x = pd.DataFrame(analyze_df.index)
y = analyze_df.iloc[:, 0]

# 2016 yılını ve intihar değerini temizliyoruz.
x = x.iloc[:-1, :].values.reshape(-1,1)
y = analyze_df.iloc[:-1, 0].values.reshape(-1,1)
# Gerekli kütüphaneleri çağırıyoruz.
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Modellerimizi oluşturuyoruz.
lr = LinearRegression()

# 4. dereceden bir fonksiyon kullanacağız.
pf = PolynomialFeatures(degree=4)
# Öncelikle girdi değerlerimizi polinomal forma çeviriyoruz. Çünkü polinomal bir fonksiyon oluşturduysak o dereceden girdiler sağlamalıyız.
x_pol = pf.fit_transform(x)
# Modelimizi eğitiyoruz
lr.fit(x_pol, y)
# Modelimizi eğittik, şimdi ise test edeceğiz. Öncelikle test datamızı polinomal hale getiriyoruz.
x_pol2 = pf.fit_transform(x)
# Çevirdiğimiz test verisini modele gönderip tahmin edilen değerlerimizi göreceğiz.
y_pred = lr.predict(x_pol2)
# Orijinal değerler ile fonksiyonumuzun türettiği verileri kıyaslamak için görselleştirelim.
plt.figure(figsize=(20,10))
plt.scatter(x, y)
plt.plot(x, y_pred, color="red")
plt.xlabel("Yıllar")
plt.ylabel("İntihar Sayıları")
# Regresyon score değerini gözlemleyelim
r2_score(y, y_pred)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)
# Kullanacağımız modellerde girdi değerlerimize normalizasyon uygulamalıyız. Bunun için gerekli kütüphaneleri çağırıyoruz.
from sklearn.preprocessing import StandardScaler
# Girdi çıktı değerleri için ayrı birer scale modeli üretmeliyiz, çağırılan her model ancak normalize ettiği veri ölçütünü de-normalize edebilir.
sc_x = StandardScaler()
sc_y = StandardScaler()
x_train = sc_x.fit_transform(x_train)
y_train = sc_y.fit_transform(y_train)

# Test edeceğimiz girdi değerlerini de normalize etmeliyiz.
x_test = sc_x.transform(x_test)
# SVR için gerekli kütüphanemizi çağırıyoruz
from sklearn.svm import SVR
# rbf
model = SVR(kernel="rbf")
model.fit(x_train, y_train)
y_pred_rbf = model.predict(x_test)
# Normalize ederek eğittiğimiz veriler bize normalize edilmiş bir halde dönecektir. 
# Gerçek değerlerini görmek için de-normalize (inverse transform) etmemiz gerekmekte.
y_pred_rbf = sc_y.inverse_transform(y_pred_rbf)
# linear
model = SVR(kernel="linear")
model.fit(x_train, y_train)
y_pred_linear =  sc_y.inverse_transform(model.predict(x_test))
# poly
model = SVR(kernel="poly")
model.fit(x_train, y_train)
y_pred_poly =  sc_y.inverse_transform(model.predict(x_test))
# sigmoid
model = SVR(kernel="sigmoid")
model.fit(x_train, y_train)
y_pred_sgm =  sc_y.inverse_transform(model.predict(x_test))
print(r2_score(y_test, y_pred_rbf))
print(r2_score(y_test, y_pred_linear))
print(r2_score(y_test, y_pred_poly)) 
print(r2_score(y_test, y_pred_sgm)) 
from sklearn.tree import DecisionTreeRegressor
# Ağacımızın derinliğini 100 birim belirledik.
model = DecisionTreeRegressor(max_depth = 100, random_state=0)
model.fit(x_train, y_train)
y_pred = sc_y.inverse_transform(model.predict(x_test))
# Model score'umuzu kontrol ediyoruz.
r2_score(y_test, y_pred)
score=[]
for i in range(1, 15):
    model = DecisionTreeRegressor(max_depth = i, random_state=0)
    model.fit(x_train, y_train)
    y_pred = sc_y.inverse_transform(model.predict(x_test))
    score.append(r2_score(y_test, y_pred))
# Score çıktılarımızı grafik üzerinde inceleyelim
plt.figure(figsize=(20,10))
plt.plot(range(1,15), score)
plt.xlabel("Derinlik")
plt.ylabel("R2 Score")
from sklearn.ensemble import RandomForestRegressor

# Desicion Tree yöntemimizde olduğu gibi en uygun dallanma sayısını görmek için döngü kuralım.
score = []
for i in range(1, 50):
    model = RandomForestRegressor(n_estimators = i, random_state = 0)
    model.fit(x_train, y_train)    
    y_pred = sc_y.inverse_transform(model.predict(x_test))    
    score.append(r2_score(y_test, y_pred))
# Score çıktılarımızı grafik üzerinde inceleyelim
plt.figure(figsize=(20,10))
plt.plot(range(1, 50), score)
plt.xlabel("Dallanma Sayısı")
plt.ylabel("R2 Score")
plt.show()
from sklearn.neighbors import KNeighborsRegressor

# Komşu sayımızı bir döngü ile belirleyip score değerini grafik üzerinde gözlemleyebiliriz
score = []
for i in range(1, 27):
    model = KNeighborsRegressor(n_neighbors = i) # n_neighbors = k
    model.fit(x_train,y_train)
    y_pred = sc_y.inverse_transform(model.predict(x_test))    
    score.append(r2_score(y_test, y_pred))
plt.figure(figsize=(20,10))
plt.plot(range(1, 27), score)
plt.xlabel("Komşuluk Sayısı")
plt.ylabel("R2 Score")

