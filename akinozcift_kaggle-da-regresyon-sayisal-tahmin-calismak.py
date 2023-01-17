
import pandas as pd # datayi CSV veya baska kaynaklardan okutmak icin kullaniyoruz

# sag tarafta draft environment bolumu var
# o bolumde input klasorumuze sistemden veya disardan ekledigimiz data dosyasi var
# yol tanimi ../input/melb_fiyat.csv     seklinde


veriyolu = '../input/melb_fiyat.csv' # dizini veriyolu olarak degiskene atadik 
data = pd.read_csv(veriyolu) #pandas'la read metoduyla yolunu yazdigimiz dosyayi aktardik 
data.describe() # bakalim neler aktarmisiz describe ile ilk bir kac satiri gorelim


data.columns # veride hangi kolonlar var gorelim
# tum veri data degiskeninde oradan alacagimiz kolonlari tanimlayalim 
secilen_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
# simdi bu diziyi kullanarak X (tanimlayan kolonlar) ve Y (tahmin kolonu, price) sekline gecelim
# X degerlerini data dan elde edelim
X = data[secilen_features]
X.describe() # sectigimiz featurelar la veri kumesini gorelim
# y degerini de data dan elde edecegiz.Bunun icin 
y= data.Price  # data'nin altindan Price kolonunu tum satirlarla y degiskenine aktar 
y.describe() # sectigimizi gorelim
# veriyi bolmek icin train_test_split metodunu kullanacagiz
from sklearn.cross_validation import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) 
# test_size 0.2 --> % 20 ise kalan % 80 train olur
# random state = 0 ile random olmasin secimi yapiyoruz her seferinde bolumleme ve icindeki ornekler ayni kaliyor
# arzu edilirse X_train yazilarak nasil bolundugu ekrana getirilebilir 
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train) # modeli train degerleriyle egitiyoruz
y_pred = regressor.predict(X_test) # modeli test bolumuyle tahmin yetenegi anlaminda test ediyoruz

# acaba tahmin ettigimiz degerler y_pred ve test degerleri y_test ne kadar yakin yanyana gorsek
#tahmin ne kadar yakinsa fark sifira o kadar yakin olur
from sklearn.metrics import mean_absolute_error
ev_fiyat_tahminleri = regressor.predict(X_test)
mean_absolute_error(y_test, ev_fiyat_tahminleri)
