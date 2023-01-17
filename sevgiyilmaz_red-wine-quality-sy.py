import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
datapath = '../input/winequality-red.csv' # veri setinin değişken haline getirilmiş oldu.
wine_dataset = pd.read_csv(datapath)
wine_df=pd.DataFrame(wine_dataset) 
wine_df
wine_df.shape #veri setinin satır ve sütun sayısını verir
wine_df.describe() #veri setinin sütunları ile hesaplanmış istatistikler
wine_df.info() #veri içerisindeki veri türleri ve bellek kullanımları
wine_df.head() #verinin ilk 5 satırını çıktı olarak verir
wine_df.tail() #verinin son 5 satırını çıktı olarak verir
wine_df.sample(10) #veri içerisinden rasgele 10 adet satır çıktı halinde verir
#HİSTOGRAM
wine_dataset.hist()
plt.show()
#KORELASYON
wine_df.corr() 
#pozitif korelasyonda özniteliklerden biri artarsa diğeri de artar demektir.
#fixed acidity ve citric acid arasında 0,67lik bir pozitif kolerasyon bulunmaktadır.
#fixed acidity ve density arasında 0,66lık bir pozitif kolerasyon bulunmaktadır.
#free sulfur dioxide ve total sulfur dioxide arasında 0,66lık bir pozitif kolerasyon bulunmaktadır.
#negatif kolerasyonda öznitelik arasında tersine ilişki olduğunu gösterir.Biri artarken birinin azaldığını gösterir.
#fixed acidity ile ph arasında -0,68lik bir negatif korelasyon bulunmaktadır.
#SEABORN VE CORRELASYONU YÜKSEK İKİ VERİNİN HİSTOGRAMI
#SEABORN
sns.lmplot(x = "density", y = "quality", data = wine_df)
#Fixed-acidity ve citric-acid öznitelikleri arasındaki korelasyon
f_acid = wine_dataset.loc[:,"fixed acidity"]
c_acid = wine_dataset.loc[:,"citric acid"]
plt.bar(f_acid, c_acid)
plt.show() #Grafik ile fixed acid ve citric acid arasında parabolik bir artış ilişkisi olduğu çıkarımı bulunabilir.
#free sulfur dioxide ve total sulfur dioxide öznitelikleri arasındaki korelasyon
f_sultur = wine_dataset.loc[:,"free sulfur dioxide"]
t_sulfur = wine_dataset.loc[:,"total sulfur dioxide"]
plt.bar(f_sultur,t_sulfur)
plt.show()
#EKSİK VERİ BULMA
import matplotlib.pyplot                                                        
from sklearn.preprocessing import Imputer
eksik_deger_toplami = wine_dataset.isnull().sum()
print("Sutünlar eksik veriler toplamı : ",eksik_deger_toplami) #Eksik değerlerin toplamını verir.Buna göre sutünlarda eksik değer olmadığı görülmüştür.
#Uç Değerlerin Bulunması - Box Plot ile görsel kontrol gerçekleştirilebilir.
wine_dataset.plot(kind ='box', subplots=True, sharex=False, sharey=False)
plt.show()
y = wine_dataset.loc[:,"quality"].values  #Quality değeri bizim y değerimiz
print ("y :" , y)
x = wine_dataset.iloc[:,:-1].values #Diğer tüm özniteliklerimiz X değerleri
print("x :", x )
from sklearn.model_selection import train_test_split #20-80 oranında test ve eğitim setleri oluşmuş olur.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test) 
from sklearn.naive_bayes import GaussianNB #NaiveBayes yöntemi il model eğitilmiş,confusion matris oluşturulmuştur.
classifier = GaussianNB()
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
print ("y_predicted :" ,y_pred)
from sklearn.metrics import confusion_matrix
cm_naive = confusion_matrix(y_test, y_pred)
print("Confusion Matrix_Naive Bayes : " ,cm_naive)
from sklearn.metrics import accuracy_score
acc_naive = accuracy_score(y_test,y_pred)
print("Accuracy Score_Naive Bayes : " , acc_naive)
clas_report_naive = classification_report(y_test,y_pred)
print (clas_report_naive)

from sklearn.tree import DecisionTreeClassifier #Decision Tree ile model eğitilmiş,confusion matris oluşturulmuştur.
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
print (y_pred)
from sklearn.metrics import confusion_matrix
cm_dtree = confusion_matrix(y_test, y_pred)
print("Confusion Matrix_Decision Tree : " ,cm_dtree)
from sklearn.metrics import accuracy_score
acc_dtree = accuracy_score(y_test,y_pred)
print("Accuracy Score_Decision Tree : " , acc_dtree)
classification_report(y_test, y_pred)
clas_report_dtree = classification_report(y_test,y_pred)
print (clas_report_dtree)
#MODEL SONUÇLARININ DEĞERLENDİRİLMESİ
#Decision Tree modeli ile 63,4% accuracy rate elde edilmiştir.Ancak Naive bayes ile model çalıştırıldığında 54 % lük bir accurcy rate bulunmuştur.
#Bu gösteriyorki veri seti ile 80-20lik oranda oluşturulan eğitim-test verisi üzerinden decision tree modeli ile daha iyi bir sonuç elde edilmektedir.
#Yeni öznitelik oluşturma ( Chlorides ve Density öznitelikleri kullanılarak yeni öznitelik oluşturulma)
ch = wine_df.chlorides.values
den = wine_df.density.values
print (ch)
print(den)
yeni_sutün_değerleri=ch/den
print(yeni_sutün_değerleri)
wine_df["yeni"]=yeni_sutün_değerleri
print(wine_df.loc[:,"yeni"])
#Normalleştirme
alcohol_max = max(wine_df['alcohol']) #Normalleştirme için sutünün max değeri bulunup,sutündaki değerler 0-1 arasına çekilmiş olur.
print ("Max Alcohol değeri : ",alcohol_max)
normalization_alcohol=wine_df.alcohol.values / alcohol_max
print("Normalize Alcohol Sutünü : ", normalization_alcohol) # Alcohol sutünü normalleştirilmiştir.
hist_x = np.arange(1,1600,1)
hist_y = normalization_alcohol
plt.plot(hist_x,hist_y)
plt.show() #Normalleştirilmiş sutündaki değerlerin 0-1 arasında olduğu görülmüştür.
