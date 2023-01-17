import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
### pandas kütüphanesindeki read_csv() fonksiyonu ile bir csv(virgülle ayrılmış değerler) dosyasını pandas dataframe'e çevirmek mümkündür
### read_csv() fonksiyonun en önemli 2 parametresi path  yani dosyanın bulunduğu adresi yazdığımız parametredir
### diğeri ise dosyanın içindeki verilerin ne ile ayrıldığını anlayan sep parametresidir. Öntanımlı değeri "," dür.
### eğerki dosyanın içindeki değerler virdülden başka bir nesne noktalama işareti ile ayrılıyorsa bunu sep=";" (örnektir) diye belirmemiz gerekmektedir
data=pd.read_csv("../input/titanic/train.csv")
### copy() fonksiyonu mevcut csv dosyamızın ilerde yapacağım değişikliklerden etkilenmemsi için kullanılan bir fonksiyondur. mevcut oluşan dataframe in özelliklerini  başka bir değişkene atamaya yarar.
df=data.copy()
### head() fonksiyonun öntanımlı parametresi n=5 tir hiç bir şey yazılmadan df.head() olarak yazdığınızda dataframedeki ilk 5 değeri getirmektedir.
### tail() fonksiyonu head() fonksiyonu gibi bir paramtreye sahiptir ve ön tanımlı değeri 5 tir bu fonksiyon ise bize dataframedeki son 5 değeri getirir.
df.head()
df.head(n=3)
df.tail()
df.tail(3)
### pd.DataFrame.shape bize bir dataframe in boyutlarını sayısal değerler olarak bir tuple ile döndürür. Bu tuple in ilk değeri Bize dataframede bulunan satır sayısını diğer değer ise bulunan sutun sayısını vermektedir
df.shape
print("Titanic Data setimizin mevcut satır sayısı {0} ve  sutun sayısı {1} dır ".format(df.shape[0],df.shape[1]))
### df.info() fonksiyonu bize dataframe in kısa bir özetini vermekte bunlar başlıca kaç satırımızın olduğu,bu satırlara denk gelen indekslerin başlangıç-bitiş noktaları,sutunların isimleri
### sutunlardaki toplam dolu değer adetini sutunların veritiplerinin ne olduğunu ve toplamda hafızada ne kadar değer tuttuğu verir.
df.info()
### df.columns bize dataframe de bulan sutunların isimlerini vermektedir.
df.columns
### df.index bize dataframe deki indexlerin başlangıç ve bitiş noktalarını ve indexler arasındaki artışın hakkında bilgi vermektedir
df.index
### df.describe() fonksiyonu bize dataframe deki veri tipi sayısal olan sutunların betimleyici istatistik değerlerini vermektedir. Sutundaki nan değerler hariç tutarak bu işlemleri yapar
df.describe()
### df.describe() fonksiyonun çıktısının daha okunaklı ve anlaması kolay olması içinsonuna .T koyarız
df.describe().T
### df.isnull() fonksiyonu bize bir boolean (True,False)dataframe döndürür ve bu dataframe de df deki satırlar ve sutunlar vardır farki ise boş değerlerin yerini True dolu değerlerin yerini False alır
### bu fonksiyonun yanına eklenen values değeri ise dataframe i bir arraya dönüştürür ve her bir elemanı bir satırdaki değerleri içerir
### bu birleşik kod bloğunun sonuna eklenecek any() fonksiyonu bu değerlerin içinde hiç True var mı diye sorar ve varsa True yoksa False döndürür.
df.isnull()
df.isnull().values
df.isnull().values.any()
df.isnull().sum()
df.info()
### df.info ile dataframe bir göz attığımızda dtype object olanlar bizim kategorik değişkenlerimizdir.
### bunun iki şekilde bulabiliriz aslında 2 sizde aynı döngü ama kod temizliği ve tekrarı olmaması adına Python un bize sunduğu "list comprehension" tekniğini öğreneceğiz
# birinci yol klasik for ve if döngüsü
cat_colsF = []
for col in df.columns:
    if df[col].dtype == "O":
        cat_colsF.append(col)
cat_colsF
### List comprehension yöntemi 

cat_cols = [col for col in df.columns if df[col].dtype=="O"]
cat_cols
### kaç adet kategorik değişkenimiz var ona bakalım 
len(cat_cols)
### Kategorik değşkenler sadece objectlerden olmaya bilir int değere sahip bir sutunda içerdiği verilere göre aslında kategorik değişken olarak davranabilir veya kabul edilebilir 
### Örnek olarak df bulunan Survived Sutunnu dtype olarak integer olsada içerdiği verilerde göreceğimiz üzere sadece 2 farklı değer vardır. Nasıl Bu bilgiye ulaşabiliriz?
## unique() fonksiyonu bize bir sutundaki eşsiz verileri vermekte
## nunique() fonksiyonu ise bize bir sutunda kaç farklı eşsiz veri olduğunu vermekte
df["Survived"].unique()
len(df["Survived"].unique())
df["Survived"].nunique()
### Tekrardan kategorik değişkenleri içinde tutan listemizi olusturalım ve bir list comprehension yazarak bir sutunda 10 dan az eşsiz veri varsa bunlarıda kategorik değişken olarak kabul etsin 
cat_cols = [col for col in df.columns if len(df[col].unique()) <10]
cat_cols
len(cat_cols)
### olusturdugumuz cat_cols ile hangi kategorik değişken sutununda kac adet eşsiz veri var bakalım..
df[cat_cols].nunique()
### ilk kategorik değişkenleri aldığımızda listemizde 5 eleman varken su an dtype i integer olan Survived sutunuda artık bizim kategorik değişkenler listemizde
### Survived sutunumuzun başlıca özelliklerine bakıp içerdiği eşsiz verilere göre bir görselleştirme yapalım 
df["Survived"].unique()
len(df["Survived"].unique())
### Survived Sutundaki her bir eşsiz değerden kaç adet var onu arıyoruz
df["Survived"].value_counts()
### Bunu birde grafik ile görselleştirelim
sns.countplot(x="Sex", data=df);
## Survived Sutundaki eşsiz değerlerin yüzdelik dilimlerini nasıl buluruz
(df["Survived"].value_counts()/len(df))*100
def cat_summary(data): ### fonksiyonumuzun adı cat_summary
    ### cat_names değişkenin içine bir list comprehension ile df içinde bulunan sutunlardaki eşsiz veri sayısı 10 dan küçükleri kategorik değişken kabul ederek ismini bu değişkene atıyoruz
    cat_names = [col for col in data.columns if len(data[col].unique()) < 10] 
    for col in cat_names:
        print(pd.DataFrame({col: data[col].value_counts(), ## dataframe deki sutunun essiz değerlerinin adetlerini raporluyoruz
                           "Ratio": 100 * data[col].value_counts()/ len (data)}), end = "\n\n\n")## dataframe deki sutunun essiz değerlerinin yüzdelik ifadelerini raporluyoruz
        sns.countplot(x = col, data = data) ## ilgi sutunun seaborn kütüphanesindeki countplot grafiğini çizdiriyoruz
        plt.show() ## jupyter notebook ve jupyter labda yazmamıza gerek yok ama IDElerde (pycharm) grafiği göstermek için gerek olan kod.
cat_summary(df)
cat_cols = [col for col in df.columns if df[col].dtype == "O"]
cat_cols
def cat_summary(data, categorical_cols, number_of_classes=10):
    
    var_count = 0  # Kaç kategorik değişken olduğu raporlanacak
    vars_more_classes = []  # Belirli bir sayıdan daha fazla sayıda sınıfı olan değişkenler saklanacak.
    
    for var in data:
        if var in categorical_cols:
            if len(list(data[var].unique())) <= number_of_classes:  # sınıf sayısına göre seç
                print(pd.DataFrame({var: data[var].value_counts(),
                                    "Ratio": 100 * data[var].value_counts() / len(data)}),end="\n\n\n")
                var_count += 1
            else:
                vars_more_classes.append(data[var].name)
    print('%d categorical variables have been described' % var_count, end="\n\n")
    print('There are', len(vars_more_classes), "variables have more than", number_of_classes, "classes", end="\n\n")
    print('Variable names have more than %d classes:' % number_of_classes, end="\n\n")
    print(vars_more_classes)
cat_summary(df, cat_cols)
### Sayısal değişkenleri içeren sutunlarımızın betimleyici istatistiksel değerlerine bakacağız ===> describe().T 
### describe() fonksiyonda quartile lardan hariç yüzdelik dilimlerdeki istatistiksel değerlere bakma
### Kaç Adet Sayısal Değişken içeren sutunumuz var bulma
### Bir Sayısal değişken sutunun hsitogram Dağılım grafiğini çizme
### Bir Df deki tüm sayısal değişkenlerin histogram grafiğini çizen bir fonksiyon olusturma 
### Sayısal değişkenleri içeren sutunlarımızın betimleyici istatistiksel değerlerine bakacağız ===> describe().T 
df.describe().T
### describe() fonksiyonda quartile lardan hariç yüzdelik dilimlerdeki istatistiksel değerlere bakma
### describe() fonksiyonun içindeki percentiles parametresi ön tanımlı olarak[.25,.5,.75] tamımmlıdır içine bir liste ile çeşitli değerler verirsek onlarıda bize özette verir
df.describe([0.05, 0.10, 0.25, 0.50, 0.75, 0.80, 0.90, 0.95, 0.99]).T
### df te kaç adet sayısal değer var ona bakıyoruz
num_cols = [col for col in df.columns if df[col].dtypes != "O"]
num_cols
print("Train data setimizde toplam {0} adet sayısal değişkenimiz vardır".format(len(num_cols)))
### BU değişkenlerden PassengerId ve Survived bu değerlerin arasında olmamalı Survived bizim hedef değişkenimiz(target) PassengerId ise yolcuları temsil eden bir değer survived ı etkilememekte 
###  Bir List comprehension işle bu iki sutun hariç yeni bir num_cols oluşturalım 
num_cols = [col for col in df.columns if df[col].dtypes != "O" 
           and col not in "PassengerId"
           and col not in "Survived"]
num_cols
### yukardaki sutunlardan bir tanesi için bir histogram grafiği çizelim
df["Age"].hist()
### Age sutunun birde boxplot grafiğine bakalım
sns.boxplot(x="Age",data=df)
### bir df te verilen lsitedeği sayısal sutunların hepsinin hist grafiğini çizdiren fonksiyon
def hist_for_nums(data, numeric_cols):
    
    col_counter = 0 
    
    for col in numeric_cols: ## verilen listedeki sayısal sutunları tek tek döndüren for döngüsü
        
        data[col].hist() ## histogram çizen satır
        plt.xlabel(col) ## x düzleminin ismi
        plt.title(col) ## grafiğin başlığı
        plt.show() ## oluşan grafiğin gösterilmesini sağlayan show() fonksiyonu
        
        col_counter += 1 ## her döngüde değeri 1 arttırıyoruz
        
    print(col_counter, "variables have been plotted") ## kaç değişkenin grafiğini raporladığımız satır.
hist_for_nums(df, num_cols)
df.head()
df["Survived"].value_counts()
#df teki bir adet kategorik değişkene göre target analizi
df.groupby("Pclass")["Survived"].mean()
def target_summary_with_cat(data, target):
    """
    data = Fonksiyon verilecek dataframe i temsil etmekte (pandas.dataframe olmak zorunda)
    target = fonksiyona verilen dataframe içindeki sutunlardan target olarak seçecek olduğumuz sutunun adı(string olarak yazılmalı)
    """    
    cats_names = [col for col in data.columns if len(data[col].unique()) < 10 and col not in target] ## verilen dataframede bulunan kategorik değişken sutunlarını bulmak 
    
    for var in cats_names:
        print(pd.DataFrame({"TARGET_MEAN": data.groupby(var)[target].mean()}), end="\n\n\n") ## bulunan kategorik değişkenlere göre target analizinin raporlandığı kod bloğu
target_summary_with_cat(df, "Survived")
num_cols
## Bir sayısal Değişkene Göre Targıt analizi ==> agg() fonksiyonun kullanımı
## agg() fonksiyonun içine bir sözlükle atama yapmaktayız sözlüğümüzün key değeri işlem yapılacak sutunun ismini value değeri ise bu sutuna uygulanacak işlemei referans olarak almaktadır.
df.groupby("Survived").agg({"Fare": "mean" , "Age":"min", "Parch":"median"})
## Sayısal Değişkenlerin target a göre raporlamasıını yapan fonksiyon yazılımı
## corr() fonskiyonu içine yazılan dataframe deki type i sayısal olan surunların bir biri ile olan korelasyonunu bize göstermektedir.
df.corr()
df.head(n=3)
df.describe().T
sns.boxplot(x=df["Age"])
### Grafikte çizgiler bizim verimizin quantile larını temsil etmekte şimdi bunlara göre aykırı gözlemlerimizi bulacağız
df["Age"].quantile(0.25)
df["Age"].quantile(.5)
df["Age"].quantile(0.75)
q1=df["Age"].quantile(0.25)
q3=df["Age"].quantile(0.75)
iqr = q3-q1
up = q3 + 1.5 * iqr
low = q1 - 1.5 *iqr
up
low
### Aşağıdaki kod bloğu ile df deki Age sutunun aykırı gözlem olan değerlerine eriştik 
df[(df["Age"]<low) | (df["Age"]> up)][["Age"]]
### Aşağıdaki kod bloğu ile df deki Age sutunun aykırı gözlem olan değerlerinin haç adet olduğunu öğrenmek için shape parametresinin return ettiği 2 elemanlı tuple dan 0. indexe erişirsek buluruz
df[(df["Age"]<low) | (df["Age"]> up)][["Age"]].shape[0]
### Bir sutunda aykır gözlem var mı yok mu diye sorgulamak için 
df[(df["Age"]<low) | (df["Age"]> up)].any(axis=None)
#Bir sutunun aykırı gözlemlerinin hangi değerlerden küçük veya büyük olduğunda aykırı gözlem kabul edildiğini anlatan fonksiyon
def outlier_thresholds(dataframe, variable):
    ## 2 değişken almakta 
    ## dataframe = çalıştığımız dataframi temsil eden değişken => pandas.DataFrame olmalı 
    ## variable = dataframe de çalışmak istediğimiz sutun ismi ==> string olmalı 
    quartile1 = dataframe[variable].quantile(0.25) ## ilgili sutunun 0.25 quantile değeri
    quartile3 = dataframe[variable].quantile(0.75)## ilgili sutunun 0.75 quantile değeri
    interquantile_range = quartile3 - quartile1 ## sutundaki 0.75 ile 0.25 değerlerin farkı
    up_limit = quartile3 + 1.5 * interquantile_range ## aykırı gözlemleri belirlem için kullancağımız üst limit değeri hesaplaması 
    low_limit = quartile1 - 1.5 * interquantile_range ## aykırı gözlemleri belirlem için kullancağımız alt limit değeri hesaplaması 
    return low_limit, up_limit ## alt ve üst limitimiz return eden kısım 
outlier_thresholds(df, "Fare")
## fonksiyonumuz bize 2 adet değer atadığından bunları tek satırda 2 değişkene atamamız mümkündür
low,up = outlier_thresholds(df, "Age")
low
up
## Bir sutunda aykırı gözlem var mı yok mu onu return eden fonksiyon  
def has_outliers(dataframe, variable):
    ## 2 değişken almakta 
    ## dataframe = çalıştığımız dataframi temsil eden değişken => pandas.DataFrame olmalı 
    ## variable = dataframe de çalışmak istediğimiz sutun ismi ==> string olmalı 
    low_limit, up_limit = outlier_thresholds(dataframe, variable) ## daha önceden olusturduğumuz outlier_thresholds fonskiyonumuzu bu fonksiyonun içinde oluşturduk ve low ve up değerlerini 2 değişkene atadık
    if dataframe[(dataframe[variable]<low_limit) | (dataframe[variable]> up_limit)].any(axis=None): ## bulunan low ve up değerlere göre bir koşul olusturduk 
        print(variable, "yes")## sonuç true dönerse yes diye bir çıktı alacağız
has_outliers(df,"Age")
### List Comprehantion ile bir değişken olusturacağız ve df te bulunan sutunlardan sayısal olup ueşsiz değeri 10 dan fazla olan sutunları bulacağız 
### Fakat Passenger Id bize targetımızın olusmasında bir katkı sağlamadığından  onun bu lsite içinde olmasını istemiyoruz
num_names = [col for col in df.columns if len(df[col].unique()) > 10
             and df[col].dtypes != 'O'
             and col not in "PassengerId"]
num_names
## Bir for döngüsü ile nu sutunlarda aykırı gözlem var mı yok mu oluşturduğumuz has_outliers fonksiyonu ile bakıyoruz
for col in num_names:
    has_outliers(df, col)
### has_outliers fonksiyonumuzu biraz daha güzelleştirelim :)
def has_outliers(dataframe, num_col_names, plot=False):
    ### fonksiyonumuzun 3 adet argumanı var 1 tanesi ön tanımlı olarak verilmiş yanı eğer biz plot özelliğini vermezsek fonksiyon ön tanımlı olan değer olarak False kabul edecek bu parametreyi
    ## dataframe = çalıştığımız dataframi temsil eden değişken => pandas.DataFrame olmalı 
    ## num_col_names = sayısal değişkenlerimizin isimlerini barındıran bir liste
    
    variable_names = []
    
    for col in num_col_names:
        low_limit, up_limit = outlier_thresholds(dataframe, col)## daha önceden olusturduğumuz outlier_thresholds fonskiyonumuzu bu fonksiyonun içinde oluşturduk ve low ve up değerlerini 2 değişkene atadık
        
        if dataframe[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)].any(axis=None): ## fonksiyona verdiğimiz listedeki sutunlardan aykırı gözlem içeren varsa
            number_of_outliers = dataframe[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)].shape[0] ## bu sutunda kaç aykırı değer var onu buluyoruz 
         
            print(col, ":", number_of_outliers) ## sutundaki aykırı gözlem değerini raporluyoruz
            variable_names.append(col) ##sutunun ismini fonksiyonun basında olusturduğumuz variable_names listesine ekliyoruz
            
            if plot: ## eğer fonksiyonun parametrelerinden 3 parametre True olarak girilirse
                sns.boxplot(x=dataframe[col]) ## bize bir boxplot çizdiriyor
                plt.show()## ve bunu IDE de görüntülememizi sağlıyor jupyter için gerek li değildir.
                
    return variable_names ## fonksiyonumuz bize verdiğimiz listede aykırı gözlem olan sutunların ismini bir liste ile döndürüyor.
has_outliers(df, num_names)
has_outliers(df, num_names, True)
## Aykırı Gözlemlerin olmadığı bir dataframe nasıl olusturulur. Ve bunu yapan fonksiyonun yazımı
## yazdığımız outkier_treshold fonksiyonu bize sutunumuzdaki aykırı gözlemleri belirleyeceğimiz low ve up değerleri vermekteydi
low, up = outlier_thresholds(df, "Age")
up
## aykırı değerlerimizin olduğu satırları aşağıdaki kod ile bulmuştuk
df[((df["Age"] < low) | (df["Age"] > up))]
## ilk df ten sonra açılan köseli parantezden hemen sonra konulacak ~ (windows için Alt Gr +ü ye aynı anda bastıktan sonra yazılacak ilk tuş sonrası kendiliğinden çıkmaktadır) 
## bu satırlar hariç diğer satırlar anlamına gelmektedir

df[~((df["Age"] < low) | (df["Age"] > up))]
df.shape ## tüm değerler varken dataframe mizde 891 satır 12 sutun bulunmaktadır
### peki bu değerleri silecek fonksiyon nasıl yazılır ? 
def remove_outliers(dataframe, variable):
    ## dataframe = çalıştığımız dataframi temsil eden değişken => pandas.DataFrame olmalı 
    ## variable = dataframe de çalışmak istediğimiz sutun ismi ==> string olmalı 
    low_limit, up_limit = outlier_thresholds(dataframe, variable) ## daha önceden olusturduğumuz outlier_thresholds fonskiyonumuzu bu fonksiyonun içinde oluşturduk ve low ve up değerlerini 2 değişkene atadık
    df_without_outliers = dataframe[~((dataframe[variable] < low_limit) | (dataframe[variable] > up_limit))] ## yeni bir dataframe olusturduk ve bunu içine çalışmak istediğimiz sutunun aykırı gözlemleri hariç olan
                                                                                                             ## dataframe e eşitledik
    return df_without_outliers ## oluşan dataframe i return parametresi ile döndürki bir değişkene fonksiyon kullanıldığında atayabilelim.
df1 = remove_outliers(df,"Age")
df1.shape
## bu fonksiyonu oluşturduğumuz num_cols için kullanmak istersek bir for döngüsü ile bunu yapabiliriz
for col in num_cols:
    new_df = remove_outliers(df,col)
new_df.shape
df.info()
# Dataframe de eksik değer var mı yok mu ?
df.isnull().values.any()
# Dataframe deki sutunlarda eksik değerlerin olup olmadığının sorgulanması
df.isnull().sum()
# Hangi sutunlarda eksik değer var bulunması
df.columns[df.isnull().any()]
# değişkenlerdeki değerlerin toplamının bulunması
df.isnull().sum().sum()
#en az bir tane eksik degere sahip olan gözlem birimleri
df[df.isnull().any(axis=1)]
# tam olan gözlem birimleri
df[df.notnull().all(axis=1)]
### Sutunlardaki eksik verinin oransal olarak görülmesi ve fazladan aza doğru sıralama
(df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)
# Dataframe deki sutunlarda eksik değer olan sutunların raporlamasını yapan fonksiyonun yazımı
def missing_values_table(dataframe):
    ## fonksiyonun parametresi bir dataframe olmalı
    variables_with_na = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0] ## dataframe de eksik veri olan sutunların adlarının bir lsiteye alınması işlemi
    
    n_miss = dataframe[variables_with_na].isnull().sum().sort_values(ascending=False) ## eksik gözlem olan sutunların kaç eksik değere sahip olduğunun bulunması ve çok olandan az olana doru sıralanması 
    ratio = (dataframe[variables_with_na].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False) ## Sutundaki eksik verinin yüzde olarak ifade edilmesi için gerekli hesaplama
    
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])## missing df seklinde bir datafram olusturmak raporlama için
    print(missing_df) ## raporlamanın yazdırıldığı kısım
    return variables_with_na ## eskik gözleme sahip sutunların fonksiyon la geri döndürülmesi
cols_with_na = missing_values_table(df) ### aşağıda cıktı fonksiyonun print kısmından çıkan kısım 
cols_with_na ## fonksiyonun return ettiği liste bir değişkene atanmıs durumda
## Ne yapacağız peki bu eksik değerlere ? 
## 1) Kullanacağımız model bir ağaç yöntemi ise dokunmayacağız
## 2) Silebiliriz.
## 3) Basit Doldurma Yöntemleri Kullanabiliriz
## silme yöntemi
df.shape
df.dropna()
df.shape
## basit doldurma yöntemleri ==> fillna() boş değerleri doldurmak istediğiniz değeri atıyoruz ilk parametre olarak sonrasında kalıcı olması için inplace parametresini True Yapıyoruz.
df["Age"].fillna(df["Age"].mean(), inplace = True)


## kayıp değerleri doldurma işlemi basarılı oldumu diye daha önceden yazdığımız missing_values_table fonksiyonu ile kontrol sağlıyoruz.
missing_values_table(df)
df["Sex"].head()
### Label Encoding yapabilmek için yüklenmesi gereken Kütüphane
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
len(df["Sex"].value_counts()) ## label encoder kullanabilmek için sutunda max 2 eşsiz değerden olmalı
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit_transform(df["Sex"])
df["Sex"].head()
## eğer bu değerleri eski haline geri çevirmek istersek 
label_encoder.inverse_transform([0,1])
#dataframe de en fazla 2 eşsiz değeri olan sutunlara label encoding uygulayan fonksiyon

def label_encoder(dataframe):
    labelencoder = preprocessing.LabelEncoder() ## burada label encoding yapacak değişkenimizi sklearn kütüphanesinde nulunan preprocessing.LabelEncoder() türettik

    label_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O" ## dataframe de Hem kategorik değişken hemde sutundaki esşiz değeri 2 ye eşit olan sutun isimlerin
                  and len(dataframe[col].value_counts()) == 2]                     ## aldık

    for col in label_cols:
        dataframe[col] = labelencoder.fit_transform(dataframe[col]) ## for döngüsü ile bu sutunlara label encoding uyguladık
    return dataframe ## fonksiyon bize label encoding yapılmış yeni dataframe i atamamız için döndürdü
df.head()
df=label_encoder(df)
df.head(n=3)
## 2 den fazla değere sahip sutun alalım 
df["Embarked"].value_counts()
df.shape
df = data.copy()
df.head()
pd.get_dummies(df,columns=["Sex"],drop_first=True).head()
pd.get_dummies(df,columns=["Embarked"]).head()
pd.get_dummies(df,columns=["Embarked"],drop_first=True).head()
pd.get_dummies(df,columns=["Embarked"],dummy_na = True).head()
## 2 den fazla eşsiz değere sahip kategorik değişkenlere one hot encoding yapacak fonksiyon 
def one_hot_encoder(dataframe, category_freq=10, nan_as_category=False):
    """
    dataframe = bir pandas dataframe i fonksiyona tanımlanmalı
    category_freq= bir sutununu en fazla kaç eşsiz değere sahip olursa kategorik değişken sayılacağını belirlediğimiz arguman int değer atanmalı
    nan_as_category = Sutundaki nan değerlere one hot encoding uygulanıp uygulanmayacağını kara vermekte olan arguman 
    """    
    categorical_cols = [col for col in dataframe.columns if len(dataframe[col].value_counts()) < category_freq
                        and dataframe[col].dtypes == 'O'] ## kategorik değişkenlerimiz belirlediğimiz list comprehension kodu
    
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, dummy_na=nan_as_category, drop_first=True) ##belirlenen kategorik değişkenlere one hot encoding uygulanan kod blogu
    
    return dataframe ##fonksiyonumuz bize one hot encoding uygulanmıs bir dataframe döndürüyor.
df = one_hot_encoder(df)
df.head()
df=data.copy()
100*(df["Parch"].value_counts()/len(df))
### Parch sutununu kategorik bir değişken olarak düşünelim ve buradaki %9 aşağısındaki değerleri birleştirip bunların tamamına rare sınıfını atayalım 
df["Parch"].unique()
len(df["Parch"].unique())
rr = df["Parch"].value_counts()/len(df)
rr
rare_labels = rr[rr<0.08].index
rare_labels
### dataframe de Parch sutununda ki 3,4,5,6 değerlerin yerine hepsine rare atayalım

df["Parch"] = np.where(df["Parch"].isin(rare_labels),"RARE",df["Parch"])
df["Parch"].value_counts()/len(df)
def rare_encoder(dataframe, rare_perc):
    
    tempr_df = dataframe.copy() ## uygulayacağımız df te değişiklik olmasın diye bir değişkene kopyalıyoruz

    rare_columns = [col for col in tempr_df.columns if tempr_df[col].dtypes == 'O'
                    and (tempr_df[col].value_counts() / len(tempr_df) < rare_perc).any(axis=None)] ## rare encodin yapacağımız kategorik değişkenlerimizin isimlerini bir listeye alıyoruz

    for var in rare_columns: 
        tmp = tempr_df[var].value_counts() / len(tempr_df) ## sutundaki sınıfların frekanslarını buluyoruz
        rare_labels = tmp[tmp < rare_perc].index ## for döngüsü ile sutundaki rare encoding yapacağımız sınıfları belirliyoruz
        tempr_df[var] = np.where(tempr_df[var].isin(rare_labels), 'Rare', tempr_df[var]) ## frekansı düşük olan sınıfların yerine "Rare" olarak atıyoruz

    return tempr_df
df=data.copy()
df["Parch"] = df["Parch"].apply(lambda x: str(x)) ### object olup değişikliği görmek için yapılan bir işlem
df.nunique()
df1=rare_encoder(df,0.08)
df1.nunique()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(df[["Age"]])
df.describe().T
df["Age"] = scaler.transform(df[["Age"]])
df.describe().T
df=data.copy()
from sklearn.preprocessing import RobustScaler

transformer = RobustScaler()
transformer.fit(df[["Age"]])
df["Age"] = transformer.transform(df[["Age"]])
df.describe().T
df=data.copy()
from sklearn.preprocessing import MinMaxScaler

transformer = MinMaxScaler((-10, 10)).fit(df[["Age"]])
df["Age"] = transformer.transform(df[["Age"]])  # on tanımlı değeri 0 ile 1 arası.
df.describe().T
df.head()
df = data.copy()
df.head()
df.loc[((df['SibSp'] + df['Parch']) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df['SibSp'] + df['Parch']) == 0), "NEW_IS_ALONE"] = "YES"
df.head()
