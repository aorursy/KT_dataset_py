import pandas as pd

import numpy as np

import seaborn as sns

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.metrics import roc_auc_score, roc_curve, recall_score, f1_score, precision_score

from sklearn.naive_bayes import GaussianNB





from sklearn.preprocessing import scale 





df = pd.read_csv("../input/tablet/tablet.csv").copy()
df.head()# veri çerçevesinin tüm sütunları ve ilk 5 gözlemini görüntüledik

df.tail() # veri çerçevesinin son 5 gözlemini görüntüledik.
df.sample(5) # rastgele 5 değişkeni yazdırdık.
df.info()
df.dtypes
df.describe().T # Bazı değişkenlerin standart sapmaları yüksek. Bu düzensiz bir dağılımın olduğunu gösterir.

df.shape # 2000 tane gözlemimiz var ve 20 tane değişkenimiz var.
df.count() #değişkenlerde kaç tane değer olduğunu bulduk.
df.isna().sum() #OnKameraMP ve RAM değişkeninde eksik verilerimiz var. 
df.mean() #değişkenlerin ortalamaları
df.std() #standart sapmalara bakıyoruz. Standart saspması yüksek olan çok fazla değişken var. Bu da bize verisetinin dengeziz dagıldıgını söyler.
df.cov() # Kovaryansa bakıyoruz.
df.corr() # Korelasyona bakıyoruz.
corr = df.corr() # Korelasyon ısı haritası çizdirdik. Buraya baktığımızda korelasyon çok düşük. Aralarındaki ilişkiler çok zayıf.

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values); 
df.groupby(["FiyatAraligi"]).mean()
df.groupby(["FiyatAraligi"]).std()
df["FiyatAraligi"].unique()# FiyatAraligi değişkeninin benzersiz değişkenleri
df["FiyatAraligi"].nunique() #FiyatAraligi değişkeninin benzersiz değişken sayısı.
df["FiyatAraligi"].value_counts() #düzenli bir dağılım var
sns.countplot(df["FiyatAraligi"]) # düzenli bir dağılım var 
sns.set(rc={'figure.figsize':(8,6)})  # grafikleri 8, 6 şeklinde boyutlandırmak için yazdık, yazmasak da olur.
sns.scatterplot(x = "CozunurlukYükseklik", y = "CozunurlukGenislik", data = df);
sns.jointplot(x = "CozunurlukYükseklik", y = "CozunurlukGenislik", data = df, color="purple");
sns.scatterplot(x = "OnKameraMP", y = "ArkaKameraMP", data = df);
sns.jointplot(x = "OnKameraMP", y = "ArkaKameraMP", data = df, color="purple");
sns.jointplot(x = df["OnKameraMP"], y = df["ArkaKameraMP"], kind = "kde", color = "purple");
sns.jointplot(x = df["CozunurlukYükseklik"], y = df["CozunurlukGenislik"], kind = "kde", color = "purple");
sns.violinplot(y = "RAM", data = df);
sns.violinplot(y = "MikroislemciHizi", data = df);
df.head()
sns.pairplot(df, hue = "FiyatAraligi", palette="Set2");
sns.barplot(x ="ArkaKameraMP" , y = "RAM" , data = df);
import missingno                  # eksik verileri daha iyi okumak için kullanacağız.

from sklearn import preprocessing   # ön işleme aşamasında label encoding vb. için dahil ettik.

import re                         # regular expression yani düzenli ifadeler kullanmak için dahil ettik.
df.columns
df.head()
df.isnull().sum().sum() # 17 tane eksik değerimiz var.
df.isnull().sum() # hangi değerlerin eksik olduğuna bakıyoruz ve OnKameraMP ve RAM değişkenleri eksik değer içeriyor.
missingno.matrix(df,figsize=(20, 10)); # veri setine bakılınca çok fazla eksik değerimiz yok.
missingno.heatmap(df, figsize= (15,8)); # Bu eksik değer barındıran değişkenler arasında anlamlı bir ilişki var mı ona bakıyoruz.

# RAM ve OnKameraMP değişkenlerinin aynı gözlemde eksik veri barındırmadığını gözlemliyoruz. Bu yok sayılabilecek kadar güçsüz bir ilişki, hatta nötr (ilişki yok) de denilebilir. 
def eksik_deger_tablosu(df): 

    eksik_deger = df.isnull().sum()

    eksik_deger_yuzde = 100 * df.isnull().sum()/len(df)

    eksik_deger_tablo = pd.concat([eksik_deger, eksik_deger_yuzde], axis=1)

    eksik_deger_tablo_son = eksik_deger_tablo.rename(columns = {0 : 'Eksik Değerler', 1 : '% Değeri'})

    return eksik_deger_tablo_son
eksik_deger_tablosu(df).T # OnkameraMP değişkeni % 0.25 ve RAM değişkeni ise %0.26 
df["RAM"].unique()
df["OnKameraMP"].unique()
df[df["RAM"].isnull()] # RAM değişkeninde eksik olan gözlemlere bakıyoruz ve NaN olduğunu görüyoruz.
sns.countplot(df[df["RAM"].isnull()]["FiyatAraligi"]); #Eksik olan RAM değişkenini FiyatAraligi değişkenine göre bakınca tek bir özelliğe bağlı olduğunu görüyoruz. RAM değişkeninin eksik olmasında fiyatının pahalı olmasından dolayı eksik girilmiş olabilir.
sns.countplot(df[df["RAM"].isnull()]["Renk"]); # Renk değişkenine göre bakıyoruz şimdi de ve en çok turuncu renkte eksik değer girilmiş.
sns.countplot(df[df["RAM"].isnull()]["BataryaGucu"]) # BataryaGucu değişkenine göre bakınca 1035 değeriçok etkili gözüküyor.
sns.distplot(df[df["RAM"].isnull()]["BataryaGucu"]);
df[df["OnKameraMP"].isnull()] #OnKameraMP değişkenin eksik değerleri NaN dğerler içeriyor.
sns.countplot(df[df["OnKameraMP"].isnull()]["Bluetooth"])
sns.countplot(df[df["OnKameraMP"].isnull()]["FiyatAraligi"]) # grafiğe bakınca çok ucuz olmasına bağlı. Fiyatı çok ucuz olduğu için Eksik girilmiş olabilir
sns.countplot(df[df["OnKameraMP"].isnull()]["Renk"]) 
df.groupby("FiyatAraligi").mean() # FiyatAraligi değişkenine göre ortalamaları bulduk. 
df.groupby("FiyatAraligi")[["RAM"]].mean()
df[(df["FiyatAraligi"] == "Pahalı") & (df["RAM"].isnull())]
Pahali_RAM = df[(df["FiyatAraligi"] == "Pahalı") & (df["RAM"].isnull())].index

Pahali_RAM
df.loc[Pahali_RAM ,"RAM"] = 3500
df.isna().sum()["RAM"] # RAM değişkeninin eksik değerleri dolduruldu.
df.groupby("FiyatAraligi")[["OnKameraMP"]].mean()
df[(df["FiyatAraligi"] == "Çok Ucuz") & (df["OnKameraMP"].isnull())]
OnKamera_Bos = df[(df["FiyatAraligi"] == "Çok Ucuz") & (df["OnKameraMP"].isnull())].index

OnKamera_Bos 
df.loc[OnKamera_Bos ,"OnKameraMP"] = 4 # OnKameraMP değişkenini çok ucuz kategorisinin ortalaması 4 ile dolduruyoruz.
df.isna().sum()["OnKameraMP"] # OnKameraMP değişkeninin eksik değerleri dolduruldu.
df["RAM"].unique()
df["OnKameraMP"].unique()
df.isna().sum() # Şimdi verisetinde eksik değerimiz kalmadı.
sns.countplot(df["FiyatAraligi"]); 
label_encoder = preprocessing.LabelEncoder()
df['CiftHat'] = label_encoder.fit_transform(df['CiftHat'])

df.head()



# Önemli not: İlk değeri 1 yaptı yani Var : 1, Yok : 0 değerlerini temsil ediyor.
df['Bluetooth'] = label_encoder.fit_transform(df['Bluetooth'])

df.head()

 #İlk değeri 1 yaptı yani Var : 0, Yok : 1 değerlerini temsil ediyor.
df['4G'] = label_encoder.fit_transform(df['4G'])

df.head()

 #İlk değeri 1 yaptı yani Var : 0, Yok : 1 değerlerini temsil ediyor.
df['3G'] = label_encoder.fit_transform(df['3G'])

df.head()

 #İlk değeri 1 yaptı yani Var : 1, Yok : 0 değerlerini temsil ediyor.
df['Dokunmatik'] = label_encoder.fit_transform(df['Dokunmatik'])

df.head()

 #İlk değeri 1 yaptı yani Var : 0, Yok : 1 değerlerini temsil ediyor.
df['WiFi'] = label_encoder.fit_transform(df['WiFi'])

df.head()

 #İlk değeri 1 yaptı yani Var : 0, Yok : 1 değerlerini temsil ediyor.
sns.set(rc={'figure.figsize':(11,8)}) # oluşacak grafiklerin uzunluğunu ve genişliğini belirliyorum.



sns.scatterplot(df["FiyatAraligi"], df["RAM"]);
df.drop(["Bluetooth","CiftHat","4G","3G" ,"WiFi" ,"Renk","Dokunmatik" ], axis = 1, inplace = True) #sayısal olmayan değişkenleri kaldırdık.
X = df.drop("FiyatAraligi", axis = 1)

y = df["FiyatAraligi"]
X  # bağımsız değişken dışında her şey
y # bağımlı değişken FiyatAraligi
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/4, random_state = 3512, shuffle=1)
X_test.head()
y_test.head()
y_test.head()
nb = GaussianNB()

nb_model = nb.fit(X_train, y_train)
nb_model #Modelin alabileceği parametreleri görüntüledik.
dir(nb_model) #Model üzerinde yazılabilecek tüm komutları görüntüledik.
X_test[0:10]
nb_model.predict(X_test)[0:10]
y_test[0:10]
y_pred = nb_model.predict(X_test)

y_pred
y_test
accuracy_score(y_test, y_pred)
karmasiklik_matrisi = confusion_matrix(y_test, y_pred)

print(karmasiklik_matrisi)
(karmasiklik_matrisi[0][0] + karmasiklik_matrisi[1][1]) / (karmasiklik_matrisi[0][0] + karmasiklik_matrisi[1][1] +  karmasiklik_matrisi[1][0] + karmasiklik_matrisi[0][1])
cross_val_score(nb_model, X_test, y_test, cv = 10)
cross_val_score(nb_model, X_test, y_test, cv = 10).mean()
print(classification_report(y_test, y_pred))
PrecisionScore = precision_score(y_test, y_pred, average='weighted')

PrecisionScore
RecallScore = recall_score(y_test, y_pred, average='weighted')

RecallScore
F1Score = f1_score(y_test, y_pred, average = 'weighted')  

F1Score
X.columns
len(X.columns) #kaç öznitelik olduğuna bakalım
from sklearn.feature_selection import *
test = SelectKBest(k = 12)

test
fit = test.fit(X, y)

fit
for indis, skor in enumerate(fit.scores_):

    print(skor, " -> ", X.columns[indis])
from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors=15) 

  

knn.fit(X_train, y_train) 
print(knn.predict(X_test)) #Modelin doğruluğunu hesaplayalım.
knn = KNeighborsClassifier(n_neighbors=15) 

  

knn.fit(X_train, y_train) 

  

#Modelin doğruluğunu hesaplayalım. 

print(knn.score(X_test, y_test))
neighbors = np.arange(2, 15) 

train_accuracy = np.empty(len(neighbors)) 

test_accuracy = np.empty(len(neighbors)) 
#K değerlerinin üzerinde döngü

for i, k in enumerate(neighbors): 

    knn = KNeighborsClassifier(n_neighbors=k) 

    knn.fit(X_train, y_train) 

      

    # Eğitim ve veri doğruluğunu test edelim

    train_accuracy[i] = knn.score(X_train, y_train) 

    test_accuracy[i] = knn.score(X_test, y_test) 

  
# Grafiği çizelim.

plt.plot(neighbors, test_accuracy, label = 'Testing dataset Accuracy') 

plt.plot(neighbors, train_accuracy, label = 'Training dataset Accuracy') 

  

plt.legend() 

plt.xlabel('n_neighbors') 

plt.ylabel('Accuracy') 

plt.show() 
# Function importing Dataset 

def importdata(): 

    balance_data = pd.read_csv('../input/tablet/tablet.csv', sep= ',', header = None) 

    # Printing the dataswet shape 

    print("Dataset Length: ", len(balance_data)) 

    print("Dataset Shape: ", balance_data.shape) 

      

    # Printing the dataset obseravtions 

    print("Dataset: ",balance_data.head()) 

    return balance_data 
# Function to split the dataset 

def splitdataset(balance_data): 

  

    # Separating the target variable 

    X = balance_data.values[:, 2:15] 

    Y = balance_data.values[:, 0] 

  

    # Splitting the dataset into train and test 

    X_train, X_test, y_train, y_test = train_test_split(  

    X, Y, test_size = 0.25, random_state = 100) 

      

    return X, Y, X_train, X_test, y_train, y_test 
# Function to perform training with giniIndex. 

def train_using_gini(X_train, X_test, y_train): 

  

    # Creating the classifier object 

    clf_gini = DecisionTreeClassifier(criterion = "gini", 

            random_state = 100,max_depth=2, min_samples_leaf=15) 

  

    # Performing training 

    clf_gini.fit(X_train, y_train) 

    return clf_gini 
# Function to perform training with entropy. 

def tarin_using_entropy(X_train, X_test, y_train): 

  

    # Decision tree with entropy 

    clf_entropy = DecisionTreeClassifier( 

            criterion = "entropy", random_state = 100, 

            max_depth = 2, min_samples_leaf = 15) 

  

    # Performing training 

    clf_entropy.fit(X_train, y_train) 

    return clf_entropy 
# Function to make predictions 

def prediction(X_test, clf_object): 

  

    # Predicton on test with giniIndex 

    y_pred = clf_object.predict(X_test) 

    print("Predicted values:") 

    print(y_pred) 

    return y_pred 
# Function to calculate accuracy 

def cal_accuracy(y_test, y_pred): 

      

    print("Confusion Matrix: ", 

        confusion_matrix(y_test, y_pred)) 

      

    print ("Accuracy : ", 

    accuracy_score(y_test,y_pred)*100) 

      

    print("Report : ", 

    classification_report(y_test, y_pred)) 
# Driver code 

def main(): 

      

    # Building Phase 

    data = importdata() 

    X, Y, X_train, X_test, y_train, y_test = splitdataset(data) 

    clf_gini = train_using_gini(X_train, X_test, y_train) 

    clf_entropy = tarin_using_entropy(X_train, X_test, y_train) 

      

    # Operational Phase 

    print("Results Using Gini Index:") 

      

    # Prediction using gini 

    y_pred_gini = prediction(X_test, clf_gini) 

    cal_accuracy(y_test, y_pred_gini) 

      

    print("Results Using Entropy:") 

    # Prediction using entropy 

    y_pred_entropy = prediction(X_test, clf_entropy) 

    cal_accuracy(y_test, y_pred_entropy) 
# Calling main function 

if __name__=="__main__": 

    main() 