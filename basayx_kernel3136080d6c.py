from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Distribution graphs (histogram/bar graph) of column data
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()

# Correlation matrix
def plotCorrelationMatrix(df, graphWidth):
    filename = df.dataframeName
    df = df.dropna('columns') # drop columns with NaN
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.show()
# Scatter and density plots
def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include =[np.number]) # keep only numerical columns
    # Remove rows and columns that would lead to df being singular
    df = df.dropna('columns')
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    columnNames = list(df)
    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.show()

nRowsRead = 1000 # specify 'None' if want to read whole file
# tablet.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
df1 = pd.read_csv('/kaggle/input/tablets/tablet.csv', delimiter=',', nrows = nRowsRead)
df1.dataframeName = 'tablet.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')

x=df1.drop('FiyatAraligi', axis=1)
y=df1['CekirdekSayisi']

df1.head(5)
plotPerColumnDistribution(df1, 10, 5)
plotCorrelationMatrix(df1, 8)
plotScatterMatrix(df1, 20, 10)

#  value_counts() fonksiyonu ile veri çerçevemizin ne kadar dengeli dağıldığını sorgulayalım.
df1["FiyatAraligi"].value_counts()

#  Keman grafiği çizdirerek FiyatAraligi değişkeninin dağılımını inceleyelim.
sns.violinplot(z=df1['FiyatAraligi'], data = df1);


# Şimdi sayısal değerler arasında korelasyon matrislerini gösterip, anlamlı değerleri gözlemleyelim.
df1.corr() 
#  Örneğin Kalinlik ve Agirlik arasında pozitif bir ilişki olduğu söylenebilir. Bunu kontrol etmek için matrisini çizdirelim: 
df1.groupby('Kalinlik')['Agirlik'].apply(lambda x: np.mean(x))

df1.corr()["Kalinlik"]["Agirlik"] # Görüldüğü gibi pozitif bir değer elde edilir.
#  Benzer bir şekilde BataryaGucu ve MikroislemciHizi değerleri arasında olduğu gibi bazı değişkenler
# arasında da pozitif ilişki gözlenebilir.
df1.groupby('BataryaGucu')['MikroislemciHizi'].apply(lambda x: np.mean(x))
df1.corr()["BataryaGucu"]["MikroislemciHizi"]
#  Ancak her değişken arasında pozitif bir ilişki bulunmaz.
#  Örneğin BataryaGucu ile Agirlik arasında negatif bir ilişki vardır.

df1.corr()["BataryaGucu"]["Agirlik"]

#  En çok hangi fiyat sınıfına ait tabletin bulunduğunu görelim.
df1['FiyatAraligi'].mode()
#  Aynı şeyi MikroislemciHizi için de yapalım.
df1['MikroislemciHizi'].mode()
#  MikroislemciHizi için ortada kalan değeri görelim.
df1['MikroislemciHizi'].median()
#  Tüm tabletler arasında ortalama mikro işlemci hzını bulalım.
df1['MikroislemciHizi'].mean()
#  Ayrıca diğer tüm sayısal değişkenler için de ortalama değerlerini listeleyebiliriz.
df1.mean(axis = 0, skipna = True)
#  Son olarak, MikroislemciHizi değişkeninin standart sapma değerini görelim.
df1['MikroislemciHizi'].std()

#  Veri setindeki eksik değerler var mı, onu gözlemleyelim.
eksik_verilerin_sayisi = df1.isnull().sum()
eksik_verilerin_sayisi  # Gördüğünüz gibi az da olsa bir takım ekisik veriler bulunuyor.

#Toplam kaç tane eksik verimiz var ona bakalım:
toplamVeriMiktari = np.product(df1.shape)
toplamEksiklikMiktari = eksik_verilerin_sayisi.sum()
#Eksik verilerin yüzde(%) olarak ne kadar olduğuna bakalım
veriSetineOranla_eksiklikYuzdesi = (toplamEksiklikMiktari/toplamVeriMiktari) * 100
veriSetineOranla_eksiklikYuzdesi #Sadece %0.06 oranında bir eksiklik var.
# Yine de bu miktar bile veri analizinde kritik sonuçlara sebep olabilir. Bu sebepten bu eksiklikleri tamamlayalım...
# Eksik kısımları doldurmak için önce onları veri setinden ayırmamız gerekiyor.
# Tüm satırlardaki eksik(NaN) değerlerini siliyoruz
nonull_data = pd.DataFrame(df1)
nonull_data.dropna()
#  Bu işlemlerden sonra kalan veri setimizin durumunu kıyaslayalım.
print("Orijinal veri setinin sütunları : %d \n" % df1.shape[1])
print("Eksik verileri çıkarıldıktan sonraki sütun sayısı : %d" % nonull_data.shape[1])

#  Veri setimizde OnKameraMP ve RAM sütunlarında eksiklikler vardı bu sebepten bu kısımlar kaldırılacaktır.
#  Şimdi boş olarak belirttiğimiz bu kısımları dolduralım.

nonull_data["OnKameraMP"].fillna(0 ,inplace = True)
nonull_data["RAM"].fillna(0 ,inplace = True)

nonull_data.isnull().sum() # Görüldüğü üzere artık boş verimiz kalmamıştır.

# Şimdi bağımlı ve bağımsız değişkenlerimizi tanımlayarak devam edelim.
from sklearn.preprocessing import scale 
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score, roc_curve, recall_score, f1_score, precision_score
from sklearn.naive_bayes import GaussianNB
#  Sırada, tanımladığımız bilgiler üzerinden GaussianNB modelini uygulayacağız.
#  Ama bundan önce bu modeli ve diğer yapacğaımız modeli kullanabilmek için veri setimizdeki değerleri 
# nümerik hale getireceğiz.

data = pd.DataFrame(nonull_data)
varYok_mapping = {"Yok": 0, "Var": 1}
data['Bluetooth'] = data['Bluetooth'].map(varYok_mapping)
data['CiftHat'] = data['CiftHat'].map(varYok_mapping)
data['4G'] = data['4G'].map(varYok_mapping)
data['3G'] = data['3G'].map(varYok_mapping)
data['Dokunmatik'] = data['Dokunmatik'].map(varYok_mapping)
data['WiFi'] = data['WiFi'].map(varYok_mapping)

fiyat_mapping = {"Çok Ucuz": 0, "Ucuz": 1, "Normal": 2, "Pahalı": 3}
data['FiyatAraligi'] = data['FiyatAraligi'].map(fiyat_mapping)
# Renkler için bir üstünlük ilişkisi kuramayacağımızdan onları farklı bir şekilde nüremik hale geriyoruz.

data = pd.concat([data, pd.get_dummies(data['Renk'], prefix='Renk')], axis=1)
data = data.drop('Renk',axis=1)
data

# Bu işlemlemden sonra artık modellerimizi uygulamaya başlayabiliriz.
# İlk önce GuassionNB modelini uygulayalım.
y = data['FiyatAraligi']
X = data.drop(['FiyatAraligi'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.25, random_state = 42)
nb = GaussianNB()
nb_model = nb.fit(X_train, y_train)
y_pred = nb_model.predict(X_test)

accuracy_score(y_test, y_pred)
karmasiklik_matrisi = confusion_matrix(y_test, y_pred)
print(karmasiklik_matrisi)
cross_val_score(nb_model, X_test, y_test, cv = 10)
cross_val_score(nb_model, X_test, y_test, cv = 10).mean()
(karmasiklik_matrisi[0][0] + karmasiklik_matrisi[1][1]) / (karmasiklik_matrisi[0][0] + karmasiklik_matrisi[1][1] +  karmasiklik_matrisi[1][0] + karmasiklik_matrisi[0][1])
accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
PrecisionScore = precision_score(y_test, y_pred, average='weighted')
PrecisionScore
RecallScore = recall_score(y_test, y_pred, average='weighted')
RecallScore
F1Score = f1_score(y_test, y_pred, average = 'weighted')  
F1Score

# Şimdi DecisionTree modelimizi uygulayalım.
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, auc, roc_curve, classification_report
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score, f1_score, precision_score
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn.metrics import confusion_matrix as cm
from matplotlib.legend_handler import HandlerLine2D
cart = DecisionTreeClassifier(random_state = 42)
cart_model = cart.fit(X_train, y_train)
cart_model


!pip install skompiler
!pip install graphviz
!pip install pydotplus

from skompiler import skompile
print(skompile(cart_model.predict).to("python/code"))
y_pred = cart_model.predict(X_test)
accuracy_score(y_test, y_pred)
karmasiklik_matrisi = confusion_matrix(y_test, y_pred)
print(karmasiklik_matrisi)
cart_grid = {"max_depth": range(1,20),
            "min_samples_split" : range(2,100)}
cart = DecisionTreeClassifier()
cart_cv = GridSearchCV(cart, cart_grid, cv = 3, n_jobs = -1, verbose = 2)
cart_cv_model = cart_cv.fit(X_train, y_train)
print("En iyi parametreler : " + str(cart_cv_model.best_params_))
print("En iyi skor : " + str(cart_cv_model.best_score_))
cart = DecisionTreeClassifier(max_depth = 7, min_samples_split =95)
cart_tuned = cart.fit(X_train, y_train)
y_pred = cart_tuned.predict(X_test)
accuracy_score(y_test, y_pred)
karmasiklik_matrisi = confusion_matrix(y_test, y_pred)
print(karmasiklik_matrisi)
cross_val_score(cart_tuned, X_test, y_test, cv = 10)
cross_val_score(cart_tuned, X, y, cv = 10).mean()

ROC_AUC_Score = roc_auc_score(y_test, y_pred)
ROC_AUC_Score
logit_roc_auc = roc_auc_score(y_test, cart_model.predict(X_test))

fpr, tpr, thresholds = roc_curve(y_test, cart_model.predict_proba(X_test)[:,1])
plt.figure(figsize=(10,7))
plt.plot(fpr, tpr, label='AUC (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Oranı')
plt.ylabel('True Positive Oranı')
plt.title('ROC')
plt.show()

max_depths = np.linspace(1, 32, 32, endpoint = True)
train_results = []
test_results = []

for max_depth in max_depths:
    dt = DecisionTreeClassifier(max_depth = max_depth)
    dt.fit(X_train, y_train)
    train_pred = dt.predict(X_train)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    
    train_results.append(roc_auc)
    y_pred = dt.predict(X_test)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    test_results.append(roc_auc)

line1,  = plt.plot(max_depths, train_results, 'b', label = "Train AUC")
line2, = plt.plot(max_depths, test_results, 'r', label = "Test AUC")

plt.legend(handler_map = {line1: HandlerLine2D(numpoints = 2)})
plt.ylabel('AUC score')
plt.xlabel('Tree depth')
plt.show()

print(classification_report(y_test, y_pred))
ranking = cart.feature_importances_
features = np.argsort(ranking)[::-1][:10]
columns = X.columns

plt.figure(figsize = (16, 9))
plt.title("Karar Ağacına Göre Özniteliklerin Önem Derecesi", y = 1.03, size = 18)
plt.bar(range(len(features)), ranking[features], color="lime", align="center")
plt.xticks(range(len(features)), columns[features], rotation = 80)
plt.show()


# Şimdi de K En Yakın Komşuluk modelini uygulayalım.
X=data.drop('FiyatAraligi', axis=1)
y=data['FiyatAraligi']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33, random_state=101)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=12,metric="minkowski")
knn.fit(X_train, y_train)
knn_params = {"n_neighbors": np.arange(1,15)}
knn_cv = GridSearchCV(knn, knn_params, cv = 3)
knn_cv.fit(X_train, y_train)
print("En iyi skor: " + str(knn_cv.best_score_))
print("En iyi parametreler: " + str(knn_cv.best_params_))
# 2'den 15'e kadar olan her sayıyı komşu sayısı olarak deneyip skorlarını listeleyelim.

score_list = []
#range(1,30,5)
for each in range(1,15,1):
    knn2 = KNeighborsClassifier(n_neighbors = each)
    knn2.fit(X_train,y_train)
    y_pred_KNN = knn2.predict(X_test)
    score_list.append(knn2.score(X_test, y_test))
    
plt.plot(range(1,15,1),score_list)
plt.xlabel("k en yakın komşu sayıları")
plt.ylabel("doğruluk skoru")
plt.show()

knn_tuned = KNeighborsClassifier(20)
knn_tuned = knn_tuned.fit(X_train, y_train)
y_pred = knn_tuned.predict(X_test)
accuracy_score(y_test, y_pred)
cross_val_score(knn_tuned, X_test, y_test, cv = 10)
cross_val_score(knn_tuned, X_test, y_test, cv = 10).mean()
ROC_AUC_Score = roc_auc_score(y_test, y_pred)
ROC_AUC_Score
logit_roc_auc = roc_auc_score(y_test, cart_model.predict(X_test))

fpr, tpr, thresholds = roc_curve(y_test, cart_model.predict_proba(X_test)[:,1])
plt.figure(figsize=(10,7))
plt.plot(fpr, tpr, label='AUC (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Oranı')
plt.ylabel('True Positive Oranı')
plt.title('ROC')
plt.show()





