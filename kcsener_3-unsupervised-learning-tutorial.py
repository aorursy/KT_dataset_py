# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
"""

K-means Clustering ile, modelde belirtilen cluster sayısı kadar cluster bulur datanın içinde.

"""
#import data

df = pd.read_csv('../input/Iris.csv') 
#see data

df.head()
#drop columns (feature olmayanları)

df.drop(['Id', 'Species'], axis=1, inplace=True) #sadece featurelar kaldı, buradan cluster bulucaz şimdi.
#tekrar bak

df.head()
#import module

from sklearn.cluster import KMeans
#algoritmayı variable'a ata

model = KMeans(n_clusters=3) #istediğimiz cluster sayısını n_cluster= parametresine argüman atıyoruz.
#df datasetini modelde fit et (train et)

model.fit(df)
#train edilmiş datanın clusterlarını predict et

labels = model.predict(df)
print(labels)
#yeni data geldi, cluster edelim

new_sample = [[5.7,4.4,1.5,0.4], [6.5, 3, 5.5, 1.8]]
#modele yerleştir yeni datayı ve predict et

new_labels = model.predict(new_sample)
print(new_labels)
#iris'in iki feature'ını visualize edelim cluster yapılmış şekilde:

#0 ve 2. columndaki featureları alıyoruz

xs = df.iloc[:, 0] 

ys = df.iloc[:, 1]

plt.scatter(xs, ys, c=labels, alpha=0.5) #c-->color, colorı cluster labellara göre kendin belirle dedik.





#bundan sonrasında ise, centroid dediğimiz, cluster'ların merkez noktalarını da belirtelim:



# Assign the cluster centers: centroids

centroids = model.cluster_centers_



# Assign the columns of centroids: centroids_x, centroids_y

centroids_x = centroids[:,0]

centroids_y = centroids[:,1]



# Make a scatter plot of centroids_x and centroids_y

plt.scatter(centroids_x, centroids_y, marker='D', s=50)

plt.show()
"""

Modelin performansını görmemiz gerekir. şu an elimizde prediction'lar ve gerçek clusterlar var.

bunları direkt olarak crosstab() metodunu kullanarak kıyaslayabiliriz:



"""
#import data (species column'ını almak için yeniden import ettik.)

df = pd.read_csv('../input/Iris.csv') 
df.head()
df['Species'].head()
#prediction ve real value'dan oluan df oluşturuyoruz daha net incelemek için:



df_new = pd.DataFrame({'labels':labels, #daha önce labels adında bir variable'a predictionları atamıştık

                       'species':df['Species']}) #şimdi de df'nin species(yani real values)'ını df_new'in diğer column'ı olarak atadık. ve df_new'i oluşturduk.
df_new.head() #ilk column prediction, ikinci column gerçek değerler
#cross-tab ile label ve species'i görelim:

ct = pd.crosstab(df_new['labels'], df_new['species'])
print(ct)



"""

setosa 50'de 50 doğru cluster edilmiş

versicolor 48'i doğru. 2 adeti virginica olarak tahmin edilmiş

virginica 36 doğru. 14 adeti versicolor olarak tahmin edilmiş.

"""
"""

cluster sayısının idealliğini konuşuyoruz evet.

bunun için 'inertia_' denilen bir ölçüm birimini kullanacağız.



inertia, herbir rowdatanın bağlı bulunduğu cluster'a olan uzaklıklar toplamını ölçer diyebiliriz.

yani, inertia'nın cluster sayısı da arttıkça azalacağını söyleyebiliriz.



örneğin, rowdata sayısı kadar cluster olsa, her cluster için centroidler aynı zamanda datanın kendisi 

olacağı için, inertia= 0 olacaktır.



ancak bu feasible değildir elbette. cluster sayısının optimum değerine ulaşmalıyız.

"""
#yukarıdaki model için, inertia değeri:

#bu metodu modülü fit() ettikten sonra kullanabiliriz.

model.inertia_ #model aslında verdiğimiz cluster sayısına göre inertia değerini minimize edecek şekilde cluster eder.
#drop columns (feature olmayanları)

df.drop(['Id', 'Species'], axis=1, inplace=True) #sadece featurelar kaldı, buradan cluster bulucaz şimdi.
#inertia plot çizeceğiz. elbow rule'a göre, inertia'nın daha yavaş azaldığı nokta bizim için ideal cluster sayısını verecektir.





ks = range(1, 10) #denenecek cluster değerleri 

inertias = [] #inertia değerlerini burada toplayacağız



for k in ks:

    # Create a KMeans instance with k clusters: model

    model = KMeans(n_clusters=k)

    

    # Fit model to samples

    model.fit(df)

    

    # Append the inertia to the list of inertias

    inertias.append(model.inertia_)

    

# Plot ks vs inertias

plt.plot(ks, inertias, '-o')

plt.xlabel('number of clusters, k')

plt.ylabel('inertia')

plt.xticks(ks)

plt.show()



#bu grafiğe göre, cluster sayısı = 3 olmalı.
"""

feature'ların clusterlarına göre variance'larının yüksek olması durumunda kmeans doğru sonuçlar vermez.

bu durumun tölere edilmesi gerekir.



"""
"""

feature'ların variance'ları eşit olacak şekilde transform edilmesi işlemi için StandardScalar() kullanılır.

Bu metod ile, her feature'ın mean'i 0, variance'ı 1'e eşitlenir.



"""
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df)
df_scaled = scaler.transform(df) #feature'ları burada scaled olacak şekilde dönüştürdük.
#artık, scaled data ile kmeans yaparak cluster yapabiliriz:

#burada istersek önce scale sonra kmeans yaparız, 





#ya da pipeline kullanarak bunu aynı anda gerçekleştirebiliriz:



from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans



scaler = StandardScaler()

kmeans = KMeans(n_clusters=3)



from sklearn.pipeline import make_pipeline



pipeline = make_pipeline(scaler, kmeans)



pipeline.fit(df)

labels = pipeline.predict(df)

#import data (species column'ını almak için yeniden import ettik.)

df = pd.read_csv('../input/Iris.csv') 
#crosstabulation ile performansa tekrar bakıyoruz:

df_new = pd.DataFrame({'labels':labels, #daha önce labels adında bir variable'a predictionları atamıştık

                       'species':df['Species']}) #şimdi de df'nin species(yani real values)'ını df_new'in diğer column'ı olarak atadık. ve df_new'i oluşturduk.



ct = pd.crosstab(df_new['labels'], df_new['species'])

ct

#dataset uygun değil, güzel sonuç vermedi ama standardizasyon(scaling) yapmadan kmeans yapmamalıyız. 
"""

Hierarchical Clustering'de, 

    herbir data point önce bir cluster olarak düşünülür. 

    sonra, birbirine en yakın iki nokta (aslında cluster) ile bir cluster oluşturur. 

    sonra en yakın iki cluster'ı tekrar bir cluster yap.

    aynı işlem 1 cluster kalana kadar tekrarlanır.



Dendogram:

    datapoint(x)-distance(y) grafiği.

    önce napmıştık, her bir data point bir cluster olarak düşünüldü.

    en yakın iki point(cluster) bir cluster oluşturdu. 

    bunu yaparken, aralarındaki distance kadar y ekseninde kaça karşılık geliyorsa o noktaya kadar bu iki point'i aşağıda yer alan dendogramdaki gibi birleştirdik.

    aynı işlemi tepede birleşene kadar tekrarladık.





"""
df = pd.read_csv('../input/Iris.csv') 
df.drop(['Id', 'Species'], axis=1, inplace=True)
df_original = pd.read_csv('../input/Iris.csv') 
# Perform the necessary imports

from scipy.cluster.hierarchy import linkage, dendrogram

import matplotlib.pyplot as plt



# Calculate the linkage: mergings

mergings = linkage(df, method='complete') #burada kritik olan şu; df dediğimiz feature'lardan oluşuyorken, aşağıda labels kısmına ID'leri verdik öylesine.



# Plot the dendrogram, using varieties as labels

dendrogram(mergings, #linkage objecti input olarak koyduk.

           labels=df_original['Id'].values, #burada bir trick kullandık: bizden argümanı array olarak istiyor.

           leaf_rotation=90,

           leaf_font_size=5,

)

plt.show()
"""

maps samples to 2D space.

represents distance between samples.

great for inspecting the data set.



"""
df = pd.read_csv('../input/Iris.csv') 
df.drop(['Id', 'Species'], axis=1, inplace=True) #features
df_original = pd.read_csv('../input/Iris.csv') 
df_original['Species'].value_counts()
df_original.Species = [2 if i == 'Iris-setosa' else 1 if i == 'Iris-versicolor' else 0 for i in df_original.Species] #grafik için target'ın float olması gerekiyordu.
species = df_original.Species
species = species.values
#import required moduls

import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
model = TSNE(learning_rate=100)
transformed = model.fit_transform(df) #tSNE sadece fit_transform metoduna sahip.
xs = transformed[:, 0]

ys = transformed[:, 1]
plt.scatter(xs, ys, c=species ) #tsne plot'un X ve Y eksenlerinin bir anlamı yoktur.

plt.show() #tsne plot, her seferinde farklı grafik çıkarır aynı kod olmasına rağmen.

#datayı anlamaya yönelik kullanılan bir görselleştirme aracı olarak nitelenebilir.
"""

Dimension Reduction datada pattern bulur, bulduğu bu pattern'ı datayı açıklamak için kullanır.

more efficient storage and computation ile yapar bunu.

less informative noisy features ı çıkartarak yapar. 



"""
"""

PCA: Principle Component Analysis

   

   PCA, dimension reduction'ı 2 stepte yapar:

        1-decorrelation 

            bu stepte PCA rotates data to be aligned with axes. mean = 0 olacak şekilde kaydırır. 

        2-dimension reduction

            

"""
df = pd.read_csv('../input/Iris.csv') 
df.drop(['Id', 'Species'], axis=1, inplace=True) #features
from sklearn.decomposition import PCA
model = PCA()
model.fit(df)
transformed = model.transform(df)
print(transformed) #bu yeni array orjinal datasetle aynı sayıda row ve column'a sahip
xs_transformed = transformed[:,0]

ys_transformed = transformed[:,1]
xs = df.iloc[:,0]

ys = df.iloc[:,1]
plt.scatter(xs,ys) 
plt.scatter(xs_transformed,ys_transformed)



#bu iki grafik arasındaki fark ilkinin decorrelated olmaması (mean=0 olacak şekilde transform edildi)
"""

intrinsic: asıl, gerçek



Intrinsic Dimension: number of features needed to aproximate the dataset yani veri setini doğrulamak için gerekli feature sayısı



Intrinsic Dimension, dimension reduction'ın arkasında yatan ana fikirdir diyebiliriz. Çünkü bize, datasetin ne kadar daraltılabileceği hakkında fikir verir.



PCA kullanarak bunu gerçekleştireceğiz.







"""
"""

daha iyi anlayabilmek için iris data setindeki 4 feature'ın 3'ünü kullanarak örnek yapalım: 

"""

df = pd.read_csv('../input/Iris.csv')
df.drop(['Id', 'Species', 'PetalLengthCm'], axis=1, inplace=True) #features 3'e düşürüldü
df.head() #her bir sample aslında 3 boyutlu olarak ifade edilmiş durumda şu an.

#aslında yapacağımız şey, bu 3 boyutu çok bilgi kaybı yaşanmadan 2 boyuta düşürmek.
#işte tam bu noktada, PCA bize yardım eder:

#intrinsic dimension: number of PCA features with significant variance

#yani, significant variance'a sahip feature sayısını bularak intrinsic dimension'a ulaşırız.

#PCA'in yaptığı işlem, sample'ları 3D iken shift ve rotate'lerle yeni axis'lere co-ordinate etmesidir.

#PCA features belirli bir sıra ile elde edilir. 

#her feature'ın variance'ını gösteren bir bar-graph olşturulur.
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
pca = PCA()
pca.fit(df)
features = range(pca.n_components_) #pca'daki feature'ları enumerate ettik.
plt.bar(features, pca.explained_variance_) #variance'lar pca.explained_variance_ attribute'unu kullanarak edindik.

plt.xticks(features)

plt.ylabel('variace')

plt.xlabel('PCA feature')

plt.show()



#variance'ın gittikçe azaldığını görüyoruz. 



#intrinsic dimension: number of PCA features with significant variance

#yani, significant variance'a sahip feature sayısını bularak intrinsic dimension'a ulaşırız. DEMİŞTİK.



#bu örnekte, ilk iki feature significant variance'a sahip.

#so, this data set has intrinsic dimension 2! 

#tabibu her zaman kesin sonuç verir mi, bu tamamen significant olarak seçtiğimiz treshold'a bağlı.

#yani, burada variace 0.2 significant değil dersek intrinsic dimension=1 de diyebilirdik.



#ŞİMDİ, INTRINSIC DIMENSION'ı DIMENSION REDUCTION'da nasıl kulanacağımıza bakalım!
"""

Dimension Reduction, aynı datayı daha az feature barındırarak verimini kaybetmeden korumasını sağlar.



PCA kullarak dimension reduction yapacağız şimdi.



yukarıda intrinsic dimension ile kaç feature'ın high variance'a sahip olduğunu çıkarmıştık. 

Bu sayıyı PCA için kullanarak dimension reduction yapacağız: PCA(n_components=2) gibi



iris dataseti kullanacağız.



"""
df = pd.read_csv('../input/Iris.csv')
df.drop(['Id', 'Species'], axis=1, inplace=True) #tüm feature'Lar duruyor.
df_original = pd.read_csv('../input/Iris.csv')
df_original.Species = [2 if i == 'Iris-setosa' else 1 if i == 'Iris-versicolor' else 0 for i in df_original.Species] #grafik için target'ın float olması gerekiyordu.
species = df_original['Species'].values
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(df)
transformed = pca.transform(df)
print(transformed.shape) #sadece 2 features var, as expected
xs = transformed[:,0]

ys = transformed[:,1]

plt.scatter(xs, ys, c=species)

plt.show()
"""

word-frequency array rowlar document number, columnlar kelimeler, değerler ise kaç kez kullanıldığı



bu array bol sıfırlı olacaktır, array'i bir düşün anlarsın. 

bu durumdan ötürü, numpy array'i yerine, sadece 0 haricindeki verileri hatırlayan scipy csr_matrix'i kullanarak save space ve fast computation yapılabilir.



sklearn'in PCA'i csr_matrix'i desteklemediği için, aynı işi yapan ama csr_matrix'i destekleyen TruncatedSVD kullanmamız gerekiyor.!!





"""
from sklearn.decomposition import TruncatedSVD
model = TruncatedSVD(n_components=3)
model.fit(df)
transformed = model.transform(df)
"""

NMF de bir dimension reduction tekniğidir.



NMF, yorumlaması ve anlaşılması PCA'ye göre kolaydır.



Ancak, her datasete uygulanamaz.! All sample features >= 0 (non negative) olmalıdır.





PCA gibi syntax'ı var, ancak, n_components= parametresi zorunlu olarak doldurulmalı.





yine, word-frequency array'ler için kullanılır.

ayrıca, images encoded as arrays ve audio spectograms, purchase histories on e-commerce sites için kullanılabilir.





"""
from sklearn.decomposition import NMF
model = NMF(n_components=2)
model.fit(df)
nmf_features = model.transform(df)
print(model.components_) #dimension of the components is the same as the dimension of df

#2 components with 4 dimensions görüyoruz
print(nmf_features) #2 features var
# components ve features 'i bulduk. bunlar, datayı baştan reconstruct etmek için kullanılabilir. (multiplying components by features, and adding up-product of matrixes)

# tabiki bunu kullanmayacağız, ama MF (matrix factorization) ismi nereden geliyor onu anladık.
# now, we learn the components of NMF represent patterns that frequently occur in the samples



#bunun için articles dataseti kullanacağız (word-frequency)
articles = pd.read_csv('../input/wiki-articles/wikipedia-vectors.csv')
articles.drop('Unnamed: 0', axis=1, inplace=True)
articles.head()
articles.shape
words = articles.columns
from sklearn.decomposition import NMF
nmf = NMF(n_components=10)
nmf.fit(articles)
print(nmf.components_.shape) #10 row, 60 column, column sayısı değişmedi, row sayısı n_components'a eşit oldu

#yani, 10 components rowlarda yer alırken, 2d array oluştu.

#13125,60'tan 10,60'a çevrildi.
#10,60'lık bu arrayin üstüne de o 60 kelimeyi (column) yazdığımızı düşünelim, şimdi yorumlanabilir bir konumda:

#herhangi bir row(component) seçip hangi kelimelerin oranla ne kadar kullanıldığını görebiliriz.





# Create a DataFrame: components_df

components_df = pd.DataFrame(nmf.components_, columns=words)



# Print the shape of the DataFrame

print(components_df.shape)



# Select row 3: component

component = components_df.iloc[3]



# Print result of nlargest

print(component.nlargest())

"""

amacımız bir müşteri tarafından okunmuş article'lara göre müşteriye öneri sunmak.



assumption: similar articles have similar topics



word-frequency array'imize (article) NMF uygulayacağız. sonucu kullanacağız.



"""
articles = pd.read_csv('../input/wiki-articles/wikipedia-vectors.csv')
articles.drop('Unnamed: 0', axis=1, inplace=True)
from sklearn.decomposition import NMF
nmf = NMF(n_components=6)
nmf_features = nmf.fit_transform(articles) #artık her article için nmf_features sahibiyiz.
# peki, NMF features i kullanarak article'lar nasıl compare edilecek?



#COSINE SIMILARITY!!!



from sklearn.preprocessing import normalize

norm_features = normalize(nmf_features)



current_article = norm_features[2,:] # rastgele bir current article seçtik



similarities = norm_features.dot(current_article) #benzerlik ilişkisine burada baktık



print(similarities)
#şimdi öneri kısmı:



norm_features = normalize(nmf_features) #nmf_features ı normalize ettik, df'e çevirdik



df = pd.DataFrame(norm_features)
current_article = df.iloc[2] #current article seçtik
similarities = df.dot(current_article) #current article ile diğer article'ların similarity'lerini hesapladık
similarities.nlargest() #en yükseğe göre sıraladık. en yüksek değer kendisi, sonra 8497 geliyor.